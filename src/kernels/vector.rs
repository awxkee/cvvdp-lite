/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! Shared monomorphized SIMD equations prevent architecture-specific model
//! drift. Math is restricted to positive normal inputs (x+epsilon), with
//! normal masking power results. Display/input validation bounds that domain.

use super::{Contrast, EPSILON, MASK_P, MASK_Q, Pool};

/// # Safety
/// Private vector implementations must provide LANES initialized f32 values,
/// exact lane-wise arithmetic/bit operations, and f64 reduction accumulators.
pub(crate) unsafe trait Vector: Copy {
    type F: Copy;
    type D: Copy;
    const LANES: usize;
    unsafe fn splat(v: f32) -> Self::F;
    unsafe fn load(p: *const f32) -> Self::F;
    unsafe fn store(p: *mut f32, v: Self::F);
    unsafe fn add(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn sub(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn mul(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn div(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn min(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn max(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn abs(a: Self::F) -> Self::F;
    /// Keep result where the original positive input exceeds zero, else zero.
    unsafe fn positive(input: Self::F, result: Self::F) -> Self::F;
    /// x = mantissa * 2^exponent, mantissa in [sqrt(1/2),sqrt(2)).
    unsafe fn decompose(x: Self::F) -> (Self::F, Self::F);
    /// Lookup 2^(p*exponent), for exponents from decompose.
    unsafe fn scale(exponent: Self::F, table: &[f32; 256]) -> Self::F;
    unsafe fn zero_d() -> Self::D;
    unsafe fn accumulate(value: Self::F, lo: &mut Self::D, hi: &mut Self::D);
    unsafe fn sum_d(value: Self::D) -> f64;
}

pub(super) struct PowerTable {
    scale: [f32; 256],
    polynomial: [f32; 11],
}

impl PowerTable {
    fn new(exponent: f32) -> Self {
        let p = exponent as f64;
        let scale = std::array::from_fn(|i| 2.0_f64.powf((i as i32 - 127) as f64 * p) as f32);
        let mut polynomial = [1.0; 11];
        let mut coefficient = 1.0_f64;
        for (i, value) in polynomial.iter_mut().enumerate().skip(1) {
            coefficient *= (p - (i - 1) as f64) / i as f64;
            *value = coefficient as f32;
        }
        Self { scale, polynomial }
    }
}

pub(super) fn power_tables() -> &'static [PowerTable; 4] {
    static TABLES: std::sync::OnceLock<[PowerTable; 4]> = std::sync::OnceLock::new();
    TABLES.get_or_init(|| [MASK_Q[0], MASK_Q[1], MASK_Q[2], MASK_P].map(PowerTable::new))
}

#[cfg(test)]
#[inline(always)]
pub(crate) fn powers<V: Vector>(input: &[f32], output: &mut [f32], exponent: f32) -> usize {
    let done = input.len() / V::LANES * V::LANES;
    let table = PowerTable::new(exponent);
    // SAFETY: The test wrapper checks CPU availability, input domain, and lengths.
    unsafe {
        for i in (0..done).step_by(V::LANES) {
            let value = pow::<V>(V::load(input.as_ptr().add(i)), &table);
            V::store(output.as_mut_ptr().add(i), value);
        }
    }
    done
}

#[inline(always)]
fn pow<V: Vector>(x: V::F, table: &PowerTable) -> V::F {
    // SAFETY: Positive normal inputs and enabled CPU features are guaranteed
    // by the caller. Vector::scale masks lookup indices to the table bounds.
    unsafe {
        let (mantissa, exponent) = V::decompose(x);
        let t = V::sub(mantissa, V::splat(1.0));
        // Binomial expansion of (1+t)^p. Centered mantissas bound t to
        // [-0.292894,0.414214). Degree ten is checked against libm for all
        // four calibrated exponents and every exponent in the masking range.
        let mut polynomial = V::splat(table.polynomial[10]);
        for i in (0..10).rev() {
            polynomial = V::add(V::mul(polynomial, t), V::splat(table.polynomial[i]));
        }
        V::mul(polynomial, V::scale(exponent, &table.scale))
    }
}

#[inline(always)]
// Private ISA types reach this helper through CPU-validated target-feature
// entry points; dispatch also validates all plane lengths.
pub(crate) fn contrast<V: Vector>(data: &mut Contrast<'_>) -> usize {
    let done = data.current[0].len() / V::LANES * V::LANES;
    // SAFETY: The safe wrapper checked all plane lengths. Each load/store
    // processes LANES initialized, non-overlapping elements before done.
    unsafe {
        let scale = V::splat(data.scale);
        let floor = V::splat(0.01);
        let ceiling = V::splat(1000.0);
        for i in (0..done).step_by(V::LANES) {
            let bg = V::max(V::load(data.expanded[0].as_ptr().add(i)), floor);
            for c in 0..3 {
                let src = V::load(data.current[c].as_ptr().add(i));
                let low = V::load(data.expanded[c].as_ptr().add(i));
                let sensitivity = V::load(data.sensitivity[c].as_ptr().add(i));
                let reference = V::load(data.reference[c].as_ptr().add(i));
                let weighted = V::mul(
                    V::mul(V::min(V::div(V::sub(src, low), bg), ceiling), scale),
                    sensitivity,
                );
                V::store(data.current[c].as_mut_ptr().add(i), weighted);
                V::store(
                    data.mask[c].as_mut_ptr().add(i),
                    V::min(V::abs(weighted), V::abs(reference)),
                );
            }
        }
    }
    done
}

#[inline(always)]
pub(crate) fn pool<V: Vector>(data: &Pool<'_>) -> (usize, [f64; 3]) {
    let done = data.current[0].len() / V::LANES * V::LANES;
    let p = data.parameters;
    let powers = power_tables();
    // SAFETY: Equal checked plane lengths cover every complete vector, CPU
    // features are enabled by the caller, and the arithmetic domain is bounded.
    unsafe {
        let mut lo = [V::zero_d(); 3];
        let mut hi = [V::zero_d(); 3];
        let epsilon = V::splat(EPSILON);
        let zero = V::splat(0.0);
        let one = V::splat(1.0);
        let mask_c = V::splat(p.mask_c);
        let dmax = V::splat(p.dmax);
        for i in (0..done).step_by(V::LANES) {
            let activity: [V::F; 3] = std::array::from_fn(|c| {
                let value = V::mul(V::load(data.blurred[c].as_ptr().add(i)), mask_c);
                let power = pow::<V>(V::add(value, epsilon), &powers[c]);
                // Preserve exact zeros despite approximation/subtraction near epsilon.
                V::positive(value, V::max(V::sub(power, V::splat(p.mask_zero[c])), zero))
            });
            for c in 0..3 {
                let mask = V::add(
                    V::add(
                        V::mul(activity[0], V::splat(p.xcm[0][c])),
                        V::mul(activity[1], V::splat(p.xcm[1][c])),
                    ),
                    V::mul(activity[2], V::splat(p.xcm[2][c])),
                );
                let diff = V::abs(V::sub(
                    V::load(data.current[c].as_ptr().add(i)),
                    V::load(data.reference[c].as_ptr().add(i)),
                ));
                let power = pow::<V>(V::add(diff, epsilon), &powers[3]);
                let numerator =
                    V::positive(diff, V::max(V::sub(power, V::splat(p.diff_zero)), zero));
                let du = V::div(numerator, V::add(one, mask));
                let d = V::div(V::mul(dmax, du), V::add(dmax, du));
                V::accumulate(d, &mut lo[c], &mut hi[c]);
            }
        }
        (
            done,
            std::array::from_fn(|c| V::sum_d(lo[c]) + V::sum_d(hi[c])),
        )
    }
}
