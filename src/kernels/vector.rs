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

#[cfg(any(
    all(feature = "neon", target_arch = "aarch64"),
    all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64"))
))]
use super::TiledPool;
use super::{Contrast, ContrastRegion, EPSILON, MASK_P, MASK_Q, POOL_TILE, Pool, PoolLanes};

/// # Safety
/// Private vector implementations must preserve the order of LANES f32 values
/// and provide exact lane-wise arithmetic and f64 reduction accumulators.
pub(crate) unsafe trait Vector<const LANES: usize>: Copy {
    type F: Copy;
    /// Integer SIMD exponents, kept exact between decomposition and lookup.
    type E: Copy;
    type D: Copy;
    unsafe fn splat(v: f32) -> Self::F;
    unsafe fn load(chunk: &[f32; LANES]) -> Self::F;
    unsafe fn store(chunk: &mut [f32; LANES], v: Self::F);
    unsafe fn add(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn sub(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn mul(a: Self::F, b: Self::F) -> Self::F;
    /// Compute `a * b + c`, fusing the operations when the backend supports FMA.
    #[inline(always)]
    unsafe fn mul_add(a: Self::F, b: Self::F, c: Self::F) -> Self::F {
        // SAFETY: The caller enables the backend's ISA.
        unsafe { Self::add(Self::mul(a, b), c) }
    }
    unsafe fn div(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn min(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn max(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn abs(a: Self::F) -> Self::F;
    /// Keep result where the original positive input exceeds zero, else zero.
    unsafe fn positive(input: Self::F, result: Self::F) -> Self::F;
    /// x = mantissa * 2^exponent, mantissa in [sqrt(1/2),sqrt(2)).
    unsafe fn decompose(x: Self::F) -> (Self::F, Self::E);
    /// Lookup 2^(p*exponent), for exponents from decompose.
    unsafe fn scale(exponent: Self::E, table: &[f32; 256]) -> Self::F;
    unsafe fn zero_d() -> Self::D;
    /// Read/write the two halves of the active f64 lanes in a pooling row.
    unsafe fn load_sums(row: &[f64; POOL_TILE]) -> (Self::D, Self::D);
    unsafe fn store_sums(row: &mut [f64; POOL_TILE], lo: Self::D, hi: Self::D);
    unsafe fn accumulate(value: Self::F, lo: &mut Self::D, hi: &mut Self::D);
    unsafe fn sum_d(value: Self::D) -> f64;
}

pub(crate) struct PowerTable {
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

pub(crate) fn power_tables() -> &'static [PowerTable; 4] {
    static TABLES: std::sync::OnceLock<[PowerTable; 4]> = std::sync::OnceLock::new();
    TABLES.get_or_init(|| [MASK_Q[0], MASK_Q[1], MASK_Q[2], MASK_P].map(PowerTable::new))
}

#[cfg(test)]
#[inline(always)]
pub(crate) fn powers<V: Vector<LANES>, const LANES: usize>(
    input: &[f32],
    output: &mut [f32],
    exponent: f32,
) -> usize {
    let (input, _) = input.as_chunks::<LANES>();
    let (output, _) = output.as_chunks_mut::<LANES>();
    let done = input.len() * LANES;
    let table = PowerTable::new(exponent);
    // SAFETY: The test wrapper checks CPU availability, input domain, and lengths.
    unsafe {
        for (input, output) in input.iter().zip(output) {
            let value = pow::<V, LANES>(V::load(input), &table);
            V::store(output, value);
        }
    }
    done
}

#[inline(always)]
fn pow<V: Vector<LANES>, const LANES: usize>(x: V::F, table: &PowerTable) -> V::F {
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
            polynomial = V::mul_add(polynomial, t, V::splat(table.polynomial[i]));
        }
        V::mul(polynomial, V::scale(exponent, &table.scale))
    }
}

#[inline(always)]
fn plane_chunks<const LANES: usize>(
    [a, b, c]: [&[f32]; 3],
) -> impl Iterator<Item = [&[f32; LANES]; 3]> {
    let [a, b, c] = [a, b, c].map(|plane| plane.as_chunks::<LANES>().0);
    a.iter().zip(b).zip(c).map(|((a, b), c)| [a, b, c])
}

#[inline(always)]
fn plane_chunks_mut<const LANES: usize>(
    [a, b, c]: [&mut [f32]; 3],
) -> impl Iterator<Item = [&mut [f32; LANES]; 3]> {
    let [a, b, c] = [a, b, c].map(|plane| plane.as_chunks_mut::<LANES>().0);
    a.iter_mut().zip(b).zip(c).map(|((a, b), c)| [a, b, c])
}

#[inline(always)]
// Private ISA types reach this helper through CPU-validated target-feature
// entry points; dispatch also validates all plane lengths.
pub(crate) fn contrast<V: Vector<LANES>, const LANES: usize>(data: &mut Contrast<'_>) -> usize {
    let done = data.current[0].len() / LANES * LANES;
    let current = plane_chunks_mut(data.current.each_mut().map(|plane| &mut **plane));
    let masks = plane_chunks_mut(data.expanded_then_mask.each_mut().map(|plane| &mut **plane));
    let weights = plane_chunks(data.sensitivity).zip(plane_chunks(data.reference));
    // SAFETY: CPU-validated dispatch enables the ISA. Chunks hold complete vectors.
    unsafe {
        let scale = V::splat(data.scale);
        let floor = V::splat(0.01);
        let ceiling = V::splat(1000.0);
        let one = V::splat(1.0);
        for ((current, masks), (sensitivity, reference)) in current.zip(masks).zip(weights) {
            // Keep the achromatic background before its lanes become masks.
            let bg = V::max(V::load(masks[0]), floor);
            let inv_bg = V::div(one, bg);
            for c in 0..3 {
                let src = V::load(current[c]);
                let low = V::load(masks[c]);
                let sensitivity = V::load(sensitivity[c]);
                let reference = V::load(reference[c]);
                let weighted = V::mul(
                    V::mul(V::min(V::mul(V::sub(src, low), inv_bg), ceiling), scale),
                    sensitivity,
                );
                V::store(current[c], weighted);
                V::store(masks[c], V::min(V::abs(weighted), V::abs(reference)));
            }
        }
    }
    done
}

/// # Safety
/// The caller must validate the source strides and plane lengths and enable
/// the ISA, as in the regional dispatch.
#[inline(always)]
pub(crate) unsafe fn contrast_region<V: Vector<LANES>, const LANES: usize>(
    data: &mut ContrastRegion<'_>,
) -> usize {
    let done = data.width / LANES * LANES;
    // SAFETY: Dispatch enables the ISA and validates every strided row and
    // compact plane. Each input slice below stays within one validated row.
    unsafe {
        let scale = V::splat(data.scale);
        let floor = V::splat(0.01);
        let ceiling = V::splat(1000.0);
        let one = V::splat(1.0);
        // Work through one channel at a time to keep fewer row streams live.
        for (c, (contrast, mask)) in data.contrast.iter_mut().zip(&mut data.mask).enumerate() {
            for y in 0..data.height {
                let src = y * data.source_stride;
                let dst = y * data.width;
                let output = dst..dst + done;
                let [current, sensitivity, reference, low, luminance] = [
                    data.current[c].get_unchecked(src..src + done),
                    data.sensitivity[c].get_unchecked(src..src + done),
                    data.reference[c].get_unchecked(src..src + done),
                    data.expanded[c].get_unchecked(output.clone()),
                    data.expanded[0].get_unchecked(output.clone()),
                ]
                .map(|plane| plane.as_chunks::<LANES>().0);
                let (contrast, _) = contrast
                    .get_unchecked_mut(output.clone())
                    .as_chunks_mut::<LANES>();
                let (mask, _) = mask.get_unchecked_mut(output).as_chunks_mut::<LANES>();
                let source = current.iter().zip(sensitivity).zip(reference);
                for ((((current, sensitivity), reference), (low, bg)), (contrast, mask)) in source
                    .zip(low.iter().zip(luminance))
                    .zip(contrast.iter_mut().zip(mask))
                {
                    let bg = V::max(V::load(bg), floor);
                    let current = V::load(current);
                    let low = V::load(low);
                    let sensitivity = V::load(sensitivity);
                    let reference = V::load(reference);
                    let weighted = V::mul(
                        V::mul(
                            V::min(V::mul(V::sub(current, low), V::div(one, bg)), ceiling),
                            scale,
                        ),
                        sensitivity,
                    );
                    V::store(contrast, weighted);
                    V::store(mask, V::min(V::abs(weighted), V::abs(reference)));
                }
            }
        }
    }
    done
}

#[inline(always)]
pub(crate) fn pool<V: Vector<LANES>, const LANES: usize>(
    data: &Pool<'_>,
    lanes: &mut PoolLanes,
    last: bool,
) -> (usize, [f64; 3]) {
    // SAFETY: Each row has space for all lanes. The calling target-feature
    // entry and checked data slices cover every SIMD operation below.
    unsafe {
        let [(lo0, hi0), (lo1, hi1), (lo2, hi2)] = lanes.each_ref().map(|row| V::load_sums(row));
        let mut lo = [lo0, lo1, lo2];
        let mut hi = [hi0, hi1, hi2];
        let done = accumulate_pool::<V, LANES>(data, power_tables(), &mut lo, &mut hi);
        if last {
            return (done, sum_pool::<V, LANES>(lo, hi));
        }
        for c in 0..3 {
            V::store_sums(&mut lanes[c], lo[c], hi[c]);
        }
        (done, [0.0; 3])
    }
}

#[cfg(test)]
#[cfg(any(
    all(feature = "neon", target_arch = "aarch64"),
    all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64"))
))]
#[inline(always)]
pub(crate) fn pool_tiles<V: Vector<LANES>, const LANES: usize>(
    data: &TiledPool<'_>,
    tiles: &mut [[f64; 3]],
    stride: usize,
) {
    let powers = power_tables();
    super::pool_tiles_with(data, tiles, stride, |row| {
        let (done, mut squares) = pool_with_powers::<V, LANES>(row, powers);
        super::pool_scalar_tail(row, done, &mut squares);
        squares
    });
}

#[cfg(any(
    all(feature = "neon", target_arch = "aarch64"),
    all(feature = "sse", any(target_arch = "x86", target_arch = "x86_64"))
))]
#[inline(always)]
pub(crate) fn pool_parts<V: Vector<LANES>, const LANES: usize>(
    data: &TiledPool<'_>,
    parts: &mut [[f64; 3]],
    parts_stride: usize,
) {
    let powers = power_tables();
    super::pool_parts_with(data, parts, parts_stride, |row| {
        let (done, mut squares) = pool_with_powers::<V, LANES>(row, powers);
        super::pool_scalar_tail(row, done, &mut squares);
        squares
    });
}

#[inline(always)]
pub(crate) fn pool_with_powers<V: Vector<LANES>, const LANES: usize>(
    data: &Pool<'_>,
    powers: &[PowerTable; 4],
) -> (usize, [f64; 3]) {
    // SAFETY: ISA support and data lengths were validated by the strip dispatch.
    unsafe {
        let mut lo = [V::zero_d(); 3];
        let mut hi = [V::zero_d(); 3];
        let done = accumulate_pool::<V, LANES>(data, powers, &mut lo, &mut hi);
        (done, sum_pool::<V, LANES>(lo, hi))
    }
}

#[inline(always)]
fn sum_pool<V: Vector<LANES>, const LANES: usize>(lo: [V::D; 3], hi: [V::D; 3]) -> [f64; 3] {
    // SAFETY: The caller enables the ISA and all sums are initialized.
    std::array::from_fn(|c| unsafe { V::sum_d(lo[c]) + V::sum_d(hi[c]) })
}

#[inline(always)]
fn mask_activity<V: Vector<LANES>, const LANES: usize>(
    blurred: &[f32; LANES],
    mask_zero: f32,
    mask_c: V::F,
    epsilon: V::F,
    zero: V::F,
    power: &PowerTable,
) -> V::F {
    // SAFETY: The caller enables the ISA and validates the vector bounds.
    unsafe {
        let blurred = V::load(blurred);
        let value = V::mul(blurred, mask_c);
        let power = pow::<V, LANES>(V::mul_add(blurred, mask_c, epsilon), power);
        // Preserve exact zeros despite approximation/subtraction near epsilon.
        V::positive(value, V::max(V::sub(power, V::splat(mask_zero)), zero))
    }
}

#[inline(always)]
fn accumulate_pool<V: Vector<LANES>, const LANES: usize>(
    data: &Pool<'_>,
    powers: &[PowerTable; 4],
    lo: &mut [V::D; 3],
    hi: &mut [V::D; 3],
) -> usize {
    let done = data.current[0].len() / LANES * LANES;
    let p = data.parameters;
    // SAFETY: Equal checked plane lengths cover every complete vector, CPU
    // features are enabled by the caller, and the arithmetic domain is bounded.
    unsafe {
        let epsilon = V::splat(EPSILON);
        let zero = V::splat(0.0);
        let one = V::splat(1.0);
        let mask_c = V::splat(p.mask_c);
        let dmax = V::splat(p.dmax);
        let inputs = plane_chunks(data.current).zip(plane_chunks(data.reference));
        for ((current, reference), blurred) in inputs.zip(plane_chunks(data.blurred)) {
            // Keep all three evaluations in the loop across SIMD backends.
            let activity = [
                mask_activity::<V, LANES>(
                    blurred[0],
                    p.mask_zero[0],
                    mask_c,
                    epsilon,
                    zero,
                    &powers[0],
                ),
                mask_activity::<V, LANES>(
                    blurred[1],
                    p.mask_zero[1],
                    mask_c,
                    epsilon,
                    zero,
                    &powers[1],
                ),
                mask_activity::<V, LANES>(
                    blurred[2],
                    p.mask_zero[2],
                    mask_c,
                    epsilon,
                    zero,
                    &powers[2],
                ),
            ];
            for c in 0..3 {
                let mask = V::mul(activity[0], V::splat(p.xcm[0][c]));
                let mask = V::mul_add(activity[1], V::splat(p.xcm[1][c]), mask);
                let mask = V::mul_add(activity[2], V::splat(p.xcm[2][c]), mask);
                let diff = V::abs(V::sub(V::load(current[c]), V::load(reference[c])));
                let power = pow::<V, LANES>(V::add(diff, epsilon), &powers[3]);
                let numerator =
                    V::positive(diff, V::max(V::sub(power, V::splat(p.diff_zero)), zero));
                // dmax·du / (dmax + du) with du = numerator / (1 + mask),
                // folded into a single division.
                let d = V::div(
                    V::mul(dmax, numerator),
                    V::mul_add(dmax, V::add(one, mask), numerator),
                );
                V::accumulate(d, &mut lo[c], &mut hi[c]);
            }
        }
    }
    done
}
