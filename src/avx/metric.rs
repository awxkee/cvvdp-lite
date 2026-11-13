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

use crate::kernels::{
    Contrast, POOL_TILE, Pool, PoolLanes,
    vector::{self, Vector},
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
struct Avx2<const FMA: bool>;
// SAFETY: AVX2 provides 8 f32 lanes. Validated dispatch enables
// the CPU features, including FMA for Avx2<true>, and checked input lengths
// cover all complete vectors.
unsafe impl<const FMA: bool> Vector<8> for Avx2<FMA> {
    type F = __m256;
    type E = __m256i;
    type D = __m256d;
    #[inline(always)]
    unsafe fn splat(v: f32) -> Self::F {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_set1_ps(v) }
    }
    #[inline(always)]
    unsafe fn load(chunk: &[f32; 8]) -> Self::F {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_loadu_ps(chunk.as_ptr()) }
    }
    #[inline(always)]
    unsafe fn store(chunk: &mut [f32; 8], v: Self::F) {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_storeu_ps(chunk.as_mut_ptr(), v) }
    }
    #[inline(always)]
    unsafe fn add(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_add_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn sub(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_sub_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_mul_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul_add(a: Self::F, b: Self::F, c: Self::F) -> Self::F {
        // SAFETY: Both variants enable AVX2. Only FMA-enabled entry points
        // instantiate Avx2<true>.
        unsafe {
            if FMA {
                _mm256_fmadd_ps(a, b, c)
            } else {
                _mm256_add_ps(_mm256_mul_ps(a, b), c)
            }
        }
    }
    #[inline(always)]
    unsafe fn div(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_div_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn min(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_min_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn max(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_max_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn abs(a: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_andnot_ps(_mm256_set1_ps(-0.0), a) }
    }
    #[inline(always)]
    unsafe fn positive(input: Self::F, result: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe {
            _mm256_and_ps(
                _mm256_cmp_ps::<_CMP_GT_OQ>(input, _mm256_setzero_ps()),
                result,
            )
        }
    }
    #[inline(always)]
    unsafe fn zero_d() -> Self::D {
        // SAFETY: Validated dispatch enables AVX2 and guarantees the Vector lane contract.
        unsafe { _mm256_setzero_pd() }
    }
    #[inline(always)]
    unsafe fn load_sums(row: &[f64; POOL_TILE]) -> (Self::D, Self::D) {
        let (chunks, _) = row.as_chunks::<4>();
        // SAFETY: The caller enables the ISA. Both chunks hold complete vectors.
        unsafe {
            (
                _mm256_loadu_pd(chunks[0].as_ptr()),
                _mm256_loadu_pd(chunks[1].as_ptr()),
            )
        }
    }
    #[inline(always)]
    unsafe fn store_sums(row: &mut [f64; POOL_TILE], lo: Self::D, hi: Self::D) {
        let (chunks, _) = row.as_chunks_mut::<4>();
        // SAFETY: The caller enables the ISA. Both chunks hold complete vectors.
        unsafe {
            _mm256_storeu_pd(chunks[0].as_mut_ptr(), lo);
            _mm256_storeu_pd(chunks[1].as_mut_ptr(), hi);
        }
    }
    #[inline(always)]
    unsafe fn decompose(x: Self::F) -> (Self::F, Self::E) {
        // SAFETY: The caller supplies positive normal inputs with AVX2 enabled.
        unsafe {
            let bits = _mm256_add_epi32(
                _mm256_castps_si256(x),
                _mm256_set1_epi32(0x3f80_0000 - 0x3f35_04f3),
            );
            let exponent = _mm256_sub_epi32(_mm256_srli_epi32::<23>(bits), _mm256_set1_epi32(127));
            let mantissa = _mm256_add_epi32(
                _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
                _mm256_set1_epi32(0x3f35_04f3),
            );
            (_mm256_castsi256_ps(mantissa), exponent)
        }
    }
    #[inline(always)]
    unsafe fn scale(exponent: Self::E, table: &[f32; 256]) -> Self::F {
        // SAFETY: AVX2 is enabled. Every lookup index is masked
        // to the allocated table's 256 initialized entries.
        unsafe {
            let index = _mm256_and_si256(
                _mm256_add_epi32(exponent, _mm256_set1_epi32(127)),
                _mm256_set1_epi32(255),
            );
            _mm256_i32gather_ps::<4>(table.as_ptr(), index)
        }
    }
    #[inline(always)]
    unsafe fn accumulate(value: Self::F, lo: &mut Self::D, hi: &mut Self::D) {
        // SAFETY: All register lanes are initialized and AVX2 is enabled.
        unsafe {
            let a = _mm256_cvtps_pd(_mm256_castps256_ps128(value));
            let b = _mm256_cvtps_pd(_mm256_extractf128_ps::<1>(value));
            let e = _mm256_set1_pd(2.0 * (crate::kernels::EPSILON as f64));
            let contribution = |d| {
                let linear = _mm256_mul_pd(d, e);
                // The products are exact in f64 with f32-precision operands.
                if FMA {
                    _mm256_fmadd_pd(d, d, linear)
                } else {
                    _mm256_add_pd(_mm256_mul_pd(d, d), linear)
                }
            };
            *lo = _mm256_add_pd(*lo, contribution(a));
            *hi = _mm256_add_pd(*hi, contribution(b));
        }
    }
    #[inline(always)]
    unsafe fn sum_d(value: Self::D) -> f64 {
        let mut values = [0.0; 4];
        // SAFETY: The stack buffer contains exactly one f64 vector's lanes.
        unsafe { _mm256_storeu_pd(values.as_mut_ptr(), value) };
        values.into_iter().sum()
    }
}
#[target_feature(enable = "avx2")]
pub(crate) fn contrast_avx2(data: &mut Contrast<'_>) -> usize {
    // Dispatch validates slice lengths, domain, and AVX2 availability.
    vector::contrast::<Avx2<false>, 8>(data)
}
/// # Safety
/// The caller must validate all regional plane lengths and strides and enable AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn contrast_region_avx2(data: &mut crate::kernels::ContrastRegion<'_>) -> usize {
    // SAFETY: The caller meets the regional contrast kernel's requirements.
    unsafe { vector::contrast_region::<Avx2<false>, 8>(data) }
}
#[target_feature(enable = "avx2")]
pub(crate) fn pool_avx2(data: &Pool<'_>, lanes: &mut PoolLanes, last: bool) -> (usize, [f64; 3]) {
    // Dispatch validates slice lengths, domain, and AVX2 availability.
    vector::pool::<Avx2<false>, 8>(data, lanes, last)
}
#[target_feature(enable = "avx2,fma")]
pub(crate) fn pool_fma_avx2(
    data: &Pool<'_>,
    lanes: &mut PoolLanes,
    last: bool,
) -> (usize, [f64; 3]) {
    // Dispatch validates slice lengths, domain, and AVX2 and FMA availability.
    vector::pool::<Avx2<true>, 8>(data, lanes, last)
}
#[cfg(test)]
#[target_feature(enable = "avx2")]
pub(crate) fn pool_tiles_avx2(
    data: &crate::kernels::TiledPool<'_>,
    tiles: &mut [[f64; 3]],
    stride: usize,
) {
    // Dispatch validates all strip strides, output lengths and AVX2 availability.
    let powers = vector::power_tables();
    crate::kernels::pool_tiles_with(data, tiles, stride, |row| pool_row_avx2(row, powers));
}
#[cfg(test)]
#[target_feature(enable = "avx2,fma")]
pub(crate) fn pool_tiles_fma_avx2(
    data: &crate::kernels::TiledPool<'_>,
    tiles: &mut [[f64; 3]],
    stride: usize,
) {
    let powers = vector::power_tables();
    crate::kernels::pool_tiles_with(data, tiles, stride, |row| pool_row_fma_avx2(row, powers));
}
#[target_feature(enable = "avx2")]
pub(crate) fn pool_parts_avx2(
    data: &crate::kernels::TiledPool<'_>,
    parts: &mut [[f64; 3]],
    stride: usize,
) {
    // Dispatch validates the strip strides, output length and CPU availability.
    let powers = vector::power_tables();
    crate::kernels::pool_parts_with(data, parts, stride, |row| pool_row_avx2(row, powers));
}
#[target_feature(enable = "avx2,fma")]
pub(crate) fn pool_parts_fma_avx2(
    data: &crate::kernels::TiledPool<'_>,
    parts: &mut [[f64; 3]],
    stride: usize,
) {
    let powers = vector::power_tables();
    crate::kernels::pool_parts_with(data, parts, stride, |row| pool_row_fma_avx2(row, powers));
}

// Keep AVX2 enabled if the compiler outlines the row callback.
#[inline]
#[target_feature(enable = "avx2")]
fn pool_row_avx2(data: &Pool<'_>, powers: &[vector::PowerTable; 4]) -> [f64; 3] {
    // SAFETY: This callback runs with AVX2 enabled and receives validated rows.
    unsafe { pool_row::<false>(data, powers) }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn pool_row_fma_avx2(data: &Pool<'_>, powers: &[vector::PowerTable; 4]) -> [f64; 3] {
    // SAFETY: This callback runs with AVX2 and FMA enabled and validated rows.
    unsafe { pool_row::<true>(data, powers) }
}

#[inline(always)]
unsafe fn pool_row<const FMA: bool>(data: &Pool<'_>, powers: &[vector::PowerTable; 4]) -> [f64; 3] {
    let (done, mut squares) = vector::pool_with_powers::<Avx2<FMA>, 8>(data, powers);
    crate::kernels::pool_scalar_tail(data, done, &mut squares);
    squares
}
#[cfg(test)]
#[target_feature(enable = "avx2")]
pub(crate) fn powers_avx2(input: &[f32], output: &mut [f32], exponent: f32) -> usize {
    // The test wrapper checks slice lengths, domain, and AVX2 availability.
    vector::powers::<Avx2<false>, 8>(input, output, exponent)
}
#[cfg(test)]
#[target_feature(enable = "avx2,fma")]
pub(crate) fn powers_fma_avx2(input: &[f32], output: &mut [f32], exponent: f32) -> usize {
    // The test wrapper checks lengths, domain, and AVX2 and FMA availability.
    vector::powers::<Avx2<true>, 8>(input, output, exponent)
}
