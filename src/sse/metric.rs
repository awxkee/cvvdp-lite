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
    Contrast, Pool,
    vector::{self, Vector},
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Clone, Copy)]
struct Sse2;
// SAFETY: SSE2 provides 4 f32 lanes. Validated dispatch enables
// the CPU feature, and checked input lengths cover all complete vectors.
unsafe impl Vector for Sse2 {
    type F = __m128;
    type E = __m128i;
    type D = __m128d;
    const LANES: usize = 4;
    #[inline(always)]
    unsafe fn splat(v: f32) -> Self::F {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_set1_ps(v) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::F {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_loadu_ps(p) }
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::F) {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_storeu_ps(p, v) }
    }
    #[inline(always)]
    unsafe fn add(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_add_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn sub(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_sub_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn mul(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_mul_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn div(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_div_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn min(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_min_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn max(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_max_ps(a, b) }
    }
    #[inline(always)]
    unsafe fn abs(a: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_andnot_ps(_mm_set1_ps(-0.0), a) }
    }
    #[inline(always)]
    unsafe fn positive(input: Self::F, result: Self::F) -> Self::F {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_and_ps(_mm_cmpgt_ps(input, _mm_setzero_ps()), result) }
    }
    #[inline(always)]
    unsafe fn zero_d() -> Self::D {
        // SAFETY: Validated dispatch enables SSE2 and guarantees the Vector lane contract.
        unsafe { _mm_setzero_pd() }
    }
    #[inline(always)]
    unsafe fn decompose(x: Self::F) -> (Self::F, Self::E) {
        // SAFETY: The caller supplies positive normal inputs with SSE2 enabled.
        unsafe {
            let bits = _mm_add_epi32(
                _mm_castps_si128(x),
                _mm_set1_epi32(0x3f80_0000 - 0x3f35_04f3),
            );
            let exponent = _mm_sub_epi32(_mm_srli_epi32::<23>(bits), _mm_set1_epi32(127));
            let mantissa = _mm_add_epi32(
                _mm_and_si128(bits, _mm_set1_epi32(0x007f_ffff)),
                _mm_set1_epi32(0x3f35_04f3),
            );
            (_mm_castsi128_ps(mantissa), exponent)
        }
    }
    #[inline(always)]
    unsafe fn scale(exponent: Self::E, table: &[f32; 256]) -> Self::F {
        // SAFETY: SSE2 is enabled. Every lookup index is masked
        // to the allocated table's 256 initialized entries.
        unsafe {
            let index = _mm_and_si128(
                _mm_add_epi32(exponent, _mm_set1_epi32(127)),
                _mm_set1_epi32(255),
            );
            let mut indices = [0_i32; 4];
            _mm_storeu_si128(indices.as_mut_ptr().cast(), index);
            _mm_set_ps(
                table[indices[3] as usize],
                table[indices[2] as usize],
                table[indices[1] as usize],
                table[indices[0] as usize],
            )
        }
    }
    #[inline(always)]
    unsafe fn accumulate(value: Self::F, lo: &mut Self::D, hi: &mut Self::D) {
        // SAFETY: All register lanes are initialized and SSE2 is enabled.
        unsafe {
            let a = _mm_cvtps_pd(value);
            let b = _mm_cvtps_pd(_mm_movehl_ps(value, value));
            let e = _mm_set1_pd(2.0 * (crate::kernels::EPSILON as f64));
            *lo = _mm_add_pd(*lo, _mm_add_pd(_mm_mul_pd(a, a), _mm_mul_pd(a, e)));
            *hi = _mm_add_pd(*hi, _mm_add_pd(_mm_mul_pd(b, b), _mm_mul_pd(b, e)));
        }
    }
    #[inline(always)]
    unsafe fn sum_d(value: Self::D) -> f64 {
        let mut values = [0.0; 2];
        // SAFETY: The stack buffer contains exactly one f64 vector's lanes.
        unsafe { _mm_storeu_pd(values.as_mut_ptr(), value) };
        values.into_iter().sum()
    }
}
#[target_feature(enable = "sse2")]
pub(crate) fn contrast_sse2(data: &mut Contrast<'_>) -> usize {
    // Dispatch validates slice lengths, domain, and SSE2 availability.
    vector::contrast::<Sse2>(data)
}
#[target_feature(enable = "sse2")]
pub(crate) fn contrast_region_sse2(data: &mut crate::kernels::ContrastRegion<'_>) -> usize {
    // Dispatch validates strided planes, compact outputs and SSE2 availability.
    vector::contrast_region::<Sse2>(data)
}
#[target_feature(enable = "sse2")]
pub(crate) fn pool_sse2(data: &Pool<'_>) -> (usize, [f64; 3]) {
    // Dispatch validates slice lengths, domain, and SSE2 availability.
    vector::pool::<Sse2>(data)
}
#[target_feature(enable = "sse2")]
pub(crate) fn pool_tiles_sse2(
    data: &crate::kernels::TiledPool<'_>,
    tiles: &mut [[f64; 3]],
    stride: usize,
) {
    // Dispatch validates all strip strides, output lengths and SSE2 availability.
    vector::pool_tiles::<Sse2>(data, tiles, stride);
}
#[cfg(test)]
#[target_feature(enable = "sse2")]
pub(crate) fn powers_sse2(input: &[f32], output: &mut [f32], exponent: f32) -> usize {
    // The test wrapper checks slice lengths, domain, and SSE2 availability.
    vector::powers::<Sse2>(input, output, exponent)
}
