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
use std::arch::aarch64::*;

#[derive(Clone, Copy)]
struct Neon;

// SAFETY: NEON has four f32 lanes and two f64 lanes per half. Entry points
// enable NEON; checked slice lengths cover the unaligned loads and stores.
unsafe impl Vector for Neon {
    type F = float32x4_t;
    type E = int32x4_t;
    type D = float64x2_t;
    const LANES: usize = 4;
    #[inline(always)]
    unsafe fn splat(v: f32) -> Self::F {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vdupq_n_f32(v) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self::F {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vld1q_f32(p) }
    }
    #[inline(always)]
    unsafe fn store(p: *mut f32, v: Self::F) {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vst1q_f32(p, v) }
    }
    #[inline(always)]
    unsafe fn add(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vaddq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn sub(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vsubq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn mul(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vmulq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn div(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vdivq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn min(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vminq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn max(a: Self::F, b: Self::F) -> Self::F {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vmaxq_f32(a, b) }
    }
    #[inline(always)]
    unsafe fn abs(a: Self::F) -> Self::F {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vabsq_f32(a) }
    }
    #[inline(always)]
    unsafe fn positive(input: Self::F, result: Self::F) -> Self::F {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe {
            vreinterpretq_f32_u32(vandq_u32(
                vcgtq_f32(input, vdupq_n_f32(0.0)),
                vreinterpretq_u32_f32(result),
            ))
        }
    }
    #[inline(always)]
    unsafe fn zero_d() -> Self::D {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vdupq_n_f64(0.0) }
    }
    #[inline(always)]
    unsafe fn sum_d(value: Self::D) -> f64 {
        // SAFETY: The caller enables NEON and satisfies the Vector lane contract.
        unsafe { vaddvq_f64(value) }
    }
    #[inline(always)]
    unsafe fn decompose(x: Self::F) -> (Self::F, Self::E) {
        // SAFETY: Positive normal inputs and enabled NEON are guaranteed by the caller.
        unsafe {
            let bits = vaddq_u32(
                vreinterpretq_u32_f32(x),
                vdupq_n_u32(0x3f80_0000 - 0x3f35_04f3),
            );
            let exponent = vsubq_s32(
                vreinterpretq_s32_u32(vshrq_n_u32::<23>(bits)),
                vdupq_n_s32(127),
            );
            let mantissa = vaddq_u32(
                vandq_u32(bits, vdupq_n_u32(0x007f_ffff)),
                vdupq_n_u32(0x3f35_04f3),
            );
            (vreinterpretq_f32_u32(mantissa), exponent)
        }
    }
    #[inline(always)]
    unsafe fn scale(exponent: Self::E, table: &[f32; 256]) -> Self::F {
        // SAFETY: NEON is enabled. Masked integer lookup indices always lie
        // within the table, including if an internal caller violates the domain.
        unsafe {
            let index = vandq_u32(
                vreinterpretq_u32_s32(vaddq_s32(exponent, vdupq_n_s32(127))),
                vdupq_n_u32(255),
            );
            let values = [
                table[vgetq_lane_u32::<0>(index) as usize],
                table[vgetq_lane_u32::<1>(index) as usize],
                table[vgetq_lane_u32::<2>(index) as usize],
                table[vgetq_lane_u32::<3>(index) as usize],
            ];
            vld1q_f32(values.as_ptr())
        }
    }
    #[inline(always)]
    unsafe fn accumulate(value: Self::F, lo: &mut Self::D, hi: &mut Self::D) {
        // SAFETY: Registers have four initialized lanes and NEON is enabled.
        unsafe {
            let a = vcvt_f64_f32(vget_low_f32(value));
            let b = vcvt_high_f64_f32(value);
            let e = vdupq_n_f64(2.0 * (crate::kernels::EPSILON as f64));
            *lo = vaddq_f64(*lo, vaddq_f64(vmulq_f64(a, a), vmulq_f64(a, e)));
            *hi = vaddq_f64(*hi, vaddq_f64(vmulq_f64(b, b), vmulq_f64(b, e)));
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn contrast_neon(data: &mut Contrast<'_>) -> usize {
    // Dispatch validates all plane lengths and NEON availability.
    vector::contrast::<Neon>(data)
}
#[target_feature(enable = "neon")]
pub(crate) fn contrast_region_neon(data: &mut crate::kernels::ContrastRegion<'_>) -> usize {
    // Dispatch validates strided planes, compact outputs and NEON availability.
    vector::contrast_region::<Neon>(data)
}
#[target_feature(enable = "neon")]
pub(crate) fn pool_neon(data: &Pool<'_>) -> (usize, [f64; 3]) {
    // Dispatch validates all plane lengths and NEON availability.
    vector::pool::<Neon>(data)
}
#[cfg(test)]
#[target_feature(enable = "neon")]
pub(crate) fn pool_tiles_neon(
    data: &crate::kernels::TiledPool<'_>,
    tiles: &mut [[f64; 3]],
    stride: usize,
) {
    // Dispatch validates all strip strides, output lengths and NEON availability.
    vector::pool_tiles::<Neon>(data, tiles, stride);
}
#[target_feature(enable = "neon")]
pub(crate) fn pool_parts_neon(
    data: &crate::kernels::TiledPool<'_>,
    parts: &mut [[f64; 3]],
    stride: usize,
) {
    // Dispatch validates the strip strides, output length and CPU availability.
    vector::pool_parts::<Neon>(data, parts, stride);
}
#[cfg(test)]
#[target_feature(enable = "neon")]
pub(crate) fn powers_neon(input: &[f32], output: &mut [f32], exponent: f32) -> usize {
    // The test wrapper checks lengths, domain, and NEON availability.
    vector::powers::<Neon>(input, output, exponent)
}
