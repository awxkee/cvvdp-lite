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

//! Interior pyramid kernels. Scalar border handling lives in `pyramid`.

use crate::pyramid::Rows;
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn reduce_vertical_neon(rows: &Rows<'_, 5>, dst: &mut [f32]) -> usize {
    assert!(rows.iter().all(|r| r.len() >= dst.len()));
    let done = dst.len() / 4 * 4;
    let [r0, r1, r2, r3, r4] = rows.map(|r| r.as_chunks::<4>().0);
    let outer = r0.iter().zip(r4);
    let inner = r1.iter().zip(r3);
    // SAFETY: CPU-validated dispatch enables NEON. All rows cover
    // dst, and the loop only loads/stores complete in-bounds vectors.
    unsafe {
        for ((((r0, r4), (r1, r3)), r2), dst) in
            outer.zip(inner).zip(r2).zip(dst.as_chunks_mut::<4>().0)
        {
            let r0 = vld1q_f32(r0.as_ptr());
            let r1 = vld1q_f32(r1.as_ptr());
            let r2 = vld1q_f32(r2.as_ptr());
            let r3 = vld1q_f32(r3.as_ptr());
            let r4 = vld1q_f32(r4.as_ptr());
            let value = vaddq_f32(
                vaddq_f32(
                    vmulq_f32(vaddq_f32(r0, r4), vdupq_n_f32(0.05)),
                    vmulq_f32(vaddq_f32(r1, r3), vdupq_n_f32(0.25)),
                ),
                vmulq_f32(r2, vdupq_n_f32(0.4)),
            );
            vst1q_f32(dst.as_mut_ptr(), value);
        }
    }
    done
}

#[target_feature(enable = "neon")]
pub(crate) fn expand_even_neon(rows: &Rows<'_, 3>, dst: &mut [f32]) -> usize {
    assert!(rows.iter().all(|r| r.len() >= dst.len()));
    let done = dst.len() / 4 * 4;
    let [r0, r1, r2] = rows.map(|r| r.as_chunks::<4>().0);
    // SAFETY: CPU-validated dispatch enables NEON. All rows cover
    // dst, and the loop only loads/stores complete in-bounds vectors.
    unsafe {
        for (((r0, r2), r1), dst) in r0.iter().zip(r2).zip(r1).zip(dst.as_chunks_mut::<4>().0) {
            let r0 = vld1q_f32(r0.as_ptr());
            let r1 = vld1q_f32(r1.as_ptr());
            let r2 = vld1q_f32(r2.as_ptr());
            let value = vaddq_f32(
                vmulq_f32(vaddq_f32(r0, r2), vdupq_n_f32(0.1)),
                vmulq_f32(r1, vdupq_n_f32(0.8)),
            );
            vst1q_f32(dst.as_mut_ptr(), value);
        }
    }
    done
}

#[target_feature(enable = "neon")]
pub(crate) fn expand_odd_neon(rows: &Rows<'_, 2>, dst: &mut [f32]) -> usize {
    assert!(rows.iter().all(|r| r.len() >= dst.len()));
    let done = dst.len() / 4 * 4;
    let [r0, r1] = rows.map(|r| r.as_chunks::<4>().0);
    // SAFETY: CPU-validated dispatch enables NEON. All rows cover
    // dst, and the loop only loads/stores complete in-bounds vectors.
    unsafe {
        for ((r0, r1), dst) in r0.iter().zip(r1).zip(dst.as_chunks_mut::<4>().0) {
            let r0 = vld1q_f32(r0.as_ptr());
            let r1 = vld1q_f32(r1.as_ptr());
            let value = vmulq_f32(vaddq_f32(r0, r1), vdupq_n_f32(0.5));
            vst1q_f32(dst.as_mut_ptr(), value);
        }
    }
    done
}

#[target_feature(enable = "neon")]
pub(crate) fn reduce_horizontal_neon(src: &[f32], dst: &mut [f32]) -> usize {
    assert!(src.len() >= 2 * dst.len() + 3);
    // Deinterleaving reads one extra odd sample beyond the final outer tap.
    let groups = dst.len().min(src.len().saturating_sub(4) / 2) / 4;
    let (groups_out, _) = dst[..4 * groups].as_chunks_mut::<4>();
    let (pairs, tail) = groups_out.as_chunks_mut::<2>();
    for (i, [first, second]) in pairs.iter_mut().enumerate() {
        // SAFETY: The group count bounds the furthest tap of both vectors.
        let taps = unsafe { &*src.as_ptr().add(16 * i).cast::<[f32; 20]>() };
        reduce_group(taps.first_chunk().unwrap(), first);
        reduce_group(taps.last_chunk().unwrap(), second);
    }
    if let [last] = tail {
        // SAFETY: The last group also fits the validated input length.
        let taps = unsafe { &*src.as_ptr().add(8 * (groups - 1)).cast::<[f32; 12]>() };
        reduce_group(taps, last);
    }
    4 * groups
}

#[inline]
#[target_feature(enable = "neon")]
fn reduce_group(src: &[f32; 12], dst: &mut [f32; 4]) {
    // SAFETY: The fixed-size arrays cover every vector load and store.
    unsafe {
        let float32x4x2_t(a, b) = vld2q_f32(src.as_ptr());
        let float32x4x2_t(c, d) = vld2q_f32(src[2..].as_ptr());
        let float32x4x2_t(e, _) = vld2q_f32(src[4..].as_ptr());
        let value = vaddq_f32(
            vaddq_f32(
                vmulq_f32(vaddq_f32(a, e), vdupq_n_f32(0.05)),
                vmulq_f32(vaddq_f32(b, d), vdupq_n_f32(0.25)),
            ),
            vmulq_f32(c, vdupq_n_f32(0.4)),
        );
        vst1q_f32(dst.as_mut_ptr(), value);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn expand_horizontal_neon(src: &[f32], dst: &mut [f32]) -> usize {
    assert!(dst.len().is_multiple_of(2));
    let samples = dst.len() / 2;
    assert!(src.len() >= samples + 2);
    let groups = samples / 4;
    let (groups_out, _) = dst[..8 * groups].as_chunks_mut::<8>();
    let (pairs, tail) = groups_out.as_chunks_mut::<2>();
    for (i, [first, second]) in pairs.iter_mut().enumerate() {
        // SAFETY: The group count bounds the furthest tap of both vectors.
        let taps = unsafe { &*src.as_ptr().add(8 * i).cast::<[f32; 10]>() };
        expand_group(taps.first_chunk().unwrap(), first);
        expand_group(taps.last_chunk().unwrap(), second);
    }
    if let [last] = tail {
        // SAFETY: The last group also fits the validated input length.
        let taps = unsafe { &*src.as_ptr().add(4 * (groups - 1)).cast::<[f32; 6]>() };
        expand_group(taps, last);
    }
    8 * groups
}

#[inline]
#[target_feature(enable = "neon")]
fn expand_group(src: &[f32; 6], dst: &mut [f32; 8]) {
    // SAFETY: The fixed-size arrays cover every vector load and store.
    unsafe {
        let a = vld1q_f32(src.as_ptr());
        let b = vld1q_f32(src[1..].as_ptr());
        let c = vld1q_f32(src[2..].as_ptr());
        let even = vaddq_f32(
            vmulq_f32(vaddq_f32(a, c), vdupq_n_f32(0.1)),
            vmulq_f32(b, vdupq_n_f32(0.8)),
        );
        let odd = vmulq_f32(vaddq_f32(b, c), vdupq_n_f32(0.5));
        vst2q_f32(dst.as_mut_ptr(), float32x4x2_t(even, odd));
    }
}
