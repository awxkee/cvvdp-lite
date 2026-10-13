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

//! Small pixel blocks with moxcms-style LUT/matrix SIMD, without row splitting.

use crate::{
    Error,
    image::{ByteTable, DKL, Output},
};
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn dkl_bytes_neon(src: &[u8], out: &mut Output<'_>, table: &ByteTable) -> usize {
    let count = out[0].len();
    assert!(out.iter().all(|p| p.len() == count));
    assert_eq!(src.len(), 3 * count);
    let done = count / 4 * 4;
    let [l, rg, yv] = out;
    let output = l
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(rg.as_chunks_mut::<4>().0)
        .zip(yv.as_chunks_mut::<4>().0);
    let pixels = src.as_chunks::<3>().0.as_chunks::<4>().0;
    // SAFETY: CPU dispatch enables NEON. Each block reads exactly
    // its RGB bytes, each u8 lookup fits 256 initialized four-float entries,
    // and all stores fit the checked, disjoint output planes.
    unsafe {
        for (pixels, ((l, rg), yv)) in pixels.iter().zip(output) {
            let [p0, p1, p2, p3] = pixels.map(|rgb| {
                let r = vld1q_f32(table[0][rgb[0] as usize].as_ptr());
                let g = vld1q_f32(table[1][rgb[1] as usize].as_ptr());
                let b = vld1q_f32(table[2][rgb[2] as usize].as_ptr());
                vaddq_f32(vaddq_f32(r, g), b)
            });
            let a = vtrn1q_f32(p0, p1);
            let b = vtrn2q_f32(p0, p1);
            let c = vtrn1q_f32(p2, p3);
            let d = vtrn2q_f32(p2, p3);
            vst1q_f32(
                l.as_mut_ptr(),
                vcombine_f32(vget_low_f32(a), vget_low_f32(c)),
            );
            vst1q_f32(
                rg.as_mut_ptr(),
                vcombine_f32(vget_low_f32(b), vget_low_f32(d)),
            );
            vst1q_f32(
                yv.as_mut_ptr(),
                vcombine_f32(vget_high_f32(a), vget_high_f32(c)),
            );
        }
    }
    done
}

#[target_feature(enable = "neon")]
pub(crate) fn dkl_linear_neon(
    src: [&[f32]; 3],
    out: &mut Output<'_>,
    scale: f32,
    black: f32,
) -> Result<usize, Error> {
    let count = out[0].len();
    assert!(out.iter().all(|p| p.len() == count));
    assert!(src.iter().all(|p| p.len() == count));
    let done = count / 4 * 4;
    let [l, rg, yv] = out;
    let output = l
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(rg.as_chunks_mut::<4>().0)
        .zip(yv.as_chunks_mut::<4>().0);
    let [r, g, b] = src.map(|p| p.as_chunks::<4>().0);
    let input = r.iter().zip(g).zip(b);
    // SAFETY: NEON is enabled. Complete vector accesses fit the
    // checked slices. Reject non-finite values before clipping or writing.
    unsafe {
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let maximum = vdupq_n_f32(f32::MAX);
        let scale = vdupq_n_f32(scale);
        let black = vdupq_n_f32(black);
        for (((r, g), b), ((l, rg), yv)) in input.zip(output) {
            let r = vld1q_f32(r.as_ptr());
            let g = vld1q_f32(g.as_ptr());
            let b = vld1q_f32(b.as_ptr());
            let valid = vandq_u32(
                vandq_u32(
                    vcleq_f32(vabsq_f32(r), maximum),
                    vcleq_f32(vabsq_f32(g), maximum),
                ),
                vcleq_f32(vabsq_f32(b), maximum),
            );
            if vminvq_u32(valid) == 0 {
                return Err(Error::NonFiniteSample);
            }
            let r = vaddq_f32(vmulq_f32(vminq_f32(vmaxq_f32(r, zero), one), scale), black);
            let g = vaddq_f32(vmulq_f32(vminq_f32(vmaxq_f32(g, zero), one), scale), black);
            let b = vaddq_f32(vmulq_f32(vminq_f32(vmaxq_f32(b, zero), one), scale), black);
            let value = vaddq_f32(
                vaddq_f32(
                    vmulq_f32(r, vdupq_n_f32(DKL[0][0])),
                    vmulq_f32(g, vdupq_n_f32(DKL[0][1])),
                ),
                vmulq_f32(b, vdupq_n_f32(DKL[0][2])),
            );
            vst1q_f32(l.as_mut_ptr(), value);
            let value = vaddq_f32(
                vaddq_f32(
                    vmulq_f32(r, vdupq_n_f32(DKL[1][0])),
                    vmulq_f32(g, vdupq_n_f32(DKL[1][1])),
                ),
                vmulq_f32(b, vdupq_n_f32(DKL[1][2])),
            );
            vst1q_f32(rg.as_mut_ptr(), value);
            let value = vaddq_f32(
                vaddq_f32(
                    vmulq_f32(r, vdupq_n_f32(DKL[2][0])),
                    vmulq_f32(g, vdupq_n_f32(DKL[2][1])),
                ),
                vmulq_f32(b, vdupq_n_f32(DKL[2][2])),
            );
            vst1q_f32(yv.as_mut_ptr(), value);
        }
    }
    Ok(done)
}
