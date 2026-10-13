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
// Adapted from libblur's symmetric f32 filter1d kernels.
use crate::blur::{BlurRows, RADIUS, WINDOW};
use std::arch::aarch64::*;

#[target_feature(enable = "neon")]
pub(crate) fn blur_horizontal_neon(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    assert!(src.len() >= dst.len() + 2 * RADIUS);
    let done = horizontal::<4>(src, k, dst);
    done + horizontal::<1>(&src[done..], k, &mut dst[done..])
}

#[inline]
#[target_feature(enable = "neon")]
fn horizontal<const N: usize>(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let center = vdupq_n_f32(k[RADIUS]);
    let mut done = 0;
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    for (block, dst) in blocks.iter_mut().enumerate() {
        let dst = dst.as_flattened_mut();
        let x = block * N * LANES;
        // Bounds are validated above; every vector fits the padded row or
        // one of the WINDOW rows, and every store fits this output chunk.
        unsafe {
            let mut acc: [float32x4_t; N] = std::array::from_fn(|lane| {
                vmulq_f32(
                    vld1q_f32(src.as_ptr().add(x + lane * LANES + RADIUS)),
                    center,
                )
            });
            for i in (0..RADIUS).rev() {
                let coeff = vdupq_n_f32(k[i]);
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let a = vld1q_f32(src.as_ptr().add(x + lane * LANES + i));
                    let b = vld1q_f32(src.as_ptr().add(x + lane * LANES + WINDOW - 1 - i));
                    *acc = vaddq_f32(*acc, vmulq_f32(vaddq_f32(a, b), coeff));
                }
            }
            for (lane, acc) in acc.into_iter().enumerate() {
                vst1q_f32(dst.as_mut_ptr().add(lane * LANES), acc);
            }
        }
        done += N * LANES;
    }
    done
}

#[target_feature(enable = "neon")]
pub(crate) fn blur_vertical_neon(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    assert!(rows.iter().all(|row| row.len() >= dst.len()));
    let done = vertical::<4>(rows, k, dst);
    done + vertical::<1>(&rows.map(|row| &row[done..]), k, &mut dst[done..])
}

#[inline]
#[target_feature(enable = "neon")]
fn vertical<const N: usize>(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let center = vdupq_n_f32(k[RADIUS]);
    let mut done = 0;
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    for (block, dst) in blocks.iter_mut().enumerate() {
        let dst = dst.as_flattened_mut();
        let x = block * N * LANES;
        // Bounds are validated above; every vector fits the padded row or
        // one of the WINDOW rows, and every store fits this output chunk.
        unsafe {
            let mut acc: [float32x4_t; N] = std::array::from_fn(|lane| {
                vmulq_f32(
                    vld1q_f32(rows[RADIUS].as_ptr().add(x + lane * LANES)),
                    center,
                )
            });
            for i in (0..RADIUS).rev() {
                let coeff = vdupq_n_f32(k[i]);
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let a = vld1q_f32(rows[i].as_ptr().add(x + lane * LANES));
                    let b = vld1q_f32(rows[WINDOW - 1 - i].as_ptr().add(x + lane * LANES));
                    *acc = vaddq_f32(*acc, vmulq_f32(vaddq_f32(a, b), coeff));
                }
            }
            for (lane, acc) in acc.into_iter().enumerate() {
                vst1q_f32(dst.as_mut_ptr().add(lane * LANES), acc);
            }
        }
        done += N * LANES;
    }
    done
}
