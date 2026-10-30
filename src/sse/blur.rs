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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse2")]
pub(crate) fn blur_horizontal_sse2(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    assert!(src.len() >= dst.len() + 2 * RADIUS);
    let done = horizontal::<4>(src, k, dst);
    done + horizontal::<1>(&src[done..], k, &mut dst[done..])
}

#[inline]
#[target_feature(enable = "sse2")]
fn horizontal<const N: usize>(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let center = _mm_set1_ps(k[RADIUS]);
    let mut done = 0;
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    for (block, dst) in blocks.iter_mut().enumerate() {
        let dst = dst.as_flattened_mut();
        let x = block * N * LANES;
        // Bounds are validated above; every vector fits the padded row or
        // one of the WINDOW rows, and every store fits this output chunk.
        unsafe {
            let mut acc: [__m128; N] = std::array::from_fn(|lane| {
                _mm_mul_ps(
                    _mm_loadu_ps(src.as_ptr().add(x + lane * LANES + RADIUS)),
                    center,
                )
            });
            for i in (0..RADIUS).rev() {
                let coeff = _mm_set1_ps(k[i]);
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let a = _mm_loadu_ps(src.as_ptr().add(x + lane * LANES + i));
                    let b = _mm_loadu_ps(src.as_ptr().add(x + lane * LANES + WINDOW - 1 - i));
                    *acc = _mm_add_ps(*acc, _mm_mul_ps(_mm_add_ps(a, b), coeff));
                }
            }
            for (lane, acc) in acc.into_iter().enumerate() {
                _mm_storeu_ps(dst.as_mut_ptr().add(lane * LANES), acc);
            }
        }
        done += N * LANES;
    }
    done
}

#[target_feature(enable = "sse2")]
pub(crate) fn blur_vertical_sse2(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    assert!(rows.iter().all(|row| row.len() >= dst.len()));
    let done = vertical::<4>(rows, k, dst);
    done + vertical::<1>(&rows.map(|row| &row[done..]), k, &mut dst[done..])
}

#[inline]
#[target_feature(enable = "sse2")]
fn vertical<const N: usize>(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    const LANES: usize = 4;
    let center = _mm_set1_ps(k[RADIUS]);
    let mut done = 0;
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    for (block, dst) in blocks.iter_mut().enumerate() {
        let dst = dst.as_flattened_mut();
        let x = block * N * LANES;
        // Bounds are validated above; every vector fits the padded row or
        // one of the WINDOW rows, and every store fits this output chunk.
        unsafe {
            let mut acc: [__m128; N] = std::array::from_fn(|lane| {
                _mm_mul_ps(
                    _mm_loadu_ps(rows[RADIUS].as_ptr().add(x + lane * LANES)),
                    center,
                )
            });
            for i in (0..RADIUS).rev() {
                let coeff = _mm_set1_ps(k[i]);
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let a = _mm_loadu_ps(rows[i].as_ptr().add(x + lane * LANES));
                    let b = _mm_loadu_ps(rows[WINDOW - 1 - i].as_ptr().add(x + lane * LANES));
                    *acc = _mm_add_ps(*acc, _mm_mul_ps(_mm_add_ps(a, b), coeff));
                }
            }
            for (lane, acc) in acc.into_iter().enumerate() {
                _mm_storeu_ps(dst.as_mut_ptr().add(lane * LANES), acc);
            }
        }
        done += N * LANES;
    }
    done
}

/// # Safety
/// The CPU feature must be available, `width >= 4`, and `rows.len() > 2 * RADIUS`.
/// Each `rows[j]` is valid for `width` samples and `out` for
/// `(rows.len() - 2 * RADIUS - 1) * out_stride + width`.
///
/// Every output is the per-row kernels' expression (center, then the pairs
/// from the outermost in, unfused), so results match them bit for bit.
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn blur_vertical_region_sse2(
    rows: &[*const f32],
    k: &[f32; WINDOW],
    width: usize,
    out: *mut f32,
    out_stride: usize,
) {
    debug_assert!(width >= 4 && rows.len() > 2 * RADIUS);
    let outputs = rows.len() - 2 * RADIUS;
    unsafe {
        let mut coeff = [_mm_set1_ps(0.0); RADIUS + 1];
        for (c, &v) in coeff.iter_mut().zip(k) {
            *c = _mm_set1_ps(v);
        }
        let tap = |window: &[__m128]| {
            let mut acc = _mm_mul_ps(window[RADIUS], coeff[RADIUS]);
            for i in (0..RADIUS).rev() {
                acc = _mm_add_ps(
                    acc,
                    _mm_mul_ps(_mm_add_ps(window[i], window[WINDOW - 1 - i]), coeff[i]),
                );
            }
            acc
        };
        for dst in [out] {
            // Complete vectors, then one overlapping vector for a short tail.
            let last = width - 4;
            let mut x = 0;
            loop {
                let column = x.min(last);
                let mut y = 0;
                while y + 4 <= outputs {
                    let mut window = [_mm_set1_ps(0.0); WINDOW + 3];
                    for (j, value) in window.iter_mut().enumerate() {
                        *value = _mm_loadu_ps(rows[y + j].add(column));
                    }
                    for o in 0..4 {
                        _mm_storeu_ps(
                            dst.add((y + o) * out_stride + column),
                            tap(&window[o..o + WINDOW]),
                        );
                    }
                    y += 4;
                }
                while y < outputs {
                    let mut window = [_mm_set1_ps(0.0); WINDOW];
                    for (j, value) in window.iter_mut().enumerate() {
                        *value = _mm_loadu_ps(rows[y + j].add(column));
                    }
                    _mm_storeu_ps(dst.add(y * out_stride + column), tap(&window));
                    y += 1;
                }
                if column == last {
                    break;
                }
                x += 4;
            }
        }
    }
}
