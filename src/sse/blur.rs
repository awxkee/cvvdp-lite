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
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    let done = blocks.len() * N * LANES;
    for (block, dst) in blocks.iter_mut().enumerate() {
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
            for (dst, acc) in dst.iter_mut().zip(acc) {
                _mm_storeu_ps(dst.as_mut_ptr(), acc);
            }
        }
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
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    let done = blocks.len() * N * LANES;
    for (block, dst) in blocks.iter_mut().enumerate() {
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
            for (dst, acc) in dst.iter_mut().zip(acc) {
                _mm_storeu_ps(dst.as_mut_ptr(), acc);
            }
        }
    }
    done
}

/// Every output uses the same center-first, unfused sum as the per-row kernel.
#[target_feature(enable = "sse2")]
pub(crate) fn blur_vertical_region_sse2(
    rows: &[&[f32]],
    k: &[f32; WINDOW],
    width: usize,
    out: &mut [f32],
    out_stride: usize,
) {
    assert!(width >= 4);
    crate::blur::validate_vertical_region(rows, width, out, out_stride);
    // Keep only one pointer per row in the hot filter. The row slices
    // remain borrowed, and the validation covers every pointer access.
    let mut pointers = [std::ptr::null(); crate::blur::VERTICAL_BATCH + 2 * RADIUS];
    let pointers = &mut pointers[..rows.len()];
    for (pointer, row) in pointers.iter_mut().zip(rows) {
        *pointer = row.as_ptr();
    }
    // SAFETY: The validated slices cover the entire region and cannot overlap.
    unsafe { vertical_region(pointers, k, width, out.as_mut_ptr(), out_stride) };
}

#[inline(never)]
#[target_feature(enable = "sse2")]
unsafe fn vertical_region(
    rows: &[*const f32],
    k: &[f32; WINDOW],
    width: usize,
    out: *mut f32,
    out_stride: usize,
) {
    let outputs = rows.len() - 2 * RADIUS;
    // SAFETY: The caller validates the input and output footprints. Each
    // block reads WINDOW + N - 1 row pointers for N remaining output rows,
    // and its last vector starts no later than width - 4.
    unsafe {
        let last = width - 4;
        let mut x = 0;
        loop {
            // The final vector may overlap its predecessor.
            let column = x.min(last);
            let mut y = 0;
            while y + 4 <= outputs {
                vertical_block::<4>(
                    rows.as_ptr().add(y),
                    k,
                    column,
                    out.add(y * out_stride + column),
                    out_stride,
                );
                y += 4;
            }
            while y < outputs {
                vertical_block::<1>(
                    rows.as_ptr().add(y),
                    k,
                    column,
                    out.add(y * out_stride + column),
                    out_stride,
                );
                y += 1;
            }
            if column == last {
                break;
            }
            x += 4;
        }
    }
}

#[inline]
#[target_feature(enable = "sse2")]
fn vertical_block<const N: usize>(
    rows: *const *const f32,
    k: &[f32; WINDOW],
    x: usize,
    out: *mut f32,
    stride: usize,
) {
    // SAFETY: The region loop provides WINDOW + N - 1 valid row pointers,
    // four readable samples per row at x, and N writable output vectors.
    unsafe {
        let load = |y: usize| _mm_loadu_ps(rows.add(y).read().add(x));
        let center = _mm_set1_ps(k[RADIUS]);
        let mut acc: [__m128; N] = std::array::from_fn(|y| _mm_mul_ps(load(y + RADIUS), center));
        // Keep only the output accumulators live across taps.
        for i in (0..RADIUS).rev() {
            let coeff = _mm_set1_ps(k[i]);
            for (y, acc) in acc.iter_mut().enumerate() {
                let pair = _mm_add_ps(load(y + i), load(y + WINDOW - 1 - i));
                *acc = _mm_add_ps(*acc, _mm_mul_ps(pair, coeff));
            }
        }
        for (y, acc) in acc.into_iter().enumerate() {
            _mm_storeu_ps(out.add(y * stride), acc);
        }
    }
}
