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
use crate::blur::{BlurRows, RADIUS, WINDOW, horizontal_scalar, vertical_scalar};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) fn blur_horizontal_avx2(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    assert!(src.len() >= dst.len() + 2 * RADIUS);
    // SAFETY: AVX2 is enabled and the padding was validated.
    unsafe {
        let done = horizontal::<4, false>(src, k, dst);
        done + horizontal::<1, false>(&src[done..], k, &mut dst[done..])
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn blur_horizontal_fma_avx2(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    assert!(src.len() >= dst.len() + 2 * RADIUS);
    // SAFETY: AVX2 and FMA are enabled and the padding was validated.
    let done = unsafe {
        let done = horizontal::<4, true>(src, k, dst);
        done + horizontal::<1, true>(&src[done..], k, &mut dst[done..])
    };
    if done < dst.len() {
        horizontal_scalar::<true>(&src[done..], k, &mut dst[done..]);
    }
    dst.len()
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn horizontal<const N: usize, const FMA: bool>(
    src: &[f32],
    k: &[f32; WINDOW],
    dst: &mut [f32],
) -> usize {
    const LANES: usize = 8;
    let center = _mm256_set1_ps(k[RADIUS]);
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    let done = blocks.len() * N * LANES;
    for (block, dst) in blocks.iter_mut().enumerate() {
        let x = block * N * LANES;
        // Bounds are validated above; every vector fits the padded row or
        // one of the WINDOW rows, and every store fits this output chunk.
        unsafe {
            let mut acc: [__m256; N] = std::array::from_fn(|lane| {
                _mm256_mul_ps(
                    _mm256_loadu_ps(src.as_ptr().add(x + lane * LANES + RADIUS)),
                    center,
                )
            });
            for i in (0..RADIUS).rev() {
                let coeff = _mm256_set1_ps(k[i]);
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let a = _mm256_loadu_ps(src.as_ptr().add(x + lane * LANES + i));
                    let b = _mm256_loadu_ps(src.as_ptr().add(x + lane * LANES + WINDOW - 1 - i));
                    *acc = add_tap256::<FMA>(_mm256_add_ps(a, b), coeff, *acc);
                }
            }
            for (dst, acc) in dst.iter_mut().zip(acc) {
                _mm256_storeu_ps(dst.as_mut_ptr(), acc);
            }
        }
    }
    done
}

#[target_feature(enable = "avx2")]
pub(crate) fn blur_vertical_avx2(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    assert!(rows.iter().all(|row| row.len() >= dst.len()));
    let done = vertical::<4>(rows, k, dst);
    done + vertical::<1>(&rows.map(|row| &row[done..]), k, &mut dst[done..])
}

#[inline]
#[target_feature(enable = "avx2")]
fn vertical<const N: usize>(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) -> usize {
    const LANES: usize = 8;
    let center = _mm256_set1_ps(k[RADIUS]);
    let (vectors, _) = dst.as_chunks_mut::<LANES>();
    let (blocks, _) = vectors.as_chunks_mut::<N>();
    let done = blocks.len() * N * LANES;
    for (block, dst) in blocks.iter_mut().enumerate() {
        let x = block * N * LANES;
        // Bounds are validated above; every vector fits the padded row or
        // one of the WINDOW rows, and every store fits this output chunk.
        unsafe {
            let mut acc: [__m256; N] = std::array::from_fn(|lane| {
                _mm256_mul_ps(
                    _mm256_loadu_ps(rows[RADIUS].as_ptr().add(x + lane * LANES)),
                    center,
                )
            });
            for i in (0..RADIUS).rev() {
                let coeff = _mm256_set1_ps(k[i]);
                for (lane, acc) in acc.iter_mut().enumerate() {
                    let a = _mm256_loadu_ps(rows[i].as_ptr().add(x + lane * LANES));
                    let b = _mm256_loadu_ps(rows[WINDOW - 1 - i].as_ptr().add(x + lane * LANES));
                    *acc = _mm256_add_ps(*acc, _mm256_mul_ps(_mm256_add_ps(a, b), coeff));
                }
            }
            for (dst, acc) in dst.iter_mut().zip(acc) {
                _mm256_storeu_ps(dst.as_mut_ptr(), acc);
            }
        }
    }
    done
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn blur_vertical_fma_avx2(
    rows: &BlurRows<'_>,
    k: &[f32; WINDOW],
    dst: &mut [f32],
) -> usize {
    let width = dst.len();
    if width >= 4 {
        blur_vertical_region_fma_avx2(rows, k, width, dst, width);
    } else {
        // Regional dispatch sends rows narrower than four here.
        assert!(rows.iter().all(|row| row.len() >= width));
        vertical_scalar::<true>(rows, k, dst);
    }
    width
}

#[inline(always)]
unsafe fn add_tap256<const FMA: bool>(pair: __m256, coeff: __m256, acc: __m256) -> __m256 {
    // SAFETY: The caller enables AVX2 and enables FMA for the fused version.
    unsafe {
        if FMA {
            _mm256_fmadd_ps(pair, coeff, acc)
        } else {
            _mm256_add_ps(acc, _mm256_mul_ps(pair, coeff))
        }
    }
}

#[inline(always)]
unsafe fn add_tap128<const FMA: bool>(pair: __m128, coeff: __m128, acc: __m128) -> __m128 {
    // SAFETY: The caller enables AVX2 and enables FMA for the fused version.
    unsafe {
        if FMA {
            _mm_fmadd_ps(pair, coeff, acc)
        } else {
            _mm_add_ps(acc, _mm_mul_ps(pair, coeff))
        }
    }
}

/// Every output uses the per-row kernels' expression: center, then pairs
/// from nearest to farthest.
#[target_feature(enable = "avx2")]
pub(crate) fn blur_vertical_region_avx2(
    rows: &[&[f32]],
    k: &[f32; WINDOW],
    width: usize,
    out: &mut [f32],
    out_stride: usize,
) {
    // SAFETY: AVX2 is enabled. The region is validated before SIMD access.
    unsafe { vertical_region::<false>(rows, k, width, out, out_stride) }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn blur_vertical_region_fma_avx2(
    rows: &[&[f32]],
    k: &[f32; WINDOW],
    width: usize,
    out: &mut [f32],
    out_stride: usize,
) {
    // SAFETY: AVX2 and FMA are enabled. The region is validated before SIMD access.
    unsafe { vertical_region::<true>(rows, k, width, out, out_stride) }
}

#[inline(always)]
unsafe fn vertical_region<const FMA: bool>(
    rows: &[&[f32]],
    k: &[f32; WINDOW],
    width: usize,
    out: &mut [f32],
    out_stride: usize,
) {
    // Thirty-two-bit x86 exposes eight vector registers.
    const BATCH: usize = if cfg!(target_arch = "x86_64") { 4 } else { 2 };
    const INPUT_ROWS: usize = WINDOW + BATCH - 1;
    assert!(width >= 4);
    let outputs = crate::blur::validate_vertical_region(rows, width, out, out_stride);
    let out = out.as_mut_ptr();
    let full = outputs / BATCH * BATCH;
    let batches = rows.array_windows::<INPUT_ROWS>().step_by(BATCH);
    let tail = rows[full..].array_windows::<WINDOW>();
    macro_rules! filter {
        ($lanes:literal, $splat:ident, $load:ident, $store:ident, $add:ident, $mul:ident, $add_tap:ident) => {{
            // SAFETY: The caller enables AVX2.
            let center = unsafe { $splat(k[RADIUS]) };
            // Complete vectors, then one overlapping vector for a short tail.
            let last = width - $lanes;
            let mut x = 0;
            loop {
                let column = x.min(last);
                // SAFETY: The windows select valid rows, and column plus the
                // vector width fits every input and output row.
                unsafe {
                    for (batch, input) in batches.clone().enumerate() {
                        let mut acc = [center; BATCH];
                        for (o, acc) in acc.iter_mut().enumerate() {
                            *acc = $mul($load(input[o + RADIUS].as_ptr().add(column)), center);
                        }
                        // Keep only the output sums live and load one tap pair
                        // at a time instead of retaining the whole input window.
                        for i in (0..RADIUS).rev() {
                            let coeff = $splat(k[i]);
                            for (o, acc) in acc.iter_mut().enumerate() {
                                let a = $load(input[o + i].as_ptr().add(column));
                                let b = $load(input[o + WINDOW - 1 - i].as_ptr().add(column));
                                *acc = $add_tap::<FMA>($add(a, b), coeff, *acc);
                            }
                        }
                        for (o, acc) in acc.into_iter().enumerate() {
                            $store(out.add((batch * BATCH + o) * out_stride + column), acc);
                        }
                    }
                    for (y, input) in tail.clone().enumerate() {
                        let mut acc = $mul($load(input[RADIUS].as_ptr().add(column)), center);
                        for i in (0..RADIUS).rev() {
                            let a = $load(input[i].as_ptr().add(column));
                            let b = $load(input[WINDOW - 1 - i].as_ptr().add(column));
                            acc = $add_tap::<FMA>($add(a, b), $splat(k[i]), acc);
                        }
                        $store(out.add((full + y) * out_stride + column), acc);
                    }
                }
                if column == last {
                    break;
                }
                x += $lanes;
            }
        }};
    }
    if width >= 8 {
        filter!(
            8,
            _mm256_set1_ps,
            _mm256_loadu_ps,
            _mm256_storeu_ps,
            _mm256_add_ps,
            _mm256_mul_ps,
            add_tap256
        );
    } else {
        filter!(
            4,
            _mm_set1_ps,
            _mm_loadu_ps,
            _mm_storeu_ps,
            _mm_add_ps,
            _mm_mul_ps,
            add_tap128
        );
    }
}
