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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) fn dkl_bytes_avx2(src: &[u8], out: &mut Output<'_>, table: &ByteTable) -> usize {
    let count = out[0].len();
    assert!(out.iter().all(|p| p.len() == count));
    assert_eq!(src.len(), 3 * count);
    let done = count / 8 * 8;
    let [l, rg, yv] = out;
    let output = l
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(rg.as_chunks_mut::<8>().0)
        .zip(yv.as_chunks_mut::<8>().0);
    let pixels = src.as_chunks::<3>().0.as_chunks::<2>().0.as_chunks::<4>().0;
    // SAFETY: CPU dispatch enables AVX2. Each block reads exactly
    // its RGB bytes, each u8 lookup fits 256 initialized four-float entries,
    // and all stores fit the checked, disjoint output planes.
    unsafe {
        let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for (pixels, ((l, rg), yv)) in pixels.iter().zip(output) {
            let [p0, p1, p2, p3] = pixels.map(|[first, second]| {
                let r = _mm256_insertf128_ps::<1>(
                    _mm256_castps128_ps256(_mm_loadu_ps(table[0][first[0] as usize].as_ptr())),
                    _mm_loadu_ps(table[0][second[0] as usize].as_ptr()),
                );
                let g = _mm256_insertf128_ps::<1>(
                    _mm256_castps128_ps256(_mm_loadu_ps(table[1][first[1] as usize].as_ptr())),
                    _mm_loadu_ps(table[1][second[1] as usize].as_ptr()),
                );
                let b = _mm256_insertf128_ps::<1>(
                    _mm256_castps128_ps256(_mm_loadu_ps(table[2][first[2] as usize].as_ptr())),
                    _mm_loadu_ps(table[2][second[2] as usize].as_ptr()),
                );
                _mm256_add_ps(_mm256_add_ps(r, g), b)
            });
            let a = _mm256_unpacklo_ps(p0, p1);
            let b = _mm256_unpackhi_ps(p0, p1);
            let c = _mm256_unpacklo_ps(p2, p3);
            let d = _mm256_unpackhi_ps(p2, p3);
            _mm256_storeu_ps(
                l.as_mut_ptr(),
                _mm256_permutevar8x32_ps(_mm256_shuffle_ps::<0x44>(a, c), order),
            );
            _mm256_storeu_ps(
                rg.as_mut_ptr(),
                _mm256_permutevar8x32_ps(_mm256_shuffle_ps::<0xee>(a, c), order),
            );
            _mm256_storeu_ps(
                yv.as_mut_ptr(),
                _mm256_permutevar8x32_ps(_mm256_shuffle_ps::<0x44>(b, d), order),
            );
        }
    }
    done
}

#[target_feature(enable = "avx2")]
pub(crate) fn dkl_linear_avx2(
    src: [&[f32]; 3],
    out: &mut Output<'_>,
    scale: f32,
    black: f32,
) -> Result<usize, Error> {
    let count = out[0].len();
    assert!(out.iter().all(|p| p.len() == count));
    assert!(src.iter().all(|p| p.len() == count));
    let done = count / 8 * 8;
    let [l, rg, yv] = out;
    let output = l
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(rg.as_chunks_mut::<8>().0)
        .zip(yv.as_chunks_mut::<8>().0);
    let [r, g, b] = src.map(|p| p.as_chunks::<8>().0);
    let input = r.iter().zip(g).zip(b);
    // SAFETY: AVX2 is enabled. Complete vector accesses fit the
    // checked slices. Reject non-finite values before clipping or writing.
    unsafe {
        let zero = _mm256_set1_ps(0.0);
        let one = _mm256_set1_ps(1.0);
        let maximum = _mm256_set1_ps(f32::MAX);
        let scale = _mm256_set1_ps(scale);
        let black = _mm256_set1_ps(black);
        for (((r, g), b), ((l, rg), yv)) in input.zip(output) {
            let r = _mm256_loadu_ps(r.as_ptr());
            let g = _mm256_loadu_ps(g.as_ptr());
            let b = _mm256_loadu_ps(b.as_ptr());
            let valid_r =
                _mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_andnot_ps(_mm256_set1_ps(-0.0), r), maximum);
            let valid_g =
                _mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_andnot_ps(_mm256_set1_ps(-0.0), g), maximum);
            let valid_b =
                _mm256_cmp_ps::<_CMP_LE_OQ>(_mm256_andnot_ps(_mm256_set1_ps(-0.0), b), maximum);
            let valid = _mm256_and_ps(_mm256_and_ps(valid_r, valid_g), valid_b);
            if _mm256_movemask_ps(valid) != 255 {
                return Err(Error::NonFiniteSample);
            }
            let r = _mm256_add_ps(
                _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(r, zero), one), scale),
                black,
            );
            let g = _mm256_add_ps(
                _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(g, zero), one), scale),
                black,
            );
            let b = _mm256_add_ps(
                _mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(b, zero), one), scale),
                black,
            );
            let value = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(r, _mm256_set1_ps(DKL[0][0])),
                    _mm256_mul_ps(g, _mm256_set1_ps(DKL[0][1])),
                ),
                _mm256_mul_ps(b, _mm256_set1_ps(DKL[0][2])),
            );
            _mm256_storeu_ps(l.as_mut_ptr(), value);
            let value = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(r, _mm256_set1_ps(DKL[1][0])),
                    _mm256_mul_ps(g, _mm256_set1_ps(DKL[1][1])),
                ),
                _mm256_mul_ps(b, _mm256_set1_ps(DKL[1][2])),
            );
            _mm256_storeu_ps(rg.as_mut_ptr(), value);
            let value = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(r, _mm256_set1_ps(DKL[2][0])),
                    _mm256_mul_ps(g, _mm256_set1_ps(DKL[2][1])),
                ),
                _mm256_mul_ps(b, _mm256_set1_ps(DKL[2][2])),
            );
            _mm256_storeu_ps(yv.as_mut_ptr(), value);
        }
    }
    Ok(done)
}
