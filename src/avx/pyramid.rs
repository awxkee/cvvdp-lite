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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) fn reduce_vertical_avx2(rows: &Rows<'_, 5>, dst: &mut [f32]) -> usize {
    assert!(rows.iter().all(|r| r.len() >= dst.len()));
    let done = dst.len() / 8 * 8;
    let [r0, r1, r2, r3, r4] = rows.map(|r| r.as_chunks::<8>().0);
    let outer = r0.iter().zip(r4);
    let inner = r1.iter().zip(r3);
    // SAFETY: CPU-validated dispatch enables AVX2. All rows cover
    // dst, and the loop only loads/stores complete in-bounds vectors.
    unsafe {
        for ((((r0, r4), (r1, r3)), r2), dst) in
            outer.zip(inner).zip(r2).zip(dst.as_chunks_mut::<8>().0)
        {
            let r0 = _mm256_loadu_ps(r0.as_ptr());
            let r1 = _mm256_loadu_ps(r1.as_ptr());
            let r2 = _mm256_loadu_ps(r2.as_ptr());
            let r3 = _mm256_loadu_ps(r3.as_ptr());
            let r4 = _mm256_loadu_ps(r4.as_ptr());
            let value = _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(_mm256_add_ps(r0, r4), _mm256_set1_ps(0.05)),
                    _mm256_mul_ps(_mm256_add_ps(r1, r3), _mm256_set1_ps(0.25)),
                ),
                _mm256_mul_ps(r2, _mm256_set1_ps(0.4)),
            );
            _mm256_storeu_ps(dst.as_mut_ptr(), value);
        }
    }
    done
}

#[target_feature(enable = "avx2")]
pub(crate) fn expand_even_avx2(rows: &Rows<'_, 3>, dst: &mut [f32]) -> usize {
    assert!(rows.iter().all(|r| r.len() >= dst.len()));
    let done = dst.len() / 8 * 8;
    let [r0, r1, r2] = rows.map(|r| r.as_chunks::<8>().0);
    // SAFETY: CPU-validated dispatch enables AVX2. All rows cover
    // dst, and the loop only loads/stores complete in-bounds vectors.
    unsafe {
        for (((r0, r2), r1), dst) in r0.iter().zip(r2).zip(r1).zip(dst.as_chunks_mut::<8>().0) {
            let r0 = _mm256_loadu_ps(r0.as_ptr());
            let r1 = _mm256_loadu_ps(r1.as_ptr());
            let r2 = _mm256_loadu_ps(r2.as_ptr());
            let value = _mm256_add_ps(
                _mm256_mul_ps(_mm256_add_ps(r0, r2), _mm256_set1_ps(0.1)),
                _mm256_mul_ps(r1, _mm256_set1_ps(0.8)),
            );
            _mm256_storeu_ps(dst.as_mut_ptr(), value);
        }
    }
    done
}

#[target_feature(enable = "avx2")]
pub(crate) fn expand_odd_avx2(rows: &Rows<'_, 2>, dst: &mut [f32]) -> usize {
    assert!(rows.iter().all(|r| r.len() >= dst.len()));
    let done = dst.len() / 8 * 8;
    let [r0, r1] = rows.map(|r| r.as_chunks::<8>().0);
    // SAFETY: CPU-validated dispatch enables AVX2. All rows cover
    // dst, and the loop only loads/stores complete in-bounds vectors.
    unsafe {
        for ((r0, r1), dst) in r0.iter().zip(r1).zip(dst.as_chunks_mut::<8>().0) {
            let r0 = _mm256_loadu_ps(r0.as_ptr());
            let r1 = _mm256_loadu_ps(r1.as_ptr());
            let value = _mm256_mul_ps(_mm256_add_ps(r0, r1), _mm256_set1_ps(0.5));
            _mm256_storeu_ps(dst.as_mut_ptr(), value);
        }
    }
    done
}

#[target_feature(enable = "avx2")]
pub(crate) fn reduce_horizontal_avx2(src: &[f32], dst: &mut [f32]) -> usize {
    assert!(src.len() >= 2 * dst.len() + 3);
    // Deinterleaving reads one extra odd sample beyond the final outer tap.
    let groups = dst.len().min(src.len().saturating_sub(4) / 2) / 8;
    let (groups_out, _) = dst[..8 * groups].as_chunks_mut::<8>();
    let (pairs, tail) = groups_out.as_chunks_mut::<2>();
    for (i, [first, second]) in pairs.iter_mut().enumerate() {
        // SAFETY: The group count bounds the furthest tap of both vectors.
        let taps = unsafe { &*src.as_ptr().add(32 * i).cast::<[f32; 36]>() };
        reduce_group(taps.first_chunk().unwrap(), first);
        reduce_group(taps.last_chunk().unwrap(), second);
    }
    if let [last] = tail {
        // SAFETY: The last group also fits the validated input length.
        let taps = unsafe { &*src.as_ptr().add(16 * (groups - 1)).cast::<[f32; 20]>() };
        reduce_group(taps, last);
    }
    8 * groups
}

#[inline]
#[target_feature(enable = "avx2")]
fn reduce_group(src: &[f32; 20], dst: &mut [f32; 8]) {
    let (a, b) = deinterleave(src.first_chunk().unwrap());
    let (c, d) = deinterleave(src[2..].first_chunk().unwrap());
    let (e, _) = deinterleave(src.last_chunk().unwrap());
    let value = _mm256_add_ps(
        _mm256_add_ps(
            _mm256_mul_ps(_mm256_add_ps(a, e), _mm256_set1_ps(0.05)),
            _mm256_mul_ps(_mm256_add_ps(b, d), _mm256_set1_ps(0.25)),
        ),
        _mm256_mul_ps(c, _mm256_set1_ps(0.4)),
    );
    // SAFETY: The fixed-size arrays cover every vector load and store.
    unsafe {
        _mm256_storeu_ps(dst.as_mut_ptr(), value);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn expand_horizontal_avx2(src: &[f32], dst: &mut [f32]) -> usize {
    assert!(dst.len().is_multiple_of(2));
    let samples = dst.len() / 2;
    assert!(src.len() >= samples + 2);
    let groups = samples / 8;
    let (groups_out, _) = dst[..16 * groups].as_chunks_mut::<16>();
    let (pairs, tail) = groups_out.as_chunks_mut::<2>();
    for (i, [first, second]) in pairs.iter_mut().enumerate() {
        // SAFETY: The group count bounds the furthest tap of both vectors.
        let taps = unsafe { &*src.as_ptr().add(16 * i).cast::<[f32; 18]>() };
        expand_group(taps.first_chunk().unwrap(), first);
        expand_group(taps.last_chunk().unwrap(), second);
    }
    if let [last] = tail {
        // SAFETY: The last group also fits the validated input length.
        let taps = unsafe { &*src.as_ptr().add(8 * (groups - 1)).cast::<[f32; 10]>() };
        expand_group(taps, last);
    }
    16 * groups
}

#[inline]
#[target_feature(enable = "avx2")]
fn expand_group(src: &[f32; 10], dst: &mut [f32; 16]) {
    // SAFETY: The fixed-size arrays cover every vector load and store.
    unsafe {
        let a = _mm256_loadu_ps(src.as_ptr());
        let b = _mm256_loadu_ps(src[1..].as_ptr());
        let c = _mm256_loadu_ps(src[2..].as_ptr());
        let even = _mm256_add_ps(
            _mm256_mul_ps(_mm256_add_ps(a, c), _mm256_set1_ps(0.1)),
            _mm256_mul_ps(b, _mm256_set1_ps(0.8)),
        );
        let odd = _mm256_mul_ps(_mm256_add_ps(b, c), _mm256_set1_ps(0.5));
        let lo = _mm256_unpacklo_ps(even, odd);
        let hi = _mm256_unpackhi_ps(even, odd);
        let (first, second) = dst.split_at_mut(8);
        _mm256_storeu_ps(first.as_mut_ptr(), _mm256_permute2f128_ps::<0x20>(lo, hi));
        _mm256_storeu_ps(second.as_mut_ptr(), _mm256_permute2f128_ps::<0x31>(lo, hi));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn deinterleave(src: &[f32; 16]) -> (__m256, __m256) {
    // SAFETY: The fixed-size input covers both loads, and AVX2 is enabled.
    unsafe {
        let a = _mm256_loadu_ps(src.as_ptr());
        let b = _mm256_loadu_ps(src[8..].as_ptr());
        let even = _mm256_shuffle_ps::<0x88>(a, b);
        let odd = _mm256_shuffle_ps::<0xdd>(a, b);
        (
            _mm256_castpd_ps(_mm256_permute4x64_pd::<0xd8>(_mm256_castps_pd(even))),
            _mm256_castpd_ps(_mm256_permute4x64_pd::<0xd8>(_mm256_castps_pd(odd))),
        )
    }
}
