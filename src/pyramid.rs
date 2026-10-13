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

//! Separable five-tap Gaussian reduction and sparse expansion. Reduction
//! reflects including edge samples; expansion replicates coarse edge samples,
//! matching the original's zero interleaving and explicit padding.

use crate::Backend;

pub(crate) type Rows<'a, const N: usize> = [&'a [f32]; N];

/// Per-workspace dispatch; border handling and scalar tails stay in this module.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    reduce_vertical: fn(&Rows<'_, 5>, &mut [f32]) -> usize,
    reduce_horizontal: fn(&[f32], &mut [f32]) -> usize,
    expand_even: fn(&Rows<'_, 3>, &mut [f32]) -> usize,
    expand_odd: fn(&Rows<'_, 2>, &mut [f32]) -> usize,
    expand_horizontal: fn(&[f32], &mut [f32]) -> usize,
}

impl Kernels {
    #[allow(unsafe_code)]
    pub(crate) fn for_backend(backend: Backend) -> Self {
        assert!(backend.is_available());
        // SAFETY: The workspace validates CPU availability before selecting
        // target-feature entries. Each entry validates its slices independently.
        match backend {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            Backend::Neon => Self {
                reduce_vertical: |s, d| unsafe { crate::neon::reduce_vertical_neon(s, d) },
                reduce_horizontal: |s, d| unsafe { crate::neon::reduce_horizontal_neon(s, d) },
                expand_even: |s, d| unsafe { crate::neon::expand_even_neon(s, d) },
                expand_odd: |s, d| unsafe { crate::neon::expand_odd_neon(s, d) },
                expand_horizontal: |s, d| unsafe { crate::neon::expand_horizontal_neon(s, d) },
            },
            #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
            Backend::Avx2 => Self {
                reduce_vertical: |s, d| unsafe { crate::avx::reduce_vertical_avx2(s, d) },
                reduce_horizontal: |s, d| unsafe { crate::avx::reduce_horizontal_avx2(s, d) },
                expand_even: |s, d| unsafe { crate::avx::expand_even_avx2(s, d) },
                expand_odd: |s, d| unsafe { crate::avx::expand_odd_avx2(s, d) },
                expand_horizontal: |s, d| unsafe { crate::avx::expand_horizontal_avx2(s, d) },
            },
            #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
            Backend::Sse2 => Self {
                reduce_vertical: |s, d| unsafe { crate::sse::reduce_vertical_sse2(s, d) },
                reduce_horizontal: |s, d| unsafe { crate::sse::reduce_horizontal_sse2(s, d) },
                expand_even: |s, d| unsafe { crate::sse::expand_even_sse2(s, d) },
                expand_odd: |s, d| unsafe { crate::sse::expand_odd_sse2(s, d) },
                expand_horizontal: |s, d| unsafe { crate::sse::expand_horizontal_sse2(s, d) },
            },
            Backend::Scalar => Self {
                reduce_vertical: |_, _| 0,
                reduce_horizontal: |_, _| 0,
                expand_even: |_, _| 0,
                expand_odd: |_, _| 0,
                expand_horizontal: |_, _| 0,
            },
            #[allow(unreachable_patterns)]
            _ => unreachable!("unavailable backend rejected above"),
        }
    }
}

pub(crate) fn reduce(
    src: &[f32],
    w: usize,
    h: usize,
    dst: &mut Vec<f32>,
    tmp: &mut Vec<f32>,
    kernels: &Kernels,
) {
    assert!(w > 0 && h > 0 && src.len() == w * h);
    let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
    ensure_len(tmp, w * nh);
    ensure_len(dst, nw * nh);
    let mut vertical = |y: usize, indices: [usize; 5]| {
        let rows = indices.map(|sy| &src[sy * w..(sy + 1) * w]);
        let out = &mut tmp[y * w..(y + 1) * w];
        let done = (kernels.reduce_vertical)(&rows, out);
        let [r0, r1, r2, r3, r4] = rows.map(|row| &row[done..]);
        let outer = r0.iter().zip(r4);
        let inner = r1.iter().zip(r3);
        for ((((a, e), (b, d)), c), dst) in outer.zip(inner).zip(r2).zip(&mut out[done..]) {
            *dst = 0.05 * (a + e) + 0.25 * (b + d) + 0.4 * c;
        }
    };
    // Only the first and final output rows need reflection. Interior row
    // offsets are direct, and SIMD spans the entire row including its columns.
    vertical(0, std::array::from_fn(|k| reflect(k as isize - 2, h)));
    let end = ((h - 1) / 2).max(1);
    for y in 1..end {
        let sy = 2 * y;
        vertical(y, [sy - 2, sy - 1, sy, sy + 1, sy + 2]);
    }
    for y in end..nh {
        vertical(
            y,
            std::array::from_fn(|k| reflect((2 * y) as isize + k as isize - 2, h)),
        );
    }
    let end = ((w - 1) / 2).max(1);
    for (row, out) in tmp[..w * nh]
        .chunks_exact(w)
        .zip(dst[..nw * nh].chunks_exact_mut(nw))
    {
        out[0] = reduce_border(row, 0);
        if end > 1 {
            let done = (kernels.reduce_horizontal)(row, &mut out[1..end]);
            let taps = row[2 * done..].array_windows::<5>().step_by(2);
            for (taps, dst) in taps.zip(&mut out[1 + done..end]) {
                *dst = 0.05 * (taps[0] + taps[4]) + 0.25 * (taps[1] + taps[3]) + 0.4 * taps[2];
            }
        }
        for (x, dst) in out.iter_mut().enumerate().skip(end) {
            *dst = reduce_border(row, x);
        }
    }
}

#[inline]
fn reduce_border(row: &[f32], x: usize) -> f32 {
    let xx = (2 * x) as isize;
    let w = row.len();
    0.05 * (row[reflect(xx - 2, w)] + row[reflect(xx + 2, w)])
        + 0.25 * (row[reflect(xx - 1, w)] + row[reflect(xx + 1, w)])
        + 0.4 * row[2 * x]
}

pub(crate) fn expand(
    src: &[f32],
    w: usize,
    h: usize,
    dst: &mut Vec<f32>,
    tmp: &mut Vec<f32>,
    kernels: &Kernels,
) {
    let (sw, sh) = (w.div_ceil(2), h.div_ceil(2));
    assert!(w > 0 && h > 0 && src.len() == sw * sh);
    ensure_len(tmp, sw * h);
    ensure_len(dst, w * h);
    // Generate even and odd rows in separate loops over pixels. Replication
    // is resolved once per coarse row, rather than inside a pixel loop.
    for (sy, pair) in tmp[..sw * h].chunks_mut(2 * sw).enumerate() {
        let before = sy.saturating_sub(1);
        let after = (sy + 1).min(sh - 1);
        let rows = [before, sy, after].map(|r| &src[r * sw..(r + 1) * sw]);
        let (out, odd) = pair.split_at_mut(sw);
        let done = (kernels.expand_even)(&rows, out);
        let [before, center, after] = rows.map(|row| &row[done..]);
        for (((before, after), center), dst) in
            before.iter().zip(after).zip(center).zip(&mut out[done..])
        {
            *dst = 0.1 * (before + after) + 0.8 * center;
        }
        if !odd.is_empty() {
            let rows = [rows[1], rows[2]];
            let done = (kernels.expand_odd)(&rows, odd);
            let [center, after] = rows.map(|row| &row[done..]);
            for ((center, after), dst) in center.iter().zip(after).zip(&mut odd[done..]) {
                *dst = 0.5 * (center + after);
            }
        }
    }
    for (row, out) in tmp[..sw * h]
        .chunks_exact(sw)
        .zip(dst[..w * h].chunks_exact_mut(w))
    {
        expand_border(row, 0, out);
        let end = sw.saturating_sub(1);
        if end > 1 {
            let done = (kernels.expand_horizontal)(row, &mut out[2..2 * end]);
            let taps = row[done / 2..].array_windows::<3>();
            for (taps, pair) in taps.zip(out[2 + done..2 * end].as_chunks_mut::<2>().0) {
                pair[0] = 0.1 * (taps[0] + taps[2]) + 0.8 * taps[1];
                pair[1] = 0.5 * (taps[1] + taps[2]);
            }
        }
        if end > 0 {
            expand_border(row, end, out);
        }
    }
}

#[inline]
fn expand_border(row: &[f32], sx: usize, out: &mut [f32]) {
    let after = (sx + 1).min(row.len() - 1);
    out[2 * sx] = 0.1 * (row[sx.saturating_sub(1)] + row[after]) + 0.8 * row[sx];
    if 2 * sx + 1 < out.len() {
        out[2 * sx + 1] = 0.5 * (row[sx] + row[after]);
    }
}

fn reflect(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    if i < 0 {
        (-i - 1) as usize
    } else if i >= n as isize {
        (2 * n) - i as usize - 1
    } else {
        i as usize
    }
}

pub(crate) fn ensure_len(buffer: &mut Vec<f32>, length: usize) {
    if buffer.len() < length {
        buffer.resize(length, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reference_reduce(src: &[f32], w: usize, h: usize, dst: &mut Vec<f32>, tmp: &mut Vec<f32>) {
        let (nw, nh) = (w.div_ceil(2), h.div_ceil(2));
        ensure_len(tmp, w * nh);
        ensure_len(dst, nw * nh);
        for y in 0..nh {
            let rows = std::array::from_fn::<_, 5, _>(|k| {
                reference_reflect((2 * y) as isize + k as isize - 2, h) * w
            });
            for x in 0..w {
                tmp[y * w + x] = 0.05 * (src[rows[0] + x] + src[rows[4] + x])
                    + 0.25 * (src[rows[1] + x] + src[rows[3] + x])
                    + 0.4 * src[rows[2] + x];
            }
        }
        for y in 0..nh {
            let row = &tmp[y * w..(y + 1) * w];
            for x in 0..nw {
                let xx = (2 * x) as isize;
                dst[y * nw + x] = 0.05
                    * (row[reference_reflect(xx - 2, w)] + row[reference_reflect(xx + 2, w)])
                    + 0.25
                        * (row[reference_reflect(xx - 1, w)] + row[reference_reflect(xx + 1, w)])
                    + 0.4 * row[2 * x];
            }
        }
    }

    fn reference_expand(src: &[f32], w: usize, h: usize, dst: &mut Vec<f32>, tmp: &mut Vec<f32>) {
        let (sw, sh) = (w.div_ceil(2), h.div_ceil(2));
        ensure_len(tmp, sw * h);
        ensure_len(dst, w * h);
        for y in 0..h {
            let sy = y / 2;
            let before = sy.saturating_sub(1) * sw;
            let center = sy * sw;
            let after = (sy + 1).min(sh - 1) * sw;
            for x in 0..sw {
                tmp[y * sw + x] = if y % 2 == 0 {
                    0.1 * (src[before + x] + src[after + x]) + 0.8 * src[center + x]
                } else {
                    0.5 * (src[center + x] + src[after + x])
                };
            }
        }
        for y in 0..h {
            let row = &tmp[y * sw..(y + 1) * sw];
            for x in 0..w {
                let sx = x / 2;
                dst[y * w + x] = if x % 2 == 0 {
                    0.1 * (row[sx.saturating_sub(1)] + row[(sx + 1).min(sw - 1)]) + 0.8 * row[sx]
                } else {
                    0.5 * (row[sx] + row[(sx + 1).min(sw - 1)])
                };
            }
        }
    }

    fn reference_reflect(i: isize, n: usize) -> usize {
        if n == 1 {
            return 0;
        }
        if i < 0 {
            (-i - 1) as usize
        } else if i >= n as isize {
            (2 * n) - i as usize - 1
        } else {
            i as usize
        }
    }

    #[test]
    fn constant_odd_even_and_degenerate_pyramids() {
        let kernels = Kernels::for_backend(Backend::detect());
        for (w, h) in [(1, 1), (1, 9), (10, 1), (3, 4), (4, 3), (63, 64), (64, 63)] {
            let src = vec![1.0; w * h];
            let (mut reduced, mut expanded, mut tmp) = (Vec::new(), Vec::new(), Vec::new());
            reduce(&src, w, h, &mut reduced, &mut tmp, &kernels);
            expand(&reduced, w, h, &mut expanded, &mut tmp, &kernels);
            assert!(
                reduced
                    .iter()
                    .chain(&expanded)
                    .all(|v| (*v - 1.0).abs() < 1e-6)
            );
        }
    }

    #[test]
    fn symmetric_reduce_duplicates_right_edge() {
        let mut reduced = Vec::new();
        reduce(
            &[0.0, 0.0, 0.0, 1.0, 1.0],
            5,
            1,
            &mut reduced,
            &mut Vec::new(),
            &Kernels::for_backend(Backend::detect()),
        );
        assert!((reduced[2] - 0.95).abs() < 1e-6);
    }

    #[test]
    fn all_backends_match_original_loops_exactly_for_borders_phases_and_tails() {
        for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
            if !backend.is_available() {
                continue;
            }
            let kernels = Kernels::for_backend(backend);
            for w in 1..=69 {
                for h in [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32, 33, 64, 65] {
                    let src: Vec<f32> = (0..w * h)
                        .map(|i| (((i * 37 + i / w * 73) % 1021) as f32 - 510.0) / 17.0)
                        .collect();
                    let (mut expected, mut actual, mut tmp) = (Vec::new(), Vec::new(), Vec::new());
                    reference_reduce(&src, w, h, &mut expected, &mut tmp);
                    reduce(&src, w, h, &mut actual, &mut tmp, &kernels);
                    assert_eq!(actual, expected, "reduce {backend:?} {w}x{h}");
                    let coarse: Vec<f32> = (0..w.div_ceil(2) * h.div_ceil(2))
                        .map(|i| (((i * 13 + i / w.div_ceil(2) * 43) % 1021) as f32 - 510.0) / 11.0)
                        .collect();
                    reference_expand(&coarse, w, h, &mut expected, &mut tmp);
                    expand(&coarse, w, h, &mut actual, &mut tmp, &kernels);
                    assert_eq!(actual, expected, "expand {backend:?} {w}x{h}");
                }
            }
        }
    }
}
