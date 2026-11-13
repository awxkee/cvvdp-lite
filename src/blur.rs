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

//! CVVDP's sigma-3, 13-tap phase uncertainty filter. The symmetric SIMD
//! kernels and streaming row ring follow the main tree's Gaussian blur.

use crate::{Backend, pyramid::ensure_len};

pub(crate) const RADIUS: usize = 6;
pub(crate) const WINDOW: usize = 2 * RADIUS + 1;
/// A multiple of the widest metric SIMD vector, so flattened strips keep
/// the same vector boundaries as a full band even at odd widths.
pub(crate) const MASK_STRIP_ROWS: usize = 8;
// While emitting a strip, retain the preceding halo and both input strips.
const MASK_RING_ROWS: usize = 2 * MASK_STRIP_ROWS + RADIUS;
const _: () = assert!(RADIUS < MASK_STRIP_ROWS && MASK_STRIP_ROWS <= VERTICAL_BATCH);
pub(crate) type BlurRows<'a> = [&'a [f32]; WINDOW];
/// Most output rows one [`GaussianBlur::vertical_rows`] call takes.
pub(crate) const VERTICAL_BATCH: usize = 32;
/// Whole-region vertical pass over validated rows and a strided output.
type VerticalRegionFn = unsafe fn(&[&[f32]], &[f32; WINDOW], usize, &mut [f32], usize);

#[derive(Clone, Copy)]
struct Kernels {
    horizontal: fn(&[f32], &[f32; WINDOW], &mut [f32]) -> usize,
    vertical: fn(&BlurRows<'_>, &[f32; WINDOW], &mut [f32]) -> usize,
    /// Optional region kernel for batched vertical passes (≥ 4 columns).
    vertical_region: Option<VerticalRegionFn>,
}

impl Kernels {
    #[inline]
    fn horizontal_row(&self, src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) {
        let done = (self.horizontal)(src, k, dst);
        if done < dst.len() {
            horizontal_scalar::<false>(&src[done..], k, &mut dst[done..]);
        }
    }

    #[allow(unsafe_code)]
    fn for_backend(backend: Backend) -> Self {
        assert!(backend.is_available());
        // SAFETY: Selection happens once for a CPU-validated Workspace backend.
        // Each entry checks slice bounds before accessing complete SIMD vectors.
        match backend {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            Backend::Neon => Self {
                horizontal: |s, k, d| unsafe { crate::neon::blur_horizontal_neon(s, k, d) },
                vertical: |s, k, d| unsafe { crate::neon::blur_vertical_neon(s, k, d) },
                vertical_region: Some(crate::neon::blur_vertical_region_neon),
            },
            #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
            Backend::Avx2 if std::arch::is_x86_feature_detected!("fma") => Self {
                horizontal: |s, k, d| unsafe { crate::avx::blur_horizontal_fma_avx2(s, k, d) },
                vertical: |s, k, d| unsafe { crate::avx::blur_vertical_fma_avx2(s, k, d) },
                vertical_region: Some(crate::avx::blur_vertical_region_fma_avx2),
            },
            #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
            Backend::Avx2 => Self {
                horizontal: |s, k, d| unsafe { crate::avx::blur_horizontal_avx2(s, k, d) },
                vertical: |s, k, d| unsafe { crate::avx::blur_vertical_avx2(s, k, d) },
                vertical_region: Some(crate::avx::blur_vertical_region_avx2),
            },
            #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
            Backend::Sse2 => Self {
                horizontal: |s, k, d| unsafe { crate::sse::blur_horizontal_sse2(s, k, d) },
                vertical: |s, k, d| unsafe { crate::sse::blur_vertical_sse2(s, k, d) },
                vertical_region: Some(crate::sse::blur_vertical_region_sse2),
            },
            Backend::Scalar => Self {
                horizontal: |_, _, _| 0,
                vertical: |_, _, _| 0,
                vertical_region: None,
            },
            #[allow(unreachable_patterns)]
            _ => unreachable!("unavailable backend rejected above"),
        }
    }
}

pub(crate) struct GaussianBlur {
    ring: [Vec<f32>; 3],
    padded: Vec<f32>,
    kernel: [f32; WINDOW],
    kernels: Kernels,
}

/// Normalized sigma-3 weights, center at `RADIUS`.
pub(crate) fn kernel() -> [f32; WINDOW] {
    let mut kernel = std::array::from_fn(|i| (-(i as f32 - RADIUS as f32).powi(2) / 18.0).exp());
    let sum: f32 = kernel.iter().sum();
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

#[inline]
pub(crate) fn validate_vertical_region(
    rows: &[&[f32]],
    width: usize,
    out: &[f32],
    out_stride: usize,
) -> usize {
    assert!((WINDOW..=VERTICAL_BATCH + 2 * RADIUS).contains(&rows.len()));
    assert!(out_stride >= width && rows.iter().all(|row| row.len() >= width));
    let outputs = rows.len() - 2 * RADIUS;
    let required = (outputs - 1)
        .checked_mul(out_stride)
        .and_then(|offset| offset.checked_add(width));
    assert!(required.is_some_and(|required| required <= out.len()));
    outputs
}

impl GaussianBlur {
    pub(crate) fn new(backend: Backend) -> Self {
        Self {
            ring: std::array::from_fn(|_| Vec::new()),
            padded: Vec::new(),
            kernel: kernel(),
            kernels: Kernels::for_backend(backend),
        }
    }

    /// One horizontal output row from `padded` (`RADIUS` reflected samples on
    /// each side).
    pub(crate) fn horizontal_row(&self, padded: &[f32], dst: &mut [f32]) {
        assert_eq!(padded.len(), dst.len() + 2 * RADIUS);
        self.kernels.horizontal_row(padded, &self.kernel, dst);
    }

    /// One vertical output row from its `WINDOW` source rows.
    pub(crate) fn vertical_row(&self, rows: &BlurRows<'_>, dst: &mut [f32]) {
        let done = (self.kernels.vertical)(rows, &self.kernel, dst);
        if done < dst.len() {
            let tail = rows.map(|row| &row[done..]);
            vertical_scalar::<false>(&tail, &self.kernel, &mut dst[done..]);
        }
    }

    /// Vertical pass over a batch of rows: output row `y` reads
    /// `rows[y..y + WINDOW]` (each starting at the first output column) and
    /// goes to `out[y * out_stride..][..width]`.
    pub(crate) fn vertical_rows(
        &self,
        rows: &[&[f32]],
        width: usize,
        out: &mut [f32],
        out_stride: usize,
    ) {
        if let Some(region) = self.kernels.vertical_region
            && width >= 4
        {
            // SAFETY: The kernel was selected for a supported backend.
            unsafe { region(rows, &self.kernel, width, out, out_stride) };
            return;
        }
        validate_vertical_region(rows, width, out, out_stride);
        if width != 0 {
            for (rows, dst) in rows
                .array_windows::<WINDOW>()
                .zip(out.chunks_mut(out_stride))
            {
                self.vertical_row(&rows.map(|row| &row[..width]), &mut dst[..width]);
            }
        }
    }

    pub(crate) fn reserve(&mut self, width: usize, height: usize) {
        if blur_active(width, height) {
            for plane in &mut self.ring {
                ensure_len(plane, width * height.min(MASK_RING_ROWS));
            }
            ensure_len(&mut self.padded, width + 2 * RADIUS);
        }
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        (self.ring.iter().map(Vec::capacity).sum::<usize>() + self.padded.capacity())
            * std::mem::size_of::<f32>()
    }

    /// Filter consecutive mask rows into the ring. The caller emits any
    /// dependent vertical output before inserting enough rows to wrap.
    pub(crate) fn horizontal_mask_rows(&mut self, mask: [&[f32]; 3], first_y: usize, width: usize) {
        let padded = &mut self.padded[..width + 2 * RADIUS];
        for (mask, ring) in mask.into_iter().zip(&mut self.ring) {
            for (offset, row) in mask.chunks_exact(width).enumerate() {
                padded[RADIUS..width + RADIUS].copy_from_slice(row);
                // Reflection excludes the edge sample.
                for i in 0..RADIUS {
                    padded[i] = row[RADIUS - i];
                    padded[width + RADIUS + i] = row[width - 2 - i];
                }
                let start = (first_y + offset) % MASK_RING_ROWS * width;
                let dst = &mut ring[start..start + width];
                self.kernels.horizontal_row(padded, &self.kernel, dst);
            }
        }
    }

    /// Emit a strip whose horizontal rows and reflected halo are in the ring.
    pub(crate) fn vertical_mask_rows(
        &self,
        out: [&mut [f32]; 3],
        first_y: usize,
        width: usize,
        height: usize,
    ) {
        let count = out[0].len() / width + 2 * RADIUS;
        let mut offsets = [0; MASK_STRIP_ROWS + 2 * RADIUS];
        for (i, offset) in offsets[..count].iter_mut().enumerate() {
            let y = reflect_exclusive((first_y + i) as isize - RADIUS as isize, height);
            *offset = y % MASK_RING_ROWS * width;
        }
        for (out, ring) in out.into_iter().zip(&self.ring) {
            let mut rows = [&[][..]; MASK_STRIP_ROWS + 2 * RADIUS];
            for (row, &offset) in rows[..count].iter_mut().zip(&offsets) {
                *row = &ring[offset..offset + width];
            }
            self.vertical_rows(&rows[..count], width, out, width);
        }
    }
}

pub(crate) fn blur_active(width: usize, height: usize) -> bool {
    width > RADIUS && height > RADIUS
}

/// Exclusive reflection for a coordinate within one radius of the image.
#[inline]
pub(crate) fn reflect_exclusive(i: isize, n: usize) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i >= n as isize {
        2 * n - i as usize - 2
    } else {
        i as usize
    }
}

// Center plus six symmetric pairs, accumulated from nearest to farthest.
// Fused kernels handle their own tails so each pixel rounds identically.
#[inline(always)]
pub(crate) fn horizontal_scalar<const FMA: bool>(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) {
    for (src, dst) in src.array_windows::<WINDOW>().zip(dst) {
        let mut sum = src[RADIUS] * k[RADIUS];
        for i in (0..RADIUS).rev() {
            let pair = src[i] + src[WINDOW - 1 - i];
            sum = if FMA {
                pair.mul_add(k[i], sum)
            } else {
                sum + pair * k[i]
            };
        }
        *dst = sum;
    }
}

#[inline(always)]
pub(crate) fn vertical_scalar<const FMA: bool>(
    rows: &BlurRows<'_>,
    k: &[f32; WINDOW],
    dst: &mut [f32],
) {
    for (x, dst) in dst.iter_mut().enumerate() {
        let mut sum = rows[RADIUS][x] * k[RADIUS];
        for i in (0..RADIUS).rev() {
            let pair = rows[i][x] + rows[WINDOW - 1 - i][x];
            sum = if FMA {
                pair.mul_add(k[i], sum)
            } else {
                sum + pair * k[i]
            };
        }
        *dst = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent literal 13-tap convolution, with modular exclusive reflection
    // and a full intermediate plane. This also catches early ring overwrite.
    fn reference(src: &[f32], width: usize, height: usize, kernel: &[f32; WINDOW]) -> Vec<f32> {
        if width <= RADIUS || height <= RADIUS {
            return src.to_vec();
        }
        let reflect = |i: isize, n: usize| {
            let period = 2 * (n as isize - 1);
            let i = i.rem_euclid(period) as usize;
            i.min(period as usize - i)
        };
        let mut tmp = vec![0.0; src.len()];
        let mut dst = vec![0.0; src.len()];
        for y in 0..height {
            for x in 0..width {
                for (i, &k) in kernel.iter().enumerate() {
                    let sx = reflect(x as isize + i as isize - RADIUS as isize, width);
                    tmp[y * width + x] += src[y * width + sx] * k;
                }
            }
        }
        for y in 0..height {
            for x in 0..width {
                for (i, &k) in kernel.iter().enumerate() {
                    let sy = reflect(y as isize + i as isize - RADIUS as isize, height);
                    dst[y * width + x] += tmp[sy * width + x] * k;
                }
            }
        }
        dst
    }

    fn stream_masks(
        blur: &mut GaussianBlur,
        src: [&[f32]; 3],
        width: usize,
        height: usize,
    ) -> [Vec<f32>; 3] {
        blur.reserve(width, height);
        if width <= RADIUS || height <= RADIUS {
            return src.map(<[f32]>::to_vec);
        }
        let mut dst = std::array::from_fn(|_| vec![0.0; width * height]);
        let mut next_y = 0;
        for y in (0..height).step_by(MASK_STRIP_ROWS) {
            let end = (y + MASK_STRIP_ROWS).min(height);
            while next_y < (end + RADIUS).min(height) {
                let next_end = (next_y + MASK_STRIP_ROWS).min(height);
                blur.horizontal_mask_rows(
                    src.map(|plane| &plane[next_y * width..next_end * width]),
                    next_y,
                    width,
                );
                next_y = next_end;
            }
            blur.vertical_mask_rows(
                dst.each_mut()
                    .map(|plane| &mut plane[y * width..end * width]),
                y,
                width,
                height,
            );
        }
        dst
    }

    #[test]
    fn streamed_blur_matches_literal_convolution_and_reuses_storage() {
        for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
            if !backend.is_available() {
                continue;
            }
            let mut blur = GaussianBlur::new(backend);
            blur.reserve(69, 51);
            let layout = |blur: &GaussianBlur| {
                [&blur.ring[0], &blur.ring[1], &blur.ring[2], &blur.padded]
                    .map(|v| (v.as_ptr(), v.capacity()))
            };
            let reserved = layout(&blur);
            for width in 1..=69 {
                for height in [1, 6, 7, 8, 9, 14, 16, 17, 21, 22, 23, 24, 25, 32, 33, 51] {
                    let original = std::array::from_fn(|c| {
                        (0..width * height)
                            .map(|i| ((i * 31 + i / width * 73 + c * 19) % 257) as f32 / 17.0)
                            .collect::<Vec<_>>()
                    });
                    let actual = stream_masks(
                        &mut blur,
                        original.each_ref().map(Vec::as_slice),
                        width,
                        height,
                    );
                    for (c, (actual, original)) in actual.iter().zip(&original).enumerate() {
                        let expected = reference(original, width, height, &blur.kernel);
                        for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                            assert!(
                                (a - b).abs() < 0.000006,
                                "{backend:?} {width}x{height} channel {c} [{i}]: {a} != {b}"
                            );
                        }
                    }
                    assert_eq!(reserved, layout(&blur));
                }
            }
        }
    }

    #[test]
    fn preserves_constant_and_skips_small_bands_exactly() {
        for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
            if !backend.is_available() {
                continue;
            }
            let mut blur = GaussianBlur::new(backend);
            for (width, height) in [(1, 17), (17, 6), (6, 17), (7, 7), (17, 33)] {
                let plane = vec![3.0; width * height];
                let [first, ..] = stream_masks(&mut blur, [&plane; 3], width, height);
                if width <= RADIUS || height <= RADIUS {
                    assert_eq!(first, plane);
                    assert_eq!(blur.allocated_bytes(), 0);
                } else {
                    assert!(first.iter().all(|&v| (v - 3.0).abs() < 0.000002));
                }
            }
        }
    }

    #[test]
    fn blur_regions_rows_and_horizontal_segments_match_tap_order_bit_exactly() {
        const GUARD: f32 = -12345.0;
        let k = kernel();
        for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
            if !backend.is_available() {
                continue;
            }
            let fused = match backend {
                Backend::Neon => true,
                #[cfg(all(feature = "avx", any(target_arch = "x86", target_arch = "x86_64")))]
                Backend::Avx2 => std::arch::is_x86_feature_detected!("fma"),
                _ => false,
            };
            let expected_sample = |taps: &[f32; WINDOW]| {
                let mut value = taps[RADIUS] * k[RADIUS];
                for i in (0..RADIUS).rev() {
                    let pair = taps[i] + taps[WINDOW - 1 - i];
                    value = if fused {
                        pair.mul_add(k[i], value)
                    } else {
                        value + pair * k[i]
                    };
                }
                value
            };
            let blur = GaussianBlur::new(backend);
            for width in 1..=40 {
                let padded: Vec<f32> = (0..width + 2 * RADIUS)
                    .map(|x| ((x * 31 + width) % 97) as f32 / 13.0)
                    .collect();
                let expected: Vec<u32> = padded
                    .array_windows::<WINDOW>()
                    .map(|taps| expected_sample(taps).to_bits())
                    .collect();
                let mut horizontal = vec![0.0; width];
                blur.horizontal_row(&padded, &mut horizontal);
                assert_eq!(
                    horizontal.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected,
                    "{backend:?} horizontal width={width}"
                );
                for split in 1..width {
                    let (left, right) = horizontal.split_at_mut(split);
                    left.fill(GUARD);
                    right.fill(GUARD);
                    blur.horizontal_row(&padded[..split + 2 * RADIUS], left);
                    blur.horizontal_row(&padded[split..], right);
                    assert_eq!(
                        horizontal.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected,
                        "{backend:?} horizontal width={width} split={split}"
                    );
                }
                for outputs in 1..=VERTICAL_BATCH {
                    let count = outputs + 2 * RADIUS;
                    let source: Vec<Vec<f32>> = (0..count)
                        .map(|j| {
                            (0..width + 3)
                                .map(|i| ((i * 31 + j * 17 + width) % 97) as f32 / 13.0)
                                .collect()
                        })
                        .collect();
                    let rows: Vec<&[f32]> = source.iter().map(|row| &row[3..]).collect();
                    let out_stride = width + 2;
                    let len = (outputs - 1) * out_stride + width + 1;
                    let mut region = vec![GUARD; len];
                    let mut per_row = region.clone();
                    let mut expected = vec![GUARD.to_bits(); len];
                    blur.vertical_rows(&rows, width, &mut region, out_stride);
                    for (y, window) in rows.array_windows::<WINDOW>().enumerate() {
                        let offset = y * out_stride;
                        blur.vertical_row(window, &mut per_row[offset..offset + width]);
                        for (x, value) in expected[offset..offset + width].iter_mut().enumerate() {
                            let taps = window.map(|row| row[x]);
                            *value = expected_sample(&taps).to_bits();
                        }
                    }
                    assert_eq!(
                        region.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected,
                        "{backend:?} vertical region width={width} outputs={outputs}"
                    );
                    assert_eq!(
                        per_row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected,
                        "{backend:?} vertical row width={width} outputs={outputs}"
                    );
                }
            }
        }
    }
}
