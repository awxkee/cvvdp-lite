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
pub(crate) type BlurRows<'a> = [&'a [f32]; WINDOW];

#[derive(Clone, Copy)]
struct Kernels {
    horizontal: fn(&[f32], &[f32; WINDOW], &mut [f32]) -> usize,
    vertical: fn(&BlurRows<'_>, &[f32; WINDOW], &mut [f32]) -> usize,
}

impl Kernels {
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
            },
            #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
            Backend::Avx2 => Self {
                horizontal: |s, k, d| unsafe { crate::avx::blur_horizontal_avx2(s, k, d) },
                vertical: |s, k, d| unsafe { crate::avx::blur_vertical_avx2(s, k, d) },
            },
            #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
            Backend::Sse2 => Self {
                horizontal: |s, k, d| unsafe { crate::sse::blur_horizontal_sse2(s, k, d) },
                vertical: |s, k, d| unsafe { crate::sse::blur_vertical_sse2(s, k, d) },
            },
            Backend::Scalar => Self {
                horizontal: |_, _, _| 0,
                vertical: |_, _, _| 0,
            },
            #[allow(unreachable_patterns)]
            _ => unreachable!("unavailable backend rejected above"),
        }
    }
}

pub(crate) struct GaussianBlur {
    ring: Vec<f32>,
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

impl GaussianBlur {
    pub(crate) fn new(backend: Backend) -> Self {
        Self {
            ring: Vec::new(),
            padded: Vec::new(),
            kernel: kernel(),
            kernels: Kernels::for_backend(backend),
        }
    }

    /// One horizontal output row from `padded` (`RADIUS` reflected samples on
    /// each side), with the same per-sample arithmetic as `blur`.
    pub(crate) fn horizontal_row(&self, padded: &[f32], dst: &mut [f32]) {
        assert_eq!(padded.len(), dst.len() + 2 * RADIUS);
        let done = (self.kernels.horizontal)(padded, &self.kernel, dst);
        horizontal_scalar(&padded[done..], &self.kernel, &mut dst[done..]);
    }

    /// One vertical output row from its `WINDOW` source rows.
    pub(crate) fn vertical_row(&self, rows: &BlurRows<'_>, dst: &mut [f32]) {
        let done = (self.kernels.vertical)(rows, &self.kernel, dst);
        if done < dst.len() {
            let tail = rows.map(|row| &row[done..]);
            vertical_scalar(&tail, &self.kernel, &mut dst[done..]);
        }
    }

    pub(crate) fn reserve(&mut self, width: usize, height: usize) {
        if width > RADIUS && height > RADIUS {
            ensure_len(&mut self.ring, width * WINDOW);
            ensure_len(&mut self.padded, width + 2 * RADIUS);
        }
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        (self.ring.capacity() + self.padded.capacity()) * std::mem::size_of::<f32>()
    }

    /// Write in place after filtering every source row needed by this output.
    /// At most WINDOW consecutive horizontal rows are live, including when
    /// exclusive reflection repeats earlier rows at either vertical edge.
    pub(crate) fn blur(&mut self, mask: &mut [f32], width: usize, height: usize) {
        assert_eq!(mask.len(), width * height);
        // Preserve the original model's small-band cutoff and reflect padding.
        if width <= RADIUS || height <= RADIUS {
            return;
        }
        self.reserve(width, height);
        let kernel = &self.kernel;
        let mut next_row = 0;
        for y in 0..height {
            let last_row = (y + RADIUS).min(height - 1);
            while next_row <= last_row {
                let row = &mask[next_row * width..(next_row + 1) * width];
                self.padded[RADIUS..width + RADIUS].copy_from_slice(row);
                // Torchvision reflection excludes the edge sample. Both
                // dimensions exceed RADIUS, so a single reflection suffices.
                for i in 0..RADIUS {
                    self.padded[i] = row[RADIUS - i];
                    self.padded[width + RADIUS + i] = row[width - 2 - i];
                }
                let slot = next_row % WINDOW;
                let dst = &mut self.ring[slot * width..(slot + 1) * width];
                let done = (self.kernels.horizontal)(&self.padded, kernel, dst);
                horizontal_scalar(&self.padded[done..], kernel, &mut dst[done..]);
                next_row += 1;
            }
            let rows: BlurRows<'_> = std::array::from_fn(|i| {
                let sy = y as isize + i as isize - RADIUS as isize;
                let sy = if sy < 0 {
                    (-sy) as usize
                } else if sy >= height as isize {
                    2 * height - sy as usize - 2
                } else {
                    sy as usize
                };
                let slot = sy % WINDOW;
                &self.ring[slot * width..(slot + 1) * width]
            });
            let dst = &mut mask[y * width..(y + 1) * width];
            let done = (self.kernels.vertical)(&rows, kernel, dst);
            if done < width {
                let tail = rows.map(|row| &row[done..]);
                vertical_scalar(&tail, kernel, &mut dst[done..]);
            }
        }
    }
}

// Center plus six symmetric pairs. Keep near-to-far accumulation and separate
// adds/multiplies identical to the previous blur, without an FMA requirement.
fn horizontal_scalar(src: &[f32], k: &[f32; WINDOW], dst: &mut [f32]) {
    for (x, dst) in dst.iter_mut().enumerate() {
        let mut sum = src[x + RADIUS] * k[RADIUS];
        for i in (0..RADIUS).rev() {
            sum += (src[x + i] + src[x + WINDOW - 1 - i]) * k[i];
        }
        *dst = sum;
    }
}

fn vertical_scalar(rows: &BlurRows<'_>, k: &[f32; WINDOW], dst: &mut [f32]) {
    for (x, dst) in dst.iter_mut().enumerate() {
        let mut sum = rows[RADIUS][x] * k[RADIUS];
        for i in (0..RADIUS).rev() {
            sum += (rows[i][x] + rows[WINDOW - 1 - i][x]) * k[i];
        }
        *dst = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent literal 13-tap convolution, with modular exclusive reflection
    // and a full intermediate plane. This also catches in-place ring overwrite.
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

    #[test]
    fn reflected_in_place_blur_matches_literal_convolution_and_reuses_storage() {
        for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
            if !backend.is_available() {
                continue;
            }
            let mut blur = GaussianBlur::new(backend);
            blur.reserve(69, 51);
            let layout = [
                (&blur.ring, blur.ring.capacity()),
                (&blur.padded, blur.padded.capacity()),
            ]
            .map(|(v, capacity)| (v.as_ptr(), capacity));
            for width in 1..=69 {
                for height in [1, 6, 7, 8, 12, 13, 14, 25, 26, 27, 51] {
                    let original: Vec<f32> = (0..width * height)
                        .map(|i| ((i * 31 + i / width * 73) % 257) as f32 / 17.0)
                        .collect();
                    let expected = reference(&original, width, height, &blur.kernel);
                    let mut actual = original.clone();
                    blur.blur(&mut actual, width, height);
                    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                        assert!(
                            (a - b).abs() < 0.000006,
                            "{backend:?} {width}x{height} [{i}]: {a} != {b}"
                        );
                    }
                    assert_eq!(
                        layout,
                        [
                            (&blur.ring, blur.ring.capacity()),
                            (&blur.padded, blur.padded.capacity())
                        ]
                        .map(|(v, capacity)| (v.as_ptr(), capacity))
                    );
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
                let mut plane = vec![3.0; width * height];
                blur.blur(&mut plane, width, height);
                if width <= RADIUS || height <= RADIUS {
                    assert!(plane.iter().all(|&v| v == 3.0));
                } else {
                    assert!(plane.iter().all(|&v| (v - 3.0).abs() < 0.000002));
                }
            }
        }
    }
}
