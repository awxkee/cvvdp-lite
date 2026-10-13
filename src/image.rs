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

use crate::{Backend, Error};
use std::sync::OnceLock;

#[derive(Clone, Copy)]
enum Pixels<'a> {
    Srgb8(&'a [u8]),
    SrgbF32(&'a [f32]),
    Linear([&'a [f32]; 3]),
}

/// A borrowed SDR image, without alpha. Composite alpha and convert ICC color
/// profiles to sRGB before constructing this view. Strides are in samples.
#[derive(Clone, Copy)]
pub struct RgbImage<'a> {
    pixels: Pixels<'a>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    stride: usize,
}

impl<'a> RgbImage<'a> {
    /// Interleaved, gamma-encoded sRGB bytes. Row padding is allowed.
    pub fn srgb8(
        pixels: &'a [u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, 3, &[pixels.len()])?;
        Ok(Self {
            pixels: Pixels::Srgb8(pixels),
            width,
            height,
            stride,
        })
    }

    /// Interleaved, gamma-encoded sRGB floats. Finite samples outside `[0,1]`
    /// are clipped by the display model; non-finite samples fail evaluation.
    pub fn srgb_f32(
        pixels: &'a [f32],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, 3, &[pixels.len()])?;
        Ok(Self {
            pixels: Pixels::SrgbF32(pixels),
            width,
            height,
            stride,
        })
    }

    /// Three linear sRGB planes, normalized to `[0,1]` (1 is display white).
    /// This is relative SDR light, not absolute HDR luminance. Finite samples
    /// outside `[0,1]` are clipped. The three planes share a stride.
    pub fn linear_rgb(
        planes: [&'a [f32]; 3],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, Error> {
        validate(width, height, stride, 1, &planes.map(|p| p.len()))?;
        Ok(Self {
            pixels: Pixels::Linear(planes),
            width,
            height,
            stride,
        })
    }

    /// The `w`×`h` window at (`x0`, `y0`), sharing this image's stride.
    pub(crate) fn crop(self, x0: usize, y0: usize, w: usize, h: usize) -> Self {
        debug_assert!(x0 + w <= self.width && y0 + h <= self.height);
        let start = y0 * self.stride;
        let pixels = match self.pixels {
            Pixels::Srgb8(p) => Pixels::Srgb8(&p[start + x0 * 3..]),
            Pixels::SrgbF32(p) => Pixels::SrgbF32(&p[start + x0 * 3..]),
            Pixels::Linear(p) => Pixels::Linear(p.map(|p| &p[start + x0..])),
        };
        Self {
            pixels,
            width: w,
            height: h,
            stride: self.stride,
        }
    }

    /// Image width, in pixels.
    pub fn width(self) -> usize {
        self.width
    }
    /// Image height, in pixels.
    pub fn height(self) -> usize {
        self.height
    }

    pub(crate) fn to_dkl(
        self,
        scale: f32,
        black: f32,
        out: &mut [Vec<f32>; 3],
        conversion: &mut RgbToDkl,
    ) -> Result<(), Error> {
        let count = self.width * self.height;
        for plane in out.iter_mut() {
            crate::pyramid::ensure_len(plane, count);
        }
        let [l, rg, yv] = out;
        let output = l[..count]
            .chunks_exact_mut(self.width)
            .zip(rg[..count].chunks_exact_mut(self.width))
            .zip(yv[..count].chunks_exact_mut(self.width));
        // Resolve format once per image. Conversion works on whole row slices,
        // with no per-pixel format branch and no full-image staging buffer.
        match self.pixels {
            Pixels::Srgb8(pixels) => {
                conversion.set_display(scale, black);
                for (input, ((l, rg), yv)) in pixels.chunks(self.stride).zip(output) {
                    let input = &input[..self.width * 3];
                    let mut output = [l, rg, yv];
                    let done = (conversion.kernels.bytes)(input, &mut output, &conversion.bytes);
                    let [l, rg, yv] = output;
                    let tail = l[done..]
                        .iter_mut()
                        .zip(&mut rg[done..])
                        .zip(&mut yv[done..]);
                    for (rgb, ((l, rg), yv)) in input.as_chunks::<3>().0[done..].iter().zip(tail) {
                        let r = &conversion.bytes[0][rgb[0] as usize];
                        let g = &conversion.bytes[1][rgb[1] as usize];
                        let b = &conversion.bytes[2][rgb[2] as usize];
                        *l = (r[0] + g[0]) + b[0];
                        *rg = (r[1] + g[1]) + b[1];
                        *yv = (r[2] + g[2]) + b[2];
                    }
                }
            }
            Pixels::Linear([r, g, b]) => {
                let input = r
                    .chunks(self.stride)
                    .zip(g.chunks(self.stride))
                    .zip(b.chunks(self.stride));
                for (((r, g), b), ((l, rg), yv)) in input.zip(output) {
                    let input = [r, g, b].map(|p| &p[..self.width]);
                    conversion.linear(input, [l, rg, yv], scale, black)?;
                }
            }
            Pixels::SrgbF32(pixels) => {
                // Exact libm transfer curve, followed by the same SIMD matrix
                // kernel as planar input. This small block remains on the stack.
                const BLOCK: usize = 32;
                let mut linear = [[0.0; BLOCK]; 3];
                for (input, ((l, rg), yv)) in pixels.chunks(self.stride).zip(output) {
                    let input = input[..self.width * 3].as_chunks::<3>().0;
                    let output = l
                        .chunks_mut(BLOCK)
                        .zip(rg.chunks_mut(BLOCK))
                        .zip(yv.chunks_mut(BLOCK));
                    for (input, ((l, rg), yv)) in input.chunks(BLOCK).zip(output) {
                        let count = input.len();
                        let [r, g, b] = &mut linear;
                        let samples = r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut());
                        for (rgb, ((r, g), b)) in input.iter().zip(samples) {
                            *r = srgb_to_linear(unit_sample(rgb[0])?);
                            *g = srgb_to_linear(unit_sample(rgb[1])?);
                            *b = srgb_to_linear(unit_sample(rgb[2])?);
                        }
                        conversion.linear(
                            linear.each_ref().map(|p| &p[..count]),
                            [l, rg, yv],
                            scale,
                            black,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

// Fused RGB -> XYZ -> LMS2006 -> DKLd65 from the original's f32 matrices.
pub(crate) static DKL: [[f32; 3]; 3] = [
    [0.23320122, 0.72883075, 0.088_995_87],
    [0.12762076, -0.08706811, -0.036777392],
    [-0.2148225, -0.62625366, 0.8514033],
];

pub(crate) type Output<'a> = [&'a mut [f32]; 3];
// Input component, byte value, output DKL component (plus one SIMD padding lane).
pub(crate) type ByteTable = [[[f32; 4]; 256]; 3];

type LinearKernel = fn([&[f32]; 3], &mut Output<'_>, f32, f32) -> Result<usize, Error>;

struct Kernels {
    bytes: fn(&[u8], &mut Output<'_>, &ByteTable) -> usize,
    linear: LinearKernel,
}

impl Kernels {
    #[allow(unsafe_code)]
    fn for_backend(backend: Backend) -> Self {
        assert!(backend.is_available());
        // SAFETY: Selection validates CPU availability once per workspace.
        // Architecture entries check all row lengths before loading/storing.
        match backend {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            Backend::Neon => Self {
                bytes: |s, d, t| unsafe { crate::neon::dkl_bytes_neon(s, d, t) },
                linear: |s, d, scale, black| unsafe {
                    crate::neon::dkl_linear_neon(s, d, scale, black)
                },
            },
            #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
            Backend::Avx2 => Self {
                bytes: |s, d, t| unsafe { crate::avx::dkl_bytes_avx2(s, d, t) },
                linear: |s, d, scale, black| unsafe {
                    crate::avx::dkl_linear_avx2(s, d, scale, black)
                },
            },
            #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
            Backend::Sse2 => Self {
                bytes: |s, d, t| unsafe { crate::sse::dkl_bytes_sse2(s, d, t) },
                linear: |s, d, scale, black| unsafe {
                    crate::sse::dkl_linear_sse2(s, d, scale, black)
                },
            },
            Backend::Scalar => Self {
                bytes: |_, _, _| 0,
                linear: |_, _, _, _| Ok(0),
            },
            #[allow(unreachable_patterns)]
            _ => unreachable!("unavailable backend rejected above"),
        }
    }
}

/// Reusable display-conditioned byte contributions plus fixed ISA dispatch.
pub(crate) struct RgbToDkl {
    kernels: Kernels,
    bytes: Box<ByteTable>,
    display: Option<(f32, f32)>,
}

impl RgbToDkl {
    pub(crate) fn new(backend: Backend) -> Self {
        let _ = srgb_table();
        Self {
            kernels: Kernels::for_backend(backend),
            bytes: Box::new([[[0.0; 4]; 256]; 3]),
            display: None,
        }
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        std::mem::size_of::<ByteTable>()
    }

    pub(crate) fn set_display(&mut self, scale: f32, black: f32) {
        if self.display == Some((scale, black)) {
            return;
        }
        let linear = srgb_table();
        for (input, table) in self.bytes.iter_mut().enumerate() {
            for (output, &linear) in table.iter_mut().zip(linear) {
                let rgb = linear * scale + black;
                for (output, coefficients) in output[..3].iter_mut().zip(&DKL) {
                    *output = rgb * coefficients[input];
                }
            }
        }
        self.display = Some((scale, black));
    }

    fn linear(
        &self,
        input: [&[f32]; 3],
        mut output: Output<'_>,
        scale: f32,
        black: f32,
    ) -> Result<(), Error> {
        let done = (self.kernels.linear)(input, &mut output, scale, black)?;
        let [r, g, b] = input;
        let [l, rg, yv] = output;
        let input = r[done..].iter().zip(&g[done..]).zip(&b[done..]);
        let output = l[done..]
            .iter_mut()
            .zip(&mut rg[done..])
            .zip(&mut yv[done..]);
        for (((&r, &g), &b), ((l, rg), yv)) in input.zip(output) {
            let rgb = [
                unit_sample(r)? * scale + black,
                unit_sample(g)? * scale + black,
                unit_sample(b)? * scale + black,
            ];
            for (output, coefficients) in [l, rg, yv].into_iter().zip(&DKL) {
                *output = (rgb[0] * coefficients[0] + rgb[1] * coefficients[1])
                    + rgb[2] * coefficients[2];
            }
        }
        Ok(())
    }
}

#[inline]
fn unit_sample(value: f32) -> Result<f32, Error> {
    if !value.is_finite() {
        return Err(Error::NonFiniteSample);
    }
    Ok(value.clamp(0.0, 1.0))
}

fn validate(
    width: usize,
    height: usize,
    stride: usize,
    channels: usize,
    lengths: &[usize],
) -> Result<(), Error> {
    pixel_count(width, height)?;
    let row = width.checked_mul(channels).ok_or(Error::InvalidLayout)?;
    let needed = (height - 1)
        .checked_mul(stride)
        .and_then(|v| v.checked_add(row))
        .ok_or(Error::InvalidLayout)?;
    if stride < row || lengths.iter().any(|&n| n < needed) {
        return Err(Error::InvalidLayout);
    }
    Ok(())
}

fn srgb_to_linear(v: f32) -> f32 {
    if v > 0.04045 {
        ((v + 0.055) / 1.055).powf(2.4)
    } else {
        v / 12.92
    }
}

fn srgb_table() -> &'static [f32; 256] {
    static TABLE: OnceLock<[f32; 256]> = OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0)))
}

pub(crate) fn pixel_count(width: usize, height: usize) -> Result<usize, Error> {
    if width == 0
        || height == 0
        || width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(12))
            .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(Error::InvalidDimensions);
    }
    Ok(width * height)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn reference(
        image: RgbImage<'_>,
        scale: f32,
        black: f32,
        out: &mut [Vec<f32>; 3],
    ) -> Result<(), Error> {
        for p in out.iter_mut() {
            crate::pyramid::ensure_len(p, image.width * image.height);
        }
        // The fused RGB -> XYZ -> LMS2006 -> DKLd65 matrix, computed using the
        // original's f32 matrices (not fcvvdp's rounded XYZ coefficients).
        const DKL: [[f32; 3]; 3] = [
            [0.23320122, 0.72883075, 0.088_995_87],
            [0.12762076, -0.08706811, -0.036777392],
            [-0.2148225, -0.62625366, 0.8514033],
        ];
        let table = srgb_table();
        for y in 0..image.height {
            for x in 0..image.width {
                let mut rgb = [0.0; 3];
                match image.pixels {
                    Pixels::Srgb8(p) => {
                        let offset = y * image.stride + x * 3;
                        for c in 0..3 {
                            rgb[c] = table[p[offset + c] as usize];
                        }
                    }
                    Pixels::SrgbF32(p) => {
                        let offset = y * image.stride + x * 3;
                        for c in 0..3 {
                            let v = p[offset + c];
                            if !v.is_finite() {
                                return Err(Error::NonFiniteSample);
                            }
                            rgb[c] = srgb_to_linear(v.clamp(0.0, 1.0));
                        }
                    }
                    Pixels::Linear(p) => {
                        for c in 0..3 {
                            let v = p[c][y * image.stride + x];
                            if !v.is_finite() {
                                return Err(Error::NonFiniteSample);
                            }
                            rgb[c] = v.clamp(0.0, 1.0);
                        }
                    }
                }
                for v in &mut rgb {
                    *v = *v * scale + black;
                }
                for c in 0..3 {
                    out[c][y * image.width + x] =
                        rgb[0] * DKL[c][0] + rgb[1] * DKL[c][1] + rgb[2] * DKL[c][2];
                }
            }
        }
        Ok(())
    }

    #[test]
    fn conversion_matches_original_for_all_formats_strides_tails_and_display_changes() {
        for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
            if !backend.is_available() {
                continue;
            }
            let mut conversion = RgbToDkl::new(backend);
            let pointer = conversion.bytes.as_ptr();
            for w in 1..=69 {
                for h in [1, 2, 7] {
                    let stride = 3 * w + 5;
                    let ls = w + 3;
                    let mut bytes = vec![0; (h - 1) * stride + 3 * w];
                    let mut floats = vec![f32::NAN; bytes.len()];
                    let mut planes: [Vec<f32>; 3] =
                        std::array::from_fn(|_| vec![f32::INFINITY; (h - 1) * ls + w]);
                    for y in 0..h {
                        for x in 0..w {
                            for c in 0..3 {
                                let value = (x * 53 + y * 79 + c * 37) % 256;
                                bytes[y * stride + 3 * x + c] = value as u8;
                                // Continuous floats and finite clipping in both directions.
                                let value = (value as f32 + 0.137) / 191.0 - 0.15;
                                floats[y * stride + 3 * x + c] = value;
                                planes[c][y * ls + x] = value;
                            }
                        }
                    }
                    let images = [
                        RgbImage::srgb8(&bytes, w, h, stride).unwrap(),
                        RgbImage::srgb_f32(&floats, w, h, stride).unwrap(),
                        RgbImage::linear_rgb(planes.each_ref().map(Vec::as_slice), w, h, ls)
                            .unwrap(),
                    ];
                    // Alternate displays on the same state, then revisit a cached display.
                    for (scale, black) in [
                        (199.8, 0.5978874),
                        (0.75, 0.0),
                        (9999.0, 1.0),
                        (199.8, 0.5978874),
                    ] {
                        for (format, image) in images.into_iter().enumerate() {
                            let mut expected = std::array::from_fn(|_| vec![-7.0; w * h + 3]);
                            let mut actual = expected.clone();
                            reference(image, scale, black, &mut expected).unwrap();
                            image
                                .to_dkl(scale, black, &mut actual, &mut conversion)
                                .unwrap();
                            assert_eq!(
                                actual, expected,
                                "{backend:?} format {format} {w}x{h} display {scale}/{black}"
                            );
                            assert_eq!(pointer, conversion.bytes.as_ptr());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn nonfinite_active_samples_are_rejected_in_vectors_and_tails() {
        for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
            if !backend.is_available() {
                continue;
            }
            let mut conversion = RgbToDkl::new(backend);
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                for x in [0, 3, 4, 7, 8, 16, 32] {
                    for c in 0..3 {
                        let mut floats = vec![0.25; 33 * 3];
                        let mut planes = [vec![0.25; 33], vec![0.25; 33], vec![0.25; 33]];
                        floats[3 * x + c] = bad;
                        planes[c][x] = bad;
                        let images = [
                            RgbImage::srgb_f32(&floats, 33, 1, 99).unwrap(),
                            RgbImage::linear_rgb(planes.each_ref().map(Vec::as_slice), 33, 1, 33)
                                .unwrap(),
                        ];
                        let mut out = std::array::from_fn(|_| Vec::new());
                        for image in images {
                            assert_eq!(
                                image.to_dkl(199.8, 0.6, &mut out, &mut conversion),
                                Err(Error::NonFiniteSample)
                            );
                        }
                    }
                }
            }
        }
    }
}
