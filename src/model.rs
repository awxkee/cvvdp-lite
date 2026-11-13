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

use crate::{
    Backend, Error, RgbImage,
    blur::{GaussianBlur, MASK_STRIP_ROWS, RADIUS, blur_active},
    image::RgbToDkl,
    kernels, pyramid, tables,
};
use kernels::{EPSILON, MASK_P, MASK_Q};

static BASE_WEIGHT: [f64; 3] = [0.0036334486212581396, 1.6627724170684814, 4.11874532699585];
const IMAGE_INTEGRATION: f64 = 0.577918291091919;
const JOD_A: f64 = 0.0439569391310215;
const JOD_EXP: f64 = 0.9302042722702026;
static CH_GAIN: [f32; 3] = [1.0, 1.45, 1.0];

/// SDR display conditions, with emitted black and reflected light specified
/// separately. Supported angular resolution is 0.01..=10000 pixels/degree;
/// total peak plus reflected luminance must not exceed 10000 cd/m².
/// Spatial resolution is the display's central pixels per visual degree,
/// independent of the dimensions of the compared images.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Display {
    pub pixels_per_degree: f32,
    pub peak_luminance: f32,
    /// Emitted black, in cd/m² (before adding reflections).
    pub black_luminance: f32,
    /// Reflected ambient luminance, in cd/m².
    pub reflected_luminance: f32,
}

impl Display {
    /// Original macbook_pro_16 (our macdisplay target): 3072×1920,
    /// 16-inch diagonal viewed from 25 inches, 500 nit, 0.37 nit black,
    /// 250 lux and reflectivity 0.005.
    pub fn macbook_pro_16() -> Self {
        let peak = 500.0;
        Self {
            pixels_per_degree: 98.792_42,
            peak_luminance: peak,
            black_luminance: peak / 1_351.351_3,
            reflected_luminance: 250.0 * 0.005 / std::f32::consts::PI,
        }
    }

    /// Original standard_4k: 30-inch 3840×2160 at 0.7472m, 200 nit,
    /// contrast 1000:1 and 250 lux, reflectivity 0.005.
    pub fn standard_4k() -> Self {
        Self {
            pixels_per_degree: 75.40245,
            peak_luminance: 200.0,
            black_luminance: 0.2,
            reflected_luminance: 250.0 * 0.005 / std::f32::consts::PI,
        }
    }

    /// Original standard_fhd: 24-inch 1920×1080 at 0.6m, with the
    /// same SDR photometry as standard_4k.
    pub fn standard_fhd() -> Self {
        let diagonal = 24.0_f64 * 0.0254;
        let width = diagonal * 16.0 / (16.0_f64 * 16.0 + 9.0 * 9.0).sqrt();
        Self {
            pixels_per_degree: (1.0 / (2.0 * (width / (2.0 * 1920.0 * 0.6)).atan().to_degrees()))
                as f32,
            ..Self::standard_4k()
        }
    }

    fn validate(self) -> Result<Self, Error> {
        if !self.pixels_per_degree.is_finite()
            || !(0.01..=10000.0).contains(&self.pixels_per_degree)
            || !self.peak_luminance.is_finite()
            || self.peak_luminance <= 0.0
            || !self.black_luminance.is_finite()
            || self.black_luminance < 0.0
            || self.black_luminance >= self.peak_luminance
            || !self.reflected_luminance.is_finite()
            || self.reflected_luminance < 0.0
            || self.peak_luminance + self.reflected_luminance > 10000.0
        {
            return Err(Error::InvalidDisplay);
        }
        Ok(self)
    }

    pub(crate) fn scale(self) -> f32 {
        self.peak_luminance - self.black_luminance
    }
    pub(crate) fn black(self) -> f32 {
        self.black_luminance + self.reflected_luminance
    }
}

impl Default for Display {
    fn default() -> Self {
        Self::standard_4k()
    }
}

/// Still-image model configuration. No temporal filters or video state.
pub struct CvvdpLite {
    display: Display,
}

impl CvvdpLite {
    pub fn new(display: Display) -> Result<Self, Error> {
        Ok(Self {
            display: display.validate()?,
        })
    }

    /// Cache the reference's perceptual contrast and all per-pixel CSF values.
    /// Reference and test use their own adaptation luminance, as in weber_g1.
    pub fn prepare_reference(&self, image: RgbImage<'_>) -> Result<PreparedReference, Error> {
        self.prepare_reference_with(image, &mut Workspace::new())
    }

    /// Prepare a reference with one sensitivity coordinate per pixel.
    ///
    /// Scoring reconstructs the three channel sensitivities from the stored
    /// coordinate. This saves reference memory and adds work to each comparison.
    pub fn prepare_reference_compact(
        &self,
        image: RgbImage<'_>,
    ) -> Result<PreparedReference, Error> {
        self.prepare_reference_compact_with(image, &mut Workspace::new())
    }

    /// Prepare a new immutable reference using reusable analysis scratch space.
    /// Call [`Workspace::reserve`] or attach it with [`PreparedReference::evaluator_with`]
    /// to reserve scratch for subsequent candidate comparisons.
    pub fn prepare_reference_with(
        &self,
        image: RgbImage<'_>,
        workspace: &mut Workspace,
    ) -> Result<PreparedReference, Error> {
        self.prepare_reference_impl(image, workspace, false)
    }

    /// Prepare a compact reference using reusable analysis scratch space.
    pub fn prepare_reference_compact_with(
        &self,
        image: RgbImage<'_>,
        workspace: &mut Workspace,
    ) -> Result<PreparedReference, Error> {
        self.prepare_reference_impl(image, workspace, true)
    }

    fn prepare_reference_impl(
        &self,
        image: RgbImage<'_>,
        workspace: &mut Workspace,
        compact: bool,
    ) -> Result<PreparedReference, Error> {
        let Workspace {
            mask: expansion,
            tmp,
            pyramid: pyramid_kernels,
            conversion,
            ..
        } = workspace;
        let (mut w, mut h) = (image.width, image.height);
        let height = pyramid_height(w, h, self.display.pixels_per_degree);
        let mut current = std::array::from_fn(|_| Vec::new());
        // Decode into the first band's final contrast buffers.
        image.to_dkl(
            self.display.scale(),
            self.display.black(),
            &mut current,
            conversion,
        )?;
        if compact && height > 0 {
            for plane in &mut *expansion {
                pyramid::ensure_len(plane, w * h.min(MASK_STRIP_ROWS));
            }
        }
        let mut bands = Vec::with_capacity(height + 1);
        for level in 0..height {
            let frequency = band_frequency(level, self.display.pixels_per_degree);
            let count = w * h;
            let mut next = std::array::from_fn(|_| Vec::new());
            // Finish reduction before replacing the Gaussian samples with contrast.
            reduce_planes(&current, &mut next, w, h, tmp, pyramid_kernels);
            let scale = if level == 0 { 1.0 } else { 2.0 };
            let csf = csf_at_frequency(frequency);
            let sensitivity = if compact {
                let mut positions = vec![0.0; count];
                for y in (0..h).step_by(MASK_STRIP_ROWS) {
                    let end = (y + MASK_STRIP_ROWS).min(h);
                    let pixels = y * w..end * w;
                    let len = pixels.len();
                    expand_strip(&next, expansion, (w, h), y..end, tmp, pyramid_kernels);
                    weight_reference(
                        current.each_mut().map(|v| &mut v[pixels.clone()]),
                        expansion.each_mut().map(|v| &mut v[..len]),
                        &csf,
                        scale,
                        Some(&mut positions[pixels]),
                    );
                }
                BandSensitivity::Compact {
                    positions,
                    csf: Box::new(csf),
                }
            } else {
                // The expansion is replaced with sensitivities as it is read.
                let mut planes = std::array::from_fn(|_| Vec::new());
                for (coarse, plane) in next.iter().zip(&mut planes) {
                    pyramid::expand(coarse, w, h, plane, tmp, pyramid_kernels);
                }
                weight_reference(
                    current.each_mut().map(Vec::as_mut_slice),
                    planes.each_mut().map(Vec::as_mut_slice),
                    &csf,
                    scale,
                    None,
                );
                BandSensitivity::Dense(planes)
            };
            bands.push(ReferenceBand {
                w,
                h,
                frequency,
                contrast: current,
                sensitivity,
            });
            current = next;
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        let weights = weight_base_reference(&mut current, w * h);
        bands.push(ReferenceBand {
            w,
            h,
            frequency: 0.1,
            contrast: current,
            sensitivity: BandSensitivity::Base(weights),
        });
        Ok(PreparedReference {
            width: image.width,
            height: image.height,
            display: self.display,
            bands,
        })
    }

    /// Compare one pair of images without caching the reference bands.
    /// For multiple candidates use [`Self::prepare_reference`].
    pub fn compare(&self, reference: RgbImage<'_>, test: RgbImage<'_>) -> Result<Score, Error> {
        if reference.width != test.width || reference.height != test.height {
            return Err(Error::DimensionMismatch);
        }
        let (mut w, mut h) = (reference.width, reference.height);
        let height = pyramid_height(w, h, self.display.pixels_per_degree);
        let mut workspace = Workspace::new();
        let mut reference_current = std::array::from_fn(|_| Vec::new());
        let mut reference_next = std::array::from_fn(|_| Vec::new());
        let scale = self.display.scale();
        let black = self.display.black();
        reference.to_dkl(
            scale,
            black,
            &mut reference_current,
            &mut workspace.conversion,
        )?;
        test.to_dkl(
            scale,
            black,
            &mut workspace.current,
            &mut workspace.conversion,
        )?;

        let strip_len = if height == 0 {
            0
        } else {
            workspace.reserve(w, h)?;
            w * h.min(MASK_STRIP_ROWS)
        };
        let mut sensitivity = std::array::from_fn(|_| vec![0.0; strip_len]);
        let mut per_channel = [0.0; 3];
        for level in 0..height {
            reduce_planes(
                &reference_current,
                &mut reference_next,
                w,
                h,
                &mut workspace.tmp,
                &workspace.pyramid,
            );
            reduce_planes(
                &workspace.current,
                &mut workspace.next,
                w,
                h,
                &mut workspace.tmp,
                &workspace.pyramid,
            );
            let csf = csf_at_frequency(band_frequency(level, self.display.pixels_per_degree));
            let band_scale = if level == 0 { 1.0 } else { 2.0 };
            let squares = workspace.pool_band(
                w,
                h,
                BandReference::OneShot {
                    current: &mut reference_current,
                    next: &reference_next,
                    sensitivity: &mut sensitivity,
                    csf: &csf,
                },
                band_scale,
            );
            accumulate_band(&mut per_channel, band_rms(squares, w * h), false);
            std::mem::swap(&mut reference_current, &mut reference_next);
            std::mem::swap(&mut workspace.current, &mut workspace.next);
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }

        let count = w * h;
        let weights = weight_base_reference(&mut reference_current, count);
        let squares = pool_baseband(&workspace.current, &reference_current, weights, count);
        accumulate_band(&mut per_channel, band_rms(squares, count), true);
        Ok(finish_score(per_channel))
    }
}

pub(crate) struct ReferenceBand {
    pub(crate) w: usize,
    pub(crate) h: usize,
    frequency: f32,
    pub(crate) contrast: [Vec<f32>; 3],
    sensitivity: BandSensitivity,
}

enum BandSensitivity {
    Base([f32; 3]),
    Dense([Vec<f32>; 3]),
    Compact {
        positions: Vec<f32>,
        csf: Box<[[f32; 32]; 3]>,
    },
}

impl BandSensitivity {
    fn allocated_bytes(&self) -> usize {
        match self {
            Self::Base(_) => 0,
            Self::Dense(planes) => planes.iter().map(|p| p.capacity() * size_of::<f32>()).sum(),
            Self::Compact { positions, csf } => {
                positions.capacity() * size_of::<f32>() + size_of_val(csf.as_ref())
            }
        }
    }
}

impl ReferenceBand {
    pub(crate) fn base_weights(&self) -> [f32; 3] {
        let BandSensitivity::Base(weights) = self.sensitivity else {
            unreachable!("base weights are only available for the baseband");
        };
        weights
    }

    pub(crate) fn dense_sensitivity(&self) -> Option<&[Vec<f32>; 3]> {
        match &self.sensitivity {
            BandSensitivity::Dense(planes) => Some(planes),
            _ => None,
        }
    }

    pub(crate) fn decode_sensitivity(&self, range: std::ops::Range<usize>, out: [&mut [f32]; 3]) {
        let BandSensitivity::Compact { positions, csf } = &self.sensitivity else {
            unreachable!("only compact bands have coordinates to decode");
        };
        decode_weights(csf, &positions[range], out);
    }
}

enum BandReference<'a> {
    Prepared(&'a ReferenceBand),
    OneShot {
        current: &'a mut [Vec<f32>; 3],
        next: &'a [Vec<f32>; 3],
        sensitivity: &'a mut [Vec<f32>; 3],
        csf: &'a [[f32; 32]; 3],
    },
}

impl BandReference<'_> {
    fn prepare_strip(
        &mut self,
        dimensions: (usize, usize),
        rows: std::ops::Range<usize>,
        scale: f32,
        tmp: &mut [f32],
        kernels: &pyramid::Kernels,
    ) {
        if let Self::OneShot {
            current,
            next,
            sensitivity,
            csf,
        } = self
        {
            let w = dimensions.0;
            let first = rows.start * w;
            let len = rows.len() * w;
            expand_strip(next, sensitivity, dimensions, rows, tmp, kernels);
            // Reduction has finished, so the Gaussian plane can hold the
            // weighted contrast until pooling consumes it after mask blur.
            weight_reference(
                current.each_mut().map(|v| &mut v[first..first + len]),
                sensitivity.each_mut().map(|v| &mut v[..len]),
                csf,
                scale,
                None,
            );
        }
    }

    fn contrast(&self, range: std::ops::Range<usize>) -> [&[f32]; 3] {
        let planes = match self {
            Self::Prepared(band) => &band.contrast,
            Self::OneShot { current, .. } => current,
        };
        planes.each_ref().map(|v| &v[range.clone()])
    }

    fn apply_contrast(
        &self,
        backend: Backend,
        current: [&mut [f32]; 3],
        mask: [&mut [f32]; 3],
        range: std::ops::Range<usize>,
        scale: f32,
    ) {
        let reference = self.contrast(range.clone());
        match self {
            Self::Prepared(band) => {
                if let Some(planes) = band.dense_sensitivity() {
                    kernels::contrast(
                        backend,
                        current,
                        mask,
                        planes.each_ref().map(|v| &v[range.clone()]),
                        reference,
                        scale,
                    );
                } else {
                    // Keep the original flattened SIMD boundaries, including the
                    // scalar tail at the end of the band.
                    const CHUNK: usize = 1024;
                    let mut weights = [[0.0; CHUNK]; 3];
                    let [a, b, c] = current;
                    let [m0, m1, m2] = mask;
                    for start in (0..range.len()).step_by(CHUNK) {
                        let end = (start + CHUNK).min(range.len());
                        let len = end - start;
                        band.decode_sensitivity(
                            range.start + start..range.start + end,
                            weights.each_mut().map(|v| &mut v[..len]),
                        );
                        kernels::contrast(
                            backend,
                            [&mut a[start..end], &mut b[start..end], &mut c[start..end]],
                            [
                                &mut m0[start..end],
                                &mut m1[start..end],
                                &mut m2[start..end],
                            ],
                            weights.each_ref().map(|v| &v[..len]),
                            reference.map(|v| &v[start..end]),
                            scale,
                        );
                    }
                }
            }
            Self::OneShot { sensitivity, .. } => kernels::contrast(
                backend,
                current,
                mask,
                sensitivity.each_ref().map(|v| &v[..range.len()]),
                reference,
                scale,
            ),
        }
    }
}

/// Immutable cached reference; may be shared among independent evaluators.
pub struct PreparedReference {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) display: Display,
    pub(crate) bands: Vec<ReferenceBand>,
}

impl PreparedReference {
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn display(&self) -> Display {
        self.display
    }
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }
    /// Heap memory retained by the reference, excluding allocator overhead.
    pub fn allocated_bytes(&self) -> usize {
        self.bands.capacity() * std::mem::size_of::<ReferenceBand>()
            + self
                .bands
                .iter()
                .map(|band| {
                    band.contrast
                        .iter()
                        .map(|v| v.capacity() * size_of::<f32>())
                        .sum::<usize>()
                        + band.sensitivity.allocated_bytes()
                })
                .sum::<usize>()
    }

    /// Create a candidate workspace with all buffers reserved before scoring.
    pub fn evaluator(&self) -> Evaluator<'_> {
        self.evaluator_with(Workspace::new())
    }

    /// Attach an existing workspace, keeping its allocations and backend.
    pub fn evaluator_with(&self, mut workspace: Workspace) -> Evaluator<'_> {
        workspace
            .reserve(self.width, self.height)
            .expect("validated reference dimensions");
        workspace
            .conversion
            .set_display(self.display.scale(), self.display.black());
        Evaluator {
            reference: self,
            workspace,
        }
    }
}

/// Epsilon-regularized RMS differences by channel in one spatial band,
/// before baseband weights (the original's safe_pow norm).
/// These diagnostics are available through Evaluator::band_scores.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandScore {
    pub width: usize,
    pub height: usize,
    pub frequency_cpd: f32,
    pub rms: [f64; 3],
}

/// A quality score. Raw distortion is nonnegative and smaller is better.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub jod: f64,
    /// Regularized fourth-norm pooling with the image integration correction,
    /// before the final JOD mapping. Identical images have zero distortion.
    pub distortion: f64,
}

/// A prepared reference paired with reusable candidate state.
/// Use into_workspace to move the state to another reference without reallocating.
pub struct Evaluator<'a> {
    reference: &'a PreparedReference,
    workspace: Workspace,
}

impl Evaluator<'_> {
    pub fn band_scores(&self) -> &[BandScore] {
        self.workspace.band_scores()
    }
    pub fn backend(&self) -> Backend {
        self.workspace.backend()
    }
    pub fn allocated_bytes(&self) -> usize {
        self.workspace.allocated_bytes()
    }
    pub fn compare(&mut self, test: RgbImage<'_>) -> Result<Score, Error> {
        self.workspace.compare(self.reference, test)
    }
    /// Detach scratch space so it can be reused for another image/reference.
    pub fn into_workspace(self) -> Workspace {
        self.workspace
    }
}

/// Reusable scratch independent of any reference lifetime. reserve front-loads
/// allocations; comparisons at those dimensions or smaller allocate nothing.
/// Separate workspaces can run on the caller's existing worker pool.
pub struct Workspace {
    backend: Backend,
    current: [Vec<f32>; 3],
    next: [Vec<f32>; 3],
    // One strip of expanded samples, reused for raw and blurred masks.
    mask: [Vec<f32>; 3],
    tmp: Vec<f32>,
    blur: GaussianBlur,
    pyramid: pyramid::Kernels,
    conversion: RgbToDkl,
    mask_parameters: kernels::MaskParameters,
    band_scores: Vec<BandScore>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

impl Workspace {
    /// Choose the fastest supported SIMD backend once, outside hot loops.
    pub fn new() -> Self {
        Self::with_backend(Backend::detect()).expect("detected backend")
    }

    /// Select a backend explicitly, for reproducibility and scalar validation.
    pub fn with_backend(backend: Backend) -> Result<Self, Error> {
        if !backend.is_available() {
            return Err(Error::UnsupportedBackend);
        }
        kernels::warmup(backend);
        Ok(Self {
            backend,
            current: std::array::from_fn(|_| Vec::new()),
            next: std::array::from_fn(|_| Vec::new()),
            mask: std::array::from_fn(|_| Vec::new()),
            tmp: Vec::new(),
            blur: GaussianBlur::new(backend),
            pyramid: pyramid::Kernels::for_backend(backend),
            conversion: RgbToDkl::new(backend),
            mask_parameters: mask_parameters(),
            band_scores: Vec::with_capacity(15),
        })
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Grow storage for an image. Buffers never shrink, and ping-pong storage
    /// returns to its initial orientation after every comparison/preparation.
    pub fn reserve(&mut self, width: usize, height: usize) -> Result<(), Error> {
        self.reserve_analysis(width, height)?;
        for plane in &mut self.mask {
            pyramid::ensure_len(plane, width * height.min(MASK_STRIP_ROWS));
        }
        self.blur.reserve(width, height);
        Ok(())
    }

    fn reserve_analysis(&mut self, width: usize, height: usize) -> Result<(), Error> {
        let count = crate::image::pixel_count(width, height)?;
        let next_count = width.div_ceil(2) * height.div_ceil(2);
        for c in 0..3 {
            pyramid::ensure_len(&mut self.current[c], count);
            pyramid::ensure_len(&mut self.next[c], next_count);
        }
        pyramid::ensure_len(&mut self.tmp, width);
        Ok(())
    }

    /// Heap bytes retained by scratch buffers (excluding allocator overhead).
    pub fn allocated_bytes(&self) -> usize {
        self.current
            .iter()
            .chain(&self.next)
            .chain(&self.mask)
            .chain([&self.tmp])
            .map(|v| v.capacity() * std::mem::size_of::<f32>())
            .sum::<usize>()
            + self.blur.allocated_bytes()
            + self.conversion.allocated_bytes()
            + self.band_scores.capacity() * std::mem::size_of::<BandScore>()
    }

    pub fn band_scores(&self) -> &[BandScore] {
        &self.band_scores
    }

    /// Evaluate a candidate, keeping storage independent of the reference.
    /// Validation errors leave diagnostics from the last successful comparison.
    pub fn compare(
        &mut self,
        reference: &PreparedReference,
        test: RgbImage<'_>,
    ) -> Result<Score, Error> {
        if test.width != reference.width || test.height != reference.height {
            return Err(Error::DimensionMismatch);
        }
        self.reserve(test.width, test.height)?;
        let display = reference.display;
        test.to_dkl(
            display.scale(),
            display.black(),
            &mut self.current,
            &mut self.conversion,
        )?;
        self.band_scores.clear();
        let mut per_channel = [0.0; 3];
        for (level, band) in reference.bands.iter().enumerate() {
            let (w, h, count) = (band.w, band.h, band.w * band.h);
            let base = level == reference.bands.len() - 1;
            let squares = if base {
                let weights = band.base_weights();
                pool_baseband(&self.current, &band.contrast, weights, count)
            } else {
                reduce_planes(
                    &self.current,
                    &mut self.next,
                    w,
                    h,
                    &mut self.tmp,
                    &self.pyramid,
                );
                let band_scale = if level == 0 { 1.0 } else { 2.0 };
                let sums = self.pool_band(w, h, BandReference::Prepared(band), band_scale);
                std::mem::swap(&mut self.current, &mut self.next);
                sums
            };
            let rms = band_rms(squares, count);
            accumulate_band(&mut per_channel, rms, base);
            self.band_scores.push(BandScore {
                width: w,
                height: h,
                frequency_cpd: band.frequency,
                rms,
            });
        }
        // Restore ownership of the full-size Gaussian allocation even when
        // the number of reductions is odd. This prevents next-call growth.
        if (reference.bands.len() - 1) % 2 == 1 {
            std::mem::swap(&mut self.current, &mut self.next);
        }
        Ok(finish_score(per_channel))
    }

    fn pool_band(
        &mut self,
        w: usize,
        h: usize,
        mut band: BandReference<'_>,
        scale: f32,
    ) -> [f64; 3] {
        let blur = blur_active(w, h);
        let halo = if blur { RADIUS } else { 0 };
        let mut pool = kernels::PoolAccumulator::new(self.backend, self.mask_parameters);
        let mut mask_end = 0;
        for y in (0..h).step_by(MASK_STRIP_ROWS) {
            let end = (y + MASK_STRIP_ROWS).min(h);
            // Compute each raw mask once. Inserting the next strip covers the
            // vertical halo and keeps the previously filtered rows in the ring.
            while mask_end < (end + halo).min(h) {
                let next = (mask_end + MASK_STRIP_ROWS).min(h);
                self.mask_strip(&mut band, (w, h), scale, mask_end..next);
                if blur {
                    let len = (next - mask_end) * w;
                    self.blur.horizontal_mask_rows(
                        self.mask.each_ref().map(|v| &v[..len]),
                        mask_end,
                        w,
                    );
                }
                mask_end = next;
            }
            let range = y * w..end * w;
            let len = range.len();
            if blur {
                self.blur
                    .vertical_mask_rows(self.mask.each_mut().map(|v| &mut v[..len]), y, w, h);
            }
            let current = self.current.each_ref().map(|v| &v[range.clone()]);
            let reference = band.contrast(range);
            let masks = self.mask.each_ref().map(|v| &v[..len]);
            if end == h {
                return pool.finish(current, reference, masks);
            }
            pool.push(current, reference, masks);
        }
        unreachable!("a band always has at least one row")
    }

    fn mask_strip(
        &mut self,
        band: &mut BandReference<'_>,
        dimensions: (usize, usize),
        scale: f32,
        rows: std::ops::Range<usize>,
    ) {
        band.prepare_strip(
            dimensions,
            rows.clone(),
            scale,
            &mut self.tmp,
            &self.pyramid,
        );
        let w = dimensions.0;
        let range = rows.start * w..rows.end * w;
        let len = range.len();
        expand_strip(
            &self.next,
            &mut self.mask,
            dimensions,
            rows,
            &mut self.tmp,
            &self.pyramid,
        );
        band.apply_contrast(
            self.backend,
            self.current.each_mut().map(|v| &mut v[range.clone()]),
            self.mask.each_mut().map(|v| &mut v[..len]),
            range,
            scale,
        );
    }
}

fn reduce_planes(
    current: &[Vec<f32>; 3],
    next: &mut [Vec<f32>; 3],
    w: usize,
    h: usize,
    tmp: &mut Vec<f32>,
    kernels: &pyramid::Kernels,
) {
    for (src, dst) in current.iter().zip(next) {
        pyramid::reduce(&src[..w * h], w, h, dst, tmp, kernels);
    }
}

fn expand_strip(
    coarse: &[Vec<f32>; 3],
    output: &mut [Vec<f32>; 3],
    (w, h): (usize, usize),
    rows: std::ops::Range<usize>,
    tmp: &mut [f32],
    kernels: &pyramid::Kernels,
) {
    let coarse_len = w.div_ceil(2) * h.div_ceil(2);
    let len = rows.len() * w;
    for (coarse, output) in coarse.iter().zip(output) {
        pyramid::expand_rows(
            &coarse[..coarse_len],
            rows.start,
            &mut output[..len],
            w,
            tmp,
            kernels,
        );
    }
}

#[inline]
fn weighted_contrast(value: f32, expanded: f32, inv_bg: f32, scale: f32, weight: f32) -> f32 {
    ((value - expanded) * inv_bg).min(1000.0) * scale * weight
}

#[inline]
fn weight_reference(
    current: [&mut [f32]; 3],
    mut expanded: [&mut [f32]; 3],
    csf: &[[f32; 32]; 3],
    scale: f32,
    mut positions: Option<&mut [f32]>,
) {
    for i in 0..current[0].len() {
        let bg = expanded[0][i].max(0.01);
        let position = csf_position(bg);
        let weights = band_weights(csf, position);
        let inv_bg = 1.0 / bg;
        for c in 0..3 {
            current[c][i] =
                weighted_contrast(current[c][i], expanded[c][i], inv_bg, scale, weights[c]);
        }
        if let Some(positions) = &mut positions {
            positions[i] = position;
        } else {
            for (plane, weight) in expanded.iter_mut().zip(weights) {
                plane[i] = weight;
            }
        }
    }
}

fn weight_base_reference(current: &mut [Vec<f32>; 3], count: usize) -> [f32; 3] {
    let bg = mean_luminance(&current[0][..count]);
    let inv_bg = 1.0 / bg;
    let weights = csf_at_position(&csf_at_frequency(0.1), csf_position(bg));
    for (plane, weight) in current.iter_mut().zip(weights) {
        for sample in &mut plane[..count] {
            *sample = base_contrast(*sample, inv_bg, weight);
        }
    }
    weights
}

#[inline]
fn base_contrast(value: f32, inv_bg: f32, weight: f32) -> f32 {
    (value * inv_bg).min(1000.0) * weight
}

pub(crate) fn pool_baseband(
    current: &[Vec<f32>; 3],
    reference: &[Vec<f32>; 3],
    weights: [f32; 3],
    count: usize,
) -> [f64; 3] {
    let inv_bg = 1.0 / mean_luminance(&current[0][..count]);
    std::array::from_fn(|c| {
        let mut sum = 0.0;
        for (&value, &reference) in current[c][..count].iter().zip(&reference[c][..count]) {
            let d = (base_contrast(value, inv_bg, weights[c]) - reference).abs() as f64;
            sum += d * d + 2.0 * (EPSILON as f64) * d;
        }
        sum
    })
}

/// Regularized RMS of one band's pooled squares.
pub(crate) fn band_rms(squares: [f64; 3], count: usize) -> [f64; 3] {
    squares.map(|sum| safe_pow(sum / count as f64, 0.5))
}

pub(crate) fn accumulate_band(per_channel: &mut [f64; 3], rms: [f64; 3], base: bool) {
    for c in 0..3 {
        per_channel[c] += safe_pow(rms[c] * if base { BASE_WEIGHT[c] } else { 1.0 }, 4.0);
    }
}

/// Pool the per-channel band sums into the score.
pub(crate) fn finish_score(per_channel: [f64; 3]) -> Score {
    // The epsilon requires both fourth-norm reductions to remain distinct.
    let total: f64 = per_channel
        .into_iter()
        .map(|sum| safe_pow(safe_pow(sum, 0.25), 4.0))
        .sum();
    let distortion = safe_pow(total, 0.25) * IMAGE_INTEGRATION;
    let deficit = if distortion <= 0.1 {
        JOD_A * 0.1_f64.powf(JOD_EXP - 1.0) * distortion
    } else {
        JOD_A * distortion.powf(JOD_EXP)
    };
    Score {
        jod: 10.0 - deficit,
        distortion,
    }
}

/// The masking-model constants every scoring path shares.
pub(crate) fn mask_parameters() -> kernels::MaskParameters {
    let xcm_log = [
        [-0.18950105, -5.962151, -4.318346],
        [2.5655594, 0.34406713, -2.7196465],
        [3.8118372, -1.0051706, -0.51933765],
    ];
    kernels::MaskParameters {
        mask_c: 10.0_f32.powf(-0.7954971),
        dmax: 10.0_f32.powf(2.5642455),
        mask_zero: MASK_Q.map(|q| EPSILON.powf(q)),
        diff_zero: EPSILON.powf(MASK_P),
        xcm: xcm_log.map(|row| row.map(|v| 2.0_f32.powf(v))),
    }
}

pub(crate) fn safe_pow(value: f64, exponent: f64) -> f64 {
    let epsilon = EPSILON as f64;
    ((value + epsilon).powf(exponent) - epsilon.powf(exponent)).max(0.0)
}

pub(crate) fn mean_luminance(plane: &[f32]) -> f32 {
    (plane.iter().map(|v| v.max(0.01) as f64).sum::<f64>() / plane.len() as f64) as f32
}

fn pyramid_height(w: usize, h: usize, ppd: f32) -> usize {
    let max_height = (w.min(h).ilog2() as usize).saturating_sub(1).min(14);
    for i in 0..=max_height {
        if band_frequency(i, ppd) <= 0.2 {
            return max_height.min(i + 1);
        }
    }
    max_height
}

fn band_frequency(level: usize, ppd: f32) -> f32 {
    if level == 0 {
        ppd * 0.5
    } else {
        ppd * 0.5 * 0.3228 / (1usize << (level - 1)) as f32
    }
}

fn csf_at_frequency(frequency: f32) -> [[f32; 32]; 3] {
    let f = frequency.log10();
    let index = tables::LOG_FREQUENCY
        .partition_point(|v| *v < f)
        .saturating_sub(1)
        .min(30);
    // Spatial frequency extrapolates outside the LUT, as in batch_interp1d.
    let t = (f - tables::LOG_FREQUENCY[index])
        / (tables::LOG_FREQUENCY[index + 1] - tables::LOG_FREQUENCY[index]);
    std::array::from_fn(|c| {
        std::array::from_fn(|l| {
            let row = &tables::LOG_SENSITIVITY[c][l];
            row[index] + (row[index + 1] - row[index]) * t
        })
    })
}

#[inline]
fn csf_position(luminance: f32) -> f32 {
    ((luminance.log10() - tables::LOG_LUMINANCE[0]) * 31.0
        / (tables::LOG_LUMINANCE[31] - tables::LOG_LUMINANCE[0]))
        .clamp(0.0, 31.0)
}

#[inline]
fn csf_at_position(csf: &[[f32; 32]; 3], pos: f32) -> [f32; 3] {
    let index = (pos as usize).min(30);
    let t = pos - index as f32;
    let correction = 10.0_f32.powf(-0.27974233 / 20.0);
    std::array::from_fn(|c| {
        10.0_f32.powf(csf[c][index] * (1.0 - t) + csf[c][index + 1] * t) * correction
    })
}

#[inline]
fn band_weights(csf: &[[f32; 32]; 3], position: f32) -> [f32; 3] {
    let weights = csf_at_position(csf, position);
    std::array::from_fn(|c| weights[c] * CH_GAIN[c])
}

fn decode_weights(csf: &[[f32; 32]; 3], positions: &[f32], [a, b, c]: [&mut [f32]; 3]) {
    for ((&position, a), (b, c)) in positions.iter().zip(a).zip(b.iter_mut().zip(c)) {
        [*a, *b, *c] = band_weights(csf, position);
    }
}
