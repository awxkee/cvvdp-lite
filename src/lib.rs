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

//! Still-image ColorVideoVDP, using the calibrated 0.5.7 base model.
//!
//! This crate supports SDR sRGB and normalized linear sRGB images. Prepare a
//! reference once, then reuse an [`Evaluator`] for encoder candidate searches.
//! The reference owns the expensive contrast and sensitivity analysis; the
//! evaluator reserves its working allocations and streams the test pyramid.
//! Keep a [`Workspace`] across reference lifetimes for long-lived arbiter workers.
//! Default SIMD dispatch supports NEON, AVX2, and SSE2, with scalar fallback.
//! [`Backend::Scalar`] selects libm powers; SIMD powers have a tested numerical
//! tolerance. See the README for error measurements and worker memory usage.
//! Scores are in JOD: 10 is identical, larger is better, and negative is valid.
//! See the crate README for upstream edge corrections and validation limits.
//!
//! ```
//! use cvvdp_lite::{CvvdpLite, Display, RgbImage};
//! let pixels = vec![128u8; 32 * 32 * 3];
//! let image = RgbImage::srgb8(&pixels, 32, 32, 32 * 3)?;
//! let metric = CvvdpLite::new(Display::standard_4k())?;
//! let reference = metric.prepare_reference(image)?;
//! let mut evaluator = reference.evaluator();
//! assert_eq!(evaluator.compare(image)?.jod, 10.0);
//! # Ok::<(), cvvdp_lite::Error>(())
//! ```

#[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(unsafe_code)]
mod avx;
mod blur;
mod image;
mod incremental;
#[allow(unsafe_code)]
mod kernels;
mod model;
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
#[allow(unsafe_code)]
mod neon;
mod pyramid;
#[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
#[allow(unsafe_code)]
mod sse;
mod tables;

pub use image::RgbImage;
pub use incremental::{Checkpoint, IncrementalState, Rect};
pub use kernels::Backend;
pub use model::{BandScore, CvvdpLite, Display, Evaluator, PreparedReference, Score, Workspace};

/// Invalid image layout, samples, viewing conditions, or backend selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Width and height must be nonzero and fit addressable memory.
    InvalidDimensions,
    /// A stride is too short, or a buffer does not cover the last pixel.
    InvalidLayout,
    /// Reference and test dimensions differ.
    DimensionMismatch,
    /// A floating-point sample is NaN or infinite.
    NonFiniteSample,
    /// Display parameters are non-finite or outside their physical ranges.
    InvalidDisplay,
    /// The requested SIMD backend is unavailable on this CPU or build.
    UnsupportedBackend,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidDimensions => "image dimensions must be nonzero and fit in memory",
            Self::InvalidLayout => "image stride or buffer length is invalid",
            Self::DimensionMismatch => "reference and test dimensions differ",
            Self::NonFiniteSample => "image contains a non-finite sample",
            Self::InvalidDisplay => "display parameters are invalid",
            Self::UnsupportedBackend => "requested SIMD backend is unavailable",
        })
    }
}

impl std::error::Error for Error {}
