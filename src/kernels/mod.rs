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

//! Private SIMD boundary. Safe entry points validate slice lengths; kernels
//! only touch complete vectors, and scalar tails handle every remaining lane.
//! Runtime CPU detection precedes every target-feature entry point.

#[cfg(any(
    all(feature = "neon", target_arch = "aarch64"),
    all(
        any(feature = "avx", feature = "sse"),
        any(target_arch = "x86", target_arch = "x86_64")
    )
))]
pub(crate) mod vector;

/// Arithmetic backend. Scalar uses libm powers; SIMD uses a bounded polynomial
/// approximation to powers in the masking stage. Calibration/pooling are shared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Scalar,
    Neon,
    Avx2,
    Sse2,
}

impl Backend {
    /// Select the best compiled backend supported by the current CPU.
    pub fn detect() -> Self {
        static DETECTED: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();
        *DETECTED.get_or_init(|| {
            for backend in [Self::Avx2, Self::Neon, Self::Sse2] {
                if backend.is_available() {
                    return backend;
                }
            }
            Self::Scalar
        })
    }

    pub fn is_available(self) -> bool {
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        if self == Self::Neon {
            return std::arch::is_aarch64_feature_detected!("neon");
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
        if self == Self::Avx2 {
            return std::arch::is_x86_feature_detected!("avx2");
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
        if self == Self::Sse2 {
            return std::arch::is_x86_feature_detected!("sse2");
        }
        self == Self::Scalar
    }
}

pub(crate) const EPSILON: f32 = 0.00001;
pub(crate) const MASK_P: f32 = 2.2643552;
pub(crate) const MASK_Q: [f32; 3] = [1.3026227, 2.8885908, 3.6807714];

#[derive(Clone, Copy)]
pub(crate) struct MaskParameters {
    pub mask_c: f32,
    pub dmax: f32,
    pub mask_zero: [f32; 3],
    pub diff_zero: f32,
    pub xcm: [[f32; 3]; 3],
}

pub(crate) struct Contrast<'a> {
    current: [&'a mut [f32]; 3],
    expanded: [&'a [f32]; 3],
    sensitivity: [&'a [f32]; 3],
    reference: [&'a [f32]; 3],
    mask: [&'a mut [f32]; 3],
    scale: f32,
}

/// A compact output strip reading Gaussian/reference rows at `source_stride`.
/// The expanded planes and both outputs contain exactly width * height pixels.
pub(crate) struct ContrastRegion<'a> {
    pub current: [&'a [f32]; 3],
    pub expanded: [&'a [f32]; 3],
    pub sensitivity: [&'a [f32]; 3],
    pub reference: [&'a [f32]; 3],
    pub contrast: [&'a mut [f32]; 3],
    pub mask: [&'a mut [f32]; 3],
    pub width: usize,
    pub height: usize,
    pub source_stride: usize,
    pub scale: f32,
}

pub(crate) struct Pool<'a> {
    current: [&'a [f32]; 3],
    reference: [&'a [f32]; 3],
    blurred: [&'a [f32]; 3],
    parameters: MaskParameters,
}

pub(crate) const POOL_TILE: usize = 8;

/// A tile-aligned strip. Each input starts at its upper-left pixel and may
/// have a different row stride (contrast halo, full reference, pooled masks).
pub(crate) struct TiledPool<'a> {
    pub current: [&'a [f32]; 3],
    pub reference: [&'a [f32]; 3],
    pub blurred: [&'a [f32]; 3],
    pub strides: [usize; 3],
    pub width: usize,
    pub height: usize,
    pub parameters: MaskParameters,
}

#[allow(unused_mut, unused_unsafe)] // Scalar-only builds have no dispatch block.
pub(crate) fn contrast(
    backend: Backend,
    current: [&mut [f32]; 3],
    expanded: [&[f32]; 3],
    sensitivity: [&[f32]; 3],
    reference: [&[f32]; 3],
    mask: [&mut [f32]; 3],
    scale: f32,
) {
    assert!(backend.is_available());
    let count = current[0].len();
    assert!(current.iter().all(|p| p.len() == count));
    assert!(
        expanded
            .iter()
            .chain(&sensitivity)
            .chain(&reference)
            .all(|p| p.len() == count)
    );
    assert!(mask.iter().all(|p| p.len() == count));
    let mut data = Contrast {
        current,
        expanded,
        sensitivity,
        reference,
        mask,
        scale,
    };
    let mut done = 0;
    // SAFETY: Equal initialized slice lengths were checked above. Workspace
    // construction validates backend availability; the dispatch checks it again
    // by architecture, and only a supported target-feature entry is reachable.
    unsafe {
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        if backend == Backend::Neon {
            done = crate::neon::contrast_neon(&mut data);
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
        if backend == Backend::Avx2 {
            done = crate::avx::contrast_avx2(&mut data);
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
        if backend == Backend::Sse2 {
            done = crate::sse::contrast_sse2(&mut data);
        }
    }
    let _ = backend;
    for i in done..count {
        let bg = data.expanded[0][i].max(0.01);
        for c in 0..3 {
            let weighted = ((data.current[c][i] - data.expanded[c][i]) / bg).min(1000.0)
                * data.scale
                * data.sensitivity[c][i];
            data.current[c][i] = weighted;
            data.mask[c][i] = weighted.abs().min(data.reference[c][i].abs());
        }
    }
}

/// Fuse the Gaussian-to-scratch copy with contrast conversion. Validate and
/// dispatch once per strip, retaining the old per-row SIMD/tail boundaries.
#[allow(unused_mut, unused_unsafe)]
pub(crate) fn contrast_region(backend: Backend, mut data: ContrastRegion<'_>) {
    assert!(backend.is_available());
    assert!(data.width > 0 && data.height > 0 && data.source_stride >= data.width);
    let count = data
        .width
        .checked_mul(data.height)
        .expect("contrast output size overflow");
    let required = (data.height - 1)
        .checked_mul(data.source_stride)
        .and_then(|n| n.checked_add(data.width))
        .expect("contrast input size overflow");
    assert!(
        data.current
            .iter()
            .chain(&data.sensitivity)
            .chain(&data.reference)
            .all(|p| p.len() >= required)
    );
    assert!(data.expanded.iter().all(|p| p.len() == count));
    assert!(
        data.contrast
            .iter()
            .chain(&data.mask)
            .all(|p| p.len() == count)
    );
    let mut done = 0;
    // SAFETY: Input strides cover every source row, compact planes have the
    // checked pixel count, and only a CPU-supported target-feature entry runs.
    unsafe {
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        if backend == Backend::Neon {
            done = crate::neon::contrast_region_neon(&mut data);
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
        if backend == Backend::Avx2 {
            done = crate::avx::contrast_region_avx2(&mut data);
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
        if backend == Backend::Sse2 {
            done = crate::sse::contrast_region_sse2(&mut data);
        }
    }
    let n = data.width - done;
    if n == 0 {
        return;
    }
    let luminance = data.expanded[0];
    // Slice each row once, then let iterators carry its bounds through the
    // scalar fallback/tail instead of indexing all six planes per sample.
    for (c, (contrast, mask)) in data.contrast.into_iter().zip(data.mask).enumerate() {
        let current = data.current[c];
        let expanded = data.expanded[c];
        let sensitivity = data.sensitivity[c];
        let reference = data.reference[c];
        for y in 0..data.height {
            let src = y * data.source_stride + done;
            let dst = y * data.width + done;
            let inputs = current[src..src + n].iter().zip(&expanded[dst..dst + n]);
            let weights = sensitivity[src..src + n]
                .iter()
                .zip(&reference[src..src + n]);
            let output = contrast[dst..dst + n]
                .iter_mut()
                .zip(&mut mask[dst..dst + n]);
            for ((((&value, &low), &bg), (&sensitivity, &reference)), (contrast, mask)) in inputs
                .zip(&luminance[dst..dst + n])
                .zip(weights)
                .zip(output)
            {
                let weighted =
                    ((value - low) / bg.max(0.01)).min(1000.0) * data.scale * sensitivity;
                *contrast = weighted;
                *mask = weighted.abs().min(reference.abs());
            }
        }
    }
}

#[allow(unused_mut, unused_unsafe)] // Scalar-only builds have no dispatch block.
pub(crate) fn pool(
    backend: Backend,
    current: [&[f32]; 3],
    reference: [&[f32]; 3],
    blurred: [&[f32]; 3],
    parameters: MaskParameters,
) -> [f64; 3] {
    assert!(backend.is_available());
    let count = current[0].len();
    assert!(
        current
            .iter()
            .chain(&reference)
            .chain(&blurred)
            .all(|p| p.len() == count)
    );
    let data = Pool {
        current,
        reference,
        blurred,
        parameters,
    };
    let (mut done, mut squares) = (0, [0.0; 3]);
    // SAFETY: All input planes have the checked initialized count, and backend
    // is CPU-validated by Workspace. SIMD loops only consume complete vectors.
    unsafe {
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        if backend == Backend::Neon {
            (done, squares) = crate::neon::pool_neon(&data);
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
        if backend == Backend::Avx2 {
            (done, squares) = crate::avx::pool_avx2(&data);
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
        if backend == Backend::Sse2 {
            (done, squares) = crate::sse::pool_sse2(&data);
        }
    }
    let _ = backend;
    pool_scalar_tail(&data, done, &mut squares);
    squares
}

#[inline]
fn pool_scalar_tail(data: &Pool<'_>, done: usize, squares: &mut [f64; 3]) {
    let parameters = data.parameters;
    let count = data.current[0].len();
    for i in done..count {
        let activity: [f32; 3] = std::array::from_fn(|c| {
            (data.blurred[c][i] * parameters.mask_c + EPSILON).powf(MASK_Q[c])
                - parameters.mask_zero[c]
        });
        for (c, sum) in squares.iter_mut().enumerate() {
            let mask = activity[0] * parameters.xcm[0][c]
                + activity[1] * parameters.xcm[1][c]
                + activity[2] * parameters.xcm[2][c];
            let diff = (data.current[c][i] - data.reference[c][i]).abs();
            let du = ((diff + EPSILON).powf(MASK_P) - parameters.diff_zero) / (1.0 + mask);
            let d = (parameters.dmax * du / (parameters.dmax + du)) as f64;
            *sum += d * d + 2.0 * (EPSILON as f64) * d;
        }
    }
}

/// Dispatch and validate once for a whole strip, writing directly into the
/// existing tile cache. Row reductions and scalar tails stay tile-local.
/// Incremental pooling uses [`pool_row_parts`]; this is its test oracle.
#[cfg(test)]
#[allow(unused_unsafe)]
pub(crate) fn pool_tiles(
    backend: Backend,
    data: &TiledPool<'_>,
    tiles: &mut [[f64; 3]],
    tile_stride: usize,
) {
    assert!(backend.is_available());
    assert!(data.width > 0 && data.height > 0);
    for (planes, stride) in [data.current, data.reference, data.blurred]
        .iter()
        .zip(data.strides)
    {
        assert!(stride >= data.width);
        let required = (data.height - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(data.width))
            .expect("pool input size overflow");
        assert!(planes.iter().all(|plane| plane.len() >= required));
    }
    let columns = data.width.div_ceil(POOL_TILE);
    let rows = data.height.div_ceil(POOL_TILE);
    assert!(tile_stride >= columns);
    let required = (rows - 1)
        .checked_mul(tile_stride)
        .and_then(|n| n.checked_add(columns))
        .expect("pool output size overflow");
    assert!(tiles.len() >= required);
    // SAFETY: Every input covers the strip at its checked stride and the tile
    // output covers every written row. Only CPU-supported entries are called.
    unsafe {
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        if backend == Backend::Neon {
            return crate::neon::pool_tiles_neon(data, tiles, tile_stride);
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
        if backend == Backend::Avx2 {
            return crate::avx::pool_tiles_avx2(data, tiles, tile_stride);
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
        if backend == Backend::Sse2 {
            return crate::sse::pool_tiles_sse2(data, tiles, tile_stride);
        }
    }
    pool_tiles_with(data, tiles, tile_stride, |row| {
        let mut squares = [0.0; 3];
        pool_scalar_tail(row, 0, &mut squares);
        squares
    });
}

/// Per-row terms of a strip: `parts[y * parts_stride + t]` receives exactly
/// the term [`pool_tiles`] adds for row `y` into tile `t`, so callers can keep
/// row terms and re-add them in row order.
#[allow(unused_unsafe)]
pub(crate) fn pool_row_parts(
    backend: Backend,
    data: &TiledPool<'_>,
    parts: &mut [[f64; 3]],
    parts_stride: usize,
) {
    assert!(backend.is_available());
    assert!(data.width > 0 && data.height > 0);
    for (planes, stride) in [data.current, data.reference, data.blurred]
        .iter()
        .zip(data.strides)
    {
        let required = (data.height - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(data.width))
            .expect("pool input size overflow");
        assert!(stride >= data.width || data.height == 1);
        assert!(planes.iter().all(|plane| plane.len() >= required));
    }
    let columns = data.width.div_ceil(POOL_TILE);
    assert!(parts_stride >= columns || data.height == 1);
    assert!(parts.len() >= (data.height - 1) * parts_stride + columns);
    // SAFETY: Every input covers the row segment, the output covers every
    // tile, and only CPU-supported entries are called.
    unsafe {
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        if backend == Backend::Neon {
            return crate::neon::pool_parts_neon(data, parts, parts_stride);
        }
        #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
        if backend == Backend::Avx2 {
            return crate::avx::pool_parts_avx2(data, parts, parts_stride);
        }
        #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
        if backend == Backend::Sse2 {
            return crate::sse::pool_parts_sse2(data, parts, parts_stride);
        }
    }
    pool_parts_with(data, parts, parts_stride, |row| {
        let mut squares = [0.0; 3];
        pool_scalar_tail(row, 0, &mut squares);
        squares
    });
}

/// Row terms of [`pool_tiles_with`], one output row per strip row.
#[inline]
fn pool_parts_with(
    data: &TiledPool<'_>,
    parts: &mut [[f64; 3]],
    parts_stride: usize,
    mut pool_row: impl FnMut(&Pool<'_>) -> [f64; 3],
) {
    let columns = data.width.div_ceil(POOL_TILE);
    for y in 0..data.height {
        let out = &mut parts[y * parts_stride..][..columns];
        for (part, x0) in out.iter_mut().zip((0..data.width).step_by(POOL_TILE)) {
            let n = POOL_TILE.min(data.width - x0);
            let [c, r, b] = data.strides.map(|stride| y * stride + x0);
            let row = Pool {
                current: data.current.map(|v| &v[c..c + n]),
                reference: data.reference.map(|v| &v[r..r + n]),
                blurred: data.blurred.map(|v| &v[b..b + n]),
                parameters: data.parameters,
            };
            *part = pool_row(&row);
        }
    }
}

#[cfg(test)]
#[inline]
fn pool_tiles_with(
    data: &TiledPool<'_>,
    tiles: &mut [[f64; 3]],
    tile_stride: usize,
    mut pool_row: impl FnMut(&Pool<'_>) -> [f64; 3],
) {
    let columns = data.width.div_ceil(POOL_TILE);
    for (ty, y0) in (0..data.height).step_by(POOL_TILE).enumerate() {
        let out = &mut tiles[ty * tile_stride..][..columns];
        for (sums, x0) in out.iter_mut().zip((0..data.width).step_by(POOL_TILE)) {
            *sums = [0.0; 3];
            let n = POOL_TILE.min(data.width - x0);
            for y in y0..(y0 + POOL_TILE).min(data.height) {
                let [c, r, b] = data.strides.map(|stride| y * stride + x0);
                let row = Pool {
                    current: data.current.map(|v| &v[c..c + n]),
                    reference: data.reference.map(|v| &v[r..r + n]),
                    blurred: data.blurred.map(|v| &v[b..b + n]),
                    parameters: data.parameters,
                };
                let part = pool_row(&row);
                for (sum, value) in sums.iter_mut().zip(part) {
                    *sum += value;
                }
            }
        }
    }
}

pub(crate) fn warmup(backend: Backend) {
    #[cfg(any(
        all(feature = "neon", target_arch = "aarch64"),
        all(
            any(feature = "avx", feature = "sse"),
            any(target_arch = "x86", target_arch = "x86_64")
        )
    ))]
    if backend != Backend::Scalar {
        let _ = vector::power_tables();
    }
    let _ = backend;
}

#[cfg(test)]
mod contrast_tests;

#[cfg(test)]
mod tile_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(unused_mut, unused_unsafe)]
    fn powers(backend: Backend, input: &[f32], output: &mut [f32], exponent: f32) {
        assert!(backend.is_available());
        assert_eq!(input.len(), output.len());
        assert!(input.iter().all(|x| x.is_normal() && *x > 0.0));
        let mut done = 0;
        // SAFETY: Feature availability, normal positive inputs, equal lengths,
        // and normal result range are established by this test's input sweep.
        unsafe {
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            if backend == Backend::Neon {
                done = crate::neon::powers_neon(input, output, exponent);
            }
            #[cfg(all(feature = "avx", any(target_arch = "x86_64", target_arch = "x86")))]
            if backend == Backend::Avx2 {
                done = crate::avx::powers_avx2(input, output, exponent);
            }
            #[cfg(all(feature = "sse", any(target_arch = "x86_64", target_arch = "x86")))]
            if backend == Backend::Sse2 {
                done = crate::sse::powers_sse2(input, output, exponent);
            }
        }
        for i in done..input.len() {
            output[i] = input[i].powf(exponent);
        }
    }

    #[test]
    fn vector_powers_match_libm_across_masking_range_and_tails() {
        let input: Vec<f32> = (0..10003)
            .map(|i| 2.0_f32.powf(-16.6 + i as f32 * 40.0 / 10002.0))
            .collect();
        for backend in [Backend::Scalar, Backend::Neon, Backend::Avx2, Backend::Sse2] {
            if !backend.is_available() {
                continue;
            }
            for exponent in [MASK_P, MASK_Q[0], MASK_Q[1], MASK_Q[2]] {
                let mut output = vec![0.0; input.len()];
                powers(backend, &input, &mut output, exponent);
                for (&x, &got) in input.iter().zip(&output) {
                    let expected = x.powf(exponent);
                    let relative = (got / expected - 1.0).abs();
                    assert!(
                        relative < 0.000002,
                        "{backend:?}: {x}^{exponent} {got} != {expected}, error={relative}"
                    );
                }
            }
        }
    }
}
