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

//! Region-incremental scoring for encoder loops that change one area of a
//! candidate at a time. The state keeps the test image's Gaussian pyramid and
//! the pooled squares of every 8x8 tile in every band. An update recomputes
//! only the changed area's footprint through each level (reduction ±2,
//! expansion ±2, masking blur ±6), then re-pools the touched tiles. The coarse
//! baseband, which adapts to the whole image's mean, is rescored in full.
//!
//! Pooling uses SIMD powers whose lanes differ slightly from the scalar tail,
//! so every pooled and converted row segment starts on a multiple of 8 and
//! tiles never move: a state built from scratch and one reached by any
//! sequence of updates hold bit-identical sums. Scores agree with
//! [`crate::Workspace::compare`] to rounding (a different summation order).

use crate::blur::{self, GaussianBlur, RADIUS};
use crate::image::RgbToDkl;
use crate::kernels::{self, EPSILON, MaskParameters};
use crate::model::{self, PreparedReference};
use crate::pyramid::{self, ensure_len};
use crate::{Backend, Error, RgbImage, Score};

const TILE: usize = kernels::POOL_TILE;
const ALIGN: usize = 8;
// Keep temporary images independent of image height. Whole tile rows preserve
// the SIMD lane placement and summation order of every pooled tile.
const POOL_ROWS: usize = 64;

/// A half-open pixel rectangle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Rect {
    pub fn new(x0: usize, y0: usize, x1: usize, y1: usize) -> Self {
        Self { x0, y0, x1, y1 }
    }

    fn full(w: usize, h: usize) -> Self {
        Self::new(0, 0, w, h)
    }

    fn is_empty(self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    fn clamp(self, w: usize, h: usize) -> Self {
        Self::new(
            self.x0.min(w),
            self.y0.min(h),
            self.x1.min(w),
            self.y1.min(h),
        )
    }

    fn grow(self, r: usize, w: usize, h: usize) -> Self {
        Self::new(
            self.x0.saturating_sub(r),
            self.y0.saturating_sub(r),
            (self.x1 + r).min(w),
            (self.y1 + r).min(h),
        )
    }

    fn union(self, o: Self) -> Self {
        Self::new(
            self.x0.min(o.x0),
            self.y0.min(o.y0),
            self.x1.max(o.x1),
            self.y1.max(o.y1),
        )
    }

    /// Columns widened to multiples of `step` (or the right edge).
    fn align_x(self, step: usize, w: usize) -> Self {
        Self::new(
            self.x0 / step * step,
            self.y0,
            self.x1.div_ceil(step).saturating_mul(step).min(w),
            self.y1,
        )
    }

    /// Rows widened to whole tiles, columns to whole tiles.
    fn align_tiles(self, w: usize, h: usize) -> Self {
        Self::new(
            self.x0 / TILE * TILE,
            self.y0 / TILE * TILE,
            self.x1.div_ceil(TILE).saturating_mul(TILE).min(w),
            self.y1.div_ceil(TILE).saturating_mul(TILE).min(h),
        )
    }

    fn width(self) -> usize {
        self.x1 - self.x0
    }

    fn height(self) -> usize {
        self.y1 - self.y0
    }
}

/// Coarse outputs of the five-tap reduction that read `fine`.
fn reduce_footprint(fine: Rect, nw: usize, nh: usize) -> Rect {
    Rect::new(
        fine.x0.saturating_sub(2) / 2,
        fine.y0.saturating_sub(2) / 2,
        fine.x1.div_ceil(2) + 1,
        fine.y1.div_ceil(2) + 1,
    )
    .clamp(nw, nh)
}

/// Fine pixels whose expansion reads `coarse`.
fn expand_footprint(coarse: Rect, w: usize, h: usize) -> Rect {
    Rect::new(
        (2 * coarse.x0).saturating_sub(2),
        (2 * coarse.y0).saturating_sub(2),
        2 * coarse.x1 + 2,
        2 * coarse.y1 + 2,
    )
    .clamp(w, h)
}

/// Inclusive edge reflection of the five-tap reduction.
#[inline]
fn reflect(i: isize, n: usize) -> usize {
    if n == 1 {
        return 0;
    }
    if i < 0 {
        (-i - 1) as usize
    } else if i >= n as isize {
        2 * n - i as usize - 1
    } else {
        i as usize
    }
}

/// Exclusive edge reflection of the masking blur.
#[inline]
fn reflect_exclusive(i: isize, n: usize) -> usize {
    if i < 0 {
        (-i) as usize
    } else if i >= n as isize {
        2 * n - i as usize - 2
    } else {
        i as usize
    }
}

fn blur_active(w: usize, h: usize) -> bool {
    w > RADIUS && h > RADIUS
}

struct Level {
    w: usize,
    h: usize,
    /// Test Gaussian level (DKL planes).
    gauss: [Vec<f32>; 3],
    tiles_x: usize,
    /// Pooled squares per tile; empty for the baseband.
    tiles: Vec<[f64; 3]>,
    /// Sum of each row of tiles, in tile order.
    row_sums: Vec<[f64; 3]>,
}

impl Level {
    fn sum_tile_row(&mut self, ty: usize) {
        let mut sums = [0.0f64; 3];
        for tile in &self.tiles[ty * self.tiles_x..(ty + 1) * self.tiles_x] {
            for (sum, value) in sums.iter_mut().zip(tile) {
                *sum += value;
            }
        }
        self.row_sums[ty] = sums;
    }
}

#[derive(Default)]
struct Scratch {
    row: [Vec<f32>; 3],
    tmp: Vec<f32>,
    expanded: [Vec<f32>; 3],
    contrast: [Vec<f32>; 3],
    mask: [Vec<f32>; 3],
    blurred: [Vec<f32>; 3],
    padded: Vec<f32>,
}

impl Scratch {
    fn reserve(&mut self, w: usize, h: usize) {
        let area = w * h.min(POOL_ROWS + 2 * RADIUS);
        let pooled = w * h.min(POOL_ROWS);
        for plane in &mut self.row {
            ensure_len(plane, w);
        }
        for plane in self
            .expanded
            .iter_mut()
            .chain(&mut self.contrast)
            .chain(&mut self.mask)
        {
            ensure_len(plane, area);
        }
        for plane in &mut self.blurred {
            ensure_len(plane, pooled);
        }
        ensure_len(&mut self.tmp, w);
        self.padded.reserve_exact(w + 2 * RADIUS);
    }
}

/// What an update overwrote, for an exact [`IncrementalState::rollback`].
#[derive(Default)]
pub struct Checkpoint {
    gauss: Vec<(usize, Rect)>,
    gauss_data: Vec<f32>,
    tiles: Vec<(usize, usize, [f64; 3])>,
    base: [f64; 3],
}

impl Checkpoint {
    /// Heap bytes retained for rollback, excluding allocator overhead.
    pub fn allocated_bytes(&self) -> usize {
        self.gauss.capacity() * std::mem::size_of::<(usize, Rect)>()
            + self.gauss_data.capacity() * std::mem::size_of::<f32>()
            + self.tiles.capacity() * std::mem::size_of::<(usize, usize, [f64; 3])>()
    }
}

/// Incrementally maintained score of one test image against a prepared reference.
pub struct IncrementalState {
    backend: Backend,
    conversion: RgbToDkl,
    blur: GaussianBlur,
    pyramid: pyramid::Kernels,
    params: MaskParameters,
    width: usize,
    height: usize,
    levels: Vec<Level>,
    base: [f64; 3],
    scratch: Scratch,
}

impl IncrementalState {
    /// Heap bytes retained by the pyramid, tile sums and scratch buffers,
    /// excluding the shared reference and allocator overhead.
    pub fn allocated_bytes(&self) -> usize {
        let scratch = &self.scratch;
        self.levels.capacity() * std::mem::size_of::<Level>()
            + self
                .levels
                .iter()
                .map(|level| {
                    level
                        .gauss
                        .iter()
                        .map(|v| v.capacity() * std::mem::size_of::<f32>())
                        .sum::<usize>()
                        + (level.tiles.capacity() + level.row_sums.capacity())
                            * std::mem::size_of::<[f64; 3]>()
                })
                .sum::<usize>()
            + scratch
                .row
                .iter()
                .chain(&scratch.expanded)
                .chain(&scratch.contrast)
                .chain(&scratch.mask)
                .chain(&scratch.blurred)
                .chain([&scratch.tmp, &scratch.padded])
                .map(|v| v.capacity() * std::mem::size_of::<f32>())
                .sum::<usize>()
            + self.blur.allocated_bytes()
            + self.conversion.allocated_bytes()
    }

    /// Score `test` in full and keep the analysis for later updates.
    pub fn new(reference: &PreparedReference, test: RgbImage<'_>) -> Result<Self, Error> {
        Self::with_backend(reference, test, Backend::detect())
    }

    pub fn with_backend(
        reference: &PreparedReference,
        test: RgbImage<'_>,
        backend: Backend,
    ) -> Result<Self, Error> {
        if !backend.is_available() {
            return Err(Error::UnsupportedBackend);
        }
        if test.width != reference.width || test.height != reference.height {
            return Err(Error::DimensionMismatch);
        }
        kernels::warmup(backend);
        let base_index = reference.bands.len() - 1;
        let levels = reference
            .bands
            .iter()
            .enumerate()
            .map(|(k, band)| {
                let tiles_x = band.w.div_ceil(TILE);
                let n_tiles = if k == base_index {
                    0
                } else {
                    tiles_x * band.h.div_ceil(TILE)
                };
                Level {
                    w: band.w,
                    h: band.h,
                    gauss: std::array::from_fn(|_| vec![0.0; band.w * band.h]),
                    tiles_x,
                    tiles: vec![[0.0; 3]; n_tiles],
                    row_sums: vec![
                        [0.0; 3];
                        if n_tiles == 0 {
                            0
                        } else {
                            band.h.div_ceil(TILE)
                        }
                    ],
                }
            })
            .collect();
        let mut state = Self {
            backend,
            conversion: RgbToDkl::new(backend),
            blur: GaussianBlur::new(backend),
            pyramid: pyramid::Kernels::for_backend(backend),
            params: model::mask_parameters(),
            width: test.width,
            height: test.height,
            levels,
            base: [0.0; 3],
            scratch: Scratch::default(),
        };
        state.scratch.reserve(test.width, test.height);
        state.apply(reference, test, Rect::full(test.width, test.height), None)?;
        Ok(state)
    }

    /// The current score.
    pub fn score(&self) -> Score {
        let mut per_channel = [0.0; 3];
        let base_index = self.levels.len() - 1;
        for (k, level) in self.levels.iter().enumerate() {
            let squares = if k == base_index {
                self.base
            } else {
                let mut sums = [0.0f64; 3];
                for row in &level.row_sums {
                    for (sum, value) in sums.iter_mut().zip(row) {
                        *sum += value;
                    }
                }
                sums
            };
            let rms = model::band_rms(squares, level.w * level.h);
            model::accumulate_band(&mut per_channel, rms, k == base_index);
        }
        model::finish_score(per_channel)
    }

    /// Re-score after `test` changed only inside `rect` (pixels outside it must
    /// equal those of the previous call).
    pub fn update(
        &mut self,
        reference: &PreparedReference,
        test: RgbImage<'_>,
        rect: Rect,
    ) -> Result<Score, Error> {
        self.apply(reference, test, rect, None)?;
        Ok(self.score())
    }

    /// [`Self::update`], first saving what it overwrites into `checkpoint`.
    pub fn update_checkpointed(
        &mut self,
        reference: &PreparedReference,
        test: RgbImage<'_>,
        rect: Rect,
        checkpoint: &mut Checkpoint,
    ) -> Result<Score, Error> {
        checkpoint.gauss.clear();
        checkpoint.gauss_data.clear();
        checkpoint.tiles.clear();
        checkpoint.base = self.base;
        self.apply(reference, test, rect, Some(checkpoint))?;
        Ok(self.score())
    }

    /// Undo the [`Self::update_checkpointed`] that filled `checkpoint`.
    pub fn rollback(&mut self, checkpoint: &Checkpoint) {
        let mut at = 0;
        for &(k, r) in &checkpoint.gauss {
            let level = &mut self.levels[k];
            let w = level.w;
            for plane in &mut level.gauss {
                for y in r.y0..r.y1 {
                    let n = r.width();
                    plane[y * w + r.x0..][..n].copy_from_slice(&checkpoint.gauss_data[at..at + n]);
                    at += n;
                }
            }
        }
        for &(k, t, sums) in checkpoint.tiles.iter().rev() {
            self.levels[k].tiles[t] = sums;
        }
        // Checkpoint tiles are stored by level, then by row, then by column.
        // Restore all tiles before summing each affected row exactly once.
        let mut previous = None;
        for &(k, t, _) in &checkpoint.tiles {
            let level = &mut self.levels[k];
            let ty = t / level.tiles_x;
            if previous != Some((k, ty)) {
                level.sum_tile_row(ty);
                previous = Some((k, ty));
            }
        }
        self.base = checkpoint.base;
    }

    fn save_gauss(&self, checkpoint: &mut Option<&mut Checkpoint>, k: usize, r: Rect) {
        let Some(cp) = checkpoint.as_deref_mut() else {
            return;
        };
        let level = &self.levels[k];
        cp.gauss.push((k, r));
        for plane in &level.gauss {
            for y in r.y0..r.y1 {
                cp.gauss_data
                    .extend_from_slice(&plane[y * level.w + r.x0..][..r.width()]);
            }
        }
    }

    fn apply(
        &mut self,
        reference: &PreparedReference,
        test: RgbImage<'_>,
        rect: Rect,
        mut checkpoint: Option<&mut Checkpoint>,
    ) -> Result<(), Error> {
        if test.width != self.width || test.height != self.height {
            return Err(Error::DimensionMismatch);
        }
        let rect = rect.clamp(self.width, self.height);
        if rect.is_empty() {
            return Ok(());
        }
        let display = reference.display;
        // Level 0: the changed pixels in DKL, converted in aligned row segments.
        let mut dirty = rect.align_x(ALIGN, self.width);
        self.save_gauss(&mut checkpoint, 0, dirty);
        for y in dirty.y0..dirty.y1 {
            test.crop(dirty.x0, y, dirty.width(), 1).to_dkl(
                display.scale(),
                display.black(),
                &mut self.scratch.row,
                &mut self.conversion,
            )?;
            let level = &mut self.levels[0];
            for (plane, row) in level.gauss.iter_mut().zip(&self.scratch.row) {
                plane[y * level.w + dirty.x0..][..dirty.width()]
                    .copy_from_slice(&row[..dirty.width()]);
            }
        }
        let base_index = self.levels.len() - 1;
        for k in 0..base_index {
            let (w, h) = (self.levels[k].w, self.levels[k].h);
            let (nw, nh) = (self.levels[k + 1].w, self.levels[k + 1].h);
            let coarse = reduce_footprint(dirty, nw, nh);
            self.save_gauss(&mut checkpoint, k + 1, coarse);
            self.reduce_region(k, coarse);
            let changed = dirty.union(expand_footprint(coarse, w, h));
            let pooled = if blur_active(w, h) {
                changed.grow(RADIUS, w, h)
            } else {
                changed
            };
            let tiles = pooled.align_tiles(w, h);
            if let Some(cp) = checkpoint.as_deref_mut() {
                let level = &self.levels[k];
                for ty in tiles.y0 / TILE..tiles.y1.div_ceil(TILE) {
                    for tx in tiles.x0 / TILE..tiles.x1.div_ceil(TILE) {
                        let t = ty * level.tiles_x + tx;
                        cp.tiles.push((k, t, level.tiles[t]));
                    }
                }
            }
            self.pool_region(reference, k, tiles);
            dirty = coarse;
        }
        self.base = self.score_base(reference);
        Ok(())
    }

    /// Recompute `out` of level `k + 1` from level `k` (the five-tap reduction).
    fn reduce_region(&mut self, k: usize, out: Rect) {
        if out.is_empty() {
            return;
        }
        let (fine, rest) = self.levels.split_at_mut(k + 1);
        let (src, dst) = (&fine[k], &mut rest[0]);
        let (w, h) = (src.w, src.h);
        let lo = (2 * out.x0).saturating_sub(2);
        let hi = (2 * out.x1 + 1).min(w);
        let tmp = &mut self.scratch.tmp;
        ensure_len(tmp, hi - lo);
        let tmp = &mut tmp[..hi - lo];
        let start = out.x0.max(1);
        let end = out.x1.min(((w - 1) / 2).max(1));
        for (s, d) in src.gauss.iter().zip(&mut dst.gauss) {
            for y in out.y0..out.y1 {
                let rows = std::array::from_fn(|i| {
                    let from = reflect((2 * y) as isize + i as isize - 2, h) * w;
                    &s[from + lo..from + hi]
                });
                self.pyramid.reduce_vertical(&rows, tmp);
                let row = &mut d[y * dst.w..(y + 1) * dst.w];
                let border = |x: usize| {
                    let xx = (2 * x) as isize;
                    let t = |i: isize| tmp[reflect(i, w) - lo];
                    0.05 * (t(xx - 2) + t(xx + 2)) + 0.25 * (t(xx - 1) + t(xx + 1)) + 0.4 * t(xx)
                };
                if out.x0 == 0 {
                    row[0] = border(0);
                }
                if start < end {
                    self.pyramid
                        .reduce_horizontal(&tmp[2 * start - 2 - lo..], &mut row[start..end]);
                }
                for (x, d) in row
                    .iter_mut()
                    .enumerate()
                    .take(out.x1)
                    .skip(out.x0.max(end))
                {
                    *d = border(x);
                }
            }
        }
    }

    /// Re-pool every tile of level `k` inside the tile-aligned `tiles`.
    fn pool_region(&mut self, reference: &PreparedReference, k: usize, tiles: Rect) {
        for y0 in (tiles.y0..tiles.y1).step_by(POOL_ROWS) {
            self.pool_strip(
                reference,
                k,
                Rect::new(tiles.x0, y0, tiles.x1, (y0 + POOL_ROWS).min(tiles.y1)),
            );
        }
    }

    fn pool_strip(&mut self, reference: &PreparedReference, k: usize, tiles: Rect) {
        if tiles.is_empty() {
            return;
        }
        let (w, h) = (self.levels[k].w, self.levels[k].h);
        let blur = blur_active(w, h);
        // Contrast and masks over the tiles plus the blur's reach.
        let area = if blur {
            tiles.grow(RADIUS, w, h)
        } else {
            tiles
        }
        .align_x(ALIGN, w);
        let (aw, ah) = (area.width(), area.height());
        let band = &reference.bands[k];
        let s = &mut self.scratch;
        for plane in s
            .expanded
            .iter_mut()
            .chain(&mut s.contrast)
            .chain(&mut s.mask)
        {
            ensure_len(plane, aw * ah);
        }
        expand_region(
            &self.levels[k + 1],
            w,
            h,
            area,
            &mut s.expanded,
            &mut s.tmp,
            &self.pyramid,
        );
        let level = &self.levels[k];
        let scale = if k == 0 { 1.0 } else { 2.0 };
        let src = area.y0 * w + area.x0;
        kernels::contrast_region(
            self.backend,
            kernels::ContrastRegion {
                current: level.gauss.each_ref().map(|v| &v[src..]),
                expanded: s.expanded.each_ref().map(|v| &v[..aw * ah]),
                sensitivity: band.sensitivity.each_ref().map(|v| &v[src..]),
                reference: band.contrast.each_ref().map(|v| &v[src..]),
                contrast: s.contrast.each_mut().map(|v| &mut v[..aw * ah]),
                mask: s.mask.each_mut().map(|v| &mut v[..aw * ah]),
                width: aw,
                height: ah,
                source_stride: w,
                scale,
            },
        );
        // Blurred masks over the tiles (exclusive reflection at image edges).
        let (tw, th) = (tiles.width(), tiles.height());
        for plane in &mut s.blurred {
            ensure_len(plane, tw * th);
        }
        // Expansion is dead after contrast conversion; reuse its storage for
        // the horizontal blur rather than retaining another three planes.
        let horizontal = &mut s.expanded;
        if blur {
            let rows = tiles.grow(RADIUS, w, h);
            // Every source row a vertical tap can read lies inside `rows`.
            let at = |x: usize, y: usize| (y - area.y0) * aw + x - area.x0;
            for ((mask, horizontal), blurred) in s.mask.iter().zip(horizontal).zip(&mut s.blurred) {
                ensure_len(horizontal, rows.height() * tw);
                let interior = tiles.x0 >= RADIUS && tiles.x1 + RADIUS <= w;
                for y in rows.y0..rows.y1 {
                    if !interior {
                        s.padded.clear();
                        for i in 0..tw + 2 * RADIUS {
                            let x = reflect_exclusive(
                                tiles.x0 as isize + i as isize - RADIUS as isize,
                                w,
                            );
                            s.padded.push(mask[at(x, y)]);
                        }
                    }
                    let dst = &mut horizontal[(y - rows.y0) * tw..][..tw];
                    if interior {
                        let from = at(tiles.x0 - RADIUS, y);
                        self.blur
                            .horizontal_row(&mask[from..from + tw + 2 * RADIUS], dst);
                    } else {
                        self.blur.horizontal_row(&s.padded, dst);
                    }
                }
                for y in tiles.y0..tiles.y1 {
                    let src: blur::BlurRows<'_> = std::array::from_fn(|i| {
                        let sy = reflect_exclusive(y as isize + i as isize - RADIUS as isize, h);
                        &horizontal[(sy - rows.y0) * tw..][..tw]
                    });
                    let dst = &mut blurred[(y - tiles.y0) * tw..][..tw];
                    self.blur.vertical_row(&src, dst);
                }
            }
        } else {
            for (mask, blurred) in s.mask.iter().zip(&mut s.blurred) {
                for y in tiles.y0..tiles.y1 {
                    let from = (y - area.y0) * aw + tiles.x0 - area.x0;
                    blurred[(y - tiles.y0) * tw..][..tw].copy_from_slice(&mask[from..from + tw]);
                }
            }
        }
        // Batch the strip with one validation/dispatch and no staging buffer.
        // Individual row reductions still start at the same tile boundary.
        let level = &mut self.levels[k];
        let c_at = (tiles.y0 - area.y0) * aw + tiles.x0 - area.x0;
        let r_at = tiles.y0 * w + tiles.x0;
        let t_at = tiles.y0 / TILE * level.tiles_x + tiles.x0 / TILE;
        kernels::pool_tiles(
            self.backend,
            &kernels::TiledPool {
                current: s.contrast.each_ref().map(|v| &v[c_at..]),
                reference: band.contrast.each_ref().map(|v| &v[r_at..]),
                blurred: s.blurred.each_ref().map(|v| v.as_slice()),
                strides: [aw, w, tw],
                width: tw,
                height: th,
                parameters: self.params,
            },
            &mut level.tiles[t_at..],
            level.tiles_x,
        );
        for ty in tiles.y0 / TILE..tiles.y1.div_ceil(TILE) {
            level.sum_tile_row(ty);
        }
    }

    fn score_base(&self, reference: &PreparedReference) -> [f64; 3] {
        let level = &self.levels[self.levels.len() - 1];
        let band = &reference.bands[self.levels.len() - 1];
        let count = level.w * level.h;
        let bg = model::mean_luminance(&level.gauss[0][..count]);
        let mut sums = [0.0_f64; 3];
        for (((sum, gauss), sensitivity), contrast) in sums
            .iter_mut()
            .zip(&level.gauss)
            .zip(&band.sensitivity)
            .zip(&band.contrast)
        {
            let sensitivity = sensitivity[0];
            for (&value, &reference) in gauss[..count].iter().zip(&contrast[..count]) {
                let d = ((value / bg).min(1000.0) * sensitivity - reference).abs() as f64;
                *sum += d * d + 2.0 * (EPSILON as f64) * d;
            }
        }
        sums
    }
}

/// The expanded coarse level over `area` of the `w`×`h` fine level.
fn expand_region(
    coarse: &Level,
    w: usize,
    h: usize,
    area: Rect,
    out: &mut [Vec<f32>; 3],
    tmp: &mut Vec<f32>,
    kernels: &pyramid::Kernels,
) {
    let (sw, sh) = (coarse.w, coarse.h);
    debug_assert_eq!((sw, sh), (w.div_ceil(2), h.div_ceil(2)));
    let cx0 = (area.x0 / 2).saturating_sub(1);
    let cx1 = ((area.x1 - 1) / 2 + 2).min(sw);
    let aw = area.width();
    ensure_len(tmp, cx1 - cx0);
    let tmp = &mut tmp[..cx1 - cx0];
    let start = (area.x0 / 2).max(1);
    let end = (area.x1 / 2).min(sw - 1);
    for (src, plane) in coarse.gauss.iter().zip(out.iter_mut()) {
        for y in area.y0..area.y1 {
            let sy = y / 2;
            let rows = [sy.saturating_sub(1), sy, (sy + 1).min(sh - 1)]
                .map(|r| &src[r * sw + cx0..r * sw + cx1]);
            if y.is_multiple_of(2) {
                kernels.expand_even(&rows, tmp);
            } else {
                kernels.expand_odd(&[rows[1], rows[2]], tmp);
            }
            let row = &mut plane[(y - area.y0) * aw..][..aw];
            let border = |x: usize| {
                let sx = x / 2;
                let next = (sx + 1).min(sw - 1);
                if x.is_multiple_of(2) {
                    0.1 * (tmp[sx.saturating_sub(1) - cx0] + tmp[next - cx0]) + 0.8 * tmp[sx - cx0]
                } else {
                    0.5 * (tmp[sx - cx0] + tmp[next - cx0])
                }
            };
            // area.x0 is aligned to ALIGN, so every interior span starts on
            // an even pixel. Only image edges need clamped coarse samples.
            if area.x0 == 0 {
                for (x, d) in row.iter_mut().enumerate().take(2) {
                    *d = border(x);
                }
            }
            if start < end {
                kernels.expand_horizontal(
                    &tmp[start - 1 - cx0..],
                    &mut row[2 * start - area.x0..2 * end - area.x0],
                );
            }
            let tail = (2 * end).max(area.x0);
            for (x, d) in (tail..area.x1).zip(&mut row[tail - area.x0..]) {
                *d = border(x);
            }
        }
    }
}
