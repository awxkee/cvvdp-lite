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
use crate::{Backend, Error, RgbImage, Score};

const TILE: usize = 8;
const ALIGN: usize = 8;

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
            for c in 0..3 {
                sums[c] += tile[c];
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
    horizontal: [Vec<f32>; 3],
    blurred: [Vec<f32>; 3],
    padded: Vec<f32>,
}

/// What an update overwrote, for an exact [`IncrementalState::rollback`].
#[derive(Default)]
pub struct Checkpoint {
    gauss: Vec<(usize, Rect)>,
    gauss_data: Vec<f32>,
    tiles: Vec<(usize, usize, [f64; 3])>,
    base: [f64; 3],
}

/// Incrementally maintained score of one test image against a prepared reference.
pub struct IncrementalState {
    backend: Backend,
    conversion: RgbToDkl,
    blur: GaussianBlur,
    params: MaskParameters,
    width: usize,
    height: usize,
    levels: Vec<Level>,
    base: [f64; 3],
    scratch: Scratch,
}

impl IncrementalState {
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
            params: model::mask_parameters(),
            width: test.width,
            height: test.height,
            levels,
            base: [0.0; 3],
            scratch: Scratch::default(),
        };
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
                    for c in 0..3 {
                        sums[c] += row[c];
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
        for &(k, t, _) in &checkpoint.tiles {
            let level = &mut self.levels[k];
            let ty = t / level.tiles_x;
            level.sum_tile_row(ty);
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
            for c in 0..3 {
                level.gauss[c][y * level.w + dirty.x0..][..dirty.width()]
                    .copy_from_slice(&self.scratch.row[c][..dirty.width()]);
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
        tmp.resize(hi - lo, 0.0);
        for c in 0..3 {
            let s = &src.gauss[c];
            for y in out.y0..out.y1 {
                let rows: [usize; 5] =
                    std::array::from_fn(|i| reflect((2 * y) as isize + i as isize - 2, h) * w);
                for (xx, t) in (lo..hi).zip(tmp.iter_mut()) {
                    *t = 0.05 * (s[rows[0] + xx] + s[rows[4] + xx])
                        + 0.25 * (s[rows[1] + xx] + s[rows[3] + xx])
                        + 0.4 * s[rows[2] + xx];
                }
                let row = &mut dst.gauss[c][y * dst.w..(y + 1) * dst.w];
                for (x, d) in row.iter_mut().enumerate().take(out.x1).skip(out.x0) {
                    let xx = 2 * x;
                    if xx >= 2 && xx + 2 < w {
                        let t = &tmp[xx - 2 - lo..xx + 3 - lo];
                        *d = 0.05 * (t[0] + t[4]) + 0.25 * (t[1] + t[3]) + 0.4 * t[2];
                        continue;
                    }
                    let xx = xx as isize;
                    let t = |i: isize| tmp[reflect(i, w) - lo];
                    *d = 0.05 * (t(xx - 2) + t(xx + 2))
                        + 0.25 * (t(xx - 1) + t(xx + 1))
                        + 0.4 * t(xx);
                }
            }
        }
    }

    /// Re-pool every tile of level `k` inside the tile-aligned `tiles`.
    fn pool_region(&mut self, reference: &PreparedReference, k: usize, tiles: Rect) {
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
        for c in 0..3 {
            for v in [&mut s.expanded[c], &mut s.contrast[c], &mut s.mask[c]] {
                v.resize(aw * ah, 0.0);
            }
        }
        expand_region(&self.levels[k + 1], w, h, area, &mut s.expanded, &mut s.tmp);
        let level = &self.levels[k];
        for c in 0..3 {
            for y in area.y0..area.y1 {
                let at = (y - area.y0) * aw;
                s.contrast[c][at..at + aw]
                    .copy_from_slice(&level.gauss[c][y * w + area.x0..][..aw]);
            }
        }
        let scale = if k == 0 { 1.0 } else { 2.0 };
        for y in 0..ah {
            let span = y * aw..(y + 1) * aw;
            let src = (area.y0 + y) * w + area.x0;
            let [t0, t1, t2] = &mut s.contrast;
            let [m0, m1, m2] = &mut s.mask;
            kernels::contrast(
                self.backend,
                [
                    &mut t0[span.clone()],
                    &mut t1[span.clone()],
                    &mut t2[span.clone()],
                ],
                s.expanded.each_ref().map(|v| &v[span.clone()]),
                band.sensitivity.each_ref().map(|v| &v[src..src + aw]),
                band.contrast.each_ref().map(|v| &v[src..src + aw]),
                [
                    &mut m0[span.clone()],
                    &mut m1[span.clone()],
                    &mut m2[span.clone()],
                ],
                scale,
            );
        }
        // Blurred masks over the tiles (exclusive reflection at image edges).
        let (tw, th) = (tiles.width(), tiles.height());
        for plane in &mut s.blurred {
            plane.resize(tw * th, 0.0);
        }
        if blur {
            let rows = tiles.grow(RADIUS, w, h);
            // Every source row a vertical tap can read lies inside `rows`.
            let at = |x: usize, y: usize| (y - area.y0) * aw + x - area.x0;
            for c in 0..3 {
                s.horizontal[c].resize(rows.height() * tw, 0.0);
                let interior = tiles.x0 >= RADIUS && tiles.x1 + RADIUS <= w;
                for y in rows.y0..rows.y1 {
                    s.padded.clear();
                    if interior {
                        let from = at(tiles.x0 - RADIUS, y);
                        s.padded
                            .extend_from_slice(&s.mask[c][from..from + tw + 2 * RADIUS]);
                    } else {
                        for i in 0..tw + 2 * RADIUS {
                            let x = reflect_exclusive(
                                tiles.x0 as isize + i as isize - RADIUS as isize,
                                w,
                            );
                            s.padded.push(s.mask[c][at(x, y)]);
                        }
                    }
                    let dst = &mut s.horizontal[c][(y - rows.y0) * tw..][..tw];
                    self.blur.horizontal_row(&s.padded, dst);
                }
                for y in tiles.y0..tiles.y1 {
                    let src: blur::BlurRows<'_> = std::array::from_fn(|i| {
                        let sy = reflect_exclusive(y as isize + i as isize - RADIUS as isize, h);
                        &s.horizontal[c][(sy - rows.y0) * tw..][..tw]
                    });
                    let dst = &mut s.blurred[c][(y - tiles.y0) * tw..][..tw];
                    self.blur.vertical_row(&src, dst);
                }
            }
        } else {
            for c in 0..3 {
                for y in tiles.y0..tiles.y1 {
                    let from = (y - area.y0) * aw + tiles.x0 - area.x0;
                    s.blurred[c][(y - tiles.y0) * tw..][..tw]
                        .copy_from_slice(&s.mask[c][from..from + tw]);
                }
            }
        }
        // Pool each tile in row segments (aligned, so SIMD lanes never move).
        let level = &mut self.levels[k];
        for ty in tiles.y0 / TILE..tiles.y1.div_ceil(TILE) {
            for tx in tiles.x0 / TILE..tiles.x1.div_ceil(TILE) {
                let (x0, x1) = (tx * TILE, ((tx + 1) * TILE).min(w));
                let mut sums = [0.0f64; 3];
                for y in ty * TILE..((ty + 1) * TILE).min(h) {
                    let c_at = (y - area.y0) * aw + x0 - area.x0;
                    let b_at = (y - tiles.y0) * tw + x0 - tiles.x0;
                    let r_at = y * w + x0;
                    let n = x1 - x0;
                    let part = kernels::pool(
                        self.backend,
                        s.contrast.each_ref().map(|v| &v[c_at..c_at + n]),
                        band.contrast.each_ref().map(|v| &v[r_at..r_at + n]),
                        s.blurred.each_ref().map(|v| &v[b_at..b_at + n]),
                        self.params,
                    );
                    for c in 0..3 {
                        sums[c] += part[c];
                    }
                }
                level.tiles[ty * level.tiles_x + tx] = sums;
            }
            level.sum_tile_row(ty);
        }
    }

    fn score_base(&self, reference: &PreparedReference) -> [f64; 3] {
        let level = &self.levels[self.levels.len() - 1];
        let band = &reference.bands[self.levels.len() - 1];
        let count = level.w * level.h;
        let bg = model::mean_luminance(&level.gauss[0][..count]);
        let mut sums = [0.0_f64; 3];
        for (c, sum) in sums.iter_mut().enumerate() {
            let sensitivity = band.sensitivity[c][0];
            for i in 0..count {
                let d = ((level.gauss[c][i] / bg).min(1000.0) * sensitivity - band.contrast[c][i])
                    .abs() as f64;
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
) {
    let (sw, sh) = (coarse.w, coarse.h);
    debug_assert_eq!((sw, sh), (w.div_ceil(2), h.div_ceil(2)));
    let cx0 = (area.x0 / 2).saturating_sub(1);
    let cx1 = ((area.x1 - 1) / 2 + 2).min(sw);
    let aw = area.width();
    tmp.resize(sw, 0.0);
    for (src, plane) in coarse.gauss.iter().zip(out.iter_mut()) {
        for y in area.y0..area.y1 {
            let sy = y / 2;
            let before = sy.saturating_sub(1) * sw;
            let center = sy * sw;
            let after = (sy + 1).min(sh - 1) * sw;
            for cx in cx0..cx1 {
                tmp[cx] = if y % 2 == 0 {
                    0.1 * (src[before + cx] + src[after + cx]) + 0.8 * src[center + cx]
                } else {
                    0.5 * (src[center + cx] + src[after + cx])
                };
            }
            let row = &mut plane[(y - area.y0) * aw..][..aw];
            for (x, d) in (area.x0..area.x1).zip(row.iter_mut()) {
                let sx = x / 2;
                let next = (sx + 1).min(sw - 1);
                *d = if x % 2 == 0 {
                    0.1 * (tmp[sx.saturating_sub(1)] + tmp[next]) + 0.8 * tmp[sx]
                } else {
                    0.5 * (tmp[sx] + tmp[next])
                };
            }
        }
    }
}
