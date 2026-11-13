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

use cvvdp_lite::{
    Backend, Checkpoint, CvvdpLite, Display, Error, IncrementalState, Rect, RgbImage, Workspace,
};

fn planes(w: usize, h: usize, seed: u32) -> [Vec<f32>; 3] {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    let mut noise = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state % 1000) as f32 / 1000.0
    };
    std::array::from_fn(|c| {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                let smooth = 0.5 + 0.35 * ((x * 0.07 + c as f32).sin() * (y * 0.05).cos());
                (smooth + 0.08 * (noise() - 0.5)).clamp(0.0, 1.0)
            })
            .collect()
    })
}

fn image(p: &[Vec<f32>; 3], w: usize, h: usize) -> RgbImage<'_> {
    RgbImage::linear_rgb([&p[0], &p[1], &p[2]], w, h, w).unwrap()
}

fn displays() -> [Display; 2] {
    [
        Display::standard_4k(),
        Display {
            pixels_per_degree: 98.79,
            peak_luminance: 500.0,
            black_luminance: 0.37,
            reflected_luminance: 0.4,
        },
    ]
}

const SIZES: [(usize, usize); 13] = [
    (1, 1),
    (1, 17),
    (17, 1),
    (2, 65),
    (7, 129),
    (129, 7),
    (8, 8),
    (13, 29),
    (64, 64),
    (67, 45),
    (131, 200),
    (300, 177),
    (257, 193),
];

#[test]
fn fresh_state_matches_full_comparison() {
    for display in displays() {
        let metric = CvvdpLite::new(display).unwrap();
        for (w, h) in SIZES {
            let reference = planes(w, h, 1);
            let test = planes(w, h, 2);
            let prepared = metric.prepare_reference(image(&reference, w, h)).unwrap();
            let full = Workspace::new()
                .compare(&prepared, image(&test, w, h))
                .unwrap();
            let state = IncrementalState::new(&prepared, image(&test, w, h)).unwrap();
            let inc = state.score();
            let rel = (inc.distortion - full.distortion).abs() / full.distortion.max(1e-12);
            assert!(
                rel < 1e-5,
                "{w}x{h}: {} vs {}",
                inc.distortion,
                full.distortion
            );
        }
    }
}

#[test]
fn updates_reach_the_fresh_state_exactly() {
    for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
        if !backend.is_available() {
            continue;
        }
        let metric = CvvdpLite::new(Display::standard_4k()).unwrap();
        for (w, h) in SIZES {
            let reference = planes(w, h, 3);
            let mut test = planes(w, h, 4);
            let prepared = metric.prepare_reference(image(&reference, w, h)).unwrap();
            let mut state =
                IncrementalState::with_backend(&prepared, image(&test, w, h), backend).unwrap();
            let mut checkpoint = Checkpoint::default();
            for step in 0..12usize {
                let x0 = (step * 37) % w;
                let y0 = (step * 23) % h;
                let rect = match step {
                    // Cross multiple scratch strips and exercise every image edge.
                    0 => Rect::new(0, 0, w, h),
                    1 => Rect::new(0, 0, w.min(9), h.min(9)),
                    2 => Rect::new(w.saturating_sub(9), h.saturating_sub(9), w, h),
                    3 => Rect::new(0, h / 3, w, (h / 3 + 81).min(h)),
                    _ => Rect::new(x0, y0, (x0 + 5 + step * 3).min(w), (y0 + 4 + step).min(h)),
                };
                let saved = test.clone();
                for (c, plane) in test.iter_mut().enumerate() {
                    let delta = 0.3 * ((step + c) as f32 * 0.7).sin();
                    for y in rect.y0..rect.y1 {
                        for v in &mut plane[y * w + rect.x0..y * w + rect.x1] {
                            *v = (*v + delta).clamp(0.0, 1.0);
                        }
                    }
                }
                let before = state.score();
                let after = state
                    .update_checkpointed(&prepared, image(&test, w, h), rect, &mut checkpoint)
                    .unwrap();
                let fresh = IncrementalState::with_backend(&prepared, image(&test, w, h), backend)
                    .unwrap()
                    .score();
                assert_eq!(
                    after.distortion.to_bits(),
                    fresh.distortion.to_bits(),
                    "{backend:?} {w}x{h} step {step}"
                );
                if step % 3 == 2 {
                    state.rollback(&checkpoint);
                    test = saved;
                    assert_eq!(
                        state.score().distortion.to_bits(),
                        before.distortion.to_bits()
                    );
                }
            }
        }
    }
}

#[test]
fn compact_reference_matches_dense_for_incremental_updates_and_rollback() {
    let metric = CvvdpLite::new(Display::standard_4k()).unwrap();
    let score_bits = |score: cvvdp_lite::Score| [score.jod.to_bits(), score.distortion.to_bits()];
    for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
        if !backend.is_available() {
            continue;
        }
        for (w, h) in [(1, 1), (7, 129), (67, 65)] {
            let reference = planes(w, h, 1);
            let reference = image(&reference, w, h);
            let dense = metric.prepare_reference(reference).unwrap();
            let compact = metric.prepare_reference_compact(reference).unwrap();
            let original = planes(w, h, 2);
            let initial = image(&original, w, h);
            let mut dense_state = IncrementalState::with_backend(&dense, initial, backend).unwrap();
            let mut compact_state =
                IncrementalState::with_backend(&compact, initial, backend).unwrap();
            let initial_bits = score_bits(dense_state.score());
            assert_eq!(score_bits(compact_state.score()), initial_bits);

            let mut dense_checkpoint = Checkpoint::default();
            let mut compact_checkpoint = Checkpoint::default();
            for rect in [
                Rect::new(0, 0, w, h),
                Rect::new(w / 2, h / 2, w / 2 + 1, h / 2 + 1),
                Rect::new(w - 1, h - 1, w, h),
            ] {
                let mut pixels = original.clone();
                for y in rect.y0..rect.y1 {
                    pixels[0][y * w + rect.x0..y * w + rect.x1].fill(0.0);
                }
                let candidate = image(&pixels, w, h);
                let expected = IncrementalState::with_backend(&dense, candidate, backend)
                    .unwrap()
                    .score();
                // Both states can switch between representations of the same reference.
                for (state, reference, checkpoint) in [
                    (&mut dense_state, &compact, &mut dense_checkpoint),
                    (&mut compact_state, &dense, &mut compact_checkpoint),
                ] {
                    let actual = state
                        .update_checkpointed(reference, candidate, rect, checkpoint)
                        .unwrap();
                    assert_eq!(
                        score_bits(actual),
                        score_bits(expected),
                        "{backend:?} {w}x{h} {rect:?}"
                    );
                    state.rollback(checkpoint);
                    assert_eq!(score_bits(state.score()), initial_bits);
                }
            }
        }
    }
}

#[test]
fn checkpoint_recovers_from_failed_conversion_and_clears_on_empty_update() {
    let (w, h) = (65, 49);
    let pixels = planes(w, h, 5);
    let initial = image(&pixels, w, h);
    let prepared = CvvdpLite::new(Display::default())
        .unwrap()
        .prepare_reference(initial)
        .unwrap();
    let mut state = IncrementalState::new(&prepared, initial).unwrap();
    let mut checkpoint = Checkpoint::default();

    let mut invalid = pixels.clone();
    invalid[0][8 * w + 8] = 0.0;
    invalid[0][9 * w + 8] = f32::NAN;
    let failed = state.update_checkpointed(
        &prepared,
        image(&invalid, w, h),
        Rect::new(8, 8, 9, 10),
        &mut checkpoint,
    );
    assert_eq!(failed, Err(Error::NonFiniteSample));
    state.rollback(&checkpoint);

    // The nearby edit reads the Gaussian rows touched by the failed update.
    let mut changed = pixels.clone();
    changed[0][10 * w + 8] = 0.0;
    let candidate = image(&changed, w, h);
    let score = state
        .update_checkpointed(
            &prepared,
            candidate,
            Rect::new(8, 10, 9, 11),
            &mut checkpoint,
        )
        .unwrap();
    let fresh = IncrementalState::new(&prepared, candidate).unwrap().score();
    assert_eq!(score.distortion.to_bits(), fresh.distortion.to_bits());
    assert!(score.distortion > 0.0);

    let empty = Rect::new(w, h, w, h);
    state
        .update_checkpointed(&prepared, candidate, empty, &mut checkpoint)
        .unwrap();
    state.rollback(&checkpoint);
    assert_eq!(
        state.score().distortion.to_bits(),
        score.distortion.to_bits()
    );
}
