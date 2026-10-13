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
    Backend, Checkpoint, CvvdpLite, Display, IncrementalState, Rect, RgbImage, Workspace,
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

const SIZES: [(usize, usize); 6] = [(8, 8), (13, 29), (64, 64), (67, 45), (131, 200), (300, 177)];

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
    for backend in [Backend::Scalar, Backend::detect()] {
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
                let rect = Rect::new(x0, y0, (x0 + 5 + step * 3).min(w), (y0 + 4 + step).min(h));
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
