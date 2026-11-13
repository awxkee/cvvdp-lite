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

use cvvdp_lite::{CvvdpLite, Display, Error, RgbImage};

mod fixtures {
    include!("data/reference_cases.rs");
}

fn pattern(w: usize, h: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(&[
                ((x * 3 + y * 5 + x * y % 17) % 200 + 20) as u8,
                ((x * 7 + y * 11) % 200 + 20) as u8,
                ((x * 13 + y * 2) % 200 + 20) as u8,
            ]);
        }
    }
    pixels
}

fn distort(reference: &[u8], w: usize, h: usize, kind: &str, amount: i32) -> Vec<u8> {
    let mut test = reference.to_vec();
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let value = reference[(y * w + x) * 3 + c] as i32;
                let value = match kind {
                    "identical" => value,
                    "noise" => value + (((x * 19 + y * 23) % 11) as i32 - 5) * amount,
                    "chroma" => {
                        value
                            + if c == 0 {
                                amount
                            } else if c == 1 {
                                -amount
                            } else {
                                0
                            }
                    }
                    "black" => 0,
                    "blur" => {
                        let mut sum = 0;
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                                let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                                sum += reference[(yy * w + xx) * 3 + c] as i32;
                            }
                        }
                        sum / 9
                    }
                    _ => panic!("unknown fixture distortion"),
                };
                test[(y * w + x) * 3 + c] = value.clamp(0, 255) as u8;
            }
        }
    }
    test
}

#[test]
fn agrees_with_pinned_original_still_model() {
    for &(w, h, kind, amount, fhd, expected) in fixtures::CASES {
        let reference = pattern(w, h);
        let test = distort(&reference, w, h, kind, amount);
        let metric = CvvdpLite::new(if fhd {
            Display::standard_fhd()
        } else {
            Display::standard_4k()
        })
        .unwrap();
        let score = metric
            .compare(
                RgbImage::srgb8(&reference, w, h, w * 3).unwrap(),
                RgbImage::srgb8(&test, w, h, w * 3).unwrap(),
            )
            .unwrap();
        assert!(
            (score.jod - expected).abs() < 0.0002,
            "{w}x{h} {kind} {amount} fhd={fhd}: {} != {expected}",
            score.jod
        );
    }
}

#[test]
fn one_shot_matches_prepared_comparison_bit_for_bit() {
    let metric = CvvdpLite::new(Display::standard_4k()).unwrap();
    for (w, h) in [
        (1, 1),
        (3, 17),
        (17, 3),
        (5, 17),
        (17, 5),
        (7, 7),
        (7, 8),
        (8, 9),
        (9, 8),
        (13, 9),
        (17, 15),
        (31, 16),
        (33, 17),
        (67, 65),
    ] {
        let reference = pattern(w, h);
        let candidate = distort(&reference, w, h, "noise", 11);
        let interleaved_stride = 3 * w + 5;
        let planar_stride = w + 3;
        let encodings = [reference, candidate].map(|pixels| {
            let mut bytes = vec![0; (h - 1) * interleaved_stride + 3 * w];
            let mut srgb = vec![f32::NAN; bytes.len()];
            let mut linear = std::array::from_fn(|_| vec![f32::NAN; (h - 1) * planar_stride + w]);
            for (y, row) in pixels.chunks_exact(3 * w).enumerate() {
                let start = y * interleaved_stride;
                bytes[start..start + 3 * w].copy_from_slice(row);
                for (x, rgb) in row.as_chunks::<3>().0.iter().enumerate() {
                    for c in 0..3 {
                        let value = f32::from(rgb[c]) / 255.0;
                        srgb[start + 3 * x + c] = value;
                        linear[c][y * planar_stride + x] = value;
                    }
                }
            }
            (bytes, srgb, linear)
        });
        let [references, candidates] = encodings.each_ref().map(|(bytes, srgb, linear)| {
            [
                RgbImage::srgb8(bytes, w, h, interleaved_stride).unwrap(),
                RgbImage::srgb_f32(srgb, w, h, interleaved_stride).unwrap(),
                RgbImage::linear_rgb(linear.each_ref().map(Vec::as_slice), w, h, planar_stride)
                    .unwrap(),
            ]
        });
        for (r, reference) in references.into_iter().enumerate() {
            let prepared = metric.prepare_reference(reference).unwrap();
            let mut evaluator = prepared.evaluator();
            for (c, candidate) in candidates.into_iter().enumerate() {
                let expected = evaluator.compare(candidate).unwrap();
                let actual = metric.compare(reference, candidate).unwrap();
                assert_eq!(
                    [actual.jod.to_bits(), actual.distortion.to_bits()],
                    [expected.jod.to_bits(), expected.distortion.to_bits()],
                    "{w}x{h}, reference format {r}, candidate format {c}"
                );
            }
        }
    }
}

#[test]
fn one_shot_rejects_bad_dimensions_and_nonfinite_samples() {
    let (w, h) = (17, 19);
    let metric = CvvdpLite::new(Display::default()).unwrap();
    let pixels = pattern(w, h);
    let good = RgbImage::srgb8(&pixels, w, h, w * 3).unwrap();
    let short = RgbImage::srgb8(&pixels, w, h - 1, w * 3).unwrap();
    let narrow = RgbImage::srgb8(&pixels, w - 1, h, w * 3).unwrap();

    let mut gamma = vec![0.5; w * h * 3];
    *gamma.last_mut().unwrap() = f32::NAN;
    let bad_gamma = RgbImage::srgb_f32(&gamma, w, h, w * 3).unwrap();
    let mut red = vec![0.5; w * h];
    let other = vec![0.5; w * h];
    red[w * (h - 1)] = f32::INFINITY;
    let bad_linear = RgbImage::linear_rgb([&red, &other, &other], w, h, w).unwrap();

    assert_eq!(metric.compare(good, short), Err(Error::DimensionMismatch));
    assert_eq!(metric.compare(good, narrow), Err(Error::DimensionMismatch));
    assert_eq!(
        metric.compare(bad_gamma, short),
        Err(Error::DimensionMismatch)
    );
    for bad in [bad_gamma, bad_linear] {
        assert_eq!(metric.compare(bad, good), Err(Error::NonFiniteSample));
        assert_eq!(metric.compare(good, bad), Err(Error::NonFiniteSample));
    }
}

#[test]
fn identity_and_finite_scores_for_tiny_and_thin_images() {
    let metric = CvvdpLite::new(Display::default()).unwrap();
    for (w, h) in [
        (1, 1),
        (1, 17),
        (17, 1),
        (2, 2),
        (3, 5),
        (7, 8),
        (8, 7),
        (63, 64),
        (64, 63),
    ] {
        let pixels = pattern(w, h);
        let image = RgbImage::srgb8(&pixels, w, h, w * 3).unwrap();
        let reference = metric.prepare_reference(image).unwrap();
        let mut evaluator = reference.evaluator();
        let score = evaluator.compare(image).unwrap();
        assert_eq!(score.jod, 10.0);
        assert_eq!(score.distortion, 0.0);
        assert_eq!(evaluator.band_scores().len(), reference.band_count());
        let changed = distort(&pixels, w, h, "chroma", 16);
        let changed = RgbImage::srgb8(&changed, w, h, w * 3).unwrap();
        let score = evaluator.compare(changed).unwrap();
        assert!(score.jod.is_finite() && score.jod < 10.0 && score.distortion > 0.0);
        assert_eq!(evaluator.compare(image).unwrap().jod, 10.0);
    }
}

#[test]
fn image_layout_and_recovery_after_bad_input() {
    assert!(matches!(
        RgbImage::srgb8(&[], 0, 1, 0),
        Err(Error::InvalidDimensions)
    ));
    assert!(matches!(
        RgbImage::srgb8(&[], usize::MAX, 2, usize::MAX),
        Err(Error::InvalidDimensions)
    ));
    assert!(matches!(
        RgbImage::srgb8(&[0; 12], 2, 2, 5),
        Err(Error::InvalidLayout)
    ));
    assert!(matches!(
        RgbImage::srgb8(&[0; 11], 2, 2, 6),
        Err(Error::InvalidLayout)
    ));
    let metric = CvvdpLite::new(Display::default()).unwrap();
    let reference = metric
        .prepare_reference(RgbImage::srgb8(&[0; 12], 2, 2, 6).unwrap())
        .unwrap();
    let mut evaluator = reference.evaluator();
    let bad = [f32::NAN; 12];
    assert_eq!(
        evaluator.compare(RgbImage::srgb_f32(&bad, 2, 2, 6).unwrap()),
        Err(Error::NonFiniteSample)
    );
    assert_eq!(
        evaluator
            .compare(RgbImage::srgb8(&[0; 12], 2, 2, 6).unwrap())
            .unwrap()
            .jod,
        10.0
    );
    assert_eq!(
        evaluator.compare(RgbImage::srgb8(&[0; 3], 1, 1, 3).unwrap()),
        Err(Error::DimensionMismatch)
    );
    // The final row need not include padding; padding samples are ignored.
    let padded = [0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0];
    assert_eq!(
        evaluator
            .compare(RgbImage::srgb8(&padded, 2, 2, 8).unwrap())
            .unwrap()
            .jod,
        10.0
    );
}

#[test]
fn linear_planes_match_gamma_encoded_input() {
    let (w, h) = (32, 32);
    let pixels = pattern(w, h);
    let mut planes: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
    for p in pixels.as_chunks::<3>().0 {
        for c in 0..3 {
            let v = p[c] as f32 / 255.0;
            planes[c].push(if v > 0.04045 {
                ((v + 0.055) / 1.055).powf(2.4)
            } else {
                v / 12.92
            });
        }
    }
    let metric = CvvdpLite::new(Display::default()).unwrap();
    let score = metric
        .compare(
            RgbImage::srgb8(&pixels, w, h, w * 3).unwrap(),
            RgbImage::linear_rgb(planes.each_ref().map(Vec::as_slice), w, h, w).unwrap(),
        )
        .unwrap();
    assert_eq!(score.jod, 10.0);
}

#[test]
fn right_edge_correction_preserves_transpose_symmetry() {
    let (w, h) = (63, 64);
    let reference = pattern(w, h);
    let test = distort(&reference, w, h, "blur", 1);
    let transpose = |pixels: &[u8]| {
        let mut out = vec![0; pixels.len()];
        for y in 0..h {
            for x in 0..w {
                out[(x * h + y) * 3..(x * h + y) * 3 + 3]
                    .copy_from_slice(&pixels[(y * w + x) * 3..(y * w + x) * 3 + 3]);
            }
        }
        out
    };
    let metric = CvvdpLite::new(Display::default()).unwrap();
    let before = metric
        .compare(
            RgbImage::srgb8(&reference, w, h, w * 3).unwrap(),
            RgbImage::srgb8(&test, w, h, w * 3).unwrap(),
        )
        .unwrap();
    let (reference, test) = (transpose(&reference), transpose(&test));
    let after = metric
        .compare(
            RgbImage::srgb8(&reference, h, w, h * 3).unwrap(),
            RgbImage::srgb8(&test, h, w, h * 3).unwrap(),
        )
        .unwrap();
    assert!((before.jod - after.jod).abs() < 0.00002);
}

#[test]
fn display_validation_and_reference_sharing() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<cvvdp_lite::PreparedReference>();
    for display in [
        Display {
            pixels_per_degree: 0.0,
            ..Display::default()
        },
        Display {
            peak_luminance: f32::INFINITY,
            ..Display::default()
        },
        Display {
            black_luminance: 200.0,
            ..Display::default()
        },
        Display {
            reflected_luminance: -1.0,
            ..Display::default()
        },
    ] {
        assert!(matches!(
            CvvdpLite::new(display),
            Err(Error::InvalidDisplay)
        ));
    }
}
