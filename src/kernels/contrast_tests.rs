//! Compare direct strided contrast conversion with the original copy + row calls.
use super::*;
use std::hint::black_box;
use std::time::Instant;

type Inputs = [[Vec<f32>; 3]; 4];
const PREFIX: usize = 3;

fn inputs(w: usize, h: usize, stride: usize) -> Inputs {
    std::array::from_fn(|group| {
        std::array::from_fn(|channel| {
            let stride = if group == 1 { w } else { stride };
            let mut plane = vec![f32::NAN; PREFIX + (h - 1) * stride + w + 3];
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x + channel * 23;
                    plane[PREFIX + y * stride + x] = match group {
                        0 => (i as f32 % 11.0 - 5.0) * [0.0, 1e-6, 0.01, 1.0, 1e4][i % 5],
                        1 if channel == 0 => {
                            [-0.0, 1e-4, 0.00999, 0.01, 0.01001, 1.0, 100.0][i % 7]
                        }
                        1 => (i as f32 % 13.0 - 6.0) * 0.25,
                        2 => [0.0, 0.01, 1.0, 10.0, 100.0][i % 5],
                        _ => (i as f32 % 17.0 - 8.0) * 17.0,
                    };
                }
            }
            plane
        })
    })
}

fn region<'a>(
    planes: &'a Inputs,
    [width, height, source_stride]: [usize; 3],
    scale: f32,
    contrast: &'a mut [Vec<f32>; 3],
    mask: &'a mut [Vec<f32>; 3],
) -> ContrastRegion<'a> {
    let count = width * height;
    ContrastRegion {
        current: planes[0].each_ref().map(|v| &v[PREFIX..]),
        expanded: planes[1].each_ref().map(|v| &v[PREFIX..PREFIX + count]),
        sensitivity: planes[2].each_ref().map(|v| &v[PREFIX..]),
        reference: planes[3].each_ref().map(|v| &v[PREFIX..]),
        contrast: contrast.each_mut().map(|v| &mut v[1..1 + count]),
        mask: mask.each_mut().map(|v| &mut v[1..1 + count]),
        width,
        height,
        source_stride,
        scale,
    }
}

fn rowwise(
    backend: Backend,
    planes: &Inputs,
    [w, h, stride]: [usize; 3],
    scale: f32,
    output: &mut [Vec<f32>; 3],
    masks: &mut [Vec<f32>; 3],
) {
    for (out, src) in output.iter_mut().zip(&planes[0]) {
        for y in 0..h {
            out[1 + y * w..][..w].copy_from_slice(&src[PREFIX + y * stride..][..w]);
        }
    }
    for y in 0..h {
        let dst = 1 + y * w;
        let expanded = PREFIX + y * w;
        let src = PREFIX + y * stride;
        contrast(
            backend,
            output.each_mut().map(|v| &mut v[dst..dst + w]),
            planes[1].each_ref().map(|v| &v[expanded..expanded + w]),
            planes[2].each_ref().map(|v| &v[src..src + w]),
            planes[3].each_ref().map(|v| &v[src..src + w]),
            masks.each_mut().map(|v| &mut v[dst..dst + w]),
            scale,
        );
    }
}

#[test]
fn fused_strips_match_copied_rows_bit_exactly_for_strides_tails_and_clamps() {
    for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
        if !backend.is_available() {
            continue;
        }
        for w in 1..=69 {
            for h in [1, 7, 8, 9, 17, 76] {
                for stride in [w, w + 13] {
                    let planes = inputs(w, h, stride);
                    for scale in [1.0, 2.0] {
                        let mut expected: [Vec<f32>; 3] =
                            std::array::from_fn(|_| vec![-12345.0; w * h + 2]);
                        let mut expected_masks = expected.clone();
                        let mut actual = expected.clone();
                        let mut masks = expected.clone();
                        rowwise(
                            backend,
                            &planes,
                            [w, h, stride],
                            scale,
                            &mut expected,
                            &mut expected_masks,
                        );
                        contrast_region(
                            backend,
                            region(&planes, [w, h, stride], scale, &mut actual, &mut masks),
                        );
                        for (channel, ((actual, expected), (mask, expected_mask))) in actual
                            .iter()
                            .zip(&expected)
                            .zip(masks.iter().zip(&expected_masks))
                            .enumerate()
                        {
                            for (i, ((a, e), (m, em))) in actual
                                .iter()
                                .zip(expected)
                                .zip(mask.iter().zip(expected_mask))
                                .enumerate()
                            {
                                assert_eq!(
                                    (a.to_bits(), m.to_bits()),
                                    (e.to_bits(), em.to_bits()),
                                    "{backend:?} {w}x{h} stride {stride} scale {scale} channel {channel} sample {i}"
                                );
                                assert!(a.is_finite() && m.is_finite());
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn fused_strips_validate_all_planes_before_simd() {
    let (w, h, stride) = (17, 9, 32);
    let planes = inputs(w, h, stride);
    for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
        if !backend.is_available() {
            continue;
        }
        for group in 0..6 {
            for channel in 0..3 {
                let mut output = std::array::from_fn(|_| vec![-12345.0; w * h + 2]);
                let mut masks = output.clone();
                let mut data = region(&planes, [w, h, stride], 1.0, &mut output, &mut masks);
                match group {
                    0 => data.current[channel] = &[],
                    1 => data.expanded[channel] = &[],
                    2 => data.sensitivity[channel] = &[],
                    3 => data.reference[channel] = &[],
                    4 => data.contrast[channel] = &mut [],
                    _ => data.mask[channel] = &mut [],
                }
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    contrast_region(backend, data)
                }));
                assert!(result.is_err());
                assert!(
                    output
                        .iter()
                        .chain(&masks)
                        .flatten()
                        .all(|&v| v == -12345.0)
                );
            }
        }
    }
}

/// Run `cargo test --release --lib contrast_region_benchmark -- --ignored --nocapture`.
#[test]
#[ignore = "isolated contrast timing probe"]
fn contrast_region_benchmark() {
    for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
        if !backend.is_available() {
            continue;
        }
        let backend = black_box(backend);
        for (w, h) in [(8, 8), (72, 76), (79, 76)] {
            let dims = [w, h, 1920];
            let planes = inputs(w, h, dims[2]);
            let mut output = std::array::from_fn(|_| vec![0.0; w * h + 2]);
            let mut masks = output.clone();
            let mut runs = [[0.0; 2]; 5];
            let n = 5000;
            for (trial, times) in runs.iter_mut().enumerate() {
                for variant in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let start = Instant::now();
                    for _ in 0..n {
                        if variant == 0 {
                            rowwise(
                                backend,
                                black_box(&planes),
                                black_box(dims),
                                2.0,
                                &mut output,
                                &mut masks,
                            );
                        } else {
                            contrast_region(
                                backend,
                                region(
                                    black_box(&planes),
                                    black_box(dims),
                                    2.0,
                                    &mut output,
                                    &mut masks,
                                ),
                            );
                        }
                        black_box((&output, &masks));
                    }
                    times[variant] = start.elapsed().as_secs_f64() * 1e6 / n as f64;
                }
            }
            let mut rowwise = runs.map(|r| r[0]);
            let mut fused = runs.map(|r| r[1]);
            rowwise.sort_by(f64::total_cmp);
            fused.sort_by(f64::total_cmp);
            println!(
                "{backend:?} {w}x{h}: copied rows {:.3} us, fused strip {:.3} us, speedup {:.3}x",
                rowwise[2],
                fused[2],
                rowwise[2] / fused[2]
            );
        }
    }
}
