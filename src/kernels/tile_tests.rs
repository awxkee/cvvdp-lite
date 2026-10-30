//! Batched pooling must preserve the original per-tile, per-row reductions.
use super::*;
use std::hint::black_box;
use std::time::Instant;

fn inputs(w: usize, h: usize, strides: [usize; 3]) -> [[Vec<f32>; 3]; 3] {
    std::array::from_fn(|source| {
        std::array::from_fn(|channel| {
            let stride = strides[source];
            let mut plane = vec![f32::NAN; (h - 1) * stride + w];
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x + channel * 23;
                    let value = (i * 31 % 1000) as f32 * 0.2 - 100.0;
                    plane[y * stride + x] = match source {
                        0 => value,
                        1 => {
                            value
                                + if i.is_multiple_of(7) {
                                    0.0
                                } else {
                                    (i % 19) as f32 * 0.01
                                }
                        }
                        _ => [0.0, 1e-6, 0.01, 1.0, 50.0, 1000.0][i % 6],
                    };
                }
            }
            plane
        })
    })
}

// The pre-batching call pattern, used as an independent exact-results oracle.
fn unbatched(backend: Backend, data: &TiledPool<'_>, tiles: &mut [[f64; 3]], tile_stride: usize) {
    for (ty, y0) in (0..data.height).step_by(POOL_TILE).enumerate() {
        for (tx, x0) in (0..data.width).step_by(POOL_TILE).enumerate() {
            let n = POOL_TILE.min(data.width - x0);
            let mut sums = [0.0; 3];
            for y in y0..(y0 + POOL_TILE).min(data.height) {
                let [c, r, b] = data.strides.map(|stride| y * stride + x0);
                let part = pool(
                    backend,
                    data.current.map(|v| &v[c..c + n]),
                    data.reference.map(|v| &v[r..r + n]),
                    data.blurred.map(|v| &v[b..b + n]),
                    data.parameters,
                );
                for (sum, value) in sums.iter_mut().zip(part) {
                    *sum += value;
                }
            }
            tiles[ty * tile_stride + tx] = sums;
        }
    }
}

#[test]
fn batches_match_individual_calls_exactly_for_strides_tails_and_partial_tiles() {
    for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
        if !backend.is_available() {
            continue;
        }
        for w in 1..=69 {
            for h in [1, 7, 8, 9, 17, 64] {
                for strides in [[w; 3], [w + 3, w + 13, w + 7]] {
                    let planes = inputs(w, h, strides);
                    let data = TiledPool {
                        current: planes[0].each_ref().map(Vec::as_slice),
                        reference: planes[1].each_ref().map(Vec::as_slice),
                        blurred: planes[2].each_ref().map(Vec::as_slice),
                        strides,
                        width: w,
                        height: h,
                        parameters: crate::model::mask_parameters(),
                    };
                    let tile_stride = w.div_ceil(POOL_TILE) + 3;
                    // Prefix, row padding and suffix must remain untouched.
                    let mut expected = vec![[12345.0; 3]; tile_stride * h.div_ceil(POOL_TILE) + 4];
                    let mut actual = expected.clone();
                    unbatched(backend, &data, &mut expected[2..], tile_stride);
                    pool_tiles(backend, &data, &mut actual[2..], tile_stride);
                    for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            actual.map(f64::to_bits),
                            expected.map(f64::to_bits),
                            "{backend:?} {w}x{h} strides {strides:?} output {i}"
                        );
                        assert!(actual.iter().all(|v| v.is_finite()));
                    }
                }
            }
        }
    }
}

#[test]
fn row_parts_summed_in_row_order_match_batches_exactly() {
    for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
        if !backend.is_available() {
            continue;
        }
        for w in 1..=69 {
            for h in [1, 7, 8, 9, 17] {
                let strides = [w + 3, w + 13, w + 7];
                let planes = inputs(w, h, strides);
                let data = TiledPool {
                    current: planes[0].each_ref().map(Vec::as_slice),
                    reference: planes[1].each_ref().map(Vec::as_slice),
                    blurred: planes[2].each_ref().map(Vec::as_slice),
                    strides,
                    width: w,
                    height: h,
                    parameters: crate::model::mask_parameters(),
                };
                let columns = w.div_ceil(POOL_TILE);
                let mut expected = vec![[0.0; 3]; columns * h.div_ceil(POOL_TILE)];
                pool_tiles(backend, &data, &mut expected, columns);
                // Row terms, computed over arbitrary tile-column ranges.
                let mut parts = vec![[[0.0; 3]; 16]; h];
                for (y, row) in parts.iter_mut().enumerate() {
                    let split = (y * 7) % columns.max(1);
                    for range in [0..split, split..columns] {
                        if range.is_empty() {
                            continue;
                        }
                        let x0 = range.start * POOL_TILE;
                        let x1 = (range.end * POOL_TILE).min(w);
                        let [c, r, b] = strides.map(|stride| y * stride + x0);
                        let one = TiledPool {
                            current: data.current.map(|v| &v[c..]),
                            reference: data.reference.map(|v| &v[r..]),
                            blurred: data.blurred.map(|v| &v[b..]),
                            strides,
                            width: x1 - x0,
                            height: 1,
                            parameters: data.parameters,
                        };
                        pool_row_parts(backend, &one, &mut row[range], 0);
                    }
                }
                for (t, expected) in expected.iter().enumerate() {
                    let (ty, tx) = (t / columns, t % columns);
                    let mut sums = [0.0f64; 3];
                    for row in &parts[ty * POOL_TILE..((ty + 1) * POOL_TILE).min(h)] {
                        for (sum, value) in sums.iter_mut().zip(row[tx]) {
                            *sum += value;
                        }
                    }
                    assert_eq!(
                        sums.map(f64::to_bits),
                        expected.map(f64::to_bits),
                        "{backend:?} {w}x{h} tile {t}"
                    );
                }
            }
        }
    }
}

#[test]
fn batches_validate_every_plane_and_output_before_simd() {
    let planes = inputs(17, 9, [17; 3]);
    let data = TiledPool {
        current: planes[0].each_ref().map(Vec::as_slice),
        reference: planes[1].each_ref().map(Vec::as_slice),
        blurred: planes[2].each_ref().map(Vec::as_slice),
        strides: [17; 3],
        width: 17,
        height: 9,
        parameters: crate::model::mask_parameters(),
    };
    for backend in [Backend::Scalar, Backend::Neon, Backend::Sse2, Backend::Avx2] {
        if !backend.is_available() {
            continue;
        }
        for group in 0..3 {
            for channel in 0..3 {
                let mut invalid = TiledPool { ..data };
                let source = match group {
                    0 => &mut invalid.current,
                    1 => &mut invalid.reference,
                    _ => &mut invalid.blurred,
                };
                source[channel] = &source[channel][..16];
                let mut tiles = [[-1.0; 3]; 6];
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    pool_tiles(backend, &invalid, &mut tiles, 3);
                }));
                assert!(result.is_err());
                assert_eq!(tiles, [[-1.0; 3]; 6]);
            }
        }
        let mut too_short = [[-1.0; 3]; 5];
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool_tiles(backend, &data, &mut too_short, 3);
            }))
            .is_err()
        );
    }
}

/// Run with `cargo test --release --lib tile_pooling_benchmark -- --ignored --nocapture`.
#[test]
#[ignore = "isolated pooling timing probe"]
fn tile_pooling_benchmark() {
    let backend = black_box(Backend::detect());
    for (w, h) in [(8, 8), (56, 56), (63, 64)] {
        let strides = [w + 16, 1920, w];
        let planes = inputs(w, h, strides);
        let data = TiledPool {
            current: planes[0].each_ref().map(Vec::as_slice),
            reference: planes[1].each_ref().map(Vec::as_slice),
            blurred: planes[2].each_ref().map(Vec::as_slice),
            strides,
            width: w,
            height: h,
            parameters: crate::model::mask_parameters(),
        };
        let stride = w.div_ceil(POOL_TILE);
        let mut tiles = vec![[0.0; 3]; stride * h.div_ceil(POOL_TILE)];
        let mut runs = [[0.0; 2]; 5];
        let n = 2000;
        for (trial, times) in runs.iter_mut().enumerate() {
            for variant in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                for _ in 0..n {
                    if variant == 0 {
                        unbatched(backend, black_box(&data), &mut tiles, stride);
                    } else {
                        pool_tiles(backend, black_box(&data), &mut tiles, stride);
                    }
                    black_box(&tiles);
                }
                times[variant] = start.elapsed().as_secs_f64() * 1e6 / n as f64;
            }
        }
        let mut individual = runs.map(|r| r[0]);
        let mut batched = runs.map(|r| r[1]);
        individual.sort_by(f64::total_cmp);
        batched.sort_by(f64::total_cmp);
        println!(
            "{backend:?} {w}x{h}: individual {:.3} us, batched {:.3} us, speedup {:.3}x",
            individual[2],
            batched[2],
            individual[2] / batched[2]
        );
    }
}
