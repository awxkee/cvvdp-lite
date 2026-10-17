//! Incremental-state agreement and timing probe: raw linear planes from a PNG-free synthetic image.
//! Run `cargo run --release --example incremental_probe -- 2000 1920x1080`;
//! omit the dimensions to probe all standard sizes.
use cvvdp_lite::{Checkpoint, CvvdpLite, Display, IncrementalState, Rect, RgbImage, Workspace};
use std::hint::black_box;
use std::time::Instant;

fn planes(w: usize, h: usize, seed: f32) -> [Vec<f32>; 3] {
    std::array::from_fn(|c| {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                (0.5 + 0.3 * ((x * 0.031 + seed + c as f32).sin() * (y * 0.027).cos())
                    + 0.05 * ((x * 1.7 + y * 2.3 + seed).sin()))
                .clamp(0.0, 1.0)
            })
            .collect()
    })
}

fn img(p: &[Vec<f32>; 3], w: usize, h: usize) -> RgbImage<'_> {
    RgbImage::linear_rgb([&p[0], &p[1], &p[2]], w, h, w).unwrap()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n = args.next().map(|s| s.parse().unwrap()).unwrap_or(200);
    assert!(n > 0, "iterations must be positive");
    let dimensions = args.next().map(|s| {
        let (w, h) = s.split_once('x').expect("expected WIDTHxHEIGHT");
        (w.parse().unwrap(), h.parse().unwrap())
    });
    let sizes = dimensions.as_ref().map(std::slice::from_ref).unwrap_or(&[
        (64, 64),
        (300, 177),
        (768, 512),
        (1024, 1024),
        (1920, 1080),
    ]);
    for &(w, h) in sizes {
        let reference = planes(w, h, 0.0);
        let mut test = planes(w, h, 0.4);
        let metric = CvvdpLite::new(Display::standard_4k()).unwrap();
        let prepared = metric.prepare_reference(img(&reference, w, h)).unwrap();
        let t = Instant::now();
        let full = Workspace::new()
            .compare(&prepared, img(&test, w, h))
            .unwrap();
        let full_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mut state = IncrementalState::new(&prepared, img(&test, w, h)).unwrap();
        let build_ms = t.elapsed().as_secs_f64() * 1e3;
        let inc = state.score();
        let mut trace = inc.distortion.to_bits() ^ inc.jod.to_bits();
        // 32x32 updates at scattered positions.
        let t = Instant::now();
        for i in 0..n {
            let x0 = (i * 97) % w.saturating_sub(32).max(1);
            let y0 = (i * 61) % h.saturating_sub(32).max(1);
            let r = Rect::new(x0, y0, (x0 + 32).min(w), (y0 + 32).min(h));
            for plane in &mut test {
                for y in r.y0..r.y1 {
                    for v in &mut plane[y * w + r.x0..y * w + r.x1] {
                        *v = (*v * 0.98 + 0.01).min(1.0);
                    }
                }
            }
            let score = state.update(&prepared, img(&test, w, h), r).unwrap();
            trace = trace.rotate_left(7) ^ score.distortion.to_bits() ^ score.jod.to_bits();
            black_box(score);
        }
        let upd_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
        let r = Rect::new(w / 3, h / 3, (w / 3 + 32).min(w), (h / 3 + 32).min(h));
        let mut checkpoint = Checkpoint::default();
        state
            .update_checkpointed(&prepared, img(&test, w, h), r, &mut checkpoint)
            .unwrap();
        state.rollback(&checkpoint);
        let t = Instant::now();
        for _ in 0..n {
            let score = state
                .update_checkpointed(&prepared, img(&test, w, h), r, &mut checkpoint)
                .unwrap();
            trace = trace.rotate_left(7) ^ score.distortion.to_bits() ^ score.jod.to_bits();
            black_box(score);
            state.rollback(&checkpoint);
        }
        let trial_ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
        println!(
            "{w}x{h}: bands {} full {:.6} inc {:.6} rel {:.2e} trace {trace:016x} | full {:.2} ms build {:.2} ms update(32x32) {:.3} ms trial+rollback {:.3} ms | state {} bytes checkpoint {} bytes",
            prepared.band_count(),
            full.distortion,
            inc.distortion,
            (inc.distortion - full.distortion).abs() / full.distortion,
            full_ms,
            build_ms,
            upd_ms,
            trial_ms,
            state.allocated_bytes(),
            checkpoint.allocated_bytes(),
        );
    }
}
