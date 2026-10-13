# cvvdp-lite

A dependency-free Rust implementation of the **still-image** ColorVideoVDP

## API

```rust
use cvvdp_lite::{CvvdpLite, Display, RgbImage};

# fn example(reference_rgb: &[u8], test_rgb: &[u8], width: usize, height: usize)
#     -> Result<(), cvvdp_lite::Error> {
let metric = CvvdpLite::new(Display::standard_4k())?;
let reference = metric.prepare_reference(
    RgbImage::srgb8(reference_rgb, width, height, width * 3)?
)?;
let mut evaluator = reference.evaluator();
let score = evaluator.compare(
    RgbImage::srgb8(test_rgb, width, height, width * 3)?
)?;
println!("{} JOD; {} distortion", score.jod, score.distortion);
let bands = evaluator.band_scores();
# let _ = bands;
# Ok(())
# }
```

## License

Licensed under either of

- BSD-3-Clause ([LICENSE.md](LICENSE.md))
- Apache License, Version 2.0 ([LICENSE-APACHE.md](LICENSE-APACHE.md))

at your option. The sustained CSF data in `src/tables.rs` is derived from
ColorVideoVDP and keeps its original MIT notice inline.
