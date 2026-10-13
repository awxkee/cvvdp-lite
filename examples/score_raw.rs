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

//! Minimal byte-RGB harness for upstream differential tests and benchmarks.
use cvvdp_lite::{Backend, CvvdpLite, Display, RgbImage, Workspace};
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 5 || args.len() > 8 {
        return Err("usage: score_raw WIDTH HEIGHT REF.rgb TEST.rgb [ITERATIONS] [4k|fhd|macdisplay] [auto|scalar|neon|avx2|sse2]".into());
    }
    let w: usize = args[1].parse()?;
    let h: usize = args[2].parse()?;
    let reference = std::fs::read(&args[3])?;
    let test = std::fs::read(&args[4])?;
    let iterations: u32 = args.get(5).map(|v| v.parse()).transpose()?.unwrap_or(1);
    if iterations == 0 {
        return Err("iterations must be nonzero".into());
    }
    let display = match args.get(6).map(String::as_str).unwrap_or("4k") {
        "4k" => Display::standard_4k(),
        "fhd" => Display::standard_fhd(),
        "macdisplay" | "macbook_pro_16" => Display::macbook_pro_16(),
        _ => return Err("display must be 4k, fhd or macdisplay".into()),
    };
    let started = Instant::now();
    let reference =
        CvvdpLite::new(display)?.prepare_reference(RgbImage::srgb8(&reference, w, h, w * 3)?)?;
    let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
    let backend = match args.get(7).map(String::as_str).unwrap_or("auto") {
        "auto" => Backend::detect(),
        "scalar" => Backend::Scalar,
        "neon" => Backend::Neon,
        "avx2" => Backend::Avx2,
        "sse2" => Backend::Sse2,
        _ => return Err("backend must be auto, scalar, neon, avx2, or sse2".into()),
    };
    let reserve_started = Instant::now();
    let mut evaluator = reference.evaluator_with(Workspace::with_backend(backend)?);
    let reserve_ms = reserve_started.elapsed().as_secs_f64() * 1000.0;
    let test = RgbImage::srgb8(&test, w, h, w * 3)?;
    let first_started = Instant::now();
    let mut score = evaluator.compare(test)?;
    let first_ms = first_started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    for _ in 0..iterations {
        score = std::hint::black_box(evaluator.compare(std::hint::black_box(test))?);
    }
    let candidate_ms = started.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    println!(
        "{{\"jod\":{},\"distortion\":{},\"prepare_ms\":{},\"first_ms\":{},\"candidate_ms\":{},\"bands\":{},\"backend\":\"{:?}\",\"reserve_ms\":{},\"workspace_bytes\":{},\"reference_bytes\":{}}}",
        score.jod,
        score.distortion,
        prepare_ms,
        first_ms,
        candidate_ms,
        reference.band_count(),
        evaluator.backend(),
        reserve_ms,
        evaluator.allocated_bytes(),
        reference.allocated_bytes()
    );
    Ok(())
}
