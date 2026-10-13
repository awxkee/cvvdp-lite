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

// ColorVideoVDP 0.5.7, revision 0c1a0e093bd0f9b7c8f4e7e502f11a4d3fa7183c.
// CPU float32; gausspyr_reduce width parity corrected. See tools/validate.py.
pub const CASES: &[(usize, usize, &str, i32, bool, f64)] = &[
    (64, 64, "identical", 1, false, 10.0),
    (64, 64, "noise", 1, false, 9.999443054199219),
    (64, 64, "noise", 4, false, 9.928873062133789),
    (64, 64, "chroma", 8, false, 9.828695297241211),
    (64, 64, "blur", 1, false, 9.541414260864258),
    (64, 64, "black", 1, false, 1.94696044921875),
    (96, 64, "identical", 1, false, 10.0),
    (96, 64, "noise", 1, false, 9.99974536895752),
    (96, 64, "noise", 4, false, 9.93760871887207),
    (96, 64, "chroma", 8, false, 9.82845401763916),
    (96, 64, "blur", 1, false, 9.543047904968262),
    (96, 64, "black", 1, false, 2.0025992393493652),
    (63, 65, "identical", 1, false, 10.0),
    (63, 65, "noise", 1, false, 9.99914836883545),
    (63, 65, "noise", 4, false, 9.922324180603027),
    (63, 65, "chroma", 8, false, 9.824838638305664),
    (63, 65, "blur", 1, false, 9.54344654083252),
    (63, 65, "black", 1, false, 2.288393497467041),
    (63, 64, "identical", 1, false, 10.0),
    (63, 64, "noise", 1, false, 9.99925422668457),
    (63, 64, "noise", 4, false, 9.924541473388672),
    (63, 64, "chroma", 8, false, 9.825945854187012),
    (63, 64, "blur", 1, false, 9.541300773620605),
    (63, 64, "black", 1, false, 2.277629852294922),
    (64, 63, "identical", 1, false, 10.0),
    (64, 63, "noise", 1, false, 9.99940299987793),
    (64, 63, "noise", 4, false, 9.92788028717041),
    (64, 63, "chroma", 8, false, 9.825933456420898),
    (64, 63, "blur", 1, false, 9.541143417358398),
    (64, 63, "black", 1, false, 2.2749366760253906),
    (8, 8, "identical", 1, false, 10.0),
    (8, 8, "noise", 1, false, 9.976774215698242),
    (8, 8, "noise", 4, false, 9.389714241027832),
    (8, 8, "chroma", 8, false, 9.765958786010742),
    (8, 8, "blur", 1, false, 9.963919639587402),
    (8, 8, "black", 1, false, 5.13847541809082),
    (64, 64, "noise", 8, true, 7.981668472290039),
    (64, 64, "chroma", 8, true, 9.828696250915527),
    (64, 64, "blur", 8, true, 8.635372161865234),
];
