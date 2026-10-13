#!/usr/bin/env python3
# Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1.  Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2.  Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3.  Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Generate the static, sustained-channel CSF data from ColorVideoVDP.

Usage: python3 tools/generate_tables.py /path/to/ColorVideoVDP
Generated sources retain the project license header and the upstream MIT
notice. This tool uses only Python's standard library; the Rust library does
not parse JSON.
"""
import argparse
import json
import math
from pathlib import Path
import struct


def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'src/tables.rs')
    args = parser.parse_args()
    lut = json.loads((args.upstream/'pycvvdp/vvdp_data/csf_lut_weber_fixed_size.json').read_text())
    assert len(lut['L_bkg']) == len(lut['rho']) == 32
    assert all(len(row) == 32 for ch in range(1, 4) for row in lut[f'o0_c{ch}'])
    project_header = (Path(__file__).resolve().parent/'license_header.txt').read_text().rstrip()
    upstream_license = (args.upstream/'LICENSE').read_text().rstrip()
    upstream_notice = '/*\n' + ''.join(' * '+line+'\n' if line else ' *\n' for line in upstream_license.splitlines()) + ' */'
    lines = [project_header, '', upstream_notice, '',
             '// Generated from ColorVideoVDP 0.5.7; see tools/generate_tables.py for attribution.',
             '#![allow(clippy::excessive_precision)]']
    for name, values in [('LOG_LUMINANCE', lut['L_bkg']), ('LOG_FREQUENCY', lut['rho'])]:
        lines.append(f'pub(crate) const {name}: [f32; 32] = ['+
                     ','.join(repr(f32(math.log10(v))) for v in values)+'];')
    lines.append('pub(crate) const LOG_SENSITIVITY: [[[f32; 32]; 32]; 3] = [')
    for channel in range(1, 4):
        lines.append('[')
        lines.extend('['+','.join(repr(f32(v)) for v in row)+'],' for row in lut[f'o0_c{channel}'])
        lines.append('],')
    lines.append('];')
    args.output.write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
