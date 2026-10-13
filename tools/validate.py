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

"""Differential validation against a local, pinned ColorVideoVDP checkout.

Requires numpy, torch, torchvision, and pycvvdp's dependencies. No Python
dependency is needed by the Rust crate. Run the release score_raw example first.
"""
import argparse
import ctypes
import inspect
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


def pattern(np, width, height):
    y, x = np.indices((height, width), dtype=np.int32)
    return np.stack(((x * 3 + y * 5 + x * y % 17) % 200 + 20,
                     (x * 7 + y * 11) % 200 + 20,
                     (x * 13 + y * 2) % 200 + 20), axis=-1).astype(np.uint8)


def distort(np, reference, kind, amount=1):
    values = reference.astype(np.int32)
    y, x = np.indices(reference.shape[:2], dtype=np.int32)
    if kind == 'noise':
        values += (((x * 19 + y * 23) % 11 - 5) * amount)[..., None]
    elif kind == 'chroma':
        values[..., 0] += amount
        values[..., 1] -= amount
    elif kind == 'blur':
        padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode='edge')
        values = sum(padded[dy:dy+values.shape[0], dx:dx+values.shape[1]]
                     for dy in range(3) for dx in range(3)) // 9
    elif kind == 'black':
        values[:] = 0
    elif kind == 'white':
        values[:] = 255
    elif kind != 'identical':
        raise ValueError(kind)
    return np.clip(values, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True, type=Path)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--rust-fixtures', type=Path)
    parser.add_argument('--fcvvdp-library', type=Path)
    parser.add_argument('--fcvvdp-source', type=Path, help='Record the baseline source revision')
    parser.add_argument('--image', type=Path, help='Optional natural-image performance case')
    parser.add_argument('--validation-images', nargs='*', type=Path, default=[])
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    import numpy as np
    import torch
    import pycvvdp
    from pycvvdp.lpyr_dec import lpyr_dec
    torch.set_num_threads(1)
    models = {name: pycvvdp.cvvdp(display_name='standard_'+name, device=torch.device('cpu'), quiet=True)
              for name in ('4k', 'fhd')}

    # Keep upstream execution untouched except when explicitly measuring the
    # width-parity correction. Record both outcomes so divergences are visible.
    original_reduce = lpyr_dec.gausspyr_reduce
    source = inspect.getsource(original_reduce)
    source = source[source.index('    def '):]
    import textwrap
    source = textwrap.dedent(source)
    old = "if (x.shape[-2] % 2)==1: # odd number of columns"
    if old not in source:
        raise RuntimeError('The pinned upstream parity defect has changed; review the harness')
    source = source.replace(old, "if (x.shape[-1] % 2)==1: # odd number of columns")
    namespace = dict(original_reduce.__globals__)
    exec(source, namespace)
    corrected_reduce = namespace['gausspyr_reduce']

    def reference_score(reference, test, display, corrected):
        lpyr_dec.gausspyr_reduce = corrected_reduce if corrected else original_reduce
        with torch.no_grad():
            score, _ = models[display].predict(test, reference, dim_order='HWC')
        return float(score)

    def rust_score(reference, test, display, iterations=1):
        with tempfile.TemporaryDirectory(prefix='cvvdp-lite-') as temporary:
            root = Path(temporary)
            (root/'ref.rgb').write_bytes(reference.tobytes())
            (root/'test.rgb').write_bytes(test.tobytes())
            return json.loads(subprocess.check_output([
                str(args.binary.resolve()), str(reference.shape[1]), str(reference.shape[0]),
                str(root/'ref.rgb'), str(root/'test.rgb'), str(iterations), display], text=True))

    fast = None
    if args.fcvvdp_library:
        class Image(ctypes.Structure):
            _fields_ = [('width', ctypes.c_int), ('height', ctypes.c_int), ('stride', ctypes.c_int),
                        ('data', ctypes.c_void_p), ('format', ctypes.c_int), ('colorspace', ctypes.c_int)]
        class Result(ctypes.Structure):
            _fields_ = [('jod', ctypes.c_double), ('quality', ctypes.c_double)]
        library = ctypes.CDLL(str(args.fcvvdp_library.resolve()))
        library.cvvdp_compare_images.argtypes = [ctypes.POINTER(Image), ctypes.POINTER(Image),
                                                ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(Result)]
        library.cvvdp_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                        ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        library.cvvdp_destroy.argtypes = [ctypes.c_void_p]
        library.cvvdp_reset.argtypes = [ctypes.c_void_p]
        library.cvvdp_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(Image), ctypes.POINTER(Image), ctypes.POINTER(Result)]
        def fast(reference, test, display):
            h, w, _ = reference.shape
            ref = Image(w, h, w*3, reference.ctypes.data, 1, 0)
            dst = Image(w, h, w*3, test.ctypes.data, 1, 0)
            result = Result()
            started = time.perf_counter()
            code = library.cvvdp_compare_images(ctypes.byref(ref), ctypes.byref(dst),
                                               1 if display == '4k' else 0, 1, None, ctypes.byref(result))
            if code:
                raise RuntimeError(f'fcvvdp returned {code}')
            return result.jod, (time.perf_counter()-started)*1000

        def fast_reused(reference, test, display):
            h, w, _ = reference.shape
            ref = Image(w, h, w*3, reference.ctypes.data, 1, 0)
            dst = Image(w, h, w*3, test.ctypes.data, 1, 0)
            result = Result()
            context = ctypes.c_void_p()
            code = library.cvvdp_create(w, h, 0.0, 1 if display == '4k' else 0, 1, None, ctypes.byref(context))
            if code:
                raise RuntimeError(f'fcvvdp context returned {code}')
            try:
                timings = []
                for _ in range(6):
                    if library.cvvdp_reset(context):
                        raise RuntimeError('fcvvdp reset failed')
                    start = time.perf_counter()
                    if library.cvvdp_process_frame(context, ctypes.byref(ref), ctypes.byref(dst), ctypes.byref(result)):
                        raise RuntimeError('fcvvdp process failed')
                    timings.append((time.perf_counter()-start)*1000)
                return statistics.median(timings[1:])
            finally:
                library.cvvdp_destroy(context)

    cases = []
    for width, height in [(64, 64), (96, 64), (63, 65), (63, 64), (64, 63), (8, 8)]:
        for kind, amount in [('identical', 1), ('noise', 1), ('noise', 4), ('chroma', 8), ('blur', 1), ('black', 1)]:
            cases.append((width, height, kind, amount, '4k'))
    cases += [(64, 64, kind, 8, 'fhd') for kind in ('noise', 'chroma', 'blur')]
    rows = []
    with torch.no_grad():
        for width, height, kind, amount, display in cases:
            reference = pattern(np, width, height)
            test = distort(np, reference, kind, amount)
            upstream = reference_score(reference, test, display, False)
            corrected = reference_score(reference, test, display, True)
            result = rust_score(reference, test, display)
            row = dict(width=width, height=height, kind=kind, amount=amount, display=display,
                       upstream_jod=upstream, corrected_jod=corrected, **result)
            if fast:
                row['fcvvdp_jod'], row['fcvvdp_ms'] = fast(reference, test, display)
            rows.append(row)
    if args.fixtures:
        args.fixtures.write_text(json.dumps(rows, indent=2)+'\n')
    if args.rust_fixtures:
        revision = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
        project_header = (Path(__file__).resolve().parent/'license_header.txt').read_text().rstrip()
        lines = [project_header, '', f'// ColorVideoVDP 0.5.7, revision {revision}.',
                 '// CPU float32; gausspyr_reduce width parity corrected. See tools/validate.py.',
                 'pub const CASES: &[(usize, usize, &str, i32, bool, f64)] = &[']
        for row in rows:
            lines.append(f'    ({row["width"]}, {row["height"]}, "{row["kind"]}", {row["amount"]}, '
                         f'{str(row["display"] == "fhd").lower()}, {row["corrected_jod"]}),')
        lines.append('];')
        args.rust_fixtures.write_text('\n'.join(lines)+'\n')
    for image_path in args.validation_images:
        from PIL import Image as PILImage
        reference = np.asarray(PILImage.open(image_path).convert('RGB')).copy()
        for kind, amount in [('noise', 2), ('chroma', 8), ('blur', 1)]:
            test = distort(np, reference, kind, amount)
            original = reference_score(reference, test, '4k', False)
            corrected = reference_score(reference, test, '4k', True)
            result = rust_score(reference, test, '4k')
            row = dict(image=image_path.name, width=reference.shape[1], height=reference.shape[0],
                       kind=kind, amount=amount, display='4k', upstream_jod=original, corrected_jod=corrected, **result)
            if fast:
                row['fcvvdp_jod'], row['fcvvdp_ms'] = fast(reference, test, '4k')
            rows.append(row)
    max_error = max(abs(row['jod']-row['corrected_jod']) for row in rows)
    report = dict(upstream_revision=subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
                  torch_version=torch.__version__, platform=platform.platform(), architecture=platform.machine(),
                  threads=1, max_corrected_error_jod=max_error, cases=rows)
    if args.fcvvdp_source:
        report['fcvvdp_revision'] = subprocess.check_output(['git', '-C', str(args.fcvvdp_source), 'rev-parse', 'HEAD'], text=True).strip()
    if args.image:
        from PIL import Image as PILImage
        reference = np.asarray(PILImage.open(args.image).convert('RGB')).copy()
        test = distort(np, reference, 'noise', 2)
        result = rust_score(reference, test, '4k', 5)
        times = []
        for _ in range(4):
            start = time.perf_counter()
            original = reference_score(reference, test, '4k', False)
            times.append((time.perf_counter()-start)*1000)
        benchmark = dict(image=args.image.name, width=reference.shape[1], height=reference.shape[0],
                         upstream_jod=original, upstream_ms=statistics.median(times[1:]), **result)
        if fast:
            timings = [fast(reference, test, '4k') for _ in range(4)]
            benchmark['fcvvdp_jod'] = timings[-1][0]
            benchmark['fcvvdp_ms'] = statistics.median(t[1] for t in timings[1:])
            benchmark['fcvvdp_reused_ms'] = fast_reused(reference, test, '4k')
        report['benchmark'] = benchmark
    lpyr_dec.gausspyr_reduce = original_reduce
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))
    if max_error > 0.0002:
        raise SystemExit(f'Differential error {max_error} exceeds 0.0002 JOD')


if __name__ == '__main__':
    main()
