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

"""Paired, decoded-pixel candidate benchmarks for cvvdp-lite.

Pillow is used only to prepare the inputs; timed Rust operations exclude file
I/O and image decoding. Baseline binaries may use the original score_raw API.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',required=True,type=Path)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--image',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--sizes',default='64x64,767x511,768x512,1920x1080,3840x2160')
    parser.add_argument('--iterations',type=int,default=7)
    parser.add_argument('--trials',type=int,default=3)
    args=parser.parse_args()
    if args.iterations<=0 or args.trials<=0:
        raise SystemExit('iterations and trials must be positive')
    from PIL import Image
    source=Image.open(args.image).convert('RGB')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='cvvdp-benchmark-') as directory:
        root=Path(directory)
        for dimensions in args.sizes.split(','):
            w,h=map(int,dimensions.split('x'))
            image=source.resize((w,h))
            pixels=image.tobytes()
            changed=bytes(min(255,max(0,value+(index*17)%21-10)) for index,value in enumerate(pixels))
            reference=root/'reference.rgb'
            test=root/'test.rgb'
            reference.write_bytes(pixels)
            test.write_bytes(changed)
            runs={'optimized':[],'scalar':[]}
            if args.baseline:
                runs['baseline']=[]
            for trial in range(args.trials):
                order=list(runs)
                if trial%2:
                    order.reverse()
                for variant in order:
                    binary=args.baseline if variant=='baseline' else args.binary
                    command=[str(binary.resolve()),str(w),str(h),str(reference),str(test),str(args.iterations),'4k']
                    if variant=='scalar':
                        command.append('scalar')
                    result=json.loads(subprocess.check_output(command,text=True))
                    runs[variant].append(result)
            medians={name:statistics.median(r['candidate_ms'] for r in trials) for name,trials in runs.items()}
            row=dict(width=w,height=h,median_candidate_ms=medians,trials=runs)
            if args.baseline:
                row['speedup']=medians['baseline']/medians['optimized']
                row['max_baseline_jod_error']=max(abs(r['jod']-runs['baseline'][0]['jod']) for r in runs['optimized'])
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='trials'}),flush=True)
    report=dict(platform=platform.platform(),architecture=platform.machine(),threads=1,
                input_image=args.image.name,iterations=args.iterations,trials=args.trials,
                optimized_binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),cases=rows)
    if args.baseline:
        report['baseline_binary_sha256']=hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
