#!/usr/bin/env python3
"""The reader gate of the resolution pair (Claude 1, 2026-11-23; Codex 2's B4): the output that four
ranks write with CUDA and MPI, from the smoke run of the build job, read by the readers of the
checks before any run of the pair.

    reader_gate.py <bp2raw> <cache> <smoke run directory>

Items (all needed):
  1. every field output: the field arrays have the shape (cells in x1, cells in x2) of the input;
  2. every particle output of the ions: ions of the counters (interpolated) / ions written equals the
     stride of the input within 1 percent;
  3. every spectra output: the ions of the two half-box spectra sum to the ions of the counters
     interpolated to that time within 1e-3;
  4. the core series (core_series.py) runs to the end on the output.
Exit 0 pass, 1 fail.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bhio import Run  # noqa: E402


def main():
    bp2raw, cache, path = sys.argv[1:4]
    run = Run(path, bp2raw, cache)
    nx, ny = (int(v) for v in re.search(r"resolution\s*=\s*\[\s*(\d+)\s*,\s*(\d+)", run.input).groups())
    stride = int(re.search(r"^\s*stride\s*=\s*(\d+)", run.input, re.M).group(1))
    c = run.counters()
    ok = True
    print(f"reader gate: {run.name}, {nx} x {ny} cells, particle stride {stride}")
    for bp, step, time in run.stamps("fields"):
        v = run.fields(bp, ["fN_1", "fN_2", "fB1"])
        good = all(v[k].shape == (nx, ny) for k in ("fN_1", "fN_2", "fB1")) and v["X1"].size == nx and v["X2"].size == ny
        ok &= good
        print(f"  1. fields of step {step}: shape {v['fN_1'].shape}, X1 {v['X1'].size}, X2 {v['X2'].size}  {'pass' if good else 'FAIL'}")
    for bp, step, time in run.stamps("particles"):
        written = np.atleast_1d(run.read(bp, ["pU1_2"])["pU1_2"]).size
        ions = float(np.interp(time, c["time"], c["n_2"]))
        f = ions / written if written else float("inf")
        good = abs(f / stride - 1.0) < 0.01
        ok &= good
        print(f"  2. particles of step {step}: ions {ions:.0f}, written {written}, ratio {f:.4f}  {'pass' if good else 'FAIL'}")
    for bp, step, time in run.stamps("spectra"):
        n = float(run.read(bp, ["sN_2"])["sN_2"].astype(np.float64).sum())
        ions = float(np.interp(time, c["time"], c["n_2"]))
        good = abs(n / ions - 1.0) < 1e-3
        ok &= good
        print(f"  3. spectra of step {step}: ions in the spectra {n:.0f}, counters {ions:.0f}  {'pass' if good else 'FAIL'}")
    out = Path(cache) / "reader-gate-core"
    r = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve().parent / "core_series.py"), bp2raw, cache,
                        path, str(out)], capture_output=True, text=True)
    good = r.returncode == 0
    ok &= good
    print(f"  4. core series: exit {r.returncode}; {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]}"
          f"  {'pass' if good else 'FAIL'}")
    print(f"reader gate verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
