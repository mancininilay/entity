#!/usr/bin/env python3
"""Check 4 of the resolution pair: the Gauss-law residual of a run against the policy-off runs of
stages B and C on the same grid (Claude 1, 2026-11-23; Codex 2's B3; standard library).

    check4_ref.py <run directory> <reference run directory> [<reference run directory> ...]

The authorised jobs of the pair contain no run without the policy. The reference is therefore the
three runs without the policy that exist on this grid (R-off of stage B, 7520718; R-off-b and
R-off-c of stage C, 7523492 and 7523494): one card, the stock injector, the same window, kappa and
filter. Normalisation: U = kappa (C(2n, n) / 4^n)^2, one filtered unit charge. Reference level:
the largest, over the reference runs, of the largest residual over their samples (and the same for
the root mean square); their range is printed as the interval of the reference. Criterion: the run
is at most 3 times the reference level, for the largest residual and for the root mean square (the
factor of check 4 of stages B and C). The differences from the reference (decomposition, injector,
realisation, the policy at ten times the optical depth) are stated where the result is reported.
The run and the references must have the same kappa and number of passes. Exit 0 pass, 1 fail.
"""
import math
import re
import sys
from pathlib import Path


def levels(run):
    lines = (Path(run) / "bh_gauss.txt").read_text().splitlines()
    kappa = float(re.search(r"kappa ([-0-9.eE+]+)", lines[0]).group(1))
    n = int(re.search(r"filter passes (\d+)", lines[0]).group(1))
    head = lines[1][1:].split()
    rows = [[float(x) for x in l.split()] for l in lines[2:] if l.strip()]
    mx = max(r[head.index("max_residual")] for r in rows)
    rms = max(r[head.index("rms_residual")] for r in rows)
    return kappa, n, len(rows), mx, rms


def main():
    run, refs = sys.argv[1], sys.argv[2:]
    k, n, m, mx, rms = levels(run)
    unit = k * (math.comb(2 * n, n) / 4.0 ** n) ** 2
    rl = [levels(r) for r in refs]
    assert all(x[0] == k and x[1] == n for x in rl), "different kappa or filter passes"
    ref_mx, ref_rms = max(x[3] for x in rl), max(x[4] for x in rl)
    print(f"check 4 (against the policy-off runs): {Path(run).name}; kappa {k:g}, {n} passes, U = {unit:.4e}; {m} samples")
    for r, x in zip(refs, rl):
        print(f"  reference {Path(r).name}: largest residual {x[3] / unit:.4f} U, largest rms {x[4] / unit:.3e} U ({x[2]} samples)")
    ok = True
    for label, v, ref in (("largest residual", mx, ref_mx), ("largest rms residual", rms, ref_rms)):
        good = v <= 3.0 * ref
        ok &= good
        print(f"  {label}: {v / unit:.4g} U against 3 x {ref / unit:.4g} U (reference interval "
              f"{min(x[3 if label == 'largest residual' else 4] for x in rl) / unit:.4g} to {ref / unit:.4g} U), "
              f"ratio {v / ref:.3f}  {'pass' if good else 'FAIL'}")
    print(f"check 4 verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
