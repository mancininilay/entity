#!/usr/bin/env python3
"""The level of the Gauss-law residual of a run in units of one filtered charge (Claude 1, 2026-11-23;
standard library). Check 4 (`scripts/check4_gauss.py`) compares two runs with the same kappa;
twice the resolution has another kappa, so the fine grid is reported here in the unit U of
check 4: U = kappa (C(2n, n) / 4^n)^2, the largest node value one particle of unit charge and
weight leaves after n passes of the filter.

    gauss_level.py <run directory> [<run directory> ...]

For each run: kappa, the number of filter passes, the samples, and the largest residual and the
root mean square over its samples, in the units of the file and in U, with the step where the
largest residual occurs.
"""
import math
import re
import sys
from pathlib import Path


def level(run):
    lines = (Path(run) / "bh_gauss.txt").read_text().splitlines()
    kappa = float(re.search(r"kappa ([-0-9.eE+]+)", lines[0]).group(1))
    n = int(re.search(r"filter passes (\d+)", lines[0]).group(1))
    head = lines[1][1:].split()
    rows = [[float(x) for x in l.split()] for l in lines[2:] if l.strip()]
    col = {h: [r[i] for r in rows] for i, h in enumerate(head)}
    unit = kappa * (math.comb(2 * n, n) / 4.0 ** n) ** 2
    k = max(range(len(rows)), key=lambda i: col["max_residual"][i])
    return kappa, n, len(rows), unit, col["max_residual"][k], col["steps"][k], max(col["rms_residual"])


for run in sys.argv[1:]:
    kappa, n, m, unit, mr, step, rr = level(run)
    print(f"{Path(run).name}: kappa {kappa:.6g}, {n} passes, {m} samples, U = {unit:.4e}; largest residual "
          f"{mr:.4e} ({mr / unit:.3e} U) at step {step:.0f}; largest rms {rr:.4e} ({rr / unit:.3e} U)")
