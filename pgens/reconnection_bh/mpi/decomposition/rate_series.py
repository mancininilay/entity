#!/usr/bin/env python3
"""Reported, no rule: the reconnection rate of check 3 at every field output of a run
(the mean of the inflow speed in the four quadrants, and the four quadrants).

    rate_series.py <bp2raw> <cache> <run directory> <output.json>

The estimator is the one of scalars.py, which is checked there against the registered
function of check 3.
"""
import json
import shutil
import sys

import numpy as np

from bhio import Run


def quadrants(run, bp):
    x0, x1, y0, y1 = run.extent()
    cx, cy, hx, hy = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    v = run.fields(bp, ["fE1", "fE3", "fB1", "fB2", "fB3"])
    x, y = v["X1"] - cx, v["X2"] - cy
    e1, e3 = v["fE1"].astype(np.float64), v["fE3"].astype(np.float64)
    b1, b2, b3 = (v[k].astype(np.float64) for k in ("fB1", "fB2", "fB3"))
    vy = (e3 * b1 - e1 * b3) / (b1 * b1 + b2 * b2 + b3 * b3)
    quad = []
    for ysign in (1.0, -1.0):
        my = (ysign * y >= 0.4 * hy) & (ysign * y <= 0.6 * hy)
        for xsign in (-1.0, 1.0):
            mx = (xsign * x > 0.0) & (np.abs(x) <= 0.8 * hx)
            quad.append(float(-ysign * vy[np.ix_(mx, my)].mean()))
    return quad            # above left, above right, below left, below right


def main():
    bp2raw, cache, path, out = sys.argv[1:5]
    run = Run(path, bp2raw, cache)
    rows = []
    for bp, step, time in run.stamps("fields"):
        q = quadrants(run, bp)
        rows.append({"t": time, "rate": float(np.mean(q)), "quadrants": q})
        shutil.rmtree(run.cache / "fields" / bp.name, ignore_errors=True)
    shutil.rmtree(run.cache, ignore_errors=True)
    with open(out, "w") as f:
        json.dump({"name": run.name, "rows": rows}, f)
    print(run.name + ": " + " ".join(f"{r['rate']:.4f}" for r in rows))


if __name__ == "__main__":
    main()
