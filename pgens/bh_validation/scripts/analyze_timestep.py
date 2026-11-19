#!/usr/bin/env python3
"""V7, time step: <K> at fixed physical times for several time steps.

    analyze_timestep.py <run with the smallest p> <run> <run> ... [--targets 0.9,0.7,0.5]

The runs share the optical depth and differ by the time step (maximum
probability per step p). At the times where the first run crosses the targets:
the deviation of <K> from the first run, in combined s.e. and relative to K,
and the least-squares slope of the deviation against p (through the origin
at the first run's p). Registered: the deviation grows at most linearly.
Verdict used here: |dev(p)| <= (p - p0) / (p_max - p0) |dev(p_max)| + 4 s.e.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhv import Series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--targets", default="0.9,0.7,0.5")
    ap.add_argument("--p", default="", help="comma-separated maximum probabilities of the runs")
    args = ap.parse_args()
    S = [Series(r) for r in args.runs]
    p = [float(x) for x in args.p.split(",")] if args.p else [s.dt / S[0].dt for s in S]
    n_fail = 0
    for f in (float(x) for x in args.targets.split(",")):
        i0 = S[0].first_below(f)
        if i0 is None:
            print(f"FAIL V7: target {f:.2f} not reached by the first run")
            n_fail += 1
            continue
        t = S[0].t[i0]
        dev, se = [0.0], [0.0]
        for s in S[1:]:
            j = s.at_time(t)
            # linear interpolation of <K> to the time t between the neighbouring records
            j2 = j + 1 if s.t[j] < t and j + 1 < len(s.t) else (j - 1 if j > 0 else j)
            if j2 != j and s.t[j2] != s.t[j]:
                w = (t - s.t[j]) / (s.t[j2] - s.t[j])
                m = s.mean[j] + w * (s.mean[j2] - s.mean[j])
            else:
                m = s.mean[j]
            dev.append(m - S[0].mean[i0])
            se.append(math.sqrt(s.se(j) ** 2 + S[0].se(i0) ** 2))
        x = [pi - p[0] for pi in p]
        slope = sum(xi * di for xi, di in zip(x, dev)) / sum(xi * xi for xi in x)
        ok = True
        for k in range(1, len(S)):
            bound = x[k] / x[-1] * abs(dev[-1]) + 4.0 * se[k]
            ok = ok and abs(dev[k]) <= bound
        n_fail += 0 if ok else 1
        cells = "; ".join(f"p = {p[k]:g}: {dev[k]:+.4f} ({dev[k] / se[k]:+.2f} s.e., {dev[k] / S[0].mean[i0]:+.2e} of <K>)" for k in range(1, len(S)))
        print(f"{'PASS' if ok else 'FAIL'} V7 at <K>/K0 = {f:.2f} (t = {t:.6g}, <K> = {S[0].mean[i0]:.4f} at p = {p[0]:g}): deviations {cells}; "
              f"slope d<K>/dp = {slope:+.4g} ({slope / S[0].mean[i0]:+.3e} of <K> per unit p)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
