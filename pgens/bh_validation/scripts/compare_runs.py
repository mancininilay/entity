#!/usr/bin/env python3
"""Two bh_validation runs against each other at the same physical times.

    compare_runs.py <run A> <run B> [--targets 0.9,0.7,0.5] [--label V9]

At the first record of A with <K>/K0 below each target, and at the record of
B nearest in time: <K> within 4 combined s.e. (s.e. = sigma_K / sqrt(N)),
Var K within 4 combined s.e. (s.e. = Var sqrt(2 / (N - 1))), pairs per emitter
within 4 combined s.e. (s.e. = sqrt(pairs per emitter / N)).
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bhv import Series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--targets", default="0.9,0.7,0.5")
    ap.add_argument("--label", default="compare")
    args = ap.parse_args()
    A, B = Series(args.run_a), Series(args.run_b)
    n_fail = 0
    for f in (float(x) for x in args.targets.split(",")):
        i = A.first_below(f)
        if i is None:
            print(f"FAIL {args.label}: target {f:.2f} not reached in {args.run_a}")
            n_fail += 1
            continue
        j = B.at_time(A.t[i])
        if abs(B.t[j] - A.t[i]) > 0.51 * max(A.dt, B.dt):
            print(f"FAIL {args.label}: no record of B at t = {A.t[i]:.6g} (nearest {B.t[j]:.6g})")
            n_fail += 1
            continue
        se = math.sqrt(A.se(i) ** 2 + B.se(j) ** 2)
        dm = (A.mean[i] - B.mean[j]) / se
        sv = math.sqrt(2.0 / (A.n_emitters - 1.0) * A.var[i] ** 2 + 2.0 / (B.n_emitters - 1.0) * B.var[j] ** 2)
        dv = (A.var[i] - B.var[j]) / sv
        sp = math.sqrt(A.pairs[i] / A.n_emitters + B.pairs[j] / B.n_emitters)
        dp = (A.pairs[i] - B.pairs[j]) / sp
        ok = abs(dm) <= 4.0 and abs(dv) <= 4.0 and abs(dp) <= 4.0
        n_fail += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} {args.label} at <K>/K0 = {f:.2f} (t = {A.t[i]:.6g}): <K> {A.mean[i]:.4f} vs {B.mean[j]:.4f} "
              f"({dm:+.2f} s.e.); Var K {A.var[i]:.5g} vs {B.var[j]:.5g} ({dv:+.2f} s.e.); pairs per emitter "
              f"{A.pairs[i]:.4f} vs {B.pairs[j]:.4f} ({dp:+.2f} s.e.); limit 4")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
