#!/usr/bin/env python3
"""V6, energy: loss of the emitters against the energy given to the pairs.

    analyze_energy.py <run dir> [--theta <angle>] [--limit <relative limit>]

From the last BHV line: the emitters' loss -sum w (K - K0) against
sum over pairs of a w (E- + E+) - (absorbed photon energy). With --limit the
relative difference is a verdict (fields off); with --theta the measured
relative excess of the pairs' energy is printed beside theta^2 / 2 (no verdict).
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = []
    for line in open(os.path.join(args.run_dir, "run.log")):
        if line.startswith("BHV step"):
            t = line.split()
            rows.append(dict(zip(t[1::2], (float(x) for x in t[2::2]))))
    r = rows[-1]
    tag = args.tag or os.path.basename(os.path.normpath(args.run_dir))
    loss = -r["ion_dK"]
    pairs = r["el_E"] + r["po_E"] - r["ne_E"] - r["absorbed"]
    rel = (pairs - loss) / pairs
    line = (f"{tag}: step {r['step']:.0f}, pairs {r['pairs']:.0f}, emitters' loss {loss:.10e}, "
            f"pairs' energy minus absorbed {pairs:.10e}, relative difference {rel:+.3e}")
    rc = 0
    if args.limit is not None:
        ok = abs(rel) <= args.limit
        rc = 0 if ok else 1
        line = ("PASS " if ok else "FAIL ") + "V6 " + line + f" (limit {args.limit:g})"
    if args.theta is not None:
        line = "(diag) V6 " + line + f"; theta = {args.theta:g}: theta^2/2 = {0.5 * args.theta ** 2:.4e}, measured/predicted = {rel / (0.5 * args.theta ** 2):.4f}"
    print(line)
    return rc


if __name__ == "__main__":
    sys.exit(main())
