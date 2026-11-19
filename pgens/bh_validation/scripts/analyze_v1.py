#!/usr/bin/env python3
"""V1: r(t) = max over nodes of |div E - rho| against the control run.

    analyze_v1.py <output root of run_v1.sh>

Pass (registered): r(t) <= 10 max_t r_off at every output, both species
orders, 1, 2 and 4 ranks. Negative control (registered): r exceeds
1000 max_t r_off.
"""
import os
import sys


def read(path):
    rows = []
    if not os.path.exists(path):
        return rows
    for line in open(path):
        if line.startswith("BHV step"):
            t = line.split()
            rows.append(dict(zip(t[1::2], (float(x) for x in t[2::2]))))
    return rows


def main():
    root = sys.argv[1]
    ok = True
    print(f"{'prec':6s} {'ranks':5s} {'case':12s} {'pairs':>9s} {'max_t r':>11s} {'max_t r_off':>11s} {'ratio':>10s}  verdict")
    for prec in ("single", "double"):
        for ranks in (1, 2, 4):
            base = os.path.join(root, prec, f"r{ranks}")
            off = {}
            for o in ("A", "B"):
                rows = read(os.path.join(base, f"{o}_off", "run.log"))
                off[o] = max(r["gauss_r"] for r in rows) if rows else float("nan")
            for case in ("A_on", "B_on", "A_on_stale", "B_on_stale", "A_neg", "A_neg_stale"):
                rows = read(os.path.join(base, case, "run.log"))
                if not rows:
                    print(f"{prec:6s} {ranks:5d} {case:12s} no output")
                    ok = False
                    continue
                r_off = off[case[0]]
                r_max = max(r["gauss_r"] for r in rows)
                pairs = rows[-1]["pairs"]
                ratio = r_max / r_off
                if "neg" in case:
                    verdict = "PASS (violates)" if ratio > 1000.0 else "FAIL (negative control too weak)"
                    registered = case == "A_neg"
                else:
                    verdict = "PASS" if (ratio <= 10.0 and pairs >= 1.0e4) else "FAIL"
                    registered = case in ("A_on", "B_on")
                if verdict.startswith("FAIL"):
                    ok = False
                tag = "" if registered else "  [additional, not registered]"
                print(f"{prec:6s} {ranks:5d} {case:12s} {pairs:9.0f} {r_max:11.3e} {r_off:11.3e} {ratio:10.3g}  {verdict}{tag}")
    print("ALL PASS" if ok else "SOME FAIL")


if __name__ == "__main__":
    main()
