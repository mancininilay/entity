#!/usr/bin/env python3
"""The registered check of the birth write inside reconnection (Claude 1, 2026-11-23; standard library).

    lf_check.py <lf-ctl run> <lf-neg run> <reference run>

lf-ctl and lf-neg: 300 steps of s1-n-a at production size on four ranks with the lepton species of
the policy listed first and unequal stale slots; lf-neg leaves out the write of the previous
position of a new pair (the generator's test-only control). The reference is R-mid of stage C
(the same grid and optical depth, the emitter first, one card), whose samples up to step 500 bound
the level of an ordinary run at this stage.
Criteria (all needed for a pass):
  (a) both runs reached step 300 and made pairs;
  (b) largest residual of lf-neg / largest residual of lf-ctl >= 1000 (the check can fail);
  (c) largest residual of lf-ctl <= 10 x the largest residual of the reference over its samples at
      steps <= 500 (with the write, the residual stays at the level of the other runs).
The three files must have the same kappa and number of filter passes. Exit 0 pass, 1 fail.
"""
import re
import sys
from pathlib import Path


def table(run):
    lines = (Path(run) / "bh_gauss.txt").read_text().splitlines()
    meta = (float(re.search(r"kappa ([-0-9.eE+]+)", lines[0]).group(1)), int(re.search(r"filter passes (\d+)", lines[0]).group(1)))
    head = lines[1][1:].split()
    rows = [[float(x) for x in l.split()] for l in lines[2:] if l.strip()]
    return meta, {h: [r[i] for r in rows] for i, h in enumerate(head)}


def last_counter(run, key):
    lines = (Path(run) / "bh_counters.txt").read_text().splitlines()
    head = lines[0][1:].split()
    row = lines[-1].split()
    return float(row[head.index(key)]), float(row[head.index("steps")])


def main():
    ctl, neg, ref = sys.argv[1:4]
    (mc, c), (mn, n), (mr, r) = table(ctl), table(neg), table(ref)
    assert mc == mn == mr, (mc, mn, mr)
    ok = True
    print(f"birth write inside reconnection: {Path(ctl).name}, {Path(neg).name}; reference {Path(ref).name} (steps <= 500)")
    for run in (ctl, neg):
        pairs, steps = last_counter(run, "pairs")
        good = steps >= 300 and pairs > 0
        ok &= good
        print(f"  (a) {Path(run).name}: last step {steps:.0f}, pairs {pairs:.0f}  {'pass' if good else 'FAIL'}")
    lc, ln = max(c["max_residual"]), max(n["max_residual"])
    lr = max(v for s, v in zip(r["steps"], r["max_residual"]) if s <= 500)
    ratio = ln / lc
    good_b, good_c = ratio >= 1000.0, lc <= 10.0 * lr
    ok &= good_b and good_c
    print(f"  (b) largest residual: without the write {ln:.4e}, with it {lc:.4e}, ratio {ratio:.4g} (at least 1000)  "
          f"{'pass' if good_b else 'FAIL'}")
    print(f"  (c) with the write {lc:.4e} against 10 x {lr:.4e} of the reference  {'pass' if good_c else 'FAIL'}")
    for name, t in (("lf-ctl", c), ("lf-neg", n)):
        print(f"      {name} by sample: " + ", ".join(f"{s:.0f}:{v:.3e}" for s, v in zip(t["steps"], t["max_residual"])))
    print(f"birth write check verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
