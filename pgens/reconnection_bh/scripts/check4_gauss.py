#!/usr/bin/env python3
"""Check 4: the residual of Gauss's law of the run with the policy at the
level of the run without.

    check4_gauss.py <bp2raw> <cache> <run without> <run with>

From bh_gauss.txt (see pgen.hpp): residual r = div E - kappa F[rho] on the
nodes of the window, its largest absolute value and its root mean square at
every sample. Level of a run: the largest value over its samples.
Criteria: level(with) <= 3 level(without), for the largest value and for
the root mean square.
Reported: the levels in units of U = kappa (C(2n, n) / 4^n)^2, the largest
node value that one particle of unit charge and weight leaves after the n
passes of the filter: a single charge created or lost in the window shows as
a residual of up to U.
"""
import math
import sys

from bhio import Run, verdict


def main():
    bp2raw, cache = sys.argv[1], sys.argv[2]
    off, low = Run(sys.argv[3], bp2raw, cache), Run(sys.argv[4], bp2raw, cache)
    g_off, g_low = off.gauss(), low.gauss()
    n = g_off["passes"]
    assert g_low["passes"] == n and g_low["kappa"] == g_off["kappa"]
    unit = g_off["kappa"] * (math.comb(2 * n, n) / 4.0 ** n) ** 2
    print(f"check 4: {low.name} against {off.name}; kappa {g_off['kappa']:.6g}, {n} filter passes, "
          f"one unit charge U = {unit:.4e}")
    ok = True
    for key, label in (("max_residual", "largest residual"), ("rms_residual", "rms residual")):
        lo, ll = g_off[key].max(), g_low[key].max()
        good = ll <= 3.0 * lo
        ok = ok and good
        print(f"  {label}: without {lo:.4e} ({lo / unit:.3e} U), with {ll:.4e} ({ll / unit:.3e} U), "
              f"ratio {ll / lo:.3f}  {verdict(good)}")
    for run, g in ((off, g_off), (low, g_low)):
        print(f"  {run.name}: {g['steps'].size} samples to step {g['steps'][-1]:.0f}; at the end "
              f"largest residual {g['max_residual'][-1]:.4e}, rms {g['rms_residual'][-1]:.4e}, "
              f"largest charge term {g['max_charge'][-1]:.4e}, rms {g['rms_charge'][-1]:.4e}")
    print(f"  sensitivity: the level without the policy is {g_off['max_residual'].max() / unit:.3e} U "
          f"(a single unit charge is resolved when this is well below 1)")
    print(f"check 4 verdict: {verdict(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
