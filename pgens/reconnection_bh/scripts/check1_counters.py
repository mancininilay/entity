#!/usr/bin/env python3
"""Check 1: the counters of the policy at the end of a run.

    check1_counters.py <bp2raw> <cache> <run with the policy> [<run without>]
"""
import sys

from bhio import Run, verdict


def main():
    bp2raw, cache = sys.argv[1], sys.argv[2]
    low = Run(sys.argv[3], bp2raw, cache)
    c = low.counters()
    last = {k: v[-1] for k, v in c.items()}
    pairs, above, pexc = last["pairs"], last["above_tables"], last["probability_exceeded"]
    refused, dropped = last["loss_bound_skipped"], last["capacity_dropped"]
    candidates = pairs + refused + dropped
    print(f"check 1: {low.name}, last line at step {last['steps']:.0f}, time {last['time']:.6g}")
    print(f"  pairs {pairs:.0f}; absorbed photon energy {last['absorbed_energy']:.9g} "
          f"({last['absorbed_energy'] / max(pairs, 1):.9f} per pair)")
    print(f"  particle-steps above the tables: {above:.0f}  {verdict(above == 0)}")
    print(f"  pairs dropped for capacity: {dropped:.0f}  {verdict(dropped == 0)}")
    share = refused / candidates if candidates > 0 else 0.0
    print(f"  candidates refused by the loss bound: {refused:.0f} of {candidates:.0f} "
          f"candidates (share {share:.3e})  reported")
    print(f"  particle-steps above the probability bound: {pexc:.0f}  reported "
          f"(the run stops if not zero)")
    for s in (1, 2, 3, 4):
        if f"n_{s}" in last:
            print(f"  species {s}: {last[f'n_{s}']:.0f} particles, largest K over the run "
                  f"{c[f'Kmax_{s}'].max():.6g}, at the end {last[f'Kmax_{s}']:.6g}")
    mass_ratio = low.param("mass_ratio")
    gamma_max = low.param("gamma_max")
    k_top = c["Kmax_2"].max()
    print(f"  largest ion K over the run {k_top:.6g}: Gamma_p {1 + k_top / mass_ratio:.4f} "
          f"against the upper edge of the tables {gamma_max:g}")
    ok = (above == 0) and (dropped == 0)
    if len(sys.argv) > 4:
        off = Run(sys.argv[4], bp2raw, cache)
        co = off.counters()
        zero = all(co[k][-1] == 0 for k in ("pairs", "above_tables", "probability_exceeded",
                                            "loss_bound_skipped", "capacity_dropped",
                                            "absorbed_energy"))
        print(f"  {off.name}: every counter zero at the end: {verdict(zero)}; "
              f"lepton species empty: {verdict(co['n_3'][-1] == 0 and co['n_4'][-1] == 0)}")
        ok = ok and zero and co["n_3"][-1] == 0 and co["n_4"][-1] == 0
    print(f"check 1 verdict: {verdict(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
