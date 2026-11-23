#!/usr/bin/env python3
"""Check 6 with plasmoids compared at the same position (series preparation, 2026-11-22).

    check6_position.py --half-length L --test <series .npz> --ref <series .npz> [<series .npz> ...]
                       [--test-dir <run directory> --ref-dirs <run directory> [...]]

The series are written by check6_series.py (one per run). The reference is an ensemble: the
members are realisations of the setup the tested run is compared with, and every one of them
supplies values. L is half the length of the box along the sheet, in skin depths; the middle of
the box is x1 = 0.

Why. The pairs of a plasmoid gain energy while the plasmoid travels to an open end of the box,
at the original optical depth too (stage C, report section 17.1). A bound taken from one plasmoid
of one run that stays in the middle of the box then fails every plasmoid that travels. Here a
plasmoid of the tested run is judged against the plasmoids of the members that were at the same
distance from the middle of the box.

  6a  stability, as in check6_health.py: w_p dt of the mixture inside 5 skin depths of the
      densest point of the pairs is below 2 at every field output of the tested run.

  6b  heating of the pairs at the same position. An output counts when the plasmoid that holds
      the densest point of the pairs has a size of at least 25 skin depths. Its position is the
      distance d of that point from the middle of the box, in bins of L / 10. For a bin, the
      reference is every counting output of every member in that bin. At every counting output
      of the tested run whose bin holds at least 3 reference outputs: inside 5 and inside 2
      skin depths the median K of the leptons of the pairs is at most twice the largest
      reference value of the bin, and their mean of 1 / gamma at least half the smallest.
      An output whose bin holds fewer than 3 reference outputs is not judged and is listed.
      Fewer than 3 judged outputs: no verdict.

  6c  the box, as in check6_health.py, when the run directories are given: the largest mean K
      of all leptons of the pairs is at most twice the largest over the members.

Exit 0 pass, 1 fail, 2 no verdict.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check6_health as h  # noqa: E402  (series, at, box_pair_energy, the constants of 6a and 6c)

SIZE_MIN = 25.0
BINS = 10
REF_MIN = 3
JUDGED_MIN = 3
FACTOR = 2.0


def points(c, radius, half_length):
    """Counting outputs of one run: (time, bin, distance, median K, <1/gamma>, size)."""
    m = h.at(c, radius) & (c["size"] >= SIZE_MIN) & np.isfinite(c["lep_medK"]) & np.isfinite(c["lep_invgamma"])
    d = np.abs(c["xc"][m])
    b = np.minimum((d / (half_length / BINS)).astype(int), BINS - 1)
    return c["time"][m], b, d, c["lep_medK"][m], c["lep_invgamma"][m], c["size"][m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-length", type=float, required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--test-dir")
    ap.add_argument("--ref-dirs", nargs="*", default=[])
    a = ap.parse_args()
    test, test_name = h.series(a.test)
    refs = [h.series(r) for r in a.ref]
    width = a.half_length / BINS
    print(f"check 6 by position: {test_name} against the members " + ", ".join(n for _, n in refs)
          + f"; bins of {width:g} skin depths of distance from the middle of the box")
    codes = []

    # ---- 6a (as check6_health.py)
    m = h.at(test, 5.0) & np.isfinite(test["wpdt_all"])
    if not m.any():
        print("  6a stability: no output with enough written leptons inside 5 skin depths  NO VERDICT")
        codes.append(2)
    else:
        i = int(np.argmax(test["wpdt_all"][m]))
        ok = bool((test["wpdt_all"][m] < h.WP_LIMIT).all())
        codes.append(0 if ok else 1)
        print(f"  6a stability: largest w_p dt inside 5 skin depths {test['wpdt_all'][m][i]:.3f} at t = "
              f"{test['time'][m][i]:.0f} ({int(m.sum())} outputs); limit {h.WP_LIMIT:g}  {'PASS' if ok else 'FAIL'}")

    # ---- 6b by position
    code_b, judged, unjudged = 0, 0, 0
    for radius in (5.0, 2.0):
        ref_b, ref_k, ref_g, ref_who = [], [], [], []
        for c, name in refs:
            _, b, _, k, g, _ = points(c, radius, a.half_length)
            ref_b += list(b)
            ref_k += list(k)
            ref_g += list(g)
            ref_who += [name] * len(b)
        ref_b, ref_k, ref_g = np.array(ref_b, dtype=int), np.array(ref_k), np.array(ref_g)
        print(f"  6b inside {radius:g}: reference outputs per bin (largest median K, smallest <1/gamma>, members):")
        for b in range(BINS):
            s = ref_b == b
            if s.any():
                who = sorted({w for w, f in zip(ref_who, s) if f})
                print(f"     {b * width:5.0f} to {(b + 1) * width:5.0f}: {int(s.sum()):3d} outputs, "
                      f"{ref_k[s].max():.3f}, {ref_g[s].min():.4f}, {', '.join(who)}")
        t, b, d, k, g, size = points(test, radius, a.half_length)
        for ti, bi, di, ki, gi, si in zip(t, b, d, k, g, size):
            s = ref_b == bi
            if int(s.sum()) < REF_MIN:
                unjudged += 1
                print(f"     t = {ti:.0f}: distance {di:.0f}, size {si:.1f}: median K {ki:.3f}, <1/gamma> {gi:.4f}; "
                      f"{int(s.sum())} reference outputs in the bin  not judged")
                continue
            hi, lo = FACTOR * ref_k[s].max(), ref_g[s].min() / FACTOR
            ok = (ki <= hi) and (gi >= lo)
            judged += 1
            if not ok:
                code_b = 1
            print(f"     t = {ti:.0f}: distance {di:.0f}, size {si:.1f}: median K {ki:.3f} (bound {hi:.3f}), "
                  f"<1/gamma> {gi:.4f} (bound {lo:.4f})  {'ok' if ok else 'OUTSIDE'}")
    if code_b == 0 and judged < JUDGED_MIN:
        code_b = 2
    codes.append(code_b)
    print(f"  6b heating of the pairs at the same position: {judged} values judged, {unjudged} without a reference  "
          f"{['PASS', 'FAIL', 'NO VERDICT'][code_b]}")

    # ---- 6c (as check6_health.py), when the run directories are given
    if a.test_dir and a.ref_dirs:
        k_test, t_test, _, _ = h.box_pair_energy(a.test_dir)
        ref_box = [h.box_pair_energy(d)[:2] for d in a.ref_dirs]
        k_ref = max((k for k, _ in ref_box if np.isfinite(k)), default=float("nan"))
        if not (np.isfinite(k_test) and np.isfinite(k_ref)):
            codes.append(2)
            print("  6c the box: not enough leptons  NO VERDICT")
        else:
            ok = k_test <= h.FACTOR * k_ref
            codes.append(0 if ok else 1)
            print(f"  6c the box: largest mean K of the leptons of the pairs {k_test:.3f} at t = {t_test:.0f}; "
                  "members " + ", ".join(f"{k:.3f} (t = {t:.0f})" for k, t in ref_box)
                  + f"; bound {h.FACTOR * k_ref:.3f}  {'PASS' if ok else 'FAIL'}")
    else:
        print("  6c the box: run directories not given, not evaluated")

    code = 1 if 1 in codes else (2 if 2 in codes else 0)
    print(f"check 6 by position verdict: {['PASS', 'FAIL', 'NO VERDICT'][code]} ("
          + ", ".join(['pass', 'fail', 'no verdict'][c] for c in codes) + ")")
    return code


if __name__ == "__main__":
    sys.exit(main())
