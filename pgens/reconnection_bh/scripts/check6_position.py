#!/usr/bin/env python3
"""Check 6 with plasmoids compared at the same position (series preparation, 2026-11-22; coverage
and missing data corrected 2026-11-23 after Codex 2's review, B1).

    check6_position.py --half-length L --test <series .npz> --ref <series .npz> [<series .npz> ...]
                       --test-dir <run directory> --ref-dirs <run directory> [...] [--frame]

The series are written by check6_series.py or by series/core_series.py (one per run). The
reference is an ensemble: the members are realisations of the setup the tested run is compared
with. L is half the length of the box along the sheet, in skin depths; the middle of the box is
x1 = 0.

Why. The pairs of a plasmoid travelling to an open end of the box were hotter the further it had
gone, at the original optical depth too (stage C, report section 17.1). A bound taken from one
plasmoid of one run that stays in the middle of the box then fails every plasmoid that travels.
Here a plasmoid of the tested run is judged against the plasmoids of the members that were at the
same distance from the middle of the box. Position does not control the history of a plasmoid
(its age, its size, how close its edge is to the outlet); the series written by core_series.py
carry those as diagnostics, and this script reports them; it does not match them.

  6a  stability, as in check6_health.py: w_p dt of the mixture inside 5 skin depths of the
      densest point of the pairs is below 2 at every field output of the tested run.

  6b  heating of the pairs at the same position.
      Eligible epoch: a field output at which the plasmoid that holds the densest point of the
      pairs has an equivalent radius (`size`) of at least 25 skin depths. Its position is the
      distance d of that point from the middle of the box, in bins of L / 10. Both apertures,
      5 and 2 skin depths, are required at every eligible epoch.
      Reference of a bin and aperture: the eligible epochs of the members in that bin with finite
      values. It covers the bin when it holds at least 3 epochs from at least 2 different members
      and at least 2 different island histories (a history is a run, or, with series of
      core_series.py, one of its numbered island histories).
      An eligible epoch of the tested run with a value that is not finite in a required aperture
      is INADEQUATE: it is listed, and the verdict cannot be a pass.
      An eligible epoch is judged when both apertures are covered; it is inside when, in each
      aperture, the median K of the leptons of the pairs is at most twice the largest reference
      value of the bin and their mean of 1 / gamma at least half the smallest. A judged epoch
      outside: fail. Epochs without coverage are listed and not judged.
      Pass needs at least 3 judged epochs (counted as epochs, not as apertures), none outside and
      none inadequate; otherwise no verdict.

  6c  the box, as in check6_health.py: the largest mean K of all leptons of the pairs is at most
      twice the largest over the members. The run directories are required; without them 6c, and
      so the check, has no verdict.

With --frame (series of `series/core_series.py`) 6b uses the median K and the mean of 1 / gamma
of the leptons of the pairs in the zero-momentum frame of the pairs in each aperture
(`lep_medK_frame`, `lep_invgamma_frame`) instead of the lab values.

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
REF_MIN = 3            # reference epochs of a bin
REF_RUNS = 2           # different members among them
REF_HISTORIES = 2      # different island histories among them
JUDGED_MIN = 3         # judged epochs for a pass
FACTOR = 2.0
APERTURES = (5.0, 2.0)

KCOL, GCOL = "lep_medK", "lep_invgamma"


def epochs(c, half_length):
    """Eligible epochs of one run: {time: (bin, distance, size, history, {aperture: (K, 1/gamma)}), ...}."""
    out = {}
    hist = c.get("history")
    for i in np.where((c["size"] >= SIZE_MIN) & np.isin(c["radius"], APERTURES))[0]:
        t = float(c["time"][i])
        d = abs(float(c["xc"][i]))
        b = min(int(d / (half_length / BINS)), BINS - 1)
        hid = int(hist[i]) if hist is not None and np.isfinite(hist[i]) else 0
        e = out.setdefault(t, (b, d, float(c["size"][i]), hid, {}))
        e[4][float(c["radius"][i])] = (float(c[KCOL][i]), float(c[GCOL][i]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-length", type=float, required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--test-dir")
    ap.add_argument("--ref-dirs", nargs="*", default=[])
    ap.add_argument("--frame", action="store_true")
    a = ap.parse_args()
    global KCOL, GCOL
    if a.frame:
        KCOL, GCOL = "lep_medK_frame", "lep_invgamma_frame"
    test, test_name = h.series(a.test)
    refs = [h.series(r) for r in a.ref]
    width = a.half_length / BINS
    print(f"check 6 by position: {test_name} against the members " + ", ".join(n for _, n in refs)
          + f"; bins of {width:g} skin depths of distance from the middle of the box; energies "
          + ("in the frame of the pairs" if a.frame else "in the lab frame")
          + f"; a bin is covered by >= {REF_MIN} epochs from >= {REF_RUNS} members and >= {REF_HISTORIES} histories")
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
    ref = {r: [] for r in APERTURES}     # (bin, K, 1/gamma, member, history)
    for c, name in refs:
        for t, (b, d, size, hid, vals) in epochs(c, a.half_length).items():
            for r in APERTURES:
                if r in vals and np.isfinite(vals[r][0]) and np.isfinite(vals[r][1]):
                    ref[r].append((b, vals[r][0], vals[r][1], name, (name, hid)))
    print("  6b reference per bin and aperture (epochs, members, histories; largest median K, smallest <1/gamma>):")
    cover = {}
    for r in APERTURES:
        for b in range(BINS):
            s = [x for x in ref[r] if x[0] == b]
            runs, hists = {x[3] for x in s}, {x[4] for x in s}
            ok = len(s) >= REF_MIN and len(runs) >= REF_RUNS and len(hists) >= REF_HISTORIES
            cover[(r, b)] = (ok, max((x[1] for x in s), default=np.nan), min((x[2] for x in s), default=np.nan))
            if s:
                print(f"     aperture {r:g}, {b * width:5.0f} to {(b + 1) * width:5.0f}: {len(s):3d} epochs, "
                      f"{len(runs)} members, {len(hists)} histories; {cover[(r, b)][1]:.3f}, {cover[(r, b)][2]:.4f}"
                      f"{'' if ok else '  (not covered)'}")
    judged, outside, inadequate, uncovered = 0, 0, 0, 0
    for t, (b, d, size, hid, vals) in sorted(epochs(test, a.half_length).items()):
        missing = [r for r in APERTURES if r not in vals or not (np.isfinite(vals[r][0]) and np.isfinite(vals[r][1]))]
        if missing:
            inadequate += 1
            print(f"     t = {t:.0f}: distance {d:.0f}, size {size:.1f}: no finite value in aperture(s) "
                  f"{', '.join(f'{r:g}' for r in missing)}  INADEQUATE")
            continue
        if not all(cover[(r, b)][0] for r in APERTURES):
            uncovered += 1
            print(f"     t = {t:.0f}: distance {d:.0f}, size {size:.1f}: bin not covered  not judged")
            continue
        judged += 1
        parts, ok = [], True
        for r in APERTURES:
            k, g = vals[r]
            hi, lo = FACTOR * cover[(r, b)][1], cover[(r, b)][2] / FACTOR
            inside = (k <= hi) and (g >= lo)
            ok &= inside
            parts.append(f"aperture {r:g}: median K {k:.3f} (bound {hi:.3f}), <1/gamma> {g:.4f} (bound {lo:.4f})")
        outside += not ok
        print(f"     t = {t:.0f}: distance {d:.0f}, size {size:.1f}, history {hid}: " + "; ".join(parts)
              + f"  {'inside' if ok else 'OUTSIDE'}")
    if outside:
        code_b = 1
    elif inadequate or judged < JUDGED_MIN:
        code_b = 2
    else:
        code_b = 0
    codes.append(code_b)
    print(f"  6b heating of the pairs at the same position: {judged} epochs judged, {outside} outside, {uncovered} not "
          f"covered, {inadequate} inadequate  {['PASS', 'FAIL', 'NO VERDICT'][code_b]}")

    # ---- 6c (as check6_health.py); the run directories are required
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
        codes.append(2)
        print("  6c the box: run directories not given  NO VERDICT")

    code = 1 if 1 in codes else (2 if 2 in codes else 0)
    print(f"check 6 by position verdict: {['PASS', 'FAIL', 'NO VERDICT'][code]} ("
          + ", ".join(['pass', 'fail', 'no verdict'][c] for c in codes) + ")")
    return code


if __name__ == "__main__":
    sys.exit(main())
