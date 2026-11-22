#!/usr/bin/env python3
"""Check 6, second part: numerical health of the loaded plasma (stage C).

    check6_health.py <series of the tested run .npz> <series of the reference run .npz>
                     <tested run directory> <reference run directory> [more reference run directories]

The two series are written by check6_series.py. The run directories give the
counters (`bh_counters.txt`); the first reference directory is the run of the
reference series, the others are further realisations of the same kind.

Three criteria, all needed for a pass:

  6a  stability. w_p dt of the mixture inside 5 skin depths of the densest
      point of the pairs is below 2 (the limit of the leapfrog scheme for
      plasma oscillations) at every field output.
  6b  no heating of the pairs where they are densest. At every field output
      of the tested run at which the plasmoid holding the densest point has a
      size of at least 25 skin depths: inside 5 and inside 2 skin depths, the
      median K of the leptons of the pairs is at most twice the largest value
      of the reference over its outputs of that size, and their mean of
      1 / gamma is at least half the smallest value of the reference.
      No output of that size in the tested run or in the reference: no
      verdict. An output whose plasmoid is larger than 1.5 times the largest
      of the reference is outside the reference and is reported, not judged.
  6c  the box. The largest value, over t >= runtime / 10 and at least 1000
      leptons present, of the mean K of all leptons of the pairs
      (bh_counters.txt) is at most twice the largest such value over the
      reference runs.

Reported without criterion: w_p dt at 1, 2, 10 and 20 skin depths; cells per
skin depth; the ratio of pairs to primaries in the circles; the mean K and
the mean of 1 / gamma of the ions there, against the band of the reference.
Exit 0 pass, 1 fail, 2 no verdict.
"""
import re
import sys
from pathlib import Path

import numpy as np

SIZE_MIN, SIZE_BEYOND = 25.0, 1.5
WP_LIMIT = 2.0
FACTOR = 2.0
MIN_LEPTONS = 1000.0


def series(path):
    z = np.load(path)
    names = [str(n) for n in z["names"]]
    t = z["table"]
    return {n: t[:, i] for i, n in enumerate(names)}, str(z["run"])


def counters(run_dir):
    path = Path(run_dir) / "bh_counters.txt"
    header = open(path).readline()[1:].split()
    d = np.loadtxt(path, ndmin=2)
    c = {h: d[:, i] for i, h in enumerate(header)}
    text = next(Path(run_dir).glob("*.toml")).read_text()
    runtime = float(re.search(r"^\s*runtime\s*=\s*([-0-9.eE+]+)", text, re.M).group(1))
    return c, runtime


def box_pair_energy(run_dir):
    c, runtime = counters(run_dir)
    n = c["n_3"] + c["n_4"]
    m = (c["time"] >= runtime / 10.0) & (n >= MIN_LEPTONS)
    if not m.any():
        return float("nan"), float("nan"), c, runtime
    k = (c["K_3"][m] + c["K_4"][m]) / n[m]
    i = int(np.argmax(k))
    return float(k[i]), float(c["time"][m][i]), c, runtime


def at(c, r):
    return c["radius"] == r


def band(c, key, r, big):
    v = c[key][at(c, r) & big]
    v = v[np.isfinite(v)]
    return (float(v.min()), float(v.max())) if v.size else (float("nan"), float("nan"))


def main():
    test, test_name = series(sys.argv[1])
    ref, ref_name = series(sys.argv[2])
    test_dir, ref_dirs = sys.argv[3], sys.argv[4:]
    print(f"check 6: {test_name} against the reference {ref_name}")
    codes = []

    # ---- 6a
    m = at(test, 5.0) & np.isfinite(test["wpdt_all"])
    worst = int(np.argmax(test["wpdt_all"][m])) if m.any() else None
    if worst is None:
        print("  6a stability: no output with enough written leptons inside 5 skin depths  NO VERDICT")
        codes.append(2)
    else:
        w, t = test["wpdt_all"][m][worst], test["time"][m][worst]
        ok = bool((test["wpdt_all"][m] < WP_LIMIT).all())
        codes.append(0 if ok else 1)
        print(f"  6a stability: largest w_p dt inside 5 skin depths {w:.3f} at t = {t:.0f} "
              f"({int(m.sum())} outputs); limit {WP_LIMIT:g}  {'PASS' if ok else 'FAIL'}")
    for r in (1.0, 2.0, 10.0, 20.0):
        mm = at(test, r) & np.isfinite(test["wpdt_all"])
        if mm.any():
            i = int(np.argmax(test["wpdt_all"][mm]))
            print(f"     reported: inside {r:g}: largest w_p dt {test['wpdt_all'][mm][i]:.3f} (pairs alone "
                  f"{test['wpdt_pair'][mm][i]:.3f}) at t = {test['time'][mm][i]:.0f}, cells per skin depth there "
                  f"{test['cells_all'][mm][i]:.2f}; outputs at or above the limit: "
                  f"{int((test['wpdt_all'][mm] >= WP_LIMIT).sum())} of {int(mm.sum())}")
    mm = at(test, 5.0) & np.isfinite(test["cells_all"])
    if mm.any():
        print(f"     reported: smallest number of cells per skin depth inside 5: {test['cells_all'][mm].min():.2f}")

    # ---- 6b
    big_ref = ref["size"] >= SIZE_MIN
    big_test = test["size"] >= SIZE_MIN
    size_max_ref = float(ref["size"][big_ref].max()) if big_ref.any() else float("nan")
    code_b, judged = 0, 0
    for r in (5.0, 2.0):
        lo_k, hi_k = band(ref, "lep_medK", r, big_ref)
        lo_g, hi_g = band(ref, "lep_invgamma", r, big_ref)
        print(f"  6b inside {r:g}: reference band over {int((at(ref, r) & big_ref).sum())} outputs of size >= "
              f"{SIZE_MIN:g} (sizes up to {size_max_ref:.1f}): median K {lo_k:.3f} to {hi_k:.3f}, <1/gamma> "
              f"{lo_g:.4f} to {hi_g:.4f}; bounds: median K <= {FACTOR * hi_k:.3f}, <1/gamma> >= {lo_g / FACTOR:.4f}")
        sel = np.where(at(test, r) & big_test)[0]
        for i in sel:
            k, g, size, t = test["lep_medK"][i], test["lep_invgamma"][i], test["size"][i], test["time"][i]
            if not (np.isfinite(k) and np.isfinite(lo_k)):
                print(f"     t = {t:.0f}: size {size:.1f}: no statistics")
                continue
            if size > SIZE_BEYOND * size_max_ref:
                print(f"     t = {t:.0f}: size {size:.1f} is beyond the reference: median K {k:.3f}, <1/gamma> "
                      f"{g:.4f}  not judged")
                continue
            ok = (k <= FACTOR * hi_k) and (g >= lo_g / FACTOR)
            judged += 1
            if not ok:
                code_b = 1
            print(f"     t = {t:.0f}: size {size:.1f}: median K {k:.3f}, mean K {test['lep_meanK'][i]:.2f}, "
                  f"<1/gamma> {g:.4f}, written {test['lep_written'][i]:.0f}  {'ok' if ok else 'OUTSIDE'}")
    if judged == 0:
        code_b = 2
    codes.append(code_b)
    print(f"  6b heating of the pairs: {judged} values judged  "
          f"{['PASS', 'FAIL', 'NO VERDICT (no output of that size)'][code_b]}")

    # ---- reported: pairs per primary, the ions
    for r in (1.0, 5.0, 20.0):
        mm = at(test, r)
        print(f"     reported: inside {r:g}, pairs per primary (N_3 + N_4) / (N_1 + N_2) by output: "
              + " ".join(f"{t:.0f}:{a / b:.3f}" for t, a, b in
                         zip(test["time"][mm], test["n_pair"][mm], test["n_prim"][mm])))
    for r in (5.0, 2.0):
        lo, hi = band(ref, "ion_meanK", r, big_ref)
        lo_g, hi_g = band(ref, "ion_invgamma", r, big_ref)
        mm = at(test, r) & big_test & np.isfinite(test["ion_meanK"])
        if mm.any():
            print(f"     reported: ions inside {r:g} at size >= {SIZE_MIN:g}: mean K {test['ion_meanK'][mm].min():.1f} "
                  f"to {test['ion_meanK'][mm].max():.1f} (reference {lo:.1f} to {hi:.1f}); <1/gamma> "
                  f"{test['ion_invgamma'][mm].min():.4f} to {test['ion_invgamma'][mm].max():.4f} "
                  f"(reference {lo_g:.4f} to {hi_g:.4f})")

    # ---- 6c
    k_test, t_test, c_test, _ = box_pair_energy(test_dir)
    refs = [box_pair_energy(d)[:2] for d in ref_dirs]
    k_ref = max((k for k, _ in refs if np.isfinite(k)), default=float("nan"))
    if not (np.isfinite(k_test) and np.isfinite(k_ref)):
        codes.append(2)
        print("  6c the box: not enough leptons  NO VERDICT")
    else:
        ok = k_test <= FACTOR * k_ref
        codes.append(0 if ok else 1)
        print(f"  6c the box: largest mean K of the leptons of the pairs {k_test:.3f} at t = {t_test:.0f}; "
              "reference runs " + ", ".join(f"{k:.3f} (t = {t:.0f})" for k, t in refs)
              + f"; bound {FACTOR * k_ref:.3f}  {'PASS' if ok else 'FAIL'}")
    n1 = c_test["n_1"]
    print("     reported: leptons of the pairs present per primary electron in the box: "
          + " ".join(f"{c_test['time'][i]:.0f}:{(c_test['n_3'][i] + c_test['n_4'][i]) / n1[i]:.4f}"
                     for i in np.linspace(0, len(n1) - 1, 13).astype(int)))

    code = 1 if 1 in codes else (2 if 2 in codes else 0)
    print(f"check 6 verdict: {['PASS', 'FAIL', 'NO VERDICT'][code]} (6a, 6b, 6c: "
          + ", ".join(['pass', 'fail', 'no verdict'][c] for c in codes) + ")")
    return code


if __name__ == "__main__":
    sys.exit(main())
