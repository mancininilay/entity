#!/usr/bin/env python3
"""Check 3 of stage C: the scatter between realisations as the reference.

    check3_ensemble.py <bp2raw> <cache> --off <run> <run> <run> --low <run> <run> <run> [--mid <run>]

The first run of each list is the original one (R-off, R-low); the others are
the members of the ensemble. The ten scalars are those of check 3 as
registered for stage B (check3_compare.py: the reconnection rate over
t >= runtime / 3; mean K, the 99 and 99.9 percent quantiles and the slope for
the electrons and the ions, and the fraction of ions above threshold, over
t >= 2 runtime / 3), each one number per run (the mean over its window).

Independence. For every pair of runs and every field output: the correlation
coefficient of the density of the primaries along the sheet (N_1 + N_2
averaged over |y| <= 20 skin depths, 256 bins in x). Two runs count as
independent realisations when the median of this correlation over the
outputs with t >= 2 runtime / 3 is below 0.5. If a pair of runs of the same
kind is not independent, the scatter is not measured and the script gives no
verdict (exit 2).

Scatter. For every scalar: the standard deviation over the three runs of each
kind (two degrees of freedom each), and the pooled one, s_p (four degrees of
freedom).

Rules, each for the ten scalars at once (Bonferroni: 0.5 percent per scalar,
two-sided, for 5 percent over the family):
  A. the kind with the policy against the kind without:
       |mean(low) - mean(off)| <= 5.598 s_p sqrt(2 / 3)
  B. the original pair, judged by the four new members only (s_new: the pooled
     standard deviation of the two pairs of members, two degrees of freedom):
       |Q(R-low) - Q(R-off)| <= 14.089 s_new sqrt(2)
  C. R-mid against the kind R-low:
       |Q(R-mid) - mean(low)| <= 5.598 s_p sqrt(4 / 3)
The values at the uncorrected 5 percent level (2.776, 4.303, 2.776) are
printed next to them. Exit 0 when nothing lies outside, 1 when something
does, 2 when the scatter is not measured.

Reported without a rule: the reconnection rate over t >= 2 runtime / 3 only
(the window in which independence is tested; the registered rate starts at
runtime / 3), the number of primaries at the end, the pairs created and the
largest ion K, with their scatter.
"""
import shutil
import sys

import numpy as np

import check3_compare as c3
from bhio import Run, threshold

T4_FAMILY, T4_SINGLE = 5.5976, 2.7764     # Student t, 4 degrees of freedom: 0.9975 and 0.975
T2_FAMILY, T2_SINGLE = 14.0890, 4.3027    # 2 degrees of freedom
INDEPENDENT_BELOW = 0.5
PROFILE_BINS, PROFILE_HALF_WIDTH = 256, 20.0


def profiles(run):
    """Density of the primaries along the sheet at every field output: (times, profiles)."""
    x0, x1, y0, y1 = run.extent()
    cy = 0.5 * (y0 + y1)
    half = PROFILE_HALF_WIDTH * run.param("skindepth0")
    times, rows = [], []
    for bp, step, time in run.stamps("fields"):
        v = run.fields(bp, ["fN_1", "fN_2"])
        y = v["X2"].astype(np.float64)
        band = np.abs(y - cy) <= half
        p = (v["fN_1"].astype(np.float64) + v["fN_2"].astype(np.float64))[:, band].mean(axis=1)
        n = p.size // PROFILE_BINS
        rows.append(p[:n * PROFILE_BINS].reshape(PROFILE_BINS, n).mean(axis=1))
        times.append(time)
    return np.array(times), np.array(rows)


def correlation(u, v):
    u, v = u - u.mean(), v - v.mean()
    d = np.sqrt((u * u).sum() * (v * v).sum())
    return float((u * v).sum() / d) if d > 0 else float("nan")


def measure(path, bp2raw, cache):
    run = Run(path, bp2raw, cache)
    _, k_th = threshold(run.param("photon_energy"), run.param("mass_ratio"))
    a = c3.rates(run)
    q = {"reconnection rate": float(a[:, 1:].mean(axis=1).mean())}
    for species, label in ((1, "electrons"), (2, "ions")):
        s = c3.spectra(run, species, k_th)
        for j, name in enumerate(c3.NAMES):
            if species == 1 and j == 4:
                continue
            q[f"{label}, {name}"] = float(s[:, 0, j].mean())
    times, prof = profiles(run)
    counters = run.counters()
    runtime = run.param("runtime")
    late = a[:, 0] >= 2.0 * runtime / 3.0
    extra = {"reconnection rate over t >= 2 runtime / 3": float(a[late, 1:].mean(axis=1).mean()),
             "primaries per species at the end": float(counters["n_1"][-1]),
             "pairs created": float(counters["pairs"][-1]),
             "largest ion K of the run": float(counters["Kmax_2"].max())}
    shutil.rmtree(run.cache, ignore_errors=True)
    return {"name": run.name, "q": q, "times": times, "profiles": prof, "extra": extra, "runtime": runtime}


def main():
    args = sys.argv[1:]
    bp2raw, cache = args[0], args[1]
    groups, key = {"off": [], "low": [], "mid": []}, None
    for a in args[2:]:
        if a in ("--off", "--low", "--mid"):
            key = a[2:]
        else:
            groups[key].append(a)
    if len(groups["off"]) != 3 or len(groups["low"]) != 3 or len(groups["mid"]) > 1:
        raise SystemExit("three runs without the policy, three with, at most one R-mid")
    runs = {k: [measure(p, bp2raw, cache) for p in v] for k, v in groups.items()}
    off, low, mid = runs["off"], runs["low"], runs["mid"]
    names = list(off[0]["q"])
    print("check 3 (stage C): the ensemble as the reference")
    print("  without the policy: " + ", ".join(r["name"] for r in off) + " (the first is the original)")
    print("  with the policy:    " + ", ".join(r["name"] for r in low) + " (the first is the original)")
    if mid:
        print("  ten times the optical depth: " + mid[0]["name"])
    for r in off + low + mid:
        print(f"  {r['name']}: " + "; ".join(f"{k} {v:.6g}" for k, v in r["extra"].items()))

    # ---- independence
    print("independence: correlation of the density of the primaries along the sheet "
          f"(|y| <= {PROFILE_HALF_WIDTH:g} skin depths, {PROFILE_BINS} bins)")
    everyone = off + low + mid
    kinds = ["off"] * len(off) + ["low"] * len(low) + ["mid"] * len(mid)
    same_kind_ok = True
    for i in range(len(everyone)):
        for j in range(i + 1, len(everyone)):
            a, b = everyone[i], everyone[j]
            n = min(len(a["times"]), len(b["times"]))
            c = np.array([correlation(a["profiles"][k], b["profiles"][k]) for k in range(n)])
            t = a["times"][:n]
            late = t >= 2.0 * a["runtime"] / 3.0
            med = float(np.median(c[late])) if late.any() else float("nan")
            below = np.where(c < INDEPENDENT_BELOW)[0]
            first = f"{t[below[0]]:.0f}" if below.size else "never"
            # last output at or above the bound: the runs are apart for good after it
            above = np.where(c >= INDEPENDENT_BELOW)[0]
            last = f"{t[above[-1]]:.0f}" if above.size else "none"
            same = kinds[i] == kinds[j] and kinds[i] != "mid"
            ok = med < INDEPENDENT_BELOW
            if same and not ok:
                same_kind_ok = False
            print(f"  {a['name']} / {b['name']} ({'same kind' if same else 'different kinds'}): median over "
                  f"t >= 2 runtime / 3 {med:+.3f} {'independent' if ok else 'NOT INDEPENDENT'}; first output "
                  f"below {INDEPENDENT_BELOW:g} at t = {first}, last at or above it at t = {last}")
            print("    by output: " + " ".join(f"{x:+.2f}" for x in c))
    if not same_kind_ok:
        print("check 3 (stage C) verdict: NO VERDICT (runs of the same kind are not independent: the "
              "scatter is not measured)")
        return 2

    # ---- scatter and the three rules
    codes = []
    print("scatter and rules (values: mean over the window of each run)")
    for name in names:
        vo = np.array([r["q"][name] for r in off])
        vl = np.array([r["q"][name] for r in low])
        so, sl = vo.std(ddof=1), vl.std(ddof=1)
        sp = np.sqrt(0.5 * (so ** 2 + sl ** 2))
        s_new = np.sqrt(0.5 * (0.5 * (vo[1] - vo[2]) ** 2 + 0.5 * (vl[1] - vl[2]) ** 2))
        scale = abs(np.concatenate([vo, vl]).mean())
        print(f"  {name}:")
        print("    without " + " ".join(f"{x:.6g}" for x in vo) + f" (mean {vo.mean():.6g}, s {so:.3g}); with "
              + " ".join(f"{x:.6g}" for x in vl) + f" (mean {vl.mean():.6g}, s {sl:.3g}); pooled s_p {sp:.3g} "
              f"({sp / scale:.1%} of the mean of the six); s_new {s_new:.3g}")
        # A
        d, unit = vl.mean() - vo.mean(), sp * np.sqrt(2.0 / 3.0)
        inside = abs(d) <= T4_FAMILY * unit
        codes.append(0 if inside else 1)
        print(f"    A  kind against kind: difference {d:+.4g} ({d / abs(vo.mean()):+.1%}); t = {d / unit:+.2f}; "
              f"bound {T4_FAMILY * unit:.4g} ({T4_FAMILY * unit / abs(vo.mean()):.1%}); at the single 5 percent "
              f"level {T4_SINGLE * unit:.4g}: {'inside the scatter' if inside else 'OUTSIDE THE SCATTER'}"
              f"{'' if abs(d) <= T4_SINGLE * unit else ' (outside at the single level)'}")
        # B
        d, unit = vl[0] - vo[0], s_new * np.sqrt(2.0)
        inside = abs(d) <= T2_FAMILY * unit
        codes.append(0 if inside else 1)
        print(f"    B  the original pair: difference {d:+.4g} ({d / abs(vo[0]):+.1%}); in units of s_new sqrt(2) "
              f"{d / unit:+.2f} (of s_p sqrt(2): {d / (sp * np.sqrt(2.0)):+.2f}); bound {T2_FAMILY * unit:.4g}; "
              f"at the single 5 percent level {T2_SINGLE * unit:.4g}: "
              f"{'inside the scatter' if inside else 'OUTSIDE THE SCATTER'}"
              f"{'' if abs(d) <= T2_SINGLE * unit else ' (outside at the single level)'}")
        # C
        if mid:
            q = mid[0]["q"][name]
            d, unit = q - vl.mean(), sp * np.sqrt(4.0 / 3.0)
            inside = abs(d) <= T4_FAMILY * unit
            codes.append(0 if inside else 1)
            print(f"    C  {mid[0]['name']} {q:.6g} against the kind with the policy: difference {d:+.4g} "
                  f"({d / abs(vl.mean()):+.1%}); t = {d / unit:+.2f}; bound {T4_FAMILY * unit:.4g} "
                  f"({T4_FAMILY * unit / abs(vl.mean()):.1%}); with the scatter of that kind alone (s {sl:.3g}, "
                  f"two degrees of freedom) t = {d / (sl * np.sqrt(4.0 / 3.0)) if sl > 0 else float('nan'):+.2f}: "
                  f"{'inside the scatter' if inside else 'OUTSIDE THE SCATTER'}"
                  f"{'' if abs(d) <= T4_SINGLE * unit else ' (outside at the single level)'}")
    print("reported without a rule (the scatter of four more numbers):")
    for key in off[0]["extra"]:
        vo = np.array([r["extra"][key] for r in off])
        vl = np.array([r["extra"][key] for r in low])
        line = (f"  {key}: without " + " ".join(f"{x:.6g}" for x in vo) + f" (s {vo.std(ddof=1):.3g}); with "
                + " ".join(f"{x:.6g}" for x in vl) + f" (s {vl.std(ddof=1):.3g})")
        if mid:
            line += f"; {mid[0]['name']} {mid[0]['extra'][key]:.6g}"
        print(line)
    n_out = codes.count(1)
    print(f"check 3 (stage C) verdict: {'NOTHING OUTSIDE THE SCATTER' if n_out == 0 else 'OUTSIDE THE SCATTER'} "
          f"({len(codes) - n_out} comparisons inside, {n_out} outside, of {len(codes)})")
    return 1 if n_out else 0


if __name__ == "__main__":
    sys.exit(main())
