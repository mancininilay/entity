#!/usr/bin/env python3
"""Does the decomposition shift the dynamics? The registered comparison (PREREGISTRATION-D.md).

    verdict.py <directory with <label>.scalars.json and <label>.interface.json>
               --single-low L L L --single-off L L L [--mid L] --m4 L [--f22 L L] [--f41 L L]

Groups of runs:
  S    one card, with the policy, this optical depth     (--single-low: R-low, R-low-b, R-low-c)
  O    one card, without the policy                      (--single-off: R-off, R-off-b, R-off-c)
  mid  one card, ten times the optical depth             (shown, never pooled)
  F22  four cards, [2, 2]: the run of M4 (--m4, which raised the question) and the new ones (--f22)
  F41  four cards, [4, 1]: the new ones (--f41)
Without --f22 and --f41 the script prints the existing runs, their scatter and what the
registered comparisons will be able to see (the gate before the runs); no verdict.

Scatter of a scalar: the pooled standard deviation within the groups O, S, F22, F41
(S, F22, F41 for the numbers of pairs, which O does not have). Comparison of two sets of
runs A and B: D = mean(A) - mean(B), standard error s sqrt(1 / n_A + 1 / n_B), Student t
with the degrees of freedom of s. For every scalar: the interval at 95 percent for this
scalar alone and the interval at 95 percent for its family (Bonferroni).
  ESTABLISHED          the family interval excludes zero
  indicated            only the single interval excludes zero
  not established      the single interval includes zero
  bound                |D| + half-width of the single interval: a shift larger than this
                       is excluded at 95 percent, for this scalar alone
Exit 0: no shift established in the confirmatory comparison; 1: established; 2: no verdict.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

TAU = 0.014333
INDEPENDENT_BELOW = 0.5
# (scalar, where it is stored, stated size as a fraction of the mean of S)
LATE = [("ions, mean K", "late", 0.25), ("ions, K 99 percent", "late", 0.25),
        ("ions, K 99.9 percent", "late", 0.25), ("reconnection rate", "late", 0.10),
        ("pairs created", "reported", 0.15)]
EARLY = [("pairs created up to t = 750", "early", 0.10), ("ions, mean K at t = 750", "early", 0.10),
         ("reconnection rate over 300 <= t <= 700", "early", 0.10)]
REPORTED = [("reconnection rate over t >= 2 runtime / 3", "reported"), ("ions, fraction above threshold", "late"),
            ("ions, slope sigma/2..2 sigma", "late"), ("electrons, mean K", "late"),
            ("electrons, K 99 percent", "late"), ("electrons, K 99.9 percent", "late"),
            ("electrons, slope sigma/2..2 sigma", "late"), ("primaries per species at the end", "reported"),
            ("largest ion K of the run", "reported")]
SEAM_FIELDS = [("fN_2", "ion density"), ("fB1", "Bx"), ("fB2", "By"), ("fE3", "Ez"), ("fJ3", "Jz")]
SEAM_TOLERANCE = 0.05
X_PLANES, Y_PLANE = ["x = -L/4", "x = 0", "x = +L/4"], "y = 0"


# ---------------------------------------------------------------- Student t
def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, nu):
    p = 0.5 * _betai(0.5 * nu, 0.5, nu / (nu + t * t))
    return 1.0 - p if t > 0 else p


def t_quantile(p, nu):
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def self_test():
    for p, nu, want in ((0.975, 4, 2.7764), (0.9975, 4, 5.5976), (0.975, 2, 4.3027), (0.9975, 2, 14.0890),
                        (0.975, 7, 2.3646), (0.975, 5, 2.5706)):
        got = t_quantile(p, nu)
        if abs(got - want) > 2e-4 * want:
            raise SystemExit(f"Student t: quantile {p} with {nu} degrees of freedom is {got}, expected {want}")


# ------------------------------------------------------------------- runs
def load(directory, label):
    with open(Path(directory) / f"{label}.scalars.json") as f:
        r = json.load(f)
    r["label"] = label
    p = Path(directory) / f"{label}.interface.json"
    r["interface"] = json.load(open(p)) if p.exists() else None
    return r


def value(run, name, where):
    v = run[where][name]
    if name.startswith("pairs") and run["emitting"]:
        v *= TAU / run["optical_depth"]          # per unit of optical depth, at the optical depth of R-low
    return v


def pooled(groups):
    """Pooled standard deviation within the groups, and its degrees of freedom."""
    ss, df = 0.0, 0
    for g in groups:
        g = np.asarray(g, dtype=float)
        if g.size >= 2:
            ss += float(((g - g.mean()) ** 2).sum())
            df += g.size - 1
    return (math.sqrt(ss / df), df) if df > 0 else (float("nan"), 0)


def compare(text, a, b, s, df, family, scale, stated):
    """One comparison of two sets of values; returns (established, indicated, bound as a fraction)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0 or df == 0:
        return None
    d = a.mean() - b.mean()
    se = s * math.sqrt(1.0 / a.size + 1.0 / b.size)
    h1 = t_quantile(0.975, df) * se
    hf = t_quantile(1.0 - 0.025 / family, df) * se
    established, indicated = abs(d) > hf, abs(d) > h1
    word = "ESTABLISHED" if established else ("indicated at the single level, not established" if indicated
                                              else "not established")
    bound = (abs(d) + h1) / scale
    print(f"    {text} ({a.size} against {b.size}): D = {d:+.5g} ({d / scale:+.1%}); standard error "
          f"{se / scale:.1%}; t = {d / se:+.2f} ({df} d.o.f.); single interval [{(d - h1) / scale:+.1%}, "
          f"{(d + h1) / scale:+.1%}]; family interval [{(d - hf) / scale:+.1%}, {(d + hf) / scale:+.1%}]: "
          f"{word}; a shift above {bound:.1%} is excluded"
          + (f" ({'below' if bound < stated else 'not below'} the stated size {stated:.0%})" if stated else ""))
    return established, indicated, bound


def correlation(u, v):
    u, v = np.asarray(u) - np.mean(u), np.asarray(v) - np.mean(v)
    d = math.sqrt(float((u * u).sum() * (v * v).sum()))
    return float((u * v).sum() / d) if d > 0 else float("nan")


def independence(groups):
    """The registered condition of check 3 for every pair of runs; False if two runs of one group fail it."""
    print("independence: median over t >= 2 runtime / 3 of the correlation of the density of the primaries "
          "along the sheet (bound 0.5)")
    runs = [(g, r) for g, rs in groups.items() for r in rs]
    ok = True
    for i in range(len(runs)):
        row = []
        for j in range(len(runs)):
            if j <= i:
                continue
            (ga, a), (gb, b) = runs[i], runs[j]
            n = min(len(a["profile_times"]), len(b["profile_times"]))
            late = [k for k in range(n) if a["profile_times"][k] >= 2.0 * a["runtime"] / 3.0]
            med = float(np.median([correlation(a["profiles"][k], b["profiles"][k]) for k in late])) if late else float("nan")
            bad = not med < INDEPENDENT_BELOW
            if bad and ga == gb:
                ok = False
            row.append(f"{b['label']} {med:+.2f}" + (" NOT INDEPENDENT" if bad else ""))
        if row:
            print(f"  {runs[i][1]['label']}: " + ", ".join(row))
    print(f"  runs of one group independent: {'yes' if ok else 'NO'}")
    return ok


# ------------------------------------------------------------- interfaces
def seam_medians(run):
    """{(plane, field): (median over the outputs after the first of the ratio to the two adjacent planes,
    of the ratio to the planes 4 to 7 cells away, of the signed jump in units of its standard error)}"""
    out = {}
    if run["interface"] is None:
        return out
    cells = {"x": run["interface"]["outputs"][0]["cells_in_x_window"], "y": run["interface"]["outputs"][0]["cells_in_y_window"]}
    t0 = min(o["t"] for o in run["interface"]["outputs"])
    acc = {}
    for row in run["interface"]["rows"]:
        if row["t"] <= t0:
            continue
        rms, jump = np.array(row["rms"]), np.array(row["jump"])
        near = 0.5 * (rms[7] + rms[9])
        far = float(np.median(rms[[1, 2, 3, 4, 12, 13, 14, 15]]))
        n = cells[row["plane"][0]]
        z = jump[8] / (rms[8] / math.sqrt(n)) if rms[8] > 0 else float("nan")
        acc.setdefault((row["plane"], row["field"]), []).append(
            (rms[8] / near if near > 0 else float("nan"), rms[8] / far if far > 0 else float("nan"), z))
    for key, rows in acc.items():
        a = np.array(rows)
        out[key] = (float(np.nanmedian(a[:, 0])), float(np.nanmedian(a[:, 1])), float(np.nanmean(a[:, 2])) * math.sqrt(len(rows)))
    return out


def is_boundary(run, plane):
    nx, ny = run["decomposition"]
    if plane == Y_PLANE:
        return ny == 2
    return (plane == "x = 0" and nx in (2, 4)) or (plane != "x = 0" and nx == 4)


def interfaces(groups):
    runs = [r for rs in groups.values() for r in rs if r["interface"] is not None]
    if not runs:
        print("interfaces: no interface file")
        return None
    med = {r["label"]: seam_medians(r) for r in runs}
    print("interfaces: median over the outputs after the first of rms(across the plane) / mean rms(across the two "
          "adjacent planes); in brackets the same against the planes 4 to 7 cells away; * marks a rank boundary of that run")
    for plane in X_PLANES + [Y_PLANE]:
        print(f"  plane {plane}")
        print("    " + f"{'run':16s}" + "".join(f"{name:>24s}" for _, name in SEAM_FIELDS))
        for r in runs:
            star = "*" if is_boundary(r, plane) else " "
            print("    " + f"{r['label'] + star:16s}" + "".join(
                f"{med[r['label']][(plane, f)][0]:14.4f} [{med[r['label']][(plane, f)][1]:6.3f}]" for f, _ in SEAM_FIELDS))
    # rule for the planes across the layer: an isolated jump at a rank boundary
    flagged_x, control_x = [], []
    for r in runs:
        for plane in X_PLANES:
            for f, name in SEAM_FIELDS:
                m = med[r["label"]][(plane, f)][0]
                if abs(m - 1.0) > SEAM_TOLERANCE:
                    (flagged_x if is_boundary(r, plane) else control_x).append(f"{r['label']} {plane} {name} {m:.4f}")
    n_b = sum(1 for r in runs for p in X_PLANES if is_boundary(r, p)) * len(SEAM_FIELDS)
    n_c = sum(1 for r in runs for p in X_PLANES if not is_boundary(r, p)) * len(SEAM_FIELDS)
    print(f"  planes across the layer (x): ratio outside 1 +- {SEAM_TOLERANCE} at a rank boundary: {len(flagged_x)} of {n_b}"
          + (": " + "; ".join(flagged_x) if flagged_x else "") + f"; where the plane is not a rank boundary: "
          f"{len(control_x)} of {n_c}" + (": " + "; ".join(control_x) if control_x else ""))
    # rule for the plane of the layer: the runs with the cut against the runs without, by rank
    cut = [r for r in runs if is_boundary(r, Y_PLANE)]
    free = [r for r in runs if not is_boundary(r, Y_PLANE) and r not in groups.get("mid", [])]
    flagged_y = []
    if cut and free:
        print(f"  plane of the layer (y = 0): the {len(cut)} runs with the cut ({', '.join(r['label'] for r in cut)}) "
              f"against the {len(free)} without ({', '.join(r['label'] for r in free)})")
        for f, name in SEAM_FIELDS:
            a = [med[r["label"]][(Y_PLANE, f)][0] for r in cut]
            b = [med[r["label"]][(Y_PLANE, f)][0] for r in free]
            out = min(a) > max(b) or max(a) < min(b)
            if out:
                flagged_y.append(name)
            four = [med[r["label"]][(Y_PLANE, f)][0] for r in free if r["decomposition"] != [1, 1]]
            print(f"    {name}: with the cut " + " ".join(f"{x:.3f}" for x in a) + "; without, one card "
                  + " ".join(f"{med[r['label']][(Y_PLANE, f)][0]:.3f}" for r in free if r["decomposition"] == [1, 1])
                  + ("; without, four cards " + " ".join(f"{x:.3f}" for x in four) if four else "")
                  + f": {'ALL OUTSIDE THE RANGE of the runs without the cut' if out else 'not separated'}")
        chance = 2.0 / math.comb(len(cut) + len(free), len(cut))
        print(f"    chance of 'all outside the range' for one field if the cut does nothing: {chance:.3f}")
    print("  signed jump across the plane, mean over the outputs, in units of its standard error (reported, no rule):")
    for plane in X_PLANES + [Y_PLANE]:
        for r in runs:
            if is_boundary(r, plane):
                print(f"    {r['label']} {plane}: " + ", ".join(f"{name} {med[r['label']][(plane, f)][2]:+.1f}" for f, name in SEAM_FIELDS))
    return flagged_x, control_x, flagged_y


# ------------------------------------------------------------------- main
def main():
    self_test()
    directory, args = sys.argv[1], sys.argv[2:]
    labels, key = {"single-low": [], "single-off": [], "mid": [], "m4": [], "f22": [], "f41": []}, None
    for a in args:
        if a.startswith("--"):
            key = a[2:]
        else:
            labels[key].append(a)
    if len(labels["single-low"]) != 3 or len(labels["single-off"]) != 3 or len(labels["m4"]) != 1:
        raise SystemExit("three single-card runs with the policy, three without, and the run of M4")
    g = {k: [load(directory, x) for x in v] for k, v in labels.items()}
    S, O, mid, m4, n22, f41 = g["single-low"], g["single-off"], g["mid"], g["m4"], g["f22"], g["f41"]
    f22 = m4 + n22
    new = n22 + f41
    for r, tiling in [(x, [1, 1]) for x in S + O + mid] + [(x, [2, 2]) for x in f22] + [(x, [4, 1]) for x in f41]:
        if r["decomposition"] != tiling:
            raise SystemExit(f"{r['label']}: decomposition {r['decomposition']}, expected {tiling}")
    for r in S + f22 + f41:
        if not r["emitting"]:
            raise SystemExit(f"{r['label']}: the policy is off")
    print("the decomposition question: runs")
    for name, rs in (("S, one card, policy", S), ("O, one card, no policy", O), ("mid, one card, ten times the optical depth", mid),
                     ("F22, four cards [2, 2]", f22), ("F41, four cards [4, 1]", f41)):
        for r in rs:
            print(f"  {name}: {r['label']} ({r['directory']}): overdensity {r['overdensity']:g}, optical depth "
                  f"{r['optical_depth']:g}, to step {r['steps']:.0f}, t = {r['end_time']:.1f}; largest counters "
                  + ", ".join(f"{k} {v:.0f}" for k, v in r["counters_max"].items()))
    complete = all(r["end_time"] >= r["runtime"] - 1.0 for r in S + O + mid + f22 + f41)
    if not complete:
        print("  A RUN DID NOT REACH ITS RUNTIME: its scalars are partial")
    independent = independence({"S": S, "O": O, "mid": mid, "F22": f22, "F41": f41})

    def table(name, where):
        def row(rs):
            return " ".join(f"{value(r, name, where):.5g}" for r in rs) if rs else "-"
        return f"S {row(S)}; O {row(O)}; mid {row(mid)}; F22 {row(f22)} (the first is the run of M4); F41 {row(f41)}"

    codes = {}
    for title, family in (("late", LATE), ("early", EARLY)):
        print(f"{title} scalars (family of {len(family)}: single level 5 percent, family level {0.05 / len(family):.3%} per scalar)")
        est, ind, bounds, under = [], [], [], []
        for name, where, stated in family:
            v = lambda rs: [value(r, name, where) for r in rs]
            has_o = not name.startswith("pairs")
            s, df = pooled([v(S), v(f22), v(f41)] + ([v(O)] if has_o else []))
            scale = abs(np.mean(v(S)))
            print(f"  {name}: {table(name, where)}")
            print(f"    scatter within the groups {s:.4g} ({s / scale:.1%} of the mean of S, {df} degrees of freedom"
                  + ("" if has_o else "; the runs without the policy have no pairs") + ")")
            if not new:
                n_f = len(family)
                df_w = df + 3 if not has_o else df + 3      # two more in F22, one in F41
                for text, na, nb in (("four new runs against S", 4, 3), ("all five on four cards against S", 5, 3),
                                     ("[4, 1] against the two new [2, 2]", 2, 2), ("[4, 1] against the three [2, 2]", 2, 3)):
                    se = s * math.sqrt(1.0 / na + 1.0 / nb)
                    print(f"    if the scatter stays as it is: {text}: standard error {se / scale:.1%}; half-width of the single "
                          f"interval {t_quantile(0.975, df_w) * se / scale:.1%}, of the family interval "
                          f"{t_quantile(1.0 - 0.025 / n_f, df_w) * se / scale:.1%} ({df_w} d.o.f.)")
                compare("the run of M4 against S (existing runs; this difference raised the question)", v(m4), v(S), s, df,
                        len(family), scale, 0)
                continue
            c1 = compare("C1 confirmatory: the four new runs against S", v(new), v(S), s, df, len(family), scale, stated)
            compare("C2 all five runs on four cards against S (includes the run that raised the question)", v(f22 + f41), v(S),
                    s, df, len(family), scale, stated)
            compare("C3 the two new [2, 2] against S", v(n22), v(S), s, df, len(family), scale, stated)
            compare("C3' the three [2, 2] against S", v(f22), v(S), s, df, len(family), scale, stated)
            compare("C4 [4, 1] against S", v(f41), v(S), s, df, len(family), scale, stated)
            c5 = compare("C5 confirmatory: [4, 1] against the two new [2, 2]", v(f41), v(n22), s, df, len(family), scale, stated)
            compare("C5' [4, 1] against the three [2, 2]", v(f41), v(f22), s, df, len(family), scale, stated)
            if has_o:
                compare("context: the four new runs against the six single-card runs (S and O; assumes that the pairs change "
                        "nothing at this optical depth)", v(new), v(S + O), s, df, len(family), scale, 0)
                compare("context: all five against the six single-card runs", v(f22 + f41), v(S + O), s, df, len(family), scale, 0)
            if c1:
                est.append((name, c1[0], c5[0] if c5 else False))
                ind.append((name, c1[1], c5[1] if c5 else False))
                bounds.append((name, c1[2], c5[2] if c5 else float("nan")))
                under.append(c1[2] < stated)
        if new:
            e1 = [n for n, a, b in est if a]
            i1 = [n for n, a, b in ind if a and n not in e1]
            e5 = [n for n, a, b in est if b]
            i5 = [n for n, a, b in ind if b and n not in e5]
            if not independent or not complete:
                word = "NO VERDICT (independence or an incomplete run)"
                codes[title] = 2
            elif e1:
                word = "A SHIFT IS ESTABLISHED in " + ", ".join(e1)
                codes[title] = 1
            elif all(under):
                word = "NO SHIFT ESTABLISHED, and a shift above the stated sizes is EXCLUDED for every scalar"
                codes[title] = 0
            else:
                word = "NO SHIFT ESTABLISHED; not excluded at the stated sizes"
                codes[title] = 0
            print(f"  verdict, {title} scalars, four cards against one (C1): {word}"
                  + (f"; indicated at the single level only: {', '.join(i1)}" if i1 else "")
                  + "; bounds: " + ", ".join(f"{n} {b:.1%}" for n, b, _ in bounds))
            print(f"  verdict, {title} scalars, [4, 1] against [2, 2] (C5): "
                  + ("A DIFFERENCE IS ESTABLISHED in " + ", ".join(e5) if e5 else "no difference established")
                  + (f"; indicated at the single level only: {', '.join(i5)}" if i5 else "")
                  + "; bounds: " + ", ".join(f"{n} {b:.1%}" for n, _, b in bounds))
    print("reported without a rule:")
    for name, where in REPORTED:
        print(f"  {name}: {table(name, where)}")
    print("series (bh_counters.txt): time; pairs created; mean K of the ions")
    for r in S + O + mid + f22 + f41:
        s = r["series"]
        print(f"  {r['label']:16s} pairs " + " ".join(f"{x:9.0f}" for x in s["pairs"]))
        print(f"  {'':16s} K     " + " ".join(f"{x:9.3f}" for x in s["ions, mean K"]))
    print(f"  {'':16s} t     " + " ".join(f"{x:9.0f}" for x in S[0]["series"]["time"]))
    seams = interfaces({"S": S, "O": O, "mid": mid, "F22": f22, "F41": f41})
    if seams and new:
        fx, cx, fy = seams
        print("verdict, interfaces: planes across the layer: "
              + ("ISOLATED JUMP at a rank boundary" if fx else "no isolated jump at a rank boundary")
              + (" (the rule is not sharp: the same excess occurs where the plane is not a rank boundary)" if cx else "")
              + "; plane of the layer: "
              + ("the runs with the cut are ALL OUTSIDE THE RANGE of the runs without it in " + ", ".join(fy) if fy
                 else "the runs with the cut are not separated from the runs without it"))
    if not new:
        print("no new run given: no verdict")
        return 0
    return max(codes.values())


if __name__ == "__main__":
    sys.exit(main())
