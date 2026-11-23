#!/usr/bin/env python3
"""The comparisons of the resolution pair, R0 to R3 of its pre-registration (Claude 1, 2026-11-23).

    compare_pair.py <bp2raw> <cache> <output directory> --present <run> <run> <run>
                    --fine <run> <run> <run> [--long <run>]

Every run directory must hold `analysis/core.npz` (series/core_series.py). Writes
<output directory>/compare.txt and compare.json.

Scalars. The ten scalars of check 3 by the registered function of stage C
(`scripts/check3_ensemble.measure`, unchanged: the reconnection rate over t >= runtime / 3, the
spectral scalars of the electrons and the ions over t >= 2 runtime / 3), one number per run.

Independence (as check 3 of stage C): for every pair of runs of one grid, the median over the
field outputs with t >= 2 runtime / 3 of the correlation of the sheet profiles is below 0.5.
If a pair of one grid fails, R1 gives no verdict.

R1, the two grids: for each scalar, the difference of the means of the grids, d, and the pooled
standard deviation s_p (four degrees of freedom). A difference is flagged when
|d| > 5.598 s_p sqrt(2 / 3) (the family of ten, Bonferroni within the rule); 2.776 for a single
scalar is printed beside it. Reported for every scalar: d in percent of the mean of the present
grid, its nominal 95 percent interval d +- 2.776 s_p sqrt(2 / 3) (independent equal-variance
Gaussian errors, three runs per grid), and the critical difference. Exit code of R1: 0 nothing
flagged, 1 something flagged, 2 no verdict.

R0, the loaded regime, on each grid: the outputs at which the pairs are at least half the
primaries inside 5 skin depths of the densest block of the pairs, in a plasmoid of at least 25
skin depths (core series), summed over the runs of the grid. At least five: the regime was
reached on that grid. Otherwise R2 for that grid reads "no test of the loaded regime".

R2, the cores: `scripts/check6_position.py --frame` (6a, 6b by position in the frame of the pairs,
6c) for every run of the fine grid against the three runs of the present grid as members, and for
every run of the present grid against the three of the fine grid; the same with the energies in
the lab frame, reported. Exit codes collected.

R3, stationarity (reported, no criterion): the inflow rate and the mean K of the ions in the
windows 1000 <= t < 2000 and 2000 <= t <= 3000 of every run of 3000 time units, and in blocks of
1000 time units from t = 1000 in the long run; the standard deviation between the windows of one
run against that between the runs of one grid.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SC = HERE.parent / "scripts"
sys.path.insert(0, str(SC))
import check3_compare as c3  # noqa: E402
import check3_ensemble as ens  # noqa: E402
from bhio import Run, threshold  # noqa: E402

T_FAMILY, T_SINGLE = 5.5976, 2.7764
LOADED_RATIO, LOADED_SIZE, LOADED_MIN = 0.5, 25.0, 5


def window_scalars(path, bp2raw, cache, windows):
    """Inflow rate (the estimator of check 3, quadrants averaged) and mean K of the ions per window."""
    run = Run(path, bp2raw, cache)
    x0, x1, y0, y1 = run.extent()
    cx, cy, hx, hy = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    rate = {w: [] for w in windows}
    for bp, step, time in run.stamps("fields"):
        ws = [w for w in windows if w[0] <= time < w[1] or (time >= w[1] - 1e-6 and w[1] == windows[-1][1])]
        if not ws:
            continue
        v = run.fields(bp, ["fE1", "fE3", "fB1", "fB2", "fB3"])
        x, y = v["X1"] - cx, v["X2"] - cy
        e1, e3 = v["fE1"].astype(np.float64), v["fE3"].astype(np.float64)
        b1, b2, b3 = (v[k].astype(np.float64) for k in ("fB1", "fB2", "fB3"))
        vy = (e3 * b1 - e1 * b3) / (b1 * b1 + b2 * b2 + b3 * b3)
        quad = []
        for ysign in (1.0, -1.0):
            my = (ysign * y >= 0.4 * hy) & (ysign * y <= 0.6 * hy)
            for xsign in (-1.0, 1.0):
                mx = (xsign * x > 0.0) & (np.abs(x) <= 0.8 * hx)
                quad.append(-ysign * vy[np.ix_(mx, my)].mean())
        for w in ws:
            rate[w].append(float(np.mean(quad)))
    _, k_th = threshold(run.param("photon_energy"), run.param("mass_ratio"))
    sigma = run.sigma()
    kmean = {w: [] for w in windows}
    for bp, step, time in run.stamps("spectra"):
        ws = [w for w in windows if w[0] <= time < w[1] or (time >= w[1] - 1e-6 and w[1] == windows[-1][1])]
        if not ws:
            continue
        v = run.read(bp, ["sEbn", "sN_2"])
        n = v["sN_2"].astype(np.float64).reshape(2, -1)
        s = c3.scalars(n[0] + n[1], v["sEbn"].astype(np.float64), sigma, k_th)
        for w in ws:
            kmean[w].append(s[0])
    return {f"{w[0]:g}-{w[1]:g}": {"rate": float(np.mean(rate[w])) if rate[w] else None,
                                   "ions, mean K": float(np.mean(kmean[w])) if kmean[w] else None,
                                   "outputs": [len(rate[w]), len(kmean[w])]} for w in windows}


def core_table(run_dir):
    z = np.load(Path(run_dir) / "analysis" / "core.npz")
    names = [str(n) for n in z["names"]]
    return {n: z["table"][:, i] for i, n in enumerate(names)}


def loaded_count(run_dir):
    c = core_table(run_dir)
    m = (c["radius"] == 5.0) & (c["size"] >= LOADED_SIZE) & (c["n_pair"] >= LOADED_RATIO * c["n_prim"])
    return int(m.sum()), [float(t) for t in c["time"][m]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bp2raw")
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("--present", nargs=3, required=True)
    ap.add_argument("--fine", nargs=3, required=True)
    ap.add_argument("--long")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    text, rec = [], {}
    say = text.append

    # ---- scalars and independence
    meas = {}
    for kind in ("present", "fine"):
        meas[kind] = [ens.measure(p, a.bp2raw, a.cache) for p in getattr(a, kind)]
    say("resolution pair: R0 to R3; present grid " + ", ".join(m["name"] for m in meas["present"])
        + "; fine grid " + ", ".join(m["name"] for m in meas["fine"]) + (f"; long run {Path(a.long).name}" if a.long else ""))
    independent = True
    rec["independence"] = {}
    for kind in ("present", "fine"):
        ms = meas[kind]
        for i in range(3):
            for j in range(i + 1, 3):
                ti, pi = ms[i]["times"], ms[i]["profiles"]
                tj, pj = ms[j]["times"], ms[j]["profiles"]
                late = [k for k, t in enumerate(ti) if t >= 2.0 * ms[i]["runtime"] / 3.0 and k < len(tj)]
                cor = [ens.correlation(pi[k], pj[k]) for k in late]
                med = float(np.median(cor)) if cor else float("nan")
                ok = med < 0.5
                independent &= ok
                rec["independence"][f"{ms[i]['name']}/{ms[j]['name']}"] = med
                say(f"  independence {ms[i]['name']} / {ms[j]['name']}: median correlation over t >= 2 runtime / 3 "
                    f"{med:+.3f} ({'independent' if ok else 'NOT independent'})")

    # ---- R1
    say("R1, the two grids (rule A of check 3 with the grids as the two kinds):")
    flagged = 0
    rec["R1"] = {}
    for key in meas["present"][0]["q"]:
        p = np.array([m["q"][key] for m in meas["present"]])
        f = np.array([m["q"][key] for m in meas["fine"]])
        sp = float(np.sqrt((p.var(ddof=1) + f.var(ddof=1)) / 2.0))
        d = float(f.mean() - p.mean())
        half_f, half_s = T_FAMILY * sp * np.sqrt(2.0 / 3.0), T_SINGLE * sp * np.sqrt(2.0 / 3.0)
        out_f = abs(d) > half_f
        flagged += out_f
        base = abs(p.mean())
        rec["R1"][key] = {"present": p.tolist(), "fine": f.tolist(), "s_p": sp, "d": d, "critical_family": half_f,
                          "critical_single": half_s, "flagged": bool(out_f)}
        say(f"  {key}: present {', '.join(f'{x:.4g}' for x in p)}; fine {', '.join(f'{x:.4g}' for x in f)}; "
            f"d {100 * d / base:+.1f}% of the present mean, nominal 95% interval [{100 * (d - half_s) / base:+.1f}, "
            f"{100 * (d + half_s) / base:+.1f}]%; critical difference {100 * half_f / base:.1f}% (family), "
            f"{100 * half_s / base:.1f}% (single); {'FLAGGED' if out_f else 'not flagged'}")
    code_r1 = 2 if not independent else (1 if flagged else 0)
    say(f"R1 verdict: {['nothing flagged', 'FLAGGED', 'NO VERDICT (independence screen failed)'][code_r1]} "
        f"({flagged} of {len(rec['R1'])} scalars flagged)")
    rec["R1_code"] = code_r1

    # ---- R0
    rec["R0"] = {}
    loaded = {}
    for kind in ("present", "fine"):
        n_tot, detail = 0, {}
        for p in getattr(a, kind):
            n, ts = loaded_count(p)
            n_tot += n
            detail[Path(p).name] = ts
        loaded[kind] = n_tot >= LOADED_MIN
        rec["R0"][kind] = {"outputs": n_tot, "reached": loaded[kind], "times": detail}
        say(f"R0 {kind} grid: {n_tot} outputs with pairs >= {LOADED_RATIO:g} of the primaries inside 5 d_e in a plasmoid "
            f">= {LOADED_SIZE:g} d_e: {'reached' if loaded[kind] else 'NOT reached (R2: no test of the loaded regime)'}")

    # ---- R2
    rec["R2"] = {}
    for kind, other in (("fine", "present"), ("present", "fine")):
        refs = [str(Path(p) / "analysis" / "core.npz") for p in getattr(a, other)]
        for p in getattr(a, kind):
            for frame in (True, False):
                cmd = [sys.executable, "-B", str(SC / "check6_position.py"), "--half-length", "500", "--test",
                       str(Path(p) / "analysis" / "core.npz"), "--ref"] + refs + ["--test-dir", p, "--ref-dirs"] \
                    + list(getattr(a, other)) + (["--frame"] if frame else [])
                r = subprocess.run(cmd, capture_output=True, text=True)
                tag = f"{Path(p).name}-against-{other}-{'frame' if frame else 'lab'}"
                (out / f"R2-{tag}.txt").write_text(r.stdout + r.stderr)
                rec["R2"][tag] = r.returncode
                say(f"R2 {tag}: exit {r.returncode} ({['pass', 'fail', 'no verdict'][r.returncode] if r.returncode in (0, 1, 2) else 'error'})"
                    + ("" if loaded[kind] or not frame else "; the loaded regime was not reached on this grid"))

    # ---- R3
    say("R3, stationarity (reported):")
    rec["R3"] = {}
    for kind in ("present", "fine"):
        for p in getattr(a, kind):
            w = window_scalars(p, a.bp2raw, a.cache, [(1000.0, 2000.0), (2000.0, 3000.0)])
            rec["R3"][Path(p).name] = w
            say(f"  {Path(p).name}: " + "; ".join(f"{k}: rate {v['rate']:.4f}, ions mean K {v['ions, mean K']:.3f}"
                                               for k, v in w.items() if v["rate"] is not None))
    if a.long:
        blocks = [(t, t + 1000.0) for t in np.arange(1000.0, 9000.0, 1000.0)]
        w = window_scalars(a.long, a.bp2raw, a.cache, blocks)
        rec["R3"][Path(a.long).name] = w
        r_ = [v["rate"] for v in w.values() if v["rate"] is not None]
        k_ = [v["ions, mean K"] for v in w.values() if v["ions, mean K"] is not None]
        say(f"  {Path(a.long).name}: blocks of 1000 from t = 1000: rate " + ", ".join(f"{x:.4f}" for x in r_)
            + "; ions mean K " + ", ".join(f"{x:.3f}" for x in k_))
        late_p = [rec["R3"][Path(p).name]["2000-3000"] for p in a.present]
        sd_members = (float(np.std([v["rate"] for v in late_p], ddof=1)), float(np.std([v["ions, mean K"] for v in late_p], ddof=1)))
        sd_blocks = (float(np.std(r_[1:], ddof=1)) if len(r_) > 2 else float("nan"),
                     float(np.std(k_[1:], ddof=1)) if len(k_) > 2 else float("nan"))
        say(f"  standard deviation of the rate and of the ions' mean K: between the three present-grid runs in 2000 to 3000 "
            f"{sd_members[0]:.4f}, {sd_members[1]:.3f}; between the blocks of the long run from t = 2000 {sd_blocks[0]:.4f}, "
            f"{sd_blocks[1]:.3f}")
        rec["R3_sd"] = {"members_2000_3000": sd_members, "long_blocks_from_2000": sd_blocks}

    (out / "compare.txt").write_text("\n".join(text) + "\n")
    (out / "compare.json").write_text(json.dumps(rec, indent=1))
    print("\n".join(text))
    return code_r1


if __name__ == "__main__":
    sys.exit(main())
