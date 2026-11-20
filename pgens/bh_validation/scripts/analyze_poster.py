#!/usr/bin/env python3
"""V5, the poster test: <K>(t) against the published-formula trajectory.

    analyze_poster.py <run dir> <fig3_analytic_data.csv> [--t-end 0.79]

Registered: initial Gamma_p = 300, eps0 = 0.05, 1e4 emitters, to 0.79 initial
cooling times; maximum residual of <K> to the published-formula trajectory
(column K_published_over_K0) <= 0.1 %. The run's records are matched to the
rows of the file by time (same time step and record interval as the file).
"""
import argparse
import csv
import os
import re
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("reference")
    ap.add_argument("--t-end", type=float, default=0.79)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    text = open(os.path.join(args.run_dir, "input.toml")).read()
    get = lambda k: float(re.search(rf"^\s*{k}\s*=\s*(.+)$", text, re.M).group(1))
    K0 = get("mass_ratio") * (get("gamma_p") - 1.0)
    rows = []
    for line in open(os.path.join(args.run_dir, "run.log")):
        if line.startswith("BHV step"):
            t = line.split()
            rows.append(dict(zip(t[1::2], (float(x) for x in t[2::2]))))
    dt = (rows[1]["time"] - rows[0]["time"]) / (rows[1]["step"] - rows[0]["step"])
    # reference rows by the number of steps (the file's time step is 0.5; the
    # run's time step differs from it by the rounding of a single-precision
    # CFL number only)
    ref = {}
    for r in csv.DictReader(open(args.reference)):
        ref[int(round(float(r["time"]) / 0.5))] = (float(r["t_over_tcool0"]), float(r["K_published_over_K0"]),
                                                   float(r["K_Born_table_over_K0"]))
    if abs(dt - 0.5) > 1.0e-6:
        print(f"FAIL V5: the time step of the run ({dt}) is not the reference's 0.5")
        return 1
    tag = args.tag or os.path.basename(os.path.normpath(args.run_dir))
    worst = worst_born = 0.0
    t_worst = 0.0
    n = 0
    last = None
    for r in rows:
        n_steps = int(round(r["step"] + 1.0))
        if n_steps not in ref:
            continue
        tc, k_pub, k_born = ref[n_steps]
        if tc > args.t_end + 1e-9:
            continue
        K_ref = r.get("K_ref", K0)
        k = (K_ref + r["ion_dK"] / r["ion_w"]) / K0
        res = abs(k / k_pub - 1.0)
        if res > worst:
            worst, t_worst = res, tc
        worst_born = max(worst_born, abs(k / k_born - 1.0))
        n += 1
        last = (tc, k, k_pub)
    if n == 0:
        print(f"FAIL V5 {tag}: no record matches the reference times (dt = {dt})")
        return 1
    ok = worst <= 1.0e-3 and last[0] >= args.t_end - 0.01
    print(f"{'PASS' if ok else 'FAIL'} V5 {tag}: {n} records to t/t_cool0 = {last[0]:.4f} (<K>/K0 = {last[1]:.5f}, published {last[2]:.5f}); "
          f"maximum residual to the published-formula trajectory {100 * worst:.4f} % at t/t_cool0 = {t_worst:.4f} (limit 0.1 %); "
          f"to the Born-table trajectory of the same file {100 * worst_born:.4f} %; pairs per emitter {rows[-1]['pairs'] / rows[-1]['ion_w']:.2f}; "
          f"bounded {rows[-1]['bounded']:.0f}, above {rows[-1]['above']:.0f}, dropped {rows[-1]['dropped']:.0f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
