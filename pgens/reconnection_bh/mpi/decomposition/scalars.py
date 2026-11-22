#!/usr/bin/env python3
"""One run -> one JSON file with what the comparison of the decompositions needs.

    scalars.py <bp2raw> <cache> <run directory> <output.json>

Late scalars: the ten scalars of check 3 and its four reported numbers, by the
registered function of stage C (check3_ensemble.measure, unchanged), with the
density profiles along the sheet for the independence condition.

Early scalars (before the boundaries open at t = 750, while the runs still
share the first chain of plasmoids):
  pairs created up to t = 750                 (bh_counters.txt, last line with time <= 750)
  mean kinetic energy of the ions at t = 750  (K_2 / w_2 of the same line)
  reconnection rate over 300 <= t <= 700      (the estimator of check 3 on the field
                                               outputs of that window)
The copy of the rate estimator made here for the other window is checked
against the registered function on the registered window (t >= runtime / 3):
the two means must agree to 1e-9, or the script stops.

Series (reported): pairs created, ions present, mean K of the ions and of the
leptons of the pairs at fixed times, from bh_counters.txt.
"""
import json
import re
import sys

import numpy as np

import check3_compare as c3
import check3_ensemble as ens
from bhio import Run

EARLY_T = 750.0
EARLY_RATE = (300.0, 700.0)
SERIES_T = [250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0, 2250.0, 2500.0, 2750.0, 3000.0]


def rate_of_output(run, bp):
    """The inflow rate of one field output: the lines of check3_compare.rates, for one file."""
    x0, x1, y0, y1 = run.extent()
    cx, cy, hx, hy = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0), 0.5 * (y1 - y0)
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
    return float(np.mean(quad))


def main():
    bp2raw, cache, path, out = sys.argv[1:5]
    run = Run(path, bp2raw, cache)
    runtime = run.param("runtime")
    stamps = run.stamps("fields")
    early = [rate_of_output(run, bp) for bp, step, time in stamps if EARLY_RATE[0] - 1.0 <= time <= EARLY_RATE[1] + 1.0]
    late = [rate_of_output(run, bp) for bp, step, time in stamps if time >= runtime / 3.0]

    c = run.counters()
    t = c["time"]
    i = int(np.searchsorted(t, EARLY_T, side="right")) - 1
    emitting = run.emitting()
    early_scalars = {
        "pairs created up to t = 750": float(c["pairs"][i]),
        "ions, mean K at t = 750": float(c["K_2"][i] / c["w_2"][i]),
        "reconnection rate over 300 <= t <= 700": float(np.mean(early)),
    }
    series = {"time": [], "pairs": [], "ions": [], "ions, mean K": [], "pair leptons, mean K": []}
    for ts in SERIES_T:
        k = int(np.searchsorted(t, ts, side="right")) - 1
        w = c["w_3"][k] + c["w_4"][k]
        series["time"].append(float(t[k]))
        series["pairs"].append(float(c["pairs"][k]))
        series["ions"].append(float(c["n_2"][k]))
        series["ions, mean K"].append(float(c["K_2"][k] / c["w_2"][k]))
        series["pair leptons, mean K"].append(float((c["K_3"][k] + c["K_4"][k]) / w) if w > 0 else None)
    zeros = {k: float(c[k].max()) for k in ("above_tables", "probability_exceeded", "loss_bound_skipped", "capacity_dropped")}

    m = ens.measure(path, bp2raw, cache)            # the registered function; it removes the cache of the run
    registered = m["q"]["reconnection rate"]
    if abs(np.mean(late) - registered) > 1e-9 * abs(registered):
        raise SystemExit(f"{run.name}: the copy of the rate estimator gives {np.mean(late):.12g} on the registered "
                         f"window, the registered function {registered:.12g}")

    tiling = re.search(r"^\s*decomposition\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", run.input, re.M)
    record = {
        "name": m["name"], "directory": str(run.dir.name),
        "decomposition": [int(tiling.group(1)), int(tiling.group(2))] if tiling else [1, 1],
        "overdensity": run.param("cs_overdensity"),
        "optical_depth": run.param("optical_depth") if emitting else 0.0,
        "emitting": emitting, "runtime": runtime,
        "steps": float(c["steps"][-1]), "end_time": float(t[-1]),
        "late": m["q"], "reported": m["extra"], "early": early_scalars,
        "early_time_used": float(t[i]), "early_rate_outputs": len(early),
        "series": series, "counters_max": zeros,
        "profile_times": [float(x) for x in m["times"]],
        "profiles": [[float(x) for x in row] for row in m["profiles"]],
    }
    with open(out, "w") as f:
        json.dump(record, f)
    print(f"{m['name']}: " + "; ".join(f"{k} {v:.6g}" for k, v in {**m['q'], **m['extra'], **early_scalars}.items()))


if __name__ == "__main__":
    main()
