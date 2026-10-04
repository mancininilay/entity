#!/usr/bin/env python3
"""C1.16 (new, D32): the in-code bulk energies Kb_own (per-species mean 3-velocity)
and Kb_all (mass-weighted velocity of all depositing species, Tristan velx)
recomputed independently from Entity's stock moment outputs V_s (const
smoothing, order 4 = 5x5 top-hat), the raw counts `count_s` and the mask.
Requires field outputs at user-output steps, downsampling 1.
Usage: check_bulk.py <sim_dir>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p13ana import readers  # noqa: E402


def gamma_of(v1, v2, v3):
    b2 = v1.astype(np.float64) ** 2 + v2.astype(np.float64) ** 2 + v3.astype(np.float64) ** 2
    return 1.0 / np.sqrt(np.clip(1.0 - b2, 1e-12, None))


def main(sim):
    usr = os.path.join(sim, "usr")
    meta, outs = readers.read_all_recmix(usr)
    by_step = {o.step: o for o in outs}
    dep = [s["index"] for s in meta["species"] if s["deposit"] and s["mass"] > 0]
    allname = "_".join(str(s) for s in dep)
    res = []
    for fp in readers.list_entity_fields(sim):
        F = readers.read_entity_fields(fp)
        if F["step"] not in by_step:
            continue
        o = by_step[F["step"]]
        mask = F["mask"] > 0.5
        gall = gamma_of(F[f"V1_{allname}"], F[f"V2_{allname}"], F[f"V3_{allname}"])
        row = {"step": F["step"], "own": [], "all": []}
        for s in range(1, 10):
            m = meta["species"][s - 1]["mass"]
            c = F[f"count_{s}"].astype(np.float64)
            g = gamma_of(F[f"V1_{s}"], F[f"V2_{s}"], F[f"V3_{s}"])
            kb_own = m * np.sum(c[mask] * (g[mask] - 1.0))  # vs Kb_own_raw (uncorrected)
            kb_all = m * np.sum(c[mask] * (gall[mask] - 1.0))
            ko, ka = o.m("Kb_own_raw")[s - 1], o.m("Kb_all")[s - 1]
            row["own"].append(float(kb_own / ko - 1) if ko > 0 else None)
            row["all"].append(float(kb_all / ka - 1) if (np.isfinite(ka) and ka > 0) else None)
            # masked count identity: sum of count_s over the mask == N_s (masked)
        row["count_vs_N_maxdiff"] = float(max(abs(np.sum(F[f"count_{s}"][mask]) - o.m("N")[s - 1]) for s in range(1, 10)))
        res.append(row)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
