#!/usr/bin/env python3
"""C1 checks on a small rec_mix run (cloud CPU). Usage: check_run.py <sim_dir>

Checks (tech spec V.2):
  C1.1  initial counts and kinetic energy per particle (unmasked, first output)
  C1.11 sum(hist) + Nskip = N_mask exactly; histogram vs direct energy
  C1.11c mixing output vs mixing recomputed from N_4_5_6, N_7_8_9 (needs
        smoothing order 0 and downsampling 1 in the field output)
  C1.15 midline rows == field output E, B at the same step (exact)
  C1.14 CV closure
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p13ana import cvbudget, partition, readers  # noqa: E402


def main(sim):
    usr = os.path.join(sim, "usr")
    meta, outs = readers.read_all_recmix(usr)
    res = {}
    names = meta["moments"]
    # ---------------- C1.11 histogram sums --------------------------------
    worst = 0.0
    erel = []
    for o in outs:
        N = o.m("N")
        H = o.hist.sum(axis=1)
        sk = o.m("Nskip")
        worst = max(worst, float(np.max(np.abs(H + sk - N))))
        eb = readers.bin_centres(meta)
        K = o.m("K")
        mass = np.array([s["mass"] for s in meta["species"]])
        Kh = mass * (o.hist * eb).sum(axis=1)
        sel = K > 0
        if sel.any():
            erel.append(np.max(np.abs(Kh[sel] / K[sel] - 1)))
    res["C1.11_hist_sum_max_abs_diff"] = worst
    res["C1.11_hist_vs_direct_energy_max_rel"] = float(np.max(erel)) if erel else None
    # ---------------- C1.1 initial state -----------------------------------
    o0 = outs[0]
    res["C1.1_first_output_step"] = o0.step
    res["C1.1_Ntot"] = o0.m("Ntot").tolist()
    with np.errstate(invalid="ignore", divide="ignore"):
        res["C1.1_Ktot_per_particle"] = (o0.m("Ktot") / o0.m("Ntot")).tolist()
    # ---------------- partition time series --------------------------------
    if meta["layout"] == "active":
        ser_d = partition.series(meta, outs, partition.active_partition,
                                 ["f_i", "eps_e", "eps_i", "epst_i_own", "epst_i_all",
                                  "epst_i_status", "i_low_occupancy_fraction", "N_pm", "N_i"])
        ser_h = partition.series(meta, outs, partition.active_partition, ["f_i", "eps_e", "eps_i"],
                                 estimator="hist")
        res["partition_last"] = {k: (str(ser_d[k][-1]) if k.endswith("status") else float(ser_d[k][-1]))
                                 for k in ser_d if k not in ("time", "step")}
        res["partition_last_hist"] = {k: float(ser_h[k][-1]) for k in ("f_i", "eps_e", "eps_i")}
    # ---------------- C1.15 midline vs field output -------------------------
    fl = readers.list_entity_fields(sim)
    mls = {readers._step_of(p): p for p in readers.list_midlines(usr)}
    c115 = []
    for fp in fl:
        F = readers.read_entity_fields(fp, ["E1", "E2", "E3", "B1", "B2", "B3"])
        if F["step"] not in mls or F["E1"].shape[0] != meta["derived"]["N1"]:
            continue
        ml = readers.read_midline(mls[F["step"]])
        d = 0.0
        for r in range(ml.rows.shape[0]):
            j = ml.j_first + r
            for c, n in enumerate(["E1", "E2", "E3", "B1", "B2", "B3"]):
                d = max(d, float(np.max(np.abs(ml.rows[r, c] - F[n][:, j].astype(np.float32)))))
        c115.append((F["step"], d))
    res["C1.15_midline_vs_fields_max_abs_diff"] = c115
    # ---------------- C1.11c mixing vs N outputs ----------------------------
    c11c = []
    for fp in fl:
        try:
            F = readers.read_entity_fields(fp, ["N_4_5_6", "N_7_8_9", "mixing"])
        except Exception:
            try:
                F = readers.read_entity_fields(fp, ["N_4_5", "N_7_8", "mixing"])
                F["N_4_5_6"], F["N_7_8_9"] = F["N_4_5"], F["N_7_8"]
            except Exception:
                continue
        ppc0 = meta["derived"]["ppc0"]
        L = np.rint(F["N_4_5_6"].astype(np.float64) * ppc0)
        U = np.rint(F["N_7_8_9"].astype(np.float64) * ppc0)
        with np.errstate(invalid="ignore", divide="ignore"):
            M = np.where(L + U > 0, 2 * np.minimum(L, U) / (L + U), 0.0)
        mask_py = M > meta["mask"]["threshold"]
        mask_cc = F["mixing"] > meta["mask"]["threshold"]
        c11c.append((F["step"], int(np.sum(mask_py != mask_cc)), float(np.max(np.abs(M - F["mixing"])))))
    res["C1.11c_mask_mismatch_cells_and_max_dM"] = c11c
    # ---------------- C1.14 CV budget ----------------------------------------
    cv = readers.read_cvbudget(usr)
    cl = cvbudget.closure(cv, meta=meta)
    res["C1.14_closure"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cl.items()}
    iv = cvbudget.intervals(cv, meta=meta)  # depositing species only (IC4)
    res["C1.14_intervals"] = [
        {"steps": [int(a), int(b)], "W_phys": float(w),
         "r_F_rel": (float(rf / wj) if "r_F" in iv else None),
         "r_C_rel": (float(rc / w) if "r_C" in iv else None),
         "r_total_rel": float(rt / w)}
        for a, b, w, rt, rf, rc, wj in zip(iv["step0"], iv["step1"], iv["W_phys"], iv["r_total"],
                                          iv.get("r_F", iv["r_total"]), iv.get("r_C", iv["r_total"]),
                                          iv.get("Wj", iv["W_phys"]))]
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
