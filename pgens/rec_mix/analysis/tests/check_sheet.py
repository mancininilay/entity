#!/usr/bin/env python3
"""C1.2 (sheet current balance and normal stress) and C1.3 (neutrality) at the
first output of a rec_mix run with field output smoothing 0, downsampling 1.
Usage: check_sheet.py <sim_dir>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p13ana import readers  # noqa: E402


def main(sim):
    meta = readers.read_meta(os.path.join(sim, "usr"))
    d, inp = meta["derived"], meta["inputs"]
    fp = readers.list_entity_fields(sim)[0]
    F = readers.read_entity_fields(fp, ["E3", "B1", "B2", "T22_1_2_3", "Charge"])
    N1, N2 = d["N1"], d["N2"]
    a, y0, x0 = inp["cs_width"], d["y0"], d["x0"]
    dt = d["dt"]
    X1, X2 = np.meshgrid(F["X1"], F["X2"], indexing="ij")
    # analytic curl B_3 = -dB1/dx2 at cell centres (B1 = tanh((x2-y0)/a))
    curlB3 = -(1.0 / a) / np.cosh((X2 - y0) / a) ** 2
    rows = np.abs(F["X2"] - y0) < 2 * a
    r2 = (X1 - x0) ** 2 + (X2 - y0) ** 2
    dip = 1 - np.exp(-r2 / (inp["dip_width"] * a) ** 2)
    far = dip > 0.9
    sel = far & rows[None, :]
    # balance: E3 after one step relative to dt * curlB3 (0 if J balances curl B,
    # 2 if the current had the wrong sign)
    bal = F["E3"][sel].sum() / (dt * curlB3[sel].sum())
    res = {"step": F["step"], "C1.2_balance_E3_over_dt_curlB": float(bal)}
    # midplane sign of the current: E3 grows opposite to J at step 1 only if
    # unbalanced; report the sign of J3 implied (D29: J3 < 0 at the midplane)
    # normal stress of the sheet species vs the shape-weighted expectation
    shape = dip / np.cosh((X2 - y0) / a) ** 2
    P0 = F["T22_1_2_3"][:, rows].sum() / shape[:, rows].sum()
    PB = 0.5 * d["sigma0"]  # B0^2/8pi in units n0 me c^2
    exp_legacy = PB / d["Dmass"]
    res["C1.2_T22_peak"] = float(P0)
    res["C1.2_T22_over_PB"] = float(P0 / PB)
    # transverse stress of a population drifting along x3: n_lab kT / Gamma_d
    G = 1.0 / np.sqrt(1.0 - d["beta_d"] ** 2)
    # legacy (p1.1): T_cs = sigma0/(2 eta D)  -> P22/PB = 1/(D Gamma_d)
    # balanced (Gamma_d included, Codex A3): T_cs = sigma0 Gamma_d/(2 eta) -> 1
    res["C1.2_expected_over_PB"] = 1.0 / (d["Dmass"] * G) if meta["sheet_init"] == "legacy" else 1.0
    # neutrality over 10x10 blocks (Charge is normalised by n0 e)
    q = F["Charge"][: N1 // 10 * 10, : N2 // 10 * 10]
    qb = q.reshape(N1 // 10, 10, N2 // 10, 10).mean(axis=(1, 3))
    res["C1.3_max_block_charge"] = float(np.abs(qb).max())
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
