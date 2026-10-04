#!/usr/bin/env python3
"""C1.16b and C1.14b from a full particle output (stride 1) at a user-output label:
recompute, independently of the generator's kernels,
  * the own-species bulk energies Kb_own (variance-corrected), Kb_own_raw and
    Nlowocc of every species, from particle positions and momenta, the mask
    (custom field `mask`) and the box rule of the meta (window w, nmin; boxes
    truncated at non-periodic walls, wrapped in x1 while x1 is periodic);
  * the CV inventory N_cv and I_cv from the particles flagged in the CV payload.
Also: window_scan() for the window convergence of eps-tilde (IC5).
Usage: check_particles.py <sim_dir> <label> [periodic_x1]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p13ana import readers  # noqa: E402


def read_particles(path):
    from adios2 import FileReader
    out = {}
    with FileReader(path) as r:
        v = r.available_variables()
        for s in range(1, 10):
            if f"pX1_{s}" not in v:
                continue
            out[s] = {k: np.asarray(r.read(f"p{k}_{s}"), dtype=np.float64)
                      for k in ("X1", "X2", "U1", "U2", "U3", "W") if f"p{k}_{s}" in v}
            if f"pPLDI0_{s}" in v:
                out[s]["CV"] = np.asarray(r.read(f"pPLDI0_{s}"))
    return out


def box_sum(a, w, periodic_x1):
    out = np.zeros_like(a)
    n1, n2 = a.shape
    for di in range(-w, w + 1):
        for dj in range(-w, w + 1):
            sh = np.roll(a, (-di, -dj), axis=(0, 1))
            if not periodic_x1:
                if di > 0:
                    sh[n1 - di:, :] = 0
                elif di < 0:
                    sh[:-di, :] = 0
            if dj > 0:
                sh[:, n2 - dj:] = 0
            elif dj < 0:
                sh[:, :-dj] = 0
            out += sh
    return out


def bulk_estimate(p, m, mask, w, nmin, N1, N2, periodic_x1):
    """The rec_mix box estimator, reimplemented on particles: masked K, N,
    Kb_own (variance-corrected), Kb_own_raw and Nlowocc for window w."""
    u = np.stack([p["U1"], p["U2"], p["U3"]], -1)
    g = np.sqrt(1 + (u ** 2).sum(-1))
    b = u / g[:, None]
    i = np.floor(p["X1"]).astype(int)
    j = np.floor(p["X2"]).astype(int)
    wt = p["W"]
    cnt = np.zeros((N1, N2)); np.add.at(cnt, (i, j), wt)
    kin = np.zeros((N1, N2)); np.add.at(kin, (i, j), wt * m * (g - 1))
    sb = [np.zeros((N1, N2)) for _ in range(3)]
    for k in range(3):
        np.add.at(sb[k], (i, j), wt * b[:, k])
    sq = np.zeros((N1, N2)); np.add.at(sq, (i, j), wt * (b ** 2).sum(-1))
    n = box_sum(cnt, w, periodic_x1)
    B = [box_sum(x, w, periodic_x1) for x in sb]
    Q = box_sum(sq, w, periodic_x1)
    sel = mask & (cnt > 0) & (n > 0)
    nn = n[sel]
    braw = (B[0][sel] ** 2 + B[1][sel] ** 2 + B[2][sel] ** 2) / nn ** 2
    bcor = braw.copy()
    two = nn > 1.5
    s2 = (Q[sel][two] - nn[two] * braw[two]) / (nn[two] - 1)
    bcor[two] = np.clip(braw[two] - s2 / nn[two], 0, None)
    cap = 1 - 1e-12
    return {"N": float(np.sum(cnt[mask])), "K": float(np.sum(kin[mask])),
            "Kb_own": float(m * np.sum(cnt[sel] * (1 / np.sqrt(1 - np.minimum(bcor, cap)) - 1))),
            "Kb_own_raw": float(m * np.sum(cnt[sel] * (1 / np.sqrt(1 - np.minimum(braw, cap)) - 1))),
            "Nlowocc": float(np.sum(cnt[sel][nn < nmin]))}, (g, wt)


def _load(sim, label):
    usr = os.path.join(sim, "usr")
    meta, outs = readers.read_all_recmix(usr)
    o = [x for x in outs if x.step == label][0]
    P = read_particles(os.path.join(sim, "particles", f"particles.{label:08d}.bp"))
    F = readers.read_entity_fields(os.path.join(sim, "fields", f"fields.{label:08d}.bp"), ["mask"])
    return meta, o, P, F["mask"] > 0.5


def window_scan(sim, label, windows=(1, 2, 3, 4, 6), periodic_x1=False):
    """Window convergence of the corrected bulk-frame energy per particle,
    eps-tilde_s = (K_s - Kb_own,s)/N_s on the mask, and of the low-occupancy
    fraction, for each species (Codex IC5). The generator supports w <= 2
    (N_GHOSTS); larger windows are evaluated here only."""
    meta, o, P, mask = _load(sim, label)
    d = meta["derived"]
    nmin = meta["bulk"]["nmin"]
    out = {}
    for s, p in P.items():
        m = meta["species"][s - 1]["mass"]
        row = {}
        for w in windows:
            e, _ = bulk_estimate(p, m, mask, w, nmin, d["N1"], d["N2"], periodic_x1)
            if e["N"] > 0:
                row[w] = {"epst": (e["K"] - e["Kb_own"]) / e["N"],
                          "epst_raw": (e["K"] - e["Kb_own_raw"]) / e["N"],
                          "lowocc_frac": e["Nlowocc"] / e["N"], "N": e["N"]}
        out[s] = row
    return out


def main(sim, label, periodic_x1=False):
    usr = os.path.join(sim, "usr")
    meta, o, P, mask = _load(sim, label)
    d = meta["derived"]
    N1, N2 = d["N1"], d["N2"]
    w, nmin = meta["bulk"]["window"], meta["bulk"]["nmin"]
    res = {"label": label, "species": {}}
    for s, p in P.items():
        m = meta["species"][s - 1]["mass"]
        est, (g, wt) = bulk_estimate(p, m, mask, w, nmin, N1, N2, periodic_x1)
        ref = {k: est[k] for k in ("Kb_own", "Kb_own_raw", "Nlowocc")}
        if "Kb_own_w1" in o.names:  # format 3: window 1 as well
            e1, _ = bulk_estimate(p, m, mask, 1, nmin, N1, N2, periodic_x1)
            ref["Kb_own_w1"], ref["Nlowocc_w1"] = e1["Kb_own"], e1["Nlowocc"]
        code = {k: float(o.m(k)[s - 1]) for k in ref}
        rel = {k: (abs(code[k] - ref[k]) / abs(ref[k]) if ref[k] != 0 else abs(code[k]))
               for k in ref}
        e = {"ref": ref, "code": code, "rel_diff": rel}
        if "CV" in p:
            flag = p["CV"] > 0
            e["Ncv_particles"] = float(wt[flag].sum())
            e["Icv_particles"] = float(m * np.sum(wt[flag] * (g[flag] - 1)))
        res["species"][s] = e
    cv = readers.read_cvbudget(usr, meta=meta)
    k = list(cv["step"]).index(label)
    for s in res["species"]:
        res["species"][s]["Ncv_record"] = float(cv["Ncv"][k, s - 1])
        res["species"][s]["Icv_record"] = float(cv["Icv"][k, s - 1])
    print(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), len(sys.argv) > 3 and sys.argv[3] == "periodic")
