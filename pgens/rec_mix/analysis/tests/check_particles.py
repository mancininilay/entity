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


def cell_and_frac(X):
    """Cell index and in-cell offset from an output position X = i + dx (unit
    cells). The output is i + dx rounded to float, so a particle with dx within
    half an ulp of 1 appears at the integer i + 1; such exact integers are
    assigned to the lower cell with dx -> 1 (a true dx = 0 has probability
    ~2^-24 per particle). Without this, one particle in ~1e5 lands in the
    neighbouring cell of its in-code one."""
    i = np.floor(X).astype(int)
    f = X - np.floor(X)
    amb = (f == 0) & (X > 0)
    i[amb] -= 1
    f = np.where(amb, 1.0, f)
    return i, f


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


def bulk_estimate(p, m, mask, w, nmin, N1, N2, periodic_x1, half=False):
    """The rec_mix box estimator, reimplemented on particles: masked K, N,
    Kb_own (variance-corrected), Kb_own_raw, Nlowocc, Klowocc and Kblowocc for
    window w. half=True: Kb_own_half (box means from the particles with
    dx1 < 1/2, weights from all particles; dx1 = X1 - floor(X1))."""
    u = np.stack([p["U1"], p["U2"], p["U3"]], -1)
    g = np.sqrt(1 + (u ** 2).sum(-1))
    b = u / g[:, None]
    i, f1 = cell_and_frac(p["X1"])
    j, _ = cell_and_frac(p["X2"])
    wt = p["W"]
    cnt = np.zeros((N1, N2)); np.add.at(cnt, (i, j), wt)
    kin = np.zeros((N1, N2)); np.add.at(kin, (i, j), wt * m * (g - 1))
    sb = [np.zeros((N1, N2)) for _ in range(3)]
    for k in range(3):
        np.add.at(sb[k], (i, j), wt * b[:, k])
    sq = np.zeros((N1, N2)); np.add.at(sq, (i, j), wt * (b ** 2).sum(-1))
    if half:
        h = f1 < 0.5
        cnth = np.zeros((N1, N2)); np.add.at(cnth, (i[h], j[h]), wt[h])
        sb = [np.zeros((N1, N2)) for _ in range(3)]
        for k in range(3):
            np.add.at(sb[k], (i[h], j[h]), wt[h] * b[h, k])
        sq = np.zeros((N1, N2)); np.add.at(sq, (i[h], j[h]), wt[h] * (b[h] ** 2).sum(-1))
        n = box_sum(cnth, w, periodic_x1)
    else:
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
    kbc = cnt[sel] * (1 / np.sqrt(1 - np.minimum(bcor, cap)) - 1)
    low = nn < nmin
    if half:
        return {"Kb_own_half": float(m * np.sum(kbc))}, (g, wt)
    return {"N": float(np.sum(cnt[mask])), "K": float(np.sum(kin[mask])),
            "Kb_own": float(m * np.sum(kbc)),
            "Kb_own_raw": float(m * np.sum(cnt[sel] * (1 / np.sqrt(1 - np.minimum(braw, cap)) - 1))),
            "Nlowocc": float(np.sum(cnt[sel][low])),
            "Klowocc": float(np.sum(kin[sel][low])),
            "Kblowocc": float(m * np.sum(kbc[low]))}, (g, wt)


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


def occupancy_scan(sim, label, populations=None, nmins=(5, 10, 20, 40), periodic_x1=False):
    """Occupancy screen against its threshold nmin, from the particle output:
    per population, the number fraction in boxes with fewer than nmin
    particles, the energy bound max(Klowocc, Kblowocc)/(K - Kb_own) and the
    reported contribution of those boxes (Klowocc - Kblowocc)/(K - Kb_own)."""
    meta, o, P, mask = _load(sim, label)
    d = meta["derived"]
    w = meta["bulk"]["window"]
    if populations is None:
        populations = {"leptons": [4, 5, 7, 8], "ions": [6, 9]} if meta["layout"] == "active" \
            else {"leptons": [4, 5, 7, 8], "tracer_3": [3], "tracer_6": [6], "tracer_9": [9]}
    out = {}
    for name, sps in populations.items():
        row = {}
        for nmin in nmins:
            t = dict(N=0.0, K=0.0, Kb=0.0, Nlo=0.0, Klo=0.0, Kblo=0.0)
            for s in sps:
                if s not in P:
                    continue
                m = meta["species"][s - 1]["mass"]
                e, _ = bulk_estimate(P[s], m, mask, w, nmin, d["N1"], d["N2"], periodic_x1)
                for k, kk in (("N", "N"), ("K", "K"), ("Kb", "Kb_own"), ("Nlo", "Nlowocc"),
                              ("Klo", "Klowocc"), ("Kblo", "Kblowocc")):
                    t[k] += e[kk]
            R = t["K"] - t["Kb"]
            row[nmin] = {"number_fraction": t["Nlo"] / t["N"] if t["N"] else None,
                         "energy_bound": max(t["Klo"], t["Kblo"]) / R if R > 0 else None,
                         "reported_contribution": (t["Klo"] - t["Kblo"]) / R if R > 0 else None}
        out[name] = row
    return out


def thinning_scan(sim, label, fractions=(1, 2, 4, 8), periodic_x1=False):
    """Finite-sample dependence of eps-tilde on the actual distribution (RV1b):
    box means from the particles with floor(k dx1) = 0 (about 1/k of them),
    weights from all particles, for each k; eps-tilde per species and the
    relative change against k = 1. A bias that depends on n shows up here."""
    meta, o, P, mask = _load(sim, label)
    d = meta["derived"]
    w, nmin = meta["bulk"]["window"], meta["bulk"]["nmin"]
    out = {}
    for s, p in P.items():
        m = meta["species"][s - 1]["mass"]
        full, _ = bulk_estimate(p, m, mask, w, nmin, d["N1"], d["N2"], periodic_x1)
        if not full["N"] > 0:
            continue
        row = {}
        frac = cell_and_frac(p["X1"])[1]
        for k in fractions:
            keep = np.floor(k * frac) == 0       # about 1/k of the particles
            kb = full["Kb_own"] if k == 1 else _kb_subset(p, keep, m, mask, w, d["N1"], d["N2"],
                                                         periodic_x1)
            row[k] = (full["K"] - kb) / full["N"]
        out[s] = {"epst": row, "change": {k: row[k] / row[1] - 1 for k in row}}
    return out


def _kb_subset(p, keep, m, mask, w, N1, N2, periodic_x1):
    u = np.stack([p["U1"], p["U2"], p["U3"]], -1)
    g = np.sqrt(1 + (u ** 2).sum(-1))
    b = u / g[:, None]
    i, _ = cell_and_frac(p["X1"])
    j, _ = cell_and_frac(p["X2"])
    wt = p["W"]
    cnt = np.zeros((N1, N2)); np.add.at(cnt, (i, j), wt)
    cs = np.zeros((N1, N2)); np.add.at(cs, (i[keep], j[keep]), wt[keep])
    sb = [np.zeros((N1, N2)) for _ in range(3)]
    for k in range(3):
        np.add.at(sb[k], (i[keep], j[keep]), wt[keep] * b[keep, k])
    sq = np.zeros((N1, N2)); np.add.at(sq, (i[keep], j[keep]), wt[keep] * (b[keep] ** 2).sum(-1))
    n = box_sum(cs, w, periodic_x1)
    B = [box_sum(x, w, periodic_x1) for x in sb]
    Q = box_sum(sq, w, periodic_x1)
    sel = mask & (cnt > 0) & (n > 0)
    nn = n[sel]
    braw = (B[0][sel] ** 2 + B[1][sel] ** 2 + B[2][sel] ** 2) / nn ** 2
    bcor = braw.copy()
    two = nn > 1.5
    s2 = (Q[sel][two] - nn[two] * braw[two]) / (nn[two] - 1)
    bcor[two] = np.clip(braw[two] - s2 / nn[two], 0, None)
    return float(m * np.sum(cnt[sel] * (1 / np.sqrt(1 - np.minimum(bcor, 1 - 1e-12)) - 1)))


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
        if "Klowocc" in o.names:  # v1.5: energy share and thinning
            ref["Klowocc"], ref["Kblowocc"] = est["Klowocc"], est["Kblowocc"]
            ref["Klowocc_w1"], ref["Kblowocc_w1"] = e1["Klowocc"], e1["Kblowocc"]
            eh, _ = bulk_estimate(p, m, mask, w, nmin, N1, N2, periodic_x1, half=True)
            ref["Kb_own_half"] = eh["Kb_own_half"]
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
