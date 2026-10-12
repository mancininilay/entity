#!/usr/bin/env python3
"""C1.16b and C1.14b from the particle state at a user-output label:
recompute, independently of the generator's kernels (a different numerical
path by the same author, not independent authorship),
  * the own-species bulk energies Kb_own (variance-corrected), Kb_own_raw,
    Nlowocc, Klowocc, Kblowocc (both windows) and Kb_own_half of every species
    from the particle cells and momenta, the mask (custom field `mask`) and the
    box rule of the meta (window w, nmin; boxes truncated at non-periodic walls,
    wrapped in x1 while x1 is periodic);
  * the CV inventory N_cv and I_cv from the particles flagged in the CV payload.
Also window_scan(), occupancy_scan() and position_subsampling_scan().

Particle source (Codex A0-2). The exact source is the generator's validation
dump (setup.validation_dump; readers.read_vdump): integer cell and float
offset as the kernels use them. Entity's particle output holds only the
rounded position X = float(i) + dx, which does not identify the cell when X is
an integer (dx just below 1 in cell i - 1 or dx ~ 0 in cell i give the same X)
and does not identify the dx1 < 1/2 subsample when X - floor(X) = 1/2. With
that source the cell is taken as floor(X) (a convention, not a recovery), the
ambiguous particles are counted and reported, and exact comparisons are not
claimed while any is present.
Usage: check_particles.py <sim_dir> <label> [periodic_x1]"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p13ana import readers  # noqa: E402


def read_particles(path):
    """Entity particle output (rounded positions; see the module docstring)."""
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
            for k in ("IDX", "RNK"):
                if f"p{k}_{s}" in v:
                    out[s][k] = np.asarray(r.read(f"p{k}_{s}"))
    return out


def classify_positions(X, subsample=0.5):
    """Cell, in-cell offset and rounding ambiguity of rounded output positions
    X = float(i) + dx (unit cells, float rounding to nearest). Since integers and
    i + 1/2 are representable and rounding is monotone, a value strictly inside
    (i, i + 1) comes from cell i, and one strictly below (above) i + subsample
    from dx below (at or above) subsample. Ambiguous are exactly: X integer (cell
    X with dx ~ 0, or cell X - 1 with dx ~ 1), and X - floor(X) == subsample (the
    dx < subsample selection). Returns (cell = floor(X), frac, amb_cell, amb_sub);
    the cell of an ambiguous particle is a convention, not a recovery (A0-2)."""
    X = np.asarray(X, dtype=np.float64)
    i = np.floor(X).astype(np.int64)
    f = X - i
    amb_cell = f == 0.0
    amb_sub = (f == subsample) | amb_cell
    return i, f, amb_cell, amb_sub


def particles_from_bp(P):
    """Rounded-position source: cells by floor (convention) and ambiguity flags."""
    out = {}
    for s, p in P.items():
        i, f1, a1, h1 = classify_positions(p["X1"])
        j, _, a2, _ = classify_positions(p["X2"])
        q = {k: p[k] for k in ("U1", "U2", "U3", "W") if k in p}
        q.update({"i": i, "j": j, "dx1": f1, "amb_cell": a1 | a2, "amb_half": h1})
        for k in ("CV", "IDX", "RNK"):
            if k in p:
                q[k] = p[k]
        out[s] = q
    return out


def particles_from_vdump(V):
    """Exact source: the generator's cells and float32 offsets (no ambiguity)."""
    out = {}
    for s, d in V["species"].items():
        n = d["i1"].size
        q = {"i": d["i1"], "j": d["i2"], "dx1": d["dx1"].astype(np.float64),
             "dx1_f32": d["dx1"], "U1": d["u1"].astype(np.float64), "U2": d["u2"].astype(np.float64),
             "U3": d["u3"].astype(np.float64), "W": d["w"].astype(np.float64), "CV": d["cv"],
             "amb_cell": np.zeros(n, bool), "amb_half": np.zeros(n, bool), "RANK": d["rank"]}
        if "idx" in d:
            q["IDX"], q["RNK"] = d["idx"], d["dom"]
        out[s] = q
    return out


def half_select(p):
    """The kernel's subsample: dx1 < 1/2 evaluated on the stored float offset."""
    if "dx1_f32" in p:
        return p["dx1_f32"] < np.float32(0.5)
    return p["dx1"] < 0.5


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
    """The rec_mix box estimator, reimplemented on particles (dict with cells
    i, j, offset dx1, U1-3, W): masked K, N, Kb_own (variance-corrected),
    Kb_own_raw, Nlowocc, Klowocc and Kblowocc for window w. half=True:
    Kb_own_half (box means from the particles with dx1 < 1/2, weights from all
    particles). Klowocc and Kblowocc are the recorded low-occupancy
    contribution: |Klowocc - Kblowocc| <= max(Klowocc, Kblowocc) bounds its
    magnitude, not its error against a physical random energy (Codex A0-3a)."""
    u = np.stack([p["U1"], p["U2"], p["U3"]], -1)
    g = np.sqrt(1 + (u ** 2).sum(-1))
    b = u / g[:, None]
    i, j = p["i"], p["j"]
    wt = p["W"]
    cnt = np.zeros((N1, N2)); np.add.at(cnt, (i, j), wt)
    kin = np.zeros((N1, N2)); np.add.at(kin, (i, j), wt * m * (g - 1))
    sb = [np.zeros((N1, N2)) for _ in range(3)]
    for k in range(3):
        np.add.at(sb[k], (i, j), wt * b[:, k])
    sq = np.zeros((N1, N2)); np.add.at(sq, (i, j), wt * (b ** 2).sum(-1))
    if half:
        h = half_select(p)
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


def _load(sim, label, source="auto"):
    """Meta, user output, particles and mask at a label. source: "vdump" (exact,
    the generator's validation dump), "bp" (Entity particle output, rounded
    positions; ambiguity flagged) or "auto" (vdump if present)."""
    usr = os.path.join(sim, "usr")
    meta, outs = readers.read_all_recmix(usr)
    o = [x for x in outs if x.step == label][0]
    if source == "auto":
        source = "vdump" if label in readers.list_vdump_labels(usr) else "bp"
    if source == "vdump":
        P = particles_from_vdump(readers.read_vdump(usr, label))
    else:
        P = particles_from_bp(read_particles(os.path.join(sim, "particles", f"particles.{label:08d}.bp")))
    F = readers.read_entity_fields(os.path.join(sim, "fields", f"fields.{label:08d}.bp"), ["mask"])
    return meta, o, P, F["mask"] > 0.5, source


def ambiguity_report(sim, label):
    """A0-2 witness on real output: classify the rounded Entity particle
    positions at a label and, where the validation dump exists, count how many
    of the ambiguous ones the floor convention puts in the wrong cell or the
    wrong dx1 < 1/2 subsample."""
    usr = os.path.join(sim, "usr")
    B = read_particles(os.path.join(sim, "particles", f"particles.{label:08d}.bp"))
    have_v = label in readers.list_vdump_labels(usr)
    V = readers.read_vdump(usr, label) if have_v else None
    out = {}
    for s, p in B.items():
        i, f1, a1, h1 = classify_positions(p["X1"])
        j, _, a2, _ = classify_positions(p["X2"])
        r = {"N": int(i.size), "ambiguous_cell": int((a1 | a2).sum()), "ambiguous_half": int(h1.sum())}
        if have_v:
            v = V["species"][s]
            # exact positions of the dump through Entity's conversion
            # (kernels/prtls_to_phys.hpp: x = x_min + (float(i_local) + dx), float)
            off1 = np.array([rk["off1"] for rk in V["ranks"]])
            off2 = np.array([rk["off2"] for rk in V["ranks"]])
            rank_index = {rk["rank"]: k for k, rk in enumerate(V["ranks"])}
            ri = np.array([rank_index[x] for x in v["rank"]], dtype=np.int64) if v["rank"].size else \
                np.zeros(0, np.int64)
            o1, o2 = off1[ri], off2[ri]
            X1 = (o1.astype(np.float32) + ((v["i1"] - o1).astype(np.float32) + v["dx1"]).astype(np.float32))
            X2 = (o2.astype(np.float32) + ((v["i2"] - o2).astype(np.float32) + v["dx2"]).astype(np.float32))
            # multiset comparison of (X1, X2, U1, U2, U3) between the BP output and the emulation
            a = np.stack([p["X1"], p["X2"], p["U1"], p["U2"], p["U3"]], 1)
            b = np.stack([X1.astype(np.float64), X2.astype(np.float64), v["u1"].astype(np.float64),
                          v["u2"].astype(np.float64), v["u3"].astype(np.float64)], 1)
            same = a.shape == b.shape and np.array_equal(a[np.lexsort(a.T[::-1])], b[np.lexsort(b.T[::-1])])
            r["bp_equals_converted_dump"] = bool(same)
            # wrong assignments of the floor convention, from the dump's own positions
            ci, _, _, _ = classify_positions(X1.astype(np.float64))
            cj, _, _, _ = classify_positions(X2.astype(np.float64))
            r["floor_wrong_cell"] = int(((ci != v["i1"]) | (cj != v["i2"])).sum())
            hx = (X1.astype(np.float64) - ci) < 0.5
            r["floor_wrong_half"] = int((hx != (v["dx1"] < np.float32(0.5))).sum())
        out[s] = r
    return out


def window_scan(sim, label, windows=(1, 2, 3, 4, 6), periodic_x1=False):
    """Window convergence of the corrected bulk-frame energy per particle,
    eps-tilde_s = (K_s - Kb_own,s)/N_s on the mask, and of the low-occupancy
    fraction, for each species (Codex IC5). The generator supports w <= 2
    (N_GHOSTS); larger windows are evaluated here only."""
    meta, o, P, mask, _ = _load(sim, label)
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
    meta, o, P, mask, _ = _load(sim, label)
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


def position_subsampling_scan(sim, label, fractions=(1, 2, 4, 8), periodic_x1=False):
    """Position-subsampling sensitivity of eps-tilde (formerly "thinning";
    Codex A0-3b): box means from the particles with floor(k dx1) = 0 (about 1/k
    of them, by in-cell position), weights from all particles, for each k;
    eps-tilde per species and the relative change against k = 1.
    Limitation: dx1 can be correlated with velocity in the evolved plasma (sub-
    cell phase-space structure, e.g. two streams at different dx1), so a change
    can reflect the selection of a different velocity population as well as a
    finite-sample bias; one split is not an error bar. Statistically valid
    subsampling independent of phase space is an R1 item (D36), not done here."""
    meta, o, P, mask, _ = _load(sim, label)
    d = meta["derived"]
    w, nmin = meta["bulk"]["window"], meta["bulk"]["nmin"]
    out = {}
    for s, p in P.items():
        m = meta["species"][s - 1]["mass"]
        full, _ = bulk_estimate(p, m, mask, w, nmin, d["N1"], d["N2"], periodic_x1)
        if not full["N"] > 0:
            continue
        row = {}
        for k in fractions:
            keep = np.floor(k * p["dx1"]) == 0       # about 1/k of the particles
            if k == 2:
                keep = half_select(p)                # the kernel's dx1 < 1/2 on the stored offset
            kb = full["Kb_own"] if k == 1 else _kb_subset(p, keep, m, mask, w, d["N1"], d["N2"],
                                                         periodic_x1)
            row[k] = (full["K"] - kb) / full["N"]
        out[s] = {"epst": row, "change": {k: row[k] / row[1] - 1 for k in row},
                  "ambiguous_particles": int(np.sum(p["amb_cell"] | p["amb_half"]))}
    return out


thinning_scan = position_subsampling_scan  # former name (v1.5)


def _kb_subset(p, keep, m, mask, w, N1, N2, periodic_x1):
    u = np.stack([p["U1"], p["U2"], p["U3"]], -1)
    g = np.sqrt(1 + (u ** 2).sum(-1))
    b = u / g[:, None]
    i, j = p["i"], p["j"]
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


def main(sim, label, periodic_x1=False, source="auto"):
    """Exact comparisons need the exact source; with rounded positions the
    result carries the ambiguity counts and `exact_source` False (A0-2)."""
    usr = os.path.join(sim, "usr")
    meta, o, P, mask, src = _load(sim, label, source)
    d = meta["derived"]
    N1, N2 = d["N1"], d["N2"]
    w, nmin = meta["bulk"]["window"], meta["bulk"]["nmin"]
    res = {"label": label, "source": src, "exact_source": src == "vdump", "species": {}}
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
        e = {"ref": ref, "code": code, "rel_diff": rel,
             "ambiguous_cell": int(p["amb_cell"].sum()), "ambiguous_half": int(p["amb_half"].sum())}
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
