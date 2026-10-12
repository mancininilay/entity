#!/usr/bin/env python3
"""C1.12: particle IDs (Entity tracking: species counter at creation and the
creating domain) are unique and stable across sorting, MPI exchange and
restart (tech spec V.2). Data: the generator's validation dumps (exact state,
all ranks; readers.read_vdump), which carry the IDs of every alive particle.

Metrics (explicit, all exact counts):
  unique      per label and species, no (idx, dom) pair occurs twice;
  stable      for every ID present at two consecutive labels of the same run,
              the displacement (minimum image in x1, which only weakens the
              test) is at most c dt (label difference) cells: an ID moved to a
              different particle by sorting or exchange would almost always
              violate it; reported: the number of violations and the largest
              displacement over the bound;
  migrated    particles whose current domain (the dump's domain index) differs
              from their creating domain (> 0 shows that MPI exchange was
              exercised);
  restart     with run A (uninterrupted) and run B (resumed from A's checkpoint
              at label L0): B's IDs at its first label that existed at L0 are
              a subset of A's IDs at L0 and satisfy the displacement bound from
              A at L0; B's IDs created after the restart do not reuse any ID
              that A had at any label <= L0 (the species counter was restored),
              per (species, domain) they are above the largest ID A had then.
Statistical agreement of A and B after the restart is not claimed here (the
device RNG state is not restored; restart_stat covers C1.7), and nothing is
bitwise.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p13ana import readers  # noqa: E402


def _ids(sp):
    return sp["idx"].astype(np.uint64) * np.uint64(1 << 20) + sp["dom"].astype(np.uint64)


def _pos(sp):
    return sp["i1"] + sp["dx1"].astype(np.float64), sp["i2"] + sp["dx2"].astype(np.float64)


def _domain_of(V, i1, i2):
    """Rank owning each global cell (from the dump headers)."""
    r = np.full(i1.size, -1, np.int64)
    for rk in V["ranks"]:
        sel = (i1 >= rk["off1"]) & (i1 < rk["off1"] + rk["n1"]) & (i2 >= rk["off2"]) & (i2 < rk["off2"] + rk["n2"])
        r[sel] = rk["rank"]
    return r


def unique(V):
    out = {}
    for s, sp in V["species"].items():
        if "idx" not in sp:
            out[s] = None
            continue
        ids = _ids(sp)
        out[s] = {"n": int(ids.size), "duplicates": int(ids.size - np.unique(ids).size),
                  "migrated": int(np.sum(sp["dom_now"] != sp["dom"].astype(np.int64)))}
    return out


def displacement(VA, VB, N1, bound):
    """For IDs present in both dumps: violations of |dx| <= bound."""
    out = {}
    for s in VA["species"]:
        a, b = VA["species"][s], VB["species"][s]
        if "idx" not in a or a["idx"].size == 0 or b["idx"].size == 0:
            out[s] = {"common": 0, "violations": 0, "max_over_bound": 0.0}
            continue
        ia, ib = _ids(a), _ids(b)
        common, ka, kb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
        xa, ya = _pos(a)
        xb, yb = _pos(b)
        dx = xb[kb] - xa[ka]
        dx = dx - N1 * np.round(dx / N1)
        dy = yb[kb] - ya[ka]
        d = np.sqrt(dx ** 2 + dy ** 2)
        out[s] = {"common": int(common.size), "violations": int(np.sum(d > bound * (1 + 1e-6) + 1e-4)),
                  "max_over_bound": float(d.max() / bound) if d.size else 0.0}
    return out


def restart(usrA, usrB, L0, LB, N1, bound):
    """IDs across a restart from A's checkpoint at L0 (B's first label LB)."""
    A0 = readers.read_vdump(usrA, L0)
    B1 = readers.read_vdump(usrB, LB)
    earlier = [readers.read_vdump(usrA, L) for L in readers.list_vdump_labels(usrA) if L <= L0]
    out = {}
    for s in A0["species"]:
        a, b = A0["species"][s], B1["species"][s]
        if "idx" not in a:
            out[s] = None
            continue
        ida, idb = _ids(a), _ids(b)
        old_all = np.unique(np.concatenate([_ids(V["species"][s]) for V in earlier]))
        # per (domain) largest idx A ever had up to L0
        mx = {}
        for V in earlier:
            sp = V["species"][s]
            for dmn in np.unique(sp["dom"]):
                m = int(sp["idx"][sp["dom"] == dmn].max())
                mx[int(dmn)] = max(mx.get(int(dmn), -1), m)
        new = ~np.isin(idb, ida)
        reused = np.isin(idb[new], old_all)
        below = np.array([int(i) <= mx.get(int(dm), -1) for i, dm in zip(b["idx"][new], b["dom"][new])],
                         dtype=bool) if new.any() else np.zeros(0, bool)
        out[s] = {"B_first_label_ids": int(idb.size), "survivors_from_L0": int((~new).sum()),
                  "created_after_restart": int(new.sum()), "reused_old_ids": int(reused.sum()),
                  "new_ids_not_above_previous_max": int(below.sum())}
    out["displacement_L0_to_B"] = displacement(A0, B1, N1, bound)
    return out


def main(simA, simB, L0, labels_A, labels_B, N1, dt):
    usrA, usrB = os.path.join(simA, "usr"), os.path.join(simB, "usr")
    res = {"A": {}, "B": {}, "stable_A": {}, "stable_B": {}}
    VA = {L: readers.read_vdump(usrA, L) for L in labels_A}
    VB = {L: readers.read_vdump(usrB, L) for L in labels_B}
    for L, V in VA.items():
        res["A"][L] = unique(V)
    for L, V in VB.items():
        res["B"][L] = unique(V)
    for X, VV, key in ((labels_A, VA, "stable_A"), (labels_B, VB, "stable_B")):
        for L1, L2 in zip(X[:-1], X[1:]):
            res[key][f"{L1}-{L2}"] = displacement(VV[L1], VV[L2], N1, (L2 - L1) * dt)
    res["restart"] = restart(usrA, usrB, L0, labels_B[0], N1, (labels_B[0] - L0) * dt)
    return res


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2], int(sys.argv[3]),
                          [int(x) for x in sys.argv[4].split(",")], [int(x) for x in sys.argv[5].split(",")],
                          int(sys.argv[6]), float(sys.argv[7])), indent=1))
