"""Control-volume energy budget (P1.3 tech spec III.13, D28).

Per species s and interval between consecutive records of one segment:

    W_s   = Delta I_s - (inE_s - outE_s)       (electric work inside the CV)
    field = Delta U_EM + int(Poynting out) dt  (should equal -sum_s W_s)

Records with segment_start = True (first post-step of a process, i.e. the run
start and every restart) open a new segment; intervals never span segments.
"""

from __future__ import annotations

import numpy as np


def depositing(meta: dict) -> list[int]:
    return [s["index"] for s in meta["species"] if s["deposit"]]


def _species(species, meta):
    """The species that exchange energy with the fields (Codex IC4): an explicit
    list, or the depositing species of the meta. No default: passive tracers
    must never enter a physical residual by omission."""
    if species is not None:
        return list(species)
    if meta is not None:
        return depositing(meta)
    raise ValueError("cvbudget: pass `meta` or an explicit `species` list (the depositing "
                     "species); passive tracers must not enter the physical residual")


def intervals(cv: dict, species: list[int] | None = None, meta: dict | None = None) -> dict:
    """Per interval (consecutive records of one segment):

    * W        [nint, nspec]: particle-side work per species (all species, for
               diagnostics, tracers included);
    * W_phys   [nint]: sum of W over the depositing species;
    * dU, flux: field energy change and integrated outward Poynting flux;
    * Wj       (format 2+): field-side work of the (filtered) current.

    Residuals (Codex A3 notation), all over the depositing species:
        r_F     = dU + flux + Wj           (field side)
        r_C     = W_phys - Wj              (particle-field coupling)
        r_total = W_phys + dU + flux = r_F + r_C."""
    sel = [k - 1 for k in _species(species, meta)]
    seg = np.cumsum(cv["segment_start"].astype(int))
    same = seg[1:] == seg[:-1]
    i1 = np.nonzero(same)[0] + 1
    i0 = i1 - 1
    dI = cv["Icv"][i1] - cv["Icv"][i0]
    W = dI - (cv["inE"][i1] - cv["outE"][i1])
    Wp = W[:, sel].sum(axis=1)
    dU = cv["U_EM"][i1] - cv["U_EM"][i0]
    F = cv["flux_int"][i1]
    out = {
        "step0": cv["step"][i0], "step1": cv["step"][i1],
        "t0": cv["time"][i0], "t1": cv["time"][i1],
        "species": [k + 1 for k in sel],
        "W": W, "W_phys": Wp,
        "dU": dU, "flux": F,
        "r_total": Wp + dU + F,
        "residual": Wp + dU + F,  # alias of r_total (physical species only)
    }
    if "jwork_int" in cv:
        Wj = cv["jwork_int"][i1]
        out.update({"Wj": Wj, "r_F": dU + F + Wj, "r_C": Wp - Wj})
    return out


def closure(cv: dict, species: list[int] | None = None, meta: dict | None = None,
            window: tuple[int, int] | None = None) -> dict:
    """Totals of the three residuals over all intervals (or over a step window),
    relative to the work they test: r_F/|Wj|, r_C/|W_phys|, r_total/|W_phys|."""
    iv = intervals(cv, species, meta)
    sel = np.ones(len(iv["W_phys"]), bool)
    if window is not None:
        sel = (iv["step0"] >= window[0]) & (iv["step1"] <= window[1])
    Wtot = float(iv["W_phys"][sel].sum())
    field = float((iv["dU"] + iv["flux"])[sel].sum())
    res = Wtot + field
    out = {"species": iv["species"], "W_total": Wtot, "field_side": field, "residual": res,
           "rel_residual": abs(res) / abs(Wtot) if Wtot != 0 else np.nan,
           "W_by_species": iv["W"][sel].sum(axis=0), "n_intervals": int(sel.sum())}
    if "Wj" in iv:
        Wj = float(iv["Wj"][sel].sum())
        out.update({"Wj_total": Wj,
                    "r_F_rel": abs(field + Wj) / abs(Wj) if Wj != 0 else np.nan,
                    "r_C_rel": abs(Wtot - Wj) / abs(Wtot) if Wtot != 0 else np.nan,
                    # names kept from format 2
                    "field_identity_rel": abs(field + Wj) / abs(Wj) if Wj != 0 else np.nan,
                    "coupling_rel": abs(Wtot - Wj) / abs(Wj) if Wj != 0 else np.nan})
    return out


def work_fraction(cv: dict, ions: list[int], leptons: list[int], window=None) -> float:
    """f_i^CV = Delta W_i / (Delta W_i + Delta W_pm) over a step window (s0, s1).
    Physical only for depositing ions (active layout)."""
    iv = intervals(cv, species=list(ions) + list(leptons))
    sel = np.ones(len(iv["W"]), bool)
    if window is not None:
        sel = (iv["step0"] >= window[0]) & (iv["step1"] <= window[1])
    Wi = iv["W"][sel][:, [s - 1 for s in ions]].sum()
    Wl = iv["W"][sel][:, [s - 1 for s in leptons]].sum()
    return float(Wi / (Wi + Wl)) if (Wi + Wl) != 0 else np.nan
