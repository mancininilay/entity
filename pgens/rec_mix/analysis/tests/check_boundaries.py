#!/usr/bin/env python3
"""C1.5: boundary equivalence (tech spec III.4).

(a) reflection(): energy reflected by a wall from a vacuum plane pulse (the
    generator's field_test pulse_x2 / pulse_x1; no plasma, zero background and
    zero MATCH targets). Entity's MATCH update F <- s F + (1 - s) F_target is
    linear in F - F_target (src/kernels/fields_bcs.hpp), so the reflection of a
    deviation from the target does not depend on the target: a zero background
    tests the same operator as the production targets. Two pulses start at
    centre -/+ offset, each moving towards its own wall; the box halves are
    measured separately. R = (energy in the half after the interaction) /
    (energy in the half at the first output). Energy: sum over cells of
    E^2 + B^2 of Entity's field output (cell-centred, smoothing irrelevant
    for fields).
(b) layers(): |F - F_target| / B0 in the absorbing layers of a reconnection run,
    per field output, for the inflow (x2) layers (target: initial field, E = 0)
    and the outflow (x1) layers after t_open (target: the reconnected fields,
    I.3/D29, evaluated at the cell centres). Operationalization of "after a
    complete step, the field profiles in the layers stay within 2% of B0 of the
    targets in steady upstream conditions" (proposal, Claude 2026-10-12): every
    layer cell, all six components, at field outputs (the strict reading; the
    maximum over cells is an extreme value that includes PIC noise, so the rms
    and the largest x1-averaged row deviation are reported next to it). "Steady
    upstream" for the
    x2 layers: the outputs before any light signal from the sheet edge can reach
    the layer, t < (y0 - a - w_in)/c (the layer then sees the injected upstream
    plasma, the injector and the padding, but no reconnection disturbance);
    those outputs are flagged `quiescent` and judged. Later outputs are reported:
    in a small box the reconnection disturbance reaches the walls within a
    fraction of t_lc. The x1 layers are reported in an upstream band (3a <=
    |x2 - y0| <= y0 - padding - w_in) after t_open; their targets assume the
    steady reconnected inflow (E_x3 = -0.1 B0), which only a production-size
    run reaches (Rusty).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p13ana import readers  # noqa: E402

COMP = ["E1", "E2", "E3", "B1", "B2", "B3"]


def _energy_profile(path, axis):
    F = readers.read_entity_fields(path, COMP)
    u = sum(F[k].astype(np.float64) ** 2 for k in COMP)
    return F["step"], u.sum(axis=1 - axis)


def reflection(sim):
    """Reflected energy fraction for the two walls of a pulse test."""
    meta = readers.read_meta(os.path.join(sim, "usr"))
    ft = meta["geometry"]["field_test"]
    axis = 1 if ft == "pulse_x2" else 0
    fl = readers.list_entity_fields(sim, allow_unmanifested=True)
    s0, p0 = _energy_profile(fl[0], axis)
    s1, p1 = _energy_profile(fl[-1], axis)
    h = p0.size // 2
    e0 = (p0[:h].sum(), p0[h:].sum())
    e1 = (p1[:h].sum(), p1[h:].sum())
    x = np.arange(p1.size) + 0.5
    cen = (float((x[:h] * p1[:h]).sum() / p1[:h].sum()) if p1[:h].sum() > 0 else None,
           float((x[h:] * p1[h:]).sum() / p1[h:].sum()) if p1[h:].sum() > 0 else None)
    g = meta["geometry"]
    return {"field_test": ft, "steps": [int(s0), int(s1)], "pulse_width": g["pulse_width"],
            "layer_width": meta["derived"]["match_ds_x2" if axis == 1 else "match_ds_x1"],
            "E_initial": e0, "E_final": e1, "R_low": e1[0] / e0[0], "R_high": e1[1] / e0[1],
            "final_centroids": cen}


def _targets(meta, X1, X2, kind):
    d, inp = meta["derived"], meta["inputs"]
    a, x0, y0, bg = inp["cs_width"], d["x0"], d["y0"], inp["b_guide"]
    br = meta["geometry"]["beta_rec_target"]
    T = {k: np.zeros_like(X1) for k in COMP}
    T["B1"] = np.tanh((X2 - y0) / a)
    T["B3"] = -bg * np.ones_like(X1)
    if kind == "x1":
        T["B2"] = br * np.tanh((X1 - x0) / a)
        T["E3"] = -br * np.ones_like(X1)
        T["E1"] = -br * bg * np.tanh((X2 - y0) / a)
    return T


def layers(sim):
    meta = readers.read_meta(os.path.join(sim, "usr"))
    d, inp = meta["derived"], meta["inputs"]
    N1, N2, a, y0 = d["N1"], d["N2"], inp["cs_width"], d["y0"]
    w_in, w_out = d["match_ds_x2"], d["match_ds_x1"]
    pad = inp["injector_padding"]
    t_open_step = inp["t_open"] / d["dt"] if inp["t_open"] > 0 else np.inf
    t_quiet = (y0 - a - w_in)  # light travel time (c = 1 cell per unit time)
    rows = []
    for p in readers.list_entity_fields(sim):
        F = readers.read_entity_fields(p, COMP)
        X1, X2 = np.meshgrid(F["X1"], F["X2"], indexing="ij")
        r = {"label": int(F["step"]), "before_t_open": bool(F["step"] < t_open_step),
             "quiescent": bool(F["step"] * d["dt"] < t_quiet)}
        # x2 (inflow) layers: cell centres within w_in of a wall
        sel2 = (X2 < w_in) | (X2 > N2 - w_in)
        T = _targets(meta, X1, X2, "x2")
        dev = np.max([np.abs(F[k] - T[k])[sel2].max() for k in COMP])
        r["x2_layers_max_dev"] = float(dev)
        r["x2_layers_by_component"] = {k: float(np.abs(F[k] - T[k])[sel2].max()) for k in COMP}
        # noise-robust companions (reported): rms over the layer cells, and the
        # largest x1-averaged deviation of any layer row (systematic part)
        r["x2_layers_rms_dev"] = float(np.sqrt(np.mean([np.mean((F[k] - T[k])[sel2] ** 2) for k in COMP])))
        rows2 = np.any(sel2, axis=0)
        r["x2_layers_max_row_mean_dev"] = float(max(np.abs((F[k] - T[k])[:, rows2].mean(axis=0)).max()
                                                    for k in COMP))
        if not r["before_t_open"]:
            band = (np.abs(X2 - y0) >= 3 * a) & (np.abs(X2 - y0) <= y0 - pad - w_in)
            sel1 = ((X1 < w_out) | (X1 > N1 - w_out)) & band
            T1 = _targets(meta, X1, X2, "x1")
            if sel1.any():
                r["x1_layers_upstream_band_max_dev"] = float(
                    np.max([np.abs(F[k] - T1[k])[sel1].max() for k in COMP]))
                r["x1_layers_by_component"] = {k: float(np.abs(F[k] - T1[k])[sel1].max()) for k in COMP}
        rows.append(r)
    return {"w_in": w_in, "w_out": w_out, "t_open_step": float(t_open_step), "t_quiescent": t_quiet,
            "outputs": rows}


if __name__ == "__main__":
    sim = sys.argv[1]
    meta = readers.read_meta(os.path.join(sim, "usr"))
    fn = reflection if meta["geometry"]["field_test"] != "none" else layers
    print(json.dumps(fn(sim), indent=1))
