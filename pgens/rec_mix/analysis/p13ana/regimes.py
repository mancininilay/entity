"""Units mapping (tech spec III.1, P13-D39) and regime numbers (III.11, VII.1).

P13-D39 (Nilay, 2026-10-12): the p1.1 labels are HOT. sigma_hot = B^2/(4 pi w)
is the total upstream magnetization including ion and thermal enthalpy, with a
common physical upstream temperature T = k_B T/(me c^2) (0.01) and Gamma = 5/3;
d_up is the total upstream skin depth including the thermal and ion response.
Tristan's sigma and c_omp (Entity's sigma0 and skindepth0) are the fiducial
cold electron-mass inputs, n0 counting every upstream particle:

    D           = (2 - q + mu q)/2                     (rho/(n0 me); testprtl: 1)
    sigma_input = sigma_hot (D + 5T/2)
    F           = [(2-q)/(1+3T/2) + q/(mu (1+3T/2))]/2  (testprtl: 1/(1+3T/2))
    c_omp       = d_up sqrt(F)
    larmor0     = c_omp/sqrt(sigma_input)

Literal (saved) inputs may replace the nominal sigma_input, c_omp; the labels
are then the provenance. T -> 0 recovers the pre-D39 cold mapping, which holds
in the cold limit only.
"""

from __future__ import annotations

import math


def scales(layout: str, *, sigma_hot: float, q: float, mu: float, dup_cells: float,
           T: float = 0.01, sigma_input: float | None = None, c_omp: float | None = None) -> dict:
    """Entity scales and derived magnetizations/skin depths (keyword-only, so a
    pre-D39 positional call with a cold sigma fails instead of being reinterpreted).

    Returns
      D, H = D + 5T/2, F (Nilay's convention), sigma_input/c_omp (nominal and used),
      sigma0 = sigma_input, skindepth0 = c_omp, larmor0,
      sigma_cold_total = sigma_input/D (cold total magnetization),
      sigma_e = 2 sigma_input/(2 - q): cold lepton rest-mass magnetization
                (testprtl: sigma_input); sigma_e_star = sigma_input,
      d_up_effective = c_omp/sqrt(F) (the thermal total skin depth the used
                c_omp implies; 5 cells only for nominal inputs),
      d_e = d_up_effective sqrt(1 + q/(mu (2 - q))): thermal lepton skin depth
            (F convention: the thermal factor cancels in d_e/d_up),
      d_e_cold = c_omp sqrt(2/(2 - q)) = d_e/sqrt(1 + 3T/2),
      chi = mu q/(2 - q), and F_common_T with the c_omp ratio it would imply
      (Codex's qualification; documented, not used).
    """
    active = layout == "active"
    if not active:
        q, mu = 0.0, 1.0
    D = (2 - q + q * mu) / 2
    H = D + 2.5 * T
    th = 1 + 1.5 * T
    F = ((2 - q) / th + q / (mu * th)) / 2
    Fc = ((2 - q) / th + q / (mu + 1.5 * T)) / 2
    s_nom, c_nom = sigma_hot * H, dup_cells * math.sqrt(F)
    if (sigma_input is None) != (c_omp is None):
        raise ValueError("give both sigma_input and c_omp (literal inputs) or neither")
    s_in = s_nom if sigma_input is None else float(sigma_input)
    c_in = c_nom if c_omp is None else float(c_omp)
    dup_eff = c_in / math.sqrt(F)
    lep = 1 + (q / (mu * (2 - q)) if q > 0 else 0.0)
    return {"mode": "nominal" if sigma_input is None else "literal",
            "D": D, "H": H, "F": F, "T": T, "sigma_hot": sigma_hot,
            "sigma_input_nominal": s_nom, "c_omp_nominal": c_nom,
            "sigma_input": s_in, "c_omp": c_in,
            "sigma0": s_in, "skindepth0": c_in, "larmor0": c_in / math.sqrt(s_in),
            "sigma_cold_total": s_in / D, "sigma_e": 2 * s_in / (2 - q), "sigma_e_star": s_in,
            "d_up_nominal": dup_cells, "d_up_effective": dup_eff,
            "d_e": dup_eff * math.sqrt(lep), "d_e_cold": c_in * math.sqrt(2 / (2 - q)),
            "chi": mu * q / (2 - q), "F_common_T": Fc, "c_omp_ratio_common_T": math.sqrt(Fc / F)}


def gamma_cool(gamma_rad: float, d_e: float, sigma_e: float, L: float) -> float:
    """Downstream cooling Lorentz factor gamma_cool = 20 gamma_rad^2 d_e/(sqrt(sigma_e) L).
    A reference label (III.11): with the D39 hot labels, pass the d_e and sigma_e
    convention explicitly (scales() gives the thermal d_e and the cold lepton
    rest-mass sigma_e; their thermal factors are O(T))."""
    return 20 * gamma_rad**2 * d_e / (math.sqrt(sigma_e) * L)


def regime(gamma_rad: float, eps_e0: float, gcool: float) -> str:
    if gcool > eps_e0:
        return "I"
    if gamma_rad > eps_e0:
        return "II"
    return "III"
