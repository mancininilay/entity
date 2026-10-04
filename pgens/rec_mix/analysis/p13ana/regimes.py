"""Units mapping (III.1) and regime numbers (III.11, VII.1)."""

from __future__ import annotations

import math


def scales(layout: str, sigma: float, q: float, mu: float, dup_cells: float) -> dict:
    if layout == "active":
        D = (2 - q + q * mu) / 2
        sigma0 = sigma * D
        skin = dup_cells * math.sqrt((2 - q + q / mu) / 2)
        de = dup_cells * math.sqrt(1 + (q / (mu * (2 - q)) if q > 0 else 0))
        chi = mu * q / (2 - q)
    else:
        D, sigma0, skin, de, chi = 1.0, sigma, dup_cells, dup_cells, 0.0
    return {"D": D, "sigma0": sigma0, "sigma_e_star": sigma0, "sigma_e": sigma * (1 + chi),
            "skindepth0": skin, "larmor0": skin / math.sqrt(sigma0), "d_e": de, "chi": chi}


def gamma_cool(gamma_rad: float, d_e: float, sigma_e: float, L: float) -> float:
    """Downstream cooling Lorentz factor gamma_cool = 20 gamma_rad^2 d_e/(sqrt(sigma_e) L)."""
    return 20 * gamma_rad**2 * d_e / (math.sqrt(sigma_e) * L)


def regime(gamma_rad: float, eps_e0: float, gcool: float) -> str:
    if gcool > eps_e0:
        return "I"
    if gamma_rad > eps_e0:
        return "II"
    return "III"
