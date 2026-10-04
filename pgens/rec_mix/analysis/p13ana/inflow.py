"""Inflow speed (SM S6-S7), port of the p1.1 script betarec4.py.

Geometry (in the paper's axes; Entity: inflow = x2, outflow = x1):
central half of the box in the outflow direction, and two upstream slabs at
x_c - 2L/5 <= x <= x_c - L/5 and x_c + L/5 <= x <= x_c + 2L/5 in the inflow
direction (L = inflow extent). beta_L = <v_in>_lower, beta_R = -<v_in>_upper,
beta_in = (beta_L + beta_R)/2. An output is "good" if both are positive and
beta_in > beta_min (betarec4: 0.1); the estimate is the mean over the latest
n_avg consecutive good outputs (betarec4: 3).

The velocity passed in decides the definition:
  * per species: Entity field output V_s (mean 3-velocity of species s);
  * p1.1 continuity: the mass-weighted velocity of all depositing species
    (Tristan velx) = Entity output V_<all depositing species>, or
    mass_weighted(...) below from V_s and Rho_s with the same smoothing.
"""

from __future__ import annotations

import numpy as np


def beta_in(v_inflow: np.ndarray, x_in: np.ndarray, x_out: np.ndarray, beta_min: float = 0.10) -> dict:
    """v_inflow, x_in, x_out: arrays of the same 2D shape (inflow component of the
    velocity, inflow and outflow coordinates)."""
    v = np.asarray(v_inflow, float)
    xi = np.broadcast_to(x_in, v.shape)
    xo = np.broadcast_to(x_out, v.shape)
    xmin, xmax = np.nanmin(xi), np.nanmax(xi)
    ymin, ymax = np.nanmin(xo), np.nanmax(xo)
    Lx, Ly = xmax - xmin, ymax - ymin
    xc, yc = xmin + 0.5 * Lx, ymin + 0.5 * Ly
    my = np.abs(xo - yc) <= Ly / 4.0
    left = my & (xi >= xc - 2 * Lx / 5) & (xi <= xc - Lx / 5)
    right = my & (xi >= xc + Lx / 5) & (xi <= xc + 2 * Lx / 5)
    if not left.any() or not right.any():
        return {"beta": np.nan, "good": False}
    bl = float(np.nanmean(v[left]))
    br = -float(np.nanmean(v[right]))
    b = 0.5 * (bl + br)
    good = bool(np.isfinite(b) and bl > 0 and br > 0 and b > beta_min)
    return {"beta": b, "beta_left_in": bl, "beta_right_in": br, "good": good,
            "n_left": int(left.sum()), "n_right": int(right.sum())}


def latest_good_block(indices, good, n_avg: int = 3):
    """Positions of the latest block of n_avg consecutive good outputs whose
    output indices are also consecutive (betarec4.find_latest_good_block)."""
    indices = np.asarray(indices, int)
    good = np.asarray(good, bool)
    for stop in range(len(good), n_avg - 1, -1):
        pos = np.arange(stop - n_avg, stop)
        if np.all(good[pos]) and np.all(np.diff(indices[pos]) == 1):
            return pos
    return None


def mass_weighted(V: dict, Rho: dict) -> np.ndarray:
    """All-species mass-weighted velocity from per-species V_s and Rho_s outputs
    (exact if both were computed with the same smoothing)."""
    num = sum(V[s] * Rho[s] for s in V)
    den = sum(Rho[s] for s in V)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)
