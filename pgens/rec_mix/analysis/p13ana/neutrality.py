"""C1.3 charge neutrality: an exact load check and a noise-aware local test
(tech spec V.2, revised after Codex A3).

(a) Exact: the load is neutral in every cell (co-located pairs). rec_mix checks
    this at initialisation (log line "C1.3a load neutrality") and aborts otherwise.

(b) Local, after the first step: particles of a co-located pair move
    independently, so a block of cells acquires a charge whenever exactly one
    member of a pair crosses its boundary. For a pair starting at a uniformly
    distributed point in a cell, the probability that a given edge separates the
    two members is E|Delta_perp| (cells), with Delta the relative displacement
    over one step, as long as |Delta| < 1. Each separated pair adds +-1 e to the
    block, independently, so

        Var(Q_block) = sum_types n_type * P * E|Delta_perp,type|,

    with n_type the pairs per cell of each pair type and P the perimeter in
    cells. The expectation is computed here from the input temperatures by an
    independent Maxwell-Juttner sampler (inverse CDF of the energy distribution
    on a grid; rec_mix uses a gamma-mixture rejection sampler), not from the
    measured charge.

    The test, on blocks in the far upstream (no sheet population, away from the
    padding and the injection rows):
      * the rms block charge is within [0.8, 1.25] of the model (the rms of
        N >= 50 blocks has relative scatter ~ 1/sqrt(2N));
      * the largest |rho_block|/sigma_model is below the Bonferroni bound
        z_N = Phi^-1(1 - alpha/(2N)), alpha = 1e-3 (a simultaneous bound over
        all blocks, not a pointwise 4 sigma rule).
    Block correlations through shared edges make the maximum test conservative.
"""

from __future__ import annotations

import math

import numpy as np


def juttner_beta_samples(theta: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Isotropic 3-velocities from a Maxwell-Juttner distribution of temperature
    theta (in the species' rest-mass units), by inverse CDF of the kinetic-energy
    density f(e) ~ (1 + e) sqrt(e (e + 2)) exp(-e/theta) on a fine grid."""
    emax = theta * 60.0 + 1e-12
    e = np.linspace(0.0, emax, 200001)
    f = (1 + e) * np.sqrt(e * (e + 2)) * np.exp(-e / theta)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(e))])
    cdf /= cdf[-1]
    ek = np.interp(rng.random(n), cdf, e)
    g = 1 + ek
    beta = np.sqrt(1 - 1 / g ** 2)
    mu = 2 * rng.random(n) - 1
    return beta * mu  # one Cartesian component


def pair_separation(theta_a: float, theta_b: float, dt: float, n: int = 2_000_000,
                    seed: int = 0) -> float:
    """E|Delta_perp| in cells after one step for a co-located pair whose members
    have temperatures theta_a and theta_b (each in its own mass units)."""
    rng = np.random.default_rng(seed)
    d = (juttner_beta_samples(theta_a, n, rng) - juttner_beta_samples(theta_b, n, rng)) * dt
    if np.any(np.abs(d) >= 1):
        raise ValueError("one-step displacement >= 1 cell; the crossing model does not apply")
    return float(np.mean(np.abs(d)))


def block_sigma(pair_types: list[tuple[float, float, float]], dt: float, block: int,
                ppc0: float) -> float:
    """Model standard deviation of the block-averaged charge density in units of
    e n0 (Entity's `Charge` output with smoothing 0 is e n0 per cell). Each pair
    type is (pairs per cell, theta_a, theta_b)."""
    var_q = 0.0
    for n_pc, ta, tb in pair_types:
        var_q += n_pc * 4 * block * pair_separation(ta, tb, dt)
    return math.sqrt(var_q) / (block * block * ppc0)


def bonferroni_z(nblocks: int, alpha: float = 1e-3) -> float:
    from statistics import NormalDist

    return NormalDist().inv_cdf(1 - alpha / (2 * nblocks))


def local_test(charge: np.ndarray, rows: list[tuple[int, int]], cols: tuple[int, int],
               block: int, sigma: float, alpha: float = 1e-3) -> dict:
    """charge: Entity `Charge` [i1, i2] (smoothing 0). rows: list of (j0, j1)
    ranges in x2 (far upstream); cols: (i0, i1) range in x1."""
    vals = []
    i0, i1 = cols
    for j0, j1 in rows:
        sub = charge[i0:i1, j0:j1]
        n1, n2 = sub.shape[0] // block, sub.shape[1] // block
        sub = sub[: n1 * block, : n2 * block].astype(np.float64)
        vals.append(sub.reshape(n1, block, n2, block).mean(axis=(1, 3)).ravel())
    v = np.concatenate(vals)
    rms = float(np.sqrt(np.mean(v ** 2)))
    zmax = float(np.max(np.abs(v)) / sigma)
    zn = bonferroni_z(v.size, alpha)
    return {"nblocks": int(v.size), "sigma_model": sigma, "rms": rms, "rms_ratio": rms / sigma,
            "max_abs": float(np.max(np.abs(v))), "z_max": zmax, "z_bound": zn,
            "pass": bool(0.8 <= rms / sigma <= 1.25 and zmax <= zn)}
