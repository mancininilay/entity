"""Energy partition in the downstream mask (P1.3 tech spec III.5).

Active and passive schemas are separate functions on purpose (Codex R4): tracer
sums can never enter a physical fraction.

Two estimators are provided for every quantity:

* ``hist``   : from the masked spectra and the geometric bin centres, exactly
               as the p1.1 script ``extractfi.py`` (continuity with p1.1);
* ``direct`` : from the double-precision moment sums written by rec_mix
               (the primary P1.3 estimator).

Energies are kinetic, in m_e c^2, per particle.
"""

from __future__ import annotations

import numpy as np

from .readers import bin_centres

# species roles (1-based indices) of the two layouts
ACTIVE = {"leptons": [4, 5, 7, 8], "ions": [6, 9], "sheet": [1, 2, 3]}
PASSIVE = {"leptons": [4, 5, 7, 8], "tracers": [3, 6, 9], "sheet": [1, 2]}


def _mass(meta, s):
    return float(meta["species"][s - 1]["mass"])


# Bulk-corrected energies: what the status means (Codex IC5, RV1, A0-3; v1.6).
#
# eps-tilde with the own-species bulk is a SECONDARY, UNVALIDATED diagnostic
# (P13-D36): f_i, eps, spectra and the CV work do not depend on it. Its status
# is an OCCUPANCY SCREEN (decided thresholds, P13-D36), not an accuracy
# certificate.
# "occupancy_ok" means: at most MAX_LOWOCC of the population's masked particles
# sit in own-species boxes with fewer than nmin (meta bulk.nmin, default 20)
# particles, AND (format 3 with Klowocc) max(Klowocc, Kblowocc) is at most
# MAX_LOWOCC of K - Kb_own. Klowocc is the masked kinetic energy of the
# particles in under-occupied boxes and Kblowocc the bulk energy the estimator
# assigns to them; both are >= 0, so the recorded low-occupancy contribution
# Klowocc - Kblowocc satisfies |Klowocc - Kblowocc| <= max(Klowocc, Kblowocc):
# the ratio bounds the MAGNITUDE OF THAT RECORDED CONTRIBUTION, which makes it a
# conservative contribution screen. It is NOT a bound on its error against the
# physical random energy of those particles: the box velocity is averaged over
# the whole (2w + 1)^2 box and assigned to the masked central-cell particles, and
# Jensen's inequality over the box says nothing about that central subset
# (Codex A0-3a witness: one stationary masked particle with nine neighbours at
# beta = 0.9 gives Klowocc = 0, Kblowocc = 0.685, a negative selected random
# energy). The v1.5 claim "true random energy in [0, Klowocc]" is withdrawn.
# A number fraction alone does not screen the energy share (Codex RV1a: 1% of
# the particles can carry 91% of the random energy). Outputs without Klowocc
# are screened on the number fraction only ("occupancy_ok_number_only").
# Otherwise the value is NaN with status "low_occupancy"; the unchecked numbers
# stay under "*_unchecked".
#
# Not covered by any screen: the finite-sample bias of the estimator in
# well-occupied boxes. The variance correction makes |b|^2 unbiased before
# clipping, but gamma(|b|^2) is nonlinear, steeply so as |b| -> 1, and rare
# high-leverage particles (Codex RV1b: a 1% counter-stream at beta = 0.99 gives
# a 69% underestimate with 20 particles per box) bias it in a
# distribution-dependent way. The Gaussian calibration in
# tests/test_safeguards.py (bias <= 1.5% of eps-tilde at n = 20 for drifting
# Gaussians with beta_b <= 0.5) is a calibration for those flows, not a bound.
# Position-subsampling sensitivity (formerly "thinning"): format 3 outputs carry
# Kb_own_half (box means from the particles with dx1 < 1/2, about n/2 per box,
# weights from all particles); "epst_<pop>_position_subsampling_change" =
# (eps-tilde(half) - eps-tilde(n))/eps-tilde(n) is reported with every value
# (also under its former name "epst_<pop>_thinning_change"). The selection is
# by in-cell position, and dx1 can be correlated with velocity in the evolved
# plasma (Codex A0-3b witness: two streams at dx1 = 1/4 and 3/4 give -100% at
# any occupancy), so a change can reflect sub-cell phase-space structure as
# well as finite-sample bias; a small change is weak evidence, not a bound, and
# one split is not an error bar. Phase-space-independent repeated subsampling
# is an R1 item (P13-D36), not implemented (tech spec III.8).
#
# Window sensitivity: eps-tilde measures random motion below the box scale
# (2w + 1 cells; w = 2, i.e. 5 x 5 cells, is Tristan's smooth_window and the
# p1.1 convention), so it depends on w; "epst_<pop>_window_change" =
# (eps-tilde(w) - eps-tilde(1))/eps-tilde(w) is reported, not judged.
# The legacy Tristan bulk subtraction is not reproduced (P13-D38).
MAX_LOWOCC = 0.02
OK = ("occupancy_ok", "occupancy_ok_number_only")


def _bulk_status(out, species, N, max_lowocc, suffix=""):
    """Occupancy screen of a population (list of species): status, number
    fraction in under-occupied boxes, energy bound fraction (NaN if not output)."""
    ncol = "Nlowocc" + suffix
    if ncol not in out.names:
        return "unknown_occupancy", np.nan, np.nan
    lo = sum(out.m(ncol)[s - 1] for s in species)
    frac = lo / N if N > 0 else np.nan
    efrac = np.nan
    if "Klowocc" + suffix in out.names:
        K = sum(out.m("K")[s - 1] for s in species)
        Kb = sum(out.m("Kb_own" + suffix)[s - 1] for s in species)
        Klo = sum(out.m("Klowocc" + suffix)[s - 1] for s in species)
        Kblo = sum(out.m("Kblowocc" + suffix)[s - 1] for s in species)
        efrac = max(Klo, Kblo) / (K - Kb) if K - Kb > 0 else (0.0 if max(Klo, Kblo) == 0 else np.inf)
    if not (np.isfinite(frac) and frac <= max_lowocc):
        return "low_occupancy", frac, efrac
    if np.isnan(efrac):
        return "occupancy_ok_number_only", frac, efrac
    return ("occupancy_ok" if efrac <= max_lowocc else "low_occupancy"), frac, efrac


def _thinning(d, out, key, species, K, N):
    """eps-tilde with the box means from the particles with dx1 < 1/2 (format 3):
    position-subsampling sensitivity (A0-3b; see above)."""
    if "Kb_own_half" not in out.names or not N > 0:
        return
    Kbh = sum(out.m("Kb_own_half")[s - 1] for s in species)
    eh = (K - Kbh) / N
    e = d.get(key + "_own_unchecked", np.nan)
    d[key + "_own_half_unchecked"] = eh
    ch = (eh - e) / e if e else np.nan
    d[key + "_position_subsampling_change"] = ch
    d[key + "_thinning_change"] = ch  # former name (v1.5)


# --------------------------------------------------------------------------
# Active schema
# --------------------------------------------------------------------------
def active_partition(meta: dict, out, estimator: str = "direct",
                     max_lowocc: float = MAX_LOWOCC) -> dict:
    """f_i, eps_e, eps_i, eps-tilde (own and all-species bulk) from one output.
    eps-tilde with the own-species bulk is NaN unless it passes the occupancy
    screen (an occupancy check, not an accuracy certificate; see above)."""
    if meta["layout"] != "active":
        raise ValueError("active_partition needs an active-layout run")
    lep, ion = ACTIVE["leptons"], ACTIVE["ions"]
    mu = _mass(meta, ion[0])
    if estimator == "hist":
        eb = bin_centres(meta)
        cl = sum(out.hist[s - 1] for s in lep)
        ci = sum(out.hist[s - 1] for s in ion)
        Nl, Ni = cl.sum(), ci.sum()
        Kl, Ki = (cl * eb).sum(), mu * (ci * eb).sum()
    elif estimator == "direct":
        N, K = out.m("N"), out.m("K")
        Nl, Ni = sum(N[s - 1] for s in lep), sum(N[s - 1] for s in ion)
        Kl, Ki = sum(K[s - 1] for s in lep), sum(K[s - 1] for s in ion)
    else:
        raise ValueError(estimator)
    res = {
        "N_pm": Nl, "N_i": Ni, "K_pm": Kl, "K_i": Ki,
        "f_i": Ki / (Ki + Kl) if (Ki + Kl) > 0 else np.nan,
        "eps_e": Kl / Nl if Nl > 0 else np.nan,
        "eps_i": Ki / Ni if Ni > 0 else np.nan,
    }
    if estimator == "direct":
        tags = ("own", "all") + (("own_raw",) if "Kb_own_raw" in out.names else ())
        for tag in tags:
            Kb = out.m(f"Kb_{tag}")
            Kbl = sum(Kb[s - 1] for s in lep)
            Kbi = sum(Kb[s - 1] for s in ion)
            res[f"epst_e_{tag}"] = (Kl - Kbl) / Nl if Nl > 0 else np.nan
            res[f"epst_i_{tag}"] = (Ki - Kbi) / Ni if Ni > 0 else np.nan
        for pop, sp, Npop in (("e", lep, Nl), ("i", ion, Ni)):
            status, frac, efrac = _bulk_status(out, sp, Npop, max_lowocc)
            res[f"epst_{pop}_status"] = status
            res[f"{pop}_low_occupancy_fraction"] = frac
            res[f"{pop}_low_occupancy_energy_bound"] = efrac  # former name
            res[f"{pop}_low_occupancy_contribution_ratio"] = efrac  # A0-3a: bounds the recorded contribution
            for tag in ("own", "own_raw"):
                key = f"epst_{pop}_{tag}"
                if key in res:
                    res[key + "_unchecked"] = res[key]
                    if status not in OK:
                        res[key] = np.nan
            Kpop = Kl if pop == "e" else Ki
            _window(res, out, f"epst_{pop}", sp, Kpop, Npop, max_lowocc)
            _thinning(res, out, f"epst_{pop}", sp, Kpop, Npop)
        res["ion_low_occupancy_fraction"] = res["i_low_occupancy_fraction"]
    return res


def _window(d, out, key, species, K, N, max_lowocc):
    """Add eps-tilde at window 1 (own bulk, validity from Nlowocc_w1) and the
    relative change from window 1 to the configured window."""
    if "Kb_own_w1" not in out.names or not N > 0:
        return
    Kb1 = sum(out.m("Kb_own_w1")[s - 1] for s in species)
    e1 = (K - Kb1) / N
    st, fr, ef = _bulk_status(out, species, N, max_lowocc, "_w1")
    d[key + "_own_w1_unchecked"] = e1
    d[key + "_own_w1"] = e1 if st in OK else np.nan
    d[key + "_w1_status"] = st
    e = d.get(key + "_own_unchecked", np.nan)
    d[key + "_window_change"] = (e - e1) / e if e else np.nan


# --------------------------------------------------------------------------
# Passive schema
# --------------------------------------------------------------------------
def passive_partition(meta: dict, out, estimator: str = "direct",
                      max_lowocc: float = MAX_LOWOCC) -> dict:
    """Leptons (eps_e) and, per tracer mass separately, eps_i and eps-tilde_i.
    No fraction involving tracer sums is defined."""
    if meta["layout"] != "testprtl":
        raise ValueError("passive_partition needs a testprtl-layout run")
    lep = PASSIVE["leptons"]
    res = {"tracers": {}}
    if estimator == "hist":
        eb = bin_centres(meta)
        cl = sum(out.hist[s - 1] for s in lep)
        Nl, Kl = cl.sum(), (cl * eb).sum()
    else:
        N, K = out.m("N"), out.m("K")
        Nl, Kl = sum(N[s - 1] for s in lep), sum(K[s - 1] for s in lep)
    res.update({"N_pm": Nl, "K_pm": Kl, "eps_e": Kl / Nl if Nl > 0 else np.nan})
    if estimator == "direct":
        Kb = {t: out.m(f"Kb_{t}") for t in ("own", "all")}
        for t in ("own", "all"):
            res[f"epst_e_{t}"] = (Kl - sum(Kb[t][s - 1] for s in lep)) / Nl if Nl > 0 else np.nan
        status, frac, efrac = _bulk_status(out, lep, Nl, max_lowocc)
        res["epst_e_status"], res["e_low_occupancy_fraction"] = status, frac
        res["e_low_occupancy_energy_bound"] = efrac  # former name
        res["e_low_occupancy_contribution_ratio"] = efrac
        res["epst_e_own_unchecked"] = res["epst_e_own"]
        if status not in OK:
            res["epst_e_own"] = np.nan
        _window(res, out, "epst_e", lep, Kl, Nl, max_lowocc)
        _thinning(res, out, "epst_e", lep, Kl, Nl)
    for s in PASSIVE["tracers"]:
        m = _mass(meta, s)
        if estimator == "hist":
            c = out.hist[s - 1]
            Ns, Ks = c.sum(), m * (c * eb).sum()
        else:
            Ns, Ks = out.m("N")[s - 1], out.m("K")[s - 1]
        d = {"mass": m, "N": Ns, "K": Ks, "eps_i": Ks / Ns if Ns > 0 else np.nan}
        if estimator == "direct":
            # Kb_all is undefined (NaN) for non-depositing tracers: the
            # depositing-species velocity applied to the tracer mass is not a
            # bulk energy of the tracer population
            d["epst_i_own"] = (Ks - Kb["own"][s - 1]) / Ns if Ns > 0 else np.nan
            d["epst_i_own_raw"] = (Ks - out.m("Kb_own_raw")[s - 1]) / Ns \
                if (Ns > 0 and "Kb_own_raw" in out.names) else np.nan
            status, frac, efrac = _bulk_status(out, [s], Ns, max_lowocc)
            d["epst_status"], d["low_occupancy_fraction"] = status, frac
            d["low_occupancy_energy_bound"] = efrac  # former name
            d["low_occupancy_contribution_ratio"] = efrac
            for key in ("epst_i_own", "epst_i_own_raw"):
                d[key + "_unchecked"] = d[key]
                if status not in OK:
                    d[key] = np.nan
            _window(d, out, "epst_i", [s], Ks, Ns, max_lowocc)
            _thinning(d, out, "epst_i", [s], Ks, Ns)
        res["tracers"][s] = d
    return res


# --------------------------------------------------------------------------
# Time series and the p1.1 steady-state protocol (last 50 outputs)
# --------------------------------------------------------------------------
def series(meta, outs, func, keys, **kw):
    rows = [func(meta, o, **kw) for o in outs]
    t = np.array([o.time for o in outs])
    step = np.array([o.step for o in outs])
    return {"time": t, "step": step, **{k: np.array([r[k] for r in rows]) for k in keys}}


def steady(x: np.ndarray, n: int = 50, nan_policy: str = "propagate") -> tuple[float, float]:
    """Mean and (population) temporal standard deviation of the last n values.
    nan_policy="propagate" (default): any invalid value in the window makes the
    result NaN, so an invalid measurement (e.g. eps-tilde at low occupancy) never
    enters an average silently. nan_policy="omit" reproduces extractfi.py
    (np.nanmean, np.nanstd) for the p1.1 continuity comparison."""
    x = np.asarray(x, float)[-n:]
    if nan_policy == "omit":
        return float(np.nanmean(x)), float(np.nanstd(x))
    if nan_policy != "propagate":
        raise ValueError(nan_policy)
    return float(np.mean(x)), float(np.std(x))


def block_error(x: np.ndarray, nblocks: int = 5) -> float:
    """Standard error of the mean from `nblocks` contiguous blocks (III.9)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < nblocks:
        return np.nan
    b = np.array([blk.mean() for blk in np.array_split(x, nblocks)])
    return float(b.std(ddof=1) / np.sqrt(nblocks))


# --------------------------------------------------------------------------
# p1.1 Tristan spectra (continuity, C1.10): exact port of extractfi.py
# --------------------------------------------------------------------------
def tristan_active_partition(spec: dict, mu: float) -> dict:
    eb = spec["ebins"]
    cl = sum(spec["counts"][s] for s in (4, 5, 7, 8))
    ci = sum(spec["counts"][s] for s in (6, 9))
    Nl, Ni = cl.sum(), ci.sum()
    El, Ei = (cl * eb).sum(), mu * (ci * eb).sum()
    return {
        "f_i": Ei / (El + Ei) if (El + Ei) > 0 else np.nan,
        "eps_e": El / Nl if Nl > 0 else np.nan,
        "eps_i": Ei / Ni if Ni > 0 else np.nan,
        "N_pm": Nl, "N_i": Ni,
    }


def compare(xE: float, xT: float, sE: float, sT: float, delta_rel: float) -> dict:
    """Tolerance formula (III.9): |X_E - X_T| <= max(delta_rel |X_T|, 2 s_comb)."""
    scomb = float(np.hypot(sE, sT))
    tol = max(delta_rel * abs(xT), 2.0 * scomb)
    return {"diff": xE - xT, "tol": tol, "pass": abs(xE - xT) <= tol,
            "ratio_to_tol": abs(xE - xT) / tol if tol > 0 else np.inf}


def mixing_from_counts(lower, upper):
    """Mask definition (P13-D13): M = 2 min(D_lo, D_up)/(D_lo + D_up) with D the
    total particle count of the side species; 0 where both are empty."""
    lo = np.asarray(lower, float)
    up = np.asarray(upper, float)
    s = lo + up
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, 2 * np.minimum(lo, up) / s, 0.0)
