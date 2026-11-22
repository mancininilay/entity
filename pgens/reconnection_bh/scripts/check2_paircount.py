#!/usr/bin/env python3
"""Check 2: the number of pairs against a prediction made outside the policy.

    check2_paircount.py <bp2raw> <cache> <run with the policy> <j_table.txt>

Prediction: optical_depth * integral over time of S(t), S = sum over the ions
of J(Gamma_p), Gamma_p = 1 + K / mass_ratio, with J from `j_table.txt` (bhref,
ln J linear in y = log10(Gamma_p - Gamma_th)) and the ions from the particle
output of species 2 (every `stride`-th particle; the sum is scaled by the
number of ions of the spectra output of the same step over the number
written). Trapezoid rule over the outputs. Observed: the counter `pairs` of
bh_counters.txt, interpolated to the times of the first and the last output.

Errors (one standard deviation each):
  statistics  sqrt(prediction)
  stride      sum over the outputs of weight * sqrt(f (f - 1) sum_sampled J^2),
              f = ions / ions written: the sampling errors of all outputs
              taken as fully correlated (an upper bound); the value for
              independent outputs is printed too
  cadence     |I(all outputs) - I(every other output)| / 3
Tolerance: 4 * sqrt(statistics^2 + stride^2 + cadence^2). If the tolerance
exceeds `CAP` of the prediction the check gives no verdict (not sharp).
"""
import sys

import numpy as np

from bhio import Run, verdict

CAP = 0.05


def load_j(path):
    rows = np.loadtxt(path)
    gth = float(open(path).readline().split("gamma_th")[1].split(";")[0])
    return gth, rows[:, 0], np.log(rows[:, 2])


def rate(gamma, gth, y, lnj):
    """J(Gamma) from the table; 0 below it, an error above it."""
    out = np.zeros_like(gamma)
    x = gamma - gth
    m = x > 10.0 ** y[0]
    yy = np.log10(x[m])
    if yy.size and yy.max() > y[-1] + 1e-12:
        raise SystemExit("an ion lies above the table of J")
    out[m] = np.exp(np.interp(yy, y, lnj))
    return out


def trapezoid(t, s):
    return float(np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(t)))


def weights(t):
    w = np.zeros_like(t)
    w[:-1] += 0.5 * np.diff(t)
    w[1:] += 0.5 * np.diff(t)
    return w


def main():
    run = Run(sys.argv[3], sys.argv[1], sys.argv[2])
    gth, y, lnj = load_j(sys.argv[4])
    tau = run.param("optical_depth")
    mass_ratio = run.param("mass_ratio")
    eps0 = run.param("photon_energy")
    assert abs(gth - (1.0 / eps0 + eps0 / 4.0)) < 1e-12, "table and input: different bath"
    spectra = {step: bp for bp, step, _ in run.stamps("spectra")}
    c = run.counters()
    t, S, V, steps = [], [], [], []
    print("check 2: outputs of the ions")
    print("  step      time   ions     written   f        S        N_eff")
    for bp, step, time in run.stamps("particles"):
        v = run.read(bp, ["pU1_2", "pU2_2", "pU3_2", "pW_2"])
        u2 = (v["pU1_2"].astype(np.float64) ** 2 + v["pU2_2"].astype(np.float64) ** 2 +
              v["pU3_2"].astype(np.float64) ** 2)
        K = u2 / (np.sqrt(1.0 + u2) + 1.0)          # species 2 has mass 1
        wj = v["pW_2"].astype(np.float64) * rate(1.0 + K / mass_ratio, gth, y, lnj)
        written = K.size
        if step in spectra:
            ions = float(run.read(spectra[step], ["sN_2"])["sN_2"].sum())
        else:
            ions = float(np.interp(time, c["time"], c["n_2"]))
        f = ions / written
        s1, s2 = wj.sum(), (wj ** 2).sum()
        t.append(time)
        steps.append(step)
        S.append(f * s1)
        V.append(f * (f - 1.0) * s2)
        n_eff = f * s1 * s1 / s2 if s2 > 0 else 0.0
        print(f"  {step:6d} {time:9.3f} {ions:9.0f} {written:8d} {f:7.4f} {f * s1:10.5g} {n_eff:9.3g}")
    t, S, V = np.array(t), np.array(S), np.array(V)
    if t.size < 5:
        raise SystemExit("check 2 needs at least five outputs of the ions")
    predicted = tau * trapezoid(t, S)
    coarse = tau * trapezoid(t[::2], S[::2]) if t.size % 2 == 1 else \
        tau * (trapezoid(t[:-1:2], S[:-1:2]) + 0.5 * (S[-1] + S[-2]) * (t[-1] - t[-2]))
    observed = float(np.interp(t[-1], c["time"], c["pairs"]) - np.interp(t[0], c["time"], c["pairs"]))
    w = weights(t)
    e_stat = np.sqrt(predicted)
    e_stride = tau * float(np.sum(w * np.sqrt(V)))
    e_stride_ind = tau * float(np.sqrt(np.sum(w * w * V)))
    e_cadence = abs(predicted - coarse) / 3.0
    sigma = np.sqrt(e_stat ** 2 + e_stride ** 2 + e_cadence ** 2)
    tol = 4.0 * sigma
    print(f"  optical depth {tau:g}; {t.size} outputs from t = {t[0]:.4g} to {t[-1]:.6g}, "
          f"mean spacing {np.diff(t).mean():.4g}")
    print(f"  pairs observed {observed:.1f}; predicted {predicted:.1f}; "
          f"observed/predicted - 1 = {observed / predicted - 1:+.4e}")
    print(f"  errors (fraction of the prediction): statistics {e_stat / predicted:.3e}; "
          f"stride {e_stride / predicted:.3e} (independent outputs {e_stride_ind / predicted:.3e}); "
          f"cadence {e_cadence / predicted:.3e}; total {sigma / predicted:.3e}")
    print(f"  difference {observed - predicted:+.1f} = {(observed - predicted) / sigma:+.2f} total "
          f"standard deviations; tolerance {tol:.1f} ({tol / predicted:.3e} of the prediction)")
    if tol > CAP * predicted:
        print(f"check 2 verdict: NO VERDICT (tolerance above {CAP:g} of the prediction: not sharp)")
        return 2
    ok = abs(observed - predicted) <= tol
    print(f"check 2 verdict: {verdict(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
