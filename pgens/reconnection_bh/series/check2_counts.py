#!/usr/bin/env python3
"""Check 2 of the resolution pair: the number of pairs against a prediction made outside the policy,
with the written ions scaled by the count of the counters (Claude 1, 2026-11-23; the deferred item 1
of the review log of stage B and Codex 2's B4).

    check2_counts.py <bp2raw> <cache> <run with the policy> <j_table.txt>

As `scripts/check2_paircount.py` (registered for stages B and C), with one change: the factor
f = ions / ions written of every particle output takes the number of ions from `bh_counters.txt`
(n_2, interpolated in time to the output, a count kept in double precision), not from the
single-precision spectrum. The factor from the spectrum is printed beside it. The errors are
printed separately and combined as there:
  statistics  sqrt(prediction)
  stride      sum over the outputs of weight * sqrt(f (f - 1) sum_sampled J^2) (fully correlated
              outputs, an upper bound; the value for independent outputs is printed too)
  cadence     |I(all outputs) - I(every other output)| / 3
Tolerance: 4 * sqrt(statistics^2 + stride^2 + cadence^2); no verdict when it exceeds 5 percent of
the prediction. Exit 0 pass, 1 fail, 2 no verdict.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check2_paircount as c2  # noqa: E402
from bhio import Run, verdict  # noqa: E402


def main():
    run = Run(sys.argv[3], sys.argv[1], sys.argv[2])
    gth, y, lnj = c2.load_j(sys.argv[4])
    tau = run.param("optical_depth")
    mass_ratio = run.param("mass_ratio")
    eps0 = run.param("photon_energy")
    assert abs(gth - (1.0 / eps0 + eps0 / 4.0)) < 1e-12, "table and input: different bath"
    spectra = {step: bp for bp, step, _ in run.stamps("spectra")}
    c = run.counters()
    t, S, V = [], [], []
    print(f"check 2 (counts of the counters): {run.name}, outputs of the ions")
    print("  step      time   ions (counters)  ions (spectrum)  written   f        S        N_eff")
    for bp, step, time in run.stamps("particles"):
        v = run.read(bp, ["pU1_2", "pU2_2", "pU3_2", "pW_2"])
        u2 = (v["pU1_2"].astype(np.float64) ** 2 + v["pU2_2"].astype(np.float64) ** 2 +
              v["pU3_2"].astype(np.float64) ** 2)
        K = u2 / (np.sqrt(1.0 + u2) + 1.0)          # species 2 has mass 1
        wj = v["pW_2"].astype(np.float64) * c2.rate(1.0 + K / mass_ratio, gth, y, lnj)
        written = K.size
        ions = float(np.interp(time, c["time"], c["n_2"]))
        ions_s = float(run.read(spectra[step], ["sN_2"])["sN_2"].sum()) if step in spectra else float("nan")
        f = ions / written
        s1, s2 = wj.sum(), (wj ** 2).sum()
        t.append(time)
        S.append(f * s1)
        V.append(f * (f - 1.0) * s2)
        n_eff = f * s1 * s1 / s2 if s2 > 0 else 0.0
        print(f"  {step:6d} {time:9.3f} {ions:16.0f} {ions_s:16.0f} {written:8d} {f:8.5f} {f * s1:10.5g} {n_eff:9.3g}")
    t, S, V = np.array(t), np.array(S), np.array(V)
    if t.size < 5:
        raise SystemExit("check 2 needs at least five outputs of the ions")
    predicted = tau * c2.trapezoid(t, S)
    coarse = tau * c2.trapezoid(t[::2], S[::2]) if t.size % 2 == 1 else \
        tau * (c2.trapezoid(t[:-1:2], S[:-1:2]) + 0.5 * (S[-1] + S[-2]) * (t[-1] - t[-2]))
    observed = float(np.interp(t[-1], c["time"], c["pairs"]) - np.interp(t[0], c["time"], c["pairs"]))
    w = c2.weights(t)
    e_stat = np.sqrt(predicted)
    e_stride = tau * float(np.sum(w * np.sqrt(V)))
    e_stride_ind = tau * float(np.sqrt(np.sum(w * w * V)))
    e_cadence = abs(predicted - coarse) / 3.0
    sigma = np.sqrt(e_stat ** 2 + e_stride ** 2 + e_cadence ** 2)
    tol = 4.0 * sigma
    print(f"  optical depth {tau:g}; {t.size} outputs from t = {t[0]:.4g} to {t[-1]:.6g}")
    print(f"  pairs observed {observed:.1f}; predicted {predicted:.1f}; observed/predicted - 1 = "
          f"{observed / predicted - 1:+.4e}")
    print(f"  errors, in pairs and as fractions of the prediction: statistics {e_stat:.1f} ({e_stat / predicted:.3e}); "
          f"stride {e_stride:.1f} ({e_stride / predicted:.3e}; independent outputs {e_stride_ind / predicted:.3e}); "
          f"cadence {e_cadence:.1f} ({e_cadence / predicted:.3e}); combined {sigma:.1f} ({sigma / predicted:.3e})")
    print(f"  difference {observed - predicted:+.1f} = {(observed - predicted) / sigma:+.2f} combined standard "
          f"deviations; tolerance {tol:.1f} ({tol / predicted:.3e} of the prediction)")
    if tol > c2.CAP * predicted:
        print(f"check 2 verdict: NO VERDICT (tolerance above {c2.CAP:g} of the prediction: not sharp)")
        return 2
    ok = abs(observed - predicted) <= tol
    print(f"check 2 verdict: {verdict(ok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
