#!/usr/bin/env python3
"""Check 3: the run with the policy against the run without.

    check3_compare.py <bp2raw> <cache> <run without> <run with>

Reconnection rate, from the field output: the inflow speed towards the sheet,
-sign(y) (E x B)_y / B^2, averaged over the slabs 0.4 <= |y| / (L_y / 2) <= 0.6
and |x| <= 0.8 L_x / 2, separately in the four quadrants (above and below the
sheet, left and right half); the rate of an output is the mean of the four;
outputs with t >= runtime / 3.

Spectra of the electrons and of the ions (species 1 and 2), from the spectra
output with two spatial bins (left and right half of the box); outputs with
t >= 2 runtime / 3. Scalars of a spectrum: mean K, the 99 and 99.9 percent
quantiles of K, the logarithmic slope of dN / d ln K between sigma / 2 and
2 sigma, and for the ions the fraction above the threshold.

For every scalar Q: Q_bar, its mean over the window, for the whole box, and a
reference scatter s_ref from the sub-samples of the same run: the standard
deviation of the mean of the sub-samples (four quadrants, or two halves),
root-mean-square over the window. Two runs agree when
|Q_bar(with) - Q_bar(without)| <= 2 sqrt(s_ref(with)^2 + s_ref(without)^2).
The sub-samples of one run are not independent realisations: the scatter
between two runs with different random streams can be larger.
"""
import sys

import numpy as np

from bhio import Run, threshold, verdict


def rates(run):
    x0, x1, y0, y1 = run.extent()
    cx, cy, hx, hy = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (x1 - x0), 0.5 * (y1 - y0)
    runtime = run.param("runtime")
    rows = []
    for bp, step, time in run.stamps("fields"):
        if time < runtime / 3.0:
            continue
        v = run.read(bp, ["X1", "X2", "fE1", "fE3", "fB1", "fB2", "fB3"])
        x, y = v["X1"] - cx, v["X2"] - cy
        e1, e3 = v["fE1"].astype(np.float64), v["fE3"].astype(np.float64)
        b1, b2, b3 = (v[k].astype(np.float64) for k in ("fB1", "fB2", "fB3"))
        assert e1.shape == (x.size, y.size), (e1.shape, x.size, y.size)
        vy = (e3 * b1 - e1 * b3) / (b1 * b1 + b2 * b2 + b3 * b3)
        quad = []
        for ysign in (1.0, -1.0):
            my = (ysign * y >= 0.4 * hy) & (ysign * y <= 0.6 * hy)
            for xsign in (-1.0, 1.0):
                mx = (xsign * x > 0.0) & (np.abs(x) <= 0.8 * hx)
                quad.append(-ysign * vy[np.ix_(mx, my)].mean())
        rows.append([time] + quad)
    return np.array(rows)


def quantile(counts, edges, q):
    """Energy below which the fraction q of the particles lies (log-linear in a bin)."""
    cum = np.cumsum(counts)
    target = q * cum[-1]
    i = int(np.searchsorted(cum, target))
    below = cum[i - 1] if i > 0 else 0.0
    frac = (target - below) / counts[i] if counts[i] > 0 else 0.0
    return float(np.exp(np.log(edges[i]) + frac * (np.log(edges[i + 1]) - np.log(edges[i]))))


def scalars(counts, edges, sigma, k_th):
    centre = np.sqrt(edges[1:] * edges[:-1])
    total = counts.sum()
    out = [float((counts * centre).sum() / total), quantile(counts, edges, 0.99),
           quantile(counts, edges, 0.999)]
    m = (centre >= 0.5 * sigma) & (centre <= 2.0 * sigma) & (counts > 0)
    out.append(float(np.polyfit(np.log(centre[m]), np.log(counts[m]), 1)[0]) if m.sum() >= 5
               else float("nan"))
    # fraction above the threshold: whole bins above it and the share of the bin it cuts
    i = int(np.searchsorted(edges, k_th)) - 1
    part = (np.log(edges[i + 1]) - np.log(k_th)) / (np.log(edges[i + 1]) - np.log(edges[i]))
    out.append(float((counts[i + 1:].sum() + part * counts[i]) / total))
    return out


NAMES = ["mean K", "K 99 percent", "K 99.9 percent", "slope sigma/2..2 sigma", "fraction above threshold"]


def spectra(run, species, k_th):
    runtime, sigma = run.param("runtime"), run.sigma()
    rows = []
    for bp, step, time in run.stamps("spectra"):
        if time < 2.0 * runtime / 3.0:
            continue
        v = run.read(bp, ["sEbn", f"sN_{species}"])
        n = v[f"sN_{species}"].astype(np.float64).reshape(2, -1)
        edges = v["sEbn"].astype(np.float64)
        rows.append([scalars(n[0] + n[1], edges, sigma, k_th), scalars(n[0], edges, sigma, k_th),
                     scalars(n[1], edges, sigma, k_th)])
    return np.array(rows)          # (outputs, [whole, left, right], scalars)


def compare(label, q_off, s_off, q_low, s_low, n_off, n_low):
    tol = 2.0 * np.sqrt(s_off ** 2 + s_low ** 2)
    ok = abs(q_low - q_off) <= tol
    print(f"  {label}: without {q_off:.6g} (scatter {s_off:.3g}, {n_off} outputs); with {q_low:.6g} "
          f"(scatter {s_low:.3g}, {n_low} outputs); difference {q_low - q_off:+.4g} "
          f"({(q_low - q_off) / q_off:+.3%}), tolerance {tol:.4g}  {verdict(ok)}")
    return ok


def main():
    bp2raw, cache = sys.argv[1], sys.argv[2]
    off, low = Run(sys.argv[3], bp2raw, cache), Run(sys.argv[4], bp2raw, cache)
    _, k_th = threshold(low.param("photon_energy"), low.param("mass_ratio"))
    n_fail = 0
    print(f"check 3: {low.name} against {off.name}")
    r = {}
    for run in (off, low):
        a = rates(run)
        if a.shape[0] < 3:
            raise SystemExit("check 3 needs at least three field outputs in the window")
        per_output = a[:, 1:].mean(axis=1)
        s_ref = np.sqrt(np.mean(a[:, 1:].var(axis=1, ddof=1) / 4.0))
        r[run.name] = (per_output.mean(), s_ref, a.shape[0], per_output.std(ddof=1))
        print(f"  {run.name}: rate per output " + " ".join(f"{x:.4f}" for x in per_output))
    (qo, so, no, to), (ql, sl, nl, tl) = r[off.name], r[low.name]
    n_fail += not compare("reconnection rate", qo, so, ql, sl, no, nl)
    print(f"    reported: standard deviation over the outputs, without {to:.3g}, with {tl:.3g}")
    for species, label in ((1, "electrons"), (2, "ions")):
        so_, sl_ = spectra(off, species, k_th), spectra(low, species, k_th)
        if so_.shape[0] < 3 or sl_.shape[0] < 3:
            raise SystemExit("check 3 needs at least three spectra in the window")
        for j, name in enumerate(NAMES):
            if species == 1 and j == 4:
                continue
            ref = []
            for a in (so_, sl_):
                ref.append(np.sqrt(np.mean((a[:, 1, j] - a[:, 2, j]) ** 2)) / 2.0)
            n_fail += not compare(f"{label}, {name}", so_[:, 0, j].mean(), ref[0],
                                  sl_[:, 0, j].mean(), ref[1], so_.shape[0], sl_.shape[0])
    print(f"check 3 verdict: {verdict(n_fail == 0)} ({n_fail} of 10 comparisons outside the tolerance)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
