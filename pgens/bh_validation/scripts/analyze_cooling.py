#!/usr/bin/env python3
"""Mean cooling (V2) and straggling (V3) of a bh_validation run.

    analyze_cooling.py <run dir> [--targets 0.9,0.7,0.5] [--d2 moments_mono05.txt]
                       [--no-verdict-v3]

Reads `run.log` (the BHV lines), `input.toml` and `bhv_tables.txt` of the run.

V2 (registered): at the first record with <K>/K0 below each target,
|<K> - K_corr| <= 4 s.e., s.e. = sigma_K / sqrt(N), K_corr the table ODE
d<K>/dt = -tau [I(<K>) + (1/2) I''(<K>) Var K(t)] with I'' by centred
differences (h = 0.02 K) and Var K(t) the run's own variance.

V3 (registered): Var K against dV/dt = -2 tau (dI/dK) V + a R (E[D^2] - p <D>^2),
E[D^2] from the frozen reference moments, p = tau J dt / a, R = tau J; tolerance
4 sqrt(2/N) at the records where sigma_K/K first exceeds 0.005 and 0.01 and at
the last record with sigma_K/K <= 0.02 and curvature budget <= 0.6 %.
"""
import argparse
import math
import os
import re
import sys

from bisect import bisect_right


def read_series(run_dir):
    rows = []
    for line in open(os.path.join(run_dir, "run.log")):
        if line.startswith("BHV step"):
            t = line.split()
            rows.append(dict(zip(t[1::2], (float(x) for x in t[2::2]))))
    return rows


def read_input(run_dir):
    text = open(os.path.join(run_dir, "input.toml")).read()

    def get(key, cast=float, default=None):
        m = re.search(rf"^\s*{key}\s*=\s*(.+)$", text, re.M)
        if not m:
            return default
        return cast(m.group(1).strip().strip('"'))

    return dict(tau=get("optical_depth"), a=get("pair_weight"), mu_r=get("mass_ratio"),
                gamma_p=get("gamma_p"), n_ions=get("n_ions"))


class Tables:
    def __init__(self, path):
        head = open(path).readline().split()
        self.gamma_th, self.gamma_min, self.gamma_max = float(head[2]), float(head[4]), float(head[6])
        d = [[float(x) for x in l.split()] for l in open(path) if l.strip() and not l.startswith("#")]
        self.y = [r[0] for r in d]
        self.gamma = [r[1] for r in d]
        self.lnJ = [math.log(r[2]) for r in d]
        self.lnI = [math.log(r[3]) for r in d]

    def _interp(self, lnv, G):
        # as the code: zero below the lower edge, the edge value above the upper edge,
        # linear in y = log10(Gamma - Gamma_th) of the logarithm
        if not (G > self.gamma_th) or not (G >= self.gamma_min):
            return 0.0
        if G >= self.gamma_max:
            return math.exp(lnv[-1])
        yy = math.log10(G - self.gamma_th)
        if yy < self.y[0]:
            return 0.0
        if yy >= self.y[-1]:
            return math.exp(lnv[-1])
        hi = bisect_right(self.y, yy)
        lo = hi - 1
        f = (yy - self.y[lo]) / (self.y[hi] - self.y[lo])
        return math.exp(lnv[lo] + f * (lnv[hi] - lnv[lo]))

    def I(self, G):
        return self._interp(self.lnI, G)

    def J(self, G):
        return self._interp(self.lnJ, G)


class Moments:
    """E[D^2] of the frozen reference, log-linear in y between its nodes."""

    def __init__(self, path, gamma_th):
        cols = None
        for line in open(path):
            if line.startswith("# columns:"):
                cols = eval(line.split(":", 1)[1])
                break
        d = [[float(x) for x in l.split()] for l in open(path) if l.strip() and not l.startswith("#")]
        ig, i2 = cols.index("gamma_p"), cols.index("D2mean")
        d = [r for r in d if r[ig] > gamma_th and r[i2] > 0]
        self.gamma = [r[ig] for r in d]
        self.lnD2 = [math.log(r[i2]) for r in d]
        self.gamma_th = gamma_th
        self.y = [math.log10(g - gamma_th) for g in self.gamma]

    def at(self, G):
        if G <= self.gamma[0]:
            return math.exp(self.lnD2[0])
        if G >= self.gamma[-1]:
            return math.exp(self.lnD2[-1])
        yy = math.log10(G - self.gamma_th)
        hi = bisect_right(self.y, yy)
        lo = hi - 1
        f = (yy - self.y[lo]) / (self.y[hi] - self.y[lo])
        return math.exp(self.lnD2[lo] + f * (self.lnD2[hi] - self.lnD2[lo]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--targets", default="0.9,0.7,0.5")
    ap.add_argument("--d2", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    targets = [float(x) for x in args.targets.split(",")]
    rows = read_series(args.run_dir)
    inp = read_input(args.run_dir)
    T = Tables(os.path.join(args.run_dir, "bhv_tables.txt"))
    tag = args.tag or os.path.basename(os.path.normpath(args.run_dir))
    mu_r, tau, a = inp["mu_r"], inp["tau"], inp["a"]
    K0 = rows[0].get("K_ref", mu_r * (inp["gamma_p"] - 1.0))
    dt = rows[1]["time"] - rows[0]["time"]
    dt /= (rows[1]["step"] - rows[0]["step"])
    N = rows[0]["ion_w"]
    # state after step n is at t = (n + 1) dt; the initial state is prepended
    t = [0.0] + [(r["step"] + 1.0) * dt for r in rows]
    mean = [K0] + [K0 + r["ion_dK"] / r["ion_w"] for r in rows]
    var = [0.0] + [(r["ion_dK2"] / r["ion_w"] - (r["ion_dK"] / r["ion_w"]) ** 2) * N / (N - 1.0) for r in rows]
    pairs = rows[-1]["pairs"]
    print(f"== {tag}: N = {N:.0f}, K0 = {K0:.6g}, dt = {dt:.9g}, tau = {tau:.9g}, a = {a:g}, "
          f"p(Gamma_p0) = {tau * T.J(inp['gamma_p']) * dt / a:.5f}, records = {len(rows)}, steps = {rows[-1]['step'] + 1:.0f}, "
          f"<K>/K0 at the end = {mean[-1] / K0:.5f}, pairs = {pairs:.0f} ({pairs / N:.4f} per emitter)")
    print(f"   counters at the end: above {rows[-1]['above']:.0f}, probability exceeded {rows[-1]['pexc']:.0f}, "
          f"bounded {rows[-1]['bounded']:.0f}, dropped {rows[-1]['dropped']:.0f}")

    I_of_K = lambda K: T.I(1.0 + K / mu_r)
    J_of_K = lambda K: T.J(1.0 + K / mu_r)

    def Ipp(K):
        h = 0.02 * K
        return (I_of_K(K + h) - 2.0 * I_of_K(K) + I_of_K(K - h)) / (h * h)

    def Ip(K):
        h = 0.02 * K
        return (I_of_K(K + h) - I_of_K(K - h)) / (2.0 * h)

    def var_at(tt):
        # linear interpolation of the measured Var K(t)
        if tt <= t[0]:
            return var[0]
        if tt >= t[-1]:
            return var[-1]
        hi = bisect_right(t, tt)
        lo = hi - 1
        return var[lo] + (tt - t[lo]) / (t[hi] - t[lo]) * (var[hi] - var[lo])

    d2 = None
    if args.d2:
        d2 = Moments(args.d2, T.gamma_th)

    def B_of_K(K):
        # variance injected per unit time: a R (E[D^2] - p <D>^2)
        J = J_of_K(K)
        if J <= 0.0:
            return 0.0
        G = 1.0 + K / mu_r
        p = tau * J * dt / a
        Dm = I_of_K(K) / J
        return a * tau * J * (d2.at(G) - p * Dm * Dm)

    def B_poisson(K):
        return a * tau * J_of_K(K) * d2.at(1.0 + K / mu_r)

    def Bpp(K):
        h = 0.02 * K
        return (B_of_K(K + h) - 2.0 * B_of_K(K) + B_of_K(K - h)) / (h * h)

    fp = lambda K: -tau * I_of_K(K)
    fc = lambda K, tt: -tau * (I_of_K(K) + 0.5 * Ipp(K) * var_at(tt))
    Kp = Kc = K0
    tt = 0.0
    vf = vf2 = vpois = 0.0
    K_plain, K_corr, var_fb, var_fb_curv, var_fb_poisson = [], [], [], [], []
    h = dt
    for t_end in t:
        while tt < t_end - 1e-12 * max(1.0, t_end):
            hh = min(h, t_end - tt)
            k1 = fp(Kp); k2 = fp(Kp + 0.5 * hh * k1); k3 = fp(Kp + 0.5 * hh * k2); k4 = fp(Kp + hh * k3)
            Kp += hh / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            Kc_prev = Kc
            k1 = fc(Kc, tt); k2 = fc(Kc + 0.5 * hh * k1, tt + 0.5 * hh)
            k3 = fc(Kc + 0.5 * hh * k2, tt + 0.5 * hh); k4 = fc(Kc + hh * k3, tt + hh)
            Kc += hh / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            if d2 is not None:
                g0, g1 = B_of_K(Kc_prev), B_of_K(Kc)
                Km = 0.5 * (Kc_prev + Kc)
                c = tau * Ip(Km)
                g = 0.5 * (g0 + g1)
                ex = math.exp(-2.0 * c * hh)
                gain = (1.0 - ex) / (2.0 * c) if abs(c) > 1e-300 else hh
                vf = vf * ex + g * gain
                g2 = g + 0.5 * Bpp(Km) * var_at(tt + 0.5 * hh)
                vf2 = vf2 * ex + g2 * gain
                gp = 0.5 * (B_poisson(Kc_prev) + B_poisson(Kc))
                vpois = vpois * ex + gp * gain
            tt += hh
        K_plain.append(Kp); K_corr.append(Kc)
        var_fb.append(vf); var_fb_curv.append(vf2); var_fb_poisson.append(vpois)

    n_fail = 0
    # ---- V2
    for f in targets:
        idx = [i for i in range(len(mean)) if mean[i] <= f * K0]
        if len(idx) == 0:
            print(f"   V2 target <K>/K0 = {f:.2f} NOT REACHED (run ends at {mean[-1] / K0:.4f}): no verdict")
            n_fail += 1
            continue
        i = int(idx[0])
        se = math.sqrt(var[i] / N)
        diff = mean[i] - K_corr[i]
        ok = abs(diff) <= 4.0 * se
        n_fail += 0 if ok else 1
        print(f"   {'PASS' if ok else 'FAIL'} V2 at <K>/K0 = {f:.2f} (t = {t[i]:.6g}, <K> = {mean[i]:.6f}): "
              f"<K> - K_corr = {diff:+.4g} = {diff / se:+.3f} s.e. (s.e. = {se:.4g}, limit 4); plain ODE {(mean[i] - K_plain[i]) / se:+.3f} s.e.; "
              f"Jensen term {(K_corr[i] - K_plain[i]) / (4.0 * se):.3f} of 4 s.e.; sigma_K/K = {math.sqrt(var[i]) / mean[i]:.4g}")
    # ---- V3
    if d2 is not None:
        tol = 4.0 * math.sqrt(2.0 / N)
        rel = [math.sqrt(max(v, 0.0)) / m for v, m in zip(var, mean)]
        curv = [abs(c / v - 1.0) if v > 0 else 1.0 for c, v in zip(var_fb_curv, var_fb)]
        i005 = next((i for i in range(1, len(t)) if rel[i] >= 0.005), None)
        i01 = next((i for i in range(1, len(t)) if rel[i] >= 0.01), None)
        last_in = None
        last_old = None
        for i in range(1, len(t)):
            if rel[i] <= 0.02:
                last_old = i
                if curv[i] <= 0.006:
                    last_in = i
        pts = [("sigma_K/K = 0.005", i005), ("sigma_K/K = 0.01", i01),
               ("last record in the domain (sigma_K/K <= 0.02, curvature <= 0.6 %)", last_in)]
        if last_old != last_in:
            pts.append(("deepest record with sigma_K/K <= 0.02 [outside the domain]", last_old))
        for name, i in pts:
            if i is None or not var_fb[i] > 0:
                # as in the fork's verdict of test E3: a point the run does not reach has no verdict
                print(f"   (diag) V3 point ({name}) not in the run: no verdict")
                continue
            ratio = var[i] / var_fb[i]
            dKK = (K0 - K_corr[i]) / K0
            if curv[i] <= 0.006:
                ok = abs(ratio - 1.0) <= tol
                n_fail += 0 if ok else 1
                print(f"   {'PASS' if ok else 'FAIL'} V3 at {name} (t = {t[i]:.6g}, Delta K/K = {dKK:.4f}): Var_meas = {var[i]:.6g}, "
                      f"Var_pred = {var_fb[i]:.6g}, |ratio - 1| = {abs(ratio - 1.0):.4g} (limit {tol:.4g}); curvature budget {100 * curv[i]:.3f} %; "
                      f"with the Poisson source instead: |ratio - 1| = {abs(var[i] / var_fb_poisson[i] - 1.0):.4g}")
            else:
                print(f"   (diag) V3 at {name} (t = {t[i]:.6g}, Delta K/K = {dKK:.4f}) outside the registered domain (curvature {100 * curv[i]:.3f} %): "
                      f"|ratio - 1| = {abs(ratio - 1.0):.4g}, curvature-corrected {abs(var[i] / var_fb_curv[i] - 1.0):.4g} (limit {tol:.4g}); no verdict")
    print(f"   {tag}: {'ALL PASS' if n_fail == 0 else str(n_fail) + ' FAIL'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
