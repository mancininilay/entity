"""Reference curves for the demonstration figures (P2 step 8), written to reproduce what the registered C++ tests
compute (tests/kernels/bh_event.cpp `reference`, T5's RK4): the plain ODE dK/dt = -tau0 I(Gamma(K)) (RK4 at dt/16),
the Jensen-corrected ODE dK/dt = -tau0 [I + (1/2) I''(K) Var K(t)] with the run's measured Var K (linear in t between
records), and along it the drift-feedback variance dV/dt = -2 c V + a R E[D^2], c = tau0 dI/dK, R = tau0 J, solved
exactly over each RK4 step for frozen c and source (the C++ form), with the curvature budget (1/2) B''(K) V added in a
second variance. E[D^2]: the frozen bhref 0.2.0 moments_mono05 D2mean (restframe; log-linear in y = log10(Gamma -
Gamma_th), as load_d2) or <D>^2 from the table (delta). Centred differences with h = 0.02 K, as the C++."""
import numpy as np
from bhtable import Table, MU_R, gamma_of_K

class D2Fixture:
    def __init__(self, path, gamma_th):
        hdr = {}; rows = []
        for l in open(path):
            if l.startswith('#'):
                k, _, v = l[1:].partition(':'); hdr[k.strip()] = v.strip()
            elif l.strip():
                rows.append([float(x) for x in l.split()])
        import json
        cols = json.loads(hdr['columns']); a = np.array(rows)
        G = a[:, cols.index('gamma_p')]; D2 = a[:, cols.index('D2mean')]; D1 = a[:, cols.index('Dmean')]
        ok = (G > gamma_th) & (D2 > 0) & (D1 > 0)
        self.gamma, self.y = G[ok], np.log10(G[ok] - gamma_th); self.lnD2, self.lnD1 = np.log(D2[ok]), np.log(D1[ok])
        self.gamma_th = gamma_th; self.sha = hdr.get('data_sha256', '').strip('"')
    def _i(self, v, G):
        G = np.atleast_1d(np.asarray(G, float)); out = np.empty_like(G)
        lo_ = G <= self.gamma[0]; hi_ = G >= self.gamma[-1]; mid = ~(lo_ | hi_)
        out[lo_] = np.exp(v[0]); out[hi_] = np.exp(v[-1])
        if mid.any():
            yy = np.log10(G[mid] - self.gamma_th)
            hi = np.clip(np.searchsorted(self.y, yy, side='right'), 1, len(self.y) - 1); lo = hi - 1
            t = (yy - self.y[lo]) / (self.y[hi] - self.y[lo]); out[mid] = np.exp(v[lo] + t * (v[hi] - v[lo]))
        return out
    def D2(self, G): return self._i(self.lnD2, G)
    def D1(self, G): return self._i(self.lnD1, G)

def plain_ode(T, tau0, K0, times, dt=0.5, sub=16):
    """K(t) of dK/dt = -tau0 I(Gamma(K)), RK4 with h = dt/sub, sampled at `times`."""
    f = lambda K: -tau0 * T.If_s(1.0 + K / MU_R)
    out = []; K = K0; t = 0.0; h = dt / sub
    for te in times:
        while t < te - 1e-12 * max(1.0, te):
            hh = min(h, te - t)
            k1 = f(K); k2 = f(K + .5*hh*k1); k3 = f(K + .5*hh*k2); k4 = f(K + hh*k3)
            K += hh / 6 * (k1 + 2*k2 + 2*k3 + k4); t += hh
        out.append(K)
    return np.array(out)

def corrected(T, tau0, K0, tm, Vm, d2fun, a=1.0, h=0.5, tv=None, Vv=None):
    """The C++ `reference` with the run's records (tm, Vm): returns K_plain, K_corr, V_fb, V_fb_curv at tm.
    h is the RK4 step (the run's dt, as the C++). (tv, Vv), default (tm, Vm), is the variance record the Jensen term
    interpolates linearly in t; it must start with the known initial record (0, 0), as the C++ reference records t = 0
    (bh_event.cpp `cooling_run`): np.interp holds the first value to the left, so a record starting at the first
    output time would invent Var K > 0 back to t = 0 (Codex 1's finding F1, 2026-11-01)."""
    I = lambda K: T.If_s(1.0 + K / MU_R); J = lambda K: T.Jf_s(1.0 + K / MU_R)
    Ipp = lambda K: (I(K + .02*K) - 2*I(K) + I(K - .02*K)) / (.02*K)**2
    Ip = lambda K: (I(K + .02*K) - I(K - .02*K)) / (2 * .02*K)
    B = lambda K: a * tau0 * J(K) * float(np.ravel(d2fun(gamma_of_K(K)))[0])
    Bpp = lambda K: (B(K + .02*K) - 2*B(K) + B(K - .02*K)) / (.02*K)**2
    tv, Vv = (tm, Vm) if tv is None else (tv, Vv)
    Vat = lambda t: float(np.interp(t, tv, Vv))
    fp = lambda K: -tau0 * I(K)
    fc = lambda K, t: -tau0 * (I(K) + .5 * Ipp(K) * Vat(t))
    Kp = Kc = K0; t = 0.0; vf = vf2 = 0.0
    R = {'K_plain': [], 'K_corr': [], 'V_fb': [], 'V_fb_curv': []}
    for te in tm:
        while t < te - 1e-12 * max(1.0, te):
            hh = min(h, te - t)
            k1 = fp(Kp); k2 = fp(Kp + .5*hh*k1); k3 = fp(Kp + .5*hh*k2); k4 = fp(Kp + hh*k3); Kp += hh/6*(k1 + 2*k2 + 2*k3 + k4)
            Kc_prev = Kc; g0 = B(Kc)
            k1 = fc(Kc, t); k2 = fc(Kc + .5*hh*k1, t + .5*hh); k3 = fc(Kc + .5*hh*k2, t + .5*hh); k4 = fc(Kc + hh*k3, t + hh)
            Kc += hh/6*(k1 + 2*k2 + 2*k3 + k4)
            g1 = B(Kc); c = tau0 * Ip(.5*(Kc_prev + Kc)); g = .5*(g0 + g1); ex = np.exp(-2*c*hh)
            vf = vf*ex + (g*(1 - ex)/(2*c) if abs(c) > 1e-300 else g*hh)
            g2 = g + .5 * Bpp(.5*(Kc_prev + Kc)) * Vat(t + .5*hh)
            vf2 = vf2*ex + (g2*(1 - ex)/(2*c) if abs(c) > 1e-300 else g2*hh)
            t += hh
        R['K_plain'].append(Kp); R['K_corr'].append(Kc); R['V_fb'].append(vf); R['V_fb_curv'].append(vf2)
    return {k: np.array(v) for k, v in R.items()}
