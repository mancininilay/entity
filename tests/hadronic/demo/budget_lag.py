"""One-ion model of the budget source's emission rule (bethe_heitler.hpp at e7ba238c, steps 1 to 4 for `budget`), to
explain the budget number yield of the demonstration (figure 4). Analysis code, not a ctest.

All ions of a budget run are identical (deterministic loss; the restframe spectrum enters only the pair sampling), so
one ion reproduces the packet positions. The model: per step dK = tau0 I(Gamma_eff) dt (Euler, dt = 0.5) into the
pending payload; application when pending >= min(eta_u K, r_th_pre) (the double-precision floor is negligible); debt
r += applied loss, K0 set when r goes from 0 to positive; packets: the K0 trigger |K - K0| > f_r K0 pays the whole debt,
otherwise r_th = min(r_th_factor <D>(Gamma), f_r K) is paid while r >= r_th; K0 reset when r returns to 0.

The lag bias: the debt is paid in order, so packet i pays the path interval [K0 - Q_i, K0 - Q_i + r_i] (Q_i the paid
energy through packet i) but its pairs are drawn at the emission point K_i = K0 - Q_i - d_i, below that interval by the
debt d_i left after the packet; N = sum r_i / <D>(K_i) exceeds the physical integral of dK / <D> over the paid interval.
To first order, with g = d ln<D>/dK (Codex 1's derivation, finding F3 of 2026-11-01), the whole-run excess is the
PAIR-NUMBER-weighted sum (r_i/<D>_i) g_i (r_i/2 + d_i) / sum (r_i/<D>_i) (`est_n`); the energy-weighted
sum r_i^2 g_i / (2 sum r_i) (`est`) drops the 1/<D> weights and d: an approximation, exact only for constant <D> and d = 0.
Usage: budget_lag.py <campaign output root> <Gamma_p0> [run name]"""
import json, sys
import numpy as np
from bhtable import Table, MU_R

ETA_U, R_TH_FACTOR, F_R, DT = 1e-4, 10.0, 0.02, 0.5
TAU0S = {6.0: 0.2564289235, 3.0: 7.862350556}; NSTEPS = {6.0: 39600, 3.0: 130000}
T = None

def integral(Ka, Kb, n=64):
    """int_{Kb}^{Ka} dK / <D>(K) by Simpson."""
    if Ka <= Kb: return 0.0
    x = np.linspace(Kb, Ka, 2 * n + 1); f = np.array([1.0 / T.Df_s(1.0 + k / MU_R) for k in x]); h = (Ka - Kb) / (2 * n)
    return h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())

def model(OUT, G0):
  """the packet list of one ion and the lag bias over the emitted interval"""
  global T
  if T is None: T = Table(f'{OUT}/hadronic_table_double.txt')
  TAU0 = TAU0S[G0]; NSTEP = NSTEPS[G0]
  K0i = MU_R * (G0 - 1); K = K0i; pend = 0.0; r = 0.0; Kk0 = 0.0
  packets = []          # (K at emission, r_packet)
  nk0 = 0
  for step in range(NSTEP):
      Keff = K - pend
      pend += TAU0 * T.If_s(1.0 + Keff / MU_R) * DT
      rth_pre = min(R_TH_FACTOR * T.Df_s(1.0 + K / MU_R), F_R * K)
      if pend >= min(ETA_U * K, rth_pre):
          if r == 0.0: Kk0 = K - pend
          K -= pend; r += pend; pend = 0.0
          rth = min(R_TH_FACTOR * T.Df_s(1.0 + K / MU_R), F_R * K)
          k0 = Kk0 > 0 and abs(K - Kk0) > F_R * Kk0
          while r > 0:
              if k0: rp = r; nk0 += 1
              elif r >= rth: rp = rth
              else: break
              r -= rp; packets.append((K, rp, r))   # (emission K, packet energy, debt left after it)
              if k0: break
          if r == 0.0: Kk0 = 0.0
  P = np.array(packets)
  N = float(np.sum(P[:, 1] / np.array([T.Df_s(1.0 + k / MU_R) for k in P[:, 0]])))
  Q = float(P[:, 1].sum()); Klast = float(P[-1, 0])
  # physical number over the emitted energy: the debt is paid in order, so the emitted energy is the K interval
  # [K0 - Q, K0] of the path (the last packet's remainder, if any, is the unpaid debt at the bottom)
  Nphys = integral(K0i, K0i - Q)
  ED = np.array([T.Df_s(1.0 + k / MU_R) for k in P[:, 0]])
  dlnD = np.gradient(np.log(ED), P[:, 0])
  est = float(np.sum(P[:, 1] ** 2 / 2 * dlnD) / Q)                       # energy-weighted: an approximation (F3)
  lnD = lambda k: np.log(T.Df_s(1.0 + k / MU_R))
  g = np.array([(lnD(k + 0.5) - lnD(k - 0.5)) for k in P[:, 0]])           # d ln<D>/dK at the emission points (h = 0.5)
  wn = P[:, 1] / ED
  est_n = float(np.sum(wn * g * (P[:, 1] / 2 + P[:, 2])) / np.sum(wn))     # number-weighted with the debt left d (F3)
  return {'P': P, 'nk0': nk0, 'N': N, 'Q': Q, 'Nphys': Nphys, 'bias': N / Nphys - 1, 'est': est, 'est_n': est_n,
          'Kfinal': K - pend, 'debt': r}

if __name__ == '__main__':
  OUT = sys.argv[1]; G0 = float(sys.argv[2]); RUN = sys.argv[3] if len(sys.argv) > 3 else f'demo_g{int(G0)}_budget_delta_double'
  M = model(OUT, G0); P = M['P']; nk0 = M['nk0']; K = M['Kfinal']; pend = 0.0; r = M['debt']; Q = M['Q']; N = M['N']; Nphys = M['Nphys']; est = M['est']
  K0i = MU_R * (G0 - 1); Klast = float(P[-1, 0])
  print(f'model Gamma_p0 = {G0:g}: packets/ion {len(P)}, K0 triggers {nk0}, Gp at emission {1 + P[:, 0].max() / MU_R:.6f} .. '
      f'{1 + Klast / MU_R:.6f}, final K {K - pend:.3f}, debt left {r:.3f}, Q {Q:.3f}, N {N:.5f}')
  print(f'  lag bias N / int dK/<D> over the emitted interval - 1 = {N / Nphys - 1:+.4%}; first order, number-weighted with the '
        f'debt left: {M["est_n"]:+.4%}; energy-weighted approximation sum r^2/2 dln<D>/dK / Q: {est:+.4%}')
  try:
    z = np.load(f'{OUT}/{RUN}/reduced.npz'); j = json.load(open(f'{OUT}/{RUN}/reduced.json')); lf = j['log']['ledger_final']; NI = 1e5
    Nr = j['pairs']['w'] / NI; Qr = j['pairs']['wD'] / NI
    print(f'run {RUN}: packets/ion {lf["n_spawned"] / NI:g}, K0 triggers/ion {lf["n_k0_trigger"] / NI:g}, Gp at emission '
          f'{j["pairs"]["Gp_max"]:.6f} .. {j["pairs"]["Gp_min"]:.6f}, final K {z["blocks"][-1, 3]:.3f}, debt left {lf["debt_outstanding"] / NI:.3f}, '
          f'Q {Qr:.3f}, N {Nr:.5f}')
    print(f'  N / int dK/<D> over the emitted interval [K0 - Q, K0] - 1 = {Nr / integral(K0i, K0i - Qr) - 1:+.4%}')
  except (FileNotFoundError, KeyError):
    print(f'run {RUN}: not available')
