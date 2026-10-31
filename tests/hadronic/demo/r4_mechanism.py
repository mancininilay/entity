"""R4 mechanism check (author's calculation, P2 step 8): the drift-feedback variance prediction along the Gamma_p0 = 3
restframe event path with the fitted step-1 table versus the Born table, everything else equal (J is the same function
of Gamma_p in both; E[D^2] the frozen bhref 0.2.0 D2mean; tau0 from J at Gamma_p0 = 3). The event dynamics follow the
Born law in both cases (the sampler draws Born outcomes; the table enters the event mode only through J); only the
prediction's feedback coefficient c = tau0 dI/dK and its mean path change with the table. Two passes: the Jensen term
fed with the pass-1 prediction. Usage: r4_mechanism.py <born table> <fit table> <bhref-0.2.0 dir>"""
import sys, numpy as np
from bhtable import Table, MU_R, gamma_of_K
from refs import D2Fixture, corrected
Tb, Tf = Table(sys.argv[1]), Table(sys.argv[2]); d2 = D2Fixture(sys.argv[3] + '/moments_mono05.txt', Tb.gamma_th)
G0 = 3.0; K0 = MU_R * (G0 - 1); lam, dt = 0.05, 0.5
tm = np.arange(12.5, 15800.1, 12.5)
for name, T in (('born', Tb), ('fit', Tf)):
    tau0 = lam / (float(T.Jf([G0])[0]) * dt)
    R1 = corrected(T, tau0, K0, tm, np.zeros_like(tm), lambda G: d2.D2(G))
    R = corrected(T, tau0, K0, tm, R1['V_fb'], lambda G: d2.D2(G))
    globals()[name] = (tau0, R)
tb, Rb = born; tf, Rf = fit
print(f"tau0 born {tb:.10g}, fit {tf:.10g}, relative {tf/tb-1:.2e} (J identical up to the grid interpolation)")
for t_ in (150.0, 1100.0, 5000.0, 10000.0, 15200.0, 15400.0, 15787.5):
    i = int(np.argmin(np.abs(tm - t_)))
    dK = (K0 - Rb['K_corr'][i]) / K0
    print(f"t = {tm[i]:8.1f}  dK/K (born path) = {dK:.4f}  V_fb fit / V_fb born - 1 = {Rf['V_fb'][i]/Rb['V_fb'][i]-1:+.4%}  "
          f"curvature budget (born) = {Rb['V_fb_curv'][i]/Rb['V_fb'][i]-1:+.4%}  K_corr fit / born - 1 = {Rf['K_corr'][i]/Rb['K_corr'][i]-1:+.2e}")
# the feedback coefficients themselves at a few K
for G in (3.0, 2.8, 2.6, 2.5, 2.45):
    K = MU_R * (G - 1); h = 0.02 * K
    cb = (Tb.If([gamma_of_K(K + h)])[0] - Tb.If([gamma_of_K(K - h)])[0]) / (2 * h); cf = (Tf.If([gamma_of_K(K + h)])[0] - Tf.If([gamma_of_K(K - h)])[0]) / (2 * h)
    print(f"Gamma_p = {G}: dI/dK fit / born - 1 = {cf/cb-1:+.4%};  I fit / born - 1 = {Tf.If([G])[0]/Tb.If([G])[0]-1:+.4%}")
