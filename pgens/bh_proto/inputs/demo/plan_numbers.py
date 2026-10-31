import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../tests/hadronic/demo'))
from bhtable import Table, K_of_gamma, gamma_of_K, MU_R
T = Table(sys.argv[1])
DT, LAM = 0.5, 0.05
print("table:", T.qk_source, "sha", T.sha[:16], "nodes", len(T.gamma))
for G0, stop in ((6.0, 0.5), (3.0, 0.65), (10.0, None)):
    J0 = float(T.Jf([G0])[0]); tau0 = LAM / (J0 * DT)
    K0 = K_of_gamma(G0)
    # plain ODE dK/dt = -tau0 I(Gamma(K)), RK4 at dt/16, until <K>/K0 = stop
    if stop is None:
        print(f"G0={G0}: J={J0:.6e} tau0={tau0:.10g} <D>={float(T.Df([G0])[0]):.6g} drag/step={tau0*float(T.If([G0])[0])*DT:.6g} K0={K0:.6g}")
        continue
    K, t, h = K0, 0.0, DT / 16
    f = lambda K: -tau0 * float(T.If([gamma_of_K(K)])[0])
    npairs = 0.0; nbudget = 0.0
    while K > stop * K0:
        k1 = f(K); k2 = f(K + .5*h*k1); k3 = f(K + .5*h*k2); k4 = f(K + h*k3)
        dK = h/6*(k1+2*k2+2*k3+k4)
        G = gamma_of_K(K); D = float(T.Df([G])[0])
        npairs += -dK / D
        rth = min(10.0 * D, 0.02 * K)
        nbudget += -dK / rth
        K += dK; t += h
    print(f"G0={G0}: J={J0:.6e} tau0={tau0:.10g} K0={K0:.6g} t_stop={t:.6g} steps={t/DT:.0f} pairs/ion (a=1)={npairs:.4g} budget packets/ion (t_latency=0)={nbudget:.4g}")
