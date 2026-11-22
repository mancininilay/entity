"""Reference rate table of check 2 of stage B: J(Gamma_p) of the monochromatic 1.8 bath from
bhref 0.3.0 (Codex 1; frozen package, called read-only with python -B), on a grid uniform in
y = log10(Gamma_p - Gamma_th) from -6 to the automatic upper edge of the policy's tables
(rest-frame photon energy 0.05 * 1836.15). Claude 1, 2026-11-21.
    make_j_table.py <bhref/v0.3.0> <j_table.txt> <check.txt>
"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import bhref
from bhref.baths import Bath
from bhref.born import BornCumulatives
from bhref.rates import Rates
EPS0, MU_R = 1.8, 1836.15
bath = Bath({'type': 'mono', 'eps0': EPS0})
rates = Rates(bath, BornCumulatives(1024))
gth = bath.gamma_th
A = 0.05 * MU_R / EPS0
genv = 0.5 * (A + 1 / A)
y = np.linspace(-6.0, np.log10(genv * (1 - 1e-12) - gth), 2961)
g = gth + 10.0 ** y
vals = rates.many(g)
with open(sys.argv[2], 'w') as f:
    f.write(f'# bhref {bhref.__version__} mono eps0={EPS0}; gamma_th {gth:.17g}; '
            f'columns: y=log10(Gamma-gamma_th) Gamma J I\n')
    for yy, gg, v in zip(y, g, vals):
        f.write(f'{yy:.17g} {gg:.17g} {v[0]:.17g} {v[1]:.17g}\n')
ym = 0.5 * (y[1:] + y[:-1]); gm = gth + 10.0 ** ym
exact = rates.many(gm)[:, 0]
lnJ = np.log(vals[:, 0])
rel = np.exp(0.5 * (lnJ[1:] + lnJ[:-1])) / exact - 1
with open(sys.argv[3], 'w') as f:
    f.write(f'# midpoint interpolation error (ln J linear in y); upper edge Gamma {g[-1]:.9g}, K {MU_R * (g[-1] - 1):.6g}\n')
    f.write(f'max_abs_rel {np.max(np.abs(rel)):.3e} at Gamma {gm[np.argmax(np.abs(rel))]:.6f}\n')
    for lo, hi in [(gth, 1.01), (1.01, 1.05), (1.05, 1.5), (1.5, 3.0), (3.0, 26.0)]:
        m = (gm >= lo) & (gm < hi)
        f.write(f'Gamma in [{lo:.5f}, {hi:.5f}): max_abs_rel {np.max(np.abs(rel[m])):.3e}\n')
print(open(sys.argv[3]).read())
