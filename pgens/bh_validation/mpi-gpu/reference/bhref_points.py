"""Reference numbers of M1 from bhref 0.3.0 (Codex 1, frozen), called read-only.

Claude 1 (local Claude Code, second session), 2026-11-22. Nothing of the
Entity policy is used here.
    python -B bhref_points.py <.../bhref/v0.3.0> <out dir>
Writes points.txt: for each case J, I, <D> = I / J and d ln J / d Gamma
(central difference, h = 1e-4), with the 512-panel values beside the
1024-panel ones.
"""
import sys
from pathlib import Path
import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root))
import bhref
from bhref.baths import Bath
from bhref.born import BornCumulatives
from bhref.rates import Rates

MU_R = 1836.15
cum, cum_med = BornCumulatives(1024), BornCumulatives(512)
with open(out / 'points.txt', 'w') as f:
    f.write(f'# bhref {bhref.__version__} frozen={bhref.__frozen__} mu_r={MU_R}\n')
    f.write('# case eps0 Gamma_p K J I Dmean dlnJ/dGamma J_512panels I_512panels\n')
    for case, eps0, g in [('H', 0.5, 6.0), ('L', 1.8, 1.2)]:
        bath = Bath({'type': 'mono', 'eps0': eps0})
        r, rm = Rates(bath, cum), Rates(bath, cum_med)
        J, I, D, _ = r.at(g)
        Jm, Im, _, _ = rm.at(g)
        h = 1e-4
        s = (np.log(r.at(g + h)[0]) - np.log(r.at(g - h)[0])) / (2 * h)
        f.write(f'{case} {eps0} {g:.17g} {MU_R * (g - 1):.17g} {J:.17g} {I:.17g} {D:.17g} {s:.17g} {Jm:.17g} {Im:.17g}\n')
print(open(out / 'points.txt').read())
