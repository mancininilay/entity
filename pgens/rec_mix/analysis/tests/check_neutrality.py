#!/usr/bin/env python3
"""C1.3b on the first field output of a rec_mix run (smoothing 0, downsampling
1, `Charge` output, active layout). Usage: check_neutrality.py <sim_dir>"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p13ana import neutrality, readers  # noqa: E402


def main(sim, block=10):
    meta = readers.read_meta(os.path.join(sim, "usr"))
    d, inp = meta["derived"], meta["inputs"]
    assert meta["layout"] == "active"
    q, mu, th, ppc0, dt = inp["q"], inp["mu"], inp["theta_up"], d["ppc0"], d["dt"]
    h = 0.5 * ppc0
    types = []
    if q < 1:
        types.append((h * (1 - q), th, th))
    if q > 0:
        types.append((h * q, th, th / mu))
    sigma = neutrality.block_sigma(types, dt, block, ppc0)
    fp = readers.list_entity_fields(sim)[0]
    F = readers.read_entity_fields(fp, ["Charge"])
    N1, N2, y0, a = d["N1"], d["N2"], d["y0"], inp["cs_width"]
    pad = inp["injector_padding"]
    lo0 = pad + 11 + 1
    lo1 = int(y0 - 10 * a) - 1
    hi0 = int(y0 + 10 * a) + 1
    hi1 = N2 - pad - 12
    rows = [(lo0, lo1), (hi0, hi1)]
    res = neutrality.local_test(F["Charge"], rows, (0, N1), block, sigma, ppc0=ppc0)
    res.update({"step": F["step"], "rows": rows, "global_mean_charge": float(F["Charge"].mean())})
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
