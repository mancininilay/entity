#!/usr/bin/env python3
"""The delivered upstream of a run of the resolution pair, from bh_profile.txt (standard library).
Claude 1, 2026-11-23; the band reader of `slab_from_profile.py` of the series preparation, with
the geometry of the production box.

    upstream.py <run directory>

The rows of bh_profile.txt are means over abs(x1) <= profile_xmax (400 in the inputs of the pair).
At every output:
  band      100 <= abs(x2) < 150: density of the primaries (n0), B^2 (B0^2, physical components),
            the magnetisation proxy sigma0 <B^2> / <N> and density over field strength
            <N> / sqrt(<B^2>);
  stream    the rows 3 to 8 skin depths inside the inner edge of the replenished regions
            (inj_ypad + 10 cells from each boundary in x2): density, B^2, density over field
            strength.
Then the ranges over t >= 1000 and over t >= 2000. A reported quantity of every run (pre-
registration of the pair); no criterion here.
"""
import re
import sys
from pathlib import Path


def param(text, key):
    m = re.search(rf"^\s*{key}\s*=\s*([-0-9.eE+]+)", text, re.M)
    if m is None:
        raise SystemExit(f"{key} not in the input")
    return float(m.group(1))


def main():
    run = Path(sys.argv[1])
    text = next(run.glob("*.toml")).read_text()
    ext = re.search(r"extent\s*=\s*\[\[\s*([-0-9.eE+]+)\s*,\s*([-0-9.eE+]+)\s*\]\s*,\s*\[\s*([-0-9.eE+]+)\s*,\s*"
                    r"([-0-9.eE+]+)\s*\]\]", text)
    res = re.search(r"resolution\s*=\s*\[\s*([0-9]+)\s*,\s*([0-9]+)\s*\]", text)
    y0, y1 = float(ext.group(3)), float(ext.group(4))
    ny = int(res.group(2))
    dy = (y1 - y0) / ny
    pad = param(text, "inj_ypad")
    sigma0 = (param(text, "skindepth0") / param(text, "larmor0")) ** 2
    inner = pad + 10 * dy
    outputs = {}
    for line in (run / "bh_profile.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t = [float(x) for x in line.split()]
        outputs.setdefault((t[0], t[1]), []).append(t)
    mid = 0.5 * (y0 + y1)

    def mean(rows, col, sel):
        c = sum(r[4] for r in rows if sel(r))
        return sum(r[col] * r[4] for r in rows if sel(r)) / c if c > 0 else float("nan")

    band = lambda r: 100.0 <= abs(r[3] - mid) < 150.0  # noqa: E731
    stream = lambda r: (inner + 3 <= y1 - r[3] < inner + 8) or (inner + 3 <= r[3] - y0 < inner + 8)  # noqa: E731
    print(f"# {run.name}: sigma0 {sigma0:g}; band 100 <= abs(y) < 150; stream 3 to 8 d_e inside {inner:.3f} d_e "
          "from each boundary in y")
    print("# time  band: N  B2  proxy  N/B  |  stream: N  B2  N/B")
    rows_out = []
    for (step, time), rows in sorted(outputs.items()):
        n, b2 = mean(rows, 5, band), mean(rows, 6, band)
        ns, bs = mean(rows, 5, stream), mean(rows, 6, stream)
        v = (time, n, b2, sigma0 * b2 / n, n / b2 ** 0.5, ns, bs, ns / bs ** 0.5)
        rows_out.append(v)
        print(f"{v[0]:9.2f}  {v[1]:.4f} {v[2]:.4f} {v[3]:7.2f} {v[4]:.4f}  |  {v[5]:.4f} {v[6]:.4f} {v[7]:.4f}")
    for t0 in (1000.0, 2000.0):
        sel = [v for v in rows_out if v[0] >= t0]
        if not sel:
            continue
        rng = lambda k: (min(v[k] for v in sel), max(v[k] for v in sel))  # noqa: E731
        print(f"# t >= {t0:g} ({len(sel)} outputs): band N {rng(1)[0]:.3f} to {rng(1)[1]:.3f}, B2 {rng(2)[0]:.3f} to "
              f"{rng(2)[1]:.3f}, proxy {rng(3)[0]:.1f} to {rng(3)[1]:.1f}, N/B {rng(4)[0]:.3f} to {rng(4)[1]:.3f}; "
              f"stream N {rng(5)[0]:.3f} to {rng(5)[1]:.3f}, N/B {rng(7)[0]:.3f} to {rng(7)[1]:.3f}")


if __name__ == "__main__":
    main()
