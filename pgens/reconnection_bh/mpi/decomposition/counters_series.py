#!/usr/bin/env python3
"""Reported, no rule: the counters of the runs side by side at fixed times, with a finer
sampling around the opening of the boundaries (t = 750). Standard library only.

    counters_series.py <run directory> [<run directory> ...]

Per run and time: the primaries of one species present, the pairs created, the share of the
leptons of the pairs that has left the box, the mean kinetic energy of the ions and of the
leptons of the pairs. From bh_counters.txt (written every 10 steps).
"""
import sys
from pathlib import Path

TIMES = [250, 500, 700, 750, 800, 850, 900, 950, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]


def table(path):
    hdr, rows = None, []
    for line in open(path):
        if line.startswith("#"):
            hdr = line[1:].split()
        elif line.strip():
            rows.append([float(x) for x in line.split()])
    return {n: i for i, n in enumerate(hdr)}, rows


def main():
    runs = []
    for d in sys.argv[1:]:
        c, rows = table(Path(d) / "bh_counters.txt")
        picked, k = [], 0
        for t in TIMES:
            while k + 1 < len(rows) and rows[k + 1][c["time"]] <= t:
                k += 1
            picked.append(rows[k] if rows[k][c["time"]] > t - 5.0 else None)
        runs.append((Path(d).name, c, picked))
    width = max(len(n) for n, _, _ in runs) + 2
    quantities = [
        ("primaries of one species present (millions)", lambda r, c: f"{r[c['n_2']] / 1e6:8.2f}"),
        ("pairs created (thousands)", lambda r, c: f"{r[c['pairs']] / 1e3:8.1f}"),
        ("leptons of the pairs that left the box (percent of the pairs created)",
         lambda r, c: f"{100 * (1 - 0.5 * (r[c['n_3']] + r[c['n_4']]) / r[c['pairs']]):8.2f}" if r[c['pairs']] > 0 else f"{'-':>8s}"),
        ("mean K of the ions", lambda r, c: f"{r[c['K_2']] / r[c['w_2']]:8.3f}"),
        ("mean K of the leptons of the pairs",
         lambda r, c: f"{(r[c['K_3']] + r[c['K_4']]) / (r[c['w_3']] + r[c['w_4']]):8.3f}" if r[c['w_3']] + r[c['w_4']] > 0 else f"{'-':>8s}"),
    ]
    for title, f in quantities:
        print(title)
        print(f"  {'t':{width}s}" + "".join(f"{t:8d}" for t in TIMES))
        for name, c, picked in runs:
            print(f"  {name:{width}s}" + "".join(f(r, c) if r is not None else f"{'':>8s}" for r in picked))


if __name__ == "__main__":
    main()
