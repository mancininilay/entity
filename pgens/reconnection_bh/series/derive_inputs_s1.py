#!/usr/bin/env python3
"""Inputs of step 1 of the resolved series, derived from ../r-low.toml by listed edits.
PREPARED, NOT RUN: no job is authorised and three choices are open (see README.md).

    derive_inputs_s1.py <directory with r-low.toml> <output directory> [--decomposition 2,2|4,1]
                        [--runtime 3000|9000] [--injector 1|2] [--toy]

The resolution pair at ten times the optical depth of R-low (0.14333), with the corrected
replenishing rule, on four ranks:

  s1-n-{a,b,c}.toml   the present grid, 4096 x 2048 cells
  s1-f-{a,b,c}.toml   twice the resolution, 8192 x 4096 cells (the time step halves with the cell)

a, b, c are realisations: the same physics, `setup.seed` absent, 1 and 2. With --runtime 9000 one
input per grid is written instead (s1-n-long.toml, s1-f-long.toml, seed absent).

Edits against r-low.toml, all runs: the optical depth; the decomposition; the capacities per rank
(see below); the corrected replenishing rule (`setup.injector`, 2 by default: the target follows
the field; 1: a fixed target); its blocks 15.6 skin depths long and three rows of the present grid
high (64 cells and one row on each side on the present grid, 128 cells and two rows on each side
on the fine one, five rows for 0.61 skin depth against 0.73); `bh_ranks.txt` and `bh_profile.txt`;
no checkpoints. The stock slab is ten cells thick on either grid, so the inner edge of the
replenished region lies 1.2 skin depths further out on the fine grid.
Fine grid in addition: the resolution; the intervals in steps doubled, so that the diagnostics keep
their cadence in time; the stride of the particle output times four, so that as many particles
are written.

Capacities per rank: 2 x (largest share of the box on one rank) x (largest number the box held),
per species, the rule of the MPI validation with the referee's numbers (Codex 1, 2026-11-22, G6):
shares 0.3288 for [2, 2] and 0.4611 for [4, 1]; largest number of a primary species in the box
8.8e7 on the present grid (the runs with the stock rule, which overfills; the corrected rule
should hold fewer, not yet measured at this size), times four on the fine grid. Leptons of the
pairs: 4e7 per rank on the present grid (the run at ten times the optical depth held at most
about 5e6 per species in the whole box), times four on the fine grid.

--toy writes the same edits on the toy box of stage A (toy-low.toml, 1024 x 512 cells, 200 steps,
capacities of the toy), to exercise every key together: toy-s1.toml.

Each edit must match exactly once: the script fails otherwise.
"""
import argparse
from pathlib import Path

SHARE = {"2,2": 0.3288, "4,1": 0.4611}
N_BOX = 8.8e7


def edit(text, pairs):
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"edit does not match once: {old!r} ({text.count(old)})")
        text = text.replace(old, new)
    return text


def capacity(share, factor):
    """Rounded up to two digits."""
    c = 2.0 * share * N_BOX * factor
    m, e = f"{c:.1e}".split("e")
    m = float(m)
    if m * 10 ** int(e) < c:
        m += 0.1
    return f"{m:.1f}e{int(e)}"


def run_input(base, name, dec, fine, seed, runtime, injector):
    f = 2 if fine else 1
    cap_p = capacity(SHARE[dec], f * f)
    cap_l = f"{4 * f * f}e7"
    setup = (f"  diag_interval  = {10 * f}\n  gauss_interval = {500 * f}\n  rank_interval  = {100 * f}\n"
             f"  injector       = {injector}\n  smooth_cells   = {64 * f}\n  smooth_rows    = {f}\n"
             f"  profile_interval = {500 * f}\n"
             "  profile_xmax   = 400.0\n" + (f"  seed           = {seed}\n" if seed else ""))
    pairs = [
        ('name    = "r-low"', f'name    = "{name}"'),
        ("runtime = 3000.0", f"runtime = {float(runtime)}"),
        ("decomposition = [1, 1]", f"decomposition = [{dec.replace(',', ', ')}]"),
        ('    label    = "e-"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 1.2e8',
         f'    label    = "e-"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = {cap_p}'),
        ('    label    = "ion"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 1.2e8',
         f'    label    = "ion"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = {cap_p}'),
        ('    label    = "e-_bh"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 4e7',
         f'    label    = "e-_bh"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = {cap_l}'),
        ('    label    = "e+_bh"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 4e7',
         f'    label    = "e+_bh"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = {cap_l}'),
        ("      optical_depth    = 0.014333", "      optical_depth    = 0.14333"),
        ("  diag_interval  = 10\n  gauss_interval = 500\n", setup),
        ("    interval_time    = 20.0\n", "    interval_time    = 20.0\n\n[checkpoint]\n  keep = 0\n"),
    ]
    if fine:
        pairs += [("resolution = [4096, 2048]", "resolution = [8192, 4096]"),
                  ("    stride        = 5\n", "    stride        = 20\n")]
    return edit(base, pairs)


def toy_input(toy, dec, injector):
    return edit(toy, [
        ('name    = "toy-low"', 'name    = "toy-s1"'),
        ("runtime = 500.0", "runtime = 17.3"),
        ("decomposition = [1, 1]", f"decomposition = [{dec.replace(',', ', ')}]"),
        ("  diag_interval  = 10\n  gauss_interval = 100\n",
         "  diag_interval  = 10\n  gauss_interval = 50\n  rank_interval  = 50\n" + f"  injector       = {injector}\n"
         "  smooth_cells   = 64\n  profile_interval = 100\n  profile_xmax   = 100.0\n  seed           = 1\n"),
        ("    interval_time    = 5.0\n", "    interval_time    = 5.0\n\n[checkpoint]\n  keep = 0\n"),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--decomposition", default="2,2", choices=sorted(SHARE))
    ap.add_argument("--runtime", type=int, default=3000, choices=(3000, 9000))
    ap.add_argument("--toy", action="store_true")
    ap.add_argument("--injector", type=int, default=2, choices=(1, 2))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.toy:
        (out / "toy-s1.toml").write_text(toy_input((Path(a.source) / "toy-low.toml").read_text(), a.decomposition, a.injector))
        return
    base = (Path(a.source) / "r-low.toml").read_text()
    for grid, fine in (("n", False), ("f", True)):
        if a.runtime == 9000:
            (out / f"s1-{grid}-long.toml").write_text(run_input(base, f"s1-{grid}-long", a.decomposition, fine, 0, 9000, a.injector))
        else:
            for tag, seed in (("a", 0), ("b", 1), ("c", 2)):
                (out / f"s1-{grid}-{tag}.toml").write_text(
                    run_input(base, f"s1-{grid}-{tag}", a.decomposition, fine, seed, 3000, a.injector))


if __name__ == "__main__":
    main()
