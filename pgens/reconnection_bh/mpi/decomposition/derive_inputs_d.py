#!/usr/bin/env python3
"""The four inputs of the decomposition question, derived from ../r-low-mpi.toml by listed edits.

    derive_inputs_d.py <directory with r-low-mpi.toml> <output directory>

  r-low-mpi-b22.toml  decomposition [2, 2], overdensity of the initial layer 2.997
  r-low-mpi-c22.toml  decomposition [2, 2], overdensity of the initial layer 3.003
  r-low-mpi-a41.toml  decomposition [4, 1], overdensity 3.000 (unchanged), capacity of the primaries 1e8 per rank
  r-low-mpi-b41.toml  decomposition [4, 1], overdensity 2.997, capacity of the primaries 1e8 per rank

Nothing else changes against the input of the run of M4 (r-low-mpi.toml, job 7523907): the
optical depth stays 0.014333, no checkpoints, the same outputs. Each edit must match exactly
once: the script fails otherwise.
"""
import sys
from pathlib import Path

LAYER = "  cs_overdensity = 3.0\n"
TILING = "    decomposition = [2, 2]\n"
ELECTRONS = '    label    = "e-"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 8e7\n'
IONS = '    label    = "ion"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 8e7\n'
SLAB = [(TILING, "    decomposition = [4, 1]\n"),
        (ELECTRONS, ELECTRONS.replace("8e7", "1e8")),
        (IONS, IONS.replace("8e7", "1e8"))]


def edit(text, pairs):
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"edit does not match once: {old!r} ({text.count(old)})")
        text = text.replace(old, new)
    return text


def name(new):
    return [('  name    = "r-low-mpi"\n', f'  name    = "{new}"\n')]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    base = (src / "r-low-mpi.toml").read_text()
    inputs = {
        "r-low-mpi-b22": edit(base, name("r-low-mpi-b22") + [(LAYER, "  cs_overdensity = 2.997\n")]),
        "r-low-mpi-c22": edit(base, name("r-low-mpi-c22") + [(LAYER, "  cs_overdensity = 3.003\n")]),
        "r-low-mpi-a41": edit(base, name("r-low-mpi-a41") + SLAB),
        "r-low-mpi-b41": edit(base, name("r-low-mpi-b41") + SLAB + [(LAYER, "  cs_overdensity = 2.997\n")]),
    }
    out.mkdir(parents=True, exist_ok=True)
    for k, text in inputs.items():
        (out / f"{k}.toml").write_text(text)


if __name__ == "__main__":
    main()
