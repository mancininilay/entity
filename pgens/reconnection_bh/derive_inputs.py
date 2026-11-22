#!/usr/bin/env python3
"""The other inputs of the shakedown, derived from r-low.toml by listed edits.

    derive_inputs.py <directory with r-low.toml>

  r-off.toml       r-low without the line `emission = "bethe_heitler"`
  toy-low.toml     a 250 x 125 d_e box for a workstation (same cell size),
                   the optical depth raised to 0.5 and the tables ended at
                   Gamma_p = 3 (the probability per step at the automatic
                   upper edge would exceed its bound at this optical depth)
  toy-off.toml     toy-low without the emission line
  identity.toml    toy-off with every particle and every field written at
                   the end of 200 steps, for the comparison with the stock
                   generator
Each edit must match exactly once: the script fails otherwise.
"""
import sys
from pathlib import Path

EMISSION = '    emission = "bethe_heitler"\n'


def edit(text, pairs):
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"edit does not match once: {old!r} ({text.count(old)})")
        text = text.replace(old, new)
    return text


TOY = [
    ('runtime = 3000.0', 'runtime = 500.0'),
    ('resolution = [4096, 2048]', 'resolution = [1024, 512]'),
    ('extent     = [[-500.0, 500.0], [-250.0, 250.0]]',
     'extent     = [[-125.0, 125.0], [-62.5, 62.5]]'),
    ('ds = [[10.0], [20.0]]', 'ds = [[4.0], [8.0]]'),
    ('    label    = "e-"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 1.2e8',
     '    label    = "e-"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 6e6'),
    ('    label    = "ion"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 1.2e8',
     '    label    = "ion"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 6e6'),
    ('    label    = "e-_bh"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 4e7',
     '    label    = "e-_bh"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 6e6'),
    ('    label    = "e+_bh"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 4e7',
     '    label    = "e+_bh"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 6e6'),
    ('optical_depth    = 0.014333', 'optical_depth    = 0.5'),
    ('gamma_max        = 0.0', 'gamma_max        = 3.0'),
    ('inj_ypad       = 50.0', 'inj_ypad       = 12.0'),
    ('cs_width       = 10.0', 'cs_width       = 4.0'),
    ('gauss_interval = 500', 'gauss_interval = 100\n  gauss_xpad     = 8.0\n  gauss_ypad     = 16.0'),
    ('  interval_time = 100.0\n\n  [output.fields]', '  interval_time = 25.0\n\n  [output.fields]'),
    ('    stride        = 5\n    interval_time = 20.0\n\n  [output.spectra]',
     '    stride        = 10\n    interval_time = 5.0\n\n  [output.spectra]'),
    ('    num_spatial_bins = [2, 1]\n    interval_time    = 20.0',
     '    num_spatial_bins = [2, 1]\n    interval_time    = 5.0'),
]

IDENTITY = [
    ('runtime = 500.0', 'runtime = 17.3'),
    ('gauss_interval = 100', 'gauss_interval = 20'),
    ('  interval_time = 25.0\n\n  [output.fields]', '  interval      = 199\n\n  [output.fields]'),
    ('    species       = [2, 3, 4]\n    stride        = 10\n    interval_time = 5.0',
     '    stride        = 1'),
    ('    num_spatial_bins = [2, 1]\n    interval_time    = 5.0', '    num_spatial_bins = [2, 1]'),
]


def main():
    d = Path(sys.argv[1])
    low = (d / "r-low.toml").read_text()
    off = edit(low, [(EMISSION, ""), ('name    = "r-low"', 'name    = "r-off"')])
    toy_low = edit(low, TOY + [('name    = "r-low"', 'name    = "toy-low"')])
    toy_off = edit(toy_low, [(EMISSION, ""), ('name    = "toy-low"', 'name    = "toy-off"')])
    identity = edit(toy_off, IDENTITY + [('name    = "toy-off"', 'name    = "identity"')])
    (d / "r-off.toml").write_text(off)
    (d / "toy-low.toml").write_text(toy_low)
    (d / "toy-off.toml").write_text(toy_off)
    (d / "identity.toml").write_text(identity)


if __name__ == "__main__":
    main()
