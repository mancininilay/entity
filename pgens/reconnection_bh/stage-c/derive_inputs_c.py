#!/usr/bin/env python3
"""The inputs of stage C, derived from ../r-low.toml by listed edits.

    derive_inputs_c.py <directory with r-low.toml> <output directory>

  r-mid.toml     R-low at ten times the optical depth (0.14333), with the ion output
  r-low-b.toml   R-low with the overdensity of the current layer changed in its fourth digit
                 (2.997) and the optical depth changed in its fourth digit (0.014323)
  r-low-c.toml   the same, the other way (3.003 and 0.014343)
  r-off-b.toml   R-off with the overdensity of the current layer at 2.997
  r-off-c.toml   R-off with the overdensity of the current layer at 3.003

All five: no checkpoints (`checkpoint.keep = 0`) and capacities of the primaries of 1.6e8.
The four members of the ensemble write no ions (particle output of the lepton species only).
`setup.t_open` is NOT written: the generator reads it as a double (`simtime_t`) while the engine
stores the floats of `[setup]` as `real_t`, so in a single-precision build any value given in
the input stops the run at start ("Bad any_cast for setup.t_open"). The default stays in force:
1.5 x half the extent in x = 750. Each edit must match exactly once: the script fails otherwise.
"""
import sys
from pathlib import Path

EMISSION = '    emission = "bethe_heitler"\n'

COMMON = [
    ('    label    = "e-"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 1.2e8',
     '    label    = "e-"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 1.6e8'),
    ('    label    = "ion"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 1.2e8',
     '    label    = "ion"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 1.6e8'),
    ('    interval_time    = 20.0\n', '    interval_time    = 20.0\n\n[checkpoint]\n  keep = 0\n'),
]
NO_IONS = [('    species       = [2, 3, 4]', '    species       = [3, 4]')]


def edit(text, pairs):
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"edit does not match once: {old!r} ({text.count(old)})")
        text = text.replace(old, new)
    return text


def name(new):
    return [('name    = "r-low"', f'name    = "{new}"')]


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    low = (src / "r-low.toml").read_text()
    if not low.endswith("    interval_time    = 20.0\n"):
        raise SystemExit("r-low.toml does not end with the spectra block")
    base = edit(low, COMMON)
    tau, layer = 'optical_depth    = 0.014333', 'cs_overdensity = 3.0'
    layer_b, layer_c = [(layer, 'cs_overdensity = 2.997')], [(layer, 'cs_overdensity = 3.003')]
    inputs = {
        "r-mid": edit(base, name("r-mid") + [(tau, 'optical_depth    = 0.14333')]),
        "r-low-b": edit(base, name("r-low-b") + NO_IONS + layer_b + [(tau, 'optical_depth    = 0.014323')]),
        "r-low-c": edit(base, name("r-low-c") + NO_IONS + layer_c + [(tau, 'optical_depth    = 0.014343')]),
        "r-off-b": edit(base, name("r-off-b") + NO_IONS + layer_b + [(EMISSION, "")]),
        "r-off-c": edit(base, name("r-off-c") + NO_IONS + layer_c + [(EMISSION, "")]),
    }
    out.mkdir(parents=True, exist_ok=True)
    for k, text in inputs.items():
        (out / f"{k}.toml").write_text(text)


if __name__ == "__main__":
    main()
