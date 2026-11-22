#!/usr/bin/env python3
"""Inputs of the Bethe-Heitler validation runs (pgens/bh_validation).

    make_input.py <out.toml> key=value ...

Keys (defaults in DEFAULTS): order = "leptons_first" | "ions_first"; policy =
"on" | "off" | "negative"; fields = true | false; hot_box and cold_box =
"x0, x1, y0, y1" (fractions of the domain); the rest are numbers.
The optical depth is given through `p_ref`, the probability per step at the
Lorentz factor whose tabulated rate is `J_ref`, or directly as `tau`.
"""
import math
import sys

DEFAULTS = dict(
    name="bhv", order="leptons_first", policy="on", fields="true",
    nx=64, ny=64, dx=0.1, cfl=0.5, steps=100, decomposition="",
    n_ions=32768, gamma_p=10.0, gamma_max=12.0, mass_ratio=1836.15,
    ion_mass=1.0, isotropic="true", neutralizer="true",
    eps0=0.5, J_ref=1.09447, p_ref=0.05, pair_weight=1.0,
    max_probability=0.1, max_loss_fraction=0.04, use_weights="false",
    maxnpart_ions=0, maxnpart_leptons=2.0e6, diag_interval=5, dump_step=-1,
    gauss="true", seed=1, bx=0.0, by=0.0, bz=0.0, sort_interval=-1,
    filters=0, stale_prev="false", recycle="false", tau=0.0,
    rank_lines="false", hot_fraction=0.0, hot_box="0.0, 1.0, 0.0, 1.0",
    cold_box="0.0, 1.0, 0.0, 1.0",
)


def flt(x):
    """A TOML float: never an integer literal."""
    t = f"{x:.12g}"
    return t if ("." in t or "e" in t) else t + ".0"


def main():
    out = sys.argv[1]
    cfg = dict(DEFAULTS)
    for kv in sys.argv[2:]:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise SystemExit(f"unknown key {k}")
        cfg[k] = type(DEFAULTS[k])(v) if not isinstance(DEFAULTS[k], str) else v
    nx, ny, dx = int(cfg["nx"]), int(cfg["ny"]), float(cfg["dx"])
    # Entity: dt = CFL * dx / sqrt(D)
    dt = float(cfg["cfl"]) * dx / math.sqrt(2.0)
    tau = float(cfg["p_ref"]) * float(cfg["pair_weight"]) / (float(cfg["J_ref"]) * dt)
    if float(cfg["tau"]) > 0.0:
        # the optical depth given directly
        tau = float(cfg["tau"])
    runtime = (int(cfg["steps"]) + 0.5) * dt
    fields = cfg["fields"] == "true"
    leptons_first = cfg["order"] == "leptons_first"
    emission = {"on": "bethe_heitler", "off": "none", "negative": "custom"}[cfg["policy"]]
    n_ions = int(cfg["n_ions"])
    max_ions = int(cfg["maxnpart_ions"]) or 2 * n_ions
    sort = "" if int(cfg["sort_interval"]) < 0 else f"\n    spatial_sorting_interval = {int(cfg['sort_interval'])}"

    def lepton(label, charge):
        return f"""  [[particles.species]]
    label    = "{label}"
    mass     = 1.0
    charge   = {charge}
    maxnpart = {float(cfg["maxnpart_leptons"]):.1f}{sort}
"""

    ion = f"""  [[particles.species]]
    label    = "ion"
    mass     = {float(cfg['ion_mass'])}
    charge   = 1.0
    maxnpart = {float(max_ions):.1f}
    emission = "{emission}"{sort}
"""
    if leptons_first:
        species = lepton("e-", -1.0) + lepton("e+", 1.0) + ion
        e_idx, p_idx, i_idx = 1, 2, 3
    else:
        species = ion + lepton("e-", -1.0) + lepton("e+", 1.0)
        i_idx, e_idx, p_idx = 1, 2, 3
    if cfg["policy"] == "negative":
        # an empty species with the built-in policy, so that its parameters are read
        species += """  [[particles.species]]
    label    = "unused"
    mass     = 1.0
    charge   = 0.0
    maxnpart = 16.0
    emission = "bethe_heitler"
"""
    decomposition = ""
    if cfg["decomposition"]:
        decomposition = f"\n  [simulation.domain]\n    decomposition = [{cfg['decomposition']}]\n"
    radiation = ""
    if cfg["policy"] != "off":
        radiation = f"""
[radiation]
  [radiation.emission]
    [radiation.emission.bethe_heitler]
      electron_species  = {e_idx}
      positron_species  = {p_idx}
      optical_depth     = {tau:.9g}
      mass_ratio        = {float(cfg['mass_ratio'])}
      pair_weight       = {float(cfg['pair_weight'])}
      max_probability   = {float(cfg['max_probability'])}
      max_loss_fraction = {float(cfg['max_loss_fraction'])}
      gamma_max         = {float(cfg['gamma_max'])}
      photon_spectrum   = "mono"
      photon_energy     = {float(cfg['eps0'])}
"""
    text = f"""[simulation]
  name    = "{cfg['name']}"
  engine  = "srpic"
  runtime = {runtime:.12g}
{decomposition}
[grid]
  resolution = [{nx}, {ny}]
  extent     = [[0.0, {flt(nx * dx)}], [0.0, {flt(ny * dx)}]]

  [grid.metric]
    metric = "minkowski"

  [grid.boundaries]
    fields    = [["PERIODIC"], ["PERIODIC"]]
    particles = [["PERIODIC"], ["PERIODIC"]]

[scales]
  larmor0    = 1.0
  skindepth0 = 1.0

[algorithms]
  current_filters = {int(cfg['filters'])}

  [algorithms.timestep]
    CFL = {float(cfg['cfl'])}

  [algorithms.fieldsolver]
    enable = {'true' if fields else 'false'}

  [algorithms.deposit]
    enable = {'true' if fields else 'false'}

[particles]
  ppc0        = 8.0
  use_weights = {cfg['use_weights']}

{species}{radiation}
[setup]
  ion_species         = {i_idx}
  neutralizer_species = {e_idx if cfg['neutralizer'] == 'true' else 0}
  electron_species    = {e_idx}
  positron_species    = {p_idx}
  n_ions              = {n_ions}
  seed                = {int(cfg['seed'])}
  gamma_p             = {float(cfg['gamma_p'])}
  mass_ratio          = {float(cfg['mass_ratio'])}
  isotropic           = {cfg['isotropic']}
  gauss               = {cfg['gauss']}
  negative_control    = {'true' if cfg['policy'] == 'negative' else 'false'}
  stale_prev          = {cfg['stale_prev']}
  recycle             = {cfg['recycle']}
  rank_lines          = {cfg['rank_lines']}
  hot_fraction        = {float(cfg['hot_fraction'])}
  hot_box             = [{cfg['hot_box']}]
  cold_box            = [{cfg['cold_box']}]
  diag_interval       = {int(cfg['diag_interval'])}
  dump_step           = {int(cfg['dump_step'])}
  bx                  = {float(cfg['bx'])}
  by                  = {float(cfg['by'])}
  bz                  = {float(cfg['bz'])}

[output]
  interval = 100000000

  [output.fields]
    enable = false

  [output.particles]
    enable = false

  [output.spectra]
    enable = false

  [output.stats]
    enable = false

[checkpoint]
  interval = 100000000

[diagnostics]
  interval = 100000000
"""
    with open(out, "w") as f:
        f.write(text)
    print(f"{out}: dt = {dt:.9g}, optical_depth = {tau:.9g}, steps = {cfg['steps']}")


if __name__ == "__main__":
    main()
