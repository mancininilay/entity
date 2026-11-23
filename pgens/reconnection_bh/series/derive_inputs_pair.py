#!/usr/bin/env python3
"""The inputs of the resolution pair that `derive_inputs_s1.py` does not write, derived by listed
edits. Claude 1, 2026-11-23 (brief `2026-11-23-claude1-local-brief-bh-resolution-pair.md`).

    derive_inputs_pair.py <directory of the generator copy> <output directory>

  s1-n-long.toml   s1-n-a with 9000 time units and `setup.seed = 3`: the long run on the
                   present grid, a realisation of its own.
  lf-ctl.toml      the birth write inside reconnection, the run with the write: s1-n-a for 300
                   steps (runtime 25.9) with the two lepton species of the policy listed first
                   (species 1 and 2; the primaries are species 3 and 4,
                   `setup.primary_species = [3, 4]`), the previous positions of their unused
                   slots made unequal (`setup.stale_prev = true`), the Gauss-law diagnostic every
                   50 steps, no output after the first step.
  lf-neg.toml      the same with the write of the previous position of a new pair left out: the
                   emitter carries `emission = "custom"`, `setup.negative_control = true`, and an
                   empty species with the built-in policy is appended so that the parameters of
                   the policy are read (the generator's test-only control; never a production
                   input).
  r-low-1card.toml the input of the runs on four cards of the decomposition question
                   (`../mpi/r-low-mpi.toml`) on one rank, with the capacity of one card for the
                   primaries (1.6e8, as the members of stage C), the Gauss-law diagnostic every
                   500 steps (one domain), and a name of its own. It runs with the executable of
                   those runs (second local session, build at b996538f): MPI build, one card.

Each edit must match exactly once: the script fails otherwise.
"""
import sys
from pathlib import Path


def edit(text, pairs):
    for old, new in pairs:
        if text.count(old) != 1:
            raise SystemExit(f"edit does not match once: {old!r} ({text.count(old)})")
        text = text.replace(old, new)
    return text


EMITTER_FIRST = '''  [[particles.species]]
    label    = "e-"
    mass     = 1.0
    charge   = -1.0
    maxnpart = 5.8e7

  [[particles.species]]
    label    = "ion"
    mass     = 1.0
    charge   = 1.0
    maxnpart = 5.8e7
    emission = "bethe_heitler"

  [[particles.species]]
    label    = "e-_bh"
    mass     = 1.0
    charge   = -1.0
    maxnpart = 4e7

  [[particles.species]]
    label    = "e+_bh"
    mass     = 1.0
    charge   = 1.0
    maxnpart = 4e7
'''


def leptons_first(emission):
    unused = '''
  [[particles.species]]
    label    = "unused"
    mass     = 1.0
    charge   = 0.0
    maxnpart = 16.0
    emission = "bethe_heitler"
''' if emission == "custom" else ""
    return f'''  [[particles.species]]
    label    = "e-_bh"
    mass     = 1.0
    charge   = -1.0
    maxnpart = 4e7

  [[particles.species]]
    label    = "e+_bh"
    mass     = 1.0
    charge   = 1.0
    maxnpart = 4e7

  [[particles.species]]
    label    = "e-"
    mass     = 1.0
    charge   = -1.0
    maxnpart = 5.8e7

  [[particles.species]]
    label    = "ion"
    mass     = 1.0
    charge   = 1.0
    maxnpart = 5.8e7
    emission = "{emission}"
''' + unused


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    base = (src / "series" / "s1-n-a.toml").read_text()
    if base.count(EMITTER_FIRST) != 1:
        raise SystemExit("the species block of s1-n-a.toml is not the expected one")

    (out / "s1-n-long.toml").write_text(edit(base, [
        ('name    = "s1-n-a"', 'name    = "s1-n-long"'),
        ("runtime = 3000.0", "runtime = 9000.0"),
        ("  profile_xmax   = 400.0\n", "  profile_xmax   = 400.0\n  seed           = 3\n"),
    ]))

    short = [
        ("runtime = 3000.0", "runtime = 25.9"),
        ("  gauss_interval = 500\n", "  gauss_interval = 50\n"),
        ("      electron_species = 3\n      positron_species = 4\n",
         "      electron_species = 1\n      positron_species = 2\n"),
        ("  interval_time = 100.0\n", "  interval_time = 1000.0\n"),
        ("    species       = [2, 3, 4]\n    stride        = 5\n    interval_time = 20.0\n",
         "    species       = [1, 2, 4]\n    stride        = 5\n    interval_time = 1000.0\n"),
        ("    num_spatial_bins = [2, 1]\n    interval_time    = 20.0\n",
         "    num_spatial_bins = [2, 1]\n    interval_time    = 1000.0\n"),
    ]
    for name, emission, extra in (("lf-ctl", "bethe_heitler", "  stale_prev     = true\n"),
                                  ("lf-neg", "custom", "  stale_prev     = true\n  negative_control = true\n")):
        (out / f"{name}.toml").write_text(edit(base, short + [
            ('name    = "s1-n-a"', f'name    = "{name}"'),
            (EMITTER_FIRST, leptons_first(emission)),
            ("  profile_xmax   = 400.0\n", "  profile_xmax   = 400.0\n  primary_species = [3, 4]\n" + extra),
        ]))

    mpi = (src / "mpi" / "r-low-mpi.toml").read_text()
    (out / "r-low-1card.toml").write_text(edit(mpi, [
        ('name    = "r-low-mpi"', 'name    = "r-low-1card"'),
        ("decomposition = [2, 2]", "decomposition = [1, 1]"),
        ('    label    = "e-"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 8e7',
         '    label    = "e-"\n    mass     = 1.0\n    charge   = -1.0\n    maxnpart = 1.6e8'),
        ('    label    = "ion"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 8e7',
         '    label    = "ion"\n    mass     = 1.0\n    charge   = 1.0\n    maxnpart = 1.6e8'),
        ("  gauss_interval = 0\n", "  gauss_interval = 500\n"),
    ]))


if __name__ == "__main__":
    main()
