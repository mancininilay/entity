# Reconnection with the Bethe-Heitler emission policy: shakedown runs

Not part of the pull request (`dev/bh-emission`). The first reconnection runs
with the policy: an electron-ion plasma with ions of mass 1, no cooling, the
ions emitting pairs on a monochromatic photon field of 1.8 m_e c^2. Two runs,
identical except for the policy.

- `pgen.hpp`: the stock generator `pgens/reconnection` with diagnostics added
  (`diff ../reconnection/pgen.hpp pgen.hpp`: two changed lines, the rest is
  added). The diagnostics read the state after a step and write
  `bh_counters.txt` and `bh_gauss.txt` (see the note at the top of the file).
- `r-low.toml`: the stock `reconnection.toml` with one rank, runtime 3000,
  species 2 as emitting ions, two empty lepton species, the parameters of the
  policy, the two diagnostics and the output the checks need
  (`diff ../reconnection/reconnection.toml r-low.toml`).
- `derive_inputs.py`: writes `r-off.toml` (no emission), the toy pair
  `toy-low.toml` and `toy-off.toml` (a 250 x 125 d_e box) and `identity.toml`
  from `r-low.toml` by listed edits.
- `PREREGISTRATION.md`: the five checks of the two runs, with tolerances,
  fixed before the runs.
- `scripts/`: `check1_counters.py` to `check5_maps.py`, `run_checks.sh`
  (needs numpy; the figures of check 5 need matplotlib, `plot_check5.py`);
  `bhio.py` reads the output; `compare_identity.py` compares two runs bit for
  bit.
- `tools/bp2raw.cpp`: variables of a BP file as raw arrays
  (`c++ -std=c++17 -O2 bp2raw.cpp $(adios2-config --cxx-flags --cxx-libs) -o bp2raw`).
- `data/j_table_mono18.txt`: the interaction rate J(Gamma_p) for this photon
  field from the independent reference package `bhref` 0.3.0, written by
  `data/make_j_table.py`; the reference of check 2.

Build: `cmake -D pgen=reconnection_bh -D output=ON ...` (zig-zag deposit, one
domain for the Gauss-law diagnostic).
