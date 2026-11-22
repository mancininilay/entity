# Stage C of the shakedown: R-mid and an ensemble (local branch `dev/bh-reconnection-c`)

Branch from 97d78619 (`dev/bh-reconnection`, which is being refereed and is not touched). No
engine file and no generator file of the runs changes: the runs use the executable of stage B.

- `derive_inputs_c.py`, `r-mid.toml`, `r-low-b.toml`, `r-low-c.toml`, `r-off-b.toml`,
  `r-off-c.toml`: the five inputs, derived from `../r-low.toml` by listed edits.
- `PREREGISTRATION-C.md`: checks 1, 2, 4 and 5 of R-mid as registered for stage B; check 3
  re-registered with the ensemble as its reference; check 6 (numerical health of the loaded
  plasma); fixed before the runs.
- `../scripts/check3_ensemble.py`, `../scripts/check6_series.py`, `../scripts/check6_health.py`:
  the new checks (they import `bhio.py` and `check3_compare.py`).
- `rusty/`: `run_c.sh` and `run_c.sbatch` (one run on one A100, with a stop flag in the driver),
  `analyze_c.sh` (the checks on a login node), `dryrun_c.sh` (the driver with stub engines).
- `SCRIPTS.sha256`, `INPUTS.sha256`: the hashes the driver and the analysis verify on Rusty.

`../../injector_slab/pgen.hpp` is a diagnostic generator for upstream issue U17 (the stock
replenishing injector on a uniform plasma, with and without a flow, and a proposed correction).
No run of the shakedown uses it.
