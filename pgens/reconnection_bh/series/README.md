# Step 1 of the resolved series: the resolution pair (pre-registered)

The pair: the reconnection setup of the shakedown at ten times its optical depth (0.14333), with
the replenishing rule that follows the field (`setup.injector = 2`), the emitter listed before the
lepton species, on the four A100 of one node ([2, 2], one rank per card), on the present grid
(4096 x 2048 cells) and at twice the resolution (8192 x 4096 cells, the time step halves with the
cell): three realisations by `setup.seed` on each grid and one run of 9000 time units on the present
grid. Decided by Nilay on 2026-11-23; the pre-registration is `PREREGISTRATION-S1.md` (its commit
and SHA-256 are recorded in the master folder of the project), which fixes the jobs, the checks,
the stop conditions and the release of the fine grid.

| File | What |
|---|---|
| `PREREGISTRATION-S1.md` | the pre-registration of the pair |
| `INPUTS.sha256`, `SCRIPTS.sha256` | the registered inputs, and the analysis scripts and table; `run.sh` and `analyze.sh` refuse anything else |
| `derive_inputs_s1.py`, `derive_inputs_pair.py` | derive every input here from `../r-low.toml` (or `../toy-low.toml`) by listed edits |
| `s1-n-a.toml`, `s1-n-b.toml`, `s1-n-c.toml` | the present grid, three realisations (`setup.seed` absent, 1, 2), 3000 time units |
| `s1-n-long.toml` | the present grid, 9000 time units, seed 3 |
| `s1-f-a.toml`, `s1-f-b.toml`, `s1-f-c.toml` | twice the resolution, three realisations, 3000 time units |
| `lf-ctl.toml`, `lf-neg.toml` | 300 steps at production size with the lepton species listed first: the birth write and its negative control (test-only options) |
| `r-low-1card.toml` | R-low of stage B with the MPI build on one card (the decomposition question; reported) |
| `toy-s1.toml` | the same edits on the toy box of stage A, the smoke run of the build job |
| `check2_counts.py`, `check4_ref.py`, `ranks_check.py`, `upstream.py`, `gauss_level.py`, `lf_check.py`, `reader_gate.py` | the checks of the pair |
| `core_series.py`, `compare_pair.py` | the core series of every run, and the comparisons R0 to R3 |
| `rusty/` | the drivers on Rusty: `prepare.sh`, `build.sh`, `submit.sh`, `run.sh`, `analyze.sh`, `reader_gate.sh`, `dryrun.sh` |

Capacities per rank follow the rule and the numbers stated at the top of `derive_inputs_s1.py`.
