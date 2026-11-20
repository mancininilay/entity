# Second correction round of 2026-11-20

`dev/bh-emission` at 43d68f45 (four commits after e92d84d8): the reused-slot
case of the charge test with previous positions in neighbouring cells, the
finite checks of the timestep and of the mass of the emitting species, the
thread count of the table build passed as an argument, and the control
wrapper of the test using its two host functions.

- `tierA/`: `ARCHETYPES::bethe_heitler` built from the pull-request branch
  (Apple clang 21, Kokkos 5.0.1 OpenMP) in single and double precision,
  plain and with Kokkos bounds checking, at 1, 2 and 8 threads: twelve runs,
  all pass; the plain and the bounds-checked logs are identical.
  `timing.txt`: wall times on an idle machine. `build_pr_tests.sh` and
  `run_tierA.sh` (one level up) are what was run.
- `mutations/`: the policy with part of the previous-position write removed
  (seven mutations: nothing written, cell indices only, offsets only,
  electrons only, positrons only, x only, y only), as include overlays, with
  the shipped test and with a diagnostic copy that goes on after a failed
  check; single and double precision; 1, 2 and 8 threads. Every mutation
  fails the shipped test in every run. The mutation that the test of
  e92d84d8 let through (offsets written, cell indices not) is caught by the
  neighbouring-cell arm alone, with a continuity residual 5250 to 6330 times
  the limit. Two more mutants remove the finite check of the timestep and
  that of the mass: both fail. `matrix_summary.md` has the tables,
  `matrix_runs.txt` every log; `make_mutants.py`, `build_mutant.py`,
  `run_matrix.sh` and `summarize_matrix.py` produce them.
- `oracle/`: tables, sampler draws and device rates dumped with the headers
  of 43d68f45, with the default thread count (the dump programs of
  `results/oracle/` unchanged) and with every threaded builder given 1 and 3
  threads (`*_capped.cpp`): nine outputs, byte-identical to those of
  66aa6fb6 (`oracle_sha256.txt`, `run_oracle.txt`).
- `a100-e92d84d8/`: tier A of the previous head, e92d84d8, on one A100
  (CUDA 12.8, Kokkos 5.2.1), single and double precision: it passes. This
  is the test before the neighbouring-cell arm was added, kept as an early
  reading of the tolerance on a GPU: in single precision the continuity
  residual of the reused-slot case is 2.3e-5 against a limit of 1.35e-2.
  `build_job_7512470_driver.txt` is the log of the build job, whose driver
  script stopped after the two builds (a defect of the script).

The standalone reproduction for the contract of `kernel::InjectParticle` is
in `pgens/bh_validation/upstream-repro/`.

Absolute paths of the machines are written as `<Entity Local Experiments>`,
`<home>` and `<ceph>` in these copies.
