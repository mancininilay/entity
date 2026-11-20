# Results of the validation runs of 2026-11-19

Compact text outputs only; the run directories (inputs, logs, dumps) stay in
the evidence folder of the project.

Commits: `dev/bh-emission` 66aa6fb6 on upstream `1.6.0rc` 861b19a7; executables
built from `dev/bh-emission-validation` d43543ad (same sources under `src/`
plus the clang fix). Apple clang 21, Kokkos 5.0.1 (OpenMP, Serial), MPICH 4.3.2.

- `analysis_final.txt`: verdicts of V2 to V9, V12, V13 (`scripts/analyze_final.sh`).
  Earlier outputs of the same script are kept beside it: before the two
  reruns (V5 single and the largest step of V7 had not started), and before
  two corrections of the analysis scripts (a point of V3 that the run does not
  reach was counted as a failure; the records of V5 in single precision were
  matched by a rounded time).
- `v1_analysis.txt`: Gauss's law, with the controls.
- `v10_verdict.txt`: identity with upstream on Serial builds.
- `v11/`: the stock suite on the branch and on upstream with 4 and 2 threads.
- `tierA/`: `ARCHETYPES::bethe_heitler` built with bounds-checked views.
- `oracle/`: sources of the three dumps (host tables, sampler draws, device
  rates) and the hashes of their outputs from the fork tree at 72427ff6 and
  from the branch (identical).
- `e10_single.log`, `e10_double.log`: the event stage of `dev/hadronic`
  72427ff6 on the Born tables (the other side of V8).
- `upstream-issues/`: outputs of the reproductions and of upstream's own
  failing tests.
- `run_final.log`, `rerun_gaps.sh`, `rerun_gaps.log`: what was run.
