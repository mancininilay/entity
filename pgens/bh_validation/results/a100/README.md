# Results on one A100 (2026-11-19)

One NVIDIA A100-SXM4-80GB, one rank, no MPI. GCC 13.3, CUDA 12.8 (nvcc
12.8.61), Kokkos 5.2.1 (Serial + CUDA, AMPERE80), `output=OFF`, single and
double precision. The unit test was built from `dev/bh-emission` 66aa6fb6, the
executable with `pgens/bh_validation` from `dev/bh-emission-validation`
a8a6f399. Drivers: `../../rusty/build-cuda.sh` (compile only, 19 min 33 s on 16
cores) and `../../rusty/run-a100.sh` (12 min 46 s); each ran as one Slurm job
of one hour at most, one GPU, no requeue. The new code compiles without
warnings; the only warnings are two unused variables in `src/output/render/`.

- `tierA_single.txt`, `tierA_double.txt`: `ARCHETYPES::bethe_heitler` (A1, A2, A3).
- `analysis.txt`: V1 on one rank (both species orders, the controls, the
  negative control, the same with stale slots), V2 and V3 at Gamma_p0 = 6, the
  energy ledger of that run, V4 at Gamma_p = 3 and 10 (ions first).
- `compare_a100_mac_v2.txt`: the cooling runs of the A100 against those of the
  Mac (`scripts/compare_runs.py`).

V4 passes the registered family rule with a smallest adjusted p of 0.0175: one
comparison (the electron energy at Gamma_p = 3) has p = 0.00175, and the mean
electron energy of the sample is 4.4 standard errors low. The files below
trace this to the random numbers and not to the GPU:

- `v4_blocks_a100.txt`, `v4_blocks_mac.txt`: the samples step by step
  (`scripts/v4_blocks.py`).
- `replay_g3_221184_seed_default.txt`: the same statistics from a replay on a
  CPU of the first uses of the 221184 generator states that the pool has on an
  A100, seeded as Entity seeds it (`tests/archetypes/bhv_replay_pool.cpp`).
  The windows of 19 uses are the steps of the run and reproduce them.
- `match_block0.txt`: every pair created on the A100 in the first step
  (209510) against the replay (`scripts/extract_block0.py`,
  `scripts/match_block0.py`): all found, largest relative difference 3e-14.
- `replay_g3_221184_seed_default_950trials.txt`: the same streams ten times
  deeper; `replay_g3_221184_other_seeds.txt`: sixteen other seeds.

A by-product of the replay, not related to the GPU:

- `replay_mean_loss_1e8_seed1.txt`, `replay_mean_loss_1e8_other_seeds.txt`,
  `replay_mean_loss_high_gamma.txt`: the mean loss per interaction of the
  sampler against the tabulated one, about 1e8 outcomes per Lorentz factor.
  They agree at Gamma_p = 3 and differ by a few 1e-4 above (1.4e-3 at 80).
- `bhref_vs_table_mean_loss.txt`: the frozen reference against the same table.

All four V4 runs of one machine are one sample: the pool is seeded with a
constant and the runs at Gamma_p = 3 and 10 consume the same numbers.
