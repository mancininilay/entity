# Corrections of 2026-11-20

`dev/bh-emission` at e92d84d8 (four commits after 66aa6fb6): the charge test
with reused lepton slots and its control, exactly opposite lepton charges,
refusal of non-finite parameters with the reworded description of the
parameters, and the bound on the threads of the table build.

- `tierA_single.txt`, `tierA_double.txt`, `tierA_bounds_single.txt`,
  `tierA_bounds_double.txt`: `ARCHETYPES::bethe_heitler` built from the
  pull-request branch (Apple clang 21, Kokkos 5.0.1 OpenMP, 8 threads), the
  last two with Kokkos bounds checking.
- `mutation_old_comparator_double.txt`: the same test with the approximate
  comparison of the lepton charges put back: it fails, as it must.
  `mutation_no_finite_check_double.txt`: the same with the finite check
  removed.
- `parser_input_nan_*.txt`: the executable with `optical_depth = nan` and
  with `max_probability = nan` in the input: refused when the input is read.
- `oracle_sha256.txt`: tables, sampler draws and device rates dumped with
  the corrected headers: the hashes are those of 66aa6fb6 and of the fork at
  72427ff6, also with the build limited to 1 and to 3 threads.
- `rusty_thread_bound.txt`: GCC 13.3 on Linux: the bound follows the
  affinity mask (32, 4 and 1 threads without a mask, with four processors
  and with one), and the tables of the old and the new headers are identical.
- `regression_v4_g3.txt`: one run of 2026-11-19 repeated with the corrected
  build: the same 1048903 pairs, bit for bit as sets (the order of the slots
  depends on the threads).
- `loss_bound_seed2.txt`: see the commit that added it.
