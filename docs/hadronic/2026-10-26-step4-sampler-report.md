# P2 step 4, C++ side: the rest-frame sampler (`restframe` spectrum for both sources). Report

Claude 1 (Claude Code cloud session), 2026-10-26. Fork `mancininilay/entity`,
branch `dev/hadronic-sampler` (created from `dev/hadronic` at `5017cac`).
Written against the step-4 cloud brief (contract revision 1.0 for the
rest-frame sampler, spec v3.3 III.2, III.4, III.6).

This session's budget is limited and may end without warning; the work is
staged, each stage ends with a commit and a push and an entry below. A
session that resumes reads this report first and continues from the first
incomplete stage. Statements are labelled fact (with the number), convention,
or open question. No pre-registered tolerance is adjusted.

## 0. Stage list (brief section 2)

| stage | content | status |
| --- | --- | --- |
| 0 | G3 fallout: the two float literals of `g3_asymmetric_capacity_2ranks.toml`; the T5 dt-convergence split into its own ctest entry with a timeout | in progress |
| 1 | the rest-frame laws g, h, the recoil moment q_k by quadrature, anchors A1 to A5, the fixture `qk_table.txt` | not started |
| 2 | CDF tables, the two sampling chains on the host, A7, A9, the `moments_*` and `hist_*` fixtures | not started |
| 3 | the UR anchor (Kelner and Aharonian 2008 eqs. 58 to 64), A8, the `ka2008_*` fixtures | not started |
| 4 | device tables, the device sampler, `spectrum = restframe` in the kernel, A11, A12 | not started |
| 5 | the reduction timing harness (G3 open item) | not started |
| 6 | closure: full ctest single and double, the report completed | not started |

## 1. Environment (facts, measured in the cloud container)

| item | value |
| --- | --- |
| `dev/hadronic` HEAD at branch creation | `5017cac` (round-3 suite counts) |
| gcc | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| cmake / ninja | 4.4.4 / 1.13.2, 4 cores |
| Kokkos, ADIOS2 | `$HOME/.entity/kokkos`, `$HOME/.entity/adios2` (Serial + OpenMP, no MPI) |
| configure | `cmake -S . -B build -G Ninja -D TESTS=ON -D output=ON -D precision=single -D mpi=OFF -D CMAKE_PREFIX_PATH=...`: OK |
| literature PDFs | **not present in the container**: there is no `Literature/` directory (`Blumenthal_1970.pdf`, `Motz_1969.pdf`, `Kelner_Aharonian_2008.pdf`) and no `Resources/` note. The formulas of stages 1 and 3 are transcribed from the author's knowledge of the papers with the formula numbers stated in the code; the report flags every such transcription as "transcribed without the PDF" (open question for the referee: verify against the PDFs). |

## 2. Stage log

### Stage 0 (G3 fallout)

- stage 0 started, HEAD `5017cac`.
- (a) `pgens/bh_proto/inputs/mpi/g3_asymmetric_capacity_2ranks.toml` lines 56
  and 66: `maxnpart = 40` -> `40.0` (the pair species; the ion species at line
  46 already reads `1e3`). The `Resources` note naming the patch is not in the
  container; the change is the one described in the brief.
- (b) `tests/kernels/bh_loss.cpp`: `test_cooling_T5` split into the base
  cooling curve (`t5`, unchanged checks: two K(t) errors, the T11-lite
  closure, two envelope checks) and `test_dt_convergence_T5fine` (`t5fine`:
  the three-resolution dt-convergence with its three verdicts and the
  envelope checks of the scaled runs). The common setup (dt from dK/K = 1e-4
  per step at Gamma_p = 6, the step count 6 -> 3 found by the probe run) is
  factored into `t5_setup` and is recomputed by each group, so the groups are
  independent. `tests/kernels/CMakeLists.txt`: `t5fine` added to the group
  list; `KERNELS::bh_loss_t5fine` and `KERNELS::bh_loss_t5b` get `TIMEOUT
  7200` (convention: the long groups only; every other entry keeps the ctest
  default). The executable without arguments still runs every group. Checked
  here by `-fsyntax-only` with the project's compile flags (the kernel test
  executable is not built before stage 4; the full ctest is stage 6).
