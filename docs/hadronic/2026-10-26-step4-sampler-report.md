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
| 0 | G3 fallout: the two float literals of `g3_asymmetric_capacity_2ranks.toml`; the T5 dt-convergence split into its own ctest entry with a timeout | done, `6348ba4` |
| 1 | the rest-frame laws g, h, the recoil moment q_k by quadrature, anchors A1 to A5, the fixture `qk_table.txt` | done, `2893cfb` |
| 2 | CDF tables, the two sampling chains on the host, A7, A9, the `moments_*` and `hist_*` fixtures | in progress |
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
- stage 0 done, commit `6348ba4`, pushed; draft PR
  https://github.com/mancininilay/entity/pull/4 opened against `dev/hadronic`.

### Stage 1 (the rest-frame laws)

- stage 1 started, HEAD `6348ba4`.

#### What was implemented (stage 1)

`src/archetypes/hadronic/bh_restframe.h` (namespace `arch::bh::rf`, host-only,
Kokkos-free):

| content | function | source stated in the code |
| --- | --- | --- |
| energy law dsigma/dE'+ (law 1), long double evaluation | `dsigma_dE`, `g` | Blumenthal 1970 eq. (10) = Heitler 1954 section 26 eq. (14) = Motz, Olsen and Koch 1969 formula 3D-1000, Z = 1, unscreened; **transcribed without the PDF** |
| fully differential law (law 2), integrated numerically over the second lepton's direction on the momentum-transfer sphere (u = ln q^2, azimuth psi, graded Gauss-Legendre panels around the second lepton's forward peak) | `dsigma_dE_dcos_numeric`, `AngularQuadrature` | Bethe and Heitler 1934 as Heitler 1954 section 26 eq. (6) = MOK 3D-3000; **transcribed without the PDF**; its normalisation is fixed to 1/(4 pi^2) (p+ p-/k^3) F/q^4 by the requirement that its angular integral be the energy law (a 1/(2 pi) prefactor gave exactly 2 pi times the energy law: fact of the scratch run) |
| single-lepton energy-angle law (law 3), closed form, long double | `dsigma_dE_dcos_closed`, `h` | Sauter-Gluckstern-Hull, MOK 1969 formula 3D-2000, Z = 1, unscreened (the formula number is from memory, **to be verified against `Literature/Motz_1969.pdf`**); reconstructed from Koch and Motz 1959 formula 2BN by the substitution E0 -> -E+, E -> E-, k -> -k; the signs of the three logarithmic groups and the overall factor, which the substitution leaves ambiguous, were fixed against the numerical integral of law 2: s0 = -1, sL = -1, se = +1, sQ = +1 is the unique combination of 16 that agrees (constant ratio to 6 digits at 12 points; the other 15 vary by factors of order one), and the normalisation is d^2 sigma/(dE+ dOmega+) = -(p+ p-/(8 pi k^3)) bracket (the 2BN prefactor) |
| recoil moment q_k = eps' - 2 <p' cos theta'>_h, the moments | `moments`, `RestFrameQuadrature` | contract section 1 item 3; composite Gauss-Legendre: E' = 1 + (eps' - 2) sin^2(pi s/2), 16 panels x 16 nodes; angle t = ln(1 - cos theta + (E - p)/p), panels of width 0.5 x 16 nodes |
| the eps' grid of the fixture (449 nodes) | `qk_grid` | convention: delta_i = 10^(-4 + i/64), i = 0..447 (eps' <= 1000), then eps' = 1000 |
| host thread pool for per-node work | `parallel_for_index` | |

`src/archetypes/hadronic/bh_sampler_io.h`: `FixtureTable`,
`format_fixture_file`, `write_fixture_file` (the contract-1.4 header style,
`columns` and `data_sha256` written last, `%.12e` one-space columns) and a
minimal validating reader `read_fixture_file` (strict JSON header values,
literal-byte `data_sha256`, finite rows of the right width).

`tools/bh_sampler_dump.cpp` (`bh_sampler_dump qk_table --out FILE [--status S]
[--threads N]`, standard library and threads only, exits 2 on option errors)
writes `tests/data/hadronic/cpp-sampler-candidate/qk_table.txt`: format
`bh-qk-table`, producer `entity-cpp`, version = git hash of the build, columns
`eps_prime q_k q_k_eq37 mean_E_minus mean_p_cos_theta sigma_from_g
sigma_maximon`, 449 rows; 2.1 s on 4 threads. File written at `4b2f129`:
`data_sha256` `8b4f6c9a2fd2fd63225ec2c9cfd605065afd87de01b9a7fc72ccaaeaa2316ee3`
(the header's `version` is the only build-dependent token; the data block is
deterministic).

`tests/archetypes/bh_restframe.cpp` (`ARCHETYPES::bh_restframe`, 4.7 s on 4
threads): 8 contract items and 12 implementation-chosen checks (prefix
`(impl)`), diagnostics as `[DIAG]`. Log of the run kept at
`docs/hadronic/logs-step4-stage1-bh_restframe.log`.

#### Anchors (facts, measured)

Nodes: eps' on the 449-node grid; E' on the 31 interior nodes x_j = (1 -
cos(pi j/32))/2 of x = (E' - 1)/(eps' - 2).

| anchor | quantity | measured | tolerance | verdict |
| --- | --- | --- | --- | --- |
| A1 | max over the 449 eps' nodes in [2.001, 1000] of \|int g dE' / sigma~ - 1\| | 1.005e-4 (at eps' = 3.981, the eq.-10 side of Maximon's junction; the series step there is 1.5e-4) | 1e-3 | pass |
| A1 (diag) | the same from eps' - 2 = 1e-4 | 1.005e-4 (at eps' = 2.0001 the long-double evaluation gives -1.2e-7) | | |
| A2 | max over 449 x 31 nodes of \|int h dcos / g - 1\| | 2.33e-5 (at eps' = 2.000107, the smallest p') | 1e-3 | pass |
| A2 | e-/e+ symmetry of g | bitwise equal at 12 exactly complementary pairs; 2.3e-14 at generic nodes (rounding of eps' - E') | exact | pass |
| A3 | q_k at eps' - 2 = 1e-3 | 1.99974999 (2 - q_k = 2.5e-4; eq. 3.7 gives 1.99979) | \|q_k - 2\| <= 1e-3 | pass |
| A3 | <E'-> at eps' - 2 = 1e-3 | 1.0005 (= eps'/2 by symmetry) | \|<E'-> - 1\| <= 1e-3 | pass |
| A3 (diag) | q_k eps'/(4 ln eps') at eps' = 10, 30, 100, 300, 1000 | 1.0566, 1.1009, 1.2332, 1.3972, 1.6032 (eq. 3.7: 1.0577, 1.1062, 1.2351, 1.3942, 1.6036) | reported | not constant on the kernel domain: eq. 3.7 carries ln^2 and ln^3 terms, so the ratio grows like 0.1 ln eps'; the limit of eq. 3.9 is not reached below 1000 |
| A4 | q_k machinery vs eq. 3.7 on the nodes in [2.05, 1000] | 5.47e-3 (at eps' = 29.4); sample ratios minus one: -9.4e-4 (2.05), -4.1e-3 (2.5), -3.3e-3 (3), +8.0e-4 (5), -1.0e-3 (10), -4.8e-3 (20), -4.5e-3 (50), -1.5e-3 (100), -2.4e-4 (1000) | 2% | pass |
| A4 | q_k machinery vs phi_fit'/(kappa sigma~) on the same nodes | 4.04e-3 (at eps' = 49.0) | 2% | pass |
| A5 | phi by int eps' sigma~ q_k (machinery q_k, linear in ln(eps' - 2) between nodes, threshold law below 1e-4) vs phi_fit, 301 kappa in [2.001, 1000] | 3.11e-3 (at kappa = 74.3); +1.15e-3 (2.5), +1.15e-3 (3), +1.6e-4 (5), -1.6e-4 (10), -1.7e-3 (25), -2.9e-3 (100), +8.5e-4 (1000) | 2% | pass |
| A5 (diag) | phi machinery vs the step-1 cumulative (eq. 3.7 q_k) | 3.9e-3 | | the fit budget of eq. 3.7 |
| A5, second part | direct-boost loss table vs I_exact (0.5%) | needs the chain: stage 2 | | not yet |

Implementation-chosen checks (all pass after the tolerance note below): A1
self-convergence 5.2e-8 (order + 8, panels x 2; the floor is the long-double
cancellation at eps' - 2 = 1e-4); A2 self-convergence 4.4e-12; closed form vs
the numerical integral of law 2 at 384 points with x in [0.01, 0.95] and 1 -
cos theta in [1e-8, 1.99]: 2.29e-7 (at eps' = 2.001, x = 0.01, i.e. p' =
4.5e-3, where the closed form's bracket cancels by ~1/p'^2 even in long
double; 1e-8 or better for p' >= 0.02); uniqueness of the sign choice;
normalisation of law 2 through its own angular integral 8.9e-12 at 3 nodes;
law-2 quadrature self-convergence 2.4e-10; q_k and <p cos> self-convergence
2.5e-8 (floor: the eps' - 2 = 1e-4 node); int int h vs sigma~ at every node
1.0e-4; the fixture vs the live computation 4.9e-13.

Tolerance note (convention, stated): the two implementation-chosen tolerances
"closed form vs numerical integral" and "q_k self-convergence" were first
written as 1e-7 and 1e-8 and failed the first run at 2.29e-7 and 2.48e-8,
both at the roundoff floor of the long-double evaluation near threshold, not
at a quadrature error; they were set to 1e-6 and 1e-7 before the commit.
No contract tolerance was touched.

Precision of the closed form (fact): for p' < 1e-3 (x <= 1e-6 at eps' = 2.001)
the closed form deviates from the numerical integral by 1e-4 to 1e-1 and at
p' = 4.5e-6 by a factor 100; h is there O(p') and the deviation is invisible
in every integral (A2 holds to 2.3e-5 at the worst node). Stage 2 builds the
CDF tables from the closed form on an x grid whose interior nodes keep p' above
1e-3 at every eps' node, and the end points carry h = 0 exactly.

Open questions for the referee: (1) the formula numbers 3D-1000, 3D-2000 and
3D-3000 of Motz, Olsen and Koch 1969 and the equation numbers of Heitler 1954
are quoted from memory; (2) the laws are validated numerically (A1 against
Maximon's sigma~, A2, A4 against eq. 3.7, A5 against the phi fits), not
against the printed formulas; (3) A6 (an external numerical table of the
energy distribution) was not attempted here, since no literature table is
available in the container: by the contract, A6 is then replaced by A1 plus A8.

- stage 1 done, commit `2893cfb`, pushed (20 of 20 checks pass; log in
  `docs/hadronic/logs-step4-stage1-bh_restframe.log`).

### Stage 2 (CDF tables and the sampling chains, host)

- stage 2 started, HEAD `2893cfb`.
