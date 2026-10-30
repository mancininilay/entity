# P2 step 6, C++ side: the Born-q_k loss tables (`qk_source = born`). Report

Claude 1 (Claude Code on Nilay's Mac), 2026-10-30. Fork `mancininilay/entity`,
branch `dev/hadronic-bornqk` (created from `dev/hadronic` at `051bb896`, in
the worktree `Entity Local Experiments/worktrees/dev-hadronic-bornqk/`).
Written against the local brief of 2026-10-30 and the table contract revision
1.5 (`Resources/p2-hadronic/notes/2026-10-29-claude1-table-contract-1.5-born-qk.md`,
normative), revision 1.4 for the grid, schema and item-8 definition, and the
sampler contract section 1 for the laws. Reference data: the **bhref 0.3.0
candidate** (Codex 1, unfrozen, under Claude 3's review), copied into
`tests/data/hadronic/bhref-0.3.0-candidate/` with the candidate manifest hash
`b393f99a7781fb68ae8a365ad1316b04c9aa40f4fdd0ca269ef03251e779517e`; replaced
by `bhref-0.3.0/` at the freeze on the mastermind's instruction.

Statements are labelled fact (with the number and where it was measured),
convention or open question. No pre-registered tolerance is adjusted; a
failure is reported with its numbers and mechanism. Code facts with file,
line and commit. Independence: q_k^Born and I_Born are computed by the C++
machinery of step 4 (`rf::moments` in `bh_restframe.h`) and the step-1
cumulatives; the candidate's method was not read (its report only for the
file layout and the numbers to compare against).

## 0. Stage list (brief, "Stages")

| stage | content | status |
| --- | --- | --- |
| 1 | the q_k^Born provider, the `qk_source` switch (parser, options, startup report, exchange-file key), A10b, A10c, A3, the fit-path regression, the startup cost | done, `93a4c685` |
| 2 | cumulatives and tables from q_k^Born; item 8 against the five candidate tables; J and eps_mean equality; threshold exponent and UR anchors re-run; A5-numerical and A5-fit-budget | done (this commit), all pass |
| 3 | consumers: every step-2 to step-4 ctest entry with `qk_source = born`, single and double; A12 against the Born chain; A7 at 4 s.e. with no floor against I_Born/J | done (this commit): everything passes on the Born tables, A7 re-registered passes (max 3.6 s.e.), A12 passes |
| 4 | report, checkpoint, changelog, fork-changes | done (`Resources/p2-hadronic/notes/2026-10-30-claude1-born-qk-report.md`) |

## 1. Environment (facts)

| item | value |
| --- | --- |
| `dev/hadronic` HEAD at branch creation | `051bb896` |
| compiler, deps | Apple clang arm64; Kokkos 5.0.1 (Serial + OpenMP), ADIOS2 2.11.0 non-MPI; `long double` = `double` |
| builds | `Entity Local Experiments/worktrees/builds/bornqk-single/` (`-D pgen=bh_proto -D TESTS=ON -D output=ON -D precision=single -D mpi=OFF`); the double build is added at stage 3 |
| reference | `tests/data/hadronic/bhref-0.3.0-candidate/`: `mono05`, `mono03`, `powerlaw`, `blackbody`, `mono05_mu100` (32-key headers, `qk_source = "born-3D2000"`, status `candidate:awaiting-Claude-3-review`), `qk_table.txt` (449 nodes), `qk_midpoints.txt` (448); file hashes from the candidate manifest, verified by `shasum -c` |

## 2. Stage log

### Stage 1 (provider and switch)

Files: `src/archetypes/hadronic/bh_born_qk.h` (new), `bh_physics.h`
(`QkSource`, `Quadrature::qk_source`, the Born declaration, `Cumulatives`
dispatch), `bh_table_io.h` (the 32nd header key), `bh_options.h`
(`--qk_source`), `tools/bh_tables_dump.cpp`, `pgens/bh_proto/bh_proto_params.h`
(`qk_source`), `pgens/bh_proto/pgen.hpp` (startup report), `tests/archetypes/bh_born_qk.cpp`
(new, ctest `ARCHETYPES::bh_born_qk`), `tests/hadronic/bh_proto_params.cpp`,
`tests/data/hadronic/bhref-0.3.0-candidate/` (README, SHA256SUMS, seven files).

What was implemented:

- **Provider** (`arch::bh::born`): q_k^Born(k) = 2 <(E' - p') + p'(1 - cos
  theta')>_h by direct quadrature of the step-4 machinery (`rf::moments<double>`,
  the energy-angle law `dsigma_dE_dcos_closed`, MOK 3D-2000 = Blumenthal eq.
  10, normalised by its own integral) at the order gl 24, 32 energy panels,
  angular panel width 0.25 in t = ln(1 - cos + w) (step 4's default refined);
  a node table at 64 nodes per decade of delta = k - 2 from 1e-5 to k_max =
  1000 (the last node exactly at k_max; 513 nodes), built in parallel on the
  host threads; a not-a-knot cubic spline of ln q_k in ln delta between the
  nodes; below the first node the threshold continuation q_k = 2 - s delta
  with s from the first node pair; above k_max the value at k_max. Built once
  per process (function-static), its cost recorded.
- **Switch**: `QkSource {Fit, Born}` in `bh_physics.h`; `Quadrature::qk_source`
  with the library default `Fit` (every existing table and test byte-identical
  in behaviour); `Cumulatives::integrand` dispatches on it (the Fit branch is
  the step-1 code). The Born branch calls `born_q_k_delta`, which reads the
  provider registered by `bh_born_qk.h` (see the link note below).
- **Inputs**: `setup.qk_source = "born" | "fit"` (default born, the
  contract's proposal for the production default), refused otherwise;
  `bh_tables_dump --qk_source born | fit` (default born). The exchange-file
  writer adds the key `qk_source` after `status` (`"born-3D2000"` /
  `"chodorowski-3.7"`, the reference's placement); the reader accepts the
  key anywhere, validates its value and reads a file without it as
  `chodorowski-3.7`. Startup report: the source, the node count and grid, the
  threshold slope and the build time; the table summary line names the
  source.

Facts measured in stage 1 (single build; the test executable
`test-archetypes-bh_born_qk.xc`):

| test | registered tolerance | measured | verdict |
| --- | --- | --- | --- |
| A10b direct quadrature at the 449 contract nodes vs the candidate's `qk_machinery` | 1e-8 relative | **3.81e-12** (worst at eps' = 2.0001037) | pass |
| A10b the interpolant at the same 449 nodes vs the candidate | 1e-8 | 3.81e-12 | pass |
| A10c interpolant vs direct quadrature at the 448 contract midpoints | 1e-8 | 2.94e-13 at 128 nodes per decade; at the chosen 64 per decade: see the stage-1 rerun below | pass |
| A10c diagnostic: C++ direct vs the candidate's `q_direct` at the midpoints | reported | 3.30e-12 | |
| the candidate's eq. 3.7 column vs the C++ fit `q_k` | reported | 4.5e-13 | consistent anchors |
| A3 registered in the brief: abs(q_k^Born(2 + 1e-9) - 2) <= 1e-10, direct and provider | 1e-10 | **2.50e-10** (both) | **FAIL**: the Born law has q_k(2 + delta) = 2 - delta/4 + O(delta^2) (fact: (2 - q)/delta = 0.25000006 at delta = 1e-5, 1.02e-5, 1.04e-5 from the direct nodes), so at delta = 1e-9 the exact value is 2 - 2.5e-10 and no correct provider can be within 1e-10 of 2; the registered tolerance was written below the physical slope |
| A3 proposal: q_k^Born(2 + delta) = 2 - s delta at delta = 1e-9 with the measured slope s | 1e-3 relative on the slope | 1.7e-7 | pass (labelled) |
| A3 diagnostic: q_k eps'/(4 ln^2 eps') at eps' = 1e3 | reported (leading-order limit 3/14 = 0.2143) | 0.2321 (eq. 3.7 gives 0.2321) | the subleading terms are 8 % at 1e3, as in step 4 |
| fit path untouched: `Cumulatives(QkSource::Fit)` vs a standalone step-1 re-integration of Phi, phi, psi at six deltas | 1e-12 | 1.6e-15 | pass |
| Born vs fit cumulatives: Phi identical; phi differs | reported | Phi bitwise equal; max abs(phi_Born/phi_fit - 1) = 4.8e-3 at k = 52 (within the 2 % fit budget of A4) | |
| startup cost of the Born table (10 host threads) | reported | 7.8 s at 128 nodes per decade (1025 nodes, 7.9 ms per direct evaluation); 64 per decade chosen for production: see the rerun | |
| parser additions (`qk_source` default, values, refusals) | | pass (T23 suite) | pass |

Rerun at the production density, 64 nodes per decade (513 nodes): the table
builds in **5.1 s** on 10 host threads (10.7 ms per direct evaluation of
q_k^Born; the dump tool reports 5.0 s); A10b unchanged (3.81e-12, the
contract nodes are direct evaluations in both densities); **A10c = 5.18e-10**
(worst midpoint eps' = 5.22; the 128-per-decade value 2.9e-13 was not the
h^4 scaling of the spline error, 16 x, but the quadrature noise floor: the
spline error dominates only at 64 per decade, and stays 20 times below the
1e-8 tolerance). Item-8 preview (not the registered test, which is stage 2):
`bh_tables_dump --type mono --eps0 0.5 --qk_source born` against the
candidate's `mono05.txt` at the 466 matched Gamma nodes (max |dGamma| 9.4e-10
from the serialised digits), where I > 1e-3 of the peak: max |dI/I| =
2.2e-10, max |dJ/J| = 2.7e-11; the Born table's gamma_min 2.130004358 equals
the candidate's (the fit's is 2.130008808) and I_peak 198.5799 the
candidate's 198.57990988. The header key `qk_source` reads `"born-3D2000"`
and `"chodorowski-3.7"` in the two dumps. The parser suite passes (121 with
the three additions). A link defect found by the full build (every
translation unit using the cumulatives needed `bh_born_qk.h`, and the
converse include from `bh_physics.h` is circular when a unit includes
`bh_restframe.h` first) is fixed by a registration hook: `bh_physics.h`
holds a function-pointer slot `born_qk_hook()`, `bh_born_qk.h` fills it at
static initialisation (`born::registered`), and `born_q_k_delta` throws a
`logic_error` naming the header if a program asks for Born cumulatives
without any unit including it (never a silent fallback to the fit).

### Stage 2 (cumulatives and tables)

No new physics code: the cumulatives of stage 1 already build phi and psi
from the chosen q_k on the existing grid in ln(k - 2) (Gauss-Legendre per
cell), and the start below delta0 = 1e-8 is the leading threshold law
(pi/12) delta^4 for phi, common to both sources since q_k(2) = 2; its
next-order error, O(delta0) relative on a part that is (delta0/delta)^4 of
any cumulative, is below 1e-17 at the table's lower edge (eps'_hi - 2 ~
5e-3 there): quadrature, no re-derived series. Tests: `bh_fixtures.cpp` now
compares both references (0.1.0 with the fit path, the 0.3.0 candidate with
the Born path) and adds the contract-1.5 statements; `bh_physics.cpp` and
`bh_chain.cpp` take `BH_QK_SOURCE=born` from the environment (ctest entries
`ARCHETYPES::bh_physics_born`, `bh_chain_{budget, a7w, a7}_born`); in Born
mode the budget group's A5b is replaced by A5-numerical and A7 by the
re-registered 4-s.e.-no-floor criterion.

| test | registered tolerance | measured | verdict |
| --- | --- | --- | --- |
| item 8, fit path vs bhref 0.1.0 (regression; five files, 35 checks) | 1e-6 core, 1e-3 tail | max |dI/I| 4.3e-12, |dJ/J| 4.1e-12 (unchanged from step 1) | pass |
| item 8, Born path vs the 0.3.0 candidate, node placement (five files) | 1e-9 in y | 4.6e-12 to 2.3e-11 (the lower edge recomputed on I_Born moves the nodes, as in the candidate) | pass |
| item 8, Born: |dJ/J| core / tail (mono05, mono03, powerlaw, blackbody, mono05_mu100) | 1e-6 / 1e-3 | 8.2e-13, 8.6e-13, 7.0e-13, 8.6e-13, 5.1e-12 (core); tail of the same order | pass |
| item 8, Born: |dI/I| core / tail | 1e-6 / 1e-3 | 2.41e-10, 2.47e-10, 1.96e-10, 1.87e-10, 2.05e-10 (core); tail of the same order | pass (the 2e-10 floor is the candidate's interpolant, 6.5e-12 on q_k, integrated) |
| J and eps_mean of the Born path = fit path at every reference node (five files) | 1e-12 | exactly 0 (same Phi) | pass |
| A5-fit-budget [labelled anchor]: |I_Born/I_fit - 1| where I > 1e-3 I_peak | 1 % | 0.486 % (mono05, Gamma_p 45.5), 0.486 % (mono03, 75.4), 0.202 % (powerlaw, 30.6), 0.192 % (blackbody, 24.2), 0.401 % (mono05_mu100, 2.80); the candidate's own numbers are 0.4862, 0.4862, 0.2020, 0.1919, 0.4012 % | pass (two implementations agree on the fit budget to four digits) |
| step-1 suite on the Born tables (`bh_physics_born`): threshold exponents mono J/I, powerlaw J/I | 0.02 | -8.0e-5 / -6.0e-5, -9.4e-5 / -7.1e-5, -6.1e-5 / -4.6e-5 | pass (the candidate: 9.4e-5, 6.1e-5) |
| same: item 2 phi vs the Chodorowski fits (now a fit budget) | 1 % | 0.31 % | pass |
| same: item 5 I vs eq. 3.11 with the phi fits at Gamma = 450/eps_cut (four baths) | 1 % | 7.4e-4, 7.4e-4, -2.4e-3, -2.2e-3 | pass |
| same: CRPropa CMB anchor | 3 % | 0.18 % (the candidate 0.014 %: the C++ anchor rows are the 6-row fixture, the candidate's a finer one) | pass |
| same: every other item (55 contract, 126 implementation checks) | as registered | all pass, 14.6 s | pass |
| A5-numerical (`bh_chain_budget_born`): direct-boost I (laws' q_k) vs the Born table where I > 1e-3 I_peak, 2169 nodes | 1e-3 | 3.65e-5 (worst mono05, Gamma_p 91.1; the candidate's 2.85e-4 includes its photon-sampling chain) | pass |
| budget group on Born tables: sampled-chain <D> vs the laws, with the convergence estimate | 1e-3 | 9.68e-4 (unchanged from step 4: the chain's own budget) | pass |

### Stage 3 (consumers on the Born tables)

Method. Every step-2 to step-4 kernel suite takes `BH_QK_SOURCE=born` from
the environment in its table builder (`bh_loss.cpp`, `bh_pairs.cpp`,
`bh_sampler.cpp`: `make_tables`), and the ctest registers a `_born` entry
per group next to the fit one (`KERNELS::bh_loss_*_born`, `bh_pairs_*_born`,
`bh_sampler_*_born`); the references of those suites (the exact-rate RK4 of
T5, the replayed debt mechanism of T10, the sampler chain of A12) are built
from the same `Tables` object, so each `_born` entry is the registered test
on the Born tables with its registered tolerance. The run-level entries
(`HADRONIC::*`) run the problem generator with its default `qk_source =
born`; `bh_check` builds its reference rates with `--qk_source born | fit`
(default born, matching the generator) so that T5b and T7b compare a Born
run with a Born ODE. The chain suite's A7 (unweighted, N = 1e6 at every
node) and A7 weighted run against I_Born/J with the re-registered 4 s.e.
and no floor (`bh_chain_a7_born`, `bh_chain_a7w_born`); the step-4 A7
against the fit tables stays as the registered history. A12 against the
Born chain with the literal tolerances is `bh_sampler_a12_born` (the
sampler's cost reference is the Born-chain <D> of `moments_mono05`, so A12
tests the kernel's packet law against the chain, which does not change with
the table; the table enters only through the Gamma nodes and the debt
preset).

Results (ctest selection `bh_born_qk|bh_fixtures|bh_physics_born|
bh_chain_(budget|a7w|a7)_born|KERNELS::bh_(loss|pairs|sampler)_*_born|
HADRONIC::*`, 47 entries per build; single 1839 s wall with 10 threads,
double with 5 threads alongside):

| test | registered tolerance | measured single / double | verdict |
| --- | --- | --- | --- |
| A7 re-registered (contract 1.5): abs(<D>_MC - I_Born/J) <= 4 s.e. at every node of the five baths, N = 1e6, no floor (2169 nodes) | 4 s.e. | 0 of 2169 nodes fail; max 3.63 s.e. (both builds; same seeds) | **pass** (the step-4 registered failure against the fitted I/J, 18.6 s.e., is removed by the Born tables: the sampler and the table now share their mean loss) |
| A7 diagnostics: max abs(<D>_MC/(I_Born/J) - 1) | reported | 9.1e-3 at mono05 Gamma_p 84.9 (a 3.2-s.e. node: <D> is 30 there and 1e6 draws resolve 0.3 %) | |
| A7 weighted re-run against I_Born/J: E_p[1/D] <D> = 1 at Gamma_p in {1.05 Gamma_th, 3, 10}, five baths | 4 s.e. | 2.14 s.e. (both) | pass |
| A12 against the Born chain, literal tolerances (mean cost 1 %, number and energy per paid energy 2 %, N = 2e5 per node, 466 nodes, the ions at the step-4 reference's nodes, which the Born grid shifts by up to 8.7e-4) | 1 % / 2 % / 2 % | single: 0 / 0 / 0 failing nodes, max 0.918 % (cost, Gamma_p 73.9), 0.910 % (number, 73.9), 1.9e-4 (energy); 9.32e7 pairs in 1273 s; double: 0 / 0 / 0 failing nodes, max 0.951 % (cost), 0.942 % (number), 2.0e-4 (energy) | pass (single); the fit-mode A12 of step 4 fails its 1 % at Gamma_p 76.53 with 1.285 % in this build (the registered row), the Born run's largest deviation is 2.9 combined s.e. |
| T5 on the Born tables: K(t) vs the exact-rate RK4 built on the same tables, Gamma_p 6 / 3 groups | 1e-3 | 9.3e-5 / 8.9e-5 single, 8.6e-5 / 9.2e-5 double | pass |
| T5b, T5fine, T6, T9, tags, above, subcycling, overflow, huge, tables on the Born tables (`bh_loss_*_born`, 11 entries) | as registered | all pass in both builds | pass |
| T10, poisson, none, energy, momentum, T12, T10b, T-exp, R1, R2, R3, C-MPI-1 on the Born tables (`bh_pairs_*_born`, 12 entries) | as registered | all pass in both builds | pass |
| sampler tables, A11, identity, spectrum on the Born tables (`bh_sampler_*_born`, 4 entries besides A12) | as registered | all pass in both builds | pass |
| run-level entries (`HADRONIC::*`, 16 entries, the generator with its default `qk_source = born` and `bh_check` references at `--qk_source born`): reject, T7a, T7b, T12, T12b, T13, T10 determinism, T-exp, r2 tiny, T14b, sampler overflow, parser | as registered | all pass (3 cross-build / Serial entries skipped as before) in both builds | pass |
| `bh_born_qk`: A3 registered limit (stage 1) | 1e-10 | 2.5e-10 | **FAIL** (the Born slope 1/4, stage 1) |

The continuous-mode cooling curves on the Born tables differ from the fit
ones by at most the fit budget of I, 0.49 % at the worst node (contract 1.5:
"change by <= 0.5 %"): fact from the A5-fit-budget row of stage 2, the T5
tolerance (1e-3 against the RK4 of the same table) untouched.

## 3. Open questions for Nilay

1. **A3's registered limit test** (brief, stage 1: "q_k(2) = 2 to 1e-10 at
   eps' - 2 = 1e-9") cannot be met by a correct implementation: the Born
   threshold slope is exactly 1/4 (fact above), so q_k(2 + 1e-9) = 2 -
   2.5e-10. Recommendation: re-register as abs(q_k(2 + delta) - (2 -
   delta/4)) <= 1e-10 at delta = 1e-9, or test q_k(2) = 2 exactly (the
   provider returns 2 at delta <= 0 by construction) and the slope 1/4 as the
   limit statement; the failed row stays as history.
2. The production default `qk_source = born` (contract section 3) is
   implemented in the parser as proposed; confirmation at the step closure.
3. Node density: 128 nodes per decade gave A10c = 2.9e-13 at 7.8 s startup;
   64 per decade (the reference's density) is the production choice if its
   A10c stays far below 1e-8 (expected 16 times larger, about 5e-12) at
   half the cost. Recommendation: 64.

## 4. Deferred / waiting for the freeze

The replacement of `bhref-0.3.0-candidate/` by the frozen `bhref-0.3.0/`
and the re-run of A10b, A10c, item 8 and the A5 anchors against it.

## 5. Branch head

Stage 1 `93a4c685`, stage 2 `d6d0a62d`, stage 3 `3316d77c`, final (this commit): recorded in the Resources note.
