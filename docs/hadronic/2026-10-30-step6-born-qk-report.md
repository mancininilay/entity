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
| A3 registered in the brief: abs(q_k^Born(2 + 1e-9) - 2) <= 1e-10, direct and provider [**history**: superseded by the slope-form registration of 2026-10-30, §7 F1; no longer an active failure] | 1e-10 | **2.50e-10** (both) | **FAIL (history)**: the Born law has q_k(2 + delta) = 2 - delta/4 + O(delta^2) (fact: (2 - q)/delta = 0.25000006 at delta = 1e-5, 1.02e-5, 1.04e-5 from the direct nodes), so at delta = 1e-9 the exact value is 2 - 2.5e-10 and no correct provider can be within 1e-10 of 2; the registered tolerance was written below the physical slope |
| A3 proposal: q_k^Born(2 + delta) = 2 - s delta at delta = 1e-9 with the measured slope s [history: replaced in round 2 by the decided gate abs(q_k(2 + delta) - (2 - delta/4)) <= 1e-10 and the implementation check abs(s - 1/4) <= 1e-6, §7 F1] | 1e-3 relative on the slope | 1.7e-7 | pass (labelled) |
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
| `bh_born_qk`: A3 registered limit (stage 1) [history: re-registered 2026-10-30, §7 F1] | 1e-10 | 2.5e-10 | **FAIL (history)** (the Born slope 1/4, stage 1) |

The continuous-mode cooling change fit -> Born (contract 1.5: "change by
<= 0.5 %") is measured on trajectories (round 2, Codex 1 F6; the
instantaneous rate budget of 0.49 % at the worst node is a budget, not the
proof): the same kernel, dt and steps on both tables, mu_d = mu_r, Gamma_p =
6 and 3 groups to t = 12262 (201 records): max |K_Born(t)/K_fit(t) - 1| =
**6.36e-4** (6) and **4.06e-4** (3) in the single build (`bh_loss_t5_born`,
F6 rows; the double build in section 7); Codex 1's independent run at
fdbef82f: 0.0637 % / 0.0405 % (single), 0.0695 % / 0.0413 % (double), 342
records. The T5 tolerance (1e-3 against the RK4 of the same table) untouched.

## 3. Open questions for Nilay

1. [Decided 2026-10-30 ("yes to all"): re-registered on the slope form, implemented in round 2, §7 F1.] **A3's registered limit test** (brief, stage 1: "q_k(2) = 2 to 1e-10 at
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
and the re-run of A10b, A10c, item 8 and the A5 anchors against it: done,
section 6.

## 6. Frozen fixtures (2026-10-30 afternoon)

bhref 0.3.0 frozen (Codex 1) and verified (Claude 3): frozen `SHA256SUMS`
sha256 9b2d709765bfd795664ba78d79581b2a02b375e712d2a03c87f152518a2eb815, 2293
entries, data bytes and every data_sha256 of the candidate unchanged, status
lines `frozen:0.3.0`. Import: `tests/data/hadronic/bhref-0.3.0-candidate/`
removed (git rm); `tests/data/hadronic/bhref-0.3.0/` = the 12 files of the
release's `FIXTURES.sha256` (tables/{mono05,mono03,powerlaw,blackbody,
mono05_mu100}.txt, fixtures/{qk_table,qk_midpoints,moments_<5 baths>}.txt),
`SHA256SUMS` = that list without the directory prefixes (shasum -c 12/12),
README with the manifest hash. `ls -lOR | grep dataless` under
`Entity Local Experiments/bhref/v0.3.0/`: 0 placeholders at import time; the
manifest hash recomputed on disk before copying.

Code:
- `tests/archetypes/bh_fixtures.cpp`: the Born reference is `bhref-0.3.0`,
  status prefix `frozen:0.3.0`.
- `tests/archetypes/bh_born_qk.cpp`: fixture path `bhref-0.3.0/`; wording
  "candidate" -> "frozen release"; new check of the header keys by name: the
  frozen `moments_mono05.txt` carries `qk_source` as its 9th header line
  (contract 1.5.1 (ii): the "after status" rule is for the 32-key rate
  tables) and `parse_fixture_file` returns it by name; the frozen `mono05.txt`
  with its `qk_source` line moved to the end of the header parses through
  `parse_table_file` to the same qk_source, status, data_sha256, nodes and I.
  Both readers were already key maps; the check makes the fact a test.
- `tests/kernels/bh_sampler.cpp` (A12): in Born mode (`BH_QK_SOURCE=born`) the
  reference is the frozen `bhref-0.3.0/moments_mono05.txt`, column
  `D_Born_table` (the release's full-chain quadrature at the Born-grid nodes;
  s.e. 0), header qk_source checked; the ions at the reference's nodes (=
  this table's nodes to 1.03e-11 relative, verdict at 1e-9). Fit mode keeps
  the 0.2.0 `Dmean`/`Dmean_se` reference. Lower-edge node: see below.
- `tests/archetypes/bh_chain.cpp` (A7): in Born mode the same N = 1e6 draws are
  also compared with the frozen tables' `Dmean` column at every node (nodes
  identical to this table's to 1e-9, checked), 4 s.e., no floor.

Lower-edge finding (test mechanics, single build): the frozen reference's
first node is the table's lower edge (gamma_min = 2.1300043576). An ion
placed exactly there stores u as a float; the recomputed Gamma_p falls below
gamma_min (relative error ~6e-8 in u, ~7e-8 in Gamma_p) and the kernel
emits one closing packet per launch: 200 packets at that node instead of
2e5 (the test's own N guard failed; the three physics comparisons passed at
every one of the 466 nodes, the edge node included: cost 3.75982 vs the
reference 3.75907). In the double build the same node gives 200000 packets.
The step-6 run used the 0.2.0 fit-grid nodes, 8.7e-4 above the Born edge,
and never met the edge. Change: in Born mode the lower-edge ions are placed
at Gamma_node (1 + 1e-6), where <D> differs by 2e-7 relative from the node
value compared (negligible against the 1 % tolerance); diag line printed.

Measured (single | double):
- A10b 3.810e-12 (direct), 3.810e-12 (interpolant) | same; A10c 5.181e-10 |
  same; A3 as written 2.50e-10 vs 1e-10 (registered failure kept at that
  head; the slope-corrected proposal row 1.67e-7; re-registered in round 2,
  §7 F1); header-key checks pass.
- item 8 vs the frozen tables (bh_fixtures, 85/85 both builds): file sha256
  and headers (status frozen:0.3.0, qk_source born-3D2000) verified; node
  placement max |dy| 4.57e-12 (mono05); |dJ/J| <= 8.2e-13; |dI/I| core and
  tail 2.41e-10 (mono05), 2.47e-10 (mono03), 1.96e-10 (powerlaw), 1.87e-10
  (blackbody), 2.05e-10 (mono05_mu100); A5-fit-budget 0.4862, 0.4862, 0.2020,
  0.1919, 0.4012 %; the fit path vs 0.1.0 unchanged (4.3e-12).
- A7 (bh_chain_a7_born, 256 s): re-registered vs the C++ I_Born/J 0 of 2169
  nodes fail, max 3.631 s.e.; vs the frozen Dmean 0 of 2169 fail, max 3.631
  s.e., node shift 0 | identical in double.
- A12 (bh_sampler_a12_born): single, exact nodes (before the nudge): 0 / 0 / 0
  failing nodes, max 0.634 % cost and 0.630 % number at Gamma_p 82.01,
  1.57e-4 energy, 200 packets at the edge node (guard FAIL), 1397 s; double,
  exact nodes: same maxima, 200000 packets everywhere, all pass. After the
  nudge: single 0 / 0 / 0 failing nodes, max 0.634 % cost, 0.630 % number
  at Gamma_p 82.01, 1.57e-4 energy, 200000 packets at every node, 1048 s;
  double identical, 987 s.

Branch head after this section: the commit carrying it (hash recorded in the
Resources note `notes/2026-10-30-claude1-born-qk-report.md` section 6).

## 5. Branch head

Stage 1 `93a4c685`, stage 2 `d6d0a62d`, stage 3 `3316d77c`, final (this commit): recorded in the Resources note.

## 7. Round 2 and integration (Claude 1, 2026-10-30 night)

Resolution of Codex 1's review `Resources/p2-hadronic/reviews/2026-10-30-codex1-cpp-born-qk.md` (master folder) (numerical acceptance, integration HOLD at fdbef82f) per the brief `Resources/p2-hadronic/notes/2026-10-30-claude1-local-brief-step6-integration.md`, then the merge of `dev/hadronic` = 757db4c3 (weights audit, event mode round 2) and `dev/hadronic-event-followups` = a8fd781c into `dev/hadronic-bornqk`, E10, the full ctests. Worktree `Entity Local Experiments/worktrees/dev-hadronic-bornqk/`; one commit per finding, each pushed; no merge into `dev/hadronic`; `dev-hadronic-perf` untouched. Master-folder report: `Resources/p2-hadronic/notes/2026-10-30-claude1-born-qk-report.md` §7; ctest logs `Entity Local Experiments/worktrees/builds/log-bornqk-{single,double}-integration-*`.

### 7.1 Resolution log

| finding | resolution | commit | evidence (single | double) |
| --- | --- | --- | --- |
| F1 (P2) A3 constant-limit gate | **accepted.** Registered gate now abs(q_k^Born(2 + δ) − (2 − δ/4)) ≤ 1e-10 at δ = 1e-9, direct quadrature and provider (Nilay's decision of 2026-10-30); the constant-limit rows printed as history without verdict; the "proposal" slope diagnostic replaced by an implementation check abs(s − 1/4) ≤ 1e-6 on the provider's continuation secant. | 4d46e7ff | direct 8.9e-16, provider 0 (both builds); s − 1/4 = 6.2e-8; `bh_born_qk` 10/10 |
| F2 (P2) weighted A7 not all-node, N not asserted | **accepted.** New group `a7wall` (ctest `ARCHETYPES::bh_chain_a7wall`, `_born`): the weighted chain at every one of the 2169 nodes, exactly N = 1e6 accepted draws asserted at every node, z = abs(mean(1/D) − 1/(I/J))/s.e. ≤ 4 with no floor, in Born mode also against the frozen tables' Dmean; a node reaching the attempts cap (4e9) aborts explicitly (named, the exact-N row fails), N is never reduced. The selected-point `a7w` is labelled a smoke test and asserts exact N. Seeds 7 500 000 + 1000 b + i. | 51a1c7ec | single: 2169/2169 nodes at exact N, 0 fail, max 3.80 s.e. (also vs the frozen Dmean), 2.18e10 proposals, 4900 s | double: the same (host double code, same seeds: 0 fail, max 3.80 s.e., 5162 s); fit tables (history): 1585 of 2169 nodes fail, max 19.8 s.e., exact N everywhere, single; double identical (host double code, same seeds) |
| F9 (P2) Born wiring of two host consumers | **accepted.** `bh_ka2008.cpp` and `bh_sampler_fixtures.cpp` build their loss tables through the `BH_QK_SOURCE` selector (born | fit, refusing anything else), print the source; ctest entries `ARCHETYPES::bh_ka2008_born`, `ARCHETYPES::bh_sampler_fixtures_born`. | 4f4c016d | both pass on the Born tables: ka2008 111 s, sampler_fixtures 18 s (single); double: both pass (163.79 s, 22.39 s) |
| F6 (doc) ≤ 0.5 % cooling change proven by the rate budget | **accepted.** New rows in `bh_loss` group t5, Born mode only: the same kernel, dt (0.2814) and 43584 steps on the Born and the fit tables, μ_d = μ_r, Γ_p = 6 and 3 groups, 201 records to t = 12262; max abs(K_Born/K_fit − 1) against 5e-3. The report paragraph cites this and Codex 1's run (0.0637 / 0.0405 % single, 0.0695 / 0.0413 % double), not the 0.49 % rate budget. | e4e6e5ad | **6.36e-4 / 4.06e-4** | **6.96e-4 / 4.11e-4** |
| F8 (P3) labels | **accepted.** `bh_chain` A7 diagnostic and `bh_sampler` A12 diagnostic and header call the Born tables' I/J the Born recoil (the eq. 3.7 wording kept for the fit tables only); `bh_physics.h` calls `born` the decided production default (2026-10-30), `Fit` the library default and regression path; `bh_born_qk.h` describes the continuation slope as the secant from the exact threshold value to the first positive node. Comments and text only. | 44cb2f39 | compiles, no computation changed |

### 7.2 Integration

- `dev/hadronic` 757db4c3 merged at **d1e58c46**: the only conflict the include block of `pgens/bh_proto/pgen.hpp` (both kept: `bh_born_qk.h`, `bh_event_host.h`); `bh_proto_params.h` auto-merged with both parser extensions (`qk_source`; `mode`, `on_bound`, the event keys), `check_weights_flag` keeps its event clause; `bh_check.cpp`, `bh_proto_params.cpp`, `tests/kernels/CMakeLists.txt` auto-merged.
- `dev/hadronic-event-followups` a8fd781c merged at **f886cb1a** (only `tests/kernels/bh_event.cpp`).
- E10 added at **bab1c3fd**: `tests/kernels/bh_event.cpp` builds its loss tables through `BH_QK_SOURCE` (as the other kernel tests), groups `e10` (Γ_p0 = 6) and `e10_g3` (Γ_p0 = 3) run E2/E3 restframe only; ctest `KERNELS::bh_event_e10_born`, `KERNELS::bh_event_e10_g3_born`; refused on the fit tables.

### 7.3 E10 (contract revision 1.1): E2 restframe on the Born tables, 4 s.e. against the Jensen-corrected ODE on the Born table

Restframe spectrum, a = 1, λ = 0.05 at Γ_p0 (dt = 0.5), N = 1e5, the loss tables with `qk_source = born` (this program's tables through `BH_QK_SOURCE`, the same selector as the other kernel tests), the Jensen-corrected ODE on the same Born table; E[D²] and the Born-chain ⟨D⟩ diagnostic from the frozen 0.2.0 `moments_mono05` as before. Fit-table rows of round 2 (`docs/hadronic/2026-10-29-step5-event-mode-report.md` §6) beside them for the closure.

| target | fit tables, round 2 (single | double) | **Born tables, E10** (single | double) | verdict |
| --- | --- | --- | --- | --- |
| Γ_p0 = 6, ⟨K⟩/K0 = 0.9 | 2.15 | 2.11 s.e. | **1.39** | **1.37** s.e. | pass |
| Γ_p0 = 6, 0.7 | 1.06 | 1.09 | **2.32** | **2.39** | pass |
| Γ_p0 = 6, 0.5 | **8.60** | **8.93** (FAIL) | **1.05** | **0.89** | pass |
| Γ_p0 = 3, 0.9 | **9.83** | **9.90** (FAIL) | **0.31** | **0.39** | pass |
| Γ_p0 = 3, 0.7 | **11.81** | **11.43** (FAIL) | **0.47** | **0.30** | pass |
| Γ_p0 = 3, 0.65 | 5.68 | 6.04 (diagnostic then; FAIL by the 4 s.e. rule) | **1.86** | **1.16** | pass |
| E3 (revision 1.1.1 domain) Γ_p0 = 6 at σ/K = 0.005, 0.01, last in domain | 0.17, 0.018, 0.72 % | 0.17, 0.047, 1.15 % | 0.36, 1.21, 0.50 % | 0.36, 1.21, 0.76 % | pass (1.79 %) |
| E3 Γ_p0 = 3 at σ/K = 0.005, last in domain (curvature 0.600 %) | 0.24 % (1.96 % at the 1.1 endpoint, outside) | 0.25 % | 0.27, 0.17 % | 0.27, 0.56 % | pass; the 1.1 endpoint (outside the domain, curvature 0.98 %) 1.65 % / 1.71 % leading, 0.67 % / 0.73 % curvature-corrected, labelled diagnostic |
| bounded, dropped, capped events | 0 | 0 | 0 | 0 | pass |

The Born-chain ⟨D⟩ diagnostic (ODE with the fixture's 0.2.0 `Dmean` in place of the table's I/J) now sits within 0.6 to 2.7 s.e. of the event mean and 4e-4 or less of the energy lost from the corrected ODE: the fitted table was the whole offset. Runtime 51 s (Γ_p0 = 6) and 148 s (Γ_p0 = 3) in the single build. E10 is **closed in both builds**: the restframe E2 rows that failed in rounds 1 and 2 of the event mode against the fitted table pass at every target on the Born tables. Logs `log-bornqk-{single,double}-integration-e10-direct.txt` and the ctest entries' logs.

### 7.4 Full ctest on the integration head

Main run (every entry except the long groups; `run_full_ctest.sh` of the session, logs `log-bornqk-<build>-integration-main-*`), then the long groups one by one (`log-bornqk-<build>-integration-long-<entry>-*`). 147 entries registered (the 121 of the event branch plus the Born `_born` entries, `a7wall`, `a7wall_born`, `ka2008_born`, `sampler_fixtures_born`, `e10_born`, `e10_g3_born`).

| build | main run | long groups |
| --- | --- | --- |
| single | **144 passed, 3 failed, 3 skipped of 147 in 1391 s** (134 entries run). Failed: `param_container`, `sr-cart-sph` (macOS baselines, as in every closure), `bh_chain_a7w` (the fit-table weighted smoke test against the fitted I/J: 15.6 s.e. at mono05 Γ_p = 2.23, the retained step-4 history row; its Born counterpart passes). Skipped: `t5b_cross_build`, `t8_bit_identity`, `t10_poisson_determinism` (cross-build / Serial, as before). | `bh_event_e10_born` pass (128 s), `bh_event_e10_g3_born` pass (409 s) (§7.3); `bh_event_e2e3` **fail** on the fit-table restframe row at 0.5 only (8.60 s.e., the round-2 row, 17/18 rows pass), `bh_event_e2e3_g3` **fail** on the three fit-table restframe rows (9.83, 11.81, 5.68 s.e., 13/16), `bh_event_e7` **fail** on the three fit-table restframe rows at 0.5 (8.65, 6.98, 4.63 s.e., 46/49): the fit-versus-Born offset, closed by E10 on the Born tables, the fit rows kept as registered history; `bh_event_e6` pass, `bh_event_e4` pass; `bh_chain_a7_born` pass (0 of 2169 nodes, max 3.63 s.e., also vs the frozen Dmean, node shift 0), `bh_chain_a7` pass (fit, max(1 %, 4 s.e.), max relative 9.47e-3); `bh_sampler_a12_born` pass (466 nodes, 200000 packets each, max cost 0.634 % and number 0.630 % at Γ_p 82.01, energy 1.6e-4), `bh_sampler_a12` pass (fit: max cost 0.951 % and number 0.942 % at Γ_p 73.90, energy 2.0e-4; see the note below); `bh_chain_a7wall_born` **pass** (2169 nodes, exactly 1e6 accepted at every node, 2.177e10 proposals, the most at one node 4.059e8 (mono03, Γ_p 153.0), min efficiency 2.46e-3; 0 nodes fail, max 3.80 s.e. (powerlaw, Γ_p 1.0719) against the C++ I_Born/J and identically against the frozen Dmean; 4900 s with the double run sharing the machine); `bh_chain_a7wall` (fit tables) **fail by construction, history**: exact N = 1e6 at every node (the resource row passes), but 1585 of 2169 nodes fail the 4 s.e. gate against the fitted I/J, max 19.8 s.e. (mono05_mu100, Γ_p 2.418; per bath 343 / 335 / 323 / 334 / 250 of 466 / 465 / 429 / 406 / 403): the fit-versus-Born offset of up to 0.5 % against a 1e6-sample s.e. of 2.5e-5, the same offset as the step-4 18.6 s.e. row; 5189 s |
| double | **145 passed, 2 failed, 3 skipped of 147 in 1459 s**. Failed: `param_container` (macOS baseline), `bh_chain_a7w` (the same fit-table history row). Skipped: the same three. | `bh_event_e10_born` pass (104 s), `bh_event_e10_g3_born` pass (241 s) (§7.3); `bh_event_e2e3` **fail** on the fit-table restframe row at 0.5 only (8.93 s.e., the round-2 row, 17/18 rows pass), `bh_event_e2e3_g3` **fail** on the three fit-table restframe rows (9.90, 11.43, 6.04 s.e., 13/16), `bh_event_e7` **fail** on the three fit-table restframe rows at 0.5 (8.88, 6.97, 4.98 s.e., 46/49): the fit-versus-Born offset, closed by E10 on the Born tables, the fit rows kept as registered history; `bh_event_e6` pass, `bh_event_e4` pass; `bh_chain_a7_born` pass (0 of 2169 nodes, max 3.63 s.e., also vs the frozen Dmean, node shift 0), `bh_chain_a7` pass (fit, max(1 %, 4 s.e.), max relative 9.47e-3); `bh_sampler_a12_born` pass (466 nodes, 200000 packets each, max cost 0.634 % and number 0.630 % at Γ_p 82.01, energy 1.6e-4), `bh_sampler_a12` pass (fit: max cost 0.951 % and number 0.942 % at Γ_p 73.90, energy 2.0e-4; see the note below); `bh_chain_a7wall_born` **pass** (identical statistics to the single build: the host chain is double code with the same seeds; 0 fail, max 3.80 s.e., 5162 s); `bh_chain_a7wall` (fit tables) **fail by construction, history**: identical statistics to the single build (host double code, same seeds): exact N everywhere, 1585 of 2169 nodes fail, max 19.8 s.e.; 4882 s |

**Note on the fit-mode A12 row (fact from the logs, mechanism not analysed here).** On the pre-weights base (step 4's cloud run; step-6 stage 3 at 3316d77c, descendant of 051bb896) the fit-mode A12 failed its 1 % at Γ_p = 76.53 with 1.285 %, the registered step-4 row. On the integration head, as already in the event round-2 full ctest at 757db4c3, the same entry passes with max 0.951 % at Γ_p = 73.90 in both builds. The change therefore entered with the weights branch 8e4ba0fb (`src/kernels/injectors.hpp` weight read and the generator's flag rule), not with step 6; the Born-mode A12 is unchanged (0.634 % at 82.01, as at fdbef82f). Flagged for Codex 1's closure; the registered 1 % tolerance is untouched.

### 7.5 Head and records

Integration head: the commit carrying this section (`git log -1 -- docs/hadronic/2026-10-30-step6-born-qk-report.md`), pushed, not merged. Records in the master folder: `Resources/CHANGELOG.md`, `Resources/entity/fork-changes.md`, the P2 checkpoint. Codex 1 closes on the integration head; the fast-forward of `dev/hadronic` follows on green; `dev/hadronic-perf` rebases onto the merged head afterwards.

### 7.6 Open points for Nilay (proposals, not decided)

1. **E7 on the Born tables.** E10 registers the restframe E2 rerun only; the E7 restframe rows (a ∈ {1, 2, 4} at f_max = 0.04, target 0.5) still fail on the fit tables by the same offset (8.65 / 6.98 / 4.63 s.e. single). Recommendation: register an `e7_born` entry (one 25-minute group per build) so that no registered restframe row stays failing for a reason already understood; or state that E10 suffices and the fit rows are history.
2. **The fit-mode A12 row** moved from a 1.285 % failure (Γ_p 76.53, pre-weights base) to a 0.951 % pass (Γ_p 73.90) with the weights branch (§7.4 note). Fact from the logs; the mechanism (a change in the packet-weight stream of the test) is not analysed here. Recommendation: Codex 1 confirms the cause in the closure; the 1 % tolerance stays.
3. **The fit-mode `a7wall` entry fails by construction** (1585 of 2169 nodes, the fit-versus-Born offset at 1e6 samples), as its selected-point counterpart `a7w` did since step 4. Recommendation: keep both as labelled history entries (they document why the Born tables were needed), or mark them `WILL_FAIL` in ctest so that a full run reports them as expected failures; Nilay decides.
4. Codex 1's review of the scaling harness (`reviews/2026-10-30-codex1-perf-harness.md`, HOLD) arrived during this task; it concerns `dev/hadronic-perf`, untouched here as the brief requires, and is answered separately.
