# P2 step 3, pair plumbing with the mean-energy spectrum: debt, packets, two pair species, the three sources, the exact ledger. Report

Claude 1 (Claude Code cloud session), 2026-10-25. Fork `mancininilay/entity`,
branch `dev/hadronic-pairs` (created from `dev/hadronic` at `ed98e1f`).
Round-1 commits: `50b1f3d` (step-2 closure housekeeping), `9b39f3f` (core P9),
`e628d08` (kernel and generator), `71fd517` (core P5), `46cb34f` (tests),
plus the fix and report commits after it (ledger npart snapshots and the
41-column T8 check, the MPI-safe tracking payloads of the P9 test, the T13
degenerate-control convention, the report; round-1 HEAD `974be5f`).
Round-2 commits (section 0): `ec938aa` (R1-R6, spec v3.3 semantics, the
re-registered tests), `6dcf658` (`core(hadronic):` production placement
adapter), `d6ea582` (MPI-capable prototype, placement switch and tests),
and the report commit. Line numbers in sections 1-10 refer to the round-1
HEAD unless a round-2 line number is given explicitly in section 0; the
round-2 HEAD is named in the final message.

Every statement below is labelled as a **fact** (measured, with the number),
a **convention** adopted here, or an **open question**. No pre-registered
tolerance was adjusted; the one pre-registered item that fails is labelled as
an expected failure before it runs (section 4.5) and reported with its number
and mechanism. Implementation-chosen checks are marked `(impl)`; `(diag)`
lines are reported only.


## 0. Round 2: resolution log, re-registered results, MPI readiness, production placement

Everything in this section was measured with the round-2 HEAD (builds A-D
rebuilt from it; the kernel test `test-kernels-bh_pairs.xc` prints 109
passed, 0 failed, 0 expected failures with every group). The same labelling
as the rest of the report: **fact**, **convention**, **open question**;
`(impl)` and `(diag)` as before. No tolerance was adjusted; where a
criterion was re-registered the old criterion stays in the test as a
labelled history row, with its number.

### 0.1 Resolution log

| finding | status | what was done (round-2 line numbers) | test |
| --- | --- | --- | --- |
| R1 stale payload slots with unequal pair-species slot counts (blocking) | accepted, fixed | `write_lepton` (`bethe_heitler.hpp:528-595`) zeroes every slot `c < dst.pld_r.extent(1)` of **its own** destination before writing the three defined ones; `PairTargets::npld_r` removed (the two species may differ). The generator no longer requires equal counts (it still requires >= 3 on each). | `KERNELS::bh_pairs_r1` (`bh_pairs.cpp:1491`): slot counts 5/7 and 7/5, every slot of both species NaN-filled, the recycled tails dirtied with a sentinel before a 300-step run (264 pairs): no NaN on a live lepton, every slot beyond the three defined ones zero on both species, no sentinel survives on a live lepton, the three defined slots identical on both leptons: 8 checks pass |
| R2 Poisson draw converted to int before the cap, unregistered Gaussian branch (blocking) | accepted, fixed | `poisson_sample` (`bethe_heitler.hpp:387-440`): lambda formed in double from `R dt / pair_weight`; non-finite or > 1e9 returns `kmax + 1` without a draw (counted in `n_capped_events`); lambda < 10: sequential inversion, which stops and returns `kmax + 1` as soon as k would exceed the cap; lambda >= 10: PTRS (Hörmann 1993) with the accept-reject on `floor` in double, compared with the cap **in double** before the one conversion to int; the Gaussian branch is gone. The parser's lower bound on `pair_weight` stays `> 0` only (no further bound: the cap path is exact for any positive value, section 0.4). | `KERNELS::bh_pairs_r2` (`bh_pairs.cpp:1609`): section 0.4 (17 checks); `HADRONIC::r2_tiny_pair_weight` (engine, `pair_weight = 1e-20`); UBSan diagnostic 0.8 |
| R3 K0 trigger not paying the full debt when both triggers hold (blocking) | accepted, fixed | packet loop (`bethe_heitler.hpp:930-970`): an active K0 trigger has precedence and pays `r_packet = r` (the whole debt); otherwise the regular trigger pays `r_th`; K0 is kept until r returns to exactly 0 and reset only then (spec v3.3, decision (b)). The host replay of the tests (`replay_ion`, `bh_pairs.cpp:531`) carries the same rule. | `KERNELS::bh_pairs_r3` (section 0.3): the referee's state K0 = 10000, r = 100 >= r_th = 83.64, K = 9180.75, one small loss: r = 0 exactly, one K0 packet, K0 reset, w_pair = w r / <D> with the whole r; the control without the K0 trigger pays r_th and leaves r - r_th = 66.36 with K0 kept: 6 checks pass. T10 re-registered against the replay with the new rule (0.3) |
| R4 closing packet of an ion at rest (u = 0) | accepted, fixed (convention) | the closing packet of an ion with u = 0 is emitted along +x (`c1 = 1, c2 = c3 = 0`, `bethe_heitler.hpp:726-728`) and counted in the new ledger counter `n_closing_at_rest` (ledger column 39 of 43); documented in the kernel header and the ledger header. **Convention**: +x; a dedicated direction key was not added (the spec may choose otherwise; the counter makes every such event visible). | `bh_pairs_r3` R4 part: ion at rest with r = 10, w = 2: one closing packet, `n_closing_at_rest = 1`, r = K0 = 0, pair along +x with w_pair = w r / <D>(Gamma_min) = 5.32, identity with `debt_initial = w r` exact: 3 checks pass |
| R5 one hadronic ion species (decision (d)) | accepted, fixed | the generator refuses `hadronic.species` with more than one entry: "bh_proto: the prototype accepts exactly one hadronic ion species per run (decision (d))" (`pgen.hpp:586-596`) | `HADRONIC::reject` input `reject_two_species.toml` (two species named, refused, message checked): pass; `reject` now 27 inputs |
| R6 Gauss samples before/after the first births | accepted, fixed | `CustomPostStep` (`pgen.hpp:1333-1415`) evaluates the Gauss residual before the launch every step until the first birth; when a launch spawns, the before and after samples are written to `hadronic_gauss.txt` as the stages `before_first_births` / `after_first_births` (columns incl. `max|R - R0|` and the number of depositing particles) and the run aborts if the depositing count did not increase. At the production placement (0.6) the "before" sample is the one taken at the end of the previous post-step (the stage ran inside `step_forward`). The double build's zero-control fallback (section 4.8) stays documented as a convention. | `HADRONIC::t13_gauss` (`bh_check gauss` reads the two rows): fact from the registered run: first births at step 9, depositing particles 1024 -> 1088 (32 pairs), `max|R - R0| / max|c rho|` 2.3266e-7 -> 2.3266e-7 (unchanged to the printed 16 digits: the two leptons at one position cancel in rho and have deposited nothing yet); pass |
| R7 T11 (iv) prediction (decision (a)) | accepted, re-registered | section 0.2; `momentum_run` / `test_T11_momentum` (`bh_pairs.cpp:772-934`) | `KERNELS::bh_pairs_momentum`: 2.69e-3 at 1%: pass |
| R8 dead-ion harvest in `CustomPostStep` is non-MPI-only (decision (e)) | accepted, documented and guarded | report section 0.5 and the generator comments (`pgen.hpp:1194-1204`, the harvest block); under `MPI_ENABLED` with more than one domain, absorbing particle boundaries are refused at construction with a message naming the mechanism (`pgen.hpp:700-706`); production harvests at the production placement (the kernel's dead branch runs before `CommunicateParticles`, where the dead slots are still the absorbed ions') | construction-time refusal: code only (no MPI build runs here; 0.5) |
| decision (c) `E_stochastic` | accepted, implemented | new signed ledger entry `E_stochastic = E_drag - sum_events W D` for `poisson` (0 otherwise), `E_lost_nopairs` keeps the `none` meaning only; the identity `E_drag + debt_initial = (E_pairs - E_absorbed) + E_round + E_stochastic + debt_outstanding + E_exported + E_lost_nopairs` in the kernel header, `HostLedger::identity_residual`, `GlobalLedger::identity_residual`, the ledger-file header (43 columns: `E_stochastic` after `E_round`, `n_closing_at_rest` after `n_k0_trigger`) and `bh_check` | `bh_pairs_t10`: `E_stochastic = E_lost_nopairs = 0` for budget; `bh_pairs_poisson`: `E_lost_nopairs = 0` and the imbalance in `E_stochastic` (fact: 266.56 of E_drag 8970.70, 3.0%, expectation 0; 1/sqrt(1901) = 2.3% is the one-sigma scale); `bh_pairs_r2` kernel run at `pair_weight = 1e-20`: identity 3.1e-33 with `E_stochastic` carrying the imbalance; `bh_pairs_none`: `E_lost_nopairs = E_drag` unchanged |
| pairs per unit energy (decision (c)) | accepted, re-registered | section 0.2; `test_pairs_per_energy` (`bh_pairs.cpp:1216`) | `KERNELS::bh_pairs_energy` |
| 3.1 MPI-capable prototype (code only) | done as code; compile-only check | section 0.5 | `HADRONIC::t8_bit_identity` and `t10_poisson_determinism` re-run with the order-independent checksums (pass) |
| 3.2 production-stage adapter (the one core change) | done | section 0.6, commit `6dcf658` | `HADRONIC::t14b_placement` (new) |
| report: "6 packets per ion" in 4.5 (iv) | corrected | 24 packets per ion (192 over 8 ions; the `(diag)` line of the test prints "192 packets (24 per ion)") | |

Deferred (not in this round, with the reason): none of the referee's
items. Still deferred to Rusty as the brief requires: an MPI **run** (T14),
GPU (T15b), T-perf (section 7, updated).

### 0.2 Re-registered criteria (decision (a), decision (c))

**T11 (iv), momentum** (`KERNELS::bh_pairs_momentum`; 8 ions with the
physical mass mu_d = mu_r at Gamma_p = 6, weights 1, 1.5, 2 cycling, dK/K =
1e-4 per step, 2000 steps). The registered quantity is now the **paid**
momentum residual from the finite-loss history: for every ion the drag
momentum of each applied loss, `mu_d (u_before - u_after)` along the ion
direction, is partitioned FIFO into its paid packets (in spawn order; the
replay and the kernel agree on the packet set, T10) and the outstanding
debt; `R_hist = P_drag,paid - (P_pairs - P_absorbed)` is compared with the
amended III.1 prediction `R = sum_j W_j [D_j (1/beta_d,j - 1) + 2 (gamma_e,j
- sqrt(gamma_e,j^2 - 1))]`, `W_j` the stored pair weight, `D_j`, `beta_d,j`
(the numerical ion's velocity) and `gamma_e,j` at the spawn state. Facts:
192 packets (24 per ion), P_drag 16822.5258 = paid 16197.5976 + outstanding
624.9280; P_pairs 16458.4926, P_absorbed 994.9126; R_hist 734.0177,
R_formula 736.0006: |R_hist - R| / R = **2.694e-3 <= 1e-2, pass**. `(impl)`
the FIFO partition conserves the drag momentum (paid + outstanding = P_drag
to 1e-9): pass. History row `(diag)`: the massless formula `sum W D
(1/beta_d - 1)` = 261.4391, |R_hist - R| / R = 1.808 (would fail the 1%
criterion: the lepton rest mass is omitted; this is the round-1
EXPECTED-FAIL of 4.5 (iv), now a history row). Diagnostics: mu_d = 1 (beta_d
= 1 - 6e-9): R_hist 329.98594 vs 329.98594 (5.7e-9 relative; the residual
is the rest-mass term alone, the massless prediction being 7.8e-5);
low-energy run at Gamma_p = 2.5 (1.5 is below this bath's threshold 2.125):
R_hist 342.263 vs 343.835 (4.57e-3 relative, 64 packets, `D (1/beta_p - 1) /
D` = 0.0911). Mechanism of the remaining 2.7e-3 at Gamma_p = 6: the drag of a
packet is accrued over ~80 steps at slightly higher beta_d than the spawn
state's; the FIFO partition removes the outstanding-debt term exactly but
not this drift. Round-1 4.5 (iv) is superseded by this paragraph (its text
is kept below with the correction "24 packets per ion").

**Pairs per unit ion energy** (`KERNELS::bh_pairs_energy`). Re-registered
as two statements: (1) exact bookkeeping, `sum_s w_s = sum_j Q_j / <D>_j`
over the paid packets (`Q_j = w r_packet,j`, `<D>_j` at the spawn state,
from the replay) at 1e-6 relative; (2) the physical ratio `sum w_s / E_drag`
against `1/<D>(Gamma_p mid)` within the **stated budget** `f_unpaid +
delta_D + 1e-6`, with `f_unpaid = debt_outstanding / E_drag` and `delta_D`
the largest relative variation of `1/<D>` over the packets, both printed
before the verdict. Facts: nominal configuration (16 ions, dK/K = 1e-5, 1000
steps, r_th_factor = 10): 16 packets, sum w_s = sum Q/<D> = 220.000000
(0 relative), f_unpaid 0.07184, delta_D 0.002175, budget 0.074017, measured
|ratio - 1/<D>| / (1/<D>) = 6.982e-2: pass; round-1 configuration (dK/K =
1e-6, r_th_factor = 2e-4): 86832 packets, 7.8e-9 and 1.095e-4 within
3.646e-4: pass; identity, array residual / B (4.9e-7, 3.6e-7) and E_round =
residual pass on both. History row `(diag)`: the nominal shortcut `|sum w_s /
E_drag - 1/<D>| / (1/<D>)` = 6.982e-2 at the brief's 1e-3 would fail (the
budget here is 7.4e-2). Round-1 4.4 is superseded.

### 0.3 K0 semantics (spec v3.3) and the T10 re-registration

With the K0 precedence (R3) the T10 run of 4.1 is unchanged in its counts
(fact: 24 packets per ion of which 8 on the K0 trigger, 192 in total, the
replay bit-exact, the stored debt after 2000 steps bit-exact), because in
that run the two triggers never hold at the same step. The **capped** run
(dK/K = 2e-2 per step, `max_events_per_step = 1`, 30 steps) changes: the K0
trigger now flushes the whole debt when it fires, so the round-1 statement
"exactly one packet per ion per step" (240) is a history row; re-registered
against the replay with the cap: n_spawned 216 = replay 216, n_capped 32 =
replay 32, K0 flushes 72 = replay 72 (fact; `bh_pairs_t10`), identity exact,
array residual / B and E_round = residual pass. The R3 replay test
(`bh_pairs_r3`) replays the referee's state and the control without the K0
trigger (0.1). Convention 7 of 3.6 is superseded: the K0 trigger has
precedence, pays the whole debt, and K0 is reset only when r is exactly 0.

### 0.4 R2: the exact Poisson sampler

`KERNELS::bh_pairs_r2` (1e5 draws per lambda from a `random_number_pool_t`
through `poisson_sample`):

| lambda | mean | |mean - lambda| / sigma (<= 4) | variance | |var - lambda| / lambda (<= 10%) | max k | draws per sample |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 0.197860 | 1.513 | 0.196491 | 1.75e-2 | 4 | 1.000 |
| 5 | 4.998490 | 0.214 | 4.981148 | 3.77e-3 | 17 | 1.000 |
| 50 | 50.047830 | 2.139 | 50.046302 | 9.26e-4 | 87 | 2.401 |
| 500 | 499.984520 | 0.219 | 502.723680 | 5.45e-3 | 619 | 2.291 |

All 8 pass. Cap: lambda = 500 with `max_events_per_step = 8`: every draw <=
cap + 1 and the fraction reported as exceeding is 1.000 (pass); lambda = 5,
cap 8: the fraction exceeding equals P(X > 8) = 0.0681 within 4 sigma (0.87
sigma; the capped draws' mean 4.66 is the truncated Poisson's) (pass);
lambda = inf, NaN and 1.1e19 (`pair_weight = 1e-20` at the test's rate) are
reported as exceeding without a draw (pass); the kernel at `pair_weight =
1e-20` (16 ions, 20 steps): every ion-step exceeds the cap (counted) and
spawns exactly the cap, 320 pairs, identity 3.1e-33 (pass). Engine level
(`HADRONIC::r2_tiny_pair_weight`, `r2_pairweight_tiny.toml`: 16 ions at
Gamma_p = 6, 200 steps, `pair_weight = 1e-20`, `max_events_per_step = 8`,
`on_max_events = warn`): n_events = n_spawned = 25600 = 16 x 200 x 8,
n_capped_events = 3200 = 16 x 200, npart 25600 / 25600 on both species,
n_rng = 0 (no draw on a capped step, none for the delta spectrum), identity
0, E_stochastic = E_drag = 23046.75 (the pairs carry 2.2e-15): pass (5
checks). Effect on the `poisson` run of 4.2 (fact): with the new sampler the
realisation changes, n_spawned 1901 vs 2000 +- 44.7 (2.21 sigma, <= 4),
variance / mean within 10%, n_rng = 10000 (one uniform per ion-step for
lambda = 0.2; Knuth drew 1.195 per ion-step): pass. UBSan: 0.8.

### 0.5 MPI-capable prototype (brief 3.1; code only, no MPI run)

Done in the generator (`pgen.hpp`, commit `d6ea582`):

- the refusal of `domain.index() != 0` for periodic boxes is removed; the
  per-domain ledgers are accumulated over the local domains of a step
  (`HostLedger::accumulate`, snapshots summed over the local domains and
  replaced at the next step);
- `reduce_ledger` (`pgen.hpp:1446-1514`): under `MPI_ENABLED` with more
  than one rank every `dd` entry is reduced as its `(hi, lo)` pair of
  doubles with `MPI_Allreduce(MPI_SUM)` and renormalised, the momenta and
  `sum_wK` as doubles, the counters as `MPI_UNSIGNED_LONG_LONG`;
  `debt_initial`, the host export sums and `n_dead_host` are reduced the
  same way (`GlobalLedger`, `pgen.hpp:331`); rank 0 writes the ledger file
  from the reduced ledger (`write_ledger`), and `finish` prints the global
  closure, array identity and exported lines from it, the pair-side array
  sums and the array `sum w K` also reduced;
- the T8 checksums are order-independent: the wraparound sum over the
  particles of splitmix64 of a per-particle hash of (cell indices,
  displacements, momenta, weight, tag, payload row) (`checksum_species`,
  `pgen.hpp:1516-1560`), and the same per cell for the fields
  (`bhproto::checksum_fields`); the per-rank values are summed over the
  ranks (`mpi_sum_u64`) and printed by rank 0 with the global `npart`. Fact:
  T8 (three runs on the Serial build) and T10 determinism pass with the new
  checksums; a consequence worth stating: on the OpenMP build the budget run
  now gives identical checksums between two runs that differ only in the
  spawn order (0.6), where the order-dependent hash of round 1 differed;
- per-ion, pair and Gauss files are written per rank with `_rank<r>` in the
  name when the rank count exceeds 1 (`per_rank_name`); the ledger file is
  written by rank 0 only;
- the dead-ion harvest (R8, decision (e)): the host sums over dead-tagged
  ions taken in `HadronicStage` before each launch and the kernel's dead
  branch are valid at the post-step placement only without MPI: under MPI
  `CommunicateParticles` receives the incoming particles into the dead
  slots before the post-step hook runs (fact:
  `src/framework/containers/comm/particles.cpp:226, 370-390`: the
  `outgoing_indices` array lists the `npart - npart_alive` slots of dead
  and send-tagged particles and `ExtractReceivedPrtls_kernel` writes the
  received particles into those holes first, then appends the rest), so the
  hook would see refilled slots. Convention: with
  `MPI_ENABLED` and more than one domain, absorbing particle boundaries are
  refused at construction (message naming the mechanism); the production
  placement (0.6) runs the stage before the exchange and harvests the
  absorbed ions there;
- the startup line prints the placement and the MPI configuration (ranks,
  domains, whether the ledger is reduced over the ranks).

Compile-only check (fact): OpenMPI 4.1.6 installed in the container
(`apt`, `mpicxx` = g++ 13.3.0); `build-mpi` configured with `-D mpi=ON -D
output=OFF -D TESTS=OFF -D precision=single -D pgen=bh_proto -D
CMAKE_CXX_COMPILER=mpicxx -D CMAKE_C_COMPILER=mpicc` against the OpenMP
Kokkos; `ninja -C build-mpi entity.xc` of the round-2 HEAD: compiles and links (exit 0, no warnings; `MPI_ENABLED` defined, so the `MPI_Allreduce` paths of `reduce_ledger`, `mpi_sum_u64` and the pair-side reductions in `finish` are compiled, `logs/build-mpi-r2-final.log`).
No MPI run was made (the brief excludes it); the reductions, the per-rank
file names and the construction-time refusal are untested at run time and
are listed as such in section 9.

### 0.6 Production-stage adapter (brief 3.2; the one core change, commit `6dcf658`)

Core: `src/global/traits/pgen.h` gains the concept
`traits::pgen::HasHadronicStage<PG, DOM>` (the generator exposes
`HadronicStage(step, time, domain)` and `hadronic_placement_production()`);
`src/engines/srpic/hadronic.h` (new) defines
`srpic::HadronicInteractions(domain, pgen, step, time)`, which launches the
generator's stage on one local domain; `src/engines/srpic/srpic.hpp` calls
it after the deposit and current-filter block and before
`CommunicateParticles`, outside the `deposit_enabled` conditional, only when
the trait holds and the runtime switch is on, under the existing `Custom`
timer. Nothing else in `src/` changed in round 2. Generator side: the stage
is split out of `CustomPostStep` as `HadronicStage` (`pgen.hpp:1156`);
`hadronic.placement = "poststep" | "production"` (default `poststep`,
validated by the parser and refused otherwise, `reject_placement.toml`);
at `production` the post-step hook skips the launch and keeps the output,
the Gauss bracket (0.1 R6) and `finish`.

Test `HADRONIC::t14b_placement` (`t14b_placement.sh`, Bash 3.2): the
production inputs are derived from the committed inputs by inserting the one
key under `[setup]` (the driver checks that the derived file differs by that
line only). Facts (OpenMP build A, one rank, periodic `pairs_smoke.toml`:
4 ions, dK/K = 1e-4 per step, 200 steps, 244 pairs): the two ledger files
(4 rows x 43 columns) are identical to the last printed digit (largest
relative difference 0, every counter equal), `E_pairs` 278.38291094841986
in both, identity 0; `(diag)` the order-independent checksums of the ions,
both pair species and the fields are identical between the placements on
the OpenMP build too. T-exp at the production placement (`texp_absorb.toml`,
absorbing walls): the dead-ion harvest works there (E_exported_host =
E_exported = 249.3964991569519, difference 0; pending_discarded -5.008e-4,
difference 0; 15 dead ions with a payload seen; identity 0) and the 60 ledger
rows are identical to the post-step run's (largest difference 0, 82 pairs in
both). Serial build C: the budget run and the poisson run
(`t10_poisson_det_a.toml`, 530 pairs) give bit-identical checksum lines
(ions, both pair species, fields) and byte-identical pair files between the
placements, and identical ledgers (largest difference 0; poisson identity
1.4e-34). 8 runs, every check passed. Why the agreement is exact (fact, by
construction): on one periodic domain the stage sees the same post-push ion
state at either placement, the loss and the pair momenta do not depend on
the position, and `CommunicateParticles` after the production stage wraps a
pair born in a ghost cell to the same `(i, dx)` the parent has at the
post-step placement; the only difference is the spawn order, which the
order-independent checksums ignore.

### 0.7 Suites with the round-2 HEAD

{{SUITES}}

### 0.8 UBSan and bounds-check diagnostics (R2, P8)

Build `build-ubsan` (fact): the single-precision OpenMP configuration with
`-fsanitize=undefined -fno-sanitize-recover=undefined -g` (gcc 13.3.0,
`output=OFF`), `entity.xc`, `test-kernels-bh_pairs.xc` and `bh_check` built
(exit 0). Runs under it (`logs/ubsan-*.log`): `test-kernels-bh_pairs.xc r2
r3 r1` 30 passed, 0 failed, exit 0, no `runtime error` line (the sampler at
lambda = 0.2, 5, 50, 500, the cap at lambda = 500 and 5, lambda = inf, NaN
and 1.1e19, the kernel at `pair_weight = 1e-20`, the K0 and R4 replays and
the unequal-slot runs); `r2_tiny.sh` with the UBSan `entity.xc`
(`pair_weight = 1e-20`, 25600 pairs, 3200 capped ion-steps): every check
passed, exit 0, no `runtime error` line in the run log. With
`-fno-sanitize-recover` any undefined behaviour (in particular a
float-to-int conversion out of range, the round-1 defect) would have
aborted the run. Diagnostic, not registered in ctest.

Bounds-check build D rebuilt from the round-2 HEAD (fact, `logs/bounds-*`,
not registered in ctest): `test-kernels-bh_pairs.xc t12 t10b texp r1 r2 r3`
49 passed, 0 failed, exit 0, no bounds error (the R1 unequal-slot runs and
the R2 cap runs included); engine inputs `t12_capacity_warn.toml` (n_spawned
60 at the full capacity, exit 0), `t13_gauss.toml` (686 pairs, exit 0),
`r2_pairweight_tiny.toml` (25600 pairs at the cap every step, exit 0) and
`texp_absorb.toml` at the production placement (82 pairs, E_exported_host =
E_exported = 249.3964991569519, exit 0): no bounds error in any run.

### 0.9 T13 replicate distribution (fact, honest reporting)

`HADRONIC::t13_gauss` on the OpenMP build compares the maximum over 20 rows
of a float-rounding residual of one realisation with that of another
realisation (the control run); the ratio is a random variable of order 1.
In the round-2 driver runs the ratio was 2.057 once (fail at <= 2; the
run of `ctest-r2-drivers`), then 0.871, 0.756, 1.122, 0.827, 0.867, 0.468,
1.222, 0.604 in eight replicates (all pass), and {{T13_FINAL}} in the
registered final suite. Mechanism: with OpenMP the per-thread order of the
atomic current deposit changes the float round-off of `J` and hence of `E`
from run to run, so the pair run's and the control run's maxima are
independent draws from a heavy-tailed (max-over-rows) distribution; a
factor-2 criterion on their ratio fails with a probability of order 10%
per run on this build. No tolerance was changed; the criterion is the
brief's. The Serial and double builds do not have this noise (the double
control residual is exactly zero, section 4.8). Suggested for the spec
(open question): compare each run with its own a-priori level n_steps x
eps(real_t) (the double build's convention of 4.8) on every build, which
is deterministic.

### 0.10 Conventions added or superseded in round 2

- superseded: 3.6 item 3 (poisson identity through `E_lost_nopairs`) by
  decision (c) (`E_stochastic`); 3.6 item 7 (K0 reset) by spec v3.3 (0.3);
  4.4 and 4.5 (iv) by the re-registrations of 0.2; 3.6 item 4 (one pair of
  species for all ion species) is moot under decision (d) (one ion species).
- new: +x for the closing packet of an ion at rest (R4); the production
  placement's "before" Gauss sample is the previous post-step's (R6);
  absorbing boundaries refused under MPI with more than one domain (R8);
  per-rank file names `_rank<r>`; order-independent checksums (the T8
  statement is now "identical multisets of particle records", not
  "identical arrays").
- open questions 1-4 of section 8 are closed by decisions (a), (c), (c) and
  (b); question 5 (the brief's count formula) is answered by the replay; 6
  is moot (decision (d)); 7 and 8 stay open.

## 1. Environment and builds

Facts, measured in the cloud container (gcc 13.3.0, cmake 4.4.4, ninja
1.13.2, 4 cores):

| build | configuration |
| --- | --- |
| A (`build-single`) | `-D TESTS=ON -D output=ON -D precision=single -D mpi=OFF -D pgen=bh_proto`, Kokkos OpenMP+Serial (`$HOME/.entity/kokkos`, 5.2.1), with `-D BH_DOUBLE_ENTITY=<B>/src/entity.xc -D BH_SERIAL_ENTITY=<C>/src/entity.xc` |
| B (`build-double`) | `-D precision=double -D TESTS=ON`, same Kokkos |
| C (`build-serial`) | `-D precision=single -D TESTS=OFF` against a Serial-only Kokkos built here from the same source into `$HOME/.entity/kokkos-serial` (`KOKKOS_ENABLE_SERIAL` defined, `KOKKOS_ENABLE_OPENMP` undefined) |
| D (`build-bounds`) | `-D precision=single -D TESTS=ON` against a Kokkos built here with `-D Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON -D Kokkos_ENABLE_DEBUG=ON` into `$HOME/.entity/kokkos-debug` (OpenMP+Serial); only `entity.xc`, `bh_check` and the `bh_pairs` unit test are built there (diagnostic, section 6) |

Logs of every build and run are under the session scratchpad
(`logs/*.log`, `runs/*`); the ctest work directories of the run-based tests
are under `build-*/tests/hadronic/work/`.

## 2. Housekeeping from the step-2 closure (commit `50b1f3d`)

- `bethe_heitler.hpp`: `HadronicParams::f_max` initialised to 0.01.
- `tests/hadronic/reject.sh` and `t7b.sh` rewritten without `declare -A`
  (indexed arrays and a `case` function instead); `reject.sh` counts the
  inputs it executed and fails when none ran; every driver under
  `tests/hadronic/` ends with an explicit summary line. Fact: `grep -c
  'declare -A\|\${[a-zA-Z_]*,,}\|mapfile' tests/hadronic/*.sh` finds
  nothing; `bash -n` accepts every driver.
- Step-2 report, section 10.1 row R7: six positive cases and 52 refusals (58
  checks).

## 3. What was implemented and where

### 3.1 How the prototype hook sees dead ions (fact, verified in the tree)

`Engine::run` (`src/engines/engine.hpp:316-348`) calls `step_forward`, then
`CustomPostStep`, then `SortParticles`. The SRPIC `step_forward`
(`src/engines/srpic/srpic.hpp:106-131`) pushes, deposits and communicates;
the pusher tags a particle leaving through an absorbing boundary as
`ParticleTag::dead` inside the push kernel (`src/kernels/pushers/sr.hpp:676,
687, 718, 732, 766, 777`). `RemoveDead` runs only inside `SortParticles`
(`src/framework/domain/metadomain_sort.cpp:20-25`), after the hook, and only
when the species' `clear_interval` is non-zero (the generator requires 0 on
every hadronic and pair species). So an ion absorbed in step n reaches the
kernel in the same step with `tag == dead`, its payloads intact, before any
compaction: the `dead` branch of II.2 (`E_exported += w r`,
`pending_discarded += w delta`, payloads zeroed) is exercised by T-exp both
at the kernel level and end to end (section 4.9), where the generator's own
host sum over the dead-tagged ions, taken before each launch, equals the
kernel's `E_exported` to the last bit.

### 3.2 Kernel: `src/kernels/hadronic/bethe_heitler.hpp` (957 lines)

| lines | content |
| --- | --- |
| 94-105 | `PairSource {None, Budget, Poisson}`, `PairSpectrum {Delta, Threshold}` |
| 112-190 | `dd`: compensated double-double accumulator (two-sum, two-product via `Kokkos::fma`), `add`, `sub`, `add_prod`, `prod`, `value` |
| 192-227 | `HadronicParams`: the step-2 scalars plus `source, spectrum, r_th_factor, f_r, t_latency (0 = off), pair_weight, max_events_per_step, kick (T13)` |
| 229-249 | `PairTargets`: the two destination `ParticleArrays`, offsets, host capacity, counters, domain index, tracking flags, `npld_r`, the device spawn counter, `valid` |
| 251-333 | `LossLedger`: every entry of II.2 step 6 (energies as `dd`), `P_*[3]` in double, the snapshots, the counters incl. `n_capped_events`, `n_rng`, `n_k0_trigger`, and `B_budget` (the rounding budget of T11); `operator+=` |
| 335-349 | `kinetic_energy`, `lepton_gamma` (the two expressions shared with the host closures) |
| 434-450 | `outcome_law(Gamma_p)`: `<D>`, `eps_mean`, the two lepton Lorentz factors (`delta`: (<D> + eps)/2 each; `threshold`: Gamma_p each, D = 2 Gamma_p - eps) |
| 452-516 | `write_lepton`: `InjectParticle` (tracking or not), `*_prev = *`, the parent's tag, every real payload slot written (birth time, parent Gamma_p, eps; the rest 0) |
| 518-591 | `spawn_pair`: atomic ticket against the host capacity, both leptons at `offset + ticket`, ledger entries from the stored values (`E_pairs`, `E_absorbed` with the stored eps payload, `P_pairs`, `P_absorbed`, `B_budget`), `E_round += intended - w_s (E_s - eps_s)`; on capacity failure `n_dropped` and `false` |
| 593-617 | `poisson_draw`: Knuth for lambda <= 30 (normal approximation above), every draw counted in `n_rng` |
| 619-643 | tag classes: `dead` exports the debt and discards the request |
| 644-695 | step 0; below the edge: discard the request, pay the debt in one closing packet with the law at Gamma_min (`n_closing`) |
| 696-766 | step 1 (unchanged from step 2) |
| 767-837 | step 2 with `delta_th = max(floor, min(eta_u K, r_th))` for the budget source, `r += dK_applied` in `dd`, `K0 <- K_stored` when r was 0, `n_latency` when the floor exceeds r_th |
| 838-915 | steps 3-4, budget: `r_th = min(r_th_factor <D>, f_r K, dK t_latency / dt)`, the packet loop (regular trigger, K0 trigger, `max_events_per_step` cap -> `n_capped`), debt stored as `real_t` with its rounding in `E_round`, `K0 = 0` when r = 0 |
| 916-947 | poisson: n ~ Poisson(coeff J dt / pair_weight), capped (`n_capped_events`), `w_pair = pair_weight w`, `E_lost_nopairs -= w_pair D` per spawned pair |

### 3.3 Device tables and host references

`src/archetypes/hadronic/bh_tables_device.h`: `log_dmean` node array (68),
`Dmean(Gamma)` (161-179, log-linear in y of the node ratios, equal to the
ratio of the interpolated I and J to double rounding), `EpsMean(Gamma)`
(185-202, linear in y); edge conventions for the outcome law: the first node
value below the lower edge (closing packets), the last node value at and
above the upper edge. `src/archetypes/hadronic/bh_interpolate.h`:
`interpolate_outcome_host` (62) and `interpolate_rate_host` (105), the
Kokkos-free references used by the tests, the generator's estimate and the
check tool.

### 3.4 Generator: `pgens/bh_proto/pgen.hpp` (1661 lines), `pgens/bh_proto/bh_proto_params.h` (792 lines)

| lines (pgen.hpp) | content |
| --- | --- |
| 327-373 | `HostLedger`: cumulative `LossLedger`, `debt_initial` (dd), host sums over dead ions, `identity_residual()` in dd |
| 375-386 | `hadronic_seed`: splitmix64 chained over (seed, species, domain, start step), each component xor-ed with a distinct tag before the next hash |
| 490-506 | box: `L_min` from the global mesh extent, `any_absorbing` from `prtl_bc()` |
| 568-612 | pair-species validation: defined, mass 1, charges -1 / +1 in the listed order, >= 3 real payload slots, no clearing or spatial sorting, tracking payload present |
| 616-628 | `t_latency`: auto = 0.1 L_min / c; 0 refused when any particle boundary is absorbing |
| 631-640 | `photon_field.eps_max < 2` required with the budget source (bath with a cutoff) |
| 741-763 | `spawn_rate_at(Gamma)`: budget coeff I / r_th with the actual minimum of the three terms, poisson coeff J / pair_weight |
| 765-822 | `capacity_estimate()` (III.8): gamma_p_typical or the power-law envelope (401-point trapezoid in ln Gamma), bare and sized (c_layer, comm_headroom) expectations, printed, `on_capacity_estimate` |
| 824-1013 | `InitPrtls`: NaN fill (T10b), ion injection through the P9 overload (the explicit zeroing pass of step 2 is kept), background pair plasma through `arch::InjectNonUniform` (917) and `arch::InjectGlobally` (954), `debt_initial` from the arrays |
| 1046-1229 | `CustomPostStep`: host sums over dead ions before the launch (T-exp), lazy pool construction, `PairTargets` with `capacity = min(maxnpart - npart)` and a zeroed spawn counter, the launch, `n_injected = min(tickets, capacity)`, `set_npart`, `set_counter`, `set_unsorted` for both species, `on_capacity` and `on_max_events` policies (error: ledger row written, then `raise::Error` naming the step and the counts; warn: message and count) |
| 1231-1283 | `write_ledger`: 41 columns with header and units, the identity residual as a column |
| 1285-1335 | `write_ions` (+ debt and K0 columns), 1337-1383 `write_pairs` |
| 1385-1468 | `write_gauss` (T13, 1D): CIC node charge of the depositing charged species in double, `R_i = (E_{i+1/2} - E_{i-1/2}) - (q0/(B0 V0)) rho_i`, reported relative to its first value R0 |
| 1470-1640 | `finish`: pair-side sums from the arrays in dd, checksums of ions and pairs, the T11 array identity with B, the full ledger line, the exported-debt line, the NaN payload scan |

`bh_proto_params.h`: the step-2 reader plus the keys of part IV listed in its
header (source, spectrum, pair_species, r_th_factor, f_r, t_latency,
pair_weight, max_events_per_step, on_*, gamma_p_typical, active_fraction,
envelope.*, c_layer, comm_headroom) and the test hooks (test.nan_payloads,
test.kick, test.gauss, background.*), every integer validated on the 64-bit
value before any cast (669-791).

### 3.5 Core changes (each its own commit, each with its own test)

- **P9** (`9b39f3f`, `src/kernels/injectors.hpp:83-125`): overload of
  `kernel::InjectParticle` taking the `pld_r` view and zeroing every slot;
  the three stock injector kernels carry `pldrs_*` views and call it. Test
  `tests/kernels/injectors_payloads.cpp` (`KERNELS::injectors_payloads`,
  30 checks): NaN-filled arrays, injection through the overload (tracking,
  not tracking, `npld_r = 0` with the unallocated view) and through
  `UniformInjector_kernel`, `NonUniformInjector_kernel`,
  `GlobalInjector_kernel` (one-domain Metadomain); every injected slot zero,
  every untouched tail slot still NaN.
- **P5** (`71fd517`, `src/kernels/reduced_stats.hpp:458-462, 526-537`):
  weight times mass (Rho) or signed charge (Charge); weight in T; `p^i = m
  u^i` in T0i and Tij (photons unchanged). Test
  `tests/kernels/reduced_stats_weights.cpp` (`KERNELS::reduced_stats_weights`,
  section 4.11).
- Nothing else in `src/` outside `src/kernels/hadronic/` and
  `src/archetypes/hadronic/`; `cmake/tests.cmake`, the parser, the schema,
  `srpic.hpp` are untouched.

### 3.6 Conventions adopted (each also an open question for the spec where marked)

1. **Ledger energies in double-double.** The rule "double for all
   bookkeeping, the ledger by a double reducer" is met with a compensated
   accumulator (two doubles per entry) so that the identity closes to the
   accumulator's round-off (~1e-30 relative) independently of the reduction
   order; a plain-double reduction over ~1e4 transfers of ~70 m_e c^2 has a
   round-off of ~1e-10 absolute, which would not meet the T11 (iii) criterion
   (E_round equals the array residual to 1e-9 relative, with E_round itself
   ~4e-4 for a 2000-step run). The host closures use the same `dd` type.
2. **E_round includes the storage rounding of the debt.** Besides the
   rounding of each transfer (w r_packet - w_s (E_s - eps_s)), `E_round`
   receives w (r_exact - r_stored) each time the debt payload is written as
   `real_t`; without it the identity would carry the float rounding of r
   (~1e-7 relative per step) and could not be exact. The absorbed energy is
   accounted with the stored eps payload (`E_absorbed = sum w_s eps_s`), so
   the array identity closes with the payload for every bath.
3. **Poisson and the identity (open question; superseded in round 2 by decision (c), section 0.1: the imbalance is the new entry `E_stochastic`).** The brief asks the counter
   identity to be exact for `poisson` too; with independent draws the drag
   and the pair energy match only in expectation. Convention: for `poisson`,
   `E_lost_nopairs` accumulates the signed imbalance w dK_applied - sum_events
   w_pair D (expectation zero), and `E_round` the storage rounding of the
   created pairs, so that the identity holds by construction; the physics is
   tested by the pairs-per-unit-energy check (4 sigma).
4. **One pair of species for all hadronic ion species.** `pair_species` is a
   single key; every species in `hadronic.species` feeds the same two species
   (the tests use one ion species).
5. **`n_events` = spawn attempts, `n_spawned` = successful insertions,
   `n_dropped` = refusals**, so `n_events = n_spawned + n_dropped`.
6. **Pair birth time = the time after the step** ((step + 1) dt, the same
   stamp as the ledger rows; "time + dt" of the spec with time = t^n).
7. **(superseded by spec v3.3, section 0.3: K0 precedence, reset only at r = 0.)** **The K0 trigger pays the whole debt** (r_packet = r), as III.4; after it
   r = 0 and K0 is reset. Because a regular packet leaves r - r_th > 0, K0 is
   reset only on the K0 trigger or when r is exactly 0; at dK/K = 1e-4 per
   step with f_r = 0.02 the K0 trigger fires every ~200 steps (section 4.1).
8. **Startup estimate.** The bare expectation (no margins) is what the
   measured count is compared with in T12b; the sized expectation (times
   c_layer (1 + comm_headroom)) is what `on_capacity_estimate` enforces.
9. **`InjectGlobally` with weights.** The weighted path of the stock
   `GlobalInjector_kernel` reads the destination array instead of the input
   weights (an upstream bug listed under P10 in the spec); P10 is out of
   scope, so the generator's `background.n_global` path injects with weight 1
   and the bug is left to the P10 commit.
10. **Gauss residual relative to R0.** A randomly loaded background plasma
    does not satisfy the discrete Gauss law at t = 0 (E = 0, rho != 0);
    charge conservation means R_i(t) - R_i(0) stays at round-off, which is
    what T13 measures (section 4.8).
11. **T11 (iv) at the physical mass.** The momentum residual's prediction
    involves beta_p of the ion; with mu_d = 1 (the convention of the other
    runs) the ion is ultra-relativistic and P_drag = E_drag, so the momentum
    run uses mu_d = mu_r.

## 4. Tests: measured values, tolerances, verdicts

Conventions stated before the runs (header of `tests/kernels/bh_pairs.cpp`):
bath mono eps0 = 0.5, mu_r = 1836.15, dt = 0.5, ions with mu_d = 1 at Gamma_p
= 6 (K = 9180.75) unless stated, the latency cap off in the kernel tests
(periodic box), r_th_factor = 10 and f_r = 0.02 so that r_th = 10 <D> = 83.64
binds (f_r K = 183.6); "exact" = 1e-12 relative to E_drag. At Gamma_p = 6
(facts, printed by the test): I = 3.2617455651, J = 0.38996804790, <D> =
8.364135428, eps_mean = 0.5. Every number below is from
`build-single/tests/kernels/test-kernels-bh_pairs.xc` (all groups, 69
passed, 0 failed, 1 expected failure) and from the verbose ctest run of the
drivers (`logs/ctest-single-drivers-verbose.log`).

### 4.1 T10, spawn statistics, `budget` + `delta` (`KERNELS::bh_pairs_t10`)

8 ions (weights 1 and 2.5 alternating), dK/K = 1e-4 per step, 2000 steps.
Facts: 192 packets in total, 24 per ion of which 8 on the K0 trigger; sum
dK_applied = 1438.70 per ion, r_final = 53.351074 (stored), Gamma_p 6 ->
5.2165, r_th 83.64 -> 76.65 over the run.

| check | criterion | measured | verdict |
| --- | --- | --- | --- |
| packets per ion = host replay of the debt mechanism (every ion) | exact | 24 = 24 for all 8 ions (192) | pass |
| brief's constant-threshold formula floor(sum dK / r_th(Gamma_p = 6)) | reported | 17 per ion (r_th drifts by 8% and the K0 trigger adds 8 packets per ion) | (diag) |
| every w_pair = w r_packet / <D>(Gamma_p at spawn) | 1e-6 | 5.06e-8 (max) | pass |
| pair payload Gamma_p = parent's Gamma_p at spawn | 1e-6 | 4.01e-8 | pass |
| pair payload birth time = time after the spawning step | exact | 0 | pass |
| stored debt after 2000 steps = replay | bit-exact | 0 | pass |
| pair (i, dx) = parent's, both leptons; *_prev = *; eps payload 0.5; tags alive; equal weights, collinear equal momenta | exact | all 0 | pass (5 checks) |
| both species' npart = n_spawned | exact | 192 = 192 | pass |
| hadronic RNG counter unchanged | exact | n_rng = 0 | pass |
| no drops, no capping at max_events_per_step = 8 | exact | 0 | pass |

Mechanism of the K0 packets (fact): a regular packet leaves r - r_th > 0, so
K0 is not reset; after ~200 steps |K - K0| exceeds f_r K0 = 2% and the whole
debt is paid in one packet (r = 0, K0 reset), hence 8 such packets per ion.

Capped loop (dK/K = 2e-2 per step, max_events_per_step = 1, 30 steps; a
100-step run would fall below the table edge after 74 steps): n_capped = 240
(every ion-step), exactly one packet per ion per step (240), the debt grows
to 6320.23, the T11 identity exact (0, 1e-12), array residual / B = 1.34e-3
(<= 3), E_round = array residual (0, 1e-9): pass (6 checks).

Closing packets (6 ions at Gamma_p = 2.2 accumulating debt for 20 steps with
r_th out of reach, then decelerated on the host to Gamma_p = 1.565 below
Gamma_min = 2.13): no packet before (sum w r = 382.10), n_closing = 6, one
packet per ion, r = K0 = 0 after, w_pair = w r / <D>(Gamma_min) to 4.7e-8,
payload Gamma_p = the below-edge Gamma_p to 4.1e-8, identity with debt_initial
= sum w r exact (0): pass (7 checks).

### 4.2 T10, `poisson` + `delta` (`KERNELS::bh_pairs_poisson`)

100 ions x 100 steps at R dt / pair_weight = 0.2 (pair_weight = 0.548816):
n_spawned 1950 vs 2000 +- 44.7 (1.12 sigma, <= 4), per-ion-step variance
0.1908 vs mean 0.1950 (2.2%, <= 10%), n_rng = 11950 (> 0), pairs per unit
ion energy sum w_pair / E_drag vs 1/<D>: 0.096 sigma (<= 4), T11 identity
1.5e-33 relative, array residual / B = 3.0e-6, E_round = array residual (0):
pass (7 checks). A different seed gives a different realisation (pass).
(diag) The same seed on OpenMP gave an identical spawn set in this run; bit
identity is registered on the Serial backend (4.8). (diag) E_lost_nopairs,
the signed drag-minus-pairs imbalance, is 42.6 of E_drag 8970.7 (0.47%,
expectation 0; 1/sqrt(1950) = 2.3% is the one-sigma scale).

### 4.3 T10, `none` (`KERNELS::bh_pairs_none`)

500 steps: no spawns, E_lost_nopairs = E_drag exactly, n_rng = 0, identity
exact: pass (5 checks).

### 4.4 Pairs per unit ion energy, `budget` (`KERNELS::bh_pairs_energy`)

**Round-1 text, superseded by section 0.2** (re-registered with the stated
budget; the 1e-3 shortcut is kept as a history row).

Registered configuration: 16 ions, dK/K = 1e-6 per step, 1000 steps (K
changes by 0.0989%), r_th_factor = 2e-4 so that the outstanding debt is
1.1e-4 of E_drag (86832 packets). sum w_pair / E_drag vs 1/<D>(Gamma_p mid):
1.10e-4 relative (<= 1e-3), pass; (impl) against E_drag - debt_outstanding:
2.9e-6, pass; identity exact, array residual / B 3.6e-7, E_round = residual:
pass. (diag) The brief's nominal configuration (dK/K = 1e-5, r_th_factor =
10, Gamma_p changing by 0.98%) gives 7.0e-2 against E_drag (the outstanding
debt, up to one r_th per ion, is 7.2% of E_drag) and 2.2e-3 against the paid
energy (the drift of <D> between the spawn Gamma_p and the mid-run
Gamma_p); at the brief's tolerance the statement holds only when the debt
fraction and the <D> drift are both below 1e-3, which fixes the registered
configuration above (convention).

### 4.5 T11, ledger closure (`bh_pairs_t10`, `_poisson`, `_energy`, `_momentum`, `_t12`, `_texp`)

(i) The counter identity (host, dd): 0 relative on every budget run, 1.5e-33
(poisson): pass on all 7 runs. (ii) The array identity residual against the
printed rounding budget B: 1.4e-5 (budget 2000 steps, B = 28.05), 1.3e-3
(capped), 3.0e-6 (poisson), 3.6e-7 (per-energy run), 3.4e-6 (physical mass),
2.6e-5 (T12 with drops), 1.5e-5 (T-exp), all <= 3: pass. (iii) E_round equals
the array residual: |E_round - residual| / |E_round| = 0 on every run
(E_round = 3.99e-4, 5.71e-4, -3.2e-5, -2e-6, 7.8e-5, 4.12e-4, -3.0e-5):
pass. Facts: with the energies in dd both sides are exact sums of the same
doubles, so the agreement is to the last bit; |residual| / B is 1e-5 or less
because B bounds every transfer's rounding by its half-ulp while the actual
roundings have random signs.

(iv) Momentum, **round-1 text, superseded by section 0.2** (re-registered
against the amended prediction, which passes at 1%). (`KERNELS::bh_pairs_momentum`, 8 ions with mu_d = mu_r at
Gamma_p = 6, 2000 steps, 24 packets per ion (192; round 1 wrote "6" here, corrected in round 2)... facts: P_drag 16822.53, P_pairs
16458.49, P_absorbed 994.91, E_drag 16551.54). Labelled before the run:
the III.1 prediction sum w D (1/beta_p - 1) treats the leptons as massless;
with the delta spectrum gamma_e = 4.432 and the pair momentum deficit
2 (gamma_e - p_e) = 0.2286 per interaction exceeds D (1/beta_p - 1) = 0.1186,
so the measured residual was expected to be 2.9 x the prediction and the 1%
criterion to fail. Measured: P_drag - (P_pairs - P_absorbed) = 1358.95, of
which 624.99 is the drag momentum of the still outstanding debt; the paid
part 733.95 against the III.1 prediction 261.44 is 2.81 x (|measured -
predicted| / predicted = 4.20 on the full residual): **EXPECTED-FAIL** at 1e-2,
reported as such; no tolerance adjusted. (diag) Against the kinematic
prediction with the lepton rest mass, sum w [D/beta_p(spawn) - (2
sqrt(gamma_e^2 - 1) - eps)] = 736.00, the paid residual agrees to 2.8e-3
(the drag of a packet is accrued at earlier, higher beta_p than the spawn
state, so this is an approximation; the exact bookkeeping is the ledger's).
(diag) Low-energy run at Gamma_p = 2.5 (Gamma_p = 1.5 of the brief is below
the threshold 2.125 of this bath): residual / E_drag = 0.296 with
D (1/beta_p - 1)/D = 0.091; the rest-mass deficit dominates there too.
Open question for the spec (section 8).

### 4.6 T12, capacity (`KERNELS::bh_pairs_t12`, `HADRONIC::t12_capacity`)

Kernel level: pair maxnpart = 88 with 8 ions at dK/K = 1e-4 per step, 2000
steps: exhaustion at step 860 (in [700, 1300]), n_injected = capacity = 8
exactly at that step, both species full and equal (88), every slot holds a
kernel-born lepton, n_dropped = 8976 (9064 attempts, 88 insertions), the
per-launch identity holds at every step including the drops (max residual
0), the debt of the dropped packets stays outstanding (debt_outstanding /
E_drag = 0.509), the end-of-run identity exact, array residual / B 2.6e-5,
E_round = residual: pass (10 checks).

Engine level (`t12.sh`, pair maxnpart = 60 for 4 ions over 2000 steps,
`on_capacity_estimate = warn` because the estimate (316 sized, 87.8 bare)
exceeds 60 by design): `on_capacity = error` aborts at step 1239 with
"on_capacity: pair capacity exhausted at step 1239 (species 1): capacity =
0, tickets = 4, inserted = 0, n_dropped = 4 this step; e- npart 60 / maxnpart
60, ..." (exit 1), the ledger row written before the abort has n_dropped = 4,
npart 60 / 60, residual 0; `on_capacity = warn` runs to the end: n_dropped =
3044, npart 60 / 60, residual 0, debt_outstanding 2184.87 of E_drag 5757.26,
array residual = E_round = 5.28e-5: pass (8 checks). The engine abort is on
the first step with a drop (the filling step had tickets = capacity exactly
and no drop); the kernel test checks the filling step itself.

Bounds-check rerun: section 6.

### 4.7 T12b, startup estimate (`HADRONIC::t12b_estimate`)

32 ions with Gamma_p log-spaced on [3, 12] (dN/dGamma ~ Gamma^-1, 8 energies
x 4), the declared envelope index 1 on [3, 12], c_layer = 1, comm_headroom =
0, t_run = 1000: printed estimate 904.40 macro-particles per pair species
(rate 0.028262 per active ion per unit time); the input with maxnpart = 452
refused with "on_capacity_estimate: the startup estimate of 904.3982562
macro-particles per pair species (bare 904.3982562) exceeds maxnpart (e- 452,
e+ 452)"; the input with maxnpart = 1809 runs; measured spawn count 816 =
0.902 x the estimate (within a factor 2); no drops; identity 0: pass (5
checks).

### 4.8 T13, discrete charge conservation (`HADRONIC::t13_gauss`)

1D, 16 cells, pairs deposit (ions do not), field solver on, no current
filter, E = B = 0 initially, a background pair plasma of 32 ppc per species
(temperature 0.1) loaded through `arch::InjectNonUniform` (512 e- and 512
e+), 32 ions at dK/K = 1e-3 per step, threshold spectrum with a longitudinal
kick of 0.3 (unequal lepton velocities), 200 steps (708 pairs born). The
residual R_i = (E_{i+1/2} - E_{i-1/2}) - (q0/(B0 V0)) rho_i (CIC rho from the
arrays in double, section 3.4) is compared with its first value R0 (fact: a
randomly loaded background has rho != 0 with E = 0, max|R0| = 2.57e-7 in
field units against max|c rho| = 1.65). Measured, in units of max|c rho|
(the registered run of the final suite; the OpenMP runs differ in
realisation from run to run): max over the 20 rows of max|R - R0| = 3.88e-6
(last row 2.44e-6) for the pair run against 2.79e-6 (last 2.40e-6) for the
control run without pair creation: ratio 1.39 (<= 2; 1.61 in the first
suite), pass; no growth trend (last quarter / max(first quarter, control) =
1.39, <= 2), pass; right after the first births max|R - R0| = 0 exactly (the
two leptons at one position cancel in rho to the last bit and have deposited
nothing yet), pass; identity 0, n_spawned >= 100 (708 and 716): pass (5
checks). Convention for a degenerate control: in the double build the
control run's residual is exactly zero (the deposit and the field update
reproduce the CIC charge to the last bit there), so the comparison "within a
factor 2 of the control" is undefined; in that case, and only then, the
reference is the a-priori accumulated round-off of the field update, n_steps
x eps(real_t) = 200 x 2.2e-16 = 4.4e-14, against which the double pair run
measures 5.5e-15 (0.12, pass). The single build keeps the brief's literal
comparison (its control residual 2.8e-6 is 8.5 times below that floor). Births in the cells next to the periodic boundary are
included (ions at hashed positions over all 16 cells). (diag, filter on) with
one pass of the stock current filter the residual against the unfiltered CIC
rho is 0.90 of max|c rho|: the filter smooths J, so div E follows a smoothed
charge density; the consistent comparison would apply the same stencil to
rho_h (not pursued; reported only).

### 4.9 T-exp, exported debt (`KERNELS::bh_pairs_texp`, `HADRONIC::texp_exported_debt`)

Kernel level: 10 ions, 50 steps, then 4 ions tagged dead on the host:
E_exported = sum w r of the dead ions (host sum before the launch) exact (0,
1e-12), pending_discarded = sum w delta exact, r = delta = K0 = 0 afterwards,
the identity with the exported term exact at the end of 100 steps (array
residual / B 1.5e-5, E_round = residual): pass (6 checks). Engine level
(`texp.sh`, 16 ions at c into absorbing x walls, dK/K = 1e-3 per step,
t_latency auto = 1.6, 60 steps): E_exported_host = E_exported = 249.3965 (difference 0),
pending_discarded_host = pending_discarded = -5.008e-4 (difference 0; a signed
rounding residual of the request), 15 dead ions seen with a payload by the
hook before the sort, the ledger identity 0, 82 pairs born before the ions
left: pass (6 checks); `t_latency = 0` in this box refused naming
`setup.t_latency` (pass).

### 4.10 T10b, stale payloads (`KERNELS::bh_pairs_t10b`, `KERNELS::injectors_payloads`, engine scan)

Kernel level: pair payload arrays with 5 slots filled with NaN, 300 steps
(264 pairs): no live lepton carries a NaN payload, the two slots beyond the
three specified are zero on every live lepton, the unused tails still hold
NaN (48680 slot pairs): pass (3 checks). `injectors_payloads` (30 checks,
section 3.5): every path of the inventory that the prototype can reach is
exercised: the overload directly (tracking, not tracking, npld_r = 0),
`UniformInjector_kernel` (not reachable from a `use_weights = true` Cartesian
run through `arch::InjectUniform`, which refuses weights in Cartesian
coordinates, fact: `particle_injector.h:213-215`; exercised by constructing
the kernel directly), `NonUniformInjector_kernel` (reached end to end by the
T13 background through `arch::InjectNonUniform`), `GlobalInjector_kernel`
(reached end to end by `background.n_global`; the test constructs a
one-domain Metadomain). End to end: every run prints "payload scan : 0 NaN
payload entries on live particles" (fact; the T13 runs with the stock
injector included).

### 4.11 T-P5 (`KERNELS::reduced_stats_weights`)

8 particles in 2D Minkowski, m = 3.5 with charges -2 and +2 (weighted and
unweighted) and m = 1836.15 with charge +1, weights in [0.5, 3.5], momenta in
[-2, 2]^3: N, Rho, Charge, Npart, T00, T0i, Tij against a direct double sum,
1e-6 relative to the sum of absolute contributions: 56 checks, 56 passed
(largest relative error of the registered cases 1.5e-7; the 64-particle
diagnostics reach 1.5e-7 too). The stock `KERNELS::reduced_stats` passes
unchanged.

### 4.12 Suites

Single build (A): 85 tests, 85 passed, 186.5 s wall with `ctest -j1`. The 53
stock tests pass unchanged; the 32 others are `KERNELS::injectors_payloads`,
`KERNELS::reduced_stats_weights`, `KERNELS::bh_loss_*` (10, unchanged step-2
groups: 57 checks + T5b), `KERNELS::bh_pairs_*` (8 groups), `ARCHETYPES::bh_*`
(2), `HADRONIC::{bh_proto_params (100 checks: 10 positive, 90 refusals),
reject (25 inputs, 25 passed), t7a, t7b, t5b_cross_build, t8_bit_identity,
t12_capacity, t12b_estimate, t13_gauss, t10_poisson_determinism,
texp_exported_debt}`. The step-2 regression: every step-2 test passes with
`source = "none"` added to its input (the only input change; the T8 check of
the check tool reads the 41-column ledger).

Double build (B): 85 tests, 81 passed, 3 skipped (`t5b_cross_build`,
`t8_bit_identity`, `t10_poisson_determinism`: no second build is given to the
double tree, as in step 2, not counted as validations), 64.2 s wall; in the
first run `t13_gauss` failed on a degenerate criterion (the control run's
residual is exactly zero in double arithmetic, so "within a factor 2 of the
control" was undefined); with the convention of 4.8 for that case the test
passes in both builds (single 1.39, double 0.12 of the respective level) and
the double tree is 85 / 85 with the 3 skips. Facts from the double build:
`bh_pairs` 69 passed, 0 failed, 1 expected failure (the same T11 (iv)
labelling; measured / III.1 = 2.807), array residual / B between 1.6e-17 and
2.1e-13 (E_round itself is at the 1e-13 level: the stored weights and
momenta are doubles), poisson n_spawned 1950 (identical draws to the single
build: the pool and the draws are double already), `injectors_payloads` 30 /
30, `reduced_stats_weights` 56 / 56, `bh_proto_params` 100 / 100; the step-2
groups pass as in the step-2 round 2.

Poisson determinism on the Serial build (C, `t10det.sh`): two runs with
`simulation.seed = 12345` give identical checksum lines for the ions, both
pair species and the fields and byte-identical pair files; seed 67890 differs
(pass, 4 checks); the hadronic seeds printed are 16993438640103495697 and
15281541011136068842. (diag) On the OpenMP build the same seed gives
different pair checksums (509 pairs in both runs, different arrays): the pool
assigns generator states by scheduling, as stated in the spec.

## 5. RNG seeding (fact)

`simulation.seed` exists on `dev/hadronic` (commit `d03cc08`): parsed in
`parameters.cpp:122-130` as a non-negative 64-bit TOML integer into the
`uint64` parameter `simulation.seed` with the flag `simulation.seed_set`; the
engine re-seeds the per-domain pools from it (`engine.hpp:192-215`). The
hadronic pool of an ion species is a separate `random_number_pool_t`
constructed in `CustomPostStep` on first use with
`hadronic_seed(base, species, domain.index(), checkpoint.start_step)` where
`base` is `simulation.seed` if set, else the compile-time
`constant::RandomSeed`; `hadronic_seed` hashes the tuple by chained
splitmix64 (each component xor-ed with a distinct high tag before the next
hash), never a sum. The seed used is printed
(`bh_proto: species 1: hadronic RNG pool seeded with splitmix64(seed ...,
species 1, domain 0, start step 0) = ...`).

## 6. Bounds-check diagnostic (build D, P8)

Kokkos installed with `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON` and
`Kokkos_ENABLE_DEBUG=ON` (fact: `KokkosCore_config.h` of
`$HOME/.entity/kokkos-debug` defines `KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK`).
Run under it: `test-kernels-bh_pairs.xc t12 t10b texp` (19 checks passed,
exit 0, no bounds error), the engine input `t12_capacity_warn.toml` (the
exhaustion case: n_dropped 3044 at a full capacity of 60, residual 0, exit 0,
no bounds error) and `t13_gauss.toml` (1222 pairs, residual 0, exit 0). The
tickets beyond the capacity never index the pair arrays (facts). Diagnostic,
not registered in ctest.

## 7. Deferred to Rusty (not attempted here, as the brief requires)

T14 (multi-rank **run**: send-tagged ions, pair insertion under MPI, the
receive capacity error; the code side of 3.1 is done in round 2, section
0.5, compile-only), T15b (GPU: CUDA build of the kernel, `Kokkos::fma`,
`atomic_fetch_add` on the device counter, the `dd` reducer on the device),
the double-typed stats entries, T-perf. The production placement
`srpic::HadronicInteractions` (T14b-lite) is done in round 2 (section 0.6).
The reasons for the rest: no MPI run and no CUDA build in this container,
as the brief requires.

## 8. Open questions

1. **T11 (iv) prediction.** The III.1 formula sum w D (1/beta_p - 1) for the
   momentum residual omits the lepton rest mass; with the mean-energy spectra
   at Gamma_p = 6 the pair momentum deficit 2 (gamma_e - p_e) is 1.9 times
   that term, so the 1% criterion cannot be met by construction (4.5). Either
   the prediction should include 2 (gamma_e - sqrt(gamma_e^2 - 1)) per
   interaction (and the outstanding debt's momentum), or the test belongs to
   the rest-frame sampler of step 4 where the lepton energies are drawn.
2. **The poisson identity.** The convention of 3.6 item 3 makes the identity
   hold by definition for `poisson`; the spec should state what
   `E_lost_nopairs` means for that source (the signed imbalance here) or
   exempt `poisson` from the exact identity.
3. **Pairs per unit energy at 1e-3.** The statement holds only when both the
   outstanding-debt fraction and the <D> drift over the run are below 1e-3
   (4.4); the brief's nominal parameters give 7e-2 and 2e-3. The spec could
   phrase it on the paid energy (E_drag - debt_outstanding) at the
   spawn-weighted Gamma_p.
4. **The K0 trigger.** Because a regular packet leaves r - r_th > 0, K0 is
   reset only on the K0 trigger; it fires every ~f_r / (dK/K) steps even when
   packets flow regularly (4.1). If the intention is a trigger only for ions
   whose debt stalls, K0 should be reset after every packet (K0 <- K).
5. **The brief's T10 count formula** floor(sum dK_applied / r_th) assumes a
   constant r_th; with Gamma_p dropping 13% over 2000 steps r_th drifts by 8%
   and the K0 trigger adds packets, so the exact statement is the replay
   (4.1), which is what the test registers.
6. **One pair of species for all hadronic ion species** (3.6 item 4).
7. **Gauss residual with the current filter** (4.8, diag): the consistent
   check needs the filtered charge density.
8. **`GlobalInjector_kernel` weights** (3.6 item 9): left to P10.

## 9. Not done

- Rest-frame sampler, `table` spectrum, event mode, non-uniform bath
  profiles (step 4 and later); `threshold` is a plumbing option only.
- MPI **run** (the reductions, per-rank file names and the construction
  refusal of 0.5 are compiled, not executed), GPU, double-typed stats
  entries, checkpoint restart of the hadronic pool (statistical restart
  designed, not tested): section 7. The production placement is done (0.6).
- P10 (injector weight relaxations) and the schema / parser plumbing: excluded
  by the brief.
- The `pending` request of a dead ion is discarded, as the spec; the T-exp
  runs show it is a signed rounding residual at the 1e-4 level.

## 10. Reproduction

```
# Kokkos variants (Serial only; bounds checking)
cmake -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ENABLE_OPENMP=OFF \
      -D CMAKE_INSTALL_PREFIX=$HOME/.entity/kokkos-serial $HOME/.entity/src/kokkos && ninja install
cmake -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ENABLE_OPENMP=ON \
      -D Kokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON -D Kokkos_ENABLE_DEBUG=ON -D CMAKE_INSTALL_PREFIX=$HOME/.entity/kokkos-debug $HOME/.entity/src/kokkos && ninja install
# builds B and C (entity.xc), then A with the tests
cmake -G Ninja -D TESTS=ON -D output=ON -D precision=double -D mpi=OFF -D pgen=bh_proto -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos;$HOME/.entity/adios2" -S . -B build-double && ninja -C build-double
cmake -G Ninja -D TESTS=OFF -D output=ON -D precision=single -D mpi=OFF -D pgen=bh_proto -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos-serial;$HOME/.entity/adios2" -S . -B build-serial && ninja -C build-serial entity.xc
cmake -G Ninja -D TESTS=ON -D output=ON -D precision=single -D mpi=OFF -D pgen=bh_proto -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos;$HOME/.entity/adios2" \
      -D BH_DOUBLE_ENTITY=$PWD/build-double/src/entity.xc -D BH_SERIAL_ENTITY=$PWD/build-serial/src/entity.xc -S . -B build-single && ninja -C build-single
cd build-single && ctest -j1 --output-on-failure
./tests/kernels/test-kernels-bh_pairs.xc            # all groups, full printout
# round 2: MPI compile-only check (OpenMPI 4.1.6 from apt) and the UBSan diagnostic
cmake -G Ninja -D mpi=ON -D TESTS=OFF -D output=OFF -D precision=single -D pgen=bh_proto -D CMAKE_CXX_COMPILER=mpicxx -D CMAKE_C_COMPILER=mpicc \
      -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos" -S . -B build-mpi && ninja -C build-mpi entity.xc
cmake -G Ninja -D TESTS=ON -D output=OFF -D precision=single -D pgen=bh_proto -D Kokkos_DIR=$HOME/.entity/kokkos/lib/cmake/Kokkos \
      -D CMAKE_CXX_FLAGS="-fsanitize=undefined -fno-sanitize-recover=undefined -g" -D CMAKE_EXE_LINKER_FLAGS="-fsanitize=undefined" -S . -B build-ubsan \
      && ninja -C build-ubsan entity.xc test-kernels-bh_pairs.xc bh_check && build-ubsan/tests/kernels/test-kernels-bh_pairs.xc r2 \
      && bash tests/hadronic/r2_tiny.sh $PWD/build-ubsan/src/entity.xc $PWD/build-ubsan/tests/hadronic/bh_check $PWD/pgens/bh_proto/inputs /tmp/w/r2
# bounds-check diagnostic (build D)
cmake -G Ninja -D TESTS=ON -D output=ON -D precision=single -D mpi=OFF -D pgen=bh_proto -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos-debug;$HOME/.entity/adios2" -S . -B build-bounds \
      && ninja -C build-bounds entity.xc test-kernels-bh_pairs.xc bh_check && build-bounds/tests/kernels/test-kernels-bh_pairs.xc t12 \
      && (cd /tmp/w && build-bounds/src/entity.xc -input pgens/bh_proto/inputs/t12_capacity_warn.toml)
```
