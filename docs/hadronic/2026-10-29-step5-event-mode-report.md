# P2 step 5, C++ side: the event (Monte Carlo) loss mode, `mode = "event"`. Report

Claude 1 (Claude Code on Nilay's Mac), 2026-10-29. Fork `mancininilay/entity`,
branch `dev/hadronic-event` (created from `dev/hadronic` at `051bb896`, in the
worktree `Entity Local Experiments/worktrees/dev-hadronic-event/`). Written
against the local brief of 2026-10-29 and the contract
`Resources/p2-hadronic/notes/2026-10-29-claude1-event-mode-contract.md`
(revision 1.0, normative), spec v3.3.2 II.2, II.3, III.5, III.6, III.7, IV.

Statements are labelled fact (with the number and where it was measured),
convention, or open question. No pre-registered tolerance is adjusted; a
failure is reported with its numbers and mechanism, and re-registration is
Nilay's decision. Code facts are quoted with file, line and commit.

## 0. Stage list (brief, "Stages")

| stage | content | status |
| --- | --- | --- |
| 1 | parser and options: `mode` enum, the refusals of contract section 3, the startup report (lambda_max, a_max, the ulp-energy ratio), the capacity estimate with a instead of r_th; T23 additions | done, `9a44ecb8` |
| 2 | kernel path: the event branch of `BetheHeitler_kernel` per contract section 2; E1, E5, E8 | done (this commit); one registered failure: E1 in the single build with the delta spectrum (section 2, stage 2) |
| 3 | statistics: E2 (corrected ODE), E3 (`D2mean` fixture), E6, E7 | pending |
| 4 | spectra and MPI: E4 (calibrated AD against the frozen histograms), E9; full ctest single and double | pending |
| 5 | report, checkpoint, changelog | pending |

## 1. Environment (facts, measured on the Mac)

| item | value |
| --- | --- |
| `dev/hadronic` HEAD at branch creation | `051bb896` (step 4 merged, 257cfaa1 preserved) |
| compiler | Apple clang (`/Library/Developer/CommandLineTools/usr/bin/clang++`), arm64; `long double` is `double` here |
| cmake / ninja | Homebrew; Kokkos 5.0.1 (Serial + OpenMP via libomp), ADIOS2 2.11.0 non-MPI in `~/.entity/` |
| MPI | not installed on the Mac (`mpicxx`, `mpirun` absent): E9 will be written as a Rusty job spec and marked deferred unless a toolchain appears |
| builds | out of tree in `Entity Local Experiments/worktrees/builds/event-single/` and `event-double/` (`-D pgen=bh_proto -D TESTS=ON -D output=ON -D precision=single|double -D mpi=OFF`, Kokkos OpenMP default space, Serial enabled) |

## 2. Stage log

### Stage 1 (parser and options)

Files: `pgens/bh_proto/bh_proto_params.h`, `pgens/bh_proto/pgen.hpp`,
`src/kernels/hadronic/bethe_heitler.hpp` (enum and parameter only),
`src/archetypes/hadronic/bh_event_host.h` (new, host-only),
`tests/hadronic/bh_proto_params.cpp`, `tests/hadronic/reject.sh`, five new
inputs `pgens/bh_proto/inputs/reject_mode.toml`, `reject_mode_source.toml`,
`reject_event_amax.toml`, `reject_event_ulp.toml`, `reject_event_lambda.toml`,
and `event_smoke.toml` (the stage-2 smoke input).

What was implemented:

- `mode = "continuous" | "event"` (default continuous), compared
  case-insensitively and stored lower-cased (`bh_proto_params.h`, the block
  "loss operator (step 5)"); `Setup::event_mode()`; `pairs_enabled()` is true
  in the event mode whatever `source`.
- Refusals of contract section 3 at parse time: `source` present with
  `mode = event` (message names `setup.source`); `pair_species` absent;
  neither `gamma_p_typical` nor an envelope (the active band of the a_max
  refusal and the capacity estimate: a new requirement, see the open
  questions); `on_bound` not in {error, warn}. In the event mode the parser
  sets `source = "none"` internally so that no continuous-mode guard keyed on
  the source applies (eps_max < 2 of the budget source, the latency cap); the
  kernel's source enum is None there and the event branch (stage 2) ignores
  it.
- New policy key `on_bound` ("error" | "warn", default error) for the
  per-event bound a D <= f_max K (spec III.7 lists it; it did not exist in
  the parser because the continuous mode clamps instead).
- `kernel::hadronic::LossMode {Continuous, Event}` and `HadronicParams::mode`
  (`bethe_heitler.hpp`); the pgen passes it (`HadronicStage`). The kernel
  body is unchanged in stage 1; a run in the event mode stops at the first
  stage call with "the kernel branch is not implemented yet (step 5 stage
  2)", after the startup report.
- `src/archetypes/hadronic/bh_event_host.h`: `dmax_unweighted(Gamma,
  eps_max)` (the closed form of spec III.2 at eps' = eps'_hi(eps_max)),
  `a_max(Gamma, eps_max, mu_r, f_max) = f_max K / D_max`, `ulp_energy<R>(K,
  mu_d)` (templated on the floating type), `table_numbers<R>` (lambda_max
  over the table and the minimal ratio a <D> / ulp-energy with their nodes).
- Startup report (`pgen.hpp`, `event_mode_report()`), printed after the
  table summary when `mode = event`: a_max and lambda = R dt / a at the table
  edges, at `gamma_p_typical`, at the envelope ends and at every initial
  `ions.K`; the active band = {gamma_p_typical} + the envelope on a 65-node
  log grid + the initial energies + every table node below the largest of
  these (the cooling path; no monotonicity of a_max is assumed, see the
  fact below); refusal when `pair_weight` exceeds the band minimum of a_max
  (message names `setup.pair_weight` and "exceeds a_max"); the minimal
  a <D> / ulp-energy over the whole table per ion species (real_t of the
  build), refused below 16 ("ulp-energy"); lambda_max over the active band,
  refused at or above 1 ("lambda_max"), reported above 0.1 under the
  `on_max_events` policy (error refuses, warn continues); for information
  the table-wide lambda_max and the table-wide maximum of <D> R dt / (f_max
  K) (the per-step condition of spec III.5 is the kernel's, stage 2).
- The capacity estimate (`spawn_rate_at`) uses coeff J / a in the event mode
  (as for the poisson source), i.e. one macro-pair of weight a per
  interaction.
- Restart refusal (contract section 3): at the first stage call in the event
  mode the three hadronic payload slots of every alive ion are scanned on
  the host, the count of ions with a non-zero slot is reduced over the ranks
  (`MPI_Allreduce`, one domain per rank in Entity) and the run is refused
  when it is positive (predicate `bhproto::event_restart_refused`, unit
  tested). A fresh run has every slot zero (P9), so this fires only on a
  restart from a continuous-mode segment. The run-level test (write a
  continuous checkpoint, restart in the event mode) is scheduled with the
  ctest work of stage 4.
- The startup line and the ledger file header carry `mode` and `on_bound`.

Facts measured in stage 1 (single build, `test-hadronic-bh_proto_params.xc`
140 passed, 0 failed; `reject.sh` 32 inputs executed, 32 passed):

- The closed-form D_max,unweighted equals the brute-force maximum of
  D = Gamma (eps' + beta p'_z) - eps over (eps', p'_z) at fixed eps for 15
  (eps, Gamma) cases on a 4001 x 401 grid (relative agreement 1e-6, never
  exceeded).
- a_max at eps0 = 0.5, mu_r = 1836.15, f_max = 0.01: 2.579 at Gamma_p = 3,
  1.341 at 6, 0.841 at 10. **The contract's orientation numbers (6.5 at 3,
  7.8 at 6) are f_max K / <D>, not f_max K / D_max** (with <D> = 5.159 and
  8.364 from the step-1 table: 36.72/5.159 = 7.1, 91.81/8.364 = 11.0; the
  contract's values are of that kind). See open question 1.
- a_max(Gamma) decreases monotonically above the threshold for eps_max in
  {0.3, 0.5, 1.0}; for eps_max = 1.9 (Gamma_th = 1.0013) it rises from the
  threshold to a maximum 0.877 at Gamma = 1.335 and decreases beyond; it
  decreases for Gamma >= 2 in every case. The band check scans the cooling
  path explicitly and does not rely on monotonicity.
- ulp-energy: `ulp_energy<float>/ulp_energy<double>` lies in (2^28, 2^30) at
  six (K, mu_d) points and the double value matches `std::nextafter` directly
  to 1e-12. On the mono05 table with a = 1, mu_d = 1 the minimal ratio
  a <D> / ulp-energy is 1754 (float) at Gamma_p = 73.9 (the table's hard
  end), far above the registered 16.
- `event_smoke.toml` (tau0 = 0.4, a = 1, f_max = 0.01, ions at Gamma_p = 3
  and 6): lambda = 0.078 at Gamma_p = 6, a_max over the band 1.341 at
  Gamma_p = 6 (admitted), lambda_max over the whole table 1.34 at the
  envelope Gamma_p = 91.8 (information). With the contract's literal "max
  over the table" this input, and every input whose ions stay below the
  envelope, would be refused: see open question 2.

Deviations from the contract text in stage 1, all listed as open questions
below: (1) the a_max orientation numbers; (2) lambda_max over the active band
instead of the table; (3) `gamma_p_typical` or an envelope required with the
event mode.

### Stage 2 (kernel path: E1, E5, E8)

Files: `src/kernels/hadronic/bethe_heitler.hpp` (`event_branch`, called from
`operator()` after the clamp above the table; header comment),
`pgens/bh_proto/pgen.hpp` (the `on_bound` host policy, failure code 3 at the
C-MPI-2 common point; the stage-1 guard removed; the test hook
`test.bypass_amax`), `pgens/bh_proto/bh_proto_params.h` (`test.bypass_amax`),
`tests/kernels/bh_event.cpp` (new: groups `e1`, `e5`, `e8`),
`tests/kernels/CMakeLists.txt`, `tests/hadronic/bh_check.cpp`
(`--min_bounded`, `--expect_bounded`, `--min_events`),
`tests/hadronic/event_e8.sh` (new, ctest `HADRONIC::event_e8`),
`tests/hadronic/CMakeLists.txt`, inputs `event_bound_{error,warn}.toml`,
`event_capacity_{error,warn}.toml`, `event_maxevents_{error,warn}.toml`.

What was implemented (contract section 2, in the order of its steps):

- Step 0 as in II.2 (the clamp above the upper edge counts
  `n_above_table`); below the lower edge, or at rest, the ion returns with
  no RNG use and no closing packet (no debt exists). The three payload slots
  are neither read nor written in the event mode.
- Step 1: R = (tau0/skindepth0) g(x) J(Gamma_p); R dt = 0 returns without
  RNG use.
- Step 2: n from the existing exact capped `poisson_sample` with lambda =
  R dt / a (compared with `max_events_per_step` in double before any
  conversion; clamped and counted in `n_capped_events`).
- Step 3 per event at the current state (K_cur, Gamma_cur; u-hat fixed):
  (a) `restframe`: `sample_restframe(Gamma_cur, weighted = false, ...)`,
  an overflow falls back to the mean-energy law and counts
  `n_sampler_overflow`; `delta`/`threshold`: `outcome_law`. (b) The bound
  a D_s <= f_max K_cur; a violation skips the event (no spawn, no recoil),
  counts `n_bounded` and `n_events`; the host applies `on_bound`. (c)
  `spawn_pair_vec` / `spawn_pair` with w_pair = a w and `intended` =
  w_pair D_s in dd; a capacity failure ends the loop with no recoil
  (`n_dropped` by the spawn). (d) K_new = K_cur - a D_s in double, u_new
  along u-hat stored as real_t, dK_applied = K_cur - K(u_new,stored) from
  the stored values, `E_drag`, `P_drag` (along u-hat), `B_budget`,
  `n_applied`. (e) `E_round += w dK_applied - intended` plus the spawn's
  pair term. (f) K_cur, Gamma_cur from the stored u (clamped above the
  table); an ion that cools below the lower edge within the step has no
  further interaction in that step.
- `n_events` = spawned + dropped + bounded (the drawn interactions).

Facts measured in stage 2 (the registered tolerances of the contract; the
per-ion closure is an additional check of mine, see below):

| test | build | registered tolerance | measured | verdict |
| --- | --- | --- | --- | --- |
| E1 array closure, delta (1000 ions at Gamma_p 3 and 6, a = 1, lambda = 0.1 at 6, 200 launches, 10035 events) | single | 1e-5 | 1.83e-5 | **FAIL** (mechanism below) |
| E1 array closure, delta | double | 1e-12 | 7.7e-15 | pass |
| E1 array closure, restframe (10123 events) | single | 1e-5 | 7.5e-7 | pass |
| E1 array closure, restframe | double | 1e-12 | 3.6e-16 | pass |
| E1 ledger identity residual / E_drag (both spectra) | single, double | 1e-12 | 0 (exactly, dd accumulators) | pass |
| E1 abs(E_round)/E_drag, delta | single | 1e-5 | 1.83e-5 (= the closure: E_round is the array residual by construction, difference 0) | **FAIL** |
| E1 abs(E_round)/E_drag, delta | double | 1e-13 | 7.7e-15 | pass |
| E1 abs(E_round)/E_drag, restframe | single / double | 1e-5 / 1e-13 | 7.5e-7 / 3.6e-16 | pass |
| E1 drawn events vs the exact Poisson mean from the per-launch states (4 sigma) | both | 4 | 0.96 (delta), 0.10 (restframe) | pass |
| E1 stateless: every payload slot 0; E_stochastic = E_lost_nopairs = E_exported = debt = pending = 0; transverse P_drag = 0 | both | exact | exact | pass |
| E5 identical event counts per ion, mu_d = 1 vs 1836.15, Serial policy, 64 ions, 100 launches, lambda = 0.2 (1295 events delta, 1299 restframe; n_rng identical: 6400 and 16792) | both | 0 ions differing | 0 | pass |
| E5 per-ion K after every launch, max relative difference | single / double | 1e-5 / 1e-12 | 5.0e-7 and 4.8e-7 / 1.2e-15 and 2.0e-15 | pass |
| E8 (i) kernel: f_max = 1.1e-3, delta, Gamma_p = 3 ions all bounded, Gamma_p = 6 none: their u bitwise unchanged, no pair attributed to them, n_bounded = n_events - n_spawned, bounded fraction vs the Gamma_p = 3 rate share (0.66 sigma) | both | exact / 4 sigma | exact / 0.66 | pass |
| E8 (ii) kernel: capacity 300 for 1962 events: n_dropped 1662, n_spawned = 300 = both npart, closure | both | E1 | closure 1.13e-5 single (same mechanism, **FAIL** against 1e-5), 8.6e-14 double | single fail, double pass |
| E8 (iii) kernel: max_events_per_step = 1 at lambda = 1: n_capped_events = 2548 vs 10000 (1 - 2/e) = 2642 +- 44 (2.1 sigma), at most one pair per (ion, launch), closure | both | 4 sigma / E1 | 2.1 / 9.3e-6 single, 9.9e-15 double | pass |
| E8 run level (`event_e8.sh`, 7 runs, single and double): `event_smoke` completes (identity 0, n_bounded = n_dropped = n_capped_events = 0, array identity line clean); `on_bound = error` aborts at step 16 with the flag, the gathered rank line and the common-point message; `on_bound = warn` completes with n_bounded = 7 and the identity exact; `on_capacity = error` aborts at step 44 with the row written (n_dropped > 0), `warn` completes with n_dropped = 192 and both species at 60; `on_max_events = error` aborts at step 6, `warn` completes with n_capped_events = 6 | both | per script | all pass | pass |
| T23 parser (142) and `reject.sh` (32) | both | | all pass | pass |

**The E1 single-build failure with the delta spectrum, mechanism (fact).**
The recoil is applied by rounding u_new = u_cur - delta_u to the nearest
float, delta_u = a D gamma_d / (mu_d u) (= a D for mu_d = 1, u >> 1). With
the delta spectrum D = <D>(Gamma_p) is the same at every event of an ion
while Gamma_p is nearly constant, so delta_u / ulp(u) has the same fractional
part at every event and the rounding error does not change sign: it
accumulates linearly, not as sqrt(N). The worst case per event is
ulp-energy / 2 = ulp(u) mu_d u / (2 gamma_d) = 4.9e-4 at Gamma_p = 6 against
a <D> = 8.36, i.e. 5.8e-5 relative; measured 1.83e-5 (E1), 1.13e-5 (E8 ii,
300 events, little averaging), 9.3e-6 (E8 iii, 31 events per ion: the
fractional part drifts as the ions cool and the bias partly averages). With
the restframe spectrum D varies from event to event, the rounding is
random in sign and the closure is 7.5e-7 (the contract's "expected ~1e-7"
is reached within an order of magnitude; 1e4 events, per-event 1e-4
relative, sqrt(N) averaging gives 1e-6). In the double build the same
quantity is 1e-14. The contract's argument "random in sign" (section 2 e)
assumed a varying D; it holds for restframe and fails for delta. Remedies
for Nilay's decision (section 3, question 5): accept as documented (delta
is a test spectrum; production is restframe), or stochastic rounding of the
stored u_new (one extra uniform per event, E[dK_applied] = a D exactly for
every spectrum).

**Per-ion closure (my addition, not in the contract).** The closure per ion
(its own loss against its own pairs) was first checked at the E1
tolerance and measured 2.6e-5 (delta) and 4.3e-5 (restframe) in the single
build at Gamma_p = 6 (one or a few events per ion: no averaging at all), so
the criterion was then set from the mechanism, ulp-energy(K_init) / (2 a
<D>(Gamma_init)) = 5.84e-5 at Gamma_p = 6, 1e-12 in double; it passes in
every group (double: 1e-13). Recorded here as a criterion set after a first
measurement.

**Contract corrections found in stage 2 (facts by algebra, tested by E1).**
(1) Section 2 e writes the ion term of E_round as w (a D_s - dK_applied);
with the identity E_drag = (E_pairs - E_absorbed) + E_round and the pair
term w_pair D_s - (stored pair energy - w_pair eps) it must be
w (dK_applied - a D_s): implemented so; the identity residual is exactly
0 in both builds. (2) Section 3's a_max orientation numbers are f_max K /
<D> (section 2, stage 1). (3) The per-step condition <D> R dt <= f_max K of
spec III.5 is not in the contract's per-ion algorithm and is not enforced
by the kernel (the per-event bound is); its table-wide maximum is printed
at startup (2.6e-2 for `event_smoke`).

**`on_bound` is unreachable inside the declared band (fact by
construction).** The startup refusal admits a run only if a <= f_max K /
D_max over the band and the cooling path, and every sampled D is <= D_max,
so no event of an ion inside the band can violate a D <= f_max K. The policy
path is reachable only for ions accelerated above the band (where a_max is
smaller) or with the refusal bypassed; the run-level test uses the
documented hook `test.bypass_amax = true` with f_max = 1.1e-3 and the
delta spectrum (every Gamma_p = 3 event bounded, no Gamma_p = 6 event). A
first version with ions accelerated by E = 40 x-hat from Gamma_p = 6 to 8.2
and a = 1.3 (admitted at 6, above a_max(8.2) = 0.98) produced no bounded
event in 594 restframe events: the D tail near D_max is far too thin (next
paragraph).

**The D tail against D_max (fact, from the frozen bhref 0.2.0 unweighted D
histograms, N = 1e6 each).** Gamma_p = 3: <D> = 5.137, quantiles 0.99 /
0.999 / 0.9999 at 1.89 / 2.23 / 2.44 <D>, largest populated bin edge 13.86
= 0.973 D_max (D_max = 14.24); P(D > 2 <D>) = 4.9e-3. Gamma_p = 10: <D> =
11.19, quantiles 0.99 / 0.999 / 0.9999 at 4.66 / 7.71 / 10.67 <D>, largest
populated bin 161.4 = 0.82 D_max (196.5); P(D > 2 <D>) = 9.5e-2, P(D > 5
<D>) = 7.6e-3. So at Gamma_p = 10 the <D>-based a_max (14.8 at f_max =
0.01) would let about 1e-2 of the events violate the bound at a ~ 3, while
the D_max-based a_max (0.84) is conservative by a factor 1.2 against the
largest D seen in 1e6 draws and 1.6 against the 0.9999 quantile. These
numbers quantify open question 1.

## 3. Open questions for Nilay (every **[proposal]** of the contract the implementation touched)

1. **a_max definition and the registered tests.** The contract defines
   a_max = f_max K / D_max,unweighted (implemented) but its orientation
   numbers come from <D>. With D_max the admitted weights are a <= 2.58 at
   Gamma_p = 3, 1.34 at 6, 0.84 at 10 (eps0 = 0.5, f_max = 0.01). Then E7's
   a in {1, 2, 4} at f_max = 0.01 is admitted only for a = 1 at Gamma_p = 6
   and a in {1, 2} at Gamma_p = 3; E4 at Gamma_p = 10 with a = 1 needs
   f_max >= 0.012 (0.02 proposed); a in {10, 30} at f_max = 0.1 are admitted
   only at Gamma_p = 3 (a_max = 25.8) and not at 6 (13.4): 10 yes, 30 no.
   Recommendation: keep the D_max form (it is the only one under which
   `on_bound = error` can never fire by construction on the band) and
   re-register E7 at f_max = 0.04 for a in {1, 2, 4} (a_max = 5.4 at
   Gamma_p = 6) and E4 at f_max = 0.02; the registered rows stay as history.
   Alternative: a_max from <D> (the contract's numbers), under which the
   tail of the outcome law fires `on_bound` at a rate the kernel will count
   in `n_bounded` (stage 2 measures that rate).
2. **lambda_max over the active band, not the table.** Implemented over the
   band (gamma_p_typical, envelope, initial energies, cooling path); the
   table-wide value is printed. Recommendation: accept the band definition.
3. **`gamma_p_typical` or an envelope required with `mode = event`** (as with
   the budget source), because the a_max refusal and the capacity estimate
   need the active band. Recommendation: accept.
4. The ulp-energy minimum 16 and the lambda thresholds 0.1 (policy) and 1
   (refusal) are implemented as proposed.
5. **E1 in the single build with the delta spectrum fails the registered
   1e-5 (1.8e-5 measured) by a systematic rounding of the recoil** (stage 2
   mechanism). Options: (a) accept and record: the delta spectrum is a
   development spectrum, the production restframe spectrum passes at 7.5e-7,
   the double build at 1e-14; (b) stochastic rounding of the stored u_new
   (unbiased for every spectrum; one more uniform per event; E5 unchanged
   in kind). Recommendation: (a), with the failed row kept; (b) only if a
   single-precision delta-spectrum production run is ever intended.
6. The test hook `test.bypass_amax` (default false, loud at startup) exists
   so that the `on_bound` host policy has a run-level test; without it the
   policy is unreachable for admitted inputs inside the band.
7. The per-step condition of spec III.5 (<D> R dt <= f_max K) is not
   enforced by the kernel (the contract's algorithm has only the per-event
   bound); it is printed at startup. Recommendation: leave it to the
   startup report (it is implied by lambda_max < 1 and the per-event bound
   up to a factor <D>/D_max).

## 4. Deferred (listed so that nothing is silently decided)

- E9 on the Mac: no MPI toolchain; Rusty job spec at stage 4.
- The run-level restart refusal test (continuous checkpoint, restart in the
  event mode): stage 4.
- The spectrum `table` of the contract's list {delta, threshold, table,
  restframe}: the kernel has no `table` spectrum (`PairSpectrum` is {Delta,
  Threshold, RestFrame} at 051bb896), and the parser keeps refusing it.

## 5. Branch head

Stage 1: `9a44ecb8`. Stage 2: (hash filled in at the next commit)
