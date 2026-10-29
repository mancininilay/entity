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
| 2 | kernel path: the event branch of `BetheHeitler_kernel` per contract section 2; E1, E5, E8 | done, `7be411a7`; one registered failure: E1 in the single build with the delta spectrum (section 2, stage 2) |
| 3 | statistics: E2 (corrected ODE), E3 (`D2mean` fixture), E6, E7 | done, registered N = 1e5 in both builds: E2 passes for delta, fails for restframe at the deepest targets (fit-vs-Born), E3 as registered fails beyond a 2 % energy loss (first-order drift feedback) while the corrected prediction passes, E6 and E7 pass |
| 4 | spectra and MPI: E4 (calibrated AD against the frozen histograms), the restart refusal at the run level, E9; full ctest single and double | E4 passes 10 of 10 in both builds, the restart test passes, full ctest done in both builds (baselines and registered failures only); E9 deferred (no MPI toolchain on the Mac; job spec in section 2) |
| 5 | report, checkpoint, changelog | done (`Resources/p2-hadronic/notes/2026-10-29-claude1-event-mode-report.md`) |

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
build at Gamma_p = 6 (one or a few events per ion: no averaging at all); the
criterion was then set relative to a <D> from the mechanism (5.84e-5 at
Gamma_p = 6), which a restframe ion at Gamma_p = 3 with one low-D event (D
about 2 against <D> = 5.2) exceeded in the single ctest (4.2e-5 against its
2.4e-5); the criterion is now the mechanism's bound in absolute terms,
|residual| <= n_events(ion) x half ulp-energy(K_init) + 1e-12 x own loss
(one rounding of the stored u per event, each at most half an ulp-energy).
Measured maxima of |residual| / bound: 0.45 (delta, E1 and E8 iii), 0.98
(restframe E1: an ion whose roundings all went the same way, the bound's
worst case reached), 0.32 and 0.25 (E8 i, ii); double build 0.10 / 0.19
(E1 delta / restframe), 0.10 / 0.10 / 0.016 (E8 i, ii, iii). Recorded here as a criterion set after measurements (three
revisions).

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

### Stage 3 (statistics: E2, E3, E6, E7) and stage 4 (E4, restart, E9)

Files: `tests/kernels/bh_event.cpp` (groups `e2e3`, `e2e3_g3`, `e6`, `e7`,
`e4`; `BH_EVENT_N` in the environment reduces N for a development loop,
the ctest entries run the registered N = 1e5), `tests/kernels/CMakeLists.txt`
(`BH_TEST_DATA_DIR` for the fixtures), `tests/hadronic/event_restart.sh`
(ctest `HADRONIC::event_restart`) with the inputs `event_restart_{a,b,c,d,d2}.toml`.

Method (contract section 4):

- The cooling runs: N ions at Gamma_p0 (mu_d = 1, +x), lambda = R dt / a =
  0.05 at Gamma_p0 with dt = 0.5, launched until <K> <= 0.5 K0 (0.69 K0 for
  Gamma_p0 = 3, see below); <K> and Var K recorded every 25 launches
  (two-pass in double from the stored u); the pairs are recycled per launch
  (the ledger carries their energy).
- E2 reference: the table-interpolated RK4 of step 2 (T5's reference) for
  d<K>/dt = -coeff I(<K>) ("plain") and the corrected ODE with the Jensen
  term (1/2) I''(<K>) Var K(t), I'' by centred differences on the
  interpolated table with h = 0.02 K, Var K(t) linearly interpolated from
  the run's records. Verdict at the first record with <K>/K0 <= {0.9, 0.7,
  0.5}: |<K> - K_corr| <= 4 s.e., s.e. = sigma_K / sqrt(N). The plain
  difference and the Jensen term's size relative to 4 s.e. are reported.
  The small-jump statement: the continuous mode (eta_u = 1e-5, source none)
  on the same ions, dt and coefficient, |<K>_event - <K>_continuous| <= 4
  s.e. at the same records.
- E3 reference (registered): Var_pred(t) = int_0^t a R(Gamma_ODE) E[D^2]
  (Gamma_ODE) dt' along the corrected mean, E[D^2] from the frozen bhref
  0.2.0 `D2mean` column of `moments_mono05.txt` (data_sha256
  `69b87f32...`, validated by the reader) interpolated log-linearly in y =
  log10(Gamma - Gamma_th); for the delta spectrum E[D^2] = <D>^2 of the
  table. Verdict at the records where sigma_K/K first reaches 0.005 and 0.01
  and at the last record with sigma_K/K <= 0.02: |Var_meas/Var_pred - 1| <=
  4 sqrt(2/N) (1.8 % at N = 1e5).
- E6: Gamma_p0 = 6, delta, a = 1, fixed t_end = 2000 (4000 launches at dt
  x 1), dt x {1, 2, 4, 8} (lambda 0.05 to 0.4); the dt x 1 run passes E2
  and E3; the deviations of <K> and Var K at t_end from the dt x 1 values
  are fitted by a line through the origin in (dt - dt_1), residuals within
  4 sqrt(2) s.e.; n_capped_events = 0 at dt x 1.
- E7: a in {1, 2, 4} at f_max = 0.01 (registered) with the delta spectrum
  (the bound cannot fire with D = <D> at these a; the restframe runs are
  reported with their bounded fraction), and a in {10, 30} at f_max = 0.1 as
  the labelled diagnostic; the variance against a R E[D^2] for every a.
- E4: restframe, Gamma_p in {3, 10}, 5e5 ions x 11 launches at lambda =
  0.2 with the ions reset to K0 after every launch (fixed Gamma_p; the mean
  relative shift of the sampled Gamma_p is the fraction of second-or-later
  events of an ion within a launch times <D>/K, measured 1.3e-4 and
  6.4e-5, below the contract's 1e-3); 1.1e6 accepted events per Gamma_p;
  E-, E+, cos-, cos+ (the ion along +x) and D (log bins of the supplement
  at Gamma_p = 10) against the frozen *unweighted* chains (column
  `unweighted_count`), the calibrated AD of A10 (COMPARISON.md: common
  partition at >= 100 counts in both samples, Gaussian-multiplier null,
  19,999 replicates, pass iff p >= 0.01, >= 95 % of the 10 comparisons).
- Restart refusal (contract section 3) at the run level: run A (continuous
  budget, checkpoints every 40 steps, labelled 41 and 81 by Entity, 32 ion
  rows with non-zero debt or pending payloads in the per-ion output), run
  B (A resumed with `-continue` in the event mode; the `[setup]` table of
  bh_proto is read from the resume input): refused with "mode = event
  entered with 16 alive ions carrying a non-zero hadronic payload
  (start_step 81)" and the payload-scan line REFUSED; run C (A resumed in
  the continuous mode) completes; runs D and D2 (the event mode
  checkpointed and resumed in the event mode) complete with the payload
  scan clean at step 0 and at the resume (start_step 81), identity exact.

**E3 as registered fails beyond a 2 % energy loss, mechanism (fact, measured
in the development runs at N = 2000 and confirmed below at N = 1e5).** The
registered prediction integrates the compound-Poisson source a R E[D^2]
along the mean with no feedback. But an ion that has lost more sits at a
lower rate (dlnI/dlnK = 4.3 at Gamma_p = 6: I(3)/I(6) = 0.0656/3.26 over
K(3)/K(6) = 0.4), so the spread is pulled back: to linear order dV/dt = -2 c
V + a R E[D^2] with c = coeff dI/dK at <K>(t), and the suppression is
first order, 2 (dlnI/dlnK) (Delta K / K), not second order in sigma_K/K as
the contract's window argument assumed (it used sigma_K/K where the mean
loss Delta K/K enters). At Delta K/K = 0.03 the ratio Var_meas/Var_pred is
0.93 to 0.95, at 0.12 it is 0.72, at 0.5 it is 0.11, with the variance
*decreasing* in absolute terms over the second half of the cooling (the
ions converge on the slow near-threshold rate). The feedback-corrected
prediction (the linear ODE integrated exactly over each RK4 step with c and
the source at the midpoint) agrees within 0.3 to 7 % at N = 2000 (tolerance
12.6 % there). Both verdicts are printed: the registered one [registered:
no feedback] and the corrected one [proposal: linear drift feedback]; the
registered failures are kept. Re-registration is Nilay's decision (section
3, question 8).

**E2 target 0.5 at Gamma_p0 = 3 is below the pair threshold (fact):** K/K0
= 0.5 is Gamma_p = 2.0 < Gamma_th = 2.125 for eps0 = 0.5; no run reaches it.
The Gamma_p0 = 3 run stops at <K>/K0 = 0.69 (the 0.7 target, reached after
the rate has fallen by two orders of magnitude; the 0.9 and 0.7 verdicts
are given).

Results at the registered N (filled in as the runs complete; the
development runs at N = 2000 are in the branch's logs):

Stage 3 at the registered N = 1e5 (s.e. of <K> = sigma_K/sqrt(N) = 0.19 to
0.31; E3 tolerance 4 sqrt(2/N) = 1.79 %); the double-build rows and E6, E7,
Gamma_p0 = 3 follow as the runs complete:

| test | build | registered tolerance | measured | verdict |
| --- | --- | --- | --- | --- |
| E2 Gamma_p0 = 6, delta, vs the corrected ODE at <K>/K0 = 0.9 / 0.7 / 0.5 (t = 1275 / 5700 / 19475; 6.47e7 events) | single | 4 s.e. | 0.88 / 2.26 / 1.28 s.e. (plain ODE: 1.1 / 3.9 / 6.3 s.e.; Jensen term 0.06 / 0.42 / 1.26 of 4 s.e.) | pass |
| E2 Gamma_p0 = 6, restframe, vs the corrected ODE at 0.9 / 0.7 / 0.5 | single | 4 s.e. | 2.15 / 1.06 / **8.60** s.e. (<K> - K_corr = +1.78 at 0.5: 3.9e-4 of the energy lost) | pass / pass / **FAIL** (mechanism below) |
| E2 Gamma_p0 = 6, delta, vs the continuous mode (eta_u = 1e-5) at 0.9 / 0.7 / 0.5 | single | 4 s.e. | 0.92 / 3.65 / **6.23** s.e. | pass / pass / **FAIL** (mechanism below) |
| E3 Gamma_p0 = 6, delta [registered: no feedback] at sigma_K/K = 0.005 (Delta K/K = 0.029) / 0.01 (0.128) / last in window (0.500) | single | 1.79 % | 6.8 % / 32.1 % / 89.1 % | **FAIL** (mechanism in the stage-3 text) |
| E3 Gamma_p0 = 6, delta [proposal: linear drift feedback], same points; max over the window with Delta K/K >= 1 % | single | 1.79 % | 0.85 % / 0.42 % / 0.17 %; max 1.27 % | pass |
| E3 Gamma_p0 = 6, restframe [registered] at sigma_K/K = 0.005 (Delta K/K = 0.021) / 0.01 (0.088) / last (0.500), E[D^2] from the frozen `D2mean` | single | 1.79 % | 5.8 % / 22.7 % / 89.9 % | **FAIL** |
| E3 Gamma_p0 = 6, restframe [proposal], same points; max over the window | single | 1.79 % | 0.17 % / 0.02 % / 0.72 %; max 0.88 % | pass |
| E1-type counters in the cooling runs (n_bounded = n_dropped = n_capped_events = 0 at a = 1, f_max = 0.01, lambda = 0.05) | single | exact | 0 | pass |
| E6 dt x {1, 2, 4, 8} (lambda 0.05, 0.1, 0.2, 0.4) at t_end = 2000, Gamma_p0 = 6, delta, 1.66e7 events each: <K> - <K>_1 at t_end | single | residuals of the linear fit <= 4 sqrt(2) s.e. | -0.32 / +0.14 / -0.06 (= -0.86 / 0.37 / -0.16 s.e.); fitted slope -0.011 per unit dt (0.37 s.e. per unit dt); max residual 0.84 s.e. | pass (no detectable dt dependence of <K> up to lambda = 0.4) |
| E6 Var K - Var_1 at t_end | single | idem | -4.9 / -31 / +31 (= -0.11 / -0.71 / +0.71 s.e.); slope 4.1 per unit dt (44 s.e. per unit dt); max residual 0.85 s.e. | pass |
| E6 dt x 1 passes E2 (0.46 s.e. at 0.9) and E3 [proposal] (0.53 / 0.05 / 0.13 %); n_capped_events = 0 | single | | | pass (E3 [registered] fails: 9.7 / 31.8 / 35.8 %) |
| E7 a = 1 / 2 / 4 at f_max = 0.01, delta: E2 at 0.9 / 0.7 / 0.5 | single | 4 s.e. | a = 1: 1.09 / 0.11 / 1.47; a = 2: 0.72 / 1.03 / 0.10; a = 4: 0.29 / 0.08 / 0.63 s.e. | pass (the mean loss is a-independent) |
| E7 a = 1 / 2 / 4, delta: E3 [registered] | single | 1.79 % | fails at every point with Delta K/K >= 1.5 % (3.8 % to 89 %); passes only at a = 4, sigma_K/K = 0.005 (Delta K/K = 0.8 %: 1.71 %) | **FAIL** (feedback mechanism) |
| E7 a = 1 / 2 / 4, delta: E3 [proposal], max over the window | single | 1.79 % | 1.58 % / 1.23 % / 0.80 %; Var_meas/Var_fb at the end 1.001 / 1.007 / 1.004 (variance a R E[D^2] with the feedback: contract (iii)) | pass |
| E7 restframe a = 1 / 2 / 4 at f_max = 0.01 (reported): bounded fraction; <K> - K_corr at 0.5 | single | reported | 0 / 4.5e-5 / 2.38e-2; +1.79 (8.7 s.e.; the fit-vs-Born offset) / +2.17 (7.4 s.e.) / +66.8 at 0.9 and +119 at 0.7 (0.8 % and 1.9 % of K: the skipped 2.4 % of the events bias the loss, as the bound was designed to show); Var_meas/Var_fb 1.011 / 1.010 / 1.011 | reported |
| E7 diagnostic a = 10 / 30 at f_max = 0.1, delta: <K> - K_corr at 0.9 / 0.7 / 0.5 (s.e. 0.75 / 0.86 / 0.60 and 1.3 / 1.5 / 1.1) | single | labelled, not passed or failed | a = 10: 0.93 / 1.51 / 1.31 s.e.; a = 30: -0.56 / -0.40 / 1.65 s.e.; Var_meas/Var_fb 1.015 and 1.029 | reported: no bias of the mean loss resolved at a <= 30 with the delta spectrum (reading below the table) |
| E7 diagnostic a = 10 / 30 at f_max = 0.1, restframe | single | labelled | a = 10: bounded 0, 0.49 / 2.09 / 3.69 s.e. (the fit-vs-Born offset); a = 30: bounded 3.8e-3 of the events, +18 to +29 (12 to 17 s.e., 0.2 to 0.45 % of K) | reported |
| E2 Gamma_p0 = 3, delta, vs the corrected ODE at 0.9 / 0.7 (t = 1088 / 18963; the 0.7 target lies at Gamma_p = 2.4 where I is 1.2 % of I(3); 2.43e7 events, s.e. 0.092 / 0.060) | single | 4 s.e. | 1.19 / 1.01 s.e. | pass |
| E2 Gamma_p0 = 3, restframe, at 0.9 / 0.7 | single | 4 s.e. | **9.83 / 11.81** s.e. (<K> - K_corr = +0.95 at 0.9, +0.71 at 0.7: 2.6e-3 and 6.4e-4 of the energy lost) | **FAIL** (the fit-vs-Born offset, largest near threshold where I_Born/I_fit - 1 reaches -0.49 %) |
| E2 Gamma_p0 = 3, delta, vs the continuous mode at 0.9 / 0.7 | single | 4 s.e. | 3.70 / **13.72** s.e. | pass / **FAIL** (the continuous mode's Euler error over 38,000 steps with the steep near-threshold rate) |
| E3 Gamma_p0 = 3, delta and restframe [registered] at sigma_K/K = 0.005 (Delta K/K = 0.02) and at the last record (0.31; sigma_K/K never reaches 0.01: the variance shrinks to 9 % of the no-feedback value) | single | 1.79 % | 13.7 % / 93.9 % (delta); 13.5 % / 94.1 % (restframe) | **FAIL** |
| E3 Gamma_p0 = 3 [proposal] at the same points; max over the window | single | 1.79 % | delta 0.62 % / 0.96 %, max 1.75 %; restframe 0.24 % / **1.81 %**, max 1.89 % | delta pass; restframe **fails at the last record by 0.02 %** (the linearised feedback at the steep near-threshold rate: Var_fb = 328 vs 333 measured; reported) |
| **Double build**, every stage-3 group at N = 1e5 (same seeds): E2 delta 0.89 / 2.28 / 1.30 s.e. (Gamma_p0 = 6), 1.19 / 1.02 (3); E2 restframe 2.11 / 1.09 / **8.93** (6), **9.90 / 11.43** (3); vs continuous 0.93 / 3.66 / **6.07** (6), 3.70 / **13.71** (3); E3 [registered] fails at every point as in single; E3 [proposal] passes at every point including the Gamma_p0 = 3 restframe last record (0.99 %; the single build's 1.81 % was the ratio's own noise at the 1 % level: Var 331 vs 333); E6 residuals 0.84 / 0.85 s.e.; E7 as in single | double | as above | | same verdicts as single, except the one E3 [proposal] marginal row, which passes |
| **Born-mean diagnostic (double build)**: <K> - K_born, the corrected ODE with the mean loss J(Gamma) <D>_Born(Gamma) from the fixture's `Dmean` in place of the table's I | double | labelled diagnostic | restframe Gamma_p0 = 6 at 0.9 / 0.7 / 0.5: -2.73 / -2.53 / -1.06 s.e. (the fit table gave 2.1 / 1.1 / 8.9); Gamma_p0 = 3 at 0.9 / 0.7: **0.22 / 0.67 s.e.** (the fit table gave 9.9 / 11.4); K_born - K_fit = 2.5e-3 and 5.9e-4 of the energy lost at Gamma_p0 = 3, 1.9e-4 to 4.5e-4 at 6 | confirms the mechanism of the restframe E2 failures: the sampler follows the Born mean, the table the fit |

**E7 diagnostic at a = 10 and 30 (reading).** With the delta spectrum no
bias of the mean loss is resolved up to a = 30 at f_max = 0.1: the
coarse-graining error of the mean is second order (the loss per event a <D>
is removed exactly and the rate is re-evaluated at the new state before the
next event; only the within-step frozen hazard remains, first order in
lambda = R dt / a, which *decreases* with a). The variance scales as a R
E[D^2] with the feedback (ratios 1.015 and 1.029 at a = 10 and 30). With
the restframe spectrum the bound skips 0.38 % of the events at a = 30 and
f_max = 0.1 and biases the mean by 0.2 to 0.45 % of K: the bias of E7 is the
bound's, not the coarse-graining's. The contract's expected bias scaling
(a <D>/K dlnI/dlnK) applies to a scheme that would draw all a events at
the start-of-step state; the implemented per-event update removes it.

**E2 restframe at <K>/K0 = 0.5 (fact, mechanism).** The restframe sampler
loses energy at the Born chain's mean <D>_Born(Gamma) (bhref 0.2.0 `Dmean`),
while the ODE reference uses the step-1 table I = J <D>_fit built on the
Chodorowski q_k fit; the two differ by -0.49 % to +0.07 % over the mono
baths (Claude 3's review, decision 2026-10-27 (a): the Born-q_k tables of
the next step remove the difference). With N = 1e5 the s.e. at 0.5 is 0.21
on a loss of 4591, i.e. 4 s.e. resolves 1.8e-4 of the energy lost, below
the fit-vs-Born offset. The delta spectrum uses D = <D>_fit = I/J of the
table and agrees with the same ODE at 1.3 s.e., which isolates the
mechanism. The diagnostic reference with J <D>_Born in place of I (the row "Born-mean
diagnostic" in the table) closes the discrepancy: 0.22 and 0.67 s.e. at
Gamma_p0 = 3 where the fit table gave 9.9 and 11.4 s.e. Registered failure kept;
expected to pass once the tables are Born-based (open question 10).

**E2 vs the continuous mode at 0.5 (fact, mechanism).** The continuous mode
applies the Euler step dK = coeff I(K_start) dt at dt = 0.5 (dK/K = 4.5e-5
per step, no sub-cycling); its first-order error per step is (1/2)
(dK_step)^2 (dlnI/dlnK)/K = 4e-5 in K-units, accumulated over 39,000 steps
to about 1 to 2 K-units: measured <K>_continuous - K_plain = -2.37 (5.2e-4
of K) at <K>/K0 = 0.5, while the event mode sits 1.3 s.e. (0.24) from the
corrected ODE. The registered statement compares two discretisations at a
resolution (4 s.e. = 0.76 in K) finer than the continuous mode's own
discretisation error at this dt; it holds at 0.9 and 0.7 and fails at 0.5.
Recommendation (open question 11): re-register the continuous comparison
with the continuous mode run at dt/8 (its error falls to 0.3 K-units), or
state it as agreement with the ODE reference separately for the two modes.

| test | build | registered tolerance | measured | verdict |
| --- | --- | --- | --- | --- |
| E4 at Gamma_p = 3: E-, E+, cos-, cos+, D (1.0986e6 events vs 1e6; merged bins 169, 168, 18, 18, 150) | double | p >= 0.01 each, >= 95 % | p = 0.148, 0.917, 0.205, 0.554, 0.536 | pass |
| E4 at Gamma_p = 10: E-, E+, cos-, cos+, D on log bins (1.0998e6 events; merged bins 170, 170, 48, 48, 160) | double | idem | p = 0.780, 0.602, 0.820, 0.493, 0.882 | pass (10 of 10) |
| E4 fixed-Gamma_p condition (mean relative shift of the sampled Gamma_p) | double | < 1e-3 | 1.3e-4 (3), 6.4e-5 (10) | pass |
| E4 incidental: n_bounded at a = 1, f_max = 0.01 | double | | 0 at Gamma_p = 3; 1 in 1.0998e6 events at Gamma_p = 10 (a_max = 0.84 there: the D tail reaches f_max K = 165 about once per 1e6 events) | reported |
| restart refusal (`event_restart.sh`, 5 runs) | double | per script | B refused naming 16 ions and start_step 81; A, C, D, D2 complete; D and D2 scans clean | pass |

**Full ctest, single build** (`ctest -j 2` with 4 OpenMP threads, the four
long statistics groups `bh_event_{e2e3, e2e3_g3, e6, e7}` excluded and run
separately above): 116 entries, 107 passed, 6 failed, 3 skipped (the
cross-build and Serial-build entries as before), 2070 s wall. The six
failures: `GLOBAL::param_container` and `METRICS::sr-cart-sph` (the
established baselines, as in step 4), `ARCHETYPES::bh_chain_a7w` (the
registered weighted A7 failure, carried), `KERNELS::bh_sampler_a12` (the
registered A12 mean-cost failure at the node Gamma_p = 76.53, 1.285 %
against 1 %: the same node as the cloud failure of step 4; the native
step-4 runs passed it, so this is the statistical fluctuation of a
different thread count and seed assignment, not a change of the sampler,
which step 5 does not touch), `KERNELS::bh_event_e1` and
`KERNELS::bh_event_e8` (the registered E1 single-build delta failures of
stage 2, 1.84e-5 and 1.12e-5 against 1e-5, plus, in that ctest run, the
per-ion criterion's second revision; the rerun with the final criterion
gives 20 passed / 2 failed and 35 / 2, the two being the delta closures).
`KERNELS::bh_event_e4` passes in single (10 of 10, 6.7 s), `HADRONIC::event_e8`
and `HADRONIC::event_restart` pass.

**Full ctest, double build** (same command): 116 entries, 113 passed, 3
failed, 3 skipped, 1745 s wall. The three failures are the baselines
`GLOBAL::param_container`, `ARCHETYPES::bh_chain_a7w` (registered) and
`KERNELS::bh_sampler_a12` at the same node Gamma_p = 76.53 with the same
value 1.285 % (identical in both precisions: the value is fixed by the
fixture test's seed, so it is the registered A12 failure row of step 4
reproducing natively with this build's thread count, not a precision
effect). Every `bh_event_*` entry passes in double (e1 22 / 0, e8 37 / 0,
e4 5 / 0, e5 6 / 0), as do `HADRONIC::event_e8` and `HADRONIC::event_restart`.

**E9 (MPI ledger over ranks): deferred.** No MPI toolchain on the Mac
(`mpicxx`, `mpirun` absent). Rusty job specification, to run under Nilay's
authorisation: build `bh_proto` with `-D mpi=ON -D precision=single` and
`double` in a fresh checkout at the branch head (the rusty-build scripts of
`Entity Local Experiments/rusty-build/`); inputs `event_smoke.toml` with
`hadronic.placement = "production"`, `[simulation.domain] number = {1, 2,
4}` with `decomposition` along x, ions.per_K scaled so that the total ion
number is the same (16 per domain at 1 rank, 8 at 2, 4 at 4: per_K is per
domain in bh_proto), periodic box; verdicts: the final ledger rows'
`E_drag`, `E_pairs - E_absorbed`, `n_events`, `n_spawned` agree in
expectation across the rank counts within 4 s.e. (s.e. of the event count
sqrt(n_events), of the energies sqrt(n_events) <D>), the identity residual
0 in every run (E1 per run via the run's array identity line), and the
counters `n_bounded`, `n_capped_events`, `n_dropped`, `n_sampler_overflow`
present in every rank's reduced row (the reducer's `join` is the one of
step 4; no new counter was added in step 5). The rank-asymmetric check of
the sampler counters remains deferred as before.

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
8. **E3's registered prediction is wrong beyond a 2 % energy loss** (stage
   3 mechanism): the straggling variance along a cooling curve is pulled
   back by the drift's dependence on K, first order in (dlnI/dlnK) Delta
   K/K. Recommendation: re-register E3 on the feedback-corrected
   prediction dV/dt = -2 coeff (dI/dK) V + a R E[D^2] (same source term
   from the frozen `D2mean`, same tolerance 4 sqrt(2/N), same window),
   keeping the registered rows as history; the registered form stays
   valid, and passes, only while Delta K/K <= 1 % (about 10 events per
   ion).
9. E2's target <K>/K0 = 0.5 at Gamma_p0 = 3 lies below the threshold
   (Gamma_p = 2 < 2.125): unreachable by physics, not a failure.
   Recommendation: replace it by 0.7 at Gamma_p0 = 3 (reached), or by a
   target in Gamma_p (e.g. Gamma_p = 2.4).
10. **E2 restframe at <K>/K0 = 0.5 fails at 8.6 s.e.** by the fit-vs-Born
    mean-loss offset (3.9e-4 of the energy lost; stage 3 mechanism), the
    same registered discrepancy as A7 weighted and A12 (decision 2026-10-27
    (a)). Recommendation: keep the failed row; rerun E2 restframe after the
    Born-q_k loss tables land, when the sampler and the loss table share
    their mean loss by construction.
11. **E2 vs the continuous mode at <K>/K0 = 0.5 fails at 6.2 s.e.** by the
    continuous mode's own Euler error at dt = 0.5 (stage 3 mechanism).
    Recommendation: re-register with the continuous reference at dt/8, or
    as separate agreements with the ODE.

## 4. Deferred (listed so that nothing is silently decided)

- E9 on the Mac: no MPI toolchain; Rusty job spec at stage 4.
- The run-level restart refusal test (continuous checkpoint, restart in the
  event mode): stage 4.
- The spectrum `table` of the contract's list {delta, threshold, table,
  restframe}: the kernel has no `table` spectrum (`PairSpectrum` is {Delta,
  Threshold, RestFrame} at 051bb896), and the parser keeps refusing it.

## 5. Branch head

Stage 1: `9a44ecb8`. Stage 2: `7be411a7`. Stages 3 and 4 (code): `bac7cafe`, single-build statistics `a5db13f6`. Final (this commit): see the Resources note `2026-10-29-claude1-event-mode-report.md`, which records the head hash.
