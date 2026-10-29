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
| 1 | parser and options: `mode` enum, the refusals of contract section 3, the startup report (lambda_max, a_max, the ulp-energy ratio), the capacity estimate with a instead of r_th; T23 additions | done (this commit) |
| 2 | kernel path: the event branch of `BetheHeitler_kernel` per contract section 2; E1, E5, E8 | pending |
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
| builds | out of tree in `Entity Local Experiments/worktrees/builds/event-single/` (`-D pgen=bh_proto -D TESTS=ON -D output=ON -D precision=single -D mpi=OFF`); the double build is added at stage 2 |

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

## 4. Deferred (listed so that nothing is silently decided)

- E9 on the Mac: no MPI toolchain; Rusty job spec at stage 4.
- The run-level restart refusal test (continuous checkpoint, restart in the
  event mode): stage 4.
- The spectrum `table` of the contract's list {delta, threshold, table,
  restframe}: the kernel has no `table` spectrum (`PairSpectrum` is {Delta,
  Threshold, RestFrame} at 051bb896), and the parser keeps refusing it.

## 5. Branch head

Stage 1: (hash filled in at the commit)
