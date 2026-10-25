# P2 step 2, the loss-only prototype: continuous Bethe-Heitler ion loss. Report

Claude 1 (Claude Code cloud session), 2026-10-25. Fork `mancininilay/entity`,
branch `dev/hadronic-loss` (created from `dev/hadronic` at `a346254`).
Commits: `b41cf59` (device tables, kernel, bh_proto generator, unit tests),
`5c5daba` (run-based tests, check tool, inputs, host interpolation header),
`a12cc24`/`ac77a59` (round-1 report). Round 2 (referee findings, the decisions
of 2026-10-25, the fixture test): `d68e162`, `8defba7`, and the commit carrying
this revision of the report; the data commit `0d5d2e8` (bhref 0.1.0 fixtures)
was added by the project. No core file was changed (section 3.6). Section 10
is the round-2 resolution log with the re-registered results; sections 4 to 6
keep the round-1 numbers as history where the round-2 registration changed the
criterion (marked "round 1" there).

Every statement below is labelled as a **measured fact** (with the number), a
**convention** adopted here, or an **open question**. No pre-registered
tolerance was adjusted; the three pre-registered items that fail are reported
as failures with their numbers and the mechanism behind each (section 4). The
T5b "design off" runs were labelled expected failures before they ran.

## 1. Environment and builds

Facts, measured in the cloud container:

| item | value |
| --- | --- |
| `dev/hadronic` HEAD at branch creation | `a346254` (fix(hadronic): strict JSON validation of exchange-file header values (round 3)) |
| gcc | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1); the only compiler used |
| cmake / ninja | 4.4.4 / 1.13.2 |
| nproc | 4 |
| Kokkos (OpenMP + Serial) | `$HOME/.entity/kokkos`, commit `89f0b12e4` (5.2.1); OpenMP is the default execution space |
| Kokkos (Serial only) | built here from the same source into `$HOME/.entity/kokkos-serial` (`Kokkos_ENABLE_SERIAL=ON`, `Kokkos_ENABLE_OPENMP=OFF`, Release, C++20); `KokkosCore_config.h` has `KOKKOS_ENABLE_SERIAL` defined and `KOKKOS_ENABLE_OPENMP` undefined |
| ADIOS2 | `$HOME/.entity/adios2`, 2.12.1, no MPI |
| build A (`build-single`) | `-D TESTS=ON -D output=ON -D precision=single -D mpi=OFF -D pgen=bh_proto`, OpenMP Kokkos; also `-D BH_DOUBLE_ENTITY=<build B>/src/entity.xc -D BH_SERIAL_ENTITY=<build C>/src/entity.xc` |
| build B (`build-double`) | same with `-D precision=double -D TESTS=OFF` (only `entity.xc`) |
| build C (`build-serial`) | `-D precision=single -D TESTS=OFF` against the Serial-only Kokkos |
| round 2: build B with `-D TESTS=ON` | the double suite (section 10.3); `BH_DOUBLE_ENTITY` and `BH_SERIAL_ENTITY` left unset there, so its cross-build and T8 entries are skipped (R6: in the double build the "single" and "double" executables would be the same binary) |
| round 2: build D (`build-ubsan`) | `-D precision=single -D pgen=bh_proto` with `CMAKE_CXX_FLAGS="-fsanitize=undefined -fno-sanitize-recover=undefined -g"`, only the kernel unit test target; a diagnostic, not a registered test (section 10.4) |
| stock build and ctest (before any change) | 160 targets, 53/53 tests passed, 6.0 s |
| compile time | summed object compile times from `.ninja_log`: 1438 s (A, all targets incl. tests), 518 s (B), 463 s (C); the five engine/entity objects of A take 75 s summed |
| run cost | 0.26 ms per step for the 1D, 16-cell, 4-ion, no-solver, no-deposit configuration on 4 OpenMP threads while other builds were running (10 000 steps in 2.63 s wall); 18 us per step on an idle machine (T7a: 2 x 11 600 steps in 0.42 s) |

Logs of every build and test run are under the session scratchpad
(`logs/*.log`); the ctest work directories of the run-based tests are under
`build-single/tests/hadronic/work/`.

## 2. The payload-slot mechanism (brief 1.3)

Fact: in this tree a species declares its real payload slots with the input key
`n_payloads_real` of `[[particles.species]]` (`parameters/particles.cpp:129`,
stored in `ParticleSpecies::m_npld_r`, allocated as `pld_r(maxnpart, npld_r)`).
There is no API for a problem generator to add slots after parsing; the spec's
"appended by the parser" is the production design and is not implemented.

Convention adopted: the generator requires every hadronic species to declare at
least `max(payload_pending, payload_debt, payload_debt_k0) + 1` real slots
(3 with the defaults 0, 1, 2) and refuses the run otherwise with a message
naming the species, its `n_payloads_real` and the three indices
(`pgen.hpp:423-432`; exercised end to end by `reject_payloads.toml`). The
indices are configurable through `[setup]` (`payload_pending`,
`payload_debt`, `payload_debt_k0`, validated distinct and non-negative); the
kernel reads only `payload_pending` in this step. After every injection the
slots of the new ions are zeroed over `[npart_before, npart_after)` by a
separate kernel (`zero_payloads`, `pgen.hpp:616-631`; P9 prototype recipe).
The injection kernel itself does not write `pld_r`, so the zeroing pass is the
only mechanism, as in the production recipe.

Two further prototype limitations are enforced at construction (facts about
the tree, conventions here): `RemoveDead` compacts alive particles with atomic
tickets (`particles_sort.cpp:135-143`), which permutes the array order on
OpenMP, and the per-ion output relies on the array order; the generator
therefore requires `clear_interval = 0` and `spatial_sorting_interval = 0` on
every hadronic species (no dead ions exist in this step, so nothing is lost).
Multi-domain runs are refused (`domain.index() != 0`).

## 3. What was implemented and where

Line numbers refer to commit `5c5daba`.

### 3.1 Device tables: `src/archetypes/hadronic/bh_tables_device.h` (231 lines)

| lines | content |
| --- | --- |
| 49-71 | `HadronicTables`: `gamma_th, gamma_min, gamma_max, y_min, y_max, dy_fine (1/256), dy_coarse (1/64), n_fine, n` and the `Kokkos::View<double*>` node arrays `y, gamma, loss (I), rate (J), dmean, eps_mean, log_loss, log_rate` |
| 73-100 | `cell_index(y)`: analytic index from the adopted placement (fine decade then coarse; the fine formula alone when the table ends inside the first decade), clamped, then corrected by at most one step against the stored node positions, so the shorter last interval is handled by the same code |
| 103-129 | `interpolate_log`: linear in y of ln I (ln J); 0 at or below the lower edge (`Gamma < gamma_min`, or `y < y_min`), the last node value exactly at and above the upper edge, `above` flagged only for `Gamma > gamma_max` (the caller counts `n_above_table`) |
| 131-150 | `I(Gamma[, above])`, `J(Gamma[, above])` |
| 156-228 | `mirror_tables_to_device(HadronicTablesHost)`: consistency checks (sizes, `n_fine`, positive finite I and J), host mirrors, `deep_copy` |

`src/archetypes/hadronic/bh_interpolate.h` (58 lines): the Kokkos-free host
reference of the same interpolation (`interpolate_loss_host`, bisection on the
stored nodes), used by the unit test, the generator's summary and the check
tool. Fact: device and host interpolation agree to 0 (exactly) at every cell
midpoint (same arithmetic, section 4.1).

### 3.2 Kernel: `src/kernels/hadronic/bethe_heitler.hpp` (353 lines), `kernel::hadronic::BetheHeitler_kernel<M>`

| lines | content |
| --- | --- |
| 62-85 | `HadronicParams`: `tau0_over_skindepth0, mu_r, eta_u, pending_floor_ulp, f_max, max_substeps`, payload indices `pld_pending, pld_debt, pld_debt_k0`, `dt, time, profile` (uniform only), `unsafe_application` (design off) |
| 87-153 | `LossLedger`: every entry of II.2 step 6 and part IV in double (`E_drag, E_pairs, E_absorbed, E_exported, E_round, E_lost_nopairs, pending_discarded, P_drag[3], P_pairs[3], P_absorbed[3]`), the snapshots `debt_outstanding, pending_outstanding, sum_wK`, the `unsigned long long` counters (`n_events, n_spawned, n_dropped, n_substeps, n_bounded, n_closing, n_capped, n_above_table, n_latency, n_resolution`) plus `n_processed, n_alive, n_applied`, and (round 2) `n_capped_loss`; `operator+=`; `Kokkos::reduction_identity<LossLedger>` so that `Kokkos::Sum<LossLedger>` reduces it. No `real_t` atomics anywhere |
| 159-166 | `kinetic_energy(mu, u1, u2, u3)`: `K = mu u^2 / (gamma + 1)` in double from the three stored components; the one expression shared by the kernel, the generator and the tests (the array-based closure relies on it) |
| 168-172 | `ulp_real(real_t)`: `nextafter(x, +inf) - x` |
| 204-360 | `operator()(p, LossLedger&)`: tag class (`dead`: `pending_discarded += w delta`, `delta <- 0`, return; anything else processed); step 0 (`K`, `Gamma_p`; above `gamma_max`: clamp and `n_above_table += 1`; below `gamma_min`: discard a pending `delta` into `pending_discarded`, snapshot, return); step 1 (`dK = coeff g I(Gamma_p) dt`, `g = 1`; if `dK > f_max K`: `n = ceil(dK/(f_max K))` midpoint-RK2 substeps on the table with `K_floor = mu_r (gamma_min - 1)` as the lower clamp, `n <= max_substeps` else `n = max_substeps`, `n_bounded += 1` and the loss clamped to `f_max K`; `n_substeps += n`; `delta += dK`; `n_processed += 1`); step 2 (`pending_floor = pending_floor_ulp ulp(|u|_real_t) mu_d u/gamma_d`, `delta_th = max(pending_floor, eta_u K)`; if `delta > 0` and `delta >= delta_th`: `K_new = min(K, max(K - delta, K_floor))`, `u_new = sqrt(K_new (K_new + 2 mu_d))/mu_d` along u-hat, stored as `real_t`, `dK_applied = K - K(u_new_stored)` in double from the stored values, `delta -= dK_applied`, `E_drag`, `E_lost_nopairs`, `P_drag`, `n_applied`; design off: `delta = 0` after the step); step 3 (`delta` stored as `real_t`, `pending_outstanding += w delta_stored`, `sum_wK`, `n_alive`) |

Conventions in the kernel (each listed as an open question in section 7 where
the brief leaves room): the `n_bounded` fallback clamps the loss to `f_max K`
(III.7 "an exceeded loss bound is clamped to f_max K"); `n_processed` counts
the ion-steps on which a loss was read from the table (in range or clamped
above), not the below-edge returns; the application is skipped when `delta`
is exactly 0 so that an ion with nothing pending is never rewritten (a
rewrite of `u` from `K` would move it by rounding).

### 3.3 Problem generator: `pgens/bh_proto/pgen.hpp` (838 lines) and `pgens/bh_proto/bh_proto_params.h` (541 lines)

| lines (pgen.hpp) | content |
| --- | --- |
| 81-107 | `UniformEMFields<D>`: `ex1..bx3` returning the `[setup]` vectors (used by the engine's `SetEMFields_kernel` through `init_flds`) |
| 113-125 | `splitmix64`, `hash_unit`: deterministic pseudo-random numbers from the particle counter (positions and isotropic directions; bit-reproducible on every backend, no RNG pool) |
| 131-209 | `InjectIons_kernel<D, Tracking>`: group `g = k / per_K` has kinetic energy `ions.K[g]`, `u = sqrt(K (K + 2 mu_d))/mu_d` along `+x` or isotropic, hash positions in the local active cells, `kernel::InjectParticle` with the requested weight (tracking overload when the species tracks) |
| 215-305 | checksums: order-sensitive 64-bit mix of the bit patterns (`mix_bits`: xor, multiply by the FNV prime, xor-shift) over each ion array in index order with `npart` folded in, and over the six field components of the whole `em` array including ghosts |
| 307-325 | `HostLedger::accumulate`: cumulative entries summed, snapshots overwritten per launch |
| 368-491 | constructor: strict `[setup]` read from the raw TOML (`params.data()`), `dt`, `skindepth0`, `omegaB0`, `max_steps = runtime/dt` as `engine.hpp`, `use_weights = true` required, species checks (defined, mass > 0, charge exactly +1, enough payload slots, no clearing, no spatial sorting; warning if `deposit = true`), tables built with the step-1 code path (default `Quadrature`, h = 0.02, as `bh_tables_dump`) when `hadronic.enabled`, mirrored to device, dumped to `hadronic_table.txt` with the step-1 writer, `print_summary` |
| 494-543 | `print_summary`: grid summary and, per species and initial K, `I_exact`, `I_table`, `dK/step`, `dK/K`, `ulp(u)`, `dK_ulp = ulp mu_d u/gamma`, `dK/dK_ulp`, `delta_th` (floor and `eta_u K`), steps per application, whether sub-cycling triggers |
| 545-614 | `InitPrtls`: capacity check naming the species and `maxnpart`, injection, `set_npart`, `set_counter`, `set_unsorted`, P9 zeroing, the injected `sum w K` from the stored arrays (double) |
| 658-702 | `CustomPostStep`: when the stage is active (`enabled` and `tau0 > 0`) one `Kokkos::parallel_reduce` per hadronic species over `[0, npart)` with `Kokkos::Sum<LossLedger>`; ledger and ions files every `ledger.every` steps and at the last step; checksums at the last step (`step + 1 == max_steps`) |
| 704-733 | `write_ledger`: header naming every column and unit; columns `step, time, E_drag, E_lost_nopairs, pending_discarded, pending_outstanding, P_drag_x/y/z, n_substeps, n_bounded, n_capped_loss (round 2), n_above_table, n_processed, sum_wK` (the last recomputed from the arrays in double) |
| 735-779 | `write_ions`: one block per output, rows `ion, group, K_init, K, Gamma_p, |u|, pitch (u.B/(|u||B|), 0 when B = 0), delta, tag` for the `ions.K.size() x ions.per_K` injected ions |
| 781-834 | `finish`: per-species and field checksums, the array-based closure line, the final ledger line |

`bh_proto_params.h`: `Setup` (57-86), dotted-path resolution through nested
TOML tables (`find_path`, 125-146), typed getters that refuse wrong types
(booleans must be TOML booleans; integers must be TOML integers; floats accept
integers; arrays element-wise), `parse_setup` (307-539): unknown keys refused,
every range check with a message `bh_proto: setup.<key> ... must ...`, the
`allow_unsafe_application` gate for `eta_u = 0` or `pending_floor_ulp = 0`.

Convention: the time stamp of a ledger row is the time after the step,
`(step + 1) dt` (P13; `CustomPostStep` receives t^n). Convention: output files
are written in the working directory of the run (`hadronic_ledger_<s>.txt`,
`hadronic_ions_<s>.txt`, `hadronic_table.txt`), the checksums to stdout.

Fact: the species mass is a `float` in `ParticleSpecies` (`mass()` returns
`float`), so `mu_d = 1836.15` is stored as 1836.1500244140625; the generator
uses that value as `mu_d` (printed in the ledger header). With `mu_r = 1836.15`
in double, `Gamma_p` and `gamma_d` differ by 1.3e-8 relative; irrelevant at the
tolerances here.

### 3.4 Tests

`tests/kernels/bh_loss.cpp` (1009 lines), registered as eight ctest entries
`KERNELS::bh_loss_{tables,t9,tags,above,subcycling,t6,t5,t5b}` (the
executable runs every group without arguments). Contents: `make_tables`
(mono eps0 = 0.5, mu_r = 1836.15), the RK4 reference (`rk4`, `rk4_curve`,
fixed step with the exact `at_gamma` rate or the interpolated table), the
device `Ions` set (`ParticleArrays` views with three payload slots),
`launch` (one `parallel_reduce` of the kernel), `closure_check` (T11-lite),
and the groups `test_device_tables`, `test_threshold` (T9), `test_tags`,
`test_above_table`, `test_subcycling`, `test_mu_independence_T6`,
`test_cooling_T5` (cooling curve, interpolation separation, dt-convergence
with fixed `eta_u` and with `eta_u` scaled as dt), `test_ulp_scan_T5b`.

`tests/hadronic/` (new test directory, added to `cmake/tests.cmake`):
`bh_check.cpp` (511 lines, host-only, no Kokkos: `tau0`, `t5b`, `t5b-off`,
`t7a`, `t7b`, `kick`, `order`, `t8`, `reject`), `bh_proto_params.cpp` (241
lines, 49 checks of the `[setup]` reader), the bash scripts `reject.sh`,
`t7a.sh`, `t7b.sh`, `t5b_cross.sh`, `t8.sh` with `common.sh`, registered as
`HADRONIC::bh_proto_params`, `HADRONIC::reject`, `HADRONIC::t7a_fields_B`,
`HADRONIC::t7b_fields_E`, `HADRONIC::t5b_cross_build` (skipped with code 77
when `BH_DOUBLE_ENTITY` is not given), `HADRONIC::t8_bit_identity` (skipped
without `BH_SERIAL_ENTITY`). Inputs for every run: `pgens/bh_proto/inputs/`
(`smoke.toml`, `t5b_template.toml`, `t7a_B.toml`, `t7a_B0.toml`,
`t7b_cfl{1,2,4}.toml`, `t8_{disabled,tau0zero,below}.toml`,
`reject_*.toml`). No Python anywhere in the test suite.

### 3.5 Common run configuration of the engine tests (convention)

1D Minkowski, 16 cells, extent [0, 16] (dx = 1), periodic, `larmor0 =
skindepth0 = 1` (so `omegaB0 = 1`), CFL 0.5 (dt = 0.5) unless stated, field
solver and deposit disabled (T8 keeps the solver on), one ion species with
`charge = 1`, `n_payloads_real = 3`, `deposit = false`, `clear_interval = 0`,
`spatial_sorting_interval = 0`, `particles.use_weights = true`, four ions per
energy, direction `+x`, all outputs and checkpoints disabled by `enable = false`
and intervals of 1e8 (the ADIOS2 engine refuses `format = "disabled"` at
checkpoint open, fact).

### 3.6 Core changes

None. `srpic.hpp`, the pushers, the deposit, the injectors, the statistics and
the parser are untouched. The only files outside the new directories are
`cmake/tests.cmake` (one `list(APPEND TEST_DIRECTORIES hadronic)`) and
`tests/kernels/CMakeLists.txt` (the eight `bh_loss` entries). Declaring the
payload slots and passing weights needed no core change (section 2; the
generator injects through `kernel::InjectParticle` directly with the weight
argument, as the rec_mix generator does).

## 4. Tests: measured values, tolerances, verdicts

Conventions stated before the runs (header of `bh_loss.cpp`): bath mono
eps0 = 0.5 (Gamma_th = 2.125), mu_r = 1836.15, skindepth0 = 1; "relative
error" is |K_sim - K_ref|/K_ref with K_sim from the stored `u` (double from
the `real_t` components); `K_eff = K_sim - delta` (the stored energy minus the
pending request) is reported alongside; the dt-convergence order is reported
for K_sim and K_eff against the exact-rate and the interpolated-table
references, the pre-registered verdict being K_sim against the exact rate;
the ulp ratio of T5b is `dK / dK_ulp` with `dK_ulp = ulp_float(|u|) mu_d
u/gamma_d` (section 7, item 1). Single precision unless stated.

### 4.1 Device tables (brief 1.1)

| check | measured | tolerance | verdict |
| --- | --- | --- | --- |
| device I vs `at_gamma`, 1000 log-uniform Gamma, where I > 1e-3 I_peak (424 points) | 1.175e-4 | 1e-2 | pass |
| device J vs `at_gamma`, same points | 1.338e-4 | 1e-2 | pass |
| none of the in-range Gamma flagged above | 0 | 0 | pass |
| I at the midpoints of all cells with I > 1e-3 I_peak | 1.176e-4 | 1.8e-4 + 1e-6 | pass |
| device vs host interpolation at the midpoints | 0 | 1e-12 | pass |
| 0 at Gamma_min (1 - 1e-9) and at Gamma_th | 0 | exact | pass |
| last node exactly at 1.5 Gamma_max (flagged) and at Gamma_max (not flagged) | exact | exact | pass |

Fact: the mono05 table has 466 nodes (257 + 209), Gamma_th 2.125, Gamma_min
2.13000880784, Gamma_max 91.8102230891, I_peak 1.9930346905e2, identical to
the step-1 report; the midpoint error 1.18e-4 reproduces the step-1 value
(1.18e-4) to three digits.

### 4.2 T5, cooling curve at physical mass

Setup (facts): mu_d = mu_r, tau0 = skindepth0 = 1, dt = 0.281467 so that
dK/K = 1.000e-4 per step at Gamma_p = 6 (I_table(6) = 3.261746, I_exact(6) =
3.261915), two ions at Gamma_p = 6 and two at Gamma_p = 3, eta_u = 1e-4,
pending_floor_ulp = 100. The Gamma_p = 6 ions reach Gamma_p = 3 after 43 456
steps (t_end = 12 231.4); 200 output times (every 217 steps); 17 782
applications in the base run; no sub-cycling. Reference: RK4 with step dt/8.

| quantity | Gamma_p = 6 group | Gamma_p = 3 group | tolerance | verdict |
| --- | --- | --- | --- | --- |
| max relative error of K(t) vs exact RK4 (pre-registered) | 9.281e-5 | 8.875e-5 | 1e-3 | pass |
| same with K_eff | 4.8e-5 | 3.0e-5 | reported | |
| K(t) vs interpolated-table RK4 | 9.1e-5 | 8.8e-5 | reported | |
| K_eff vs interpolated-table RK4 | 6.7e-5 | 3.5e-5 | reported | |
| table-interpolation contribution (exact RK4 vs table RK4 at the output times) | 2.2e-5 | 5e-6 | reported | |
| end values K_sim / K_ref | 3671.661 / 3671.560 | 3212.444 / 3212.517 | | |
| T11-lite closure of the base run | 2.4e-15 | 1e-9 | pass |

Convention (where the error lives): the stored K lags the requested cooling by
the pending `delta` in [0, eta_u K] (a sawtooth; its instantaneous value is
what makes K_sim differ from K_eff), the rate is evaluated at the stored
(lagging) K, and the interpolated table differs from the exact rate by up to
1.2e-4 (2.2e-5 after integration here). All three are below 1e-4; the
time-discretisation error at dK/K = 1e-4 per step is smaller than any of them
(next item).

### 4.3 T5, dt-convergence (pre-registered: order 1.0 +- 0.3 of the end-time error, K_sim vs exact RK4)

Fixed end time t_end = 12 231.4, dt, dt/2, dt/4, same eta_u = 1e-4 (the
brief's conditions). End-time relative errors (Gamma_p = 6 group unless
stated):

| dt | K_sim vs exact | K_eff vs exact | K_sim vs table | K_eff vs table | K_sim (3) vs exact | K_eff (3) vs table |
| --- | --- | --- | --- | --- | --- | --- |
| 0.281467 | 2.750e-5 | 4.784e-5 | 1.172e-5 | 6.362e-5 | 2.265e-5 | 3.548e-5 |
| 0.140734 | 2.772e-5 | 4.515e-5 | 1.194e-5 | 6.093e-5 | 6.482e-5 | 3.522e-5 |
| 0.070367 | 3.006e-5 | 4.386e-5 | 4.584e-5 | 5.964e-5 | 1.264e-5 | 3.508e-5 |

Orders (log2 of successive ratios): K_sim vs exact -0.012, -0.117
(**FAIL**, pre-registered 1.0 +- 0.3); K_eff vs exact 0.083, 0.042; K_sim vs
table -0.027, -1.94; K_eff vs table 0.062, 0.031; Gamma_p = 3 group K_sim vs
exact -1.52, +2.36.

Diagnostic run with eta_u scaled as dt (1e-4, 5e-5, 2.5e-5), everything else
equal: K_eff vs the interpolated-table reference 6.362e-5, 3.181e-5,
1.591e-5, orders **1.000019 and 0.999931** (fact). K_sim vs exact in that run:
2.750e-5, 9.4e-7, 2.4e-6 (dominated by the interpolation and the lag, not
monotone).

Reading (interpretation): the loss operator's time-splitting error is first
order and is what the scaled-eta_u diagnostic isolates; at the brief's
conditions the end-time error is dominated by three dt-independent terms,
the application lag (up to eta_u K = 1e-4 K on the stored value, roughly half
of it on average on the rate evaluation), the table interpolation (about
2e-5), and float rounding, so the order of the pre-registered quantity is
undefined (near 0). The pre-registered expectation ("first-order operator
splitting") holds for the operator but not for the measured quantity at fixed
eta_u. Recorded as a failure of the pre-registered test; see section 7, item 3.

### 4.4 T5b, single build part (`KERNELS::bh_loss_t5b`)

Gamma_p = 3 (K = 3672.3), dt = 1, tau0 chosen so that dK/dK_ulp is exactly
0.3, 1, 3, 10, 100 (achieved ratios printed: 0.3, 1, 3, 10, 100), run until K
has dropped by 1% of the reference (capped at 400 000 steps; the 0.3 case for
mu_d = 1 needs 519 882 and was capped, the double-build cross test runs it
in full). Pass criteria here (convention, the cross-build criterion is in 4.5):
design on, K_eff vs exact RK4 <= 1e-5 and K_sim vs exact RK4 <= 1e-4.

| mu_d | ratio | design on: K_sim/K_ref - 1 | K_eff/K_ref - 1 | applications | design off: K_sim/K_ref - 1 | realised / expected loss |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3 | +8.68e-5 | -2.27e-6 | 77 | +7.82e-3 (stall) | 0.0000 |
| 1 | 1 | +4.69e-5 | -2.98e-6 | 100 | -3.73e-4 | 1.0369 |
| 1 | 3 | +4.07e-5 | -2.99e-6 | 100 | -3.73e-4 | 1.0369 |
| 1 | 10 | +1.83e-5 | -2.99e-6 | 100 | -4.36e-5 | 1.0043 |
| 1 | 100 | +4.69e-5 | -3.09e-6 | 97 | +4.1e-7 | 1.0000 |
| 1836.15 | 0.3 | +4.81e-5 | -2.98e-6 | 100 | +1.010e-2 (stall) | 0.0000 |
| 1836.15 | 1 | +4.54e-5 | -2.98e-6 | 100 | -3.68e-4 | 1.0364 |
| 1836.15 | 3 | +3.38e-5 | -2.99e-6 | 100 | -3.68e-4 | 1.0364 |
| 1836.15 | 10 | +9.52e-5 | -3.01e-6 | 99 | -4.80e-5 | 1.0048 |
| 1836.15 | 100 | +1.79e-5 | -3.23e-6 | 94 | +2.1e-7 | 1.0000 |

Verdicts: all 20 design-on checks pass (K_eff within 3.3e-6 of the reference
in every case; the stored K within eta_u K). The design-off runs at ratios 0.3,
1, 3 fail the 1e-4 criterion as expected (labelled EXPECTED-FAIL before the
run); at 10 and 100 they happen to pass it. T11-lite closure passes in all 20
runs (0 or round-off).

### 4.5 T5b, cross-build (`HADRONIC::t5b_cross_build`, single vs double Entity builds)

Same conditions in the engine (dt = 0.5, 4 ions, run until the double build
has lost 1%: 519 882 steps for mu_d = 1 at ratio 0.3 down to 923 steps at
mu_d = 1836.15, ratio 100; 40 runs, 82 s in total when run alone, 352 s when
run concurrently with the kernel tests). Pre-registered pass:
design on, |K_single/K_double - 1| <= 1e-4 at the end time.

| mu_d | ratio | tau0 | steps | design on: K_single/K_double - 1 | verdict | design off: K_single/K_double - 1 | realised / expected |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3 | 2.2326e-3 | 519 882 | 7.93e-8 | pass | +1.010e-2 | 0.000002 |
| 1 | 1 | 7.4419e-3 | 155 965 | 7.87e-8 | pass | -3.731e-4 | 1.0369 |
| 1 | 3 | 2.2326e-2 | 51 989 | 1.39e-7 | pass | -3.731e-4 | 1.0369 |
| 1 | 10 | 7.4419e-2 | 15 597 | 1.23e-6 | pass | -4.415e-5 | 1.0044 |
| 1 | 100 | 0.74419 | 1 560 | 2.87e-8 | pass | +9.7e-8 | 1.0000 |
| 1836.15 | 0.3 | 3.7743e-3 | 307 520 | 1.16e-7 | pass | +1.010e-2 | 0.000001 |
| 1836.15 | 1 | 1.2581e-2 | 92 256 | 6.87e-7 | pass | -3.685e-4 | 1.0365 |
| 1836.15 | 3 | 3.7743e-2 | 30 752 | 6.53e-7 | pass | -3.685e-4 | 1.0365 |
| 1836.15 | 10 | 0.12581 | 9 226 | 2.36e-8 | pass | -4.851e-5 | 1.0048 |
| 1836.15 | 100 | 1.2581 | 923 | 8.78e-9 | pass | +6.6e-8 | 1.0000 |

The design-off table is the demonstration of why the pending design exists
(facts): at dK/dK_ulp = 0.3 the subtraction `u - du` rounds back to `u` every
step and nothing is lost (realised fraction 2e-6, the ion keeps its initial
energy while the double build lost 1%); at 1 and 3 the realised loss is biased
by +3.7% with a fixed sign (the request falls below one, respectively three,
ulps as the ion cools and I decreases, yet every step still removes a whole
ulp); at 10 the bias is +0.4%; at 100 it is at the 1e-7 level. The spec's
guess of a bias "of order +-30% at 1 to 3" is not what is measured here: the
bias is set by how far the request drifts from an integer number of ulps over
the 1% cooling: d ln I/d ln K = 7.15 at Gamma_p = 3 (measured by a centred
difference of the exact rate, R7), so a 1% fall in K lowers the request by
7.0%, and a request that starts at exactly one (three) ulps ends at 0.93
(2.79) ulps while every step still removes a whole (three whole) ulps,
averaging to the measured +3.7%.

### 4.6 T6, mu_d independence (`KERNELS::bh_loss_t6`)

mu_d in {1, 25, 1836.15}, K = 1836.15 x 5 (Gamma_p = 6), tau0 = 1, dt =
0.104143 (dK/K = 3.7e-5 per step, chosen so that delta_th / dK is not an
integer), 40 000 steps, 100 output times, design on. Final K: 4828.849 (1),
4828.646 (25), 4828.646 (1836.15), Gamma_p 3.63; applications 5884, 5884,
5885.

| pairwise max relative difference | 1 vs 25 | 1 vs 1836.15 | 25 vs 1836.15 | tolerance | verdict |
| --- | --- | --- | --- | --- | --- |
| stored K (pre-registered) | 9.9e-5 | 9.8e-5 | 9.6e-5 | 1e-5 | **FAIL** |
| K_eff = K - delta | < 1e-6 | < 1e-6 | < 1e-6 | reported | |

Mechanism (interpretation, supported by the numbers): the three runs request
identical losses; the realised `dK_applied` differs between them by the float
rounding of `u` (1e-7 relative), which shifts the step at which `delta`
crosses `delta_th` by one step on some applications, and then the stored K
differs by one application quantum, `eta_u K = 1e-4 K`, until the other run
catches up. The pre-registered parenthetical ("the only difference is the
float rounding of u at each application") is right about the requested
cooling (K_eff agrees to the rounding level) and wrong about the stored K,
whose granularity is the application quantum. Recorded as a failure of the
pre-registered test; see section 7, item 4. T11-lite closure passes for all
three runs (0).

### 4.7 T7a, uniform B perpendicular to u (`HADRONIC::t7a_fields_B`)

mu_d = 1, Gamma_p = 6 (u = 9181.75 along x), B = 100 z-hat, E = 0, omegaB0 = 1
(gyro-period 2 pi gamma/B = 577, 10.05 gyrations in runtime 5800 = 11 600
steps, Boris rotation angle 5.4e-3 per step, gyro-radius 92 cells in a 16-cell
periodic box), tau0 = 0.208288 (dK/K = 3.7e-5 per step), 116 output times,
compared with the identical run at B = 0.

| check | measured | tolerance | verdict |
| --- | --- | --- | --- |
| K(t) with B vs B = 0, all ions, every output time, stored K | 9.873e-5 | 1e-4 | pass (marginal) |
| same with K_eff | 2.25e-6 | reported | |
| pitch cosine u.B/(|u||B|), max - min over the run | 0 (exactly) | 1e-5 | pass |
| final K: 6719.640 (B) vs 6719.367 (B = 0) | | | |

Facts worth recording: the 9.87e-5 is again the application quantum (one
desynchronised application between the two runs, as in T6); with u along x and
B along z the Boris rotation keeps u_z = 0 exactly (0 x anything), so the
pitch check is exact by construction for this geometry. The T11-lite closure
line of the B run is -7.3e-6 relative (sum w (K_init - K_final) = 9844.44 vs
E_drag = 9844.51): the float Boris rotation changes |u| at the rounding
level (1.2e-7 relative per step on average over 11 600 steps), which the loss
ledger does not and should not record; the B = 0 run closes to 0 exactly.

### 4.8 T7b, uniform E parallel to u (`HADRONIC::t7b_fields_E`)

mu_d = 1, Gamma_p = 6, E = 11/32 = 0.34375 x-hat, B = 0, tau0 = 0.208288 (loss
0.34 per step at dt = 0.5, field gain omegaB0 E dt = 0.172 per step), runtime
5000, CFL 0.5, 0.25, 0.125 (10 000, 20 000, 40 000 steps), outputs every 50
time units (100 blocks). Pusher convention read from `sr.hpp:119, 345-375`
(fact): the Boris update applies two half-kicks `normalized_dt_half E` with
`normalized_dt_half = (1/2)(q/m) omegaB0 dt`, so `du = (q/m) omegaB0 E dt` per
step and the field work per unit time is `dK/dt = q omegaB0 E . beta_d`; with
q = +1 the reference is RK4 of `dK/dt = omegaB0 E beta_d(K) - (tau0/skindepth0)
I(Gamma_p(K))` from K0 = 9180.75, compared at t = (n + 1) dt (the half-step
convention t = (n + 1/2) dt is reported alongside).

| check | CFL 0.5 | CFL 0.25 | CFL 0.125 | tolerance | verdict |
| --- | --- | --- | --- | --- | --- |
| max relative error of K(t) vs RK4, stored K | 7.28e-5 | 8.78e-5 | 8.975e-5 | 1e-3 at the finest | pass |
| end-time error, stored K | 3.246e-5 | 6.845e-5 | 6.387e-5 | | |
| orders from the end-time errors | | -1.08 | +0.10 | 1.0 +- 0.3 | **FAIL** |
| end-time error with the half-step convention | 2.837e-5 | 6.640e-5 | 6.285e-5 | orders -1.23, +0.08 | reported |
| end-time error, K_eff vs interpolated-table RK4 | 4.393e-5 | 3.738e-5 | 3.460e-5 | orders 0.23, 0.11 | reported |

Reading: as in T5 the end-time error sits on a dt-independent floor (lag,
interpolation, and the rate evaluated at the lagging K), so the order of the
pre-registered quantity is undefined; the accuracy criterion passes with a
margin of 11. Recorded as a failure of the pre-registered order test.

Why E = 11/32 (fact, measured first with E = 0.34): with E = 0.34 the finest
run gave a 1.590e-3 error at the end (pre-registered 1e-3: that configuration
fails), the errors growing as dt shrinks (8.98e-5, 1.297e-3, 1.590e-3). The
field-only diagnostic (tau0 = 0, `bh_check kick`) measures the pusher's mean
du per step: 0.169921875 for a requested 0.17 at CFL 0.5 (ratio -4.6e-4;
the half-kick is 87.04 ulps of u, rounded to 87 every step), 0.04296875 for a
requested 0.0425 at CFL 0.125 (ratio +1.103e-2; 21.76 ulps rounded to 22 every
step). This is the single-precision rounding bias of the Boris kick at
u ~ 1e4 (ulp 2^-10), not the loss operator; it is the same mechanism the
pending design removes from the loss. With E = 11/32 the half-kicks are 88,
44 and 22 ulps at the three CFL values and the measured ratio is exactly 0 at
both dt. The choice isolates the loss operator; both runs are kept in the test
output (`[DIAG E = 0.34 ...]` lines). Open question 7.

### 4.9 T8, bit identity on the Serial backend (`HADRONIC::t8_bit_identity`)

Build C. mu_d = mu_r, eight ions at Gamma_p = 2.0 and 2.05 (below Gamma_min =
2.13), B = 10 z-hat, field solver on, 2000 steps, three runs: `hadronic.enabled
= false`; `tau0 = 0` (stage not launched); `tau0 = 1` (stage active, every
ion below the edge).

| check | measured | verdict |
| --- | --- | --- |
| species checksum (i1, dx1, ux1..3, weight, tag, pld_r over npart = 8) in the three runs | `1433f43866d18d21` in all three | pass |
| field checksum (six components, whole array) | `2df29ccf0d17fdce` in all three | pass |
| ledger of runs 2 and 3 identically zero (every column but step, time, sum_wK) | max |value| = 0 | pass |

Fact: run 3 prints "stage active" and launches the kernel every step; its
ledger shows `n_processed = 0` and nothing written.

### 4.10 T9, threshold (`KERNELS::bh_loss_t9`)

| check | measured | tolerance | verdict |
| --- | --- | --- | --- |
| table `gamma_th` vs 1/eps0 + eps0/4 | 0 | 1e-12 | pass |
| ions at Gamma_min (1 - 1e-6) (two, along x and y): u unchanged over 1000 steps (bitwise), delta = 0 | 0 | exact | pass |
| ions at Gamma_min (1 + 1e-3): K decreased over 1000 steps | 1.88e-3 relative (2078.777 -> 2074.866) | > 0 | pass |
| the in-range ions stop at K_floor = mu_r (Gamma_min - 1) = 2074.8656 | 5.16e-5 absolute | 2 dK_ulp = 3.87e-4 | pass |
| n_processed counts in-range ion-steps only | 8 | 2 <= n <= 2000 | pass |

tau0 for this test is chosen so that dK/K = 1e-3 per step just above the edge;
the two in-range ions reach the floor in two steps and then sit at Gamma_min,
where the float rounding of u puts Gamma_p marginally below the edge, so they
are not processed afterwards (hence 8, a fact that first surprised the test's
author, whose first version expected 2000).

### 4.11 T11-lite, closure from the arrays

For every kernel-only run above (T9, tags, above-table, sub-cycling x 3, T6
x 3, T5 base, T5b x 20): |sum w (K_init - K_final) - E_drag| / E_drag <=
1e-9. Measured: 0 in most runs, 1.2e-13 (T9), 2.4e-15 (T5), 8.7e-16 and
1.4e-16 (sub-cycling), 1.1e-14 (above-table). All pass. The same identity in
the engine runs (printed at the end of every run as `bh_proto: closure`):
5.5e-14 (smoke), 0 (T7a with B = 0, T5b runs); not applicable when the
fields change K (T7a with B, section 4.7; T7b).

### 4.12 Sub-cycling (`KERNELS::bh_loss_subcycling`)

tau0 such that dK = 0.3 K per step at Gamma_p = 6 (mu_d = mu_r), f_max = 0.1,
60 steps, reference RK4 with step dt/256.

| check | measured | tolerance | verdict |
| --- | --- | --- | --- |
| n_substeps > 0 (max_substeps = 64) | 16 | > 0 | pass |
| n_bounded = 0 with max_substeps = 64 | 0 | 0 | pass |
| max relative error of K(t) vs exact RK4 over 60 steps | 2.358e-2 | 1e-2 | **FAIL** |
| with max_substeps = 2: n_bounded > 0 | 4 | > 0 | pass |
| with max_substeps = 2: the run continues (finite, decreasing K) | K = 2881.3 after 60 steps | | pass |

Facts: the first step (3 RK2 substeps) has an error of 1.26e-3 (K 7245.99 vs
7236.87); the maximum 2.36e-2 is reached later, when dK/K per step has fallen
below f_max = 0.1 and the steps are no longer sub-cycled: an Euler step with
dK/K up to 0.1 has a relative error of order (1/2)(dK/K)^2 d ln I/d ln K,
about 1e-2 per step, and a few such steps accumulate. Diagnostic with f_max =
0.01 (everything else equal): max error 2.017e-3, n_substeps 250. With
max_substeps = 2 the clamped loss gives a 0.20 error (reported only). The
pre-registered 1e-2 at f_max = 0.1 is not met by the design "RK2 only when
dK > f_max K, Euler otherwise". Recorded as a failure; open question 5.

### 4.13 Above the table (`KERNELS::bh_loss_above`)

Three ions at Gamma_p = 1.1 Gamma_max = 101.0 with weights 1, 2, 0.5, 20 steps
with dK/K = 1e-4 per step (eta_u = 1e-5 so every step is applied).

| check | measured | tolerance | verdict |
| --- | --- | --- | --- |
| n_above_table counts every ion every step | 60 | 3 x 20 | pass |
| E_drag + pending equals 20 steps of the edge-value loss sum w dK | 3.18e-13 relative | 1e-12 | pass |
| final Gamma_p still above the edge | 100.79 | | |

### 4.14 Tag classes (`KERNELS::bh_loss_tags`)

Two ions below the application threshold: five requests accumulate without
an application (delta = 0.0459, n_applied = 0); `pending_outstanding` equals
sum w delta (1.8e-1, to 1e-12); after tagging ion 1 dead, one launch discards
w delta = 0.1377 into `pending_discarded`, zeroes its payload, and counts
n_alive = 1, n_processed = 1. All pass.

### 4.15 Rejection tests (`HADRONIC::bh_proto_params`, `HADRONIC::reject`)

`bh_proto_params`: 49 checks in round 1 (58 in round 2, section 10.1), all
passed: five positive cases (the valid setup parses with the documented
values; defaults; the disabled stage needs only the ion keys; a TOML integer is
accepted for a float key; design off with the flag) and 44 refusals (53 in
round 2), each with a message naming `setup.<key>`: negative tau0, tau0 as a
string, mu_r = 0 and negative, eps0 >= 2 (mono), eps_max >= 2 (powerlaw and
blackbody), f_max = 0 and > 1, max_substeps = 0 and non-integer, unknown and
missing photon_field.type, allow_unsafe_application absent or false with
eta_u = 0 or pending_floor_ulp = 0, ions.K empty, non-positive, negative, not
an array, missing, containing a string, non-boolean hadronic.enabled (integer
and string) and allow_unsafe_application, missing hadronic.enabled,
hadronic.species empty, index 0, duplicate, ions.per_K = 0, unknown
ions.direction, ions.weight = 0, fields.B of wrong length, ledger.every = 0,
negative eta_u and pending_floor_ulp, gamma_p_max <= 1 and a bad string,
non-distinct payload slots, an unknown key and an unknown nested key, missing
tau0 and mu_r when enabled, a table bath without a file.

`reject`: `entity.xc` exits with code 1 and the message names the key for
`reject_tau0`, `reject_fmax`, `reject_bool`, `reject_unsafe`, `reject_type`,
`reject_ionsK` (`setup.<key>`), `reject_payloads` (`n_payloads_real`) and
`reject_weights` (`use_weights`). 8 of 8 pass.

### 4.16 Summary of verdicts

| test | verdict |
| --- | --- |
| device tables (7 checks) | pass |
| T5 cooling curve, both groups | pass (9.3e-5, 8.9e-5 vs 1e-3) |
| T5 dt-convergence order (pre-registered quantity) | **fail** (order ~ 0; order 1.000 when eta_u scales with dt and the lag and interpolation are removed) |
| T5b single-build, design on (20 checks) | pass |
| T5b single-build, design off (6 of 10 fail the 1e-4 criterion) | expected failures, tabulated |
| T5b cross-build, design on (10 runs) | pass (<= 1.2e-6 vs 1e-4) |
| T5b cross-build, design off | expected failures, tabulated |
| T6 stored K | **fail** (9.9e-5 vs 1e-5; K_eff < 1e-6) |
| T7a | pass (9.87e-5 vs 1e-4; pitch exact) |
| T7b accuracy | pass (9.0e-5 vs 1e-3) |
| T7b order | **fail** (orders -1.1, +0.1) |
| T8 | pass |
| T9 (5 checks) | pass |
| T11-lite (all kernel-only runs) | pass |
| sub-cycling counters and continuation | pass |
| sub-cycling accuracy | **fail** (2.36e-2 vs 1e-2; 2.0e-3 at f_max = 0.01) |
| above the table | pass |
| tag classes | pass |
| rejections (49 parser checks incl. 5 positive cases, 8 end to end; round 2: 58 + 9) | pass |

ctest in build A: see section 9 for the round-1 counts (the stock 53 tests
unchanged and passing). Round 2: the four round-1 failures are re-registered
or re-run at the new default and pass (section 10); both suites complete with
no failure (10.3).

## 5. The T5b design-off table

Section 4.5 (cross-build, the pre-registered form) and 4.4 (single-build
against the double RK4 reference) carry the two tables. Summary of the
measured facts: at dK/dK_ulp = 0.3 the stored momentum never changes (realised
loss 2e-6 of the expected 1%); at 1 and 3 the realised loss is 1.0369
(mu_d = 1) and 1.0365 (mu_d = 1836.15) of the expected; at 10 it is 1.0044 /
1.0048; at 100 it is 1.0000. With the pending design on, the same runs agree
with the double build to <= 1.2e-6 and with the RK4 reference to <= 3.3e-6
(K_eff).

## 6. dt-convergence orders

| experiment | quantity | order dt -> dt/2 | order dt/2 -> dt/4 |
| --- | --- | --- | --- |
| T5, eta_u fixed (pre-registered) | K_sim vs exact RK4, Gamma_p = 6 | -0.012 | -0.117 |
| T5, eta_u fixed | K_eff vs exact RK4 | 0.083 | 0.042 |
| T5, eta_u fixed | K_eff vs table RK4 | 0.062 | 0.031 |
| T5, eta_u scaled as dt (diagnostic) | K_eff vs table RK4 | 1.000019 | 0.999931 |
| T7b, E = 11/32 (pre-registered) | K_sim vs exact RK4 | -1.076 | +0.100 |
| T7b | K_sim, half-step convention | -1.227 | +0.080 |
| T7b | K_eff vs table RK4 | 0.233 | 0.112 |

## 7. Ambiguities, conventions adopted, open questions

1. **The ulp ratio of T5b (convention).** The brief writes "dK/ulp(u)", whose
   units match only for mu_d = 1 and u >> 1. The ratio used everywhere here is
   `dK / dK_ulp` with `dK_ulp = ulp_float(|u|) mu_d u / gamma_d`, the energy
   resolution of the stored momentum (the quantity the pending floor is built
   from); it equals dK/ulp(u) for mu_d = 1, u >> 1, and is what the stall and
   bias depend on at mu_d = 1836.15 (where dK/ulp(u) in the literal sense would
   be 1700 times larger). The ratio is defined on the single-precision ulp
   regardless of the build (the double build runs the same tau0).
2. **Which K is compared (convention).** The brief's "K_sim" is the stored
   value; K_eff = K - delta is reported alongside in every test because the
   stored K carries the application quantum eta_u K = 1e-4 K as granularity.
   The pending delta appears in no conservation identity (spec III.3); using
   it to define a smoother diagnostic energy does not change that.
3. **T5 and T7b order tests (open question for the spec).** At fixed eta_u
   the end-time error of the stored K has a dt-independent floor (lag,
   interpolation, stale rate) that exceeds the first-order splitting error at
   dK/K <= 1e-4 per step, so "order 1.0 +- 0.3" is not a property of the
   quantity the brief names. The operator is first order (orders 1.000019 and
   0.999931 with eta_u scaled as dt against the interpolated-table reference).
   Decided 2026-10-25 (section 10.2): the order is judged on K_eff against
   the interpolated-table reference with eta_u scaled as dt.
4. **T6 on the stored K (open question for the spec).** The application
   quantum makes pairwise agreement to 1e-5 of the stored K impossible by
   design when eta_u = 1e-4; K_eff agrees to < 1e-6 (the brief's
   parenthetical). Decided 2026-10-25 (section 10.2): T6 is judged on K_eff
   at 1e-6.
5. **Sub-cycling accuracy (open question for the spec).** With f_max = 0.1
   the un-sub-cycled Euler steps with dK/K in (0.03, 0.1] dominate the error
   (2.4e-2 measured against 1e-2). Lowering f_max to 0.01 brings it to 2.0e-3
   at a cost of 250 instead of 16 substeps over the 60 steps. Decided
   2026-10-25 (section 10.2): f_max defaults to 0.01.
6. **The n_bounded fallback (convention).** When `ceil(dK/(f_max K))` exceeds
   `max_substeps`, the RK2 integration runs with `max_substeps` substeps and
   the resulting loss is clamped to `f_max K` (III.7 "clamped to f_max K");
   the brief's "apply the capped value" was read this way. The alternative
   (apply the max_substeps RK2 result unclamped) would give 0.26 K instead of
   0.1 K in the test; the choice follows the failure policy's "safe fallback".
   Decided 2026-10-25 (section 10.2): this reading is the specified one, with
   the new counter `n_capped_loss`.
7. **The pusher's own single-precision rounding bias (fact, outside this
   module).** A uniform E whose half-kick is not an integer number of ulps of
   u biases the field work systematically (+1.1% at 21.76 ulps, measured);
   at mu_d = 1, Gamma_p = 6 the float u ~ 1e4 has ulp 1e-3. T7b uses E = 11/32
   to isolate the loss operator. This affects any float PIC run with a steady
   field at large u, is independent of the hadronic module, and is the same
   mechanism the pending design addresses for the loss; recorded for the
   project, no action taken here.
8. **T7a geometry (fact).** With u along x and B along z the pitch cosine is
   exactly zero and stays so in the Boris rotation (0 x anything); the
   pre-registered pitch check is therefore exact by construction and does not
   probe an oblique case. An oblique diagnostic was not added because the
   float Boris rotation's own drift of u.B over 1e4 steps would be of order
   1e-5, confounding the 1e-5 criterion.
9. **Species mass is float (fact).** mu_d = 1836.15 is 1836.1500244 in the
   species; recorded in the ledger header; no effect at the tolerances here.
10. **n_processed (convention).** Counts ion-steps on which a table loss was
    read (in range or clamped above); below-edge returns and dead ions are not
    counted, so the T8 ledger of a stage that is launched on below-edge ions
    is identically zero.
11. **Table provenance (fact).** `hadronic_table.txt` is written by the
    step-1 writer with `status = cpp_candidate` and `version` equal to the
    configure-time git hash of the build (here `a346254`, the configure of
    build A predates the commits of this step); the table bytes are those of
    the mono05 candidate of step 1 (same code path, same quadrature).
12. **Brief versus spec.** No disagreement was found between the brief and the
    spec excerpts for the loss-only subset. The brief's T6 and order
    expectations (items 3, 4) are expectations about outcomes, not conflicting
    requirements; they are reported as measured.

## 8. Not done

- Debt, packets, pair sources, RNG, the latency cap, the startup estimate and
  the failure-policy flags: out of this step (hooks present as zero ledger
  fields and the three payload indices).
- Non-uniform bath profiles (`g(x)`): only `uniform` is accepted (`profile`
  selector present).
- The production placement (`srpic::HadronicInteractions`), parser plumbing,
  schema and `input.default.toml` entries: excluded by the brief.
- Multi-rank, multi-domain, GPU, clang and MPI builds: not attempted (the
  generator refuses multi-domain runs).
- `CustomStat` entries: the prototype writes its own ledger file (P14).
- Powerlaw, blackbody and table baths are accepted by the generator (same
  parser as `bh_tables_dump`) but every test here uses the mono eps0 = 0.5
  bath.

## 9. Reproduction

```
# Kokkos (Serial only) for T8
cmake -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 -D Kokkos_ENABLE_SERIAL=ON \
      -D Kokkos_ENABLE_OPENMP=OFF -D CMAKE_INSTALL_PREFIX=$HOME/.entity/kokkos-serial $HOME/.entity/src/kokkos
ninja install
# builds B and C (entity.xc only)
cmake -G Ninja -D TESTS=OFF -D output=ON -D precision=double -D mpi=OFF -D pgen=bh_proto \
      -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos;$HOME/.entity/adios2" -S . -B build-double && ninja -C build-double entity.xc
cmake -G Ninja -D TESTS=OFF -D output=ON -D precision=single -D mpi=OFF -D pgen=bh_proto \
      -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos-serial;$HOME/.entity/adios2" -S . -B build-serial && ninja -C build-serial entity.xc
# build A with the tests
cmake -G Ninja -D TESTS=ON -D output=ON -D precision=single -D mpi=OFF -D pgen=bh_proto \
      -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos;$HOME/.entity/adios2" \
      -D BH_DOUBLE_ENTITY=$PWD/build-double/src/entity.xc -D BH_SERIAL_ENTITY=$PWD/build-serial/src/entity.xc \
      -S . -B build-single && ninja -C build-single
cd build-single && ctest -R "KERNELS::bh_loss|HADRONIC|ARCHETYPES::bh_" --output-on-failure
# round 2: f_max defaults to 0.01 in the generator and in every input under pgens/bh_proto/inputs/;
# the double suite: cmake -D TESTS=ON in build-double, ninja, ctest (its cross-build and T8 entries skip)
# UBSan diagnostic: cmake -D CMAKE_CXX_FLAGS="-fsanitize=undefined -fno-sanitize-recover=undefined -g" ... -B build-ubsan
#   && ninja -C build-ubsan test-kernels-bh_loss.xc && build-ubsan/tests/kernels/test-kernels-bh_loss.xc
./tests/kernels/test-kernels-bh_loss.xc            # all groups, full printout
./src/entity.xc -input ../pgens/bh_proto/inputs/smoke.toml
```

Final ctest counts of build A: 67 tests (the 53 stock tests plus 14 new),
63 passed, 4 failed, 114 s wall with `ctest -j1` (`T5b cross-build` 82 s when
run alone). The four failures are the pre-registered items reported in section
4: `KERNELS::bh_loss_subcycling` (4.12), `KERNELS::bh_loss_t6` (4.6),
`KERNELS::bh_loss_t5` (4.3, the dt-order; the cooling-curve checks of the same
group pass) and `HADRONIC::t7b_fields_E` (4.8, the order; the accuracy check of
the same script passes). Every stock test passes unchanged.

## 10. Round 2 (2026-10-25): referee resolution log, re-registered tests, fixture test

### 10.1 Referee findings

| finding | status | resolution |
| --- | --- | --- |
| R1 (blocking): integers narrowed before validation | accepted, fixed | `bh_proto_params.h`: a range-checked getter (`get_int_in`) validates the 64-bit TOML value against the destination range before any cast: payload slots in [0, 65534] (the species slot count is an `unsigned short`, so max(index) + 1 fits without overflow), `ledger.every` in [1, INT_MAX], `max_substeps` in [1, 1e6], `ions.per_K` in [1, INT_MAX / ions.K.size()] (the injected count fits `npart_t`); the distinctness check runs on the validated values. Regressions (all refused naming the key, fact): `payload_pending = 4294967295`, `payload_pending = 4294967296` with `payload_debt = 0`, `ledger.every = 4294967296`, `ledger.every = 2147483648`, `payload_debt_k0 = 65535`, `ions.per_K = 2147483648`, `ions.per_K = 1073741824` with two energies, `max_substeps = 4294967296`; `payload_debt_k0 = 65534` and `ledger.every = 2147483647` accepted. `bh_proto_params`: 58 checks, 58 passed. |
| R2 (blocking): `ceil(dK/(f_max K))` converted to int before the cap | accepted, fixed | `bethe_heitler.hpp` step 1: the request is tested for finiteness; the substep count is compared with `max_substeps` in double and converted only when it fits; a non-finite request takes the capped path (n = max_substeps, loss capped at f_max K), and a non-finite integrated loss is replaced by K - K_floor before the cap. Regressions (`KERNELS::bh_loss_overflow`, fact): coefficient 1e20 (3.6e17 substeps requested) and coefficient 1e308 with dt 1e10 (non-finite request) both give a finite result, n_bounded = 1, n_capped_loss = 1, n_substeps = 64, applied loss = f_max K to 2e-6 relative. Sanitizer diagnostic: section 10.4. |
| R3 (should fix): GCA pusher on a hadronic species | accepted, fixed | `pgen.hpp` constructor refuses `(pusher & GCA) != 0` naming the species and the pusher string; `reject_gca.toml` (`pusher = "Boris,GCA"`) added to `HADRONIC::reject` (refused with exit 1, message names GCA, fact). |
| R4 (should fix): ulp floor from a `real_t` norm | accepted, fixed | the floor uses the double norm `u` rounded to `real_t`, with a representability guard (a norm beyond the largest finite `real_t` takes the ulp of that value; `ulp_real` is also guarded for non-finite input). Regression (`KERNELS::bh_loss_huge`, the referee's probe mu_d = 1, K = 1e20, coefficient 1e10, dt = 1, eta_u = 1e-12, pending_floor_ulp = 100): in single precision the floor is 8.80e14, above eta_u K = 1.0e8 and above the ten accumulated requests (1.99e13), and no application happens in 10 steps (n_applied = 0, fact); in the double build the floor is finite (1.6e6) but below eta_u K, so the first request is applied; the registered statement is "applications happen only once max(pending_floor, eta_u K) is reached, with a finite floor", checked in both builds. |
| R6 (should fix): T5b unit test in the double build | accepted, fixed | `dK_ulp_of` uses the single-precision ulp whatever the build (`ulp_float_of`), so the double build runs the same ten cases (153 s under load, section 10.3). The double suite's cross-build entry is skipped (no `BH_DOUBLE_ENTITY` given there) and is not counted as a validation; the only cross-precision validation is the single suite's `HADRONIC::t5b_cross_build` against build B. |
| R7 (note): d ln I/d ln K at Gamma_p = 3; check counts | accepted, corrected | measured by centred difference: d ln I/d ln K = 7.1527 at Gamma_p = 3 and I(0.99 K)/I(K) - 1 = -7.00% (printed by the unit test as a (diag) line); section 4.5 corrected; the parser test has 49 checks of which 5 are positive cases (44 refusals), now 58 checks (six positive cases and 52 refusals; corrected 2026-10-25 from "five and 53"); sections 4.15 and 4.16 corrected. |
| R8 (project note): float pusher steady-E bias | no action in this branch | T7b keeps E = 11/32 (integer ulps); the E = 0.34 field-only diagnostic stays in the test output. |

Deferred: nothing from this list.

### 10.2 Decisions of 2026-10-25 and the re-registered results (single build unless stated)

(a) Re-registered tests on K_eff.

| test | criterion | measured | verdict |
| --- | --- | --- | --- |
| T5 order (re-registered) | K_eff vs the interpolated-table RK4, eta_u = 1e-4, 5e-5, 2.5e-5 at dt, dt/2, dt/4, both ratios within 1.0 +- 0.3 | Gamma_p = 6 group: 1.000019, 0.999931; Gamma_p = 3 group: 0.999771, 0.999179 | pass |
| T5 order: eta_u K above the pending floor at every dt used | ratio > 1 | at the end of the runs: 8.90, 4.45, 2.22 (6) and 7.88, 3.94, 1.97 (3) | pass (min 1.97) |
| T5 accuracy (unchanged) | stored K vs exact RK4, both groups, <= 1e-3 | 9.281e-5, 8.875e-5 | pass |
| T5 interpolation separation (unchanged) | reported | exact vs table RK4 at the output times 2.2e-5 (6), 5e-6 (3) | reported |
| T6 (re-registered) | pairwise K_eff over mu_d in {1, 25, 1836.15}, <= 1e-6 at every output time | 1.85e-8, 5.05e-8, 5.31e-8 (max 5.31e-8) | pass |
| T7b accuracy (unchanged) | stored K vs exact RK4 at the finest dt, <= 1e-3 | 2.921e-5 (the eta_u-scaled run at CFL 0.125) | pass |
| T7b order (re-registered) | K_eff vs the interpolated-table RK4, eta_u = 1e-4, 5e-5, 2.5e-5 (inputs `t7b_cfl{1,2,4}.toml`) | end errors 4.393e-5, 2.199e-5, 1.105e-5; orders 0.9984, 0.9931 | pass |
| T7b: eta_u K above the floor | ratio > 1 at every dt | 16.59, 8.29, 4.15 | pass |
| stored-K envelope, every curve | max over output times of |K_stored - K_eff| / (eta_u K + pending_floor) <= 1 | T5 base 0.904 (6), 0.891 (3); T5 scaled runs 0.677, 0.246, 0.071 (6) and 0.069, 0.742, 0.527 (3); T6 0.933, 0.923, 0.892; T7a 0.737 (B), 0.933 (B = 0); T7b 0.665, 0.609, 0.727 | pass (14 checks) |
| T11-lite closure (unchanged) | <= 1e-9 on every kernel-only run | round-off (0 to 2.4e-15) | pass |

Round-1 criteria kept as history, printed with the `(diag)` label (facts, unchanged numbers): T5 fixed-eta_u stored-K orders -0.012, -0.117 (would fail 1.0 +- 0.3); T6 stored-K pairwise 9.93e-5, 9.82e-5, 9.55e-5 (would fail 1e-5); T7b fixed-eta_u stored-K end errors 3.246e-5, 6.845e-5, 6.387e-5, orders -1.08, +0.10 (would fail); T7b stored K in the eta_u-scaled runs: 3.246e-5, 4.376e-6, 2.860e-5 (orders 2.89, -2.71, the lag and interpolation floor, reported only).

(b) f_max default 0.01: `bh_proto_params.h` default, every input under `pgens/bh_proto/inputs/`, `default_params` of the unit test, and the reproduction notes. Sub-cycling accuracy at the new default (registered): max relative error vs exact RK4 over 60 steps 2.017e-3 (tolerance 1e-2, pass), n_substeps 250; the f_max = 0.1 result stays as (diag): 2.358e-2 (would fail), n_substeps 16. The T5, T6, T7 runs never trigger sub-cycling (dK/K <= 1e-4 per step), so their numbers are unchanged by the default.

(c) Clamp reading: when ceil(dK/(f_max K)) > max_substeps the kernel integrates max_substeps RK2 substeps, caps the applied loss at f_max K, counts the exhaustion in `n_bounded` and the steps on which the cap changed the value in the new counter `n_capped_loss`; both are ledger columns (15 columns now; the check tool and T8's zero-ledger check read them). With max_substeps = 2 and f_max = 0.01 over 60 steps: n_bounded = 120, n_capped_loss = 120 (two ions, every step), the run continues (final K 5023, fact); the R2 regressions count 1 and 1.

### 10.3 Suites

Single build (A): 70 tests, 70 passed, 134 s wall with `ctest -j1`
on an otherwise idle machine (`t5b_cross_build` 88 s). The 53 stock tests are
unchanged and pass; the 17 new entries are `KERNELS::bh_loss_{tables, overflow,
huge, t9, tags, above, subcycling, t6, t5, t5b}`, `ARCHETYPES::bh_fixtures`,
`HADRONIC::{bh_proto_params, reject, t7a_fields_B, t7b_fields_E,
t5b_cross_build, t8_bit_identity}`. Running the kernel unit executable with
every group: 101 checks passed, 0 failed, 6 expected failures (the T5b
design-off runs).

Double build (B, `-D TESTS=ON`): 70 tests, 68 passed, 2 skipped
(`t5b_cross_build` and `t8_bit_identity`, no second build given to the double
tree; not counted as validations), 0 failed, 42.6 s wall. The suite completes
(R6): `bh_loss_t5b` takes 16.9 s. Diagnostic values from the double build
(facts): T5 re-registered orders 1.000003, 0.999991 (Gamma_p = 6) and 1.000122,
1.000038 (Gamma_p = 3), with eta_u K / pending_floor of order 1e9 (the double
ulp floor is 9e-11 at K ~ 8e3); T5 accuracy 8.562e-5 and 9.326e-5; T6 K_eff
pairwise 1.8e-16 (the stored K agrees to 5.7e-16 too, since the application
quantum is reached on the same step in every run at double resolution); T7a
K(B) vs K(0) 4.4e-15; T7b accuracy 2.131e-5, re-registered orders 1.0001,
1.0000 (end errors 4.399e-5, 2.199e-5, 1.100e-5), stored-K orders in the same
runs -0.59, 0.90 and in the fixed-eta_u runs 0.42, -1.52 (the lag floor,
reported only); sub-cycling at f_max = 0.01: 2.017e-3; R4 probe: floor 1.6e6,
applications from the first step as registered. The round-1 history lines
print "would pass" for T6 on the stored K in the double build and "would fail"
for the T5 fixed-eta_u order (-1.47, 2.01) and for f_max = 0.1 (2.34e-2).

### 10.4 Sanitizer diagnostic (build D, `-fsanitize=undefined -fno-sanitize-recover=undefined`)

Build D compiles the kernel unit test and every library it links with
`-fsanitize=undefined -fno-sanitize-recover=undefined -g` (231 compile commands
carry the flag; the binary links `libubsan.so.1`, facts). Running every group
of `test-kernels-bh_loss.xc` under the sanitizer: 101 checks passed, 0 failed,
6 expected failures (the T5b design-off runs), exit code 0, and **no "runtime
error" line** in the output, including the R2 regressions (coefficient 1e20
and the non-finite request) that exercise the previously undefined
double-to-int conversion. Before the fix the same input would convert 3.6e17 to
`int` (undefined; the referee's UBSan report). The sanitizer build is a
diagnostic and is not registered in ctest.

### 10.5 Fixture test (`ARCHETYPES::bh_fixtures`, contract 1.4 item 8)

`tests/archetypes/bh_fixtures.cpp`, registered next to `bh_physics` with `BH_TEST_DATA_DIR`. For each of the five `bhref` 0.1.0 files: the file hash matches `SHA256SUMS` and the reader verifies `data_sha256` (producer `bhref`, status `frozen:0.1.0`); the bath is rebuilt from the header's `photon_field` object (mono eps0 0.5 and 0.3, powerlaw [0.1, 1.5] index 2, blackbody theta 0.1 on [0, 1.9], mono 0.5 with mu_r = 100); `build_tables` with the default quadrature reproduces the reference node set (same count, max |dy| below 4.5e-13 for every file, tolerance 1e-9); `at_gamma` at every reference `gamma_p`:

| file | nodes | core nodes (I_ref > 1e-3 I_peak_ref) | max |dJ/J| core | max |dI/I| core | tail nodes (> 1e-9) | max |dJ/J| tail | max |dI/I| tail | over all nodes (J, I) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mono05 | 466 | 120 | 9.64e-13 | 1.01e-12 | 273 | 5.92e-11 | 5.95e-11 | 3.84e-10, 3.84e-10 |
| mono03 | 465 | 119 | 8.22e-13 | 1.11e-12 | 272 | 3.82e-11 | 3.85e-11 | 1.96e-10, 1.97e-10 |
| powerlaw | 429 | 88 | 6.22e-13 | 9.35e-13 | 264 | 7.52e-11 | 7.83e-11 | 3.34e-10, 3.38e-10 |
| blackbody | 406 | 71 | 8.78e-13 | 1.10e-12 | 190 | 1.75e-11 | 2.02e-11 | 1.12e-10, 1.28e-10 |
| mono05_mu100 | 403 | 64 | 4.10e-12 | 4.30e-12 | 210 | 1.94e-10 | 1.94e-10 | 1.07e-9, 1.07e-9 |

Tolerances 1e-6 (core) and 1e-3 (tail): 35 checks, 35 passed. <D> agrees to 1.6e-11 or better and eps_mean to 9.2e-13 or better over all nodes (diag). The referee's expectation (core 1e-12, tail 2e-10) is reproduced. `tests/data/hadronic/cpp-candidate/README.md` now states that this test is the registered item 8.
