# P2 step 3, pair plumbing with the mean-energy spectrum: debt, packets, two pair species, the three sources, the exact ledger. Report

Claude 1 (Claude Code cloud session), 2026-10-25. Fork `mancininilay/entity`,
branch `dev/hadronic-pairs` (created from `dev/hadronic` at `ed98e1f`).
Commits: `50b1f3d` (step-2 closure housekeeping), `9b39f3f` (core P9),
`e628d08` (kernel and generator), `71fd517` (core P5), `46cb34f` (tests),
plus the report and result commits listed in section 11. Line numbers refer
to the HEAD named in the final message.

Every statement below is labelled as a **fact** (measured, with the number),
a **convention** adopted here, or an **open question**. No pre-registered
tolerance was adjusted; the one pre-registered item that fails is labelled as
an expected failure before it runs (section 4.5) and reported with its number
and mechanism. Implementation-chosen checks are marked `(impl)`; `(diag)`
lines are reported only.

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
3. **Poisson and the identity (open question).** The brief asks the counter
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
7. **The K0 trigger pays the whole debt** (r_packet = r), as III.4; after it
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

(filled in sections 4.1 to 4.12 from the runs; see the final message for the
suite counts)

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

## 6. Bounds-check diagnostic (build D)

(filled from the run)

## 7. Deferred to Rusty (not attempted here, as the brief requires)

T14 (multi-rank: send-tagged ions, pair insertion under MPI, the receive
capacity error), T15b (GPU: CUDA build of the kernel, `Kokkos::fma`,
`atomic_fetch_add` on the device counter, the `dd` reducer on the device),
T14b-lite (the production placement `srpic::HadronicInteractions` and the
double-typed stats entries), T-perf. The reasons: no MPI or CUDA build exists
in this container, and the brief excludes the production placement.

## 8. Open questions

(filled)

## 9. Not done

(filled)

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
# bounds-check diagnostic (build D)
cmake -G Ninja -D TESTS=ON -D output=ON -D precision=single -D mpi=OFF -D pgen=bh_proto -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos-debug;$HOME/.entity/adios2" -S . -B build-bounds \
      && ninja -C build-bounds entity.xc test-kernels-bh_pairs.xc bh_check && build-bounds/tests/kernels/test-kernels-bh_pairs.xc t12 \
      && (cd /tmp/w && build-bounds/src/entity.xc -input pgens/bh_proto/inputs/t12_capacity_warn.toml)
```
