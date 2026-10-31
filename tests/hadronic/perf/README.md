# Scaling harness of the Bethe-Heitler stage (P2 step 7)

Contract: `Resources/p2-hadronic/notes/2026-10-30-claude1-scaling-harness-contract.md` (revision 1.0). Measurement
code only: nothing here changes the module's production source (`src/kernels/hadronic/`, `src/archetypes/hadronic/`,
`pgens/bh_proto/`). Author Claude 1; referee Codex 1 before the Rusty job is authorised.

| measurement | where | how |
| --- | --- | --- |
| M1 stage cost per ion-step vs N_ion, four modes | `perf_driver.sh` + `inputs/` | `entity.xc` runs of `bh_proto` at the production placement; Entity's `Custom` timer (the hadronic stage in `srpic.hpp` plus the generator's `CustomPostStep`, a measured proxy for the stage) summed over all `PERF_MEASURED` steps (`PERF_DIAG = 1`: every step printed, no phase sampling; a run whose block count or values are wrong is a failed run); one CSV row per (mode, N, variant, repeat), medians over repeats. `Timestep duration` is a running average over up to 1000 steps (`progressbar.h`) and is reported as `progress_avg_ms`, never as the step cost; the whole-step cost is the block's `Total` (`total_ms_per_step`) |
| M2 cold ions | same, variant `cold` | ions at Γ_p = 2 (below the table): the early-exit cost |
| M3 lookup breakdown | `tests/kernels/bh_perf.cpp variants`, `tests/kernels/perf/` | the production kernel against a generated templated copy with (i) constants in place of the lookup, (ii) float tables with linear interpolation, under MATCHED WORK (the ion state restored and the RNG pool re-seeded before every launch, identically for every kernel; the residual work mismatch printed); nsys/ncu on the A100 at 2²² (fixed), one capture window per configuration (`--mode`, `--kernel`) |
| M4 reducer vs atomics | `tests/kernels/bh_reduction_timing.cpp` | double-double reducer vs plain atomics vs per-team scratch plus one atomic per team |
| M5 pusher reference | `perf_driver.sh` M5 runs | `ParticlePusher` and `CurrentDeposit` timers with deposit on: hadronic off (the PIC floor), poisson a = 1 (pair-loaded), budget |
| M6 stride | `tests/kernels/bh_perf.cpp stride` | the stage every n steps with dt_BH = n dt (n = 1, 4, 16) at fixed physical time; ⟨K⟩ against the ODE (event: Jensen-corrected, 4 s.e.), identity, wall time |

M3 attribution method (round 2, Codex 1 F6). The timed interval of a variant row is the `parallel_reduce` of the
kernel (the body and its ledger reduction are one Kokkos call; the reducer returns on the host) plus the fence;
`pairs.finish()` (the spawn-counter `deep_copy`) is timed separately (column "finish ms"); the production stage adds
the generator's post-step (ledger accumulation), measured by the engine's `Custom` timer in M1, not here. On the A100
the breakdown is read from two profiles per configuration: (a) nsys with NVTX ranges `<launch label>` (kernel +
reduction + fence), `bh_perf_finish_readback` and `bh_perf_restore_and_reseed` (outside the timed interval), from
which the kernel time and the reduction-plus-readback tail are the CUDA kernel rows and the gap to the range end;
(b) ncu on 12 launches after the 20 warm-up launches (`--kernel-name-base demangled --kernel-name regex:BetheHeitler`,
the Kokkos CUDA function names carry the functor type), for the kernel's own metrics (`--set basic`). The RNG pool's
share cannot be ranged inside the kernel: it is bounded from above by `bh_perf rngpool`, a kernel that only acquires a
pool state, draws once and releases it over the same N threads (its time per thread against the production kernel's
ns per ion). The lookup fraction is (production − constants) / production under matched work, with the stated
limitation that the constants variant removes the lookup's memory traffic and arithmetic together and the float
variant changes the interpolation coordinate with the precision. The job asserts `BH_PERF_N = 4194304` in every
profiled log and that each profile holds the expected ranges and launches.

Driver notes (facts found on the Mac smoke, 2026-10-30): Entity's `Custom` timer also wraps the generator's
post-step, whose ledger row and per-ion text dump are O(N) host passes; the driver therefore writes the ledger row
once, at the last step, and excludes the final diagnostics block (which also carries `finish()`) from the measured
mean. The budget source's auto latency cap (`t_latency = 6.4` at these parameters) emits a packet per ion every 13
steps, 46 per ion over 600 steps; the perf budget input sets `t_latency = 0` (packets at the request threshold
only) and `on_capacity_estimate = "warn"` (the runtime capacity policy stays `error`). Species counts in the
diagnostics are printed in `%.2e`; the exact pair counts come from the ledger row.

Smoke on the Mac: `ctest -R "HADRONIC::perf_smoke|KERNELS::bh_perf"` (2¹⁶ ions, OpenMP): a SMOKE of the harness
mechanics on a CPU, not a performance measurement (uncontrolled host load, no registered equivalence tolerance;
Codex 1's review of 2026-10-30 reproduced the states and counts but not the timings). Rusty: `rusty/` (the one
A100 job, `p2-perf-a100.sbatch`; `preflight.sh` before submission; `PERF_SELFTEST=1` runs the script's negative
probes; the job ends with `status.json` and an exit code per class: complete 0, scientific_check_failure 10,
infrastructure_failure 20, incomplete_coverage 30).
