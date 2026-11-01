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
| M6 stride | `tests/kernels/bh_perf.cpp stride` | the stage every n steps with dt_BH = n dt (n = 1, 4, 16) at fixed physical time; event: ⟨K⟩ against the Jensen-corrected ODE, 4 s.e. verdict (the n = 16 failure retained); continuous (none, poisson, budget): DIAGNOSTIC ONLY (Nilay's decision of 2026-10-31, contract revision 1.1): the relative deviation of ⟨K⟩(t_end) between strides and the convergence order in the stride (reference-free and against the plain ODE), printed beside step 2's registered dt-convergence order 1.0 ± 0.3 (T5), no pass/fail; identity on every run; stage-launch time |

M3 attribution method (round 2, Codex 1 F6). The timed interval of a variant row is the `parallel_reduce` of the
kernel (the body and its ledger reduction are one Kokkos call; the reducer returns on the host) plus the fence;
`pairs.finish()` (the spawn-counter `deep_copy`) is timed separately (column "finish ms"); the production stage adds
the generator's post-step (ledger accumulation), measured by the engine's `Custom` timer in M1, not here. On the A100
the breakdown is read from two profiles per configuration: (a) nsys with NVTX ranges `<launch label>` (kernel +
reduction + fence), `bh_perf_finish_readback` and `bh_perf_restore_and_reseed` (outside the timed interval), from
which the kernel time and the reduction-plus-readback tail are the CUDA kernel rows and the gap to the range end;
(b) ncu on 12 launches after the 20 warm-up launches (`--kernel-name-base demangled --kernel-name regex:BetheHeitler`,
the Kokkos CUDA function names carry the functor type), for the kernel's own metrics (`--set basic`). The RNG pool's
share cannot be ranged inside the kernel; `bh_perf rngpool`, a kernel that only acquires a pool state, draws once and
releases it over the same N threads, gives an ISOLATED PROXY of its cost (its time per thread beside the production
kernel's ns per ion), not a bound on the in-kernel share (Codex 1's C4): occupancy, contention for pool states, the
number of draws, the overlap with memory traffic and the production critical path differ, so the isolated cost can be
larger or smaller than the marginal cost inside the production kernel. The lookup fraction is (production − constants) / production under matched work, with the stated
limitation that the constants variant removes the lookup's memory traffic and arithmetic together and the float
variant changes the interpolation coordinate with the precision. The job fixes N at 4194304 for M3, M4, M6 and the
profiles (C4): it refuses to start (exit 6) on any other `PERF_N_M3`, explicit or inherited through `--export=ALL`;
`bh_perf` refuses an invalid `BH_PERF_N` (exit 2) instead of falling back to its default; `rusty/profile_check.py`
asserts in every profiled run's own log `BH_PERF_N = 4194304`, the configuration filter and one configuration of 120
launches, in the nsys report the three ranges of the configuration and no other configuration's launch range, and in
the ncu raw page at least 12 UNIQUE launch IDs (a launch with many metric rows counts once), all of the
configuration's functor type. Its negative controls on synthetic profiler output (one launch with 24 metric rows,
wrong N, wrong kernel type, another configuration's range) are in `HADRONIC::perf_job_selftest`.

Driver notes (facts found on the Mac smoke, 2026-10-30): Entity's `Custom` timer also wraps the generator's
post-step, whose ledger row and per-ion text dump are O(N) host passes; the driver therefore writes the ledger row
once, at the last step, and excludes the final diagnostics block (which also carries `finish()`) from the measured
mean. The budget source's auto latency cap (`t_latency = 6.4` at these parameters) emits a packet per ion every 13
steps, 46 per ion over 600 steps; the perf budget input sets `t_latency = 0` (packets at the request threshold
only) and `on_capacity_estimate = "warn"` (the runtime capacity policy stays `error`). Species counts in the
diagnostics are printed in `%.2e`; the exact pair counts come from the ledger row.

Row validity (Codex 1's closure C2, 2026-10-31): every timing column of a CSV row (Custom, pusher, deposit, Total,
progress_avg_ms) is gated in both driver paths, the fresh runs and the M5 restarts: exactly the measured window's
samples, all finite, else the run is an infrastructure failure (`timers_ok = 0`); a deposit-enabled run (the five M5
rows) must have measured a positive deposit. `nan` in the CSV means missing or invalid; `n/a` means not applicable: the
restart rows' `custom_ns_per_ion_step`, whose stage is disabled (their `Custom`, the generator's post-step, is reported
and gated). The restart takes the source run's checkpoint by its exact name `<run>.ckpt`. Negative controls on the real
smoke output: `perf_driver_replay.sh`, registered as `HADRONIC::perf_driver_replay` (fixture on `HADRONIC::perf_smoke`).

Smoke on the Mac: `ctest -R "HADRONIC::perf_smoke|KERNELS::bh_perf"` (2¹⁶ ions, OpenMP): a SMOKE of the harness
mechanics on a CPU, not a performance measurement (uncontrolled host load, no registered equivalence tolerance;
Codex 1's review of 2026-10-30 reproduced the states and counts but not the timings). Rusty: `rusty/` (the one
A100 job, `p2-perf-a100.sbatch`; `preflight.sh` before submission; `PERF_SELFTEST=1` runs the script's negative
controls, registered as `HADRONIC::perf_job_selftest`; the job ends with `status.json` at the run root and an exit
code per class: complete 0, scientific_check_failure 10, infrastructure_failure 20, incomplete_coverage 30).

Failure classes (Codex 1's closure C1, 2026-10-31). A step is a scientific failure only when an EXECUTED check
failed: the program exited 1 and printed a line starting with `FAIL: ` (`perf_driver.sh`: a ledger identity that
bh_check rejected; `bh_perf`: an M3/M6 verdict; the F8 entries: an executed assertion). Everything else is
infrastructure: `perf_driver.sh` prints `INFRA: ` and exits 3 for an `entity.xc` failure, missing or malformed timer
data, a missing checkpoint, a restart that kept the stage on, or a ledger bh_check could not read; `rusty/f8_ctest.py`
(the F8 step) runs each named entry on its own through Rusty's validated launcher plan (Codex 1's G3 plan: direct
commands, `f8_tracers` among them, wrapped in `mpirun -n 1`; the stock two-rank commands kept) and classifies it from
CTest's JUnit result and, for a failed entry, the native exit value in CTest's Test.xml: Not Run, Timeout, Exception
(signals, aborts), an MPI launcher failure, a native exit other than 1 (the caught-exception exit 2 of `f8_tracers`
and the two weights tests) and an exit 1 without the entry's anchored check-failure evidence are infrastructure
(exit 4); scientific (exit 1) needs that evidence (the verdict programs' `FAIL: ` lines with a summary failed count
>= 1; `deposit`'s four check messages; a `raise::Error` at a line of the framework test's own source), since a stock
non-zero exit is not presumed to be a check failure (Codex 1's reclosure R1); a missing entry is the selection gate
(exit 3). Each
precision's F8 run starts only when its own build exited 0 and produced the complete inventory of its binaries.
