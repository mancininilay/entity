# Pre-registration: the Bethe-Heitler policy with MPI on several GPUs, M0 to M4 (Claude 1, local Claude Code, second session, 2026-11-22)

Written before any job of this validation was submitted. Brief: `Resources/p2-hadronic/notes/2026-11-22-claude1-local-brief-bh-mpi-gpu-validation.md`. Nothing here is changed after a run: a failure is kept and reported with its numbers, no tolerance is touched, and no run is repeated with changed parameters. Every verdict will be the author's and is not refereed.

## 0. What is run, and what is fixed

- **Code.** The policy, the engine and the unit test are those of `dev/bh-emission` 43d68f45 (pull request entity-toolkit/entity#242), unchanged. Two branches of mine, not pushed:
  - `dev/bh-mpi-gpu`, from `dev/bh-emission-validation` f31c8a06. Its `src/` differs from 43d68f45 in `checkpoint/init.cpp` only (the three-line clang fix, no effect with GCC). Changed here: the validation generator `pgens/bh_validation` (one line per rank at each diagnostic output, `setup.rank_lines`; an optional box for the emitters, `setup.hot_fraction`, `hot_box`, `cold_box`; with the defaults the positions and the `BHV` lines are those of the earlier version), `make_input.py` (the new keys), and this folder.
  - `dev/bh-reconnection-mpi`, from `dev/bh-reconnection` 97d78619 (the commit of R-off and R-low). Changed: the generator copy `pgens/reconnection_bh` writes a third text file, `bh_ranks.txt` (`setup.rank_interval`; read-only, as its other diagnostics), and the folder `pgens/reconnection_bh/mpi/`.
- **Build (M0).** Rusty, the installs of `~/work/env_rusty_entity.sh` with UCX as the P1.3 runs on A100s: GCC 13.3, CUDA 12.8, Open MPI with CUDA 4.1.8, Kokkos 5.2.1 (CUDA), ADIOS2 2.12.1; `-Dmpi=ON -Dgpu_aware_mpi=OFF -Doutput=ON`. Seven trees (`rusty/build.sh`).
- **Launcher.** `mpirun --bind-to none --map-by ppr:<ranks>:node -n <ranks> rusty/rank-gpu.sh entity.xc -input ...`: the wrapper gives each rank the GPU of its index on the node (as P1.3's launcher). The GPU of each rank is written to a file per run; a run of M1 to M4 whose ranks do not sit on distinct A100-80GB cards does not count.
- **Random numbers (fact, `src/framework/domain/domain.h` 107 and `src/global/utils/numeric.h` 91 at 43d68f45).** The pool of rank r is seeded with a constant plus r. A run is therefore reproducible; rank 0 of every decomposition uses the same streams; single and double precision draw the same numbers. Verdicts that share streams are not independent, and this is said with each of them.
- **Fact about the engine under MPI that M3 and the capacities of M4 rest on** (`src/framework/containers/comm/particles.cpp` 221 to 223 at 43d68f45): in a build with MPI, `Particles::Communicate` raises "Too many particles to receive (cannot fit into maxptl)" when `npart() + received >= maxnpart()`, for every species and every rank, at the particle communication of every step, also when nothing is received and also on one rank (a periodic box communicates with itself). The policy fills its lepton species up to exactly `maxnpart` (`capacity = maxnpart - npart`, `bethe_heitler.h` 529 to 530, 578, 643 to 648). A build without MPI never makes this call.

## 1. M0: the unit test in this build

`ARCHETYPES::bethe_heitler` (tier A of the pull request: A1 charge, A2 rate and loss, A3 refusals and capacity), executable built with CUDA, MPI and output, run on one rank on one A100, in single and in double precision, and in single precision with bounds-checked Kokkos views (`-DKOKKOS_ENABLE_DEBUG_BOUNDS_CHECK`). **Criterion: the test's own assertions, unchanged (exit status 0), in the three builds.** The numbers of A1 are reported.

## 2. M1: rate and loss on 1, 2 and 4 ranks

Uniform periodic box, no fields, no deposit; one A100 per rank; decompositions [1, 1], [2, 1], [2, 2] of the 64 x 64 box. Two cases, each also with the policy off (same seed), in single and in double precision: 24 runs.

- **Definition: that of test A1 of the reconnection shakedown** (`Entity Local Experiments/p2-bh-reconnection-claude1-20261121/a1/PREREGISTRATION.md`), unchanged where it applies: 8,000,000 emitters of mass 1 with isotropic momenta of one magnitude, species order ion, e-, e+, `mass_ratio` 1836.15, `pair_weight` 1, defaults for `max_probability` and `max_loss_fraction`, no sorting, about 0.05 interactions per emitter, about 4e5 pairs (at least 1e5 required).
- **Case L, the 1.8 bath at Gamma_p = 1.2: the point of test A1b, unchanged** (`gamma_max` 0.0, the automatic tables of the reconnection runs; `optical_depth` 0.3378; 100 steps). On the Mac without MPI it gave 394,744 pairs against 394,861.6.
- **Case H, the 0.5 bath at Gamma_p = 6:** `gamma_max` 6.5 (the tables of V2 at this energy), `optical_depth` 0.0906 (probability per step 1.249e-3), 40 steps. No earlier test has this definition; it is A1's with these numbers.
- **Reference, outside the policy:** `bhref` 0.3.0 (Codex 1, frozen), `Rates(Bath(mono, eps0), BornCumulatives(1024)).at(Gamma_p)`, written to `reference/points.txt` before any run:

| Case | eps0 | Gamma_p | J | <D> | d ln J / d Gamma |
|---|---|---|---|---|---|
| H | 0.5 | 6 | 0.389990095916 | 8.367496540301 | 0.439505 |
| L | 1.8 | 1.2 | 0.041333003092 | 1.059179706798 | 9.260863 |

The policy's own tables are not used anywhere in the comparison.

**Quantities** (n steps, N emitters, dt from the times printed by the run, p = optical_depth x J x dt, s = (d ln J / d Gamma) <D> / 1836.15):

1. **Number of pairs** (A1, unchanged). Observed: the counter `pairs` summed over the ranks after n steps. Expected: N p [n - s p n (n - 1) / 2]: 399,730 for H, 394,862 for L. Standard error sqrt(expected). **Pass: within 4 standard errors, and at least 1e5 pairs.**
2. **Mean loss per interaction** (A1's quantity; its standard error is obtained differently). Observed: (sum of the emitters' kinetic energies with the policy off minus the same sum with it on) / pairs, after n steps, sums over all ranks. Expected: <D>. **Standard error, a departure from A1 forced by MPI:** A1 took the standard deviation of D_i = gamma_- + gamma_+ - eps0 over the pairs of a dump in which electron i and positron i are one pair; under MPI the two leptons of a pair travel between ranks separately and that pairing is lost. Here Var D = [S2(on) - S2(off) + 2 (S1(off) / N) L - N n (n - 1) p^2 <D>^2] / pairs - (L / pairs)^2, with S1 and S2 the sums of (K - K_ref) and (K - K_ref)^2 over the emitters (printed by the generator, summed over the ranks) and L the loss; the term in S1 removes the offset of single-precision momenta, the last term in the bracket the emitters that interacted twice (5 percent of the pairs; it enters the error estimate only). Standard error sqrt(Var D / pairs). The standard deviation of D is printed and can be compared with A1b's 0.4497 at case L. **Pass: within 4 standard errors.**
3. **Closed box.** After the last step: the counter `pairs` summed over the ranks = electrons present = positrons present, summed over the ranks, exactly; counters `above`, `pexc`, `bounded`, `dropped` zero; the number of emitters unchanged; the sum over the ranks of the per-rank counters equals the total; with the policy off every counter and both lepton species are zero. **Pass: all of it, exactly.**
4. **Per rank.** Pairs counted by rank r against p times the number of emitter-steps on rank r (the emitters present on the rank before each step, taken from the per-rank lines; the emitters of the first step are taken as those present after it, an error below 1e-3 of one step's share). **Pass: within 4 standard errors (sqrt of the expected number) for every rank.**
5. **The three decompositions agree.** For each case and precision, each pair of decompositions: pairs within 4 sqrt(expected_a + expected_b); mean loss within 4 times the two standard errors in quadrature (V12's rule). **Pass: all three pairs.** The decompositions share the streams of rank 0, so their difference scatters less than this rule assumes: a pass is weaker than it looks, a failure would be significant.

**Error budget, separate from the statistics and not added to the tolerance because small:** the policy's lookup gives J below `bhref` by 8e-5 at case H (table value 0.389958) and by 1e-5 to 6e-5 at case L (measured in stage A of the shakedown); the sampler's mean loss exceeds the table by (1.0 +- 0.4)e-4 at Gamma_p = 6 (report of 2026-11-19, section 11.5); the second-order term of the expected count is below 1e-5. Statistical standard errors: 1.6e-3 of the count; about 1e-3 of the mean loss.

Verdicts: per precision and case, 3 decompositions x (run complete, pairs, mean loss, closed box twice, sum of the ranks, each rank) and 3 x 2 comparisons. The two precisions are one sample, not two.

Reported, no criterion: the lepton-side sum of D against the emitters' loss; the absorbed photon energy per pair (eps0).

## 3. M2: charge across rank boundaries, both species orders

**V1, unchanged** (pre-registration of 2026-11-19 and `scripts/run_v1.sh`; decision P10 of 2026-11-20 for the negative control): 64 x 64 periodic box, field solver and deposit on, current filters off, 32,768 emitters at Gamma_p = 10 with isotropic momenta and one electron on top of each, 100 steps, about 1.6e5 pairs; r(t) = largest |div E - kappa rho| over the nodes of the global grid (fields and charge of all ranks summed), 20 outputs; control: the same run with the policy off. Decompositions [2, 1] and [2, 2] (32 cells per rank in the cut directions; a particle moves up to 0.35 cell per step, so most particles cross a rank boundary during the run). Single and double precision. A = leptons listed before the emitter, B = after it.

**Registered criteria:**

1. **Pass criterion of V1:** largest r over the outputs <= 10 x the same of the control, and at least 1e4 pairs: `A_on`, `B_on`, on 2 and on 4 ranks, both precisions (8 verdicts).
2. **Negative control of V1 as re-registered (P10):** leptons first, the previous-position write left out, the unused lepton slots holding stale positions that differ between the two species (`A_neg_stale`), **on 2 ranks: the ratio must exceed 1000** (2 verdicts). The same on 4 ranks is run and reported, not registered.
3. **Crossing in the step of birth, measured.** `B_birth`: order B, every emitter placed in the part of the box that rank 0 holds (`hot_box`), same density of interactions; one more diagnostic line after the first step. After that step the positrons found on the other ranks crossed a rank boundary in the step of their birth (no emitter is there, so none was born there). **Pass: at least 5 such positrons (expected: about 19 on 2 ranks and about 38 on 4, of about 1700 born in that step), no pair counted by another rank in that step, and the pass criterion of V1 against its own control (`B_birth_off`) over the 100 steps;** 2 and 4 ranks, both precisions (4 verdicts). The first step is the only one in which these crossings can be counted from outside; they happen in every step.
4. **The same with the leptons listed first** (`A_birth`): the new leptons are not pushed in the step of their birth, so **after the first step no positron is on another rank (exactly 0), and the pass criterion of V1 holds** (4 verdicts). A positron elsewhere would contradict my reading of the update order.
5. **Prediction for the order B without the write** (`B_neg_stale`: ions first, the write left out, stale slots): **the pass criterion of V1 holds (ratio <= 10)**, on 2 and on 4 ranks (4 verdicts). This is the statement of Codex 1's S2 made into a test: with the emitter listed first every new lepton is pushed before the deposit, and the push writes the previous position itself, so the missing write cannot show. If the ratio exceeds 10 the prediction is wrong and I say so.

Reported, not registered: `A_on_stale`, `B_on_stale` (the policy as built with stale slots; expected within the pass criterion, as on the Mac and on one A100); per rank, the pairs born and the positrons present at the end.

**Which configuration can fail at all** (to be confirmed or contradicted by the numbers): order A with stale slots can fail on a missing previous-position write (criterion 2 shows it); order A without stale slots cannot (the phantom currents of the two species cancel: the recorded failure of V1's first control); order B cannot fail on that write in any state of the slots (criterion 5), but it can fail on what only it exercises: a lepton that is pushed, deposited and sent to another rank in the step of its birth (criterion 3). A lost or duplicated charge leaves a residual of the order of the charge of one particle, 1e5 times the control's.

## 4. M3: a full lepton species on one rank only

**The run:** uniform box without fields, 200,000 emitters at Gamma_p = 6 on the 0.5 bath, probability per step 0.05, 10 steps, lepton capacity 30,000 per rank; nine tenths of the emitters well inside rank 0, one tenth well inside the last rank, no particle within reach of a rank boundary in 10 steps (3.6 cells). Rank 0 creates about 9000 pairs per step and reaches its capacity in the fourth step; the last rank creates about 1000 per step and stays at a tenth of its capacity. 2 ranks ([2, 1]) and 4 ranks ([2, 2]); single and double precision; and single precision with bounds-checked views.

**Criterion of the brief:** the run completes its 10 steps; on rank 0 the candidates that do not fit are skipped and counted (`dropped` > 0, lepton arrays at their capacity); the other ranks go on and count no dropped pair; the summed counters are consistent (pairs = electrons = positrons present; the sum of the per-rank `dropped` = the total); the bounds-checked build reports no access out of bounds.

**Prediction, from the source (section 0) and from the same input on the Mac (MPICH, 2 and 4 ranks, `mac-probes/` of the evidence folder): the criterion FAILS at its first clause.** In the step in which rank 0 fills, the policy skips and counts and prints its warning; then the engine's particle communication of that same step finds `npart = maxnpart` on rank 0 and aborts all ranks. Expected in the log: diagnostic lines after steps 1 to 3, the warning "Bethe-Heitler pairs are dropped", the error "Too many particles to receive (cannot fit into maxptl)", exit status not 0. This is the brief's "MPI receive error at capacity". It is in the engine, not in the policy; changing either is outside this brief.

**What the GPU runs can still decide:** whether the prediction holds in the build with CUDA; from the bounds-checked build, whether any access out of bounds precedes the engine's error (it would be reported first); that the last diagnostic lines before the stop show the other ranks creating pairs at their own rate. The policy's own handling of a full species (no recoil, no write out of bounds, the counter) is covered at kernel level by A3 of the unit test in M0, in the same build.

M3 runs last in its job. Its failure does not stop M4 (the brief: a failure of M1 or M2 does).

## 5. M4: reconnection on four GPUs

**Input.** `pgens/reconnection_bh/mpi/r-low-mpi.toml` on `dev/bh-reconnection-mpi`: `r-low.toml` of 97d78619 with exactly these changes (the `diff` is seven lines): the name; `decomposition = [2, 2]`; `maxnpart` of the two primary species 8e7 per rank (was 1.2e8 for the one rank); `gauss_interval = 0`; `rank_interval = 10`; `[checkpoint] keep = 0`. Optical depth 0.014333, tables, output and everything else unchanged.

**The generator copy under MPI (fact, read in the source at 97d78619).** (i) The counters of the policy are per rank; `WriteCounters` already sums them, and the sums per species, over the ranks (`MPI_Allreduce`), so `bh_counters.txt` needs no change. (ii) The Gauss diagnostic is valid on one domain only and raises an error otherwise (`pgen.hpp`, `WriteGauss`): it is switched off for this run, not rewritten. No check of the charge is made in M4; M2 is the test of the charge under MPI. (iii) The stock injector computes the density it replenishes with a moment of smoothing order 0 (`src/archetypes/utils.h` 143), each particle in its own cell, so it needs no ghost-cell exchange at a rank boundary. (iv) Added: `bh_ranks.txt`, the fill of every species' arrays per rank, every 10 steps. (v) Checkpoints are switched off (no restart is in scope; they were 16 percent of the single-card run), as in the runs of stage C.

**Decomposition [2, 2], and why.** From the saved density of the primaries in the two single-card runs (`m4/capacity/` of the evidence folder; 30 outputs each), the largest number of particles per primary species that one rank would have held:

| Decomposition | R-off | R-low | Largest share of the box on one rank |
|---|---|---|---|
| [4, 1], a quarter along x anywhere (the plasmoids of another realisation sit elsewhere) | 2.53e7 | 3.66e7 | 0.46 |
| [1, 4], the quarters along y | 2.47e7 | 2.97e7 | 0.37 |
| [2, 2], half along x anywhere times half along y | 2.12e7 | 2.57e7 | 0.33 |

[2, 2] has the smallest peak and the most even load late in the run. Its boundary at y = 0 runs along the whole current sheet: the pairs are born on it and the leptons of the plasmoids cross it all the time, which is the hardest case for the policy under MPI and the one this validation is for; it also has cuts in both directions, as a larger decomposition will. The price is more particle communication than [4, 1], so the cost per particle-step measured here is not the lowest that four GPUs can reach.

**Capacities per rank.** Under MPI a species that reaches its capacity on any rank ends the run (section 0), so the capacity must never be reached. Primaries: estimate 2.57e7 per species on one rank ([2, 2], the larger of two realisations); capacity **8e7**, 3.1 times the estimate and equal to the largest number the whole box held in R-low (8.07e7): a rank could hold nearly all the plasma of the box. Leptons of the pairs: the whole box held at most 4.8e5 per species in R-low; capacity **4e7** per rank, unchanged, 80 times that. Memory: about 12 GB of each 80 GB card for the particle arrays.

**Registered checks, those of the shakedown, unchanged** (`pgens/reconnection_bh/PREREGISTRATION.md` at 97d78619, sections 1 and 2; scripts `check1_counters.py` and `check2_paircount.py` of that commit, the same reference table `data/j_table_mono18.txt`):

1. **Counters (check 1):** particle-steps above the tables 0; pairs dropped for capacity 0; refused candidates and particle-steps above the probability bound reported.
2. **Pair count (check 2), same estimator:** |N_obs - N_pred| <= 4 sqrt(statistics^2 + stride^2 + cadence^2), N_pred = optical depth x time integral of the sum over the saved ions (stride 5, every 20 time units) of the `bhref` rate; no verdict if the tolerance exceeds 5 percent of N_pred. With MPI every rank writes every fifth particle of its own arrays; the estimator scales by (ions / ions written) from the spectrum, as before.

**Also required for a pass of M4:** the run completes (exit status 0, the last line of `bh_counters.txt` at the last step); the four ranks sat on four distinct A100-80GB; the sum over the ranks of the per-rank `pairs` of `bh_ranks.txt` equals the summed counter at the last line.

**Reported, no verdict:** the late-time scalars of check 3 of the shakedown (inflow rate; mean K, quantiles, slope and fraction above threshold of the spectra) beside those of the single-card R-low, and of the ensemble of stage C if it exists by then (the 4-rank run is another realisation: other streams, other order of operations); the largest fill of every species on any rank against the capacity, and the largest share of the primaries on one rank against the estimate above; GPU-ns per particle-step (4 GPUs x elapsed time / particle-steps, for the whole job and for the step timers only) against the single-card R-low (1.46 and 1.04); the engine's timers, communication in particular.

**Before M4 (a gate, not a verdict):** the toy input on four ranks, run in the build job, is read on the login node with the readers of the checks (fields with reversed dimensions, particle output, spectra, counters), as the reader gate of stage B. If the readers cannot read output written by four ranks, M4 is not submitted and I report.

## 6. Stop rules and budget

As the brief: a crash, a non-finite value, an MPI receive error at capacity, or a registered check outside its tolerance is reported and nothing is rerun with changed parameters; a failure of M1 or M2 stops M4. M3's predicted stop is reported as a failure of M3. Jobs: `gpupreempt`, no requeue, `sbatch --test-only` first, fresh directories under `~/ceph/bh-mpi-gpu-claude1-20261122/`: the build job (1 A100, 50 minutes at most), the job of M1 to M3 (4 A100-80GB on one node, 50 minutes at most), the job of M4 (4 A100-80GB, 2 hours at most). One resubmission of a job that is preempted or fails for a reason of the machine, then once on `gpu`. Cap: 15 A100 GPU-hours in all, counted as GPUs x elapsed time. Nothing is deleted on Rusty.

Signed **Claude 1** (local Claude Code, second session).
