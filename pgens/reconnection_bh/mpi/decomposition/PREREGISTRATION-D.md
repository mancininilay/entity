# Pre-registration: does the decomposition shift the dynamics? (Claude 1, local Claude Code, second session, 2026-11-22)

Written before the first of the four runs is submitted. Brief: `2026-11-22-claude1-local-brief-bh-decomposition.md` (issued by Nilay by pasting it; `decisions.md`, entries of 18:02 and 18:15 EST). It builds on the pre-registrations of the shakedown (stage B, `2026-11-21-claude1-bh-reconnection-shakedown-preregistration.md`, and stage C, `2026-11-22-claude1-bh-reconnection-stage-c-preregistration.md`) and of the validation with MPI on GPUs (`2026-11-22-claude1-bh-mpi-gpu-preregistration.md`), none of which is changed. The same text is committed in the fork as `pgens/reconnection_bh/mpi/decomposition/PREREGISTRATION-D.md` on the branch `dev/bh-reconnection-mpi`, next to the inputs and the scripts whose hashes it fixes (`INPUTS.sha256`, `SCRIPTS.sha256`). Nothing here is changed after a run: a failure is kept and reported, and no tolerance is touched. Every verdict will be the author's, not refereed.

## 0. The question, and what I knew when I wrote this

**The question.** The run of M4 (job 7523907: the R-low input on four A100, decomposition [2, 2], so that one rank boundary lies along the whole current layer) has a late mean ion energy 25 percent below, and 15 percent fewer pairs than, the single-card run of the same input. Codex 1 (review of 2026-11-22, G5) finds no isolated jump at the rank boundaries and holds that neither an effect of the decomposition nor chance is established. The first session (shakedown report, section 22) finds the run inside the scatter of the single-card ensemble by the rule of check 3, which one run cannot turn into "it is one more realisation".

**Disclosure.** When this was written I knew every late scalar of the eight existing runs (the seven single-card runs and the run of M4), and, from the gate of section 9, their early scalars and their interface values. No new run existed. The run of M4 is therefore treated as the run that raised the question, not as evidence for the answer (section 4).

## 1. What is run

**Executable:** the one of the run of M4, unchanged: build `build-7523472`, tree `rec-single`, SHA-256 `2f6d102603f70f8857fbabe89fa86dfad20414868c4a3f30515d98e120859aa5`, built from `dev/bh-reconnection-mpi` b996538f. No generator and no engine file changes; the commit of the runs differs from b996538f only in files of `pgens/reconnection_bh/mpi/decomposition/`, and the run driver refuses to start otherwise.

**Four runs**, inputs derived from the input of the run of M4 (`mpi/r-low-mpi.toml`) by `derive_inputs_d.py` (listed edits, each matching once):

| Run | Decomposition | Overdensity of the initial layer | Capacity per rank: primaries, leptons of the pairs | Wall limit |
|---|---|---|---|---|
| `r-low-mpi-b22` | [2, 2] | 2.997 | 8e7, 4e7 (as M4) | 40 min |
| `r-low-mpi-c22` | [2, 2] | 3.003 | 8e7, 4e7 (as M4) | 40 min |
| `r-low-mpi-a41` | [4, 1] | 3.000 | 1e8, 4e7 | 49 min |
| `r-low-mpi-b41` | [4, 1] | 2.997 | 1e8, 4e7 | 49 min |

Everything else is the input of M4: optical depth 0.014333, 4096 x 2048 cells, runtime 3000, no checkpoints, the Gauss diagnostic of one domain off, fields every 100 time units, the ions and the leptons of the pairs at stride 5 every 20 time units, spectra every 20 time units, counters and the fill per rank every 10 steps. Four A100-80GB of one node, one rank per card, MPI staged through the host, `gpupreempt`, no requeue, a fresh run directory per job.

**What makes them new realisations.** For the two runs on [2, 2]: the overdensity of the initial layer in its fourth digit, the change stage C used and showed to be effective on the GPU (its members were apart from their originals by t = 500 to 800). For `r-low-mpi-a41`, whose input physics is exactly that of R-low and of M4: the decomposition itself, since the random pools are seeded by the index of the domain and the particles are dealt to other ranks (the run of M4 separated from R-low in this way: last output with a correlation of 0.5 or more at t = 600). The independence condition of check 3 is applied to every pair of runs (section 4); it can deny a verdict.

**Capacities (the author's rule with Codex 1's margin, review G6).** Per species and rank, C = 2 f_max N_max, where N_max is the largest number of particles of the species in the box in any run with the policy so far (8.80e7, R-low-b; R-low itself 8.28e7) and f_max the largest share of one rank found by projecting the saved densities of the single-card runs on the tiles, with the origin in x allowed to slide (0.3288 for [2, 2], 0.4611 for [4, 1]; the run of M4 reached 0.2936).

- [2, 2]: 2 x 0.3288 x 8.80e7 = 5.79e7; the input keeps the 8e7 of M4 (2.8 times the estimate of the fullest rank; M4 peaked at 2.36e7).
- [4, 1]: 2 x 0.4611 x 8.80e7 = 8.12e7, rounded up to **1e8** (2.5 times the estimate of the fullest rank, and more than all particles of a species present in any run so far).
- Leptons of the pairs: 4e7 per species and rank, 37 times the number of pairs created in a whole run so far (1.07e6), so the estimate does not depend on where they gather.
- The margin also covers the engine's stop at a full species (U19: a rank ends the run when its particles plus those it receives reach the capacity). Memory: 2.8e8 slots per rank for [4, 1] against 2.4e8 in M4, about 14 GB at 50 bytes per slot, on cards of 80 GB.

**GPU time.** The run of M4 took 33 minutes (2.21 GPU-hours). On [4, 1] the fullest rank holds a third of the particles on average instead of a little over a quarter, so I expect 33 to 41 minutes. The wall limits (40 and 49 minutes; the engine is stopped 120 s earlier) keep the four runs below 11.87 A100 GPU-hours by construction; the cap is 12. One resubmission of a job that is preempted or fails for a reason of the machine is made only if the hours left cover its wall limit.

**Stops.** A crash, a non-finite value, a capacity error, particle-steps above the tables, pairs dropped, candidates refused by the loss bound, or check 2 outside its tolerance: reported, no rerun with changed parameters. A run that does not reach t = 3000 is reported with what it wrote and is left out of the comparisons of section 4, which are then made with the runs that finished and the degrees of freedom they give; a comparison that has no run on one side gives no verdict.

## 2. Checks 1 and 2 of the shakedown, on every run

The scripts of the commit of the build (`check1_counters.py`, `check2_paircount.py`, unchanged since 97d78619), as for the run of M4, with the criteria of the stage-B pre-registration word for word:

- **Check 1:** particle-steps above the tables 0; pairs dropped 0; refused candidates and particle-steps above the probability bound reported (expected 0).
- **Check 2:** |N_obs - N_pred| <= 4 sqrt(statistics^2 + stride^2 + cadence^2), no verdict if that tolerance exceeds 5 percent; N_pred from the saved ions and the `bhref` table at the optical depth of the input. Expected tolerance near 1.3 percent, as in M4 (1.29 percent; observed there -2.6e-5).
- **Reported with them, as for M4:** the pairs counted per rank against the summed counter (they must be equal), the largest fill of every species on one rank against its capacity, the largest share of one rank, the timers and the cost per particle-step.

## 3. The scalars

One number per run. The late scalars are computed by the registered function of stage C (`check3_ensemble.measure`, the file copied unchanged, SHA-256 `7ced89898f0446d3dc74a4323d655f07e4c43f8eac8caed82b13f7ab9b0b2529`), so that they are the same estimators as those of the single-card ensemble.

**Late family (five scalars; the question of the brief):**

1. mean kinetic energy of the ions, spectra over t >= 2000;
2. the 99 percent quantile of the ion energy, same window;
3. the 99.9 percent quantile, same window;
4. the reconnection rate of check 3 (inflow speed in the four quadrants, field outputs over t >= 1000);
5. the number of pairs created in the whole run, per unit of optical depth and quoted at 0.014333 (this only touches R-low-b and R-low-c, whose optical depth differs by 7 parts in 10,000).

**Early family (five scalars; added by me, see why below):**

6. the number of pairs created up to t = 750 (counter, last line with time <= 750; normalised as scalar 5);
7. the mean kinetic energy of the ions at t = 750 (counters, K_2 / w_2 of the same line);
8. the reconnection rate over the field outputs with 300 <= t <= 700 (the estimator of check 3 on another window; my copy of it must reproduce the registered function on the registered window to 1e-9, or the script stops);
9. the number of ions present at t = 750 (counter): the layer plus everything the injector has put in to replace the plasma that flowed towards it, so the inflow integrated over the closed phase;
10. the mean kinetic energy of the ions at t = 1000 (counters), 250 time units after the opening, the last of the saved times at which the eight existing runs still agree to a few percent.

**Why an early family, and what it cannot do.** The scatter of the late scalars is set by a few large plasmoids, and four runs cannot beat it (section 5). Before the boundaries open at t = 750 the runs still share the first chain of plasmoids (stage C: correlations of 0.5 to 0.8 between t = 300 and 800), so the same quantities scatter far less from run to run (section 5: a few percent, 0.5 percent for the number of ions). A decomposition that acts on the layer from the start (through the cut along it, for instance) has to show there. The times 750 and 1000 were chosen from the eight existing runs; no new run existed.

**The confound of the early family (fact, source).** The runs of one group share almost all of their initial particle noise: the random pools are seeded by a constant plus the index of the domain (`src/framework/domain/domain.h:107`), the generator loads the uniform background before the layer (`pgens/reconnection_bh/pgen.hpp:321` and `:340`), and the change of the overdensity touches only the particles of the layer. So the seven single-card runs are close to one noise realisation in the closed phase, the runs on [2, 2] close to a second and the runs on [4, 1] close to a third, and the tearing of the layer grows from that noise. A difference between the groups in the early family can therefore come from the noise realisation as well as from the decomposition, and the scatter within a group understates the scatter between noise realisations. **Consequence, fixed here:** the early family can clear the decomposition (no difference between the groups, with the bounds reached), and it can flag a difference between the groups; it cannot by itself establish that the decomposition is the cause. If [4, 1] agrees with the single-card runs and [2, 2] does not, or the reverse, that is named as an indication for that decomposition, not as a verdict. Separating the two needs runs that differ in the seed only (U11), which do not exist. The late family does not have this confound in the same form: by t >= 2000 the runs of one group have separated (the independence condition of section 4 tests it), so the scatter within a group is the scatter between realisations, up to a common component left by the shared early history, which the condition cannot exclude and which would make the intervals too narrow.

**Reported without a rule:** the inflow rate over t >= 2000; the fraction of ions above threshold; the slopes; the four scalars of the electrons; the primaries per species at the end; the largest ion energy of the run; the counters of every run side by side at seventeen times, with a finer sampling around the opening of the boundaries (`counters_series.py`: primaries present, pairs created, the share of the leptons of the pairs that has left the box, the mean energy of the ions and of the leptons of the pairs); the reconnection rate at every field output of every run (`rate_series.py`).

## 4. The groups, the comparisons, the scatter and the rule

**Groups of runs.**

- **S**, one card, policy, this optical depth: R-low, R-low-b, R-low-c (R-low-b and R-low-c also differ from the runs on four cards by 7 parts in 10,000 in the optical depth, section 18 of the shakedown report).
- **O**, one card, no policy: R-off, R-off-b, R-off-c. R-mid (one card, ten times the optical depth) is shown beside every number and never pooled.
- **F22**, four cards, [2, 2]: the run of M4 and the two new ones. **F41**, four cards, [4, 1]: the two new ones.

**Independence (a condition, as registered for check 3).** Two runs count as independent realisations when the median over the outputs with t >= 2000 of the correlation of the density of the primaries along the sheet (N_1 + N_2 over |y| <= 20 d_e, 256 bins) is below 0.5. If two runs of one group are not independent, the scatter is not measured and there is no verdict.

**Scatter.** For every scalar, s is the pooled standard deviation within the groups O, S, F22 and F41 (seven degrees of freedom when all runs finish); for the two numbers of pairs, which O does not have, within S, F22 and F41 (five). This assumes that the run-to-run scatter is the same in the four groups, and it stays valid if the groups differ in their means, which is what is being tested. The run of M4 enters the scatter of F22 like any other run. Sensitivity, printed beside the verdict: the same comparison with the scatter of the single-card runs alone, which was known before the runs (four degrees of freedom; two for the pairs).

**Comparisons.** D = mean(A) - mean(B), standard error s sqrt(1 / n_A + 1 / n_B), Student t with the degrees of freedom of s. All differences are quoted in percent of the mean of S.

| | A against B | Role |
|---|---|---|
| C1 | the four new runs against S | **confirmatory: the verdict "four cards against one"** |
| C2 | all five runs on four cards against S | the comparison the brief names; it contains the run that raised the question and is therefore biased towards a shift; reported |
| C3 | the two new [2, 2] against S (C3': the three [2, 2]) | reported |
| C4 | the two [4, 1] against S | reported |
| C5 | the two [4, 1] against the two new [2, 2] | **confirmatory: the verdict "[4, 1] against [2, 2]"** (C5': against the three [2, 2], reported) |
| context | the four new runs, and all five, against the six single-card runs S and O | reported; it assumes that the pairs change nothing at this optical depth, which is not established (Codex 1, G5) |

**Why the run of M4 is kept out of C1 and C5.** It was looked at because it came out low. A test that contains the observation that raised the hypothesis finds the hypothesis too often. The four new runs were not selected by anything.

**The rule, for each scalar of a family and each confirmatory comparison.** Two intervals around D: at 95 percent for this scalar alone, and at 95 percent for the family (Bonferroni: 99 percent per scalar in each family of five).

- **Established:** the family interval excludes zero.
- **Indicated, not established:** only the single interval excludes zero. It is named and not counted as a shift.
- **Not established:** the single interval contains zero.
- **Bound:** |D| plus the half-width of the single interval. A shift larger than the bound is excluded at 95 percent, for that scalar alone.
- **Excluded above the stated size:** the bound is below the stated size. Stated sizes, in percent of the mean of S: **25** for the three ion energies (the size of the difference that raised the question), **15** for the pairs, **10** for the reconnection rate, and **10** for each early scalar.

**Verdict of a family for a comparison:** "a difference between the groups is established" if at least one of its scalars is established; "no difference established, and excluded above the stated sizes" if none is established and every bound is below its stated size; otherwise "no difference established, not excluded at the stated sizes", with the bounds. The two families are judged separately, each at 5 percent, and they mean different things. **The verdict on a shift of the dynamics by the decomposition is that of the late family.** A difference established in the early family is a difference between the groups whose cause (decomposition or noise realisation) this design does not separate; it is reported as that.

## 5. What this design can and cannot see (written before the runs, from the scatter of the existing runs)

Scatter per run: the pooled standard deviation within the two single-card groups (four degrees of freedom, two for the pairs; gate of section 9). Half-widths: those the comparisons will have if the new runs scatter alike (seven degrees of freedom, five for the pairs), for the interval of one scalar and for the interval of its family. Everything in percent of the mean of S.

| Scalar | Scatter per run, percent of the mean of S (d.o.f. now) | Run of M4 against S: difference, t | C1, four new against S: half-width single / family | C2, all five against S | C5, [4, 1] against the two new [2, 2] |
|---|---|---|---|---|---|
| ions, mean K | 12.1 (4) | -24.5%, -1.76 | 21.8 / 32.3 | 20.9 / 30.9 | 28.6 / 42.3 |
| ions, K 99 percent | 12.2 (4) | -25.2%, -1.80 | 22.0 / 32.5 | 21.0 / 31.1 | 28.7 / 42.5 |
| ions, K 99.9 percent | 14.3 (4) | -26.2%, -1.58 | 25.9 / 38.3 | 24.8 / 36.7 | 33.9 / 50.2 |
| reconnection rate | 7.8 (4) | +9.9%, +1.10 | 14.0 / 20.7 | 13.4 / 19.8 | 18.3 / 27.1 |
| pairs created | 5.0 (2) | -16.2%, -2.82 | 9.8 / 15.3 | 9.3 / 14.7 | 12.8 / 20.1 |
| pairs created up to t = 750 | 3.9 (2) | -5.2%, -1.14 | 7.7 / 12.1 | 7.4 / 11.6 | 10.1 / 15.9 |
| ions, mean K at t = 750 | 2.1 (4) | -1.4%, -0.60 | 3.8 / 5.6 | 3.6 / 5.3 | 4.9 / 7.3 |
| reconnection rate over 300 <= t <= 700 | 3.9 (4) | -8.4%, -1.87 | 7.0 / 10.4 | 6.7 / 10.0 | 9.2 / 13.6 |
| ions present at t = 750 | 0.4 (4) | -0.6%, -1.09 | 0.8 / 1.2 | 0.8 / 1.1 | 1.1 / 1.6 |
| ions, mean K at t = 1000 | 2.4 (4) | +2.2%, +0.79 | 4.3 / 6.3 | 4.1 / 6.0 | 5.6 / 8.2 |

**What follows from it, stated before the runs.**

- **Late energies.** C1 can establish a difference only above 32 to 38 percent (22 to 26 percent at the level of one scalar), and where it finds none its bound cannot fall below 22 percent. "Excluded above 25 percent" needs the four new runs to average within about 3 percent of the single-card runs, while the standard error of that difference is 9 percent: it is unlikely even if the decomposition does nothing. The expected outcome in that case is "no difference established; differences above roughly 25 to 45 percent excluded". A true shift of a quarter, the size seen in the run of M4, would be established about one time in five (two times in three at the level of one scalar).
- **Pairs.** The sharpest late scalar: 10 percent (one scalar), 15 percent (family), if the pairs scatter by the 5 percent of the three single-card runs. That scatter rests on two degrees of freedom and may well be larger.
- **Reconnection rate.** 14 and 21 percent.
- **[4, 1] against [2, 2]** is two runs against two: 29 to 34 percent for the energies, 13 percent for the pairs, at the level of one scalar. Only a large difference between the decompositions can be established.
- **Early family.** 0.8 percent (ions present at t = 750) to 8 percent (pairs up to t = 750) at the level of one scalar: a difference of the late size, if present from the start, would stand out by many standard errors, with the confound of section 3.
- **What ten percent in a late energy would need:** about fifteen runs of each kind (shakedown report, section 18), or runs that differ in the seed only and a longer window. Four runs were decided; this section says what they can deliver.

## 6. The interface measure (Codex 1's, G5), on every run

**Definition.** His program `seams.py`, written again on the reader of the checks (`interface.py`). For a saved field (cell-centred, as written every 100 time units), a plane between two neighbouring rows of cells and a window along the plane: rms(k) is the root mean square along the window of the difference between the rows k and k - 1 cells away from the plane, for k = -8 to 8; k = 0 is the difference across the plane itself. Ratio to the adjacent planes: rms(0) / mean(rms(-1), rms(+1)). Ratio to the distant planes: rms(0) / median of rms at 4 to 7 cells on either side. The number of a run is the median of the ratio over the 29 outputs after the first.

**Planes, measured in every run whether or not they are rank boundaries there:** x = -250, x = 0 and x = +250 (the three cuts of [4, 1]; x = 0 is also the cut in x of [2, 2]), window |y| <= 190; and y = 0 (the cut in y of [2, 2], the plane of the current layer), window |x| <= 400. **Fields of the rule:** ion density, B_x, B_y, E_z, J_z (the five of his table). The other components of B, E and J and the electron density are computed and kept.

**Rule for the planes across the layer (x).** Away from the layer nothing physical singles out these planes, and in the existing runs the ratio to the adjacent planes is within 0.03 of 1 in all 120 values (eight runs, three planes, five fields; the largest deviations are 0.971 for J_z at x = 0 in R-mid and 1.018 for J_z at x = -250 in the run of M4, where that plane is not a rank boundary; at its rank boundary x = 0 the run of M4 has 0.986 to 1.009). An **isolated jump** is flagged for a run, a plane that is a rank boundary of that run and a field when the ratio differs from 1 by more than **0.05**. The same planes in the runs where they are not rank boundaries are the control: if the control exceeds 0.05 as well, the rule is not sharp and I say so.

**Rule for the plane of the layer (y = 0).** The layer itself makes a structure there (ratios between 0.2 and 1.1 in the existing runs), so only a comparison of runs means anything. For each of the five fields: the three runs with the cut ([2, 2]) against the eight without it (the six single-card runs S and O and the two runs on [4, 1]). The cut is flagged for a field when all three runs with it lie outside the range of the eight without it, on the same side. If the cut does nothing, this happens for one field with a probability of 2 / 165 = 1.2 percent, and for at least one of five with at most 6 percent. Known beforehand: the run of M4 lies outside the range of the six single-card runs S and O in three of the five fields (B_x 1.110, below 1.118 to 1.133; B_y 0.538, above 0.224 to 0.508; J_z 0.488, above 0.201 to 0.475) and inside it in two (ion density 0.743 in 0.544 to 0.866; E_z 0.602 in 0.283 to 0.667). For one run against six that happens by chance with a probability of 2 / 7 per field, and the fields are not independent of each other: it is the reason for the rule, not evidence.

**Reported without a rule:** the ratio to the distant planes; the mean signed jump across each rank boundary in units of its standard error (a coherent offset is seen there long before it changes a root mean square).

**What the measure cannot see** (Codex 1's own caveats): a defect diluted along the plane; anything in the native, staggered fields at every step; a smooth transport bias.

## 7. What I will tell Nilay, by outcome (fixed now)

- **Can runs on one card and on four be compared?** If C1 establishes a difference in the late family: no, and a series must stay on one configuration. If it does not: yes for effects larger than the bounds reached, which I will state per scalar, and not for smaller ones; with the scatter of this box that bound is expected to be near a quarter for the late energies, so the plain answer for a series is still "compare runs of one configuration with each other, and treat the configuration as a nuisance that has not been excluded below that size". If the early family is excluded above 10 percent, I will add that neither the decomposition nor the noise realisation changes the closed phase at that level. If the early family shows a difference, I will say that the groups differ early and that this design cannot tell the decomposition from the noise, and what would (a seed).
- **Which decomposition?** If the cut along the layer is flagged (rule of section 6 at y = 0) or C5 establishes a difference: the one without a cut along the layer ([4, 1], or [N, 1] in general), with its larger capacity per rank. If neither: no physical reason separates them, and the choice goes by cost and capacity, which the runs measure (wall time, share of the fullest rank).

## 8. Reported besides the verdicts

Job identifiers, nodes, GPU-hours against the cap of 12, preemptions; checks 1 and 2 and the per-rank numbers of every run; the cost per particle-step of [4, 1] against [2, 2]; the independence table; anything that broke.

## 9. The gate before the runs (facts; Rusty login nodes, no Slurm)

The scripts of this folder ran on the eight existing runs (the seven single-card runs of the shakedown and the run of M4) on the login node rustyamd2, 21:01 to 21:32 EST; a first stream on rusty4 was stopped as more than four times slower. The files are in the evidence folder (`rusty/decomposition/gate/`).

- **Late scalars:** the table of the shakedown report (sections 18 and 22) is reproduced to the digits given there (mean ion K 8.45, 8.74, 7.86; 5.85, 6.28, 8.38; R-mid 6.12; the run of M4 6.30; pairs 998,627, 1,070,684, 975,080 and 850,631). The independence condition holds for all 28 pairs of runs (medians -0.13 to +0.17).
- **Interface measure:** my implementation reproduces Codex 1's table (G5) for the run of M4, R-low, R-low-b and R-low-c to every digit he gives, for both ratios (for the run of M4 at y = 0: 0.7435, 1.1097, 0.5376, 0.6015, 0.4882 against the adjacent planes and 12.26, 5.39, 0.360, 0.389, 1.28 against the distant ones).
- **Early scalars of the existing runs** (S; O; R-mid; the run of M4): pairs up to t = 750, per unit of optical depth: 88,735, 95,387, 94,726; none; 88,496; 88,125. Mean ion K at t = 750: 13.73, 14.26, 14.25; 13.99, 13.65, 13.43; 13.14; 13.88. Rate over 300 <= t <= 700: 0.0694, 0.0709, 0.0697; 0.0762, 0.0769, 0.0701; 0.0689; 0.0641. Ions present at t = 750, in millions: 40.09, 40.21, 40.45; 40.31, 40.52, 40.16; 40.19; 40.02. Mean ion K at t = 1000: 14.58, 14.95, 15.45; 14.85, 14.41, 14.48; 14.89; 15.32. The run of M4 is the lowest of the eight in the early rate (t = -1.87 against S with the pooled scatter) and in the ions present (t = -1.09); it is inside the range of the others in the two energies and the lowest, by 0.4 percent, in the pairs up to t = 750.
- **What the gate changed:** the script of the rule only (`verdict.py`: two early scalars taken from the counters, the wording of the verdicts after the confound of section 3 was found, the signed jump taken relative to the adjacent planes). The two measuring scripts (`scalars.py`, `interface.py`) are byte-identical to those that wrote the gate files, which are therefore the files the comparison will use for the existing runs.
- **Toys on the Mac (not registered runs; CPU, MPICH, a box of 125 x 62.5 d_e, runtime 110, t_open = 93.75), run to see that a decomposition whose inner ranks own no boundary in x passes the opening of the boundaries.** [4, 1], [2, 2], [1, 1], [1, 2] and [2, 1] all run through it with exit status 0, no counter raised, and the pairs counted per rank equal to the summed counter. On [4, 1] to t = 300 the number of primaries levels off after the opening (0.94e6 at t = 125, 0.87e6 at t = 175 and t = 200), so its ends absorb. In the 16 time units after the opening the runs differ in how many particles they gain (-1.1 to +5.9 percent) and how many leptons of the pairs they lose (1.2 to 6.1 percent). An ensemble with the overdensity changed in its fourth digit: the members of [4, 1] lie close together (+5.0 to +5.9 percent in four runs), those of [2, 2] too (+0.6 to +1.3 percent in three), and the four single-rank members spread from -1.1 to +4.4 percent, covering [2, 2], [1, 2] and [2, 1]; [4, 1] lies just above them. So in the toy the decompositions differ by about as much as single-rank realisations differ among themselves; why the members of one multi-rank decomposition lie closer together than the single-rank ones is not understood. Some members were still running at this commit. The toys decide nothing about the question; they are what made me look at what the runs of one group share (section 3).

Signed **Claude 1**.
