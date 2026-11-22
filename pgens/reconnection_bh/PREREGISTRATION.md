# Pre-registration of the checks of stage B: reconnection with the Bethe-Heitler policy, R-off and R-low (Claude 1, local Claude Code, 2026-11-21)

Written before any run on Rusty. Brief: `2026-11-21-claude1-local-brief-bh-reconnection-shakedown.md`, A4. The same text is committed in the fork as `pgens/reconnection_bh/PREREGISTRATION.md` on `dev/bh-reconnection`, next to the scripts that compute the checks (`pgens/reconnection_bh/scripts/`); the commit and the hashes are in the report. Nothing here is changed after a run: a failure is kept and reported, and no tolerance is touched.

## 0. What is run, and what is fixed

- **Two runs**, `r-off.toml` and `r-low.toml` of `pgens/reconnection_bh`. They differ in two lines: the name of the simulation, and `emission = "bethe_heitler"` on species 2. Everything else is identical, the output included (the particle output reorders the particle arrays, so two runs with different output are not the same run).
- **Policy (R-low):** monochromatic bath 1.8, `mass_ratio` 1836.15, `pair_weight` 1, `optical_depth` 0.014333, `max_probability` and `max_loss_fraction` at their defaults (0.1, 0.04), upper edge of the tables automatic (`gamma_max = 0.0`: Gamma_p = 25.51, K = 45,008).
- **Scripts:** `check1_counters.py` to `check5_maps.py` and `run_checks.sh`, exercised end to end on a toy pair of runs (A3). The reference rate table is `data/j_table_mono18.txt`, J(Gamma_p) from `bhref` 0.3.0 (Codex 1; written by `data/make_j_table.py`) on 2961 points uniform in log10(Gamma_p - Gamma_th) up to the upper edge of the policy's tables, interpolation error below 2.1e-6. An ion of the saved output above that edge stops the script of check 2 (it would also be counted by the policy as above its tables, a failure of check 1).
- **A run that stops early** is reported as such; the checks are then evaluated on what was written and labelled partial, with no pass.
- **Facts about the output the checks rely on** (read in the source at 43d68f45 and seen in the toy runs): the output of step-based or time-based trackers is written after the first step and then every interval; the spectra are counts per logarithmic bin of gamma - 1 (K, for species of mass 1), with the first and the last bin collecting what lies outside; the particle output without tracking writes the particles of index 0, s, 2 s, ... after the dead ones are removed (s the stride); `divE` and `Charge` of the field output are not on the same points (node divergence, cell-centred moments of order 0), which is why the residual of Gauss's law comes from the generator's own diagnostic.

## 1. Counters at the end of R-low

From the last line of `bh_counters.txt` (written after the last step).

| Quantity | Criterion |
|---|---|
| particle-steps above the tables | **0** |
| pairs dropped for capacity | **0** |
| candidates refused by the loss bound | reported, with their share of all candidates (pairs + refused + dropped) |
| particle-steps above the probability bound | reported (the run stops by itself if not 0) |
| every counter of R-off, and its species 3 and 4 | **0** (the policy is never called) |

**Expectation for the refused candidates (fact from the bound of the policy, D_max of `bh_event_host.h`):** at pair weight 1 on this bath no outcome can exceed 0.04 K for Gamma_p below 9.18 (K below 15,019); the smallest admissible pair weight over that range is 1.83, at the lower edge of the tables. So the counter is expected to be exactly zero unless an ion exceeds K = 15,019. A non-zero value with all ions below that energy would contradict the bound and is reported as a defect, although the brief does not make it a criterion. (The brief expected some refusals just above threshold; there D is at most 0.21 against a bound of 0.41.)

Also reported: the largest K of each species over the run and at the end, the absorbed photon energy per pair (1.8).

## 2. Number of pairs against a prediction made outside the policy

**Prediction.** N_pred = optical_depth x integral over time of S(t), S(t) = sum over the ions of J(Gamma_p), Gamma_p = 1 + K / 1836.15, K = u^2 / (gamma + 1) (species 2 has mass 1). J from `data/j_table_mono18.txt` (`bhref`), ln J linear in log10(Gamma_p - Gamma_th); J = 0 below the first point of the table (Gamma_p - Gamma_th = 1e-6). The policy's own tables are not used.

**Ions.** The particle output of species 2: stride 5, every 20 time units (about 150 outputs; first output after the first step). S of an output = (number of ions / number of ions written) x sum over the written ions; the number of ions is the sum of the spectrum of species 2 written at the same step. Trapezoid rule over the outputs, from the first to the last.

**Observed.** The counter `pairs` of `bh_counters.txt` (every 10 steps, 0.86 time units), interpolated linearly to the times of the first and of the last output; their difference.

**Why this stride and cadence.** S is carried by the ions near the top of the spectrum (J rises by a factor 100 between K = 18 and K = 50 and by another 100 up to K = 370). With N_eff = (sum J)^2 / sum J^2 ions carrying the sum, a stride s gives a relative sampling error sqrt((s - 1) / N_eff) per output. The toy run (2.7e6 ions) had N_eff between 1.6e5 early and 1e4 late; the production box has 16 times the ions. Stride 5 keeps the sampling error per output near 0.2 to 0.6 percent if N_eff is 1e5 to 1e6. The cadence error of the toy at the equivalent cadence was 8e-4. A stride of 5 and 150 outputs cost about 36 GB per run.

**Error budget (one standard deviation each, computed by the script from the saved output with these formulas):**

- statistics: sqrt(N_pred) (the pairs are independent draws; about 0.1 to 0.3 percent for 1e5 to 1e6 pairs);
- stride: sum over the outputs of (trapezoid weight) x sqrt(f (f - 1) sum over the written ions of J^2), f = ions / ions written. This takes the sampling errors of all outputs as fully correlated, an upper bound: the output writes fixed positions of the particle arrays, which are reshuffled only when dead particles are removed, so the same ions can be written at successive outputs. The value for independent outputs is printed next to it;
- cadence: |I(all outputs) - I(every other output)| / 3, the Richardson estimate for the trapezoid rule;
- not in the budget because small: interpolation of the reference table (2e-6); the policy's lookup between its nodes, which gives J lower than `bhref` by 3e-5 to 4e-5 in this range (measured in stage A); interpolation of the counter (below 1e-4); single-precision momenta (1e-7).

**Criterion.** |N_obs - N_pred| <= 4 sqrt(statistics^2 + stride^2 + cadence^2). **If this tolerance exceeds 5 percent of N_pred the check gives no verdict** (it would not be sharp); it is then reported with its numbers. A priori estimate of the tolerance: 1 to 3 percent. The difference is also printed in units of the total with the stride error of independent outputs, for information.

**What this check can and cannot see.** It tests the rate actually applied in the run (normalisation, time step, the mapping from K to Gamma_p, the lookup) at the percent level; the test at 0.1 to 0.3 percent is A1 (uniform box). The stride sample is systematic, not random; no order of the particle arrays is known that correlates the position modulo 5 with the energy, and the toy run agreed to 0.08 percent with a stride of 10.

## 3. R-low against R-off

Pairs at about one percent of the leptons and born nearly at rest should not change the layer. With one run of each and a fixed seed there is no ensemble, and the two runs are different realisations from the first step (the policy draws from the same random pool as the injector). **The scatter taken as reference is the scatter between sub-samples of each run:** the four quadrants of the upstream region for the rate, the left and the right half of the box for the spectra (the spectra output has two spatial bins along the sheet).

- **Reconnection rate.** From the field output (every 100 time units): inflow speed towards the sheet, -sign(y) (E x B)_y / B^2, averaged over 0.4 <= |y| / (L_y / 2) <= 0.6 and |x| <= 0.8 L_x / 2, in each of the four quadrants; the rate of an output is their mean. Window: outputs with t >= runtime / 3 (about 20).
- **Spectra of the electrons and of the ions** (species 1 and 2; every 20 time units). Window: t >= 2 runtime / 3 (about 50 outputs). Scalars of a spectrum: mean K; the 99 and the 99.9 percent quantiles of K; the logarithmic slope of dN / d ln K between sigma / 2 and 2 sigma (K from 50 to 200); for the ions the fraction above threshold (K > 10.2).

For each of the ten scalars Q (the rate, four for the electrons, five for the ions): Q_bar, the mean over the window of the value for the whole box; s_ref, the standard deviation of the mean of the sub-samples (four quadrants or two halves), root mean square over the window. **Criterion: |Q_bar(R-low) - Q_bar(R-off)| <= 2 sqrt(s_ref(R-low)^2 + s_ref(R-off)^2). A comparison whose tolerance exceeds 25 percent of the value of R-off gives no verdict** (it would not be sharp) and is reported with its numbers.

**What to expect of the tolerances (toy pair, a box a quarter of the size, so few plasmoids per sub-sample):** 18 percent for the rate, 30 percent for the mean K, 36 and 64 percent for the two quantiles, 0.98 for the slope, 29 percent for the fraction of ions above threshold. In the toy only the rate was sharp. The production box holds more structures per sub-sample and the scatter should be smaller, but this check is not expected to resolve differences of a few percent; that follows from having one run of each.

**Limits, stated plainly.** The sub-samples of one run share its large structures and are not independent realisations, so s_ref can be smaller than the true run-to-run scatter; no averaging over the window is credited, which goes the other way. A comparison outside the tolerance is therefore a signal to look at the two runs (plasmoid history, time of the largest merger), not a proven effect of the pairs, and it does not stop the brief. The standard deviation over the outputs of the window is printed for the rate as a second reference. Reported without criterion: the largest ion K against time in both runs (`Kmax_2` of `bh_counters.txt`), and the cutoff reached against the expectation of the brief (order 4 sigma).

## 4. Charge: the residual of Gauss's law

**What the engine can write.** `divE` on the nodes and `Charge` at the cell centres with a shape of order 0: not the same points, and not the shape of the deposit; their difference is of the order of the charge density itself. No stock output supports the residual. **Diagnostic added to the generator copy (no change to the engine):** every 500 steps, the charge of all species on the nodes with the first-order shape of the zig-zag deposit, the binomial filter of the currents applied `current_filters` = 8 times, kappa = q0 / (B0 V0), and r = div E - kappa F[rho] on the nodes of a window that leaves out 20 d_e in x and 60 d_e in y at the boundaries (where particles are absorbed and injected and the fields are matched). It writes the largest |r| and the root mean square of r, and the same for the charge term. In the toy runs the residual is at single-precision round-off (2.5e-6 against a charge term of 0.15 after 5800 steps) with and without the policy.

**Criterion (R-low at the level of R-off).** With the level of a run defined as the largest value over its samples: level(R-low) <= 3 level(R-off), for the largest |r| and for the root mean square. Factor 3 because the round-off of two different realisations differs by a factor of order one, while a single lepton born with a wrong previous position leaves a charge of the order of one unit.

**Reported: the sensitivity.** One particle of unit charge leaves on a node at most U = kappa (C(16, 8) / 4^8)^2 = 0.0386 kappa after the filter. The levels are given in units of U; the check resolves a single lost or created charge in the window when the level of R-off is well below 1 U. (Toy: 5e-3 U.)

## 5. Where the pairs are, and their spectrum

At the field outputs nearest to runtime / 4, runtime / 2, 3 runtime / 4 and at the last one:

- two maps on a grid 16 times coarser than the cells: the density of the Bethe-Heitler leptons (N_3 + N_4 of the field output), and the number of written ions above threshold (K > 10.2) from the particle output nearest in time;
- **reported numbers:** the correlation coefficient of the two maps, and the share of the lepton density in coarse cells that hold at least one written ion above threshold. **Expectation, not a criterion of the policy:** correlation above 0.5 and share above 0.9 at each time (toy: 0.87 to 0.92 and 0.98 to 1.00). A value outside is a finding about the transport of the pairs, to be understood before a third run;
- the spectra of the two lepton species at the same times with those of the electrons and ions at the end (figure and table), their number, mean K and the K at 50, 90 and 99 percent. Expectation of the brief, for orientation: born at Lorentz factors of 1 to 3 (stage A, uniform box: mean 1.18 to 1.71 for Gamma_p of 1.05 to 1.5, tail to 8.6 at 1.5); in the layer they are then accelerated like any lepton, so the spectrum of the leptons present is not the birth spectrum.

The figures are drawn from the files the script writes (`check5_maps.npz`, `check5_spectra.txt`); the Python of Rusty has no matplotlib, so they are drawn on the Mac from these compact files.

## 6. What is reported besides the five verdicts

Job identifiers and GPU-hours; the ion spectrum reached (cutoff, range of Gamma_p) against the expectation; the number of pairs per primary lepton; the largest number of particles per species against the capacities; anything that broke; what should change for a third run with about a hundred times the optical depth.

Signed **Claude 1**.
