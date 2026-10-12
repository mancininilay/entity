# `rec_mix`: pair-ion reconnection for P1.3 (and P2's P7)

Entity port of the Tristan v2 user files `user_rec-mix.F90` (active ions, used for
the p1.1 production runs) and `user_rec-mix_testprtl.F90` (passive test ions).
Specification: `Resources/p1.3/technical-spec.md` (II.2, III.1 to III.13, IV).
Author: Claude. Reviewer: Codex (A3).

## Build and inputs

```
cmake -B build -D pgen=rec_mix -D precision=single -D output=ON [-D mpi=ON] ...
python3 pgens/rec_mix/make_input.py --name c2a --layout active --sigma 40 --q 0.1 --mu 25 --decomposition 4 8
python3 pgens/rec_mix/make_input.py --name c2d --layout testprtl --sigma 160 --test-masses 25 100 400 --decomposition 4 8
mpirun -n <N> build/src/entity.xc -input c2a.toml
```

`make_input.py` derives `[scales]` from the physical inputs (spec III.1) and writes
the 9-species table, MATCH widths, outputs and checkpoints. The generator refuses
`[scales]` that disagree with the physical inputs (relative 1e-4).

* **Pusher.** Every species gets `pusher = "Vay"` (`--pusher`; D3). Entity's
  default would be Boris.
* **Capacity** (Codex IC2). Entity allocates `maxnpart` for every species on
  every domain. `make_input.py` therefore needs the decomposition
  (`--decomposition N1D N2D`, both > 0) and sizes each species for the largest
  local domain: upstream species `--reserve` (3) times their nominal per-cell
  density, sheet species `--sheet-reserve` (2) times the rejection-sampling trial
  density (eta times upstream), tracers both halves. It prints the particle-array
  memory per domain (2D single precision: 54 B per slot with one integer
  payload, two more int slots with MPI tracking; sorting and communication
  buffers come on top) and refuses more than `--max-gib-per-domain` (30).
  `--maxnpart` overrides the estimate. Example: C2-a on [4, 8] domains,
  15.1 GiB per domain. The generator writes the fill fraction npart/maxnpart
  (max over ranks) of every species into each user output and warns above 80%;
  the injector and the load stop cleanly when a species would exceed its
  capacity ("capacity guard").

Requirements checked at start-up: 2D Minkowski; `extent = [[0, N1], [0, N2]]` with
unit cells (Tristan cell units); x1 PERIODIC at start (or MATCH + ABSORB when
resuming past t_open), x2 MATCH + ABSORB; 9 species with the layout's masses and
charges; every species pushed with Vay (unless `allow_any_pusher`);
`n_payloads_int >= 1` for every species (the control-volume flag uses the last
integer slot); tracers (`testprtl`) must have `deposit = false` (fork change
CH1). After the load: zero net charge in every cell (C1.3a; co-located pairs).

**Restarts.** Entity takes `[setup]` from the checkpoint's meta file, not from
the restart input (`src/framework/simulation.cpp`), so every key below is
fixed at the first start; `[output]`, `[checkpoint]`, the runtime and the
boundaries come from the restart input.

## Coordinates (P1.3 D29)

(x1, x2, x3) = (y_T, x_T, -z_T): x1 is the outflow direction, x2 the inflow
direction, the sheet sits at x2 = y0 = N2/2. B_x1 = tanh((x2 - y0)/a). Species
drift with u_x3 = -sgn(Q) u_d, so the sheet current is along -x3. The outflow
MATCH target has E_x3 = -0.1 B0 and B_x2 = 0.1 tanh((x1 - x0)/a).

## Species

| # | `active` | `testprtl` |
|---|---|---|
| 1, 2 | e-, e+ of the sheet | e-, e+ of the sheet |
| 3 | ion of the sheet (mass mu) | tracer a (`deposit = false`) |
| 4, 5 | e-, e+ lower side (x2 < y0) | same |
| 6 | ion lower side | tracer b |
| 7, 8 | e-, e+ upper side | same |
| 9 | ion upper side | tracer c |

Tracers are loaded on both sides at 0.5 ppc0 per mass (as Tristan), absent from
the sheet, and loaded after all depositing species so that the depositing
species draw the same random numbers with or without them.

## `[setup]` keys

| key | default | meaning |
|---|---|---|
| `layout` | "active" | "active" or "testprtl" |
| `sheet_init` | "legacy" | "legacy": T_cs = 0.5 sigma0 (1 + b_g^2)/(eta D) (p1.1); "balanced": T_cs = 0.5 sigma0 Gamma_d/eta, so that the transverse stress eta n0 T_cs/Gamma_d equals B0^2/8pi |
| `sigma`, `q`, `mu` | | p1.1 sigma (total mass), q = n_i/n_e-, mu (active only) |
| `test_masses`, `test_temperatures` | species masses, theta_up | tracers (testprtl); temperatures are physical kT |
| `dup_cells` | 5 | d_up in cells |
| `theta_up` | 0.01 | upstream kT/(m_e c^2), same physical kT for all species |
| `cs_width`, `cs_overdensity` | 400, 3 | a in cells, eta |
| `dip_width`, `sheet_extent` | 5, 10 | dip radius and sheet half-width in units of a |
| `b_guide` | 0 | Tristan b_guide (B_x3 = -b_guide) |
| `t_open_tlc` (or `t_open`) | 1.35 | opening time of the outflow walls in t_lc (or code time) |
| `injector_padding`, `ds_abs` | 12, 10 | forced >= nfilter + 4 as in Tristan |
| `mixing_threshold` | 0.01 | mask M > threshold |
| `usr_interval`, `line_interval`, `cv_interval` | 500, 100, 1 | steps |
| `cv_margin` | 10 | cells |
| `line_halfwidth` | 3 | midline rows j0 - 3 ... j0 + 2 (cell centres) |
| `spec_min`, `spec_max`, `spec_num` | 1e-3, 1e5, 300 | log bins of gamma - 1 |
| `bulk_window` | 2 | half-width of the top-hat box for bulk velocities (Tristan smooth_window), at most N_GHOSTS = 2 |
| `bulk_nmin` | 20 | own-species box occupancy below which a particle counts in `Nlowocc` |
| `inj_avg_cells` | 11 | rows averaged by the injector |
| `inject`, `load_tracers` | true, true | test switches (C0.4, C1.7) |
| `allow_any_pusher` | false | test switch: accept species not pushed with Vay |
| `cv_payload_slot` | last | integer payload slot of the CV flag |

## Algorithms (ports of the Tristan routines)

* **Load** (Tristan 119-200): co-located (e-, e+) at 0.5 ppc0 (1 - q) and (e-, ion)
  at 0.5 ppc0 q per half box; sheet within |x2 - y0| <= 10a, trials at eta times
  the upstream density accepted with probability
  cosh^-2((x2 - y0)/a) (1 - exp(-r^2/(5a)^2)); number of trials per subdomain
  Poisson(n) for n < 10, else ceiling(n) (`fillRegionWithThermalPlasma`).
* **Injector** (Tristan 264-345, every step): on subdomains containing an
  injection cell (x2 = padding or N2 - padding - 1), count the upstream species
  (active 4-9, testprtl 4, 5, 7, 8) in the 11 rows next to the plane over the
  subdomain's x1 extent; if ppc0 > average, inject 0.5 (ppc0 - avg)(1 - q) pairs
  and 0.5 (ppc0 - avg) q (e-, ion) per cell (testprtl: pairs and 0.5 (ppc0 - avg)
  per tracer mass) in the 1-cell layer, with the same rounding rule. Capacity is
  checked before any write (clean error). All payload slots are zeroed.
* **Padding** (Tristan 347-372): particles with x2 < padding get u = (0, -0.1, 0),
  with x2 >= N2 - padding u = (0, 0.1, 0); beyond padding - nfilter they are removed.
* **Boundaries** (D23): Entity MATCH. x2 walls: initial field, E -> 0. x1 walls:
  periodic until t_open, then MATCH to the reconnected targets and ABSORB.
  For a restart past t_open, give x1 as MATCH + ABSORB in the restart input (no
  step runs with periodic x1; refused before t_open). With periodic x1 in the
  restart input the walls re-open at the first post-step, with a warning.

## Outputs (`<simulation.name>/usr/`, little-endian; format 3)

* `recmix.meta.json`: format version, run id, species (roles, mass, charge,
  deposit flag, pusher, maxnpart), inputs, derived scales, mask definition, bins,
  moment names, bulk-estimator settings, CV box and record layout, midline
  layout, git hash. Rewritten at every process start.
* `segments.jsonl` (appended): one line per process start: `run_id` (52 bits,
  random), `start_label` (0 for a fresh start, the checkpoint label when
  resuming), `resuming`, time, checkpoint path, git hash.
* `fields_manifest.jsonl` (appended): `{label, run_id}` for each Entity field
  output that contains a rec_mix custom field (`mixing`, `mask`, `count_s`).
* `recmix.<label>.bin` every `usr_interval` (label = Entity's output label,
  step + 1): `"P13RMX02"`, int64 label, float64 time, uint64 run_id, int32
  (nspec, nbins, nmom, 0), float64 (masked cells, total cells), float64
  hist[nspec][nbins], float64 moms[nspec][nmom], float64 fill[nspec]
  (npart/maxnpart, max over ranks). Moments (double, over masked cells unless
  noted): N, K = sum w m (gamma - 1), G = sum w gamma, U1-3 = sum w u,
  B1-3 = sum w beta, Nunder/Eunder and Nover/Eover (particles clamped into the
  end bins and their energy), Nskip (gamma - 1 = 0, outside the log histogram),
  Kb_own and Kb_all (bulk kinetic energy, below), Ntot and Ktot (whole domain,
  unmasked), Kb_own_raw (without the variance correction), Nlowocc (masked
  particles whose own-species box holds fewer than `bulk_nmin` particles),
  Kb_own_w1 and Nlowocc_w1 (the same with window 1, for the window sensitivity),
  Klowocc and Kblowocc (kinetic and bulk energy of the masked particles in
  under-occupied boxes, also with window 1), Kb_own_half (box means from the
  particles with dx1 < 1/2, weighted by the full cell counts: thinning test).
  Format 1 (`"P13RMX01"`: no run id, no fill) is still read.
* `cvbudget.bin` (appended): records of 61 float64: label, time, segment_start,
  run_id, then per species (N_cv, I_cv = sum_CV m (gamma - 1), in_E, out_E, in_N,
  out_N since the previous record), U_EM in the CV, the integrated outward
  Poynting flux and the integrated field-side work of the (filtered) current
  W_J. Records at every user output and at every process start
  (segment_start = 1). Energies in m_e c^2 per unit-weight macroparticle:
  U = (sigma0 ppc0/2) sum(e^2 + b^2) over the Yee components owned by CV cells,
  flux = sigma0 ppc0 (e x b).n dl dt. W_s = Delta I_s - (in_E - out_E).
  Residuals over the depositing species (C1.14): r_F = Delta U + flux + W_J,
  r_C = sum_s W_s - W_J, r_total = r_F + r_C.
* `midline.<label>.bin` every `line_interval`: `"P13MID02"`, int64 label,
  float64 time, uint64 run_id, int32 (N1, nrows, 6, j_first), float32
  rows[nrows][6][N1] (E1 E2 E3 B1 B2 B3 at cell centres, same arithmetic as the
  field output), float32 E3[N1] on the nodes (i, j0) and B2[N1] at (i + 1/2, j0)
  of the midplane node line.
* Custom field outputs `mixing`, `mask` and `count_<s>` (raw particle count of
  species s, for C1.16).

**Provenance after restarts** (Codex IC3). A file or record written by the
segment with `run_id` k at label s is valid if and only if s <= `start_label`
of every later segment in `segments.jsonl`. This rejects the outputs of an
abandoned future after a rollback restart, also before the new segment has
caught up and after a cadence change, and everything of an earlier run when
a fresh run reuses the directory. `p13ana.readers` applies the rule to the
spectra, midlines, CV records and (through `fields_manifest.jsonl`) to
Entity's field files. Entity's particle, spectra and stats files carry no
writer id: move the later ones aside after a rollback before reading them.

### Mask

D = particle count per cell of the side species (active 4, 5, 6 vs 7, 8, 9;
testprtl 4, 5 vs 7, 8), smoothing 0; M = 2 min(D_lo, D_up)/(D_lo + D_up);
mask = M > threshold. This is the correct total-number mask (P13-D13), not
Tristan's 3N_e + 2N_i + N_p.

### Bulk velocities and bulk-corrected energies (Nilay, 2026-10-04)

Tristan's `velx` was the mass-weighted velocity of all depositing species
(`computeFluidVelocity`, top-hat over (2 smooth_window + 1)^2 cells). rec_mix
outputs the mean 3-velocity of **each species**:

* in-code, for eps-tilde (SM S2: eps-tilde_s = eps_s - m_s(<gamma_b,s> - 1)):
  beta_b,s(cell) = sum beta / N over the species' particles in a top-hat box of
  half-width `bulk_window`; Kb_own = sum over masked cells of N_s m_s (gamma_b,s - 1).
  Kb_all is the same with Tristan's all-species mass-weighted velocity (continuity);
* as field output: Entity's moment `V_<s>` for a single species is its mean
  3-velocity; `make_input.py` requests `V_1` ... `V_9` and the all-depositing
  `V_...` (Tristan velx) with the same top-hat smoothing (`order = 4`,
  `method = "const"`, i.e. 5 x 5 cells).

What the bulk-energy status means (Codex IC5, RV1). The analysis applies an
occupancy screen, not an accuracy certificate: eps-tilde of a population is
reported (`occupancy_ok`) only if at most 2% of its masked particles sit in
boxes with fewer than `bulk_nmin` (20) particles and the energy those boxes can
carry, max(Klowocc, Kblowocc), is at most 2% of K - Kb_own; otherwise NaN with
status `low_occupancy`. No occupancy rule bounds the finite-sample bias of the
estimator in well-occupied boxes (gamma of the box-mean velocity is nonlinear,
steeply so near beta = 1); the thinning change, from Kb_own_half, measures that
dependence on the actual distribution and is reported with every value, as is
the change from window 1 to `bulk_window`. The bulk frame is defined on a
(2w + 1)^2-cell top-hat box (w = 2: 5 x 5 cells, Tristan's convention); the
own-species eps-tilde is a secondary observable until validated on pilot data
(tech spec III.8).

## Analysis and tests

* `analysis/`: the `p13ana` package (readers, partition, CV budget, zones,
  inflow, regimes, neutrality), versioned with the generator; data-free tests
  `python3 -m pytest analysis/tests` and the C1 check scripts.
* `tests/harness.py`: the reproducible C0/C1 harness (inputs from
  `make_input.py` with explicit seeds, runs, pre-registered tolerances,
  `results.json`). See its docstring for the builds it needs.

Specification and status: `Resources/p1.3/technical-spec.md`, V.2 and Part VIII.
