# P2 step 1, C++ side: Bethe-Heitler startup tables. Report

Claude 1 (Claude Code cloud session), 2026-10-24. Fork `mancininilay/entity`,
branch `dev/hadronic-tables` (created from `dev/hadronic` at `b2dd625`).
Source commit `c070ed2` (physics, format, tool, test); the candidate tables,
their README and this report are in the following commit. This report is
written for a reader who has not seen the session: every statement below is
labelled as a measured fact (with the number), a convention adopted here, or
an open question. No pre-registered tolerance was changed.

The Python reference `bhref` was neither read nor reconstructed; the
comparison with it (contract section 5 item 8) is to be done elsewhere, on the
five candidate files listed in section 6.

## 1. Phase 0: environment and stock build

Facts, measured in the cloud container:

| item | value |
| --- | --- |
| `dev/hadronic` HEAD | `b2dd625` (feat(rec-mix): decouple injector segments from outflow domains) |
| gcc | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1); used for the build |
| clang | 18.1.3 (present, not used) |
| cmake | 4.4.4 |
| ninja | 1.13.2 |
| nproc | 4 |
| Kokkos | `$HOME/.entity/kokkos`, commit `89f0b12e43583d02bd19739810ceadcedf1ccd56` (5.2.1), `KokkosConfig.cmake` present, Serial + OpenMP |
| ADIOS2 | `$HOME/.entity/adios2`, commit `f5267290f06980acaecaf54688d0980958eb86bf` (2.12.1), config present, `ADIOS2_HAVE_MPI` undefined (no MPI) |
| configure | `cmake -G Ninja -D TESTS=ON -D output=ON -D precision=single -D mpi=OFF -D CMAKE_PREFIX_PATH=...` : OK |
| build | 156 targets, 3 min 52 s wall on 4 cores, no errors |
| ctest (stock) | 52/52 passed, 6.2 s |

No GCC internal compiler error was hit with GCC 13.3; the second compiler was
not needed.

After phase 1 the same configuration builds 160 targets and ctest reports
53/53 passed (9.0 s); the new test `ARCHETYPES::bh_physics` takes 2.5 s.

## 2. What was implemented and where

All new code is host-only, double precision, with no dependence on Kokkos,
the problem generator or the input parser. Line numbers refer to commit
`c070ed2`.

`src/archetypes/hadronic/bh_physics.h` (namespace `arch::bh`, 1358 lines):

| lines | content |
| --- | --- |
| 76 to 131 | `sigma_tilde_series10_delta` (Maximon eq. 10 in rho, taking delta = k - 2), `sigma_tilde_series12` (eq. 12), `sigma_tilde_delta`, `sigma_tilde`; junction at k = 4 |
| 133 to 146 | `q_k_delta`, `q_k` (Chodorowski eq. 3.7 fit; ln(omega - 1) = log1p(delta)) |
| 149 to 186 | `GaussLegendre` (nodes and weights by Newton iteration, any order 2 to 64) |
| 188 to 209 | `Quadrature` (the step parameters; `halved()` for the convergence test) |
| 211 to 372 | `Cumulatives`: Phi, phi, psi on a grid in t = ln(k - 2), analytic (k - 2)^4 start, breakpoint at k = 4, Gauss-Legendre partial-cell evaluation at any argument |
| 374 to 436 | `PhotonBath` interface (`gamma_th`, `gamma_th_beta_th`, clipping fractions, breakpoints) |
| 438 to 495 | `MonoBath` |
| 497 to 554 | `PowerlawBath` (exact normalisation) |
| 556 to 710 | `BlackbodyBath` (composite GL normalisation, discarded fractions from the tail series, closure residuals) |
| 712 to 866 | `TableBath` (two columns, log-log linear segments, exact segment integrals, optional clip, zero outside) |
| 868 to 1093 | `RateValues`, `BetheHeitlerRates`: `delta_hi_cut` (cancellation-free eps_c Gamma (1 + beta) - 2), `at_offset(x)` with x = Gamma - Gamma_th (exact-beta J, I, <D>, eps_mean), `at_gamma`, `ur_limit` |
| 1095 to 1325 | `HadronicTablesHost`, `table_upper_edge`, `build_tables` (peak search, lower edge by bisection, node placement), `validate_tables` |
| 1327 to 1355 | `BathSpec`, `make_bath` |

`src/archetypes/hadronic/bh_table_io.h` (342 lines): `Sha256` (lines 36 to
183, FIPS 180-4, known-answer test in the test suite), `format_table_data`
(185), `format_table_file` (210), `write_table_file` (268), `read_table_file`
(290, recomputes and checks the hash).

`tools/bh_tables_dump.cpp` (116 lines) and `tools/CMakeLists.txt`: the
command-line generator, built by the top-level CMake (`add_subdirectory(tools)`
added after the `output` library). It writes the file, reads it back, verifies
the hash and prints the grid summary.

`tests/archetypes/bh_physics.cpp` (947 lines), registered in
`tests/archetypes/CMakeLists.txt` as `gen_test(bh_physics)`, test name
`ARCHETYPES::bh_physics`. Functions: `test_maximon` (264), `test_phi` (296),
`test_qk` (342), `test_threshold` (391), `test_ur` (456), `test_crpropa`
(489), `test_ur_ratios` (523), `test_table` (571), `test_table_bath` (668),
`test_blackbody` (742), `test_file` (785), `design_numbers` (830). 72
registered checks print `[PASS]`/`[FAIL]` with the measured number and the
tolerance; `[DIAG]` lines are reported only.

`tests/data/hadronic/cpp-candidate/`: the five candidate tables and a README
stating that they are candidates, not fixtures.

Not touched: `srpic.hpp`, pushers, injectors, stats, `parameters/`,
`entity.schema.json`, `input.default.toml`.

## 3. Formulas, numerical choices and conventions

### 3.1 Formulas (as in the contract)

- sigma~: eq. (10) for 2 < k < 4 with rho = 2(k - 2)/(k + 2 + 2 sqrt(2k)) and
  coefficients 1, 1/2, 23/40, 11/60, 29/960; eq. (12) for k >= 4 with the
  coefficients of contract section 1. Measured relative step at k = 4:
  sigma~_12(4)/sigma~_10(4) - 1 = +1.5076e-4 (eq. 10 gives 0.3264913212, eq. 12
  gives 0.3265405433), matching the 1.5e-4 of the contract.
- q_k: eq. 3.7, (4/omega)[1 + 0.3958 L + 0.1000 L^2 + 0.00781 L^3], L =
  ln(omega - 1).
- Rates: J = (1/(2 Gamma^2 beta)) int deps n~/eps^2 [Phi(eps'_hi) -
  Phi(eps'_lo)]; I = (1/(2 Gamma beta)) int deps n~/eps^2 {[phi(hi) -
  phi(lo)] - (eps/Gamma)[psi(hi) - psi(lo)]} with psi(x) = int_2^x sigma~ q_k
  dk, i.e. the inner integral of sigma~ q_k (eps' - eps/Gamma) is written with
  the cumulatives, never by raw quadrature. eps'_lo = max(2, eps/(Gamma (1 +
  beta))) is below 2 for every eps < 2, so its cumulatives vanish; the code
  still evaluates the general bracket. eps_mean = int n~/eps [..] / int
  n~/eps^2 [..].
- Gamma_th = 1/eps_c + eps_c/4 with eps_c = eps0 (mono) or eps_max (every
  other bath, all of which have a cutoff since eps_max < 2 is enforced);
  Gamma_th beta_th = 1/eps_c - eps_c/4 exactly, used for the cancellation-free
  evaluation of eps_c Gamma (1 + beta) - 2 = eps_c x [1 + (Gamma +
  Gamma_th)/(Gamma beta + Gamma_th beta_th)], x = Gamma - Gamma_th.

### 3.2 Quadrature

- Cumulatives: variable t = ln(k - 2) from ln(1e-8) to ln(998), two segments
  [ln 1e-8, ln 2] and [ln 2, ln 998] (node exactly at k = 4), cells of width
  <= h_log_delta = 0.02 (1267 cells), 16-point Gauss-Legendre per cell; the
  start values at k - 2 = 1e-8 are the threshold laws Phi = (pi/24) d^4, phi =
  (pi/12) d^4, psi = (pi/24) d^4 (the neglected relative correction is O(d) =
  1e-8 of a 1e-32 quantity). Evaluation at an arbitrary k: cumulative at the
  cell start plus a 16-point Gauss-Legendre integral over the partial cell.
  Above k = 1000 the code throws (a relative overshoot of 1e-9 is clamped);
  below k = 2 + 1e-8 it returns the threshold laws.
- Outer photon-energy integral (broad baths): variable v = ln(eps/eps_th),
  eps_th = 2/(Gamma (1 + beta)), from max(0, ln(eps_min/eps_th)) to
  ln(eps_max/eps_th) = log1p(d_max/2); breakpoints at eps' = 4 (v = ln 2) and
  at the table nodes; panels of width <= h_log_eps = 0.02, 16-point
  Gauss-Legendre each; eps'_hi - 2 = 2 expm1(v), cancellation-free.
- Mono baths: no outer integral.
- Blackbody normalisation: composite 16-point Gauss-Legendre with panels of
  0.05 theta (at least 64), discarded tail from the series sum_n e^{-na}(a^2/n
  + 2a/n^2 + 2/n^3) (number) and the analogous energy series; closure
  (head + retained + tail)/total - 1 = 6.7e-16 (number), -3.3e-16 (energy)
  for theta = 0.1 on [0, 1.9].

### 3.3 Grid

- Upper edge: eps'_max = min(0.05 mu_r, 1000); Gamma_max = (A + 1/A)/2 with A
  = eps'_max/eps_c, or `gamma_p_max` if given and smaller. The last node is
  exactly Gamma_max; eps'_hi there equals eps'_max to 1e-12 (measured
  91.807500 for mu_r = 1836.15, 5.000000 for mu_r = 100).
- Peak of I: scan of 141 points over 14 decades of y below y_max, then
  golden-section refinement; the peak is at Gamma_max for all five baths.
- Lower edge: bisection in y on I(y) = 1e-12 I_peak to 1e-13 in y; measured
  I(Gamma_min)/I_peak = 1.000e-12 for all five tables.
- Nodes (convention, see section 7): y_min + i/256 for i = 0..256, then
  y_min + 1 + j/64 for j >= 1 while below y_max - 1e-9, then y_max appended.
  The final interval is therefore shorter than 1/64 of a decade; the spacing
  of all other nodes is exactly 1/256 and 1/64.

| table | Gamma_th | Gamma_min | Gamma_max | y_min | y_max | nodes (fine + coarse) | I_peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mono05 | 2.125 | 2.130008808 | 91.81022309 | -2.300266 | 1.952721 | 466 (257 + 209) | 1.993035e+02 |
| mono03 | 3.408333333 | 3.416953717 | 153.0141339 | -2.064473 | 2.174948 | 465 (257 + 208) | 3.321903e+02 |
| powerlaw | 1.041666667 | 1.047816677 | 30.61066927 | -2.211124 | 1.470837 | 429 (257 + 172) | 4.466219e+01 |
| blackbody | 1.001315789 | 1.012206257 | 24.17021616 | -1.962953 | 1.364905 | 406 (257 + 149) | 3.083121e+01 |
| mono05_mu100 | 2.125 | 2.126533256 | 5.05 | -2.814385 | 0.466126 | 403 (257 + 146) | 1.758894e+00 |

Blackbody theta = 0.1 on [0, 1.9]: discarded number fraction 9.345320e-7 of
2 zeta(3) theta^3, discarded energy fraction 6.955677e-6 of (pi^4/15) theta^4.
All other acceptance baths: 0.

### 3.4 Exchange file

Header keys (JSON values): `format`, `format_revision`, `producer`, `status`
(= `cpp_candidate`), `commit`, `bath` (object with `type` and every
parameter), `physical_mass_ratio`, `gamma_p_max` (number or `null`),
`eps_prime_max`, `gamma_envelope`, `gamma_th`, `gamma_min`, `gamma_max`,
`y_min`, `y_max`, `grid_variable`, `nodes_per_decade_fine`,
`nodes_per_decade_coarse`, `n_fine`, `n_coarse`, `n_nodes`, `I_peak`,
`lower_edge_fraction`, `clipped_number_fraction`, `clipped_energy_fraction`,
`quadrature` (object), `columns`, `format_columns`, `sha256_data`. Columns
`gamma_p beta_p y J I Dmean eps_mean` at `%.12e`, two spaces between columns.
The hash covers every byte of the data block; `grep -v '^#' FILE | sha256sum`
reproduces it (verified for all five files).

## 4. Acceptance items (contract section 5), measured

All numbers from `ARCHETYPES::bh_physics` at commit `c070ed2`, GCC 13.3,
Release. Verdicts: 72 registered checks, 72 passed, 0 failed.

### Item 1: sigma~ against Maximon 1968 Table 1 (tolerance 0.1%)

| MeV | k | sigma~ (this code) | Table 1 | relative deviation | verdict |
| --- | --- | --- | --- | --- | --- |
| 2.5 | 4.892368 | 0.583593 | 0.5836 | -1.17e-5 | pass |
| 3 | 5.870841 | 0.871636 | 0.8716 | +4.15e-5 | pass |
| 4 | 7.827789 | 1.415751 | 1.416 | -1.76e-4 | pass |
| 5 | 9.784736 | 1.899865 | 1.900 | -7.08e-5 | pass |
| 10 | 19.569472 | 3.647117 | 3.647 | +3.19e-5 | pass |
| 20 | 39.138943 | 5.618142 | 5.618 | +2.54e-5 | pass |
| 50 | 97.847358 | 8.377368 | 8.377 | +4.39e-5 | pass |
| 100 | 195.694716 | 10.511174 | 10.511 | +1.66e-5 | pass |

Worst deviation 1.76e-4 (the contract's independent reproduction quotes
1.8e-4). Diagnostics: sigma~/((pi/12)(k - 2)^3) = 0.99999863, 0.99986251,
0.98638065 at k - 2 = 1e-6, 1e-4, 1e-2. All anchors lie on the eq. 12 branch;
eq. 10 is checked by the junction step (section 3.1) and indirectly by item 2.

### Item 2: phi(kappa) by quadrature against the Chodorowski fits (1% budget)

- Max |phi_quad/phi_fit - 1| over 301 log-spaced kappa in [2.001, 1000]:
  5.26e-3 at kappa = 2.69. Verdict: pass.
- Sample ratios minus one: kappa = 2.5: +5.0e-3; 3: +4.9e-3; 5: +4.8e-4; 10:
  -2.4e-4; 25-: +3.7e-3; 25+: +2.2e-3; 50: +2.2e-3; 100: +4.7e-4; 1000:
  -3.1e-4.
- Fit junction continuity measured here: phi_fit(25-)/phi_fit(25+) - 1 =
  -1.54e-3 (the spec says "continuous to 0.15%").
- Threshold law: phi/((pi/12) d^4) - 1 = -7.83e-5 at d = 1e-4 (registered
  check, tolerance 1e-3): pass; 0.99999922 at d = 1e-6, 0.99221119 at d =
  1e-2. Phi/((pi/24) d^4) = 0.99993000 and psi/((pi/24) d^4) = 0.99988167 at
  d = 1e-4.

Convention: the functional forms of the fits, (pi/12)(kappa - 2)^4/(1 + sum
c_i (kappa - 2)^i) for kappa < 25 and kappa sum d_i ln^i kappa/(1 - sum f_i
kappa^-i) for kappa >= 25, are not written out in the brief; they are
transcribed from the paper (eqs. 3.14, 3.16, 3.18) and reproduce the stated
junction continuity, so they are taken as correct.

### Item 3: q_k eq. 3.7 against phi_fit'/(kappa sigma~) (tolerance 1%)

- Max |q_k/(phi_fit'/(kappa sigma~)) - 1| over 403 kappa in (2, 1000]
  including both sides of 25: 7.80e-3, at the one-sided derivative at 25+
  (25-: 5.18e-3). Verdict: pass. The contract quotes 0.77% at the same place.
- Study-guide K(eps') diagnostics (two-digit roundings, half-unit intervals):
  q_k = 1.98937, 1.88391, 1.76666, 1.40935, 0.97412, 0.64635, 0.36123,
  0.22752, 0.04431 at eps' = 2.05, 2.5, 3, 5, 10, 20, 50, 100, 1000. Eight of
  nine lie inside their rounding interval; K(20) = 0.64 is outside (q_k =
  0.646, interval [0.635, 0.645]). Reported as a diagnostic, as amendment A2
  prescribes.

### Item 4: threshold exponents (33 log nodes, x/(Gamma_th - 1) in [1e-5, 1e-4])

| bath | slope of J | slope of I | expected | tolerance | verdict |
| --- | --- | --- | --- | --- | --- |
| mono05 | 3.99992 | 3.99994 | 4 | 0.02 | pass |
| mono03 | 3.99991 | 3.99993 | 4 | 0.02 | pass |
| powerlaw | 4.99994 | 4.99995 | 5 | 0.02 | pass |
| blackbody (clipped) | 4.9999 | 5.0000 | diagnostic | | reported |

First table decade, diagnostic only (amendment A3): mono05 slopes 3.9651 (J),
3.9739 (I) over x/(Gamma_th - 1) from 4.45e-3 to 4.45e-2; mono03 3.9671,
3.9752; powerlaw 4.4004, 4.5469 over 0.148 to 1.48 (the amendment's "4.4 to
4.5"); blackbody 3.0916, 3.5718 over 8.3 to 83 (far from asymptotic since
Gamma_th - 1 = 0.0013).

### Item 5: UR limit

Exact I against Chodorowski eq. 3.11 evaluated with the phi fits, at a Lorentz
factor deep in the UR regime but inside the kernel domain (2 Gamma eps_c =
900); tolerance 1% (fit budget):

| bath | Gamma | I (exact) | I_UR (fits) | ratio - 1 | verdict |
| --- | --- | --- | --- | --- | --- |
| mono05 | 900 | 8.65028509e+02 | 8.65510086e+02 | -5.56e-4 | pass |
| mono03 | 1500 | 1.44171545e+03 | 1.44251681e+03 | -5.56e-4 | pass |
| powerlaw | 300 | 5.87766797e+02 | 5.87569931e+02 | +3.35e-4 | pass |
| blackbody | 236.84 | 4.74650254e+02 | 4.74331567e+02 | +6.72e-4 | pass |

Diagnostics of the beta -> 1 convergence with this code's own cumulatives at
the same Gamma: J/J_UR - 1 = -7.0e-8, -2.5e-8, -1.1e-6, -1.9e-6; I/I_UR - 1 =
-1.4e-6, -5.0e-7, -1.0e-5, -1.7e-5. At the table edge: mono05 (Gamma_max =
91.81, beta = 0.999941) I/I_UR(fits) - 1 = +5.2e-4, J/J_UR(own) - 1 = -1.3e-5,
I/I_UR(own) - 1 = -1.1e-4; blackbody (Gamma_max = 24.17, beta = 0.999144)
+2.5e-6, -4.9e-4, -1.6e-3.

External anchor (CRPropa `ElectronPairProduction`, blackbody, tolerance 3%).
The distributed CRPropa data file could not be downloaded: the host
`ruhr-uni-bochum.sciebo.de` named in CRPropa3's `CMakeLists.txt` is refused by
the container's proxy (HTTP 403 on CONNECT). The public generator of that
file is reachable (`CRPropa/CRPropa3-data`: `calc_pairproduction.py`,
`photonField.py` CMB class, `units.py`); it is Chodorowski eq. 3.11 with the
phi fits integrated by `scipy.integrate.quad` over the Planck spectrum at
T = 2.72548 K with CODATA 2006 constants. I ran that generator code verbatim
(as a scratch step, outside the test suite) at log10 gamma = 9.00, 9.25,
9.50, 9.75, 10.00, 10.25 and transcribed the six values into the test with
this provenance. The comparison is therefore against a regeneration of the
public table from its own public code, not against the distributed file
(open question, section 7). Conversion: (1/gamma) dgamma/dx [1/Mpc] = alpha
r_e^2 (m_e/m_p) Mpc n_ph I(gamma)/gamma, with CRPropa's constants (alpha r_e^2
(m_e/m_p) Mpc = 9.7380184875e-13 m^3, unclipped CMB n_ph = 4.1071773549e+08
m^-3), and a `BlackbodyBath` with theta = 4.5961675847e-10 clipped at 40
theta (discarded number fraction 3.0e-15).

| log10 gamma | this code [1/Mpc] | CRPropa generator [1/Mpc] | ratio - 1 | eps'_hi |
| --- | --- | --- | --- | --- |
| 9.00 | 3.225962e-05 | 3.223586e-05 | +7.4e-4 | 36.8 |
| 9.25 | 1.322240e-04 | 1.322157e-04 | +6.3e-5 | 65.4 |
| 9.50 | 3.164133e-04 | 3.162189e-04 | +6.1e-4 | 116.3 |
| 9.75 | 5.360063e-04 | 5.350970e-04 | +1.7e-3 | 206.8 |
| 10.00 | 7.207793e-04 | 7.191011e-04 | +2.3e-3 | 367.7 |
| 10.25 | 8.256458e-04 | 8.238957e-04 | +2.1e-3 | 653.9 |

Max deviation 2.33e-3; verdict pass against 3%, with the provenance caveat.

### Item 6: <D> -> 2 Gamma beta and the UR-error ratios

<D>/(2 Gamma beta) - 1 on the 33 asymptotic nodes (tolerance 1%): mono05
1.16e-5, mono03 1.30e-5, powerlaw 8.11e-6, blackbody 8.27e-6; all pass. First
table decade (diagnostic): max 5.18e-3 (mono05), 4.71e-3 (mono03), 6.19e-2
(powerlaw), 1.24e-1 (blackbody).

UR-error ratios of contract section 2 (mono baths, tolerance 1% on each
quoted number):

| Gamma, eps0 | J_UR/J | ref | <D>_UR/<D> | ref | I_UR/I (diag) | spec III.2 | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.5, 1.0 | 4.2211 | 4.2 | 1.3168 | 1.317 | 5.5583 | 5.6 | pass, pass |
| 2.5, 0.5 | 2.2022 | 2.2 | 1.0836 | 1.084 | 2.3865 | 2.4 | pass, pass |
| 3.0, 0.5 | 1.2959 | 1.3 | 1.0568 | 1.057 | 1.3694 | 1.37 | pass, pass |
| 5.0, 0.5 | 1.0337 | 1.03 | 1.0207 | 1.021 | 1.0551 | 1.06 | pass, pass |
| 10, 0.5 | 1.0039 | 1.00 | 1.0057 | 1.006 | 1.0096 | 1.01 | pass, pass |

Largest deviation from a quoted value: 5.0e-3 (J_UR/J at Gamma = 1.5, where
the reference 4.2 is itself quoted to two digits).

### Item 7

Not required in C++ by the brief; not done.

### Item 8

Not possible here (needs `bhref`); the candidate files are in section 6.

## 5. Self-convergence (contract section 4) and interpolation (spec T3)

Both quadrature steps halved (h_log_delta and h_log_eps 0.02 -> 0.01; the
cumulative grid goes from 1267 to 2533 cells), J and I re-evaluated at every
table node; tolerance 1e-8 wherever I > 1e-6 I_peak:

| table | max dJ/J (I > 1e-6 peak) | max dI/I (I > 1e-6 peak) | max over all nodes (J, I) | verdict |
| --- | --- | --- | --- | --- |
| mono05 | 5.6e-16 | 8.9e-16 | 8.9e-16, 1.1e-15 | pass |
| mono03 | 6.7e-16 | 1.1e-15 | 6.7e-16, 1.1e-15 | pass |
| powerlaw | 5.1e-15 | 6.0e-15 | 5.1e-15, 6.0e-15 | pass |
| blackbody | 7.3e-15 | 6.3e-15 | 7.3e-15, 6.3e-15 | pass |
| mono05_mu100 | 6.7e-16 | 8.9e-16 | 8.9e-16, 8.9e-16 | pass |

The changes are at double-precision round-off: the Gauss-Legendre rules are
already converged at the registered step. A three-node table bath (break at
eps = 0.5) converges the same way (5.8e-15).

Interpolation, linear in y of log J and log I, tested at the midpoints of all
intervals where I > 1e-3 I_peak (target 1%): max error of I 1.18e-4
(mono05), 1.20e-4 (mono03), 9.99e-5 (powerlaw), 1.79e-4 (blackbody), 1.18e-4
(mono05_mu100); of J up to 1.81e-4; of <D> up to 2.7e-5. All pass.

Validity: all five tables are monotone in Gamma, finite, with J, I > 0 at
every node (registered checks, pass). Table build times: < 0.01 s (mono),
0.20 s (powerlaw), 0.29 s (blackbody) on one core.

## 6. Candidate tables

Directory `tests/data/hadronic/cpp-candidate/`, produced by `bh_tables_dump`
from commit `c070ed2`, `status: cpp_candidate`. `sha256_data` is the hash of
the data block carried in the header (reproduced by `grep -v '^#' FILE |
sha256sum`); the file hash is `sha256sum FILE`.

| file | nodes | sha256_data | sha256 of the file |
| --- | --- | --- | --- |
| `mono05.txt` | 466 | `e16c0408be9ed231da1ac163c12e77b32308ded366b3f8b497af92ea650fb3a3` | `09d15dcfe5b85ca065038d9c42ca2e4f024e28ff8985da600c52fc0a6137e2a8` |
| `mono03.txt` | 465 | `881ce6ae674ce27b22f4f62b319f62abdcccaae7f9d39496d09c72268d67e7dc` | `32544c9a9ec7fb17a92988db1b43833db80c74745eebf16c1ca9b356df6677b0` |
| `powerlaw.txt` | 429 | `07c70efa86b257c6c00946751f04958b9f94e3078497f0edb3d38fb63364152a` | `192baff86e8fce1b1cc67bd3f09f7127e7b351f31d3f1ca83db6fdac97648599` |
| `blackbody.txt` | 406 | `f10280c1d787533ccc2558054efdf182068ab7b5b80d0a4dcf79aec548eb7647` | `f58e59c0e0e56a45d5e82a2fc285529ed0742d1be7daf1ce71baf360dd5611d5` |
| `mono05_mu100.txt` | 403 | `ed3ef74ad74b9a82b3f317e4d8cdb696b65cbed839cc33c3bb1b5fdb98a7ee7c` | `79f1456fb790260dd2cf31ab5fe944502000fc01fba59d39245159e4d960b98a` |

Design numbers of contract section 6 (diagnostics, from the test output):

| bath | Gamma_p | J | I | <D> | eps_mean | K/<D> (mu_r = 1836.15) |
| --- | --- | --- | --- | --- | --- | --- |
| mono05 | 3 | 1.272033e-02 | 6.561922e-02 | 5.158609 | 0.5 | 711.9 |
| mono05 | 4 | 9.195621e-02 | 5.898421e-01 | 6.414380 | 0.5 | 858.8 |
| mono05 | 6 | 3.899901e-01 | 3.261915e+00 | 8.364096 | 0.5 | 1097.6 |
| mono03 | 4 | 6.379032e-04 | 4.775289e-03 | 7.485915 | 0.3 | 735.8 |
| mono03 | 6 | 5.583494e-02 | 5.671021e-01 | 10.156759 | 0.3 | 903.9 |
| powerlaw | 1.5 | 1.176829e-03 | 2.203835e-03 | 1.872689 | 1.317340 | 490.2 |
| powerlaw | 2 | 7.412077e-03 | 1.986082e-02 | 2.679521 | 1.213306 | 685.3 |
| powerlaw | 3 | 3.410130e-02 | 1.337562e-01 | 3.922318 | 1.079208 | 936.3 |
| powerlaw | 4 | 7.404743e-02 | 3.684287e-01 | 4.975577 | 0.988729 | 1107.1 |
| powerlaw | 6 | 1.776372e-01 | 1.216768e+00 | 6.849735 | 0.866488 | 1340.3 |
| blackbody | 1.5 | 4.537816e-05 | 8.804615e-05 | 1.940276 | 1.209486 | 473.2 |
| blackbody | 2 | 7.339454e-04 | 2.143813e-03 | 2.920943 | 0.974679 | 628.6 |
| blackbody | 3 | 9.096309e-03 | 4.088303e-02 | 4.494463 | 0.768428 | 817.1 |
| blackbody | 4 | 3.197702e-02 | 1.864532e-01 | 5.830851 | 0.669489 | 944.7 |
| blackbody | 6 | 1.202305e-01 | 9.765354e-01 | 8.122197 | 0.569475 | 1130.3 |

Integrated interaction counts N_int(Gamma0) = mu_r int_{Gamma_th}^{Gamma0}
dGamma/<D>: mono05 360.9 (Gamma0 = 3) and 1175.0 (Gamma0 = 6), against the
conceptual plan's 361 and 1175; mono03 571.8 (6); powerlaw 1649.7 (3), 2687.9
(6); blackbody 1827.4 (3), 2712.7 (6); mono05 with mu_r = 100: 19.7 (3).

## 7. Ambiguities, conventions adopted, open questions

1. **Grid remainder (convention, open for the item 8 comparison).** The
   contract fixes 256 and 64 nodes per decade and "the last node exactly at
   the upper edge" but not how the non-integer remainder of the coarse decades
   is handled, nor how y_min's dependence on the 1e-12 bisection should be
   matched between implementations. Here the spacings are exactly 1/256 and
   1/64 and y_max is appended (shorter last interval). The device struct of
   spec IV (`dlgamma_fine, dlgamma, n_fine, n`) is compatible with this if the
   last interval is special-cased. If `bhref` adjusts the coarse spacing
   instead, the two node sets differ and item 8 should compare at the C++
   file's `gamma_p` column (the C++ rates are available at any Gamma through
   `BetheHeitlerRates::at_gamma`).
2. **y_min depends on I_peak and the bisection.** Two implementations will
   place y_min identically only to the precision of their peak search and
   root find (here 1e-13 in y). This shifts every node by the same tiny
   amount; it does not affect the comparison if done at matched Gamma.
3. **CRPropa anchor provenance (open question).** The anchor was taken from
   the public generator code of the CRPropa table, not from the distributed
   file (unreachable from the container, section 4 item 5). Whether this
   satisfies item 5 is for the mastermind/Nilay to decide; the test will
   accept transcribed values from the distributed file in the same place.
4. **Fit junction continuity.** The spec quotes the Chodorowski phi fits as
   "continuous to 0.15%" at kappa = 25; with the coefficients as listed the
   measured discontinuity is 0.154%. Consistent within rounding; noted because
   it sets the one-sided derivative that dominates item 3.
5. **Study-guide K(20).** Eq. 3.7 gives 0.646 where the study guide has 0.64;
   the rounding interval is missed by 0.001. Diagnostic only (amendment A2).
6. **J_UR/J reference 4.2.** Measured 4.2211, inside the 1% tolerance but the
   reference is quoted to two digits (its own rounding is 1.2%); the pass
   depends on the measured value happening to be close to 4.2.
7. **Table bath with nodes above 2.** `TableBath` refuses a node range ending
   at or above 2 unless an explicit `eps_max` clip below 2 is given, and then
   reports the discarded fractions relative to the full table integral. The
   blackbody with `eps_min = 0` is handled as the brief says (Planck regular
   at 0, normalisation over [0, eps_max], fractions relative to 2 zeta(3)
   theta^3 and (pi^4/15) theta^4).
8. **No conflict between the brief and the papers was found** in the formulas
   used: the Maximon anchors, the junction step, the UR-error ratios, the
   study-guide K values and the conceptual plan's N_int values are all
   reproduced from the formulas as written in the contract.
9. **mu_r enters only the envelope.** The mono05_mu100 table has the same
   rates as mono05 at equal Gamma (verified: identical J and I at Gamma = 3
   and 4); only Gamma_max (5.05) and hence Gamma_min and the node set change.

## 8. Not done

- Item 7 (polylogarithm closed forms of Zou et al. 2025) and item 8 (`bhref`
  comparison): not part of this step on the C++ side.
- Parser plumbing (`parameters/hadronic.{h,cpp}`, schema, `input.default.toml`):
  excluded by the brief.
- Download of the distributed CRPropa data file: blocked (see above).
- Builds with clang, MPI, CUDA, or double `real_t`: not attempted (the new
  code is `real_t`-independent and Kokkos-free; only GCC 13.3 Release on
  Linux was run).

## 9. Reproduction

```
cmake -S . -B build -G Ninja -D TESTS=ON -D output=ON -D precision=single -D mpi=OFF \
      -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos;$HOME/.entity/adios2"
cmake --build build
ctest --test-dir build -R bh_physics --output-on-failure
./build/tools/bh_tables_dump --type blackbody --theta 0.1 --eps_min 0 --eps_max 1.9 \
      --mu_r 1836.15 --out blackbody.txt
```
