# P2 step 1, C++ side: Bethe-Heitler startup tables. Report

Claude 1 (Claude Code cloud session), 2026-10-24. Fork `mancininilay/entity`,
branch `dev/hadronic-tables` (created from `dev/hadronic` at `b2dd625`).

Round 1: source commit `c070ed2` (physics, format, tool, test), tables and
report at `1ebd2c4`. Round 2 (referee findings R1 to R8, this revision of the
report): source commit `5aadf9d`, regenerated tables and this report in the
commit that follows it. Section 10 is the referee resolution log, section 11
the round-2 test summary.

This report is written for a reader who has not seen the session: every
statement below is labelled as a measured fact (with the number), a convention
adopted here, or an open question. No pre-registered tolerance was changed.

The Python reference `bhref` was neither read nor reconstructed; the
comparison with it (contract section 5 item 8) is done elsewhere, on the five
candidate files listed in section 6.

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

With the branch (round 1 and round 2) the same configuration builds 160
targets and ctest reports 53/53 passed; the new test `ARCHETYPES::bh_physics`
takes 2.3 to 2.5 s.

## 2. What was implemented and where

All new code is host-only, double precision, with no dependence on Kokkos,
the problem generator or the input parser. Line numbers refer to commit
`5aadf9d` (round 2).

`src/archetypes/hadronic/bh_physics.h` (namespace `arch::bh`, 1517 lines):

| lines | content |
| --- | --- |
| 79 to 134 | `sigma_tilde_series10_delta` (Maximon eq. 10 in rho, taking delta = k - 2), `sigma_tilde_series12` (eq. 12), `sigma_tilde_delta`, `sigma_tilde`; junction at k = 4 |
| 136 to 149 | `q_k_delta`, `q_k` (Chodorowski eq. 3.7 fit; ln(omega - 1) = log1p(delta)) |
| 153 to 166 | `parse_number_strict` (whole token consumed, finite) |
| 169 to 206 | `GaussLegendre` (nodes and weights by Newton iteration, any order 2 to 64) |
| 208 to 229 | `Quadrature` (the step parameters; `halved()` for the convergence test) |
| 231 to 392 | `Cumulatives`: Phi, phi, psi on a grid in t = ln(k - 2), analytic (k - 2)^4 start, breakpoint at k = 4, Gauss-Legendre partial-cell evaluation at any argument |
| 394 to 470 | `PhotonBath` interface (`gamma_th`, `gamma_th_beta_th`, clipping fractions, breakpoints, `number_normalization`, `mean_eps`, `source_file`) |
| 472 to 575 | `MonoBath`, `json_number` (shortest round-trip decimal), `json_string` (escaped JSON literal) |
| 577 to 647 | `PowerlawBath` (exact normalisation; `power_integral` at 633, the stable a^t L expm1(tL)/(tL) form) |
| 649 to 812 | `BlackbodyBath` (composite GL normalisation, discarded fractions from the tail series, closure residuals, mean energy) |
| 814 to 1000 | `TableBath` (two columns, strict row parsing naming file and line, log-log linear segments, exact segment integrals, optional clip, zero outside) |
| 1002 to 1227 | `RateValues`, `BetheHeitlerRates`: `delta_hi_cut` (cancellation-free eps_c Gamma (1 + beta) - 2), `at_offset(x)` with x = Gamma - Gamma_th (exact-beta J, I, <D>, eps_mean), `at_gamma`, `ur_limit` |
| 1229 to 1484 | `HadronicTablesHost`, `table_upper_edge` (envelope guards), `build_tables` (peak search with `gamma_at_I_peak`, lower edge by bisection, node placement), `validate_tables` |
| 1486 to 1515 | `BathSpec`, `make_bath` |

`src/archetypes/hadronic/bh_table_io.h` (693 lines): `Sha256` (39 to 175,
FIPS 180-4, unchanged since round 1, known-answer test in the test suite),
JSON helpers (`json_unescape` 177, `json_to_number` 246, `json_string_array`
258, `json_number_checked` 292), `TableFileMeta` (303), `format_table_data`
(339), `format_table_file` (364, contract 1.4 schema), `write_table_file`
(442), `TableFileContents` (458), `required_header_keys` (481),
`parse_table_file` (523, literal-byte hashing and content validation),
`read_table_file` (681).

`src/archetypes/hadronic/bh_options.h` (163 lines): `DumpOptions` and
`parse_dump_options` (38), the strict option parser of the tool.

`tools/bh_tables_dump.cpp` (95 lines) and `tools/CMakeLists.txt`: the
command-line generator, built by the top-level CMake (`add_subdirectory(tools)`
after the `output` library; its target drops the directory-wide Kokkos and
ADIOS2 link dependencies). It writes the file, reads it back through the
validating reader and prints the grid summary; option errors exit with 2.

`tests/archetypes/bh_physics.cpp` (1682 lines), registered in
`tests/archetypes/CMakeLists.txt` as `gen_test(bh_physics)` with the compile
definition `BH_TEST_DATA_DIR`, test name `ARCHETYPES::bh_physics`. Functions:
`test_maximon` (341), `test_phi` (373), `test_qk` (419), `test_threshold`
(468), `test_ur` (533), `test_crpropa` (566, reads the fixture),
`test_ur_ratios` (604), `test_table` (652), `test_table_bath` (749),
`test_blackbody` (823), `test_file` (894, round trip and the R2 negative
tests), `test_foreign_reader` (1140), `test_envelope` (1274), `test_options`
(1379), `test_power_integral` (1496), `design_numbers` (1558). The checks are
counted in two classes: **contract items** (the pre-registered acceptance
items of contract section 5, the validity and self-convergence requirements of
section 4 and the interpolation target T3: 55 checks) and
**implementation-chosen checks** (reader and writer behaviour, parsing,
regression guards, consistency checks, 89 checks), printed with the prefix
`(impl)`. `[DIAG]` lines are reported only.

`tests/data/hadronic/anchors/crpropa-lossrate-CMB-6rows.txt`: the six CRPropa
anchor rows with their provenance (section 4, item 5).

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
  beta))) is below 2 for every eps < 2, so its cumulatives vanish. The mono
  branch uses this directly: it evaluates only the upper cumulatives
  Phi(eps'_hi), phi(eps'_hi), psi(eps'_hi) (zero lower cumulatives). The
  broad-bath branch evaluates the general bracket, where the lower cumulatives
  return zero for eps'_lo <= 2. eps_mean = int n~/eps [..] / int n~/eps^2 [..].
- Gamma_th = 1/eps_c + eps_c/4 with eps_c = eps0 (mono) or eps_max (every
  other bath, all of which have a cutoff since eps_max < 2 is enforced);
  Gamma_th beta_th = 1/eps_c - eps_c/4 exactly, used for the cancellation-free
  evaluation of eps_c Gamma (1 + beta) - 2 = eps_c x [1 + (Gamma +
  Gamma_th)/(Gamma beta + Gamma_th beta_th)], x = Gamma - Gamma_th.
- Power-segment integral (powerlaw normalisation and mean energy, table-bath
  segments): int_a^b eps^p deps = a^t L expm1(tL)/(tL) with t = p + 1, L =
  ln(b/a); for |tL| < 1e-18 the limit a^t L is used (round 2, R4).

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

### 3.3 Grid and envelope

- Envelope guards (round 2, R1): `physical_mass_ratio` must be finite and
  positive; the kernel cap eps'_max = min(0.05 mu_r, 1000) must exceed 2,
  otherwise there is no Bethe-Heitler-active interval and `build_tables`
  throws naming mu_r and the cap (mu_r = 1 gives 0.05, mu_r = 40 gives
  exactly 2: both refused). An explicit `gamma_p_max` must be finite and > 1,
  and Gamma_max must exceed Gamma_th.
- Upper edge: Gamma_max = (A + 1/A)/2 with A = eps'_max/eps_c, or
  `gamma_p_max` if given and smaller. The last node is exactly Gamma_max;
  eps'_hi there equals eps'_max to 2.2e-16 relative (measured 91.807500 for
  mu_r = 1836.15, 5.000000 for mu_r = 100, 999.95 for mu_r = 19999, 1000 for
  mu_r = 20001).
- Peak of I: scan of 141 points over 14 decades of y below y_max, then
  golden-section refinement; the peak is at Gamma_max for all five baths
  (`gamma_at_I_peak` is set to Gamma_max exactly when the refined position is
  within 1e-9 in y of y_max).
- Lower edge: bisection in y on I(y) = 1e-12 I_peak to 1e-13 in y; measured
  I(Gamma_min)/I_peak = 1.000e-12 for all five tables.
- Nodes (convention, adopted in contract 1.4): y_min + i/256 for i = 0..256,
  then y_min + 1 + j/64 for j >= 1 while below y_max - 1e-9
  (`endpoint_coalescing`), then y_max appended with Gamma set exactly to
  Gamma_max. The final interval is therefore shorter than 1/64 of a decade;
  the spacing of all other nodes is exactly 1/256 and 1/64.

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

### 3.4 Exchange file (contract 1.4 section 2)

Header lines `# key: value` with strict JSON values, one key per line, in
this order: `format` ("bh-rate-table"), `format_revision` (1), `producer`,
`version` (git commit), `status` (`cpp_candidate`), `physical_mass_ratio`,
`photon_field` (object with `type`, every parameter, and `file`,
`file_sha256` for a table bath), `gamma_p_max_input` (number or `null`),
`eps_prime_max`, `gamma_envelope`, `gamma_th`, `gamma_min`, `gamma_max`,
`y_min`, `y_max`, `grid_variable`, `lower_edge_fraction`,
`first_decade_intervals_per_decade`, `other_intervals_per_decade`,
`endpoint_coalescing`, `first_decade_node_count`, `node_count`, `I_peak`,
`gamma_at_I_peak`, `number_normalization`, `mean_photon_energy`,
`discarded_number_fraction`, `discarded_energy_fraction`, `quadrature`
(object), `columns`, `data_sha256`. No blank lines anywhere. The data block
starts at the first line not beginning with `#` and runs to the end of the
file; each line is seven numbers `gamma_p beta_p y J I Dmean eps_mean` at
`%.12e` separated by one space and terminated by a newline. `data_sha256`
covers every byte of the data block; `grep -v '^#' FILE | sha256sum`
reproduces it (verified for all five files). String values are JSON-escaped;
non-finite numeric metadata is refused by the writer and by the reader.

Reader (round 2, R2): the header/data boundary is found once; the remaining
bytes are hashed verbatim and parsed separately; a missing or mismatched
digest throws; then the content is validated (required keys, format and
revision, canonical columns, exactly seven finite tokens per row, node_count
equal to the row count, at least two rows, Gamma strictly increasing, J and I
strictly positive). Unknown extra keys are ignored.

## 4. Acceptance items (contract section 5), measured

All numbers from `ARCHETYPES::bh_physics` (unchanged between round 1 at
`c070ed2` and round 2 at `5aadf9d`, since no physics changed), GCC 13.3,
Release. Verdicts: 55 contract items, 55 passed, 0 failed; 89
implementation-chosen checks, 89 passed, 0 failed.

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
  5.26e-3 at kappa = 2.69. Verdict: pass (contract item).
- Sample ratios minus one: kappa = 2.5: +5.0e-3; 3: +4.9e-3; 5: +4.8e-4; 10:
  -2.4e-4; 25-: +3.7e-3; 25+: +2.2e-3; 50: +2.2e-3; 100: +4.7e-4; 1000:
  -3.1e-4.
- Fit junction continuity measured here: phi_fit(25-)/phi_fit(25+) - 1 =
  -1.54e-3 (the spec says "continuous to 0.15%").
- Threshold law: phi/((pi/12) d^4) - 1 = -7.83e-5 at d = 1e-4 against an
  **implementation-chosen** tolerance of 1e-3 (the contract states the
  threshold law but registers no tolerance for it): pass. Diagnostics:
  0.99999922 at d = 1e-6, 0.99221119 at d = 1e-2; Phi/((pi/24) d^4) =
  0.99993000 and psi/((pi/24) d^4) = 0.99988167 at d = 1e-4.

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

The mu_r = 100 repeat of mono05 **skips items 4, 5 and 6**: J(Gamma) and
I(Gamma) do not depend on mu_r (it enters only Gamma_p(K) and the envelope),
so the rate functions of mono05_mu100 are the same functions as mono05 and
the asymptotic-node and UR checks would repeat the mono05 numbers identically;
the repeat is tested for what changes, the grid (Gamma_max = 5.05, Gamma_min,
node set), its validity, its self-convergence and its interpolation error.

First table decade, diagnostic only (amendment A3): mono05 slopes 3.9651 (J),
3.9739 (I) over x/(Gamma_th - 1) from 4.45e-3 to 4.45e-2; mono03 3.9671,
3.9752; powerlaw 4.4004, 4.5469 over 0.148 to 1.48 (the amendment's "4.4 to
4.5"); blackbody 3.0916, 3.5718 over 8.3 to 83 (far from asymptotic since
Gamma_th - 1 = 0.0013).

### Item 5: UR limit

Exact I against Chodorowski eq. 3.11 evaluated with the phi fits, at a Lorentz
factor deep in the UR regime but inside the kernel domain (2 Gamma eps_c =
900); tolerance 1% (fit budget). These are **kernel-limit tests outside the
production envelope**: the production tables stop at eps'_hi = 0.05 mu_r =
91.8 (Gamma_max of section 3.3), while these checks evaluate the same
functions at eps'_hi ~ 900, inside the q_k fit domain but ten times above the
static-source validity envelope. They test the beta -> 1 limit of the
formulas, not a production operating point.

| bath | Gamma | I (exact) | I_UR (fits) | ratio - 1 | verdict |
| --- | --- | --- | --- | --- | --- |
| mono05 | 900 | 8.65028509e+02 | 8.65510086e+02 | -5.56e-4 | pass |
| mono03 | 1500 | 1.44171545e+03 | 1.44251681e+03 | -5.56e-4 | pass |
| powerlaw | 300 | 5.87766797e+02 | 5.87569931e+02 | +3.35e-4 | pass |
| blackbody | 236.84 | 4.74650254e+02 | 4.74331567e+02 | +6.72e-4 | pass |

Diagnostics of the beta -> 1 convergence with this code's own cumulatives at
the same Gamma: J/J_UR - 1 = -7.0e-8, -2.5e-8, -1.1e-6, -1.9e-6; I/I_UR - 1 =
-1.4e-6, -5.0e-7, -1.0e-5, -1.7e-5. At the table edge (inside the envelope):
mono05 (Gamma_max = 91.81, beta = 0.999941) I/I_UR(fits) - 1 = +5.2e-4,
J/J_UR(own) - 1 = -1.3e-5, I/I_UR(own) - 1 = -1.1e-4; blackbody (Gamma_max =
24.17, beta = 0.999144) +2.5e-6, -4.9e-4, -1.6e-3.

External anchor (CRPropa `ElectronPairProduction`, blackbody, tolerance 3%).
The six anchor values live in the fixture
`tests/data/hadronic/anchors/crpropa-lossrate-CMB-6rows.txt`: rows of the
distributed file `lossrate_CMB.txt` at log10 gamma = 9.00, 9.25, 9.50, 9.75,
10.00, 10.25, units 1/Mpc, distributed-file SHA-256
`3ceafd2d613de596c6917b0fbb0c29fde2fe97ae81b6ccc49ab242dae8f142c7`,
CRPropa3-data commit `d34b15be5436423de1be5c5745b3c4a686cd46ae`, transcribed
from the distributed file and verified by Codex 1 on 2026-10-24. (History: in
round 1 the distributed file was unreachable from the build container, and
the six values were regenerated by running CRPropa's public generator code;
the referee verified them against the distributed file, where they match
exactly at its printed precision, so the fixture now carries the file's
provenance and the test reads the fixture instead of embedding values.)
Conversion: (1/gamma) dgamma/dx [1/Mpc] = alpha r_e^2 (m_e/m_p) Mpc n_ph
I(gamma)/gamma, with CRPropa's CODATA 2006 constants (alpha r_e^2 (m_e/m_p)
Mpc = 9.7380184875e-13 m^3, unclipped CMB n_ph = 4.1071773549e+08 m^-3), and
a `BlackbodyBath` with theta = 4.5961675847e-10 clipped at 40 theta
(discarded number fraction 3.0e-15).

| log10 gamma | this code [1/Mpc] | CRPropa [1/Mpc] | ratio - 1 | eps'_hi |
| --- | --- | --- | --- | --- |
| 9.00 | 3.225962e-05 | 3.223586e-05 | +7.4e-4 | 36.8 |
| 9.25 | 1.322240e-04 | 1.322157e-04 | +6.3e-5 | 65.4 |
| 9.50 | 3.164133e-04 | 3.162189e-04 | +6.1e-4 | 116.3 |
| 9.75 | 5.360063e-04 | 5.350970e-04 | +1.7e-3 | 206.8 |
| 10.00 | 7.207793e-04 | 7.191011e-04 | +2.3e-3 | 367.7 |
| 10.25 | 8.256458e-04 | 8.238957e-04 | +2.1e-3 | 653.9 |

Max deviation 2.33e-3; verdict pass against 3%.

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

Not possible here (needs `bhref`); the candidate files are in section 6. The
mastermind's precheck and the referee's review (round 2 brief, section 0)
report agreement with the independent reference at 1e-12 (core) and 2e-10
(tail) for all five tables.

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
every node (contract section 4 requirement, pass). Table build times: < 0.01 s
(mono), 0.20 s (powerlaw), 0.29 s (blackbody) on one core.

## 6. Candidate tables

Directory `tests/data/hadronic/cpp-candidate/`, produced by `bh_tables_dump`
from commit `5aadf9d` (round 2), `status: cpp_candidate`, schema of contract
1.4. `data_sha256` is the hash of the data block carried in the header
(reproduced by `grep -v '^#' FILE | sha256sum`); the file hash is `sha256sum
FILE`. Measured against the round-1 files: every data token of every row is
byte-identical (the mono baths as required; the powerlaw and the blackbody
too, the R4 change of the segment integral being below the printed 13
digits), so the data hashes differ from round 1 only through the column
separator (one space instead of two) and the file hashes through the header.

| file | nodes | data_sha256 | sha256 of the file |
| --- | --- | --- | --- |
| `mono05.txt` | 466 | `f7e28521b5d846dfb06793369ae708eaa460b0bc4d0e64f5994cff2dfdb6a189` | `e9699811ca88fe4e592bce462738bb55a7f666ef5853c30e9c808b3cb3c7e0d4` |
| `mono03.txt` | 465 | `a482ff4598e0703f3b445e26cbe007883e406d3fa8c520953c2d4bdd87ea7a64` | `3e9276db752dcb45bd18bcf568a696e67d71269ebef0a6f6060e89c0b314879e` |
| `powerlaw.txt` | 429 | `53c2ed5c74bd2bfc9e7fba194f8c97756a7444bb2c222256e69bb8b33eaf5fa1` | `2bc1a60a28c7bdee7dd84be1ab647eddeb3b7a95056fdcd9e9f805f6512197ab` |
| `blackbody.txt` | 406 | `6ac8e2063863b01d5d7c4a8374d5f2e1ef299022f184306ef3bbeaa31bebf10e` | `8c16875ad3c3afbd8b4dd47c666ec8c5d25784b2243189945688b2e0c05e0642` |
| `mono05_mu100.txt` | 403 | `f8a253176397dfce51f4b7850debcb2295b0d6ddcd3a1fa482b6ba30274fbb19` | `be4a79d76ad6028f598830d7bc0f1de93ca1386a1a298a9e6763fade98487390` |

Round-1 data hashes (two-space separator), for the record: mono05
`e16c0408...`, mono03 `881ce6ae...`, powerlaw `07c70efa...`, blackbody
`f10280c1...`, mono05_mu100 `ed3ef74a...` (full values in the round-1 report
version at commit `1ebd2c4`).

New header quantities of the schema, as written: `number_normalization` 1
(mono), 0.10714285714285714 (powerlaw: 1/int_0.1^1.5 eps^-2), 415.95407501276327
(blackbody); `mean_photon_energy` 0.5 and 0.3 (mono),
0.29014823583237964 (powerlaw, ln 15/(10 - 2/3)), 0.2701161768719336 (blackbody);
`gamma_at_I_peak` equals `gamma_max` for all five tables.

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

1. **Grid remainder (resolved: convention adopted in contract 1.4).** The
   spacings are exactly 1/256 and 1/64 and y_max is appended (shorter last
   interval), with the 1e-9 coalescing rule stated in the header. If the
   comparison needs the C++ rates at other Gamma, they are available at any
   Gamma through `BetheHeitlerRates::at_gamma`.
2. **y_min depends on I_peak and the bisection.** Two implementations will
   place y_min identically only to the precision of their peak search and
   root find (here 1e-13 in y). This shifts every node by the same tiny
   amount; the item-8 precheck reports agreement at 1e-12, so it has not
   mattered.
3. **CRPropa anchor provenance (resolved).** The six values were verified by
   the referee against the distributed file and now live in a fixture with
   that file's hash and the CRPropa3-data commit (section 4, item 5).
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
   This is why the threshold and UR items are not repeated for it (section 4,
   item 4).
10. **Library sentinel for the automatic cap.** Inside the library,
    `build_tables(rates, mu_r, gamma_p_max = 0)` keeps 0 as "automatic".
    The tool never produces that value from user input: its parser accepts
    the automatic cap only through omission or the word `auto`, and rejects
    `--gamma_p_max 0` (and anything not > 1) explicitly.

## 8. Not done

- Item 7 (polylogarithm closed forms of Zou et al. 2025) and item 8 (`bhref`
  comparison): not part of this step on the C++ side.
- Parser plumbing (`parameters/hadronic.{h,cpp}`, schema, `input.default.toml`):
  excluded by the brief.
- Builds with clang, MPI, CUDA, or double `real_t`: not attempted here (the
  new code is `real_t`-independent and Kokkos-free; only GCC 13.3 Release on
  Linux was run in the container; the referee reports the native macOS build
  passing the round-1 test).

## 9. Reproduction

```
cmake -S . -B build -G Ninja -D TESTS=ON -D output=ON -D precision=single -D mpi=OFF \
      -D CMAKE_PREFIX_PATH="$HOME/.entity/kokkos;$HOME/.entity/adios2"
cmake --build build
ctest --test-dir build -R bh_physics --output-on-failure
./build/tools/bh_tables_dump --type blackbody --theta 0.1 --eps_min 0 --eps_max 1.9 \
      --mu_r 1836.15 --out blackbody.txt
```

## 10. Referee resolution log (round 2)

Every finding of the referee review (as reproduced in the round-2 brief) with
its resolution. Code changes are in commit `5aadf9d`; the regenerated tables,
the README and this report are in the following commit.

| finding | status | resolution |
| --- | --- | --- |
| R1 (blocking): envelope guard | accepted, `5aadf9d` | `table_upper_edge` requires mu_r finite and positive and the kernel cap min(0.05 mu_r, 1000) > 2, else throws naming mu_r and the cap; an explicit gamma_p_max must be finite and > 1; Gamma_max > Gamma_th as before. Tests (`test_envelope`): mu_r = 1 and 40 rejected (caps 0.05 and 2), NaN and negative mu_r rejected, gamma_p_max = inf, 1 and 2 rejected; mu_r = 100 gives Gamma_max = 5.05 and eps'_hi(last) = 5.0; mu_r = 19999 and 20001 give eps'_max = 999.95 and 1000; eps'_hi at the last node equals eps'_max to 2.2e-16 relative in all cases (tolerance 1e-12). |
| R2 (blocking): literal-byte hashing and reader validation | accepted, `5aadf9d` | `parse_table_file` finds the header/data boundary once, hashes the remaining bytes verbatim, parses the same bytes separately; a missing or mismatched `data_sha256` throws; then required keys, format and revision, canonical `columns`, exactly seven finite tokens per row, `node_count` equal to the row count, at least two rows, Gamma strictly increasing and J, I > 0 are enforced. Negative tests (run on the mono05 and blackbody files): changed data token, inserted blank line (stale and recomputed digest), inserted comment line, missing terminal newline (stale and recomputed digest), empty data block with a valid digest, one-row and two-row tables with negative J and a correct digest, non-increasing Gamma, header claiming 999 nodes, missing `data_sha256`, eight-token row: all rejected. The positive round trip is kept. `Sha256` is untouched. |
| R3 (should fix): strict spectrum rows and options | accepted, `5aadf9d` | `TableBath` rejects any non-comment, non-empty row that is not exactly two finite numbers, naming file and line (tests: non-numeric token, one and three columns, `inf`; comments and blank lines still accepted). The tool's options are parsed by `parse_dump_options` (`bh_options.h`): `strtod` with end pointer, whole string consumed, finite; eps0 and eps_max in (0, 2), eps_min in (0, 2) (blackbody: [0, 2)), theta > 0, mu_r > 0, gamma_p_max > 1 or the word `auto`; unknown or repeated options and missing required options are errors; failures exit with 2 and a message. Tests include the referee's strings `.5 malformed` and `--gamma_p_max not-a-number`, and `--gamma_p_max 0`. |
| R4 (should fix): power-segment integral | accepted, `5aadf9d` | `power_integral` = a^t L expm1(tL)/(tL), limit a^t L for abs(tL) < 1e-18 (threshold on the product, not on t alone); used by the powerlaw normalisation and mean energy and by the table-bath segments. Test against a 256-panel Gauss-Legendre reference at p = -1 + 2e-12, -1 + 1e-11, -0.9, -1.1, -1, -2 on [0.1, 1.5] and at p = -1 + 2e-12, -1 + 1e-11, -0.9, -1.1, -1 on [0.5, 0.5001]: max relative deviation 7.3e-15 (tolerance 1e-13). Regenerated tables: every data token byte-identical to round 1 (change < 1e-12 as required; measured 0 at the printed 13 digits). |
| R5 (should fix): contract 1.4 schema | accepted, `5aadf9d` | Writer and reader implement the section-2 schema exactly (key list and order, one-space columns, strict JSON, `json_string` escaping of quotes, backslashes and control characters for every string including file names, non-finite numeric metadata refused, `endpoint_coalescing: 1e-9` stated, `gamma_at_I_peak`, `number_normalization`, `mean_photon_energy`, `file_sha256` for table baths). Node placement unchanged (the adopted convention). `test_foreign_reader` parses a hand-written file with different number formatting, a `bhref` producer, escaped strings, a different `quadrature` object and an unknown extra key, and checks the decoded metadata and values; it also checks rejection of a missing key, non-canonical columns, an unsupported revision and `NaN` metadata. |
| R6 (should fix): CRPropa fixture | accepted, `5aadf9d` | The six values moved to `tests/data/hadronic/anchors/crpropa-lossrate-CMB-6rows.txt` with the distributed-file SHA-256, the CRPropa3-data commit, the six log10 gamma values, the units and the verification note; the test reads the fixture (path from the compile definition `BH_TEST_DATA_DIR`) and regenerates nothing. C++ only. |
| R7 (note): report wording | accepted, this report | "72 registered checks" replaced by the two-class count (55 contract items, 89 implementation-chosen checks); the phi threshold check at d = 1e-4 labelled implementation-chosen (item 2); the mu_r = 100 repeat's skipped items and the reason stated (item 4 and section 7.9); the deep-UR checks at 2 Gamma eps_c = 900 stated as kernel-limit tests outside the production envelope (item 5); section 3.1 corrected to say the mono branch uses zero lower cumulatives directly. |
| R8 (note): tool link dependencies | accepted (optional), `5aadf9d` | `tools/CMakeLists.txt` clears the target's inherited `LINK_LIBRARIES`; `ldd` on the built tool shows no Kokkos, ADIOS2 or OpenMP library. Top-level link settings untouched. |

Nothing was deferred or rejected.

## 11. Round 2: build and test summary

Same container and configuration as phase 0 (GCC 13.3, Serial + OpenMP, no
MPI, single precision, output on). At `5aadf9d`:

| item | value |
| --- | --- |
| build | 160 targets, no errors, no warnings in the new files |
| ctest | 53/53 passed, 4.6 s |
| `ARCHETYPES::bh_physics` | contract items 55 passed, 0 failed; implementation-chosen checks 89 passed, 0 failed; 2.3 s |
| tables | regenerated from `5aadf9d`; all data tokens byte-identical to round 1; header hashes verified with `sha256sum` |
| tool | `--eps0 ".5 malformed"` and `--gamma_p_max not-a-number` exit 2 with a message; `--mu_r 40` exits 1 with the R1 message; `--gamma_p_max auto --mu_r 100` reproduces the mono05_mu100 table |

The physics numbers of section 4 are unchanged from round 1.
