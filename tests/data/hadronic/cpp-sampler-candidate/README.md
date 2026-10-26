# C++ step-4 sampler fixtures (candidates)

Files produced by `tools/bh_sampler_dump` (producer `entity-cpp`, status
`cpp_sampler_candidate`) in the contract-1.4 header style: `# key: value`
with strict JSON values, `columns`, `data_sha256` over the literal data block
(`grep -v '^#' FILE | sha256sum` reproduces it), one-space `%.12e` columns.
They are **candidates**, not frozen fixtures: the independent reference for
the comparison of contract (step 4) section 4 is `bhref` 0.2, whose files
enter the fork under `tests/data/hadronic/bhref-0.2.0/` when frozen.

- `qk_table.txt` (section 4 item 1): the 449 eps' nodes (64 per decade in
  log10(eps' - 2) from 1e-4 while eps' <= 1000, then eps' = 1000), q_k from
  the rest-frame machinery, q_k from Chodorowski eq. 3.7, <E'->, <p' cos
  theta'>, sigma~ from the integral of the energy law and Maximon's sigma~.
  `ARCHETYPES::bh_restframe` compares the file with the live computation
  (1e-10) when it is present. Regenerate with
  `build/tools/bh_sampler_dump qk_table --out tests/data/hadronic/cpp-sampler-candidate/qk_table.txt`.
- `moments_<bath>.txt` (section 4 item 2), five baths: at every Gamma_p node
  of the rate table, 1e6 unweighted draws (`Dmean_mc`, `Dmean_se`, `D2mean`,
  `Emean_minus`, `Emean_plus`, `cos_minus_lab_mean`, `eps_mean_mc`, `D_min`,
  `D_max`) and the weighted chain within a budget of 1e6 attempts
  (`n_weighted_accepted`, `n_weighted_attempts`, `Einv_Dmean` = E_p[1/D] (I/J),
  `Einv_Dmean_se`, `efficiency`, `inv_min_D_weighted`); `Dmean_table` = I/J of
  the step-1 table. Seed 20261026 (header `rng`).
- `hist_<bath>_G<Gamma>.txt` (item 3), mono05 and powerlaw at Gamma_p = 1.05
  Gamma_th, 3, 10, 100: 200 log bins in E- and E+ (lab, [1, Gamma (E'_max +
  beta p'_max)]), 200 linear bins in cos theta- and cos theta+ (lab, to u^),
  200 linear bins in D over [D_min, D_max] of the bath; unweighted counts and
  normalised 1/D weights of the weighted chain (1e6 accepted or 2e7 attempts,
  header `n_weighted_accepted`).
- Regenerate everything with `tools/bh_sampler_fixtures.sh build/tools/bh_sampler_dump tests/data/hadronic/cpp-sampler-candidate`
  (about 25 minutes on 4 cores; add `--n-weighted-attempts 20000000` for the
  histogram statistics used here).
- `ka2008_mono05_G100.txt`, `ka2008_mono05_G1000.txt` (item 4): the lab e-
  spectrum of the UR anchor (Kelner and Aharonian 2008 eqs. 58 to 64, as the
  exact boost of the rest-frame laws by quadrature, `bh_lab_spectrum.h`) as a
  CDF at the upper edges of the 200 log-spaced E- bins of the `hist` files at
  the same Gamma_p (`cdf_at_hi`, `probability_in_bin`); the header carries the
  quadrature parameters, the change under refinement and the mean lab energy.
