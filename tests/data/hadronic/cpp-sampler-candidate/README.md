# C++ step-4 sampler fixtures (candidates)

Files produced by `tools/bh_sampler_dump` (producer `entity-cpp`, status
`cpp_sampler_candidate`) in the contract-1.4 header style: `# key: value`
with strict JSON values, `columns`, `data_sha256` over the literal data block
(`grep -v '^#' FILE | sha256sum` reproduces it), one-space `%.12e` columns.
They are **candidates**, not frozen fixtures. The independent reference for
the comparison of contract (step 4) section 4 is `bhref` 0.2.0 candidate
(`../bhref-0.2.0-candidate/`); `ARCHETYPES::bh_sampler_fixtures` compares the
two file against file (contract A10).

Round 2 (2026-10-27). The `qk_table`, `moments_*` and `hist_*` files were
regenerated with the tool as committed in `3d17ee1` (stage 2). Their `version`
header reads `31e82eb...-mod`: that is the build's configure-time hash, one
commit older than the tool's code.

- `qk_table.txt` (section 4 item 1): the 449 eps' nodes (64 per decade in
  log10(eps' - 2) from 1e-4 while eps' <= 1000, then eps' = 1000), q_k from
  the rest-frame machinery, q_k from Chodorowski eq. 3.7, <E'->, <p' cos
  theta'>, sigma~ from the integral of the energy law and Maximon's sigma~.
  Round 2: the laws are evaluated in double with the stable threshold
  representation, and q_k is computed without cancellation.
  `ARCHETYPES::bh_restframe` compares the file with the live computation
  (1e-10). Regenerate with
  `build/tools/bh_sampler_dump qk_table --out tests/data/hadronic/cpp-sampler-candidate/qk_table.txt`.
- `moments_<bath>.txt` (section 4 item 2), five baths. At every Gamma_p node
  of the rate table:
  - 1e6 unweighted draws: `Dmean_mc`, `D2mean`, `Emean_minus`, `Emean_plus`,
    `cos_minus_lab_mean` and `eps_mean_mc`, each with its `_se` (Welford's
    update), plus `D_min` and `D_max`.
  - The weighted chain, run to a fixed 1e5 accepted packets per node
    (`n_weighted_attempts` reports the trials): `Einv` = E_p[1/D], its
    products with I/J (`Einv_Dmean`) and with `Dmean_mc` (`Einv_Dmean_mc`,
    delta-method s.e.), `efficiency` with its s.e., and `inv_min_D_weighted`.
  - Deterministic columns: `gamma_p`, `beta_p`, `J`, `I`, `Dmean_table`
    (= I/J of the step-1 table), and `Dmean_sampled`, the <D> of the density
    the unweighted chain samples (quadrature).
  - In a mono bath, `eps_mean_mc` is eps0 exactly with `eps_mean_mc_se` = 0
    (header `zero_variance_columns`).
  - Seed 20261026 (header `rng`).
- `hist_<bath>_G<Gamma>.txt` (item 3), mono05 and powerlaw at Gamma_p = 1.05
  Gamma_th, 3, 10, 100.
  - Layout: that of the bh-sampler-fixture histograms (quantity_id 0..4 =
    E-, E+, cos theta-, cos theta+, D; bin; lo; hi; unweighted_count;
    weighted_packet_count; sum_1_over_D; sum_1_over_D_squared;
    normalized_packet_weight).
  - Bin edges: taken from the reference's file of the same bath and Gamma_p
    (`--bins-from`).
  - Statistics: 1e6 unweighted draws and a fixed 1e6 accepted weighted
    packets. The header gives `n_weighted_attempts` (up to 4.6e8 at mono05
    Gamma_p = 100) and `weighted_ESS`.
  - Generation: 64 chunks with seeds of their own, so the result does not
    depend on the thread count.
- `ka2008_mono05_G100.txt`, `ka2008_mono05_G1000.txt` (item 4, round 1;
  renamed in round-2 stage 3, finding 2): the lab e- spectrum of the exact
  boost of the rest-frame laws by quadrature, as a CDF at the upper edges of
  200 log-spaced E- bins.
- Regenerate everything with
  `tools/bh_sampler_fixtures.sh build/tools/bh_sampler_dump tests/data/hadronic/cpp-sampler-candidate`
  (about 40 minutes on 4 cores).
