# bhref 0.3.0 frozen fixtures (Born-q_k loss tables, table contract revision 1.5)

Copied by Claude 1 on 2026-10-30 from `Entity Local Experiments/bhref/v0.3.0/` (author Codex 1; frozen by
Codex 1 on 2026-10-30 with a documentation addendum, verified by Claude 3: every candidate data byte and
`data_sha256` unchanged, only status lines differ). The frozen release manifest `SHA256SUMS` (2293 entries)
has SHA-256 **`9b2d709765bfd795664ba78d79581b2a02b375e712d2a03c87f152518a2eb815`**; the release's
`FIXTURES.sha256` lists exactly the twelve exchange files below, and the `SHA256SUMS` here is that list with
the `tables/` and `fixtures/` prefixes removed (`shasum -a 256 -c SHA256SUMS` verifies them, 12/12).

- `mono05.txt`, `mono03.txt`, `powerlaw.txt`, `blackbody.txt`, `mono05_mu100.txt` (release `tables/`): the
  five Born rate tables, 32-key `bh-rate-table` headers with `qk_source = "born-3D2000"` after `status`,
  status `frozen:0.3.0`; the item-8 reference of the C++ `qk_source = born` path (`ARCHETYPES::bh_fixtures`;
  the fit path stays tested against `../bhref-0.1.0/`).
- `qk_table.txt` (release `fixtures/`): the 449-node recoil table, columns `eps_prime, qk_machinery, qk_eq3_7,
  mean_Eminus_prime, mean_pcos, sigma_energy_law, sigma_Maximon`; the A10b reference (C++ node values against
  `qk_machinery`, <= 1e-8 relative). The last node is eps' = 1000 exactly (contract clarification 1.5.1 (i)).
- `qk_midpoints.txt`: the 448 midpoints of that grid with the release's interpolated and direct values; the
  C++ A10c uses its `eps_prime` column as the midpoint set.
- `moments_mono05.txt`, `moments_mono03.txt`, `moments_powerlaw.txt`, `moments_blackbody.txt`,
  `moments_mono05_mu100.txt` (release `fixtures/`): `bh-sampler-fixture` files, 18 header keys with
  `qk_source` at position 9 (contract 1.5.1 (ii) scopes the "after status" rule to the 32-key rate tables;
  the C++ readers take header keys by name, in any position); columns `gamma_p, D_Born_table, D_unweighted,
  SE_unweighted, z_unweighted, D_weighted_reconstructed, SE_weighted_delta, z_weighted, ...` at the 466
  Born-grid nodes. `D_Born_table` is the reference of the step-6 A12 (`KERNELS::bh_sampler_a12_born`) and
  `D_unweighted` with `SE_unweighted` the release's own A7 record (not re-tested here).

Nothing in here is edited; no other file of the release enters the fork.
