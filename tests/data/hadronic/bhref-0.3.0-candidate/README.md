# bhref 0.3.0 candidate tables and recoil fixtures (UNFROZEN)

Copied by Claude 1 on 2026-10-30 from `Entity Local Experiments/bhref/v0.3.0/` (author Codex 1, table
contract revision 1.5, 2026-10-29; **candidate under Claude 3's review, not frozen, not tagged**). The
candidate manifest `SHA256SUMS` of that delivery (2271 entries) has SHA-256
`b393f99a7781fb68ae8a365ad1316b04c9aa40f4fdd0ca269ef03251e779517e`; the `SHA256SUMS` here lists the
seven copied files with the hashes taken from that manifest (`shasum -a 256 -c SHA256SUMS` verifies them).

- `mono05.txt`, `mono03.txt`, `powerlaw.txt`, `blackbody.txt`, `mono05_mu100.txt`: the five Born rate
  tables (`tables/` of the candidate; 32-key exchange header with `qk_source = "born-3D2000"`,
  status `candidate:awaiting-Claude-3-review`), the item-8 reference of the C++ `qk_source = born`
  path (ARCHETYPES::bh_fixtures with the contract tolerances once stage 2 of step 6 adds the Born
  comparison; the fit path stays tested against `../bhref-0.1.0/`).
- `qk_table.txt`: the 449-node recoil table (`fixtures/`), columns `eps_prime, qk_machinery, qk_eq3_7,
  mean_Eminus_prime, mean_pcos, sigma_energy_law, sigma_Maximon`; the A10b reference (C++ node values
  against `qk_machinery`, <= 1e-8 relative).
- `qk_midpoints.txt`: the 448 midpoints of that grid with the candidate's interpolated and direct
  values (`q_interpolated, q_direct, relative_error`); the C++ A10c uses its `eps_prime` column as the
  midpoint set and compares its own interpolant with its own direct quadrature there (and reports the
  difference to the candidate's `q_direct` as a diagnostic).

When the release is frozen the mastermind replaces this directory by `bhref-0.3.0/` with the frozen
manifest hashes and the comparisons are re-run; nothing in here is edited meanwhile. No other file of
the candidate enters the fork.
