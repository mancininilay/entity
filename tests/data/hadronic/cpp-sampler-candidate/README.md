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
