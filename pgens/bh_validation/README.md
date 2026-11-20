# Validation of the Bethe-Heitler emission policy

Not part of the pull request (`dev/bh-emission`): the problem generator,
inputs, run scripts, analysis and frozen reference data behind the tests V1
to V13.

- `pgen.hpp`: periodic 2D box with emitters, an optional co-located
  neutraliser, diagnostics printed as `BHV` lines (sums over the emitters and
  the leptons, the policy counters, the residual of Gauss's law on the nodes),
  dumps of the lepton momenta and a hash of the fields and particles.
- `make_input.py`: writes the input of one run.
- `scripts/run_v1.sh`, `scripts/run_validation.sh`: the runs.
- `scripts/analyze_*.py`, `scripts/compare_*.py`: the verdicts (standard
  library only).
- `tools/bhv_spectra.cpp`: the pair spectra against the frozen histograms
  with the calibrated two-sample test of the fork (headers copied verbatim
  from `dev/hadronic` 72427ff6 into `tools/fork-72427ff6/`) and the Holm rule.
- `scripts/v4_blocks.py`, `scripts/extract_block0.py`, `scripts/match_block0.py`
  with `tests/archetypes/bhv_replay_pool.cpp`: a V4 sample step by step, and
  the pairs of a run on a GPU against the same generator streams replayed on
  a CPU.
- `rusty/`: the drivers of the CUDA builds and of the run on one A100.
- `data/`: frozen reference files, verbatim, with `SHA256SUMS`:
  `bhref-0.2.0` and `bhref-0.2.0-suppl1` from `tests/data/hadronic/` at
  `dev/hadronic` 72427ff6; `poster/fig3_analytic_data.csv`, the
  published-formula trajectory of the Gamma_p = 300 test.

Build: `cmake -D pgen=bh_validation ...`; the spectra tool:
`clang++ -std=c++20 -O2 -I tools/fork-72427ff6 tools/bhv_spectra.cpp -o bhv_spectra`.
