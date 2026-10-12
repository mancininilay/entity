# p13ana

Analysis package for P1.3 (energy partition in radiative pair-ion reconnection)
and the `rec_mix` generator. It is versioned with the generator in
`entity-dev/pgens/rec_mix/analysis` (Codex A3); the earlier unversioned copy
in `Entity Local Experiments/p13ana` is superseded. Specification:
`Resources/p1.3/technical-spec.md`, II.5 and III.5 to III.13.

Modules:

* `readers`: `rec_mix` user outputs (`recmix.*.bin` formats 1 and 2,
  `cvbudget.bin`, `midline.*.bin`, `recmix.meta.json`, `segments.jsonl`,
  `fields_manifest.jsonl`), Entity field outputs (BP, via the `adios2` Python
  package) and the p1.1 Tristan user spectra (`output/usr/reconn_spec.*`, via
  `h5py`). Every reader applies the restart provenance rule (Codex IC3; see the
  generator README): outputs of an abandoned future or of an earlier run in the
  same directory are never returned.
* `partition`: active and passive schemas, kept separate (Codex R4). Each
  quantity has a `hist` estimator (bin centres, exactly as `extractfi.py`) and a
  `direct` estimator (double moment sums). eps-tilde carries an occupancy
  status (Codex IC5, RV1): NaN with `low_occupancy` when more than 2% of the
  population's masked particles, or more than 2% of its random energy (bound
  from Klowocc, Kblowocc), sit in boxes with fewer than `bulk_nmin` particles.
  The status is an occupancy screen, not an accuracy certificate; the thinning
  change (finite-sample dependence on the actual distribution) and the window
  change are reported with every value; the unchecked values stay under
  `*_unchecked`. `steady()` is the p1.1 protocol (last 50
  outputs, mean and population standard deviation; NaN propagates unless
  `nan_policy="omit"`); `compare()` is the C2 tolerance formula.
* `cvbudget`: control-volume work per species and the three residuals r_F, r_C,
  r_total over the depositing species (Codex IC4: a species set or the meta is
  required; tracers never enter a physical residual), f_i^CV.
* `ezones`: zone finder (|E_x3| > |B_x2|), trimming, snapshot means, the
  continuity subsampling (stride 8) and the node-line sensitivity.
* `inflow`: port of `betarec4.py` (slabs, inward-positive convention, latest
  block of 3 consecutive good outputs) for any velocity definition.
* `regimes`: units mapping (III.1) and regime numbers.
* `neutrality`: the C1.3b crossing-noise model (independent Maxwell-Juttner
  sampler) and the block test with a Bonferroni bound calibrated on the
  discrete crossing distribution (Skellam; the Gaussian bound is reported as
  an approximation).

Tests:

* `python3 -m pytest tests` (no data needed): `test_basic` (mask witnesses,
  zones, inflow geometry, formulas) and `test_safeguards` (regression witnesses
  for IC3 to IC5, the Gaussian calibration of the estimator and the RV1
  counterexamples).
* Check scripts on a `rec_mix` output directory: `check_run.py` (C1.1, C1.11,
  C1.14, C1.15), `check_sheet.py` (C1.2), `check_neutrality.py` (C1.3b),
  `check_bulk.py` (C1.16), `check_particles.py` (C1.16b, C1.14b and the window,
  thinning and occupancy scans, from a full particle output).
* The assertions and tolerances live in `../tests/harness.py`.

Status (2026-10-12): written by Claude, reviewed by Codex (A3 2026-10-04;
revision referee 2026-10-12, RV1 and RV2 resolved in tech spec VI.8); the
C1.10 validation on the original Tristan outputs is pending (needs the Rusty
data and the A1 extraction).
