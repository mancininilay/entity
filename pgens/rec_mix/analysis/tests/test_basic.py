"""Unit tests for p13ana that do not need simulation data (pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p13ana import ezones, inflow, partition, regimes  # noqa: E402


def test_zones_cut_and_drop():
    c = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1], bool)
    assert list(ezones.zones_1d(c, 1.0, "cut")) == [2, 3, 1]
    assert list(ezones.zones_1d(c, 1.0, "drop")) == [3]
    assert list(ezones.zones_1d(c, 8.0, "cut")) == [16, 24, 8]


def test_line_trim():
    e3 = np.ones(20)
    b2 = np.zeros(20)
    b2[[0, 1, 10]] = 2.0  # zone broken at 10
    L = ezones.line_zones(e3, b2, trim_cells=2, pixel=1.0)
    assert list(L) == [8, 7]


def test_mask_witness_codex_R1():
    # (e, i, p) = (200, 0, 200) on one side and (1, 1, 0) on the other:
    # total-number mask M = 2*2/402 = 0.00995 (not masked at 0.01);
    # Tristan's 3Ne + 2Ni + Np would give 2*5/805 = 0.0124 (masked).
    M = partition.mixing_from_counts(200 + 0 + 200, 1 + 1 + 0)
    assert abs(M - 4 / 402) < 1e-15 and not (M > 0.01)
    M_legacy = 2 * 5 / (800 + 5)
    assert M_legacy > 0.01


def test_mask_witness_second():
    # one more independent configuration: (e, i, p) = (30, 3, 27) vs (10, 1, 9)
    M = partition.mixing_from_counts(60, 20)
    assert abs(M - 0.5) < 1e-15


def test_compare_formula():
    r = partition.compare(1.08, 1.0, 0.01, 0.01, 0.10)
    assert r["pass"] and abs(r["tol"] - 0.10) < 1e-12
    r = partition.compare(1.2, 1.0, 0.01, 0.01, 0.10)
    assert not r["pass"]


def test_tristan_partition_formula():
    eb = np.array([1.0, 10.0])
    spec = {"ebins": eb, "counts": {4: np.array([1.0, 1.0]), 5: np.zeros(2), 7: np.zeros(2),
                                    8: np.zeros(2), 6: np.array([0.0, 2.0]), 9: np.zeros(2)}}
    r = partition.tristan_active_partition(spec, mu=25.0)
    El, Ei = 11.0, 25.0 * 20.0
    assert abs(r["f_i"] - Ei / (Ei + El)) < 1e-12
    assert abs(r["eps_e"] - 5.5) < 1e-12 and abs(r["eps_i"] - 250.0) < 1e-12


def test_inflow_geometry():
    n2, n1 = 100, 40
    x2 = np.arange(n2) + 0.5
    x1 = np.arange(n1) + 0.5
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    v = np.where(X2 < 50, 0.2, -0.2)
    r = inflow.beta_in(v, X2, X1)
    assert abs(r["beta"] - 0.2) < 1e-12 and r["good"]
    pos = inflow.latest_good_block([1, 2, 3, 5, 6, 7], [True] * 6, 3)
    assert list(pos) == [3, 4, 5]


# --------------------------------------------------------------------------
# P13-D39 hot normalization. Anchors are fixed numbers from outside the
# converters under test: saved p1.1 cfg inputs (plasma:sigma, c_omp) from
# Codex's 48-cfg audit (Entity Local Experiments/p13-reference-20261012/
# normalization-audit.json), Codex's independently derived nominal values and
# larmor0 (normalization review 2026-10-12), and the pre-D39 cold-limit values.
# --------------------------------------------------------------------------
# (layout, mu, q, sigma_hot, d_up, saved sigma, saved c_omp)
CFG_ANCHORS = [
    ("active", 25, 0.1, 40, 5, 89.0, 4.84),        # C2-a  runs_mu25/sigma40_k19
    ("active", 100, 0.001, 40, 5, 42.98, 4.96),    # C2-b  runs_mu100/sigma40_k1999
    ("active", 25, 1.0, 40, 5, 521.0, 3.58),       # C2-c  runs_mu25/sigma40_k1
    ("testprtl", 1, 0.0, 160, 5, 164.0, 4.96),     # C2-d  multi-ion/sigma160_mu_25_100_400
    ("active", 100, 1.0, 40, 5, 2021.0, 3.52),     # runs_mu100/sigma40_k1 (truncated c_omp)
    ("active", 100, 0.1, 40, 5, 239.0, 4.83),
    ("active", 100, 0.01, 40, 5, 60.8, 4.95),
    ("active", 100, 1e-4, 40, 5, 41.2, 4.96),      # saved 41.2 for 41.198 (4.9e-5)
    ("active", 100, 1e-4, 80, 5, 82.4, 4.96),      # saved 82.4 for 82.396
    ("active", 25, 0.5, 40, 5, 281.0, 4.32),
    ("active", 100, 0.5, 80, 5, 2062.0, 4.3),
    ("active", 25, 0.1, 40, 2.5, 89.0, 2.42),      # legacy 3D, about 2.5 cells
    ("active", 25, 0.01, 40, 2.5, 45.8, 2.475),    # legacy 3D
    ("active", 25, 1.0, 40, 2.5, 521.0, 1.79),     # legacy 3D
]
# Codex's derived nominal c_omp and literal / ideal larmor0 (review table)
DERIVED = {"C2-a": (4.842341981, 0.513038974, 0.513287223), "C2-b": (4.961688196, 0.756568876, 0.756826383),
           "C2-c": (3.578810106, 0.156842648, 0.156790517), "C2-d": (4.962916670, 0.387310929, 0.387538683)}


def _make_input():
    import importlib.util
    p = os.path.join(os.path.dirname(__file__), "..", "..", "make_input.py")
    spec = importlib.util.spec_from_file_location("make_input", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_d39_saved_inputs_reproduced():
    """Both converters (p13ana.regimes and make_input) give the saved inputs of
    the audited p1.1 cfgs to their saved precision (sigma 1e-4 relative, c_omp
    0.01 cells), and make_input accepts them as literal inputs."""
    mi = _make_input()
    for layout, mu, q, sh, dup, s_cfg, c_cfg in CFG_ANCHORS:
        r = regimes.scales(layout, sigma_hot=sh, q=q, mu=mu, dup_cells=dup, T=0.01)
        assert abs(r["sigma_input"] / s_cfg - 1) <= 1e-4, (layout, mu, q, r["sigma_input"], s_cfg)
        assert abs(r["c_omp"] - c_cfg) <= 0.01, (layout, mu, q, r["c_omp"], c_cfg)
        d = mi.derived(layout, sh, q, mu, dup, 0.01, s_cfg, c_cfg)
        assert d["mode"] == "literal" and d["sigma0"] == s_cfg and d["skindepth0"] == c_cfg
        assert abs(d["sigma_input_nominal"] / r["sigma_input"] - 1) < 1e-14


def test_d39_codex_anchors():
    """The nominal c_omp and the literal/ideal larmor0 match Codex's independent
    numbers to their quoted precision (1e-9)."""
    mi = _make_input()
    for ref, (c_nom, l_lit, l_ideal) in DERIVED.items():
        R = mi.REFERENCES[ref]
        r = regimes.scales(R["layout"], sigma_hot=R["sigma_hot"], q=R["q"], mu=R["mu"], dup_cells=5, T=0.01)
        assert abs(r["c_omp"] - c_nom) < 1e-9, (ref, r["c_omp"])
        assert abs(r["larmor0"] - l_ideal) < 1e-9, (ref, r["larmor0"])
        lit = regimes.scales(R["layout"], sigma_hot=R["sigma_hot"], q=R["q"], mu=R["mu"], dup_cells=5,
                             T=0.01, sigma_input=R["sigma_input"], c_omp=R["c_omp"])
        assert abs(lit["larmor0"] - l_lit) < 1e-9, (ref, lit["larmor0"])
        d = mi.derived(R["layout"], R["sigma_hot"], R["q"], R["mu"], 5.0, 0.01, R["sigma_input"], R["c_omp"])
        assert abs(d["larmor0"] - l_lit) < 1e-9


def test_d39_cold_limit():
    """T -> 0 recovers the pre-D39 cold mapping (an identity of the cold limit
    only): sigma0 = 88, 41.98, 520, 160 and skindepth0 = d_up sqrt((2-q+q/mu)/2)."""
    for (mu, q, s_old) in ((25, 0.1, 88.0), (100, 1e-3, 41.98), (25, 1.0, 520.0)):
        r = regimes.scales("active", sigma_hot=40, q=q, mu=mu, dup_cells=5, T=0.0)
        assert abs(r["sigma0"] - s_old) < 1e-9
        assert abs(r["skindepth0"] - 5 * np.sqrt((2 - q + q / mu) / 2)) < 1e-12
        assert abs(r["sigma_cold_total"] - 40) < 1e-12 and r["c_omp_ratio_common_T"] == 1.0
    r = regimes.scales("testprtl", sigma_hot=160, q=0, mu=1, dup_cells=5, T=0.0)
    assert r["sigma0"] == 160 and r["skindepth0"] == 5
    mi = _make_input()
    d = mi.derived("active", 40, 0.1, 25, 5, 0.0)
    assert abs(d["sigma0"] - 88.0) < 1e-12


def test_d39_structure_and_qualification():
    """The thermal enthalpy term is 5T/2 for any q at fixed n0 (every particle
    carries 5 kT/2); D (the sheet's mass factor) is unchanged; the common-T ion
    denominator mu + 3T/2 (documented, not used) changes c_omp by at most
    0.027672% (mu 25, q 1), 0.001512% for C2-a, 3.714e-6% for C2-b (Codex)."""
    for q in (0.0, 0.1, 0.5, 1.0):
        r = regimes.scales("active", sigma_hot=1.0, q=q, mu=25, dup_cells=5, T=0.01)
        assert abs(r["H"] - r["D"] - 0.025) < 1e-15
        assert abs(r["D"] - (2 - q + 25 * q) / 2) < 1e-15
    f = lambda mu, q: regimes.scales("active", sigma_hot=40, q=q, mu=mu, dup_cells=5)["c_omp_ratio_common_T"] - 1  # noqa: E731
    assert abs(f(25, 1.0) - 2.7672e-4) < 1e-8
    assert abs(f(25, 0.1) - 1.512e-5) < 1e-8
    assert abs(f(100, 1e-3) - 3.714e-8) < 1e-10
    worst = max(f(mu, q) for mu in (25, 100, 400) for q in np.linspace(0, 1, 41))
    assert worst <= 2.7672e-4 + 1e-9


def test_d39_cold_refused():
    """make_input refuses the cold reading in its literal check and the old flag."""
    mi = _make_input()
    import pytest
    with pytest.raises(SystemExit):
        mi.derived("active", 40, 0.1, 25, 5.0, 0.01, 88.0, 4.84)        # cold sigma0
    with pytest.raises(SystemExit):
        mi.derived("active", 40, 0.1, 25, 5.0, 0.01, 89.0, 4.878524)    # cold skindepth0
    with pytest.raises(TypeError):
        regimes.scales("active", 40, 0.1, 25, 5)                         # positional (pre-D39) call


# Rusty CUDA regression: orientation must follow the BP attribute even when
# a square grid makes both candidate shapes identical.
def test_entity_field_layout_metadata():
    import types
    from unittest.mock import patch
    from p13ana import readers

    for shape in ((3, 5), (4, 4)):
        physical = 100 * np.arange(shape[0])[:, None] + np.arange(shape[1])[None, :]
        for layout in (0, 1):
            class Reader:
                def __init__(self, path): pass
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def available_variables(self): return {"fB1": {}}
                def read_attribute(self, name):
                    return {"Dimension": 2, "LayoutRight": layout}[name]
                def read(self, name):
                    return {"fB1": physical if layout else physical.T,
                            "X1": np.arange(shape[0]), "X2": np.arange(shape[1]),
                            "Step": 1, "Time": 0.45}[name]
            with patch.dict(sys.modules, {"adios2": types.SimpleNamespace(FileReader=Reader)}):
                fields = readers.read_entity_fields("mock.bp")
            np.testing.assert_array_equal(fields["B1"], physical)
            np.testing.assert_array_equal(fields["X1"], np.arange(shape[0]))
            np.testing.assert_array_equal(fields["X2"], np.arange(shape[1]))
