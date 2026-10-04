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


def test_scales_c2():
    # tech spec III.1: sigma0 = 88, 41.98, 520 for C2-a, b, c
    assert abs(regimes.scales("active", 40, 0.1, 25, 5)["sigma0"] - 88.0) < 1e-9
    assert abs(regimes.scales("active", 40, 1e-3, 100, 5)["sigma0"] - 41.98) < 1e-9
    assert abs(regimes.scales("active", 40, 1.0, 25, 5)["sigma0"] - 520.0) < 1e-9
