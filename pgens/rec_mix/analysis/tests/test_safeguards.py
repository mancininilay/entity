"""Regression witnesses for the Codex A3 findings IC3-IC5 (pytest, no simulation data).

IC3: stale files of an abandoned future after a rollback restart, also before the
     new segment has caught up and after a cadence change, for every output type.
IC4: passive tracers never enter a physical energy residual by default.
IC5: eps-tilde is invalid (NaN, explicit status) at low box occupancy, and the
     variance-corrected estimator meets its stated bias bound at n >= nmin.
"""

import json
import os
import struct
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p13ana import cvbudget, partition, readers  # noqa: E402

NAMES = ["N", "K", "G", "U1", "U2", "U3", "B1", "B2", "B3", "Nunder", "Eunder", "Nover",
         "Eover", "Nskip", "Kb_own", "Kb_all", "Ntot", "Ktot", "Kb_own_raw", "Nlowocc"]


def _meta(layout="active", deposit=None):
    deposit = deposit or [True] * 9
    masses = [1, 1, 4, 1, 1, 4, 1, 1, 4]
    return {"format_version": 3, "layout": layout,
            "species": [{"index": k + 1, "mass": masses[k], "deposit": deposit[k]} for k in range(9)],
            "spectra": {"min": 1e-3, "max": 1e5, "num": 4}, "moments": NAMES,
            "cv": {"record_doubles": 61}}


def _write_recmix(d, label, run_id, moms=None, nbins=4):
    moms = np.zeros((9, len(NAMES))) if moms is None else moms
    with open(os.path.join(d, f"recmix.{label:08d}.bin"), "wb") as f:
        f.write(b"P13RMX02")
        f.write(struct.pack("<qdQ", label, float(label), run_id))
        f.write(struct.pack("<4i", 9, nbins, len(NAMES), 0))
        f.write(struct.pack("<2d", 1.0, 1.0))
        f.write(np.zeros((9, nbins)).astype("<f8").tobytes())
        f.write(moms.astype("<f8").tobytes())
        f.write(np.zeros(9).astype("<f8").tobytes())


def _write_midline(d, label, run_id, n1=8, nrows=2):
    with open(os.path.join(d, f"midline.{label:08d}.bin"), "wb") as f:
        f.write(b"P13MID02")
        f.write(struct.pack("<qdQ", label, float(label), run_id))
        f.write(struct.pack("<4i", n1, nrows, 6, 0))
        f.write(np.zeros(nrows * 6 * n1 + 2 * n1, "<f4").tobytes())


def _segments(d, segs):
    with open(os.path.join(d, "segments.jsonl"), "w") as f:
        for rid, start in segs:
            f.write(json.dumps({"run_id": rid, "start_label": start}) + "\n")


def _scenario(d, cadence_b):
    """Run A (id 11) wrote labels 1, 101, 201, 301; run B (id 22) resumed from the
    checkpoint at label 101 and has written `cadence_b` labels so far."""
    os.makedirs(os.path.join(d, "usr"), exist_ok=True)
    os.makedirs(os.path.join(d, "fields"), exist_ok=True)
    usr = os.path.join(d, "usr")
    with open(os.path.join(usr, "recmix.meta.json"), "w") as f:
        json.dump(_meta(), f)
    _segments(usr, [(11, 0), (22, 101)])
    man = open(os.path.join(usr, "fields_manifest.jsonl"), "w")
    for lab in (1, 101, 201, 301):
        _write_recmix(usr, lab, 11)
        _write_midline(usr, lab, 11)
        os.makedirs(os.path.join(d, "fields", f"fields.{lab:08d}.bp"), exist_ok=True)
        man.write(json.dumps({"label": lab, "run_id": 11}) + "\n")
    for lab in cadence_b:
        _write_recmix(usr, lab, 22)
        _write_midline(usr, lab, 22)
        os.makedirs(os.path.join(d, "fields", f"fields.{lab:08d}.bp"), exist_ok=True)
        man.write(json.dumps({"label": lab, "run_id": 22}) + "\n")
    man.close()
    return usr


@pytest.mark.parametrize("cadence_b,expected", [
    ((201,), [1, 101, 201]),            # B has not reached A's 301 yet
    ((151, 201), [1, 101, 151, 201]),   # B changed the cadence
])
def test_ic3_rollback(cadence_b, expected):
    with tempfile.TemporaryDirectory() as d:
        usr = _scenario(d, cadence_b)
        _, outs = readers.read_all_recmix(usr)
        assert [o.step for o in outs] == expected
        assert [readers._step_of(p) for p in readers.list_midlines(usr)] == expected
        labs = [int(p.split("fields.")[-1][:8]) for p in readers.list_entity_fields(d)]
        assert labs == expected


def test_ic3_unknown_writer_rejected():
    with tempfile.TemporaryDirectory() as d:
        usr = _scenario(d, (201,))
        _write_recmix(usr, 251, 99)  # a writer that is not in the manifest
        _, outs = readers.read_all_recmix(usr)
        assert 251 not in [o.step for o in outs]


def test_ic3_cv_records():
    with tempfile.TemporaryDirectory() as d:
        usr = os.path.join(d, "usr")
        os.makedirs(usr)
        with open(os.path.join(usr, "recmix.meta.json"), "w") as f:
            json.dump(_meta(), f)
        _segments(usr, [(11, 0), (22, 101)])
        recs = []
        for lab, seg, rid in ((1, 1, 11), (101, 0, 11), (201, 0, 11), (301, 0, 11),
                              (102, 1, 22), (151, 0, 22), (201, 0, 22)):
            r = np.zeros(61)
            r[:4] = (lab, lab, seg, rid)
            recs.append(r)
        np.array(recs).astype("<f8").tofile(os.path.join(usr, "cvbudget.bin"))
        cv = readers.read_cvbudget(usr)
        assert list(cv["step"]) == [1, 101, 102, 151, 201]


def _cv_two_records(dW_dep, dW_pas, dU):
    """One interval: depositing species 4 gains dW_dep, tracer 3 gains dW_pas,
    the field energy changes by dU, no flux, no crossings."""
    cv = {"step": np.array([1, 2]), "time": np.array([0.0, 1.0]),
          "segment_start": np.array([True, False]),
          "Icv": np.zeros((2, 9)), "inE": np.zeros((2, 9)), "outE": np.zeros((2, 9)),
          "U_EM": np.array([0.0, dU]), "flux_int": np.array([0.0, 0.0]),
          "jwork_int": np.array([0.0, dW_dep])}
    cv["Icv"][1, 3] = dW_dep
    cv["Icv"][1, 2] = dW_pas
    return cv


def test_ic4_tracers_excluded():
    meta = _meta("testprtl", deposit=[True, True, False, True, True, False, True, True, False])
    cv = _cv_two_records(10.0, 5.0, -10.0)
    iv = cvbudget.intervals(cv, meta=meta)
    assert iv["r_total"][0] == 0.0 and iv["r_F"][0] == 0.0 and iv["r_C"][0] == 0.0
    assert iv["W"][0, 2] == 5.0  # tracer work still available for diagnostics
    assert cvbudget.closure(cv, meta=meta)["residual"] == 0.0
    with pytest.raises(ValueError):
        cvbudget.intervals(cv)
    with pytest.raises(ValueError):
        cvbudget.closure(cv)


def _out_with(Nion, Kion, Kb, lowocc):
    moms = np.zeros((9, len(NAMES)))
    for s in (6, 9):
        moms[s - 1, NAMES.index("N")] = Nion / 2
        moms[s - 1, NAMES.index("K")] = Kion / 2
        moms[s - 1, NAMES.index("Kb_own")] = Kb / 2
        moms[s - 1, NAMES.index("Kb_own_raw")] = Kb / 2
        moms[s - 1, NAMES.index("Nlowocc")] = lowocc / 2
    for s in (4, 5, 7, 8):
        moms[s - 1, NAMES.index("N")] = 100
        moms[s - 1, NAMES.index("K")] = 300
    return readers.RecmixOutput(1, 1.0, 1, 1, np.zeros((9, 4)), moms, NAMES, 11, None)


def test_ic5_low_occupancy_invalid():
    meta = _meta()
    # every ion alone in its box: Kb = K, eps-tilde = 0, occupancy fraction 1
    r = partition.active_partition(meta, _out_with(10.0, 50.0, 50.0, 10.0))
    assert r["epst_i_status"] == "invalid_low_occupancy"
    assert np.isnan(r["epst_i_own"]) and r["epst_i_own_unchecked"] == 0.0
    assert r["i_low_occupancy_fraction"] == 1.0
    # well-occupied boxes: valid
    r = partition.active_partition(meta, _out_with(10.0, 50.0, 20.0, 0.1))
    assert r["epst_i_status"] == "valid" and abs(r["epst_i_own"] - 3.0) < 1e-12
    # an invalid value in the averaging window propagates
    assert np.isnan(partition.steady([1.0, np.nan, 1.0], n=3)[0])
    assert partition.steady([1.0, np.nan, 1.0], n=3, nan_policy="omit")[0] == 1.0


def _kb_estimator(b):
    """Reference implementation of the rec_mix box estimator (BulkEnergy_kernel):
    b has shape [nbox, n, 3]; returns mean (gamma_b - 1) raw and corrected."""
    n = b.shape[1]
    s = b.sum(1)
    q = (b ** 2).sum((1, 2))
    braw = (s ** 2).sum(-1) / n ** 2
    if n > 1:
        s2 = (q - n * braw) / (n - 1)
        bcor = np.clip(braw - s2 / n, 0.0, 1 - 1e-12)
    else:
        bcor = braw
    g = lambda x: 1 / np.sqrt(1 - np.minimum(x, 1 - 1e-12)) - 1  # noqa: E731
    return g(braw).mean(), g(bcor).mean()


@pytest.mark.parametrize("beta_b,sig", [(0.3, 0.3), (0.0, 0.3), (0.5, 0.2)])
def test_ic5_estimator_bias_bound(beta_b, sig):
    """Known bulk flow: drifting isotropic Gaussian in four-velocity. The bias of
    the corrected bulk energy, in units of the true eps-tilde, is <= 1.5% at
    n = 20 (the default nmin) and decreases with n; the raw estimator is ~1/n."""
    rng = np.random.default_rng(7)

    def sample(n, nbox):
        u = rng.normal(0.0, sig, (nbox, n, 3))
        u[..., 0] += beta_b / np.sqrt(1 - beta_b ** 2)
        g = np.sqrt(1 + (u ** 2).sum(-1))
        return u / g[..., None]

    big = sample(400000, 1)[0]
    bt = big.mean(0)
    gt = 1 / np.sqrt(1 - (bt ** 2).sum()) - 1
    eps = (1 / np.sqrt(1 - (big ** 2).sum(-1)) - 1).mean()
    epst = eps - gt
    raw20, cor20 = _kb_estimator(sample(20, 40000))
    raw5, cor5 = _kb_estimator(sample(5, 40000))
    assert abs(cor20 - gt) / epst < 0.015
    assert abs(cor5 - gt) / epst < 0.07
    assert (raw20 - gt) / epst > 0.025  # the uncorrected estimator is visibly biased


def test_ic5_window_change_reported():
    """Format 3: eps-tilde at window 1 and the relative change to the configured
    window are reported (not judged); window-1 validity uses Nlowocc_w1."""
    names = NAMES + ["Kb_own_w1", "Nlowocc_w1"]
    moms = np.zeros((9, len(names)))
    for s in (6, 9):
        moms[s - 1, names.index("N")] = 5
        moms[s - 1, names.index("K")] = 25
        moms[s - 1, names.index("Kb_own")] = 10      # eps-tilde(w) = 3
        moms[s - 1, names.index("Kb_own_raw")] = 10
        moms[s - 1, names.index("Kb_own_w1")] = 12.5  # eps-tilde(1) = 2.5
        moms[s - 1, names.index("Nlowocc_w1")] = 1    # 10% at window 1: invalid
    for s in (4, 5, 7, 8):
        moms[s - 1, names.index("N")] = 100
        moms[s - 1, names.index("K")] = 300
    out = readers.RecmixOutput(1, 1.0, 1, 1, np.zeros((9, 4)), moms, names, 11, None)
    r = partition.active_partition(_meta(), out)
    assert r["epst_i_own"] == 3.0
    assert r["epst_i_own_w1_unchecked"] == 2.5 and np.isnan(r["epst_i_own_w1"])
    assert abs(r["epst_i_window_change"] - 0.5 / 3.0) < 1e-12
