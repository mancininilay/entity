"""Regression witnesses for the Codex A3 findings IC3-IC5 (pytest, no simulation data).

IC3: stale files of an abandoned future after a rollback restart, also before the
     new segment has caught up and after a cadence change, for every output type.
IC4: passive tracers never enter a physical energy residual by default.
IC5: eps-tilde is NaN with an explicit status when it fails the occupancy screen.
RV1 (Codex, 2026-10-12): the screen is an occupancy check, not an accuracy
     bound; the energy share of under-occupied boxes is screened (RV1a), and
     the finite-sample bias of the estimator, which no occupancy rule bounds,
     is exposed by the position-subsampling diagnostic (RV1b, Codex's exact
     two-stream counterexample kept as a witness of the limitation).
A0 (Codex, 2026-10-12): A0-1 a refused harness rerun preserves every previous
     record (collision, skipped case, disjoint addition with provenance,
     --overwrite moves instead of deleting); A0-2 rounded output positions are
     ambiguous at integers and at dx1 = 1/2 (both sides witnessed), the exact
     source is unambiguous; A0-3 witnesses of the two qualified claims (the
     box/central-cell Jensen mismatch and velocity-correlated position
     subsampling).
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

sys.path.insert(0, os.path.dirname(__file__))
import check_particles  # noqa: E402

import subprocess  # noqa: E402

HARNESS = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "harness.py")

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
    assert r["epst_i_status"] == "low_occupancy"
    assert np.isnan(r["epst_i_own"]) and r["epst_i_own_unchecked"] == 0.0
    assert r["i_low_occupancy_fraction"] == 1.0
    # well-occupied boxes: valid
    r = partition.active_partition(meta, _out_with(10.0, 50.0, 20.0, 0.1))
    # format 3 without Klowocc: screened on the number fraction only
    assert r["epst_i_status"] == "occupancy_ok_number_only" and abs(r["epst_i_own"] - 3.0) < 1e-12
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
def test_ic5_estimator_gaussian_calibration(beta_b, sig):
    """Calibration on drifting isotropic Gaussians in four-velocity (beta_b <= 0.5):
    the bias of the corrected bulk energy is <= 1.5% of the true eps-tilde at
    n = 20 and the raw estimator's is ~1/n. A calibration for these flows only,
    not a bound for other distributions (see test_rv1b_two_stream_limitation)."""
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


NAMES3 = NAMES + ["Kb_own_w1", "Nlowocc_w1", "Klowocc", "Kblowocc", "Klowocc_w1",
                  "Kblowocc_w1", "Kb_own_half"]


def _out3(ion):
    """Format 3 output; ion = dict of per-species moment values for species 6, 9."""
    moms = np.zeros((9, len(NAMES3)))
    for s in (6, 9):
        for k, v in ion.items():
            moms[s - 1, NAMES3.index(k)] = v / 2
    for s in (4, 5, 7, 8):
        moms[s - 1, NAMES3.index("N")] = 100
        moms[s - 1, NAMES3.index("K")] = 300
    return readers.RecmixOutput(1, 1.0, 1, 1, np.zeros((9, 4)), moms, NAMES3, 11, None)


def test_rv1a_energy_share_screened():
    """Codex RV1a: 990 well-sampled ions with one unit of random energy each and
    10 isolated ions with 1000 units each. The number fraction (1%) passes; the
    energy share of the singleton boxes, max(Klowocc, Kblowocc)/(K - Kb_own) ~ 10,
    does not."""
    ion = {"N": 1000.0, "K": 990.0 + 10000.0, "Kb_own": 10000.0, "Kb_own_raw": 10000.0,
           "Nlowocc": 10.0, "Klowocc": 10000.0, "Kblowocc": 10000.0}
    r = partition.active_partition(_meta(), _out3(ion))
    assert r["epst_i_status"] == "low_occupancy" and np.isnan(r["epst_i_own"])
    assert r["i_low_occupancy_fraction"] == 0.01
    assert abs(r["i_low_occupancy_energy_bound"] - 10000.0 / 990.0) < 1e-12
    # without the energy columns (format 3 before v1.5) the screen cannot see it
    r = partition.active_partition(_meta(), _out_with(1000.0, 10990.0, 10000.0, 10.0))
    assert r["epst_i_status"] == "occupancy_ok_number_only"


def _expected_epst_two_stream(n, p=0.99, b=0.99):
    """Exact expectation (binomial enumeration) of the rec_mix corrected estimator
    for boxes of n independent particles with beta_x = +b (prob p) or -b."""
    import math
    gp = 1 / math.sqrt(1 - b * b)
    kb = 0.0
    for k in range(n + 1):
        prob = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
        braw = ((2 * k - n) * b / n) ** 2
        s2 = (n * b * b - n * braw) / (n - 1)
        bc = max(braw - s2 / n, 0.0)
        kb += prob * (1 / math.sqrt(1 - min(bc, 1 - 1e-12)) - 1)
    mb = (2 * p - 1) * b
    return gp - 1 - kb, gp - 1 / math.sqrt(1 - mb * mb)


def test_rv1b_two_stream_limitation():
    """Codex RV1b, kept as a witness of a documented limitation: with 20 particles
    per box (every box passes the occupancy screen) the corrected estimator
    underestimates eps-tilde by 69%. The thinning diagnostic exposes it: halving
    the sample changes the estimate by ~40%, far above a few percent."""
    e20, true = _expected_epst_two_stream(20)
    e10, _ = _expected_epst_two_stream(10)
    assert abs(e20 / true - 1 - (-0.6932490337670086)) < 1e-9
    assert abs(e10 / e20 - 1) > 0.3
    # and the partition reports that change from Kb_own_half
    ion = {"N": 1000.0, "K": 1000.0 * (1 / np.sqrt(1 - 0.99 ** 2) - 1),
           "Kb_own": 1000.0 * (1 / np.sqrt(1 - 0.99 ** 2) - 1 - e20),
           "Kb_own_raw": 0.0, "Kb_own_half": 1000.0 * (1 / np.sqrt(1 - 0.99 ** 2) - 1 - e10)}
    r = partition.active_partition(_meta(), _out3(ion))
    assert r["epst_i_status"] == "occupancy_ok"
    assert abs(r["epst_i_thinning_change"] - (e10 / e20 - 1)) < 1e-9


# --------------------------------------------------------------------------
# A0-1: harness evidence preservation
# --------------------------------------------------------------------------
def _harness(out, *args):
    return subprocess.run([sys.executable, "-B", HARNESS, "--recmix", "/bin/false", "--out", out, *args],
                          capture_output=True, text=True)


def _campaign(d, cases):
    os.makedirs(d, exist_ok=True)
    for n in cases:
        os.makedirs(os.path.join(d, n), exist_ok=True)
        with open(os.path.join(d, n, "sentinel"), "w") as f:
            f.write("preserve " + n)
    res = {"cases": {n: {"pass": True, "checks": [{"old_evidence": n}]} for n in cases},
           "invocations": [{"index": 0, "cases": list(cases)}]}
    with open(os.path.join(d, "results.json"), "w") as f:
        json.dump(res, f)
    return open(os.path.join(d, "results.json"), "rb").read()


def test_a01_refusal_preserves_records():
    """Codex's witness: rerunning an existing case without --overwrite is
    refused before anything is written; the results.json bytes and the case
    directory are unchanged; the binary is never run."""
    with tempfile.TemporaryDirectory() as d:
        before = _campaign(d, ["capacity"])
        p = _harness(d, "--cases", "capacity")
        assert p.returncode == 2 and "refused" in p.stdout
        assert open(os.path.join(d, "results.json"), "rb").read() == before
        assert open(os.path.join(d, "capacity", "sentinel")).read() == "preserve capacity"


def test_a01_collision_with_record_only_and_skipped_path():
    """A record without a directory collides too, and a case that would be
    skipped for a missing binary (mpi, no --recmix-mpi) does not overwrite its
    record; a mixed selection with one collision writes nothing at all."""
    with tempfile.TemporaryDirectory() as d:
        before = _campaign(d, ["mpi", "capacity"])
        import shutil
        shutil.rmtree(os.path.join(d, "capacity"))  # temp test data only
        for sel in ("mpi", "capacity", "c04_deposit,mpi"):
            p = _harness(d, "--cases", sel)
            assert p.returncode == 2, (sel, p.stdout)
            assert open(os.path.join(d, "results.json"), "rb").read() == before
        assert not os.path.exists(os.path.join(d, "c04_deposit"))


def test_a01_disjoint_addition_with_provenance():
    """New cases may be added to an existing campaign: previous records are kept
    byte for byte inside the new file, and the invocation lists what it added."""
    with tempfile.TemporaryDirectory() as d:
        _campaign(d, ["capacity"])
        old = json.load(open(os.path.join(d, "results.json")))
        p = _harness(d, "--cases", "c04_deposit,mpi")
        assert p.returncode == 1  # /bin/false makes c04_deposit fail; mpi is skipped
        new = json.load(open(os.path.join(d, "results.json")))
        assert new["cases"]["capacity"] == old["cases"]["capacity"]
        inv = new["invocations"][-1]
        assert inv["index"] == 1 and inv["cases"] == ["c04_deposit", "mpi"]
        assert inv["preexisting_cases"] == ["capacity"] and inv["superseded"] == []
        assert new["cases"]["c04_deposit"]["invocation"] == 1 and new["cases"]["c04_deposit"]["pass"] is False
        assert new["cases"]["mpi"]["pass"] is None and new["cases"]["mpi"]["invocation"] == 1
        assert open(os.path.join(d, "capacity", "sentinel")).read() == "preserve capacity"


def test_a01_overwrite_moves_never_deletes():
    """--overwrite supersedes: the old directory is renamed and its record kept."""
    with tempfile.TemporaryDirectory() as d:
        _campaign(d, ["capacity"])
        _harness(d, "--cases", "capacity", "--overwrite")
        new = json.load(open(os.path.join(d, "results.json")))
        sup = [k for k in new["superseded"] if k.startswith("capacity@")]
        assert len(sup) == 1
        rec = new["superseded"][sup[0]]
        assert rec["record"]["checks"] == [{"old_evidence": "capacity"}]
        assert open(os.path.join(d, rec["directory"], "sentinel")).read() == "preserve capacity"


# --------------------------------------------------------------------------
# A0-2: rounding ambiguity of output positions
# --------------------------------------------------------------------------
def _rounded(i, dx, off=0):
    """Entity's conversion (kernels/prtls_to_phys.hpp): x = x_min + (float(i) + dx), float."""
    v = (np.float32(i) + np.float32(dx)).astype(np.float32)
    return (np.float32(off) + v).astype(np.float32).astype(np.float64)


def test_a02_integer_boundary_both_sides():
    """Codex's witness: (99, 1 - 1e-6) and (100, 1e-6) both give X = 100; the
    rounded source flags both, any single rule misassigns one of them; the exact
    source (cell, offset) assigns both correctly. Also with a domain offset."""
    for off in (0, 4160):
        i = np.array([99, 100]) - off * 0
        dx = np.array([1 - 1e-6, 1e-6], dtype=np.float32)
        X = _rounded(i, dx, off)
        assert X[0] == X[1] == 100 + off
        cell, frac, amb, _ = check_particles.classify_positions(X)
        assert amb.all() and (cell == 100 + off).all()            # floor: the first is wrong
        assert (cell != i + off).sum() == 1
    # unambiguous neighbours on both sides
    X = _rounded(np.array([99, 100]), np.array([0.999, 0.001], dtype=np.float32))
    cell, _, amb, _ = check_particles.classify_positions(X)
    assert not amb.any() and list(cell) == [99, 100]
    # exact source: no flags, correct cells
    V = {"species": {4: {"i1": np.array([99, 100]), "i2": np.array([3, 3]), "dx1": dx, "dx2": dx,
                         "u1": np.zeros(2, np.float32), "u2": np.zeros(2, np.float32),
                         "u3": np.zeros(2, np.float32), "w": np.ones(2, np.float32),
                         "cv": np.zeros(2, np.uint64), "rank": np.zeros(2, np.int64)}}}
    P = check_particles.particles_from_vdump(V)[4]
    assert list(P["i"]) == [99, 100] and not P["amb_cell"].any()


def test_a02_half_boundary_both_sides():
    """The dx1 < 1/2 subsample: dx just below 1/2 rounds to X = i + 1/2 like
    dx = 1/2 itself; the rounded source flags both, the exact source selects
    exactly the first (the kernel's float comparison); just inside on either
    side is unambiguous."""
    below = np.nextafter(np.float32(0.5), np.float32(0))
    dx = np.array([below, 0.5, 0.4999, 0.5001], dtype=np.float32)
    X = _rounded(np.full(4, 10), dx)
    assert X[0] == X[1] == 10.5
    _, frac, _, amb = check_particles.classify_positions(X)
    assert list(amb) == [True, True, False, False]
    assert list(frac[2:] < 0.5) == [True, False]
    P = {"dx1_f32": dx, "dx1": dx.astype(np.float64)}
    assert list(check_particles.half_select(P)) == [True, False, True, False]


def test_a02_exact_source_changes_occupancy_count():
    """A particle at the largest float dx1 below 1 in cell 3 (its own, masked
    cell) next to a crowd in cell 4: X rounds to 4.0, the floor rule moves it
    into cell 4 and the masked count changes; the exact cells keep it."""
    n = 25
    i = np.r_[3, np.full(n, 4)]
    dx1 = np.r_[np.nextafter(np.float32(1), np.float32(0)), np.full(n, 0.25, np.float32)]
    j = np.full(n + 1, 3)
    u = np.zeros(n + 1, np.float32)
    V = {"species": {4: {"i1": i, "i2": j, "dx1": dx1, "dx2": np.full(n + 1, 0.5, np.float32),
                         "u1": u, "u2": u, "u3": u, "w": np.ones(n + 1, np.float32),
                         "cv": np.zeros(n + 1, np.uint64), "rank": np.zeros(n + 1, np.int64)}}}
    exact = check_particles.particles_from_vdump(V)[4]
    B = {4: {"X1": _rounded(i, dx1), "X2": _rounded(j, np.full(n + 1, 0.5, np.float32)),
             "U1": u.astype(float), "U2": u.astype(float), "U3": u.astype(float), "W": np.ones(n + 1)}}
    rounded = check_particles.particles_from_bp(B)[4]
    assert rounded["amb_cell"].sum() == 1
    mask = np.zeros((8, 8), bool)
    mask[3, 3] = True
    e, _ = check_particles.bulk_estimate(exact, 1.0, mask, 0, 20, 8, 8, False)
    r, _ = check_particles.bulk_estimate(rounded, 1.0, mask, 0, 20, 8, 8, False)
    assert e["N"] == 1 and e["Nlowocc"] == 1 and r["N"] == 0


# --------------------------------------------------------------------------
# A0-3: witnesses of the qualified claims (documented limitations)
# --------------------------------------------------------------------------
def _parts(i, j, dx1, beta1):
    n = len(i)
    g = 1 / np.sqrt(1 - np.asarray(beta1) ** 2)
    return {"i": np.asarray(i), "j": np.asarray(j), "dx1": np.asarray(dx1, float),
            "U1": np.asarray(beta1) * g, "U2": np.zeros(n), "U3": np.zeros(n), "W": np.ones(n),
            "amb_cell": np.zeros(n, bool), "amb_half": np.zeros(n, bool)}


def test_a03a_jensen_box_mismatch():
    """Codex's witness: a stationary masked particle with nine unmasked
    neighbours at beta 0.9 in its box. Klowocc = 0 and Kblowocc > 0, so the
    selected random energy is negative: [0, Klowocc] does not bound the
    physical random energy. What holds is the algebraic bound on the recorded
    contribution, |Klowocc - Kblowocc| <= max(Klowocc, Kblowocc)."""
    p = _parts([3] + [4] * 9, [3] * 10, [0.25] * 10, [0.0] + [0.9] * 9)
    mask = np.zeros((8, 8), bool)
    mask[3, 3] = True
    e, _ = check_particles.bulk_estimate(p, 1.0, mask, 1, 20, 8, 8, False)
    assert e["K"] == 0 and e["Klowocc"] == 0 and e["Kblowocc"] > 0.68
    assert e["K"] - e["Kb_own"] < 0                                   # negative selected energy
    assert abs(e["Klowocc"] - e["Kblowocc"]) <= max(e["Klowocc"], e["Kblowocc"])


def test_a03b_position_subsampling_selects_a_stream():
    """Codex's witness: two streams (beta_x = +-0.8) at dx1 = 1/4 and 3/4 in one
    masked cell: the full-sample eps-tilde is 2/3, the dx1 < 1/2 subsample sees
    one stream and gives ~0 (-100%), at any occupancy: the change measures the
    phase-space selection, not a finite-sample bias."""
    for n in (100, 1000):
        h = n // 2
        p = _parts([3] * n, [3] * n, [0.25] * h + [0.75] * h, [0.8] * h + [-0.8] * h)
        mask = np.zeros((8, 8), bool)
        mask[3, 3] = True
        full, _ = check_particles.bulk_estimate(p, 1.0, mask, 1, 20, 8, 8, False)
        half, _ = check_particles.bulk_estimate(p, 1.0, mask, 1, 20, 8, 8, False, half=True)
        e_full = (full["K"] - full["Kb_own"]) / n
        e_half = (full["K"] - half["Kb_own_half"]) / n
        assert abs(e_full - 2 / 3) < 1e-9 and abs(e_half) < 1e-9
