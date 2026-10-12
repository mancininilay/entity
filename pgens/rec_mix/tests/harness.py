#!/usr/bin/env python3
"""Reproducible C0/C1 test harness for rec_mix (P1.3 tech spec V.2; Codex A3).

Writes every input with make_input.py (explicit arguments and seeds), runs the
given binaries, checks the registered tolerances and writes <out>/results.json;
the inputs and logs stay next to it, one directory per case. Exit status 1 if
any check fails, 2 if the invocation was refused before anything was written.

Evidence preservation (Codex A0-1). Before anything is written, every selected
case is checked against the existing <out>: a case whose directory or
results.json record already exists is a collision, and the whole invocation is
refused with nothing modified (also for cases that would be skipped for a
missing binary). An invocation with only new cases adds to an existing
campaign: results.json keeps every previous record and lists, per invocation,
the cases it added (provenance), and every case record names its invocation.
--overwrite never deletes: the old case directory is renamed
<case>.superseded.<time> and its record moved to results["superseded"].
results.json is replaced atomically. Use a new --out directory per campaign.

    python3 harness.py --recmix BUILD/src/entity.xc --out DIR
        [--recmix-mpi BUILD_MPI/src/entity.xc] [--recmix-mpio BUILD_MPIO/src/entity.xc]
        [--mpirun "mpirun -n 4"]
        [--stock-recon STOCK/src/entity.xc --fork-recon FORK/src/entity.xc]
        [--cases name1,name2,...] [--overwrite]

Builds: rec_mix with output=ON (single precision, CPU); for c04_default and
c05_seed the stock `reconnection` pgen built from the base commit 887c92d9
(--stock-recon) and from this branch (--fork-recon); for `mpi` a rec_mix
build with mpi=ON (output may be OFF); for `tracking` (C1.12: MPI exchange and
a restart) a rec_mix build with mpi=ON and output=ON (ADIOS2 with MPI;
--recmix-mpio, may be the same binary as --recmix-mpi). Cases whose binary is
missing are recorded as skipped.

Requirements: Python >= 3.10, NumPy, SciPy (restart_stat), the ADIOS2 Python
bindings, mpirun for the MPI cases. All cases take about 40 minutes on two
CPU cores.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.join(HERE, "..", "analysis")
sys.path.insert(0, ANA)
sys.path.insert(0, os.path.join(ANA, "tests"))
from p13ana import cvbudget, partition, readers  # noqa: E402

MAKE_INPUT = os.path.join(HERE, "..", "make_input.py")
DT = 0.45  # CFL 0.6364 / sqrt(2) with the make_input defaults (cell units)

# the small active box shared by most cases (cell units); explicit seeds per case.
# P13-D39 nominal hot labels: sigma_hot 4, d_up 4 cells, T = 0.01 give
# sigma0 = 4 (1.75 + 0.025) = 7.1, skindepth0 = 4 sqrt(0.8125/1.015) = 3.5788
# (v1.5 campaigns used the cold mapping, sigma0 7, skindepth0 3.6056)
BASE = ["--layout", "active", "--sigma-hot", "4", "--q", "0.5", "--mu", "4", "--dup", "4",
        "--ppc0", "16", "--n1", "192", "--n2", "128", "--cs-width", "8", "--padding", "8",
        "--filters", "4", "--ds-abs", "5", "--t-open-tlc", "0.5", "--decomposition", "1", "1",
        "--checkpoint-interval", "100000"]

# Pre-registered tolerances (P1.3 tech spec V.2, v1.4). Entries marked
# "proposal" in the spec await Nilay's confirmation.
TOL = {
    "C1.1_floor": 0.01,          # max(1%, 3 x Poisson) for densities and temperatures
    "C1.2_balance": 0.05,        # |E3 after one step| / (dt |curl B|) outside the dip
    "C1.2_stress": 0.02,         # T22 vs 1/(D Gamma_d) (legacy) or 1 (balanced), x B0^2/8pi
    "C1.3_global": 1e-4,         # |mean charge| in e n0 (proposal)
    "C1.3_rms": [0.8, 1.25],     # rms block charge / crossing model (proposal)
    "C1.14_rF": 0.01,            # |dU + F + Wj| / |Wj| over the run (proposal)
    "C1.14_rC": 0.02,            # |sum W_dep - Wj| / |sum W_dep| (proposal)
    "C1.14_rtot": 0.02,          # |sum W_dep + dU + F| / |sum W_dep| (V.2)
    "C1.14_cv_interval": 0.01,   # species work change, cv_interval 5 vs 1 (V.2; informational)
    "C1.16": 1e-4,               # in-code vs stock V_s output bulk energies
    "C1.16b": 1e-3,              # in-code vs particle-output bulk energies
    "C1.5_reflection": 0.05,     # reflected / incident pulse energy, each wall type (III.4)
    "C1.5_layer": 0.02,          # |F - F_target|/B0 in the layers, steady upstream (III.4)
    "C1.7_alpha": 0.01,          # family-wise false-alarm rate of the statistical restart test
                                 # (operationalization: proposal, Claude 2026-10-12)
    "D39_scales_rel": 1e-6,      # meta scales vs fixed anchors (float32 inputs)
    "D39_theta_cs_b": 5e-6,      # C2-b sheet-ion theta vs the quoted .06825 (rounding)
}


class Case:
    def __init__(self, name, out):
        self.name, self.dir = name, os.path.join(out, name)
        # second guard (main() preflights every collision): never reuse a directory
        if os.path.exists(self.dir):
            raise FileExistsError(f"{self.dir} exists; use a new --out directory for each campaign")
        os.makedirs(self.dir)
        self.metrics, self.checks = {}, []

    def check(self, label, ok, value=None, tol=None):
        self.checks.append({"check": label, "pass": bool(ok), "value": value, "tol": tol})
        return ok

    def result(self):
        return {"pass": all(c["pass"] for c in self.checks), "checks": self.checks,
                "metrics": self.metrics}


def make_input(case, name, extra, edits=(), subdir=None):
    d = os.path.join(case.dir, subdir) if subdir else case.dir
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.toml")
    cmd = [sys.executable, MAKE_INPUT, "--name", name, "--out", path] + BASE + list(extra)
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    case.metrics.setdefault("make_input", []).append(
        {"args": " ".join(cmd[2:]), "stdout": p.stdout.strip()})
    s = open(path).read()
    for old, new in edits:
        assert old in s, (old, path)
        s = s.replace(old, new)
    open(path, "w").write(s)
    return path


def edit(path, edits, dst=None):
    """Regex edits, each matching exactly once; written to dst (default: in place)."""
    s = open(path).read()
    for old, new in edits:
        n = len(re.findall(old, s))
        assert n == 1, (old, n, path)
        s = re.sub(old, new, s)
    open(dst or path, "w").write(s)
    return dst or path


def run(case, binary, toml, threads=1, extra=(), mpirun=None, expect_fail=False):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND="false")
    cmd = (mpirun.split() if mpirun else []) + [binary, "-input", os.path.basename(toml), *extra]
    log = toml.replace(".toml", ".log")
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.run(cmd, cwd=os.path.dirname(toml), env=env, stdout=f, stderr=subprocess.STDOUT)
    case.metrics.setdefault("runs", []).append(
        {"cwd": os.path.relpath(os.path.dirname(toml), case.dir), "cmd": " ".join(cmd),
         "threads": threads, "exit": p.returncode, "seconds": round(time.time() - t0, 1)})
    text = open(log, errors="replace").read()
    if not expect_fail and p.returncode != 0:
        raise RuntimeError(f"{case.name}: {' '.join(cmd)} failed ({p.returncode}); see {log}")
    return p.returncode, text


def check_pusher(case, logtext, nspec=9):
    n = len(re.findall(r"Pusher: vay", logtext))
    return case.check("IC1 every species uses the Vay pusher", n == nspec, n, nspec)


def fields_equal(case, d1, d2, names, label):
    A = readers.list_entity_fields(d1, allow_unmanifested=True)
    B = readers.list_entity_fields(d2, allow_unmanifested=True)
    mx = 0.0
    for a, b in zip(A, B):
        Fa, Fb = readers.read_entity_fields(a, names), readers.read_entity_fields(b, names)
        for k in names:
            mx = max(mx, float(np.max(np.abs(Fa[k].astype(np.float64) - Fb[k]))))
    ok = len(A) == len(B) and len(A) > 0 and mx == 0.0
    return case.check(label, ok, {"outputs": [len(A), len(B)], "max_abs_diff": mx}, 0.0)


def copy_checkpoint(src_ckpt, dst_ckpt, label):
    os.makedirs(dst_ckpt, exist_ok=True)
    tag = f"{label:08d}"
    found = False
    for f in os.listdir(src_ckpt):
        if tag in f:
            s, d = os.path.join(src_ckpt, f), os.path.join(dst_ckpt, f)
            (shutil.copytree if os.path.isdir(s) else shutil.copy)(s, d)
            found = True
    assert found, (src_ckpt, label)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x


NOINJ = ("  bulk_window      = 2", "  bulk_window      = 2\n  inject           = false")
SMOOTH0 = ("      order  = 4", "      order  = 0")
OPEN_X1 = [(r'fields    = \[\["PERIODIC"\], \["MATCH", "MATCH"\]\]',
            'fields    = [["MATCH", "MATCH"], ["MATCH", "MATCH"]]'),
           (r'particles = \[\["PERIODIC"\], \["ABSORB", "ABSORB"\]\]',
            'particles = [["ABSORB", "ABSORB"], ["ABSORB", "ABSORB"]]')]
EBJ = ["E1", "E2", "E3", "B1", "B2", "B3", "J1", "J2", "J3"]


# --------------------------------------------------------------------------
# C0: core changes (stock reconnection generator)
# --------------------------------------------------------------------------
STOCK_RECON = """[simulation]
  name    = "recon"
  engine  = "srpic"
  runtime = 150.0
{seed}
  [simulation.domain]
    decomposition = [-1, -1]

[grid]
  resolution = [256, 128]
  extent     = [[-64.0, 64.0], [-32.0, 32.0]]

  [grid.metric]
    metric = "minkowski"

  [grid.boundaries]
    fields    = [["PERIODIC"], ["MATCH", "MATCH"]]
    particles = [["PERIODIC"], ["ABSORB", "ABSORB"]]

    [grid.boundaries.match]
      ds = [[5.0], [5.0]]

[scales]
  larmor0    = 0.1
  skindepth0 = 1.0

[algorithms]
  current_filters = 4

  [algorithms.timestep]
    CFL = 0.5

[particles]
  ppc0 = 8.0

  [[particles.species]]
    label    = "e-"
    mass     = 1.0
    charge   = -1.0
    maxnpart = 1e6

  [[particles.species]]
    label    = "e+"
    mass     = 1.0
    charge   = 1.0
    maxnpart = 1e6

[setup]
  bg_B           = 1.0
  bg_Bguide      = 0.0
  bg_temperature = 1e-4
  inj_ypad       = 5.0
  cs_width       = 2.0
  cs_overdensity = 3.0

[output]
  format   = "bpfile"
  interval = 50

  [output.fields]
    quantities = ["N_1", "N_2", "E", "B", "J", "T00_1_2"]

  [output.particles]
    enable = false

  [output.spectra]
    enable = false
"""
RECON_FIELDS = ["N_1", "N_2", "E1", "E2", "E3", "B1", "B2", "B3", "J1", "J2", "J3", "T00_1_2"]


def case_c04_default(a, out):
    """C0.4/C0.5 default: stock reconnection pgen, base binary vs fork binary, no new
    keys; all field outputs and the stats file bit-identical (one thread, 333 steps,
    across the stock t_open)."""
    c = Case("c04_default", out)
    for tag, b in (("stock", a.stock_recon), ("fork", a.fork_recon)):
        os.makedirs(os.path.join(c.dir, tag))
        t = os.path.join(c.dir, tag, "recon.toml")
        open(t, "w").write(STOCK_RECON.format(seed=""))
        run(c, b, t)
    fields_equal(c, os.path.join(c.dir, "stock", "recon"), os.path.join(c.dir, "fork", "recon"),
                 RECON_FIELDS, "C0.4 stock vs fork field outputs bit-identical")
    sa = open(os.path.join(c.dir, "stock", "recon", "recon_stats.csv")).read()
    sb = open(os.path.join(c.dir, "fork", "recon", "recon_stats.csv")).read()
    c.check("C0.4 stats file bit-identical", sa == sb and len(sa) > 0)
    return c


def case_c05_seed(a, out):
    """C0.5: two seeds give different realisations with statistically equal loads."""
    c = Case("c05_seed", out)
    n1, last = {}, {}
    for sd in (1, 2):
        os.makedirs(os.path.join(c.dir, f"s{sd}"))
        t = os.path.join(c.dir, f"s{sd}", "recon.toml")
        open(t, "w").write(STOCK_RECON.format(seed=f"  seed    = {sd}\n"))
        run(c, a.fork_recon, t)
        lg = open(os.path.join(c.dir, f"s{sd}", "recon", "recon.log")).read()
        c.check(f"C0.5 seed {sd} applied", f"Re-seeding RNG pools with base {sd}" in lg)
        fl = readers.list_entity_fields(os.path.join(c.dir, f"s{sd}", "recon"), allow_unmanifested=True)
        n1[sd] = float(readers.read_entity_fields(fl[0], ["N_1"])["N_1"].sum()) * 8.0
        last[sd] = readers.read_entity_fields(fl[-1], ["E3"])["E3"]
    z = abs(n1[1] - n1[2]) / math.sqrt(n1[1] + n1[2])
    c.metrics["N_1_initial"] = n1
    c.check("C0.5 initial N equal within 4 sigma (Poisson)", z < 4, z, 4)
    c.check("C0.5 realisations differ", float(np.max(np.abs(last[1] - last[2]))) > 0)
    return c


def case_c04_deposit(a, out):
    """C0.4 CH1: non-depositing tracers leave E, B, J bit-identical (test layout,
    injection off, one thread, across t_open at step 86)."""
    c = Case("c04_deposit", out)
    common = ["--layout", "testprtl", "--test-masses", "4", "16", "64", "--steps", "101",
              "--n1", "128", "--usr-interval", "50", "--line-interval", "50",
              "--field-interval", "50", "--field-stride", "1", "--t-open-tlc", "0.3", "--seed", "11"]
    t1 = make_input(c, "tr", common, [NOINJ])
    t2 = make_input(c, "notr", common, [NOINJ, ("  inject           = false",
                                                "  inject           = false\n  load_tracers     = false")])
    _, l1 = run(c, a.recmix, t1)
    run(c, a.recmix, t2)
    check_pusher(c, l1)
    c.check("C0.4 tracers report Deposit OFF", l1.count("Deposit: OFF") == 3, l1.count("Deposit: OFF"), 3)
    fields_equal(c, os.path.join(c.dir, "tr"), os.path.join(c.dir, "notr"), EBJ,
                 "C0.4 E, B, J bit-identical with and without tracers")
    _, oa = readers.read_all_recmix(os.path.join(c.dir, "tr", "usr"))
    _, ob = readers.read_all_recmix(os.path.join(c.dir, "notr", "usr"))
    lep = [0, 1, 3, 4, 6, 7]
    same = len(oa) == len(ob) and all(np.array_equal(x.m("Ntot")[lep], y.m("Ntot")[lep]) and
                                      np.array_equal(x.m("Ktot")[lep], y.m("Ktot")[lep])
                                      for x, y in zip(oa, ob))
    c.check("C0.4 lepton moments identical", same)
    c.check("C0.4 tracers present and moving", oa[-1].m("Ntot")[2] > 0 and
            oa[-1].m("Ktot")[2] != oa[0].m("Ktot")[2])
    return c


def case_c06_restart(a, out):
    """C0.6 and C1.7 exact: restart at label 51 of a deterministic run (test layout,
    no injection, one thread, t_open at step 86): deposit = false survives; fields,
    spectra and moments at 101 bit-identical; the restart is a new segment."""
    c = Case("c06_restart", out)
    common = ["--layout", "testprtl", "--test-masses", "4", "16", "64", "--steps", "101",
              "--n1", "128", "--usr-interval", "50", "--line-interval", "50",
              "--field-interval", "50", "--field-stride", "1", "--t-open-tlc", "0.3", "--seed", "12"]
    tA = make_input(c, "rs", common, [NOINJ, ("  interval = 100000", "  interval = 50")], subdir="A")
    run(c, a.recmix, tA)
    os.makedirs(os.path.join(c.dir, "B"))
    tB = os.path.join(c.dir, "B", "rs.toml")
    shutil.copy(tA, tB)
    copy_checkpoint(os.path.join(c.dir, "A", "rs.ckpt"), os.path.join(c.dir, "B", "rs.ckpt"), 51)
    _, lb = run(c, a.recmix, tB, extra=("-continue",))
    check_pusher(c, lb)
    c.check("C0.6 tracers report Deposit OFF after restart", lb.count("Deposit: OFF") == 3,
            lb.count("Deposit: OFF"), 3)
    fa = os.path.join(c.dir, "A", "rs", "fields", "fields.00000101.bp")
    fb = os.path.join(c.dir, "B", "rs", "fields", "fields.00000101.bp")
    Fa, Fb = readers.read_entity_fields(fa, EBJ), readers.read_entity_fields(fb, EBJ)
    mx = max(float(np.max(np.abs(Fa[k] - Fb[k]))) for k in EBJ)
    c.check("C1.7 fields bit-identical after restart (label 101, across t_open)", mx == 0.0, mx, 0.0)
    _, oa = readers.read_all_recmix(os.path.join(c.dir, "A", "rs", "usr"))
    _, ob = readers.read_all_recmix(os.path.join(c.dir, "B", "rs", "usr"))
    xa = [o for o in oa if o.step == 101][0]
    xb = [o for o in ob if o.step == 101][0]
    c.check("C1.7 spectra and moments bit-identical after restart",
            np.array_equal(xa.hist, xb.hist) and
            np.array_equal(np.nan_to_num(xa.moms, nan=-7.0), np.nan_to_num(xb.moms, nan=-7.0)))
    segs = readers.read_segments(os.path.join(c.dir, "B", "rs", "usr"))
    c.check("IC3 restart recorded as a segment starting at label 51",
            len(segs) == 1 and segs[0]["start_label"] == 51 and segs[0]["resuming"], segs)
    return c


def case_c07_open_restart(a, out):
    """K5 and C1.7 across t_open (exact): a restart past t_open with the x1 walls
    open in the restart input is bit-identical to the uninterrupted run (active
    layout, no injection, one thread); an open-wall restart before t_open is refused."""
    c = Case("c07_open_restart", out)
    common = ["--steps", "351", "--usr-interval", "50", "--line-interval", "50",
              "--field-interval", "50", "--field-stride", "1", "--seed", "13"]
    tR = make_input(c, "op", common, [NOINJ, ("  interval = 100000", "  interval = 100")], subdir="ref")
    _, lr = run(c, a.recmix, tR)
    check_pusher(c, lr)
    m = re.search(r"outflow boundaries opened at t = ([0-9.e+]+) \(step (\d+)\)", lr)
    c.metrics["opened"] = m.group(0) if m else None
    for sub, lab in (("past", 301), ("before", 201)):
        os.makedirs(os.path.join(c.dir, sub))
        copy_checkpoint(os.path.join(c.dir, "ref", "op.ckpt"), os.path.join(c.dir, sub, "op.ckpt"), lab)
        edit(tR, OPEN_X1, os.path.join(c.dir, sub, "op.toml"))
    rc, lg = run(c, a.recmix, os.path.join(c.dir, "before", "op.toml"), extra=("-continue",),
                 expect_fail=True)
    c.check("K5 open x1 walls refused when resuming before t_open",
            rc != 0 and "only allowed when resuming" in lg, rc)
    _, lp = run(c, a.recmix, os.path.join(c.dir, "past", "op.toml"), extra=("-continue",))
    c.check("K5 no second opening after an open-wall restart", "outflow boundaries opened" not in lp)
    Fa = readers.read_entity_fields(os.path.join(c.dir, "ref", "op", "fields", "fields.00000351.bp"), EBJ)
    Fb = readers.read_entity_fields(os.path.join(c.dir, "past", "op", "fields", "fields.00000351.bp"), EBJ)
    mx = max(float(np.max(np.abs(Fa[k] - Fb[k]))) for k in EBJ)
    c.check("K5 restart past t_open with open walls bit-identical at label 351", mx == 0.0, mx, 0.0)
    return c


def case_provenance(a, out):
    """IC3 end to end in one simulation directory.
    (1) Rollback: run A to label 301 (checkpoints every 100); B resumes from 101
        with a changed field-output cadence (50; Entity takes [setup], hence the
        rec_mix cadences, from the checkpoint, but [output] from the new input)
        and stops at 201. Valid: A's outputs <= 101 and B's, none of A's later.
    (2) Fresh rerun in the same directory with a new user-output cadence (25):
        only its own outputs are valid."""
    c = Case("provenance", out)
    common = ["--steps", "301", "--usr-interval", "50", "--line-interval", "50",
              "--field-interval", "100", "--field-stride", "4", "--seed", "14"]
    t = make_input(c, "pv", common, [("  interval = 100000", "  interval = 100"),
                                     ("  keep     = 2", "  keep     = 5")])
    run(c, a.recmix, t)
    ckd = os.path.join(c.dir, "pv.ckpt")
    for f in os.listdir(ckd):  # roll back: only the checkpoint at 101 remains
        if "00000201" in f or "00000301" in f:
            p = os.path.join(ckd, f)
            (shutil.rmtree if os.path.isdir(p) else os.remove)(p)
    tb = edit(t, [(r"runtime = [0-9.e+]+", f"runtime = {201 * DT:.6f}"),
                  (r"    interval     = 100\n", "    interval     = 50\n")], t.replace(".toml", "_b.toml"))
    run(c, a.recmix, tb, extra=("-continue",))
    usr = os.path.join(c.dir, "pv", "usr")
    sim = os.path.join(c.dir, "pv")

    def labels():
        _, outs = readers.read_all_recmix(usr)
        ml = [readers._step_of(p) for p in readers.list_midlines(usr)]
        fl = [int(re.search(r"fields\.(\d+)", p).group(1)) for p in readers.list_entity_fields(sim)]
        st = readers.read_cvbudget(usr)["step"]
        return [o.step for o in outs], ml, fl, st

    r, ml, fl, st = labels()
    c.metrics["rollback"] = {"recmix": r, "midline": ml, "fields": fl}
    c.check("IC3 rollback: spectra, A's future rejected", r == [1, 51, 101, 151, 201], r)
    c.check("IC3 rollback: midlines, A's future rejected", ml == [1, 51, 101, 151, 201], ml)
    c.check("IC3 rollback: fields, A's 301 rejected, B's new cadence kept", fl == [1, 101, 151, 201], fl)
    c.check("IC3 rollback: CV records (user-output labels and the segment start 102)",
            st.tolist() == [1, 51, 101, 102, 151, 201], st.tolist())
    tc = edit(t, [(r"runtime = [0-9.e+]+", f"runtime = {101 * DT:.6f}"),
                  (r"  usr_interval     = 50", "  usr_interval     = 25")], t.replace(".toml", "_c.toml"))
    run(c, a.recmix, tc)
    r, ml, fl, st = labels()
    c.metrics["fresh_rerun"] = {"recmix": r, "midline": ml, "fields": fl}
    c.check("IC3 fresh rerun: spectra, only its own (cadence 25)", r == [1, 26, 51, 76, 101], r)
    c.check("IC3 fresh rerun: midlines, only its own", ml == [1, 51, 101], ml)
    c.check("IC3 fresh rerun: fields, only its own", fl == [1, 101], fl)
    c.check("IC3 fresh rerun: CV records, only its own", st.tolist() == [1, 26, 51, 76, 101], st.tolist())
    return c


# --------------------------------------------------------------------------
# C1: initial state
# --------------------------------------------------------------------------
def _sheet_case(a, out, init):
    c = Case(f"sheet_{init}", out)
    q = ", ".join(['"T22_1_2_3"', '"Charge"'] + [f'"T22_{s}"' for s in range(1, 10)])
    t = make_input(c, "sh", ["--sheet-init", init, "--steps", "3", "--usr-interval", "1",
                             "--field-interval", "1", "--field-stride", "1", "--seed", "21"],
                   [SMOOTH0, ('quantities   = ["E", "B", "J",', f'quantities   = ["E", "B", "J", {q},')])
    _, lg = run(c, a.recmix, t, threads=2)
    check_pusher(c, lg)
    sim = os.path.join(c.dir, "sh")
    import check_sheet
    r = check_sheet.main(sim)
    c.metrics.update(r)
    c.check("C1.2 sheet current balances curl B (E3 after one step / dt curl B)",
            abs(r["C1.2_balance_E3_over_dt_curlB"]) <= TOL["C1.2_balance"],
            r["C1.2_balance_E3_over_dt_curlB"], TOL["C1.2_balance"])
    rel = r["C1.2_T22_over_PB"] / r["C1.2_expected_over_PB"] - 1
    c.check(f"C1.2 normal stress vs {'1/(D Gamma_d)' if init == 'legacy' else '1'} x B0^2/8pi",
            abs(rel) <= TOL["C1.2_stress"], rel, TOL["C1.2_stress"])
    # C1.1 at label 1: kinetic temperatures <m u2^2/gamma> per species (sheet
    # species over the whole box: kT_cs/Gamma_d; upstream species in their own
    # band, 3a to 5.6a from the plane: theta_up) and upstream densities [1, 1-q, q] n0/2
    meta = readers.read_meta(os.path.join(sim, "usr"))
    d, inp = meta["derived"], meta["inputs"]
    F = readers.read_entity_fields(readers.list_entity_fields(sim)[0])
    G = 1 / math.sqrt(1 - d["beta_d"] ** 2)
    ppc0, qq = d["ppc0"], inp["q"]
    band = {"lower": (F["X2"] > 19) & (F["X2"] < 40), "upper": (F["X2"] > 88) & (F["X2"] < 109)}
    rows = []
    for s in range(1, 10):
        if s <= 3:
            Ns, Ts, T0 = F[f"N_{s}"].sum() * ppc0, F[f"T22_{s}"].sum() * ppc0, d["T_cs"] / G
        else:
            up = band["lower" if s <= 6 else "upper"]
            Ns, Ts, T0 = F[f"N_{s}"][:, up].sum() * ppc0, F[f"T22_{s}"][:, up].sum() * ppc0, inp["theta_up"]
        relT = (Ts / Ns) / T0 - 1
        tolT = max(TOL["C1.1_floor"], 3 * math.sqrt(3 / Ns))
        row = {"species": s, "N": Ns, "T_rel": relT, "T_tol": tolT}
        c.check(f"C1.1 kinetic temperature, species {s}", abs(relT) <= tolT, relT, tolT)
        if s >= 4:
            nexp = [0.5, 0.5 * (1 - qq), 0.5 * qq][(s - 4) % 3] * ppc0 * F[f"N_{s}"][:, up].size
            reln = Ns / nexp - 1
            toln = max(TOL["C1.1_floor"], 3 / math.sqrt(nexp))
            row.update({"n_rel": reln, "n_tol": toln})
            c.check(f"C1.1 upstream density, species {s}", abs(reln) <= toln, reln, toln)
        rows.append(row)
    c.metrics["C1.1"] = rows
    return c


def case_sheet_legacy(a, out):
    return _sheet_case(a, out, "legacy")


def case_sheet_balanced(a, out):
    return _sheet_case(a, out, "balanced")


def case_neutrality(a, out):
    """C1.3a (exact load neutrality, checked by the generator) and C1.3b (global
    charge and the noise-aware local test far upstream, at label 1)."""
    c = Case("neutrality", out)
    t = make_input(c, "ne", ["--n1", "160", "--n2", "256", "--cs-width", "4", "--steps", "3",
                             "--usr-interval", "1", "--field-interval", "1", "--field-stride", "1",
                             "--seed", "3", "--t-open-tlc", "5"],
                   [SMOOTH0, ('quantities   = ["E", "B", "J",', 'quantities   = ["E", "B", "J", "Charge",')])
    _, lg = run(c, a.recmix, t, threads=2)
    check_pusher(c, lg)
    c.check("C1.3a load neutral cell by cell", "max |net charge| per cell = 0 e" in lg)
    import check_neutrality
    r = check_neutrality.main(os.path.join(c.dir, "ne"))
    c.metrics.update(r)
    c.check("C1.3 global mean charge", abs(r["global_mean_charge"]) <= TOL["C1.3_global"],
            r["global_mean_charge"], TOL["C1.3_global"])
    c.check("C1.3b rms block charge within [0.8, 1.25] of the crossing model",
            TOL["C1.3_rms"][0] <= r["rms_ratio"] <= TOL["C1.3_rms"][1], r["rms_ratio"], TOL["C1.3_rms"])
    c.check("C1.3b largest block charge below the discrete (Skellam) Bonferroni bound, alpha = 1e-3",
            r["q_max"] <= r["q_bound"], r["q_max"], r["q_bound"])
    c.metrics["C1.3b_gaussian_approximation"] = {"z_max": r["z_max"], "z_bound": r["z_bound"]}
    return c


# --------------------------------------------------------------------------
# C1: reconnecting box (601 steps; t_open = 96 at step 214)
# --------------------------------------------------------------------------
def _recon(c, a, name, extra=(), edits=(), threads=2, steps=601):
    t = make_input(c, name, ["--steps", str(steps), "--usr-interval", "50", "--line-interval", "50",
                             "--field-interval", "100", "--field-stride", "1", "--seed", "31"] + list(extra),
                   list(edits))
    _, lg = run(c, a.recmix, t, threads=threads)
    check_pusher(c, lg)
    c.check("C1.6 outflow walls opened at t_open", "outflow boundaries opened" in lg)
    return os.path.join(c.dir, name)


def _cv_checks(c, sim, tag):
    """C1.14: the three residuals over the whole run (label 1 to the end); the
    cumulative residuals at every user-output label are reported, not judged."""
    usr = os.path.join(sim, "usr")
    meta = readers.read_meta(usr)
    cv = readers.read_cvbudget(usr, meta=meta)
    cl = cvbudget.closure(cv, meta=meta)
    c.metrics[f"{tag}_closure"] = jsonable(cl)
    iv = cvbudget.intervals(cv, meta=meta)
    cum = []
    for lab in range(51, int(iv["step1"][-1]) + 1, 50):
        s = iv["step1"] <= lab
        W, Wj = iv["W_phys"][s].sum(), iv["Wj"][s].sum()
        cum.append({"label": lab, "W": float(W), "Wj": float(Wj),
                    "r_F_rel": float(iv["r_F"][s].sum() / Wj), "r_C_rel": float(iv["r_C"][s].sum() / W),
                    "r_total_rel": float(iv["r_total"][s].sum() / W)})
    c.metrics[f"{tag}_cumulative"] = cum
    c.check(f"C1.14 r_F ({tag})", cl["r_F_rel"] <= TOL["C1.14_rF"], cl["r_F_rel"], TOL["C1.14_rF"])
    c.check(f"C1.14 r_C ({tag})", cl["r_C_rel"] <= TOL["C1.14_rC"], cl["r_C_rel"], TOL["C1.14_rC"])
    c.check(f"C1.14 r_total ({tag})", cl["rel_residual"] <= TOL["C1.14_rtot"], cl["rel_residual"],
            TOL["C1.14_rtot"])
    return meta, cv, cl


def case_recon_s0(a, out):
    """C1.11, C1.15 and C1.14 (4 filter passes) on a reconnecting box with
    unsmoothed moment outputs; partition summary with the bulk status."""
    c = Case("recon_s0", out)
    sim = _recon(c, a, "r0", edits=[SMOOTH0])
    import check_run
    r = check_run.main(sim)
    c.check("C1.11 sum(hist) + Nskip = N_mask exactly", r["C1.11_hist_sum_max_abs_diff"] == 0.0,
            r["C1.11_hist_sum_max_abs_diff"], 0.0)
    c.metrics["C1.11_hist_vs_direct_energy_max_rel"] = r["C1.11_hist_vs_direct_energy_max_rel"]
    mm = r["C1.11c_mask_mismatch_cells_and_max_dM"]
    c.check("C1.11 mask equals the mask from Entity's N outputs", len(mm) > 0 and all(x[1] == 0 for x in mm),
            mm, 0)
    ml = r["C1.15_midline_vs_fields_max_abs_diff"]
    c.check("C1.15 midline rows equal the field output", len(ml) > 0 and all(x[1] == 0.0 for x in ml), ml, 0.0)
    _cv_checks(c, sim, "filters4")
    # C1.5 second criterion (III.4): layer profiles against the targets; see
    # check_boundaries.layers for the operationalization (proposal)
    import check_boundaries
    lay = check_boundaries.layers(sim)
    c.metrics["C1.5_layers"] = lay
    pre = [o["x2_layers_max_dev"] for o in lay["outputs"] if o["quiescent"]]
    c.check("C1.5 x2 (inflow) layers within 2% of B0 of the targets, quiescent upstream (causal cut)",
            len(pre) > 0 and max(pre) <= TOL["C1.5_layer"], max(pre) if pre else None, TOL["C1.5_layer"])
    c.metrics["partition_last"] = r.get("partition_last")
    c.metrics["fill_last"] = readers.read_all_recmix(os.path.join(sim, "usr"))[1][-1].fill
    return c


def case_recon_f8(a, out):
    """C1.14 at the production filter count (8 passes; padding 12 = nfilter + 4)."""
    c = Case("recon_f8", out)
    sim = _recon(c, a, "r8", ["--filters", "8", "--padding", "12"])
    _cv_checks(c, sim, "filters8")
    return c


def case_filter_sensitivity(a, out):
    """C1.14 sensitivity (Codex A3): 4 against 8 current-filter passes at the same
    padding (12), two seeds each. Reports r_C, the CV work fraction f_i^CV over
    labels 251 to 601 (after t_open) and the masked f_i (mean of the last four
    outputs); the filter effect is compared with the seed spread. Informational:
    no tolerance is registered for it (proposal in tech spec V.2)."""
    c = Case("filter_sensitivity", out)
    rows = []
    for nf in (4, 8):
        for seed in (31, 32):
            name = f"f{nf}s{seed}"
            t = make_input(c, name, ["--steps", "601", "--usr-interval", "50", "--line-interval", "300",
                                     "--field-interval", "100000", "--filters", str(nf), "--padding", "12",
                                     "--seed", str(seed)])
            _, lg = run(c, a.recmix, t, threads=2)
            check_pusher(c, lg)
            usr = os.path.join(c.dir, name, "usr")
            meta, outs = readers.read_all_recmix(usr)
            cv = readers.read_cvbudget(usr, meta=meta)
            cl = cvbudget.closure(cv, meta=meta)
            fcv = cvbudget.work_fraction(cv, partition.ACTIVE["ions"], partition.ACTIVE["leptons"], (251, 601))
            fi = float(np.mean([partition.active_partition(meta, o)["f_i"] for o in outs[-4:]]))
            rows.append({"filters": nf, "seed": seed, "r_F_rel": cl["r_F_rel"], "r_C_rel": cl["r_C_rel"],
                         "r_total_rel": cl["rel_residual"], "f_i_CV": fcv, "f_i_mask": fi})
    c.metrics["runs_table"] = rows
    for key in ("f_i_CV", "f_i_mask", "r_C_rel"):
        m4 = np.mean([r[key] for r in rows if r["filters"] == 4])
        m8 = np.mean([r[key] for r in rows if r["filters"] == 8])
        spread = np.mean([abs(rows[0][key] - rows[1][key]), abs(rows[2][key] - rows[3][key])])
        c.metrics[f"{key}_filter_effect"] = {"mean_4": m4, "mean_8": m8, "diff": m8 - m4,
                                             "mean_seed_spread": spread}
    return c


def case_cv_interval(a, out):
    """C1.14: cv_interval 5 against 1 on identical dynamics (one thread). V.2
    allows intervals > 1 only if the species work changes by < 1% (reported)."""
    c = Case("cv_interval", out)
    s1 = _recon(c, a, "cv1", threads=1)
    s5 = _recon(c, a, "cv5", ["--cv-interval", "5"], threads=1)
    meta = readers.read_meta(os.path.join(s1, "usr"))
    w1 = cvbudget.closure(readers.read_cvbudget(os.path.join(s1, "usr")), meta=meta)["W_by_species"]
    w5 = cvbudget.closure(readers.read_cvbudget(os.path.join(s5, "usr")), meta=meta)["W_by_species"]
    rel = [float(x / y - 1) for x, y in zip(w5, w1)]
    c.metrics["W_rel_change_by_species"] = rel
    c.metrics["cv_interval_5_allowed"] = bool(max(abs(x) for x in rel) < TOL["C1.14_cv_interval"])
    _, o1 = readers.read_all_recmix(os.path.join(s1, "usr"))
    _, o5 = readers.read_all_recmix(os.path.join(s5, "usr"))
    c.check("cv_interval does not change the dynamics (one thread)",
            len(o1) == len(o5) and all(np.array_equal(x.m("Ntot"), y.m("Ntot")) and
                                       np.array_equal(x.m("Ktot"), y.m("Ktot")) for x, y in zip(o1, o5)))
    return c


def case_recon_s4(a, out):
    """C1.16: in-code bulk energies against the stock V_s / V_all outputs (5 x 5
    top-hat smoothing) and the raw count_s outputs."""
    c = Case("recon_s4", out)
    cnt = ", ".join(f'"count_{s}"' for s in range(1, 10))
    sim = _recon(c, a, "r4", edits=[('custom       = ["mixing", "mask"]',
                                     f'custom       = ["mixing", "mask", {cnt}]')])
    import check_bulk
    res = check_bulk.main(sim)
    worst = max(abs(v) for row in res for v in row["own"] + row["all"] if v is not None)
    c.metrics["C1.16"] = res
    c.check("C1.16 Kb_own_raw and Kb_all from the stock V outputs", worst <= TOL["C1.16"], worst, TOL["C1.16"])
    c.check("C1.16 masked count_s sums equal N_s", all(r["count_vs_N_maxdiff"] == 0 for r in res))
    return c


def _particles_case(a, out, name, q):
    """C1.16b and C1.14b against the exact particle state (the generator's
    validation dump, A0-2), plus the A0-2 witness on real output: how many of
    Entity's rounded particle positions are ambiguous and how many of those the
    floor convention misassigns."""
    c = Case(name, out)
    t = make_input(c, "pp", ["--q", str(q), "--steps", "301", "--usr-interval", "100",
                             "--line-interval", "300", "--field-interval", "100", "--field-stride", "1",
                             "--seed", "41", "--validation-dump"],
                   [("  [output.particles]\n    enable = false",
                     "  [output.particles]\n    enable   = true\n    interval = 100\n    stride   = 1")])
    _, lg = run(c, a.recmix, t, threads=2)
    check_pusher(c, lg)
    sim = os.path.join(c.dir, "pp")
    import check_particles
    r = check_particles.main(sim, 301)
    c.check("A0-2 exact particle source (validation dump) used", r["exact_source"], r["source"])
    worst, low_ok, cv_ok = 0.0, True, True
    for s, e in r["species"].items():
        for k, v in e["rel_diff"].items():
            if k.startswith("K"):  # Kb_own(_raw, _w1, _half), Klowocc, Kblowocc (both windows)
                worst = max(worst, v)
        low_ok &= all(e["ref"][k] == e["code"][k] for k in e["ref"] if k.startswith("Nlowocc"))
        cv_ok &= (e["Ncv_particles"] == e["Ncv_record"] and
                  abs(e["Icv_particles"] - e["Icv_record"]) <= 1e-9 * max(1.0, abs(e["Icv_record"])))
    c.metrics["C1.16b"] = r
    c.check("C1.16b bulk-energy moments (Kb_own, raw, w1, half; Klowocc, Kblowocc) from the particle output",
            worst <= TOL["C1.16b"], worst, TOL["C1.16b"])
    c.check("C1.16b Nlowocc (both windows) from the exact particle state, exact",
            low_ok and r["exact_source"])
    c.check("C1.14b CV inventory N_cv, I_cv from the exact particle state (CV flags)", cv_ok)
    amb = check_particles.ambiguity_report(sim, 301)
    c.metrics["A0-2_rounded_position_report"] = amb
    c.check("A0-2 Entity's particle output equals the dump through Entity's position conversion",
            all(v.get("bp_equals_converted_dump", False) for v in amb.values()),
            {s: v.get("bp_equals_converted_dump") for s, v in amb.items()})
    c.metrics["window_scan"] = check_particles.window_scan(sim, 301)
    c.metrics["position_subsampling_scan"] = check_particles.position_subsampling_scan(sim, 301)
    c.metrics["occupancy_scan"] = check_particles.occupancy_scan(sim, 301)
    meta, outs = readers.read_all_recmix(os.path.join(sim, "usr"))
    c.metrics["partition_last"] = jsonable(partition.active_partition(meta, outs[-1]))
    return c


def case_particles(a, out):
    c = _particles_case(a, out, "particles", 0.5)
    p = c.metrics["partition_last"]
    # code behaviour: the status follows the rule from the reported fractions;
    # the classification of each population is a measured finding, not a check
    for pop in ("e", "i"):
        nf, ef = p[f"{pop}_low_occupancy_fraction"], p[f"{pop}_low_occupancy_energy_bound"]
        expect = "occupancy_ok" if (nf <= partition.MAX_LOWOCC and ef <= partition.MAX_LOWOCC) \
            else "low_occupancy"
        c.check(f"RV1 occupancy screen applies its rule ({pop})", p[f"epst_{pop}_status"] == expect,
                [p[f"epst_{pop}_status"], nf, ef], partition.MAX_LOWOCC)
    c.metrics["classification"] = {pop: p[f"epst_{pop}_status"] for pop in ("e", "i")}
    c.metrics["position_subsampling_change_in_code"] = {
        k: p.get(f"epst_{k}_position_subsampling_change") for k in ("e", "i")}
    return c


def case_particles_lowq(a, out):
    c = _particles_case(a, out, "particles_lowq", 0.01)
    p = c.metrics["partition_last"]
    c.check("IC5 sparse ions (q = 0.01): eps-tilde fails the occupancy screen (NaN)",
            p["epst_i_status"] == "low_occupancy" and p["epst_i_own"] is None,
            [p["epst_i_status"], p["i_low_occupancy_fraction"], p.get("i_low_occupancy_energy_bound")])
    c.metrics["position_subsampling_change_in_code"] = {
        k: p.get(f"epst_{k}_position_subsampling_change") for k in ("e", "i")}
    return c


def case_capacity(a, out):
    """IC2: the capacity guard fails cleanly at load (capacity 1) and later in the
    injector (species 4 sized 100 slots above its use after the first step)."""
    c = Case("capacity", out)
    t = make_input(c, "cap0", ["--steps", "11", "--maxnpart", "1"])
    rc, lg = run(c, a.recmix, t, expect_fail=True)
    m = re.search(r"capacity guard: species (\d+) needs (\d+) > maxnpart (\d+)", lg)
    c.check("IC2 guard at load: clean failure", rc != 0 and m is not None, m.group(0) if m else rc)
    t = make_input(c, "cap1", ["--steps", "2", "--usr-interval", "1", "--seed", "61"])
    run(c, a.recmix, t)
    meta, outs = readers.read_all_recmix(os.path.join(c.dir, "cap1", "usr"))
    slots = int(round(outs[0].fill[3] * meta["species"][3]["maxnpart"]))
    blocks = open(t).read().split("[[particles.species]]")
    blocks[4] = re.sub(r"maxnpart       = [0-9.e+]+", f"maxnpart       = {float(slots + 100)!r}", blocks[4])
    s = "[[particles.species]]".join(blocks).replace('name    = "cap1"', 'name    = "cap2"')
    s = re.sub(r"runtime = [0-9.e+]+", f"runtime = {301 * DT:.6f}", s)
    t2 = os.path.join(c.dir, "cap2.toml")
    open(t2, "w").write(s)
    rc, lg = run(c, a.recmix, t2, expect_fail=True)
    m = re.search(r"capacity guard: species 4 needs (\d+) > maxnpart (\d+)", lg)
    c.metrics["species4_slots_after_step_0"] = slots
    c.check("IC2 guard in the injector: clean failure after the load", rc != 0 and m is not None,
            m.group(0) if m else rc)
    return c


def case_mpi(a, out):
    """C1.6: 4 ranks [2, 2] across t_open: no NaN, complete midline, CV closure
    (rec_mix outputs only; the MPI build may have output = OFF)."""
    c = Case("mpi", out)
    t = make_input(c, "mp", ["--steps", "601", "--usr-interval", "50", "--line-interval", "50",
                             "--field-interval", "100000", "--decomposition", "2", "2", "--seed", "51"])
    _, lg = run(c, a.recmix_mpi, t, mpirun=a.mpirun)
    check_pusher(c, lg)
    c.check("C1.6 outflow walls opened at t_open", "outflow boundaries opened" in lg)
    usr = os.path.join(c.dir, "mp", "usr")
    meta, outs = readers.read_all_recmix(usr)
    c.check("C1.6 user moments finite", all(np.all(np.isfinite(o.moms)) for o in outs), len(outs))
    ml = [readers.read_midline(p) for p in readers.list_midlines(usr)]
    c.check("C1.6 midline gathered from all ranks",
            len(ml) > 0 and all(np.all(m.rows[:, 3, :] != 0) for m in ml), len(ml))
    _cv_checks(c, os.path.join(c.dir, "mp"), "mpi_2x2")
    return c


# --------------------------------------------------------------------------
# P13-D39 normalization (generator level)
# --------------------------------------------------------------------------
# Fixed anchors, not computed by the converter under test: the saved p1.1 cfg
# inputs (sigma_input, c_omp), Codex's independently derived nominal values and
# literal/ideal larmor0 (normalization review 2026-10-12), and the C2-b sheet-ion
# temperature quoted there.
D39_ANCHORS = {
    "C2-a": {"sigma_input": 89.0, "c_omp": 4.84, "c_omp_nominal": 4.842341981, "larmor0": 0.513038974,
             "larmor0_ideal": 0.513287223},
    "C2-b": {"sigma_input": 42.98, "c_omp": 4.96, "c_omp_nominal": 4.961688196, "larmor0": 0.756568876,
             "larmor0_ideal": 0.756826383},
    "C2-c": {"sigma_input": 521.0, "c_omp": 3.58, "c_omp_nominal": 3.578810106, "larmor0": 0.156842648,
             "larmor0_ideal": 0.156790517},
    "C2-d": {"sigma_input": 164.0, "c_omp": 4.96, "c_omp_nominal": 4.962916670, "larmor0": 0.387310929,
             "larmor0_ideal": 0.387538683},
}
NORM_GRID = ["--n1", "64", "--n2", "128", "--cs-width", "40", "--ppc0", "8", "--steps", "2",
             "--padding", "12", "--filters", "8", "--ds-abs", "10", "--decomposition", "1", "1",
             "--usr-interval", "1", "--line-interval", "1", "--field-interval", "100000",
             "--checkpoint-interval", "100000", "--seed", "81"]


def _mk(c, name, args, subdir=None):
    """make_input without BASE (the normalization case sets everything)."""
    d = os.path.join(c.dir, subdir) if subdir else c.dir
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.toml")
    cmd = [sys.executable, MAKE_INPUT, "--name", name, "--out", path] + list(args)
    p = subprocess.run(cmd, capture_output=True, text=True)
    c.metrics.setdefault("make_input", []).append(
        {"args": " ".join(cmd[2:]), "exit": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()})
    return path, p


def case_normalization(a, out):
    """P13-D39: the generator with the literal p1.1 inputs of C2-a..d (tiny grid,
    2 steps) reproduces the fixed anchors; one nominal (ideal) run; the old cold
    interpretation is refused in every form."""
    c = Case("normalization", out)
    rows = {}
    for ref, A in D39_ANCHORS.items():
        t, p = _mk(c, ref.replace("-", "").lower(), ["--reference", ref] + NORM_GRID)
        _, lg = run(c, a.recmix, t, threads=2)
        meta = readers.read_meta(os.path.join(os.path.dirname(t), os.path.basename(t)[:-5], "usr"))
        n, dv = meta["normalization"], meta["derived"]
        r = {"mode": n["mode"], "label": n["label"], "sigma0": dv["sigma0"], "skindepth0": dv["skindepth0"],
             "larmor0": dv["larmor0"], "sigma_input_nominal": n["sigma_input_nominal"],
             "c_omp_nominal": n["c_omp_nominal"], "T_cs": dv["T_cs"], "beta_d": dv["beta_d"]}
        tol = TOL["D39_scales_rel"]
        c.check(f"D39 {ref}: literal inputs used (sigma0, skindepth0 = saved cfg values)",
                n["mode"] == "literal" and abs(dv["sigma0"] / A["sigma_input"] - 1) <= tol and
                abs(dv["skindepth0"] / A["c_omp"] - 1) <= tol, [dv["sigma0"], dv["skindepth0"]], tol)
        c.check(f"D39 {ref}: larmor0 equals the literal anchor",
                abs(dv["larmor0"] / A["larmor0"] - 1) <= tol, dv["larmor0"], A["larmor0"])
        c.check(f"D39 {ref}: nominal sigma_input and c_omp equal the independent anchors",
                abs(n["sigma_input_nominal"] / A["sigma_input"] - 1) <= tol and
                abs(n["c_omp_nominal"] / A["c_omp_nominal"] - 1) <= tol,
                [n["sigma_input_nominal"], n["c_omp_nominal"]], tol)
        if ref == "C2-b":
            th = dv["T_cs"] / meta["inputs"]["mu"]
            r["theta_i_cs"] = th
            # closed form from the saved inputs alone: 42.98/(2 eta (1 - q/2 + q mu/2) mu)
            th_cf = 42.98 / (2 * 3 * (1 - 0.0005 + 0.05) * 100)
            c.check("D39 C2-b: legacy sheet-ion theta = .06825 (Codex, 4 digits; Tristan Gaussian branch, "
                    "<= 0.1) and the closed form from the saved inputs (1e-6)",
                    abs(th - 0.06825) <= TOL["D39_theta_cs_b"] and th <= 0.1 and abs(th / th_cf - 1) <= 1e-6,
                    [th, th_cf], TOL["D39_theta_cs_b"])
        rows[ref] = r
    c.metrics["references"] = rows
    # ideal nominal mode (C2-a labels, unrounded)
    t, _ = _mk(c, "c2a_ideal", ["--layout", "active", "--sigma-hot", "40", "--q", "0.1", "--mu", "25"] + NORM_GRID)
    run(c, a.recmix, t, threads=2)
    m = readers.read_meta(os.path.join(c.dir, "c2a_ideal", "usr"))
    c.check("D39 nominal mode: larmor0 equals the ideal five-cell anchor (C2-a)",
            m["normalization"]["mode"] == "nominal" and
            abs(m["derived"]["larmor0"] / D39_ANCHORS["C2-a"]["larmor0_ideal"] - 1) <= TOL["D39_scales_rel"],
            m["derived"]["larmor0"], D39_ANCHORS["C2-a"]["larmor0_ideal"])
    # refusals of the superseded cold interpretation
    t, _ = _mk(c, "cold_scales", ["--reference", "C2-a"] + NORM_GRID)
    cold = 5 * math.sqrt((2 - 0.1 + 0.1 / 25) / 2)  # pre-D39 skindepth0, sigma0 = 88
    edit(t, [(r"larmor0    = [0-9.e+-]+", f"larmor0    = {cold / math.sqrt(88)!r}"),
             (r"skindepth0 = [0-9.e+-]+", f"skindepth0 = {cold!r}")])
    rc, lg = run(c, a.recmix, t, expect_fail=True)
    c.check("D39 refusal: cold-mapping [scales] with hot labels", rc != 0 and "[scales] disagree" in lg, rc)
    t, _ = _mk(c, "cold_literal", ["--reference", "C2-a"] + NORM_GRID)
    edit(t, [(r"sigma_input      = [0-9.e+-]+", "sigma_input      = 88.0")])
    rc, lg = run(c, a.recmix, t, expect_fail=True)
    c.check("D39 refusal: literal sigma_input from the cold mapping (88)",
            rc != 0 and "disagree with the hot labels" in lg, rc)
    t, _ = _mk(c, "old_key", ["--reference", "C2-a"] + NORM_GRID)
    edit(t, [(r"sigma_hot        = [0-9.e+-]+", "sigma            = 40.0")])
    rc, lg = run(c, a.recmix, t, expect_fail=True)
    c.check("D39 refusal: the pre-D39 key setup.sigma", rc != 0 and "P13-D39" in lg, rc)
    _, p = _mk(c, "old_cli", ["--sigma", "40", "--q", "0.1", "--mu", "25"] + NORM_GRID)
    c.check("D39 refusal: make_input --sigma", p.returncode != 0 and "P13-D39" in p.stderr, p.returncode)
    return c


# --------------------------------------------------------------------------
# C1.5 boundaries
# --------------------------------------------------------------------------
PULSE = ["--padding", "12", "--filters", "8", "--ds-abs", "10", "--steps", "1001", "--usr-interval", "1000",
         "--line-interval", "1000", "--field-interval", "1000", "--field-stride", "1",
         "--pulse-offset", "400", "--seed", "91"]


def case_reflection(a, out):
    """C1.5 (III.4), first criterion: a vacuum pulse reflects < 5% of its energy
    at each wall type, at production layer widths (inflow w_in = 1.5 (12 - 8) =
    6 cells, outflow w_out = 2 ds_abs = 20 cells); Gaussian widths 2.5, 5, 10,
    20 cells (d_up = 5 in production); every width judged. Control: the same
    x2 test with CONDUCTOR walls must reflect (R > 0.9), showing that the
    metric sees a reflection."""
    c = Case("reflection", out)
    import check_boundaries
    rows = []
    for wall, ft, grid in (("x2 (inflow)", "pulse_x2", ["--n1", "8", "--n2", "1280"]),
                           ("x1 (outflow)", "pulse_x1", ["--n1", "1280", "--n2", "8"])):
        for w in (2.5, 5.0, 10.0, 20.0):
            name = f"{ft}_w{w:g}".replace(".", "p")
            t = make_input(c, name, grid + ["--field-test", ft, "--pulse-width", str(w)] + PULSE)
            run(c, a.recmix, t, threads=2)
            r = check_boundaries.reflection(os.path.join(c.dir, name))
            r["wall"] = wall
            rows.append(r)
            for side, key in (("low", "R_low"), ("high", "R_high")):
                c.check(f"C1.5 reflection < 5%: {wall} wall ({side}), pulse width {w:g} cells",
                        r[key] < TOL["C1.5_reflection"], r[key], TOL["C1.5_reflection"])
    t = make_input(c, "control_conductor", ["--n1", "8", "--n2", "1280", "--field-test", "pulse_x2",
                                            "--pulse-width", "5", "--x2-field-bc", "CONDUCTOR"] + PULSE)
    run(c, a.recmix, t, threads=2)
    r = check_boundaries.reflection(os.path.join(c.dir, "control_conductor"))
    c.metrics["control_conductor"] = r
    c.check("C1.5 control: CONDUCTOR walls reflect (R > 0.9), the metric sees reflection",
            min(r["R_low"], r["R_high"]) > 0.9, [r["R_low"], r["R_high"]], 0.9)
    c.metrics["pulses"] = rows
    return c


def case_layers(a, out):
    """C1.5 (III.4), second criterion, on a tall box (N2 = 512) whose x2 layers
    stay causally out of reach of the sheet for the whole run (t < (y0 - a -
    w_in)/c = 242 > 501 dt = 225): every field output is a steady-upstream
    state with the injected plasma, the injector and the padding; all six
    components in every x2 layer cell within 2% of B0 of the targets (initial
    B, E = 0). Production widths (padding 12, 8 filters, w_in = 6). The x1
    layers need a steady reconnected inflow (Rusty)."""
    c = Case("layers", out)
    t = make_input(c, "ly", ["--n2", "512", "--steps", "501", "--usr-interval", "500", "--line-interval", "500",
                             "--field-interval", "50", "--field-stride", "1", "--padding", "12", "--filters", "8",
                             "--ds-abs", "10", "--t-open-tlc", "100", "--seed", "92"])
    _, lg = run(c, a.recmix, t, threads=2)
    check_pusher(c, lg)
    import check_boundaries
    lay = check_boundaries.layers(os.path.join(c.dir, "ly"))
    c.metrics["C1.5_layers"] = lay
    q = [o["x2_layers_max_dev"] for o in lay["outputs"] if o["quiescent"]]
    c.check("C1.5 all outputs causally quiescent", len(q) == len(lay["outputs"]) and len(q) > 1, len(q))
    c.check("C1.5 x2 (inflow) layers within 2% of B0 of the targets in steady upstream conditions",
            len(q) > 0 and max(q) <= TOL["C1.5_layer"], max(q) if q else None, TOL["C1.5_layer"])
    return c


# --------------------------------------------------------------------------
# C1.12 tracking (sorting, MPI exchange, restart)
# --------------------------------------------------------------------------
def case_tracking(a, out):
    """C1.12: tracking IDs unique and stable through sorting (clear_interval 10,
    spatial_sorting_interval 20), MPI exchange (4 ranks, [2, 2]) and a restart
    from label 151 (before t_open at step 214, so B crosses t_open). Exact
    state from the validation dumps; metrics in check_tracking.py."""
    c = Case("tracking", out)
    common = ["--steps", "301", "--usr-interval", "50", "--line-interval", "300", "--field-interval", "100000",
              "--decomposition", "2", "2", "--tracking", "--validation-dump", "--clear-interval", "10",
              "--sort-interval", "20", "--checkpoint-interval", "150", "--seed", "52"]
    tA = make_input(c, "tk", common, [("  keep     = 2", "  keep     = 10")], subdir="A")
    _, lg = run(c, a.recmix_mpio, tA, mpirun=a.mpirun)
    check_pusher(c, lg)
    os.makedirs(os.path.join(c.dir, "B"))
    tB = os.path.join(c.dir, "B", "tk.toml")
    shutil.copy(tA, tB)
    copy_checkpoint(os.path.join(c.dir, "A", "tk.ckpt"), os.path.join(c.dir, "B", "tk.ckpt"), 151)
    _, lb = run(c, a.recmix_mpio, tB, mpirun=a.mpirun, extra=("-continue",))
    c.check("C1.12 B crosses t_open after the restart", "outflow boundaries opened" in lb)
    import check_tracking
    LA = [1, 51, 101, 151, 201, 251, 301]
    LB = [201, 251, 301]
    r = check_tracking.main(os.path.join(c.dir, "A", "tk"), os.path.join(c.dir, "B", "tk"), 151, LA, LB,
                            192, DT)
    c.metrics["C1.12"] = r
    dup = sum(v["duplicates"] for run_ in ("A", "B") for L in r[run_].values() for v in L.values() if v)
    c.check("C1.12 IDs unique (every label, species, both runs)", dup == 0, dup, 0)
    viol = sum(v["violations"] for key in ("stable_A", "stable_B") for pr in r[key].values()
               for v in pr.values())
    common_n = sum(v["common"] for key in ("stable_A", "stable_B") for pr in r[key].values() for v in pr.values())
    c.check("C1.12 IDs stable between labels (displacement <= c dt), through sorting and exchange",
            viol == 0 and common_n > 0, {"violations": viol, "pairs": common_n}, 0)
    mig = sum(v["migrated"] for L in r["A"].values() for v in L.values() if v)
    c.check("C1.12 MPI exchange exercised (particles away from their creating domain)", mig > 0, mig)
    rs = r["restart"]
    sp = [k for k in rs if k != "displacement_L0_to_B" and rs[k]]
    reused = sum(rs[s]["reused_old_ids"] for s in sp)
    below = sum(rs[s]["new_ids_not_above_previous_max"] for s in sp)
    new = sum(rs[s]["created_after_restart"] for s in sp)
    c.check("C1.12 restart: no ID reused, new IDs above every earlier ID (counter restored)",
            reused == 0 and below == 0 and new > 0, {"reused": reused, "not_above": below, "new": new}, 0)
    rv = sum(v["violations"] for v in rs["displacement_L0_to_B"].values())
    rc_ = sum(v["common"] for v in rs["displacement_L0_to_B"].values())
    c.check("C1.12 restart: surviving IDs stable from A at 151 to B at 201", rv == 0 and rc_ > 0,
            {"violations": rv, "pairs": rc_}, 0)
    return c


# --------------------------------------------------------------------------
# C1.7 statistical restart with injection
# --------------------------------------------------------------------------
def _series(sim):
    """f_i (masked, direct), total particle number and whole-box field energy
    (Entity stats, sum of B^2 + E^2 means) per user-output label."""
    usr = os.path.join(sim, "usr")
    meta, outs = readers.read_all_recmix(usr)
    fi = {o.step: partition.active_partition(meta, o)["f_i"] for o in outs}
    npart = {o.step: float(np.sum(o.m("Ntot"))) for o in outs}
    st = np.genfromtxt(os.path.join(sim, os.path.basename(sim) + "_stats.csv"), delimiter=",", names=True)
    U = {int(r["step"]): float(sum(r[k] for k in ("B12", "B22", "B32", "E12", "E22", "E32"))) for r in st}
    return {"f_i": fi, "Npart": npart, "U_field": U}


def case_restart_stat(a, out):
    """C1.7, statistical part (D14): with injection, f_i(t), N_part(t) and the
    field energy after a restart lie within the run-to-run envelope of the
    uninterrupted run, across t_open. Run A (seed 101, checkpoint at label 151)
    and B (resumed from it; t_open at step 214 lies after the restart); the
    envelope is K = 8 uninterrupted runs with other seeds (102..109), so that B
    is exchangeable with them under the null.
    Operationalization (proposal, Claude 2026-10-12; registered criterion text
    in tech spec V.2): z = (X_B - mean_K)/(s_K sqrt(1 + 1/K)) at the labels
    201..601 (M = 9) for the 3 quantities; under the null (exchangeable,
    approximately Gaussian) z ~ t_{K-1}; pass if max |z| <= t_{K-1} quantile at
    1 - alpha/(2 M 3), alpha = 0.01 (Bonferroni). Reported, not judged: the
    min-max envelope fraction, A against the same envelope (positive control),
    a run without opening (seed 101, t_open never; negative control, the
    failure mode of a restart that loses the opening) and the A-B difference
    (statistical, not bitwise: the device RNG is not restored)."""
    from scipy import stats
    c = Case("restart_stat", out)
    common = ["--steps", "601", "--usr-interval", "50", "--line-interval", "600", "--field-interval", "100000",
              "--checkpoint-interval", "150"]
    stats50 = ("    interval = 100", "    interval = 50")
    tA = make_input(c, "rs", common + ["--seed", "101"], [stats50, ("  keep     = 2", "  keep     = 10")],
                    subdir="A")
    run(c, a.recmix, tA, threads=2)
    os.makedirs(os.path.join(c.dir, "B"))
    tB = os.path.join(c.dir, "B", "rs.toml")
    shutil.copy(tA, tB)
    copy_checkpoint(os.path.join(c.dir, "A", "rs.ckpt"), os.path.join(c.dir, "B", "rs.ckpt"), 151)
    _, lb = run(c, a.recmix, tB, threads=2, extra=("-continue",))
    c.check("C1.7 B opens the outflow walls after the restart (restart across t_open)",
            "outflow boundaries opened" in lb)
    ens = []
    for sd in range(102, 110):
        t = make_input(c, "rs", common + ["--seed", str(sd)], [stats50], subdir=f"s{sd}")
        run(c, a.recmix, t, threads=2)
        ens.append(_series(os.path.join(c.dir, f"s{sd}", "rs")))
    tN = make_input(c, "rs", common + ["--seed", "101", "--t-open-tlc", "100"], [stats50], subdir="noopen")
    run(c, a.recmix, tN, threads=2)
    SA, SB, SN = (_series(os.path.join(c.dir, d, "rs")) for d in ("A", "B", "noopen"))
    labels = list(range(201, 602, 50))
    K, M, Q = len(ens), len(labels), 3
    thr = float(stats.t.ppf(1 - TOL["C1.7_alpha"] / (2 * M * Q), K - 1))

    def zmax(S):
        zs, inside, rows = [], 0, {}
        for q in ("f_i", "Npart", "U_field"):
            for L in labels:
                e = np.array([x[q][L] for x in ens])
                m, sd = e.mean(), e.std(ddof=1)
                if not (np.all(np.isfinite(e)) and np.isfinite(S[q][L])):
                    z = np.inf                       # an undefined value fails, never skipped
                elif sd > 0:
                    z = (S[q][L] - m) / (sd * math.sqrt(1 + 1 / K))
                else:
                    z = 0.0 if S[q][L] == m else np.inf
                zs.append(abs(z))
                inside += int(e.min() <= S[q][L] <= e.max())
                rows.setdefault(q, []).append({"label": L, "value": S[q][L], "mean": m, "sd": sd, "z": z})
        return max(zs), inside / len(zs), rows

    zB, fB, rowsB = zmax(SB)
    zA, fA, _ = zmax(SA)
    zN, fN, _ = zmax(SN)
    c.metrics.update({"threshold": thr, "K": K, "M": M, "B": {"max_abs_z": zB, "minmax_fraction": fB},
                      "A_positive_control": {"max_abs_z": zA, "minmax_fraction": fA},
                      "noopen_negative_control": {"max_abs_z": zN, "minmax_fraction": fN},
                      "B_series": rowsB,
                      "A_minus_B": {q: {L: SA[q][L] - SB[q][L] for L in labels} for q in SA}})
    c.check("C1.7 statistical: restarted run within the run-to-run envelope (max |z| <= t threshold)",
            zB <= thr, zB, thr)
    c.check("C1.7 positive control: the uninterrupted seed-101 run passes the same test", zA <= thr, zA, thr)
    c.check("C1.7 power: a run that never opens the outflow walls fails it", zN > thr, zN, thr)
    c.metrics["bitwise_A_equals_B"] = all(SA[q][L] == SB[q][L] for q in SA for L in labels)
    return c


# --------------------------------------------------------------------------
# C1.14 closed box (numerical energy drift)
# --------------------------------------------------------------------------
def case_closed_box(a, out):
    """C1.14 closed-box branch (V.2, D35): measure the scheme's energy drift in
    the doubly periodic double-sheet layout at the numerics of recon_f8 (8 filter
    passes, padding 12, ppc0 16, sigma_hot 4, T 0.01, d_up 4 cells, CFL, Vay,
    cv_interval 1) and the same duration (601 steps, whole run from label 1),
    next to its open twin (= recon_f8, same seed). Energy normalization: the
    drift Delta(sum_s K_s + U_EM) of the closed box over the whole run, in
    m_e c^2, divided by |sum_dep W_s| of the open twin over the same steps (the
    registered normalization); the particle numbers differ (two sheets, no
    padding) and are reported, as is a per-particle variant (proposal, not
    adopted). Cross-check: the drift from Entity's stats (T00 - Rho and the
    field sums, an independent code path). The registered combined branch
    max(2%, 2 x drift) is evaluated for this small box as information only: no
    drift-derived tolerance is adopted before the production measurement, and
    no long-run drift is inferred from this 601-step box."""
    c = Case("closed_box", out)
    numer = ["--steps", "601", "--usr-interval", "50", "--line-interval", "600", "--field-interval", "100000",
             "--filters", "8", "--padding", "12", "--seed", "31"]
    stats50 = ("    interval = 100", "    interval = 50")
    tO = make_input(c, "op", numer, [stats50], subdir="open")
    _, lo = run(c, a.recmix, tO, threads=2)
    check_pusher(c, lo)
    tC = make_input(c, "cb", numer + ["--geometry", "closed_box"], [stats50], subdir="closed")
    _, lc = run(c, a.recmix, tC, threads=2)
    check_pusher(c, lc)
    simO, simC = os.path.join(c.dir, "open", "op"), os.path.join(c.dir, "closed", "cb")
    mO, mC = readers.read_meta(os.path.join(simO, "usr")), readers.read_meta(os.path.join(simC, "usr"))
    cvO = readers.read_cvbudget(os.path.join(simO, "usr"), meta=mO)
    cvC = readers.read_cvbudget(os.path.join(simC, "usr"), meta=mC)
    clO, clC = cvbudget.closure(cvO, meta=mO), cvbudget.closure(cvC, meta=mC)
    dep = [k for k, s in enumerate(mC["species"]) if s["deposit"]]
    E = cvC["Icv"][:, dep].sum(1) + cvC["U_EM"]
    drift = E - E[0]
    WO = clO["W_total"]
    NC = cvC["Ncv"][:, dep].sum(1)
    _, outsO = readers.read_all_recmix(os.path.join(simO, "usr"))
    NO = np.array([float(np.sum(o.m("Ntot"))) for o in outsO])
    # independent cross-check from Entity's stats: K = (T00 - Rho) V ppc0, U = sigma0/2 ppc0 V sum(F^2)
    st = np.genfromtxt(os.path.join(simC, "cb_stats.csv"), delimiter=",", names=True)
    V, ppc0, s0 = mC["derived"]["N1"] * mC["derived"]["N2"], mC["derived"]["ppc0"], mC["derived"]["sigma0"]
    Ks = (st["T00"] - st["Rho"]) * V * ppc0
    Us = 0.5 * s0 * ppc0 * V * sum(st[k] for k in ("B12", "B22", "B32", "E12", "E22", "E32"))
    labs = [int(x) for x in cvC["step"]]
    k_st = [list(st["step"].astype(int)).index(L) for L in labs]
    Es = Ks[k_st] + Us[k_st]
    rel_K = float(np.max(np.abs(Ks[k_st] - cvC["Icv"][:, dep].sum(1)) / cvC["Icv"][:, dep].sum(1)))
    rel_U = float(np.max(np.abs(Us[k_st] - cvC["U_EM"]) / cvC["U_EM"]))
    c.metrics.update({
        "duration_steps": 600, "labels": labs, "normalization": "|sum_dep W_s| of the open twin, whole run",
        "closed_drift_abs_by_label": drift.tolist(), "closed_drift_abs_final": float(drift[-1]),
        "closed_drift_abs_max_over_run": float(np.max(np.abs(drift))),
        "closed_drift_max_over_open_W": float(np.max(np.abs(drift)) / abs(WO)),
        "closed_drift_stats_by_label": (Es - Es[0]).tolist(),
        "stats_vs_cv": {"K_max_rel": rel_K, "U_max_rel": rel_U},
        "closed_E_total_initial": float(E[0]),
        "closed_residuals": {k: clC[k] for k in ("r_F_rel", "r_C_rel", "rel_residual", "W_total", "Wj_total")},
        "open_residuals": {k: clO[k] for k in ("r_F_rel", "r_C_rel", "rel_residual", "W_total", "Wj_total")},
        "closed_drift_over_open_W": float(abs(drift[-1]) / abs(WO)),
        "particles": {"closed": float(NC.mean()), "open_mean": float(NO.mean()),
                      "ratio_closed_over_open": float(NC.mean() / NO.mean())},
        "per_particle_variant_proposal": float(abs(drift[-1]) / NC.mean() * NO.mean() / abs(WO)),
    })
    branch = max(TOL["C1.14_rtot"], 2 * abs(drift[-1]) / abs(WO))
    branch_max = max(TOL["C1.14_rtot"], 2 * np.max(np.abs(drift)) / abs(WO))
    c.metrics["combined_branch_small_box_information"] = {
        "open_r_total_rel": clO["rel_residual"], "max(2%, 2 drift_end/|W|)": branch,
        "max(2%, 2 max|drift|/|W|)": float(branch_max),
        "open_r_total_within_branch": bool(clO["rel_residual"] <= branch),
        "open_r_total_within_branch_max": bool(clO["rel_residual"] <= branch_max),
        "status": "measurement in the small cloud box; not the gate (production numerics on Rusty)"}
    c.check("C1.14 closed box: no particle leaves or enters (N constant)", float(np.ptp(NC)) == 0.0, float(np.ptp(NC)))
    c.check("C1.14 closed box: CV energy equals Entity's stats energy (independent path; float "
            "stats, 1e-4)", rel_K <= 1e-4 and rel_U <= 1e-4, [rel_K, rel_U], 1e-4)
    return c


CASES = {
    "c04_default": (case_c04_default, ("stock_recon", "fork_recon")),
    "c05_seed": (case_c05_seed, ("fork_recon",)),
    "c04_deposit": (case_c04_deposit, ()),
    "c06_restart": (case_c06_restart, ()),
    "c07_open_restart": (case_c07_open_restart, ()),
    "provenance": (case_provenance, ()),
    "sheet_legacy": (case_sheet_legacy, ()),
    "sheet_balanced": (case_sheet_balanced, ()),
    "neutrality": (case_neutrality, ()),
    "recon_s0": (case_recon_s0, ()),
    "recon_f8": (case_recon_f8, ()),
    "filter_sensitivity": (case_filter_sensitivity, ()),
    "cv_interval": (case_cv_interval, ()),
    "recon_s4": (case_recon_s4, ()),
    "particles": (case_particles, ()),
    "particles_lowq": (case_particles_lowq, ()),
    "capacity": (case_capacity, ()),
    "mpi": (case_mpi, ("recmix_mpi",)),
    "normalization": (case_normalization, ()),
    "reflection": (case_reflection, ()),
    "layers": (case_layers, ()),
    "tracking": (case_tracking, ("recmix_mpio",)),
    "restart_stat": (case_restart_stat, ()),
    "closed_box": (case_closed_box, ()),
}


def describe_env(a):
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True).stdout.strip()
        except OSError:
            return None
    try:
        import adios2
        adios = adios2.__version__
    except Exception:  # noqa: BLE001
        adios = None
    def build_hash(binary):
        f = os.path.join(os.path.dirname(binary or ""), "CMakeFiles", "entity.xc.dir", "flags.make")
        m = re.search(r'ENTITY_GIT_HASH=\\?"([^"\\]+)', open(f).read()) if os.path.exists(f) else None
        return m.group(1) if m else None
    return {"harness_commit": git("rev-parse", "HEAD"),
            "harness_tree_dirty": bool(git("status", "--porcelain", "--", os.path.join(HERE, ".."))),
            "python": platform.python_version(), "numpy": np.__version__, "adios2": adios,
            "host": platform.platform(), "cpus": os.cpu_count(),
            "binaries": {k: {"path": getattr(a, k), "entity_git_hash": build_hash(getattr(a, k))}
                         for k in BINARIES if getattr(a, k)},
            "mpirun": a.mpirun, "date": time.strftime("%Y-%m-%d %H:%M:%S %Z")}


BINARIES = ("recmix", "recmix_mpi", "recmix_mpio", "stock_recon", "fork_recon")


def write_results(path, results):
    """Atomic replacement: a crash never leaves a truncated results.json."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, path)


def preflight(out, names, results):
    """Collisions of the selected cases with the existing campaign (A0-1)."""
    coll = []
    for n in names:
        if os.path.exists(os.path.join(out, n)):
            coll.append({"case": n, "reason": "case directory exists"})
        if n in results.get("cases", {}):
            coll.append({"case": n, "reason": "results.json record exists"})
    return coll


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recmix", required=True)
    ap.add_argument("--recmix-mpi")
    ap.add_argument("--recmix-mpio", help="MPI build with output = ON (tracking case)")
    ap.add_argument("--mpirun", default="mpirun --oversubscribe -n 4")
    ap.add_argument("--stock-recon")
    ap.add_argument("--fork-recon")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--overwrite", action="store_true",
                    help="supersede colliding cases: rename their directories and move their records "
                         "(nothing is deleted)")
    a = ap.parse_args()
    for k in BINARIES:
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    a.out = os.path.abspath(a.out)
    names = list(CASES) if a.cases == "all" else a.cases.split(",")
    unknown = [n for n in names if n not in CASES]
    if unknown or len(set(names)) != len(names):
        print(f"[harness] unknown {unknown} or repeated cases in {names}; nothing written", flush=True)
        sys.exit(2)
    path = os.path.join(a.out, "results.json")
    results = {"cases": {}}
    if os.path.exists(path):
        try:
            results = json.load(open(path))
        except (OSError, ValueError) as e:
            print(f"[harness] {path} unreadable ({e!r}); refused, nothing written", flush=True)
            sys.exit(2)
    coll = preflight(a.out, names, results)
    superseded = []
    if coll and not a.overwrite:
        print("[harness] refused, nothing written: the selected cases collide with existing "
              f"evidence in {a.out}:", flush=True)
        for c in coll:
            print(f"    {c['case']}: {c['reason']}", flush=True)
        print("[harness] use a new --out directory (or select only new cases)", flush=True)
        sys.exit(2)
    os.makedirs(a.out, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for n in sorted({c["case"] for c in coll}):
        d = os.path.join(a.out, n)
        moved = None
        if os.path.exists(d):
            moved = f"{n}.superseded.{stamp}"
            os.rename(d, os.path.join(a.out, moved))
        rec = results.get("cases", {}).pop(n, None)
        results.setdefault("superseded", {})[f"{n}@{stamp}"] = {"directory": moved, "record": rec}
        superseded.append(n)
    inv_index = len(results.get("invocations", []))
    results.setdefault("tolerances", TOL)  # first invocation's; every invocation keeps its own copy
    results.setdefault("cases", {})
    results.setdefault("invocations", []).append(
        {**describe_env(a), "index": inv_index, "cases": names, "tolerances": TOL,
         "preexisting_cases": sorted(set(results["cases"]) - set(names)), "superseded": superseded})
    for n in names:
        fn, needs = CASES[n]
        if any(getattr(a, k) is None for k in needs):
            results["cases"][n] = {"pass": None, "skipped": f"needs {needs}", "invocation": inv_index}
            write_results(path, results)
            continue
        print(f"[harness] {n} ...", flush=True)
        t0 = time.time()
        try:
            res = fn(a, a.out).result()
        except Exception as e:  # noqa: BLE001  (a crash is a recorded failure)
            res = {"pass": False, "error": repr(e)}
        res["seconds"] = round(time.time() - t0, 1)
        res["invocation"] = inv_index
        results["cases"][n] = jsonable(res)
        print(f"[harness] {n}: {res['pass']} ({res['seconds']} s)", flush=True)
        for ch in res.get("checks", []):
            if not ch["pass"]:
                print(f"    FAILED {ch['check']}: {ch['value']} (tol {ch['tol']})", flush=True)
        if "error" in res:
            print(f"    ERROR {res['error']}", flush=True)
        write_results(path, results)
    bad = [n for n in names if results["cases"][n].get("pass") is False]
    print("[harness] failed cases of this invocation:", bad if bad else "none")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
