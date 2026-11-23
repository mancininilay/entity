#!/usr/bin/env python3
"""Made-up series for check6_position.py: the cases its rule must tell apart (2026-11-22; the
coverage, missing-data and 6c cases of Codex 2's review, B1 and P6, added 2026-11-23).

    test_check6_position.py <work directory>

Three members: A travels from the middle of the box to 240, B from 200 to 460, C stays at 20; in
all of them the median K of the pairs rises with the distance, K = 1 + d / 100. A bin is covered
only where two members contribute: 0 to 50 (A and C) and 200 to 250 (A and B). Each made-up run
has a run directory with an input and bh_counters.txt, for 6c. Tested runs:
  same        a copy of member B                                        -> pass (4 epochs judged)
  hot         B with three times its median K                           -> fail
  nowhere     a plasmoid at 480 to 499 with K = 50, nothing else         -> no verdict
  mixed       B, and in addition K = 50 at 480 to 499                    -> pass
  two         two eligible epochs in a covered bin (two apertures each)  -> no verdict
  nonfinite   three covered epochs and one eligible epoch whose values
              are not finite                                            -> no verdict
  onemember   epochs only where one member contributes (300 to 400)     -> no verdict
  nodirs      a copy of member B, run directories not given             -> no verdict
Exit 0 if every exit code is as expected.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

NAMES = ['time', 'step', 'radius', 'xc', 'yc', 'pxc', 'pyc', 'ptime', 'size', 'n_pair', 'n_prim', 'lep_written',
         'lep_medK', 'lep_meanK', 'lep_invgamma', 'ion_written', 'ion_medK', 'ion_meanK', 'ion_invgamma',
         'wpdt_pair', 'wpdt_all', 'cells_all', 'lep_written_box']
HEAD = ("# steps time pairs above_tables probability_exceeded loss_bound_skipped capacity_dropped absorbed_energy "
        + " ".join(f"n_{s} w_{s} K_{s} Kmax_{s}" for s in (1, 2, 3, 4)))


def write(work, run, xs, ks, kbox=5.0):
    rows = []
    for n, (x, k) in enumerate(zip(xs, ks)):
        for radius in (5.0, 2.0):
            r = dict.fromkeys(NAMES, 0.0)
            r.update(time=100.0 * (n + 1), step=1000 * (n + 1), radius=radius, xc=x, pxc=x, size=30.0, n_pair=5.0,
                     n_prim=10.0, lep_written=500.0, lep_medK=k, lep_meanK=2 * k, lep_invgamma=1.0 / (1.0 + k),
                     wpdt_pair=0.2, wpdt_all=0.5, cells_all=1.0, lep_written_box=5000.0)
            rows.append([r[name] for name in NAMES])
    np.savez(work / f"{run}.npz", names=np.array(NAMES), table=np.array(rows), radii=np.array([5.0, 2.0]), dt=0.0863,
             cells=4.096, run=run)
    d = work / run
    d.mkdir(exist_ok=True)
    (d / f"{run}.toml").write_text("[simulation]\n  runtime = 3000.0\n")
    lines = [HEAD]
    for t in (500.0, 1000.0, 2000.0, 3000.0):
        lines.append(f"{t / 0.0863:.0f} {t} 1000 0 0 0 0 1.0 1e7 1e7 1e8 10 1e7 1e7 1e8 10 "
                     f"5000 5000 {5000 * kbox} 50 5000 5000 {5000 * kbox} 50")
    (d / "bh_counters.txt").write_text("\n".join(lines) + "\n")


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    k = lambda x: 1.0 + np.abs(x) / 100.0  # noqa: E731
    xa, xb, xc = np.linspace(0, 240, 20), np.linspace(200, 460, 20), np.full(20, 20.0)
    for name, x in (("a", xa), ("b", xb), ("c", xc)):
        write(work, f"member-{name}", x, k(x))
    far = np.linspace(480, 499, 5)
    covered = np.array([205.0, 215.0, 225.0, 235.0])
    write(work, "same", xb, k(xb))
    write(work, "hot", xb, 3.0 * k(xb))
    write(work, "nowhere", far, np.full(5, 50.0))
    write(work, "mixed", np.concatenate([xb, far]), np.concatenate([k(xb), np.full(5, 50.0)]))
    write(work, "two", covered[:2], k(covered[:2]))
    ks = k(covered).copy()
    ks[3] = np.nan
    write(work, "nonfinite", covered, ks)
    one = np.linspace(300, 400, 6)
    write(work, "onemember", one, k(one))
    script = Path(__file__).resolve().parent / "check6_position.py"
    members = [str(work / f"member-{m}.npz") for m in "abc"]
    dirs = [str(work / f"member-{m}") for m in "abc"]
    cases = [("same", 0, True), ("hot", 1, True), ("nowhere", 2, True), ("mixed", 0, True), ("two", 2, True),
             ("nonfinite", 2, True), ("onemember", 2, True), ("same", 2, False)]
    got, want = [], []
    for case, expect, with_dirs in cases:
        cmd = [sys.executable, "-B", str(script), "--half-length", "500", "--test", str(work / f"{case}.npz"),
               "--ref"] + members
        if with_dirs:
            cmd += ["--test-dir", str(work / case), "--ref-dirs"] + dirs
        out = subprocess.run(cmd, capture_output=True, text=True)
        label = case if with_dirs else "nodirs"
        (work / f"{label}.txt").write_text(out.stdout + out.stderr)
        got.append(out.returncode)
        want.append(expect)
        last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-300:]
        print(f"{label}: exit {out.returncode} (expected {expect}): {last}")
    print("expected exit codes", want, "got", got, "OK" if got == want else "WRONG")
    return 0 if got == want else 1


if __name__ == "__main__":
    sys.exit(main())
