#!/usr/bin/env python3
"""Made-up series for check6_position.py: the cases its rule must tell apart.

    test_check6_position.py <work directory>

Three members: A travels from the middle of the box to 240, B from 200 to 460, C stays at 20; in
all of them the median K of the pairs rises with the distance, K = 1 + d / 100. Tested runs:
  same     a copy of member B                                   -> pass
  hot      B with three times its median K                      -> fail
  nowhere  a plasmoid at 480 to 499 with K = 50, where no member
           has three outputs, and nothing else                   -> no verdict
  mixed    B, and in addition K = 50 at 480 to 499               -> pass, with outputs not judged
Exit 0 if the four exit codes are 0, 1, 2, 0.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

NAMES = ['time', 'step', 'radius', 'xc', 'yc', 'pxc', 'pyc', 'ptime', 'size', 'n_pair', 'n_prim', 'lep_written',
         'lep_medK', 'lep_meanK', 'lep_invgamma', 'ion_written', 'ion_medK', 'ion_meanK', 'ion_invgamma',
         'wpdt_pair', 'wpdt_all', 'cells_all', 'lep_written_box']


def write(path, run, xs, ks):
    rows = []
    for n, (x, k) in enumerate(zip(xs, ks)):
        for radius in (5.0, 2.0):
            r = dict.fromkeys(NAMES, 0.0)
            r.update(time=100.0 * (n + 1), step=1000 * (n + 1), radius=radius, xc=x, pxc=x, size=30.0, n_pair=5.0,
                     n_prim=10.0, lep_written=500.0, lep_medK=k, lep_meanK=2 * k, lep_invgamma=1.0 / (1.0 + k),
                     wpdt_pair=0.2, wpdt_all=0.5, cells_all=1.0, lep_written_box=5000.0)
            rows.append([r[name] for name in NAMES])
    np.savez(path, names=np.array(NAMES), table=np.array(rows), radii=np.array([5.0, 2.0]), dt=0.0863, cells=4.096,
             run=run)


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    k = lambda x: 1.0 + np.abs(x) / 100.0  # noqa: E731
    xa, xb, xc = np.linspace(0, 240, 20), np.linspace(200, 460, 20), np.full(20, 20.0)
    for name, x in (("a", xa), ("b", xb), ("c", xc)):
        write(work / f"member-{name}.npz", f"member-{name}", x, k(x))
    far = np.linspace(480, 499, 5)
    write(work / "same.npz", "same", xb, k(xb))
    write(work / "hot.npz", "hot", xb, 3.0 * k(xb))
    write(work / "nowhere.npz", "nowhere", far, np.full(5, 50.0))
    write(work / "mixed.npz", "mixed", np.concatenate([xb, far]), np.concatenate([k(xb), np.full(5, 50.0)]))
    script = Path(__file__).resolve().parent / "check6_position.py"
    got = []
    for case in ("same", "hot", "nowhere", "mixed"):
        out = subprocess.run([sys.executable, "-B", str(script), "--half-length", "500", "--test",
                              str(work / f"{case}.npz"), "--ref"] + [str(work / f"member-{m}.npz") for m in "abc"],
                             capture_output=True, text=True)
        (work / f"{case}.txt").write_text(out.stdout + out.stderr)
        got.append(out.returncode)
        print(f"{case}: exit {out.returncode}: {out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-200:]}")
    want = [0, 1, 2, 0]
    print("expected exit codes", want, "got", got, "OK" if got == want else "WRONG")
    return 0 if got == want else 1


if __name__ == "__main__":
    sys.exit(main())
