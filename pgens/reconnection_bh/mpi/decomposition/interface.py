#!/usr/bin/env python3
"""Codex 1's interface measure, for one run: the cell-to-cell difference of the saved fields
across a plane against the same difference across the neighbouring parallel planes.

    interface.py <bp2raw> <cache> <run directory> <output.json>

Definition (his program seams.py of the review of 2026-11-22, G5, written again here on the
reader of the checks). For a field f of the field output (cell-centred, as saved), a plane
between two neighbouring rows of cells, and a window along the plane:
  rms(k)  = root mean square, along the window, of f(row + k) - f(row + k - 1),
  jump(k) = mean, along the window, of the same difference,
for k = -8 ... 8; k = 0 is the difference across the plane itself.
Planes: x = -L_x / 4, x = 0, x = +L_x / 4 (the cuts of a [4, 1] decomposition; x = 0 is also the
cut in x of [2, 2]), window |y| <= 190; and y = 0 (the cut in y of [2, 2], which is the plane of
the current layer), window |x| <= 400. Every plane is measured in every run, whether it is a
rank boundary there or not.
Fields: the densities of the primaries, and the three components of B, E and J.
The ratios and their median over the outputs are formed by verdict.py.
"""
import json
import shutil
import sys

import numpy as np

from bhio import Run

FIELDS = ["fN_2", "fB1", "fB2", "fE3", "fJ3", "fN_1", "fB3", "fE1", "fE2", "fJ1", "fJ2"]
OFFSETS = np.arange(-8, 9)
X_WINDOW, Y_WINDOW = 190.0, 400.0


def lines(sub):
    """sub: 20 rows of cells around the plane (the plane lies between rows 9 and 10) x window."""
    d = np.diff(sub, axis=0)[9 + OFFSETS]
    return np.sqrt(np.mean(d * d, axis=1)).tolist(), d.mean(axis=1).tolist()


def main():
    bp2raw, cache, path, out = sys.argv[1:5]
    run = Run(path, bp2raw, cache)
    x0, x1, y0, y1 = run.extent()
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    rows, times = [], []
    for bp, step, time in run.stamps("fields"):
        v = run.fields(bp, FIELDS)
        x, y = v["X1"].astype(np.float64) - cx, v["X2"].astype(np.float64) - cy
        nx, ny = x.size, y.size
        wy, wx = np.abs(y) <= X_WINDOW, np.abs(x) <= Y_WINDOW
        planes = {"x = -L/4": ("x", nx // 4), "x = 0": ("x", nx // 2), "x = +L/4": ("x", 3 * nx // 4),
                  "y = 0": ("y", ny // 2)}
        finite = True
        for name in FIELDS:
            a = v[name].astype(np.float64)
            finite = finite and bool(np.isfinite(a).all())
            for label, (axis, i0) in planes.items():
                sub = a[i0 - 10:i0 + 10, wy] if axis == "x" else a[wx, i0 - 10:i0 + 10].T
                rms, jump = lines(sub)
                rows.append({"t": time, "field": name, "plane": label, "rms": rms, "jump": jump})
        times.append({"t": time, "step": step, "finite": finite,
                      "x_planes_between_cells": [[i - 1, i] for i in (nx // 4, nx // 2, 3 * nx // 4)],
                      "y_plane_between_cells": [ny // 2 - 1, ny // 2],
                      "x_plane_positions": [float(0.5 * (x[i - 1] + x[i])) for i in (nx // 4, nx // 2, 3 * nx // 4)],
                      "y_plane_position": float(0.5 * (y[ny // 2 - 1] + y[ny // 2])),
                      "cells_in_x_window": int(wy.sum()), "cells_in_y_window": int(wx.sum())})
        shutil.rmtree(run.cache / "fields" / bp.name, ignore_errors=True)
    shutil.rmtree(run.cache, ignore_errors=True)
    with open(out, "w") as f:
        json.dump({"name": run.name, "directory": run.dir.name, "outputs": times, "rows": rows}, f)
    print(f"{run.name}: {len(times)} field outputs, {len(rows)} lines; all finite: {all(o['finite'] for o in times)}")


if __name__ == "__main__":
    main()
