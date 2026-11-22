#!/usr/bin/env python3
"""Check 6, first part: the densest region of the pairs against time.

    check6_series.py <bp2raw> <cache> <run directory> <output prefix>

At every field output that holds pairs: the densest point of N_3 + N_4 (blocks
of 4 x 4 cells) and, inside circles of radius 1, 2, 5, 10 and 20 skin depths
around it, the mean densities of the pairs (N_3 + N_4) and of the primaries
(N_1 + N_2) from the field output. From the particle output nearest in time
(ions and leptons of the pairs at the stride of the run; the primary electrons
are not written and are taken to be like the ions, which have the same mass):
the number written, the median and the mean of K and the mean of 1 / gamma,
inside the same circles around the densest point of the written leptons.

Plasma frequency of the mixture in units of the upstream one,
w^2 = sum over species of N_s <1 / gamma>_s (densities in units of n0, to
which skindepth0 refers), given as w dt and as cells per skin depth.

Size of the plasmoid that holds the densest point: N_1 + N_2 averaged over
blocks of 16 x 16 cells; the connected set of blocks above 10 n0 that
contains the densest point; its equivalent radius sqrt(area / pi). Zero when
the block of the densest point is not above 10 n0.

Writes <prefix>.txt (a table) and <prefix>.npz (the same numbers). The raw
extraction of a particle output is removed from the cache once it is used.
"""
import re
import shutil
import sys

import numpy as np

from bhio import Run

RADII = (1.0, 2.0, 5.0, 10.0, 20.0)
SIZE_BLOCK, SIZE_LEVEL = 16, 10.0
MIN_WRITTEN = 20        # fewer written particles in a circle: no energy statistics


def block_mean(a, c):
    nx, ny = a.shape[0] // c, a.shape[1] // c
    return a[:nx * c, :ny * c].reshape(nx, c, ny, c).mean(axis=(1, 3))


def nearest(stamps, time):
    return min(stamps, key=lambda s: abs(s[2] - time))


def connected(mask, seed):
    """The 4-connected part of `mask` that contains `seed` (a boolean array)."""
    part = np.zeros_like(mask)
    if not mask[seed]:
        return part
    part[seed] = True
    while True:
        grown = part.copy()
        grown[1:, :] |= part[:-1, :]
        grown[:-1, :] |= part[1:, :]
        grown[:, 1:] |= part[:, :-1]
        grown[:, :-1] |= part[:, 1:]
        grown &= mask
        if grown.sum() == part.sum():
            return part
        part = grown


def energy_stats(x, y, u, xc, yc, r):
    """Number, median K, mean K, mean 1 / gamma of the particles within r of (xc, yc)."""
    m = (x - xc) ** 2 + (y - yc) ** 2 < r * r
    n = int(m.sum())
    if n < MIN_WRITTEN:
        return n, np.nan, np.nan, np.nan
    u2 = (u[:, m].astype(np.float64) ** 2).sum(axis=0)
    g = np.sqrt(1.0 + u2)
    k = u2 / (g + 1.0)
    return n, float(np.median(k)), float(k.mean()), float((1.0 / g).mean())


def load(run, bp, species, xc, yc, rmax):
    """Positions and momenta of the written particles of `species` within rmax of (xc, yc)."""
    xs, ys, us = [], [], []
    for s in species:
        p = run.read(bp, [f"pX1_{s}", f"pX2_{s}"])
        x, y = np.atleast_1d(p[f"pX1_{s}"]).astype(np.float64), np.atleast_1d(p[f"pX2_{s}"]).astype(np.float64)
        m = (x - xc) ** 2 + (y - yc) ** 2 < rmax * rmax
        q = run.read(bp, [f"pU1_{s}", f"pU2_{s}", f"pU3_{s}"])
        xs.append(x[m])
        ys.append(y[m])
        us.append(np.stack([np.atleast_1d(q[f"pU{i}_{s}"])[m] for i in (1, 2, 3)]))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(us, axis=1)


def main():
    run = Run(sys.argv[3], sys.argv[1], sys.argv[2])
    prefix = sys.argv[4]
    x0, x1, y0, y1 = run.extent()
    nx = int(re.search(r"resolution\s*=\s*\[\s*(\d+)", run.input).group(1))
    cells = nx / (x1 - x0) * run.param("skindepth0")
    c = run.counters()
    dt = float(c["time"][0] / c["steps"][0])
    fields, particles = run.stamps("fields"), run.stamps("particles")
    rows = []
    text = [f"check 6 series: {run.name}; dt = {dt:.9g}; cells per upstream skin depth {cells:.4g}; "
            f"radii {RADII}; size: blocks of {SIZE_BLOCK} cells above {SIZE_LEVEL:g} n0",
            "time | radius | pairs N_3+N_4 | primaries N_1+N_2 | ratio | leptons written, median K, mean K, "
            "<1/gamma> | ions written, median K, mean K, <1/gamma> | w_p dt: pairs only, all | "
            "cells per skin depth, all"]
    for bp, step, time in fields:
        v = run.fields(bp, ["fN_1", "fN_2", "fN_3", "fN_4"])
        x, y = v["X1"].astype(np.float64), v["X2"].astype(np.float64)
        pair = v["fN_3"].astype(np.float64) + v["fN_4"].astype(np.float64)
        prim = v["fN_1"].astype(np.float64) + v["fN_2"].astype(np.float64)
        sm = block_mean(pair, 4)
        if sm.max() <= 0.0:
            text.append(f"t = {time:.1f} (step {step}): no pairs")
            continue
        i, j = np.unravel_index(sm.argmax(), sm.shape)
        xc, yc = x[4 * i:4 * i + 4].mean(), y[4 * j:4 * j + 4].mean()
        coarse = block_mean(prim, SIZE_BLOCK)
        seed = (min(4 * i // SIZE_BLOCK, coarse.shape[0] - 1), min(4 * j // SIZE_BLOCK, coarse.shape[1] - 1))
        part = connected(coarse > SIZE_LEVEL, seed)
        dx = (x1 - x0) / nx
        size = float(np.sqrt(part.sum() * (SIZE_BLOCK * dx) ** 2 / np.pi))
        pbp, pstep, ptime = nearest(particles, time)
        # the densest point of the written leptons (as core_pairs.py of stage B)
        lx, ly = [], []
        for s in (3, 4):
            p = run.read(pbp, [f"pX1_{s}", f"pX2_{s}"])
            lx.append(np.atleast_1d(p[f"pX1_{s}"]).astype(np.float64))
            ly.append(np.atleast_1d(p[f"pX2_{s}"]).astype(np.float64))
        lx, ly = np.concatenate(lx), np.concatenate(ly)
        if lx.size == 0:
            text.append(f"t = {time:.1f} (step {step}): no written leptons")
            continue
        h, xe, ye = np.histogram2d(lx, ly, bins=[x.size // 8, y.size // 8], range=[[x0, x1], [y0, y1]])
        a, b = np.unravel_index(h.argmax(), h.shape)
        pxc, pyc = 0.5 * (xe[a] + xe[a + 1]), 0.5 * (ye[b] + ye[b + 1])
        rmax = max(RADII)
        px, py, pu = load(run, pbp, (3, 4), pxc, pyc, rmax)
        ix, iy, iu = load(run, pbp, (2,), pxc, pyc, rmax)
        shutil.rmtree(run.cache / pbp.parent.name / pbp.name, ignore_errors=True)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        d2 = (xx - xc) ** 2 + (yy - yc) ** 2
        text.append(f"t = {time:.1f} (step {step}): densest block of the pairs at ({xc:.1f}, {yc:.1f}), "
                    f"N_3 + N_4 = {sm[i, j]:.2f}; particles of t = {ptime:.1f}, densest point of the written "
                    f"leptons ({pxc:.1f}, {pyc:.1f}), written in the box {lx.size}; size {size:.1f}")
        for r in RADII:
            m = d2 < r * r
            n_pair, n_prim = float(pair[m].mean()), float(prim[m].mean())
            nl, medl, meanl, igl = energy_stats(px, py, pu, pxc, pyc, r)
            ni, medi, meani, igi = energy_stats(ix, iy, iu, pxc, pyc, r)
            w2_pair = n_pair * igl
            w2_all = w2_pair + n_prim * igi
            wp_pair, wp_all = dt * np.sqrt(w2_pair), dt * np.sqrt(w2_all)
            rows.append([time, step, r, xc, yc, pxc, pyc, ptime, size, n_pair, n_prim, nl, medl, meanl, igl,
                         ni, medi, meani, igi, wp_pair, wp_all, cells / np.sqrt(w2_all), lx.size])
            text.append(f"  {time:7.1f} | {r:4.0f} | {n_pair:9.3f} | {n_prim:9.2f} | {n_pair / n_prim:6.3f} | "
                        f"{nl:7d} {medl:7.3f} {meanl:8.3f} {igl:.3f} | {ni:8d} {medi:8.2f} {meani:8.2f} {igi:.4f} | "
                        f"{wp_pair:.3f} {wp_all:.3f} | {cells / np.sqrt(w2_all):.2f}")
    names = ["time", "step", "radius", "xc", "yc", "pxc", "pyc", "ptime", "size", "n_pair", "n_prim",
             "lep_written", "lep_medK", "lep_meanK", "lep_invgamma", "ion_written", "ion_medK", "ion_meanK",
             "ion_invgamma", "wpdt_pair", "wpdt_all", "cells_all", "lep_written_box"]
    a = np.array(rows, dtype=np.float64).reshape(-1, len(names))
    np.savez(prefix + ".npz", names=np.array(names), table=a, radii=np.array(RADII), dt=dt, cells=cells,
             run=run.name)
    open(prefix + ".txt", "w").write("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
