#!/usr/bin/env python3
"""The core series of the resolution pair: `scripts/check6_series.py` with the three changes that
Codex 1's review of stage C (C3) asks for (Claude 1, 2026-11-23).

    core_series.py <bp2raw> <cache> <run directory> <output prefix>

What check6_series.py does, kept: at every field output that holds pairs, the densest block of the
pairs; the mean densities of the pairs and of the primaries inside circles of radius 1, 2, 5, 10
and 20 skin depths around it; from the particle output nearest in time, the number of written
leptons of the pairs and of ions, their median and mean K and their mean of 1 / gamma inside the
same circles (no ion statistics when the run writes no ions); w_p dt and the cells per skin depth of the
mixture; the size of the plasmoid (the
connected blocks above 10 n0 holding the densest point, as an equivalent radius).

Changed:
  1. **One centre.** The particle apertures are centred on the densest block of the pairs of the
     field output, not on a separate densest point of the written leptons (they were up to 2.5
     skin depths apart in R-mid). The time between the field output and the particle output
     used is written (`dtime`).
  2. **Footprints in physical units.** The densest block is 4 cells and the blocks of the size 16
     cells on a grid of 4.096 cells per skin depth (the present grid), and scale with the cells
     per skin depth (8 and 32 at twice the resolution).
  3. **The frame of the pairs.** For the written leptons of the pairs inside each circle: the
     total four-momentum (unit weights, unit mass), the Lorentz factor of its zero-momentum frame
     (`bulk_gamma`), and the median K and the mean of 1 / gamma of the same particles in that
     frame (`lep_medK_frame`, `lep_invgamma_frame`). It is the frame of an instantaneous sample
     and follows no particle.
Diagnostics added after Codex 2's review (B2), not used by any criterion of 6b:
  `lep_peak_offset`  distance between the centre used and the densest point of the written
                     leptons of the pairs (histogram bins of 8 cells of the present grid, in
                     physical units), as check6_series.py centred the particles;
  `edge_outlet`      distance of the nearest x edge of the plasmoid (its connected blocks above
                     10 n0) from the nearer open end of the box, x = x0 or x = x1;
  `history`, `age`   island history: consecutive field outputs whose centres lie within 75 skin
                     depths in x and 30 in y of each other, with a plasmoid at each, belong to one
                     history (numbered from 1 in each run); `age` is the time since its first
                     output.
`size` is the equivalent RADIUS of the plasmoid, sqrt(area / pi), not a diameter.

The species are read from the input (`electron_species`, `positron_species`; the primaries
`setup.primary_species`, 1 and 2 by default; the ions are the primary species with the policy).

Writes <prefix>.txt and <prefix>.npz (the columns of check6_series.py, then dtime, bulk_gamma,
lep_medK_frame, lep_invgamma_frame), which `scripts/check6_position.py` reads.
"""
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
from bhio import Run  # noqa: E402

RADII = (1.0, 2.0, 5.0, 10.0, 20.0)
LINK_DX, LINK_DY = 75.0, 30.0   # skin depths: the largest jump of a centre between two outputs of one history
CELLS_REF = 4.096          # cells per skin depth of the present grid
BLOCK_REF, SIZE_BLOCK_REF, SIZE_LEVEL = 4, 16, 10.0
MIN_WRITTEN = 20


def block_mean(a, c):
    nx, ny = a.shape[0] // c, a.shape[1] // c
    return a[:nx * c, :ny * c].reshape(nx, c, ny, c).mean(axis=(1, 3))


def nearest(stamps, time):
    return min(stamps, key=lambda s: abs(s[2] - time))


def connected(mask, seed):
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


def energy_stats(x, y, u, xc, yc, r, frame=False):
    """Number, median K, mean K, mean 1/gamma within r; with frame=True also the Lorentz factor of
    the zero-momentum frame of the sample and the median K and mean 1/gamma in it."""
    m = (x - xc) ** 2 + (y - yc) ** 2 < r * r
    n = int(m.sum())
    if n < MIN_WRITTEN:
        return (n, np.nan, np.nan, np.nan) + ((np.nan, np.nan, np.nan) if frame else ())
    v = u[:, m].astype(np.float64)
    u2 = (v ** 2).sum(axis=0)
    g = np.sqrt(1.0 + u2)
    k = u2 / (g + 1.0)
    out = (n, float(np.median(k)), float(k.mean()), float((1.0 / g).mean()))
    if not frame:
        return out
    p0, p = g.sum(), v.sum(axis=1)
    beta = p / p0
    b2 = float((beta ** 2).sum())
    gb = 1.0 / np.sqrt(1.0 - b2)
    gp = gb * (g - beta @ v)
    kp = np.maximum(gp - 1.0, 0.0)
    return out + (float(gb), float(np.median(kp)), float((1.0 / gp).mean()))


def load(run, bp, species, xc, yc, rmax):
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


def species_of(run):
    e = int(run.param("electron_species", float))
    p = int(run.param("positron_species", float))
    m = re.search(r"^\s*primary_species\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", run.input, re.M)
    prim = (int(m.group(1)), int(m.group(2))) if m else (1, 2)
    blocks = run.input.split("[[particles.species]]")[1:]
    emitting = [i + 1 for i, b in enumerate(blocks) if re.search(r'emission\s*=\s*"(bethe_heitler|custom)"', b)]
    ions = [s for s in prim if s in emitting]
    if len(ions) != 1:
        raise SystemExit(f"{run.name}: cannot tell the emitting primary species ({emitting})")
    return (e, p), prim, ions[0]


def main():
    run = Run(sys.argv[3], sys.argv[1], sys.argv[2])
    prefix = sys.argv[4]
    leptons, prim, ion = species_of(run)
    x0, x1, y0, y1 = run.extent()
    nx = int(re.search(r"resolution\s*=\s*\[\s*(\d+)", run.input).group(1))
    cells = nx / (x1 - x0) * run.param("skindepth0")
    blk = max(1, int(round(BLOCK_REF * cells / CELLS_REF)))
    sblk = max(1, int(round(SIZE_BLOCK_REF * cells / CELLS_REF)))
    c = run.counters()
    dt = float(c["time"][0] / c["steps"][0])
    fields, particles = run.stamps("fields"), run.stamps("particles")
    rows = []
    hist_id, hist_t0, last = 0, None, None    # last: (time, xc, yc) of the previous output with a plasmoid
    text = [f"core series: {run.name}; dt = {dt:.9g}; cells per upstream skin depth {cells:.4g}; densest block "
            f"{blk} cells, size blocks {sblk} cells above {SIZE_LEVEL:g} n0; leptons of the pairs {leptons}, "
            f"primaries {prim}, ions {ion}; one centre for fields and particles",
            "time | radius | pairs | primaries | ratio | leptons written, median K, mean K, <1/gamma>, "
            "bulk gamma, median K and <1/gamma> in the frame of the pairs | ions written, median K, mean K, "
            "<1/gamma> | w_p dt: pairs, all | cells per skin depth"]
    fpair = [f"fN_{s}" for s in leptons]
    fprim = [f"fN_{s}" for s in prim]
    for bp, step, time in fields:
        v = run.fields(bp, fpair + fprim)
        x, y = v["X1"].astype(np.float64), v["X2"].astype(np.float64)
        pair = sum(v[k].astype(np.float64) for k in fpair)
        primd = sum(v[k].astype(np.float64) for k in fprim)
        sm = block_mean(pair, blk)
        if sm.max() <= 0.0:
            text.append(f"t = {time:.1f} (step {step}): no pairs")
            continue
        i, j = np.unravel_index(sm.argmax(), sm.shape)
        xc, yc = x[blk * i:blk * i + blk].mean(), y[blk * j:blk * j + blk].mean()
        coarse = block_mean(primd, sblk)
        seed = (min(blk * i // sblk, coarse.shape[0] - 1), min(blk * j // sblk, coarse.shape[1] - 1))
        part = connected(coarse > SIZE_LEVEL, seed)
        dx = (x1 - x0) / nx
        size = float(np.sqrt(part.sum() * (sblk * dx) ** 2 / np.pi))
        if part.any():
            cols = np.where(part.any(axis=1))[0]
            xmin, xmax = x0 + cols.min() * sblk * dx, x0 + (cols.max() + 1) * sblk * dx
            edge = float(min(xmin - x0, x1 - xmax))
            if last is not None and abs(xc - last[1]) <= LINK_DX and abs(yc - last[2]) <= LINK_DY:
                pass
            else:
                hist_id, hist_t0 = hist_id + 1, time
            last = (time, xc, yc)
            hist, age = hist_id, time - hist_t0
        else:
            edge, hist, age, last = float("nan"), 0, float("nan"), None
        pbp, pstep, ptime = nearest(particles, time)
        rmax = max(RADII)
        # the densest point of the written leptons, as a diagnostic of the centre (bins of 8 cells of
        # the present grid)
        lx, ly = [], []
        for s in leptons:
            q = run.read(pbp, [f"pX1_{s}", f"pX2_{s}"])
            lx.append(np.atleast_1d(q[f"pX1_{s}"]).astype(np.float64))
            ly.append(np.atleast_1d(q[f"pX2_{s}"]).astype(np.float64))
        lx, ly = np.concatenate(lx), np.concatenate(ly)
        nb = max(1, int(round(8 * cells / CELLS_REF)))
        if lx.size:
            hh, xe, ye = np.histogram2d(lx, ly, bins=[x.size // nb, y.size // nb], range=[[x0, x1], [y0, y1]])
            ia, ib = np.unravel_index(hh.argmax(), hh.shape)
            poff = float(np.hypot(0.5 * (xe[ia] + xe[ia + 1]) - xc, 0.5 * (ye[ib] + ye[ib + 1]) - yc))
        else:
            poff = float("nan")
        px, py, pu = load(run, pbp, leptons, xc, yc, rmax)
        try:
            ix, iy, iu = load(run, pbp, (ion,), xc, yc, rmax)
        except subprocess.CalledProcessError:      # no ions in the particle output (members of stage C)
            ix, iy, iu = np.zeros(0), np.zeros(0), np.zeros((3, 0))
        shutil.rmtree(run.cache / pbp.parent.name / pbp.name, ignore_errors=True)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        d2 = (xx - xc) ** 2 + (yy - yc) ** 2
        text.append(f"t = {time:.1f} (step {step}): densest block of the pairs at ({xc:.1f}, {yc:.1f}), "
                    f"{sm[i, j]:.2f}; particles of t = {ptime:.1f} (offset {ptime - time:+.2f}); size (radius) {size:.1f}; "
                    f"history {hist}, age {age:.0f}; edge to the outlet {edge:.1f}; densest written leptons {poff:.1f} away")
        for r in RADII:
            m = d2 < r * r
            n_pair, n_prim = float(pair[m].mean()), float(primd[m].mean())
            nl, medl, meanl, igl, gb, medf, igf = energy_stats(px, py, pu, xc, yc, r, frame=True)
            ni, medi, meani, igi = energy_stats(ix, iy, iu, xc, yc, r)
            w2_pair = n_pair * igl
            w2_all = w2_pair + n_prim * igi
            wp_pair, wp_all = dt * np.sqrt(w2_pair), dt * np.sqrt(w2_all)
            rows.append([time, step, r, xc, yc, xc, yc, ptime, size, n_pair, n_prim, nl, medl, meanl, igl,
                         ni, medi, meani, igi, wp_pair, wp_all, cells / np.sqrt(w2_all), px.size,
                         ptime - time, gb, medf, igf, poff, edge, hist, age])
            text.append(f"  {time:7.1f} | {r:4.0f} | {n_pair:9.3f} | {n_prim:9.2f} | {n_pair / n_prim:6.3f} | "
                        f"{nl:7d} {medl:7.3f} {meanl:8.3f} {igl:.3f} {gb:6.3f} {medf:7.3f} {igf:.3f} | "
                        f"{ni:8d} {medi:8.2f} {meani:8.2f} {igi:.4f} | {wp_pair:.3f} {wp_all:.3f} | "
                        f"{cells / np.sqrt(w2_all):.2f}")
    names = ["time", "step", "radius", "xc", "yc", "pxc", "pyc", "ptime", "size", "n_pair", "n_prim",
             "lep_written", "lep_medK", "lep_meanK", "lep_invgamma", "ion_written", "ion_medK", "ion_meanK",
             "ion_invgamma", "wpdt_pair", "wpdt_all", "cells_all", "lep_written_box",
             "dtime", "bulk_gamma", "lep_medK_frame", "lep_invgamma_frame",
             "lep_peak_offset", "edge_outlet", "history", "age"]
    a = np.array(rows, dtype=np.float64).reshape(-1, len(names))
    np.savez(prefix + ".npz", names=np.array(names), table=a, radii=np.array(RADII), dt=dt, cells=cells,
             run=run.name)
    open(prefix + ".txt", "w").write("\n".join(text) + "\n")
    print("\n".join(text[:2]))
    print(f"{len(rows) // len(RADII)} outputs")


if __name__ == "__main__":
    main()
