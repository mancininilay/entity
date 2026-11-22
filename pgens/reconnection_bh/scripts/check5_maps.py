#!/usr/bin/env python3
"""Check 5: where the pairs are against where the ions above threshold are,
and the spectrum of the pairs.

    check5_maps.py <bp2raw> <cache> <run with the policy> <output directory>

At the field outputs nearest to runtime / 4, / 2, 3 / 4 and at the last one:
the density of the Bethe-Heitler leptons (N_3 + N_4 of the field output) and
the density of the ions above threshold (particle output of species 2 nearest
in time, K > K_th, counted on a grid 16 times coarser than the cells and
scaled by the stride), as two maps; the correlation coefficient of the two on
the coarse grid; the share of the pairs in coarse cells that hold at least
one written ion above threshold. Then the spectra of the two lepton species
at the same times, with those of the electrons and of the ions at the end.
Numbers go to `check5.txt`, the coarse maps to `check5_maps.npz`, the spectra
to `check5_spectra.txt`; the figures `check5_maps_<k>.png` and
`check5_spectra.png` are drawn from these files by plot_check5.py, here if
matplotlib is available, else wherever the three files are copied.
"""
import sys
from pathlib import Path

import numpy as np

from bhio import Run, threshold

COARSE = 16


def nearest(stamps, time):
    return min(stamps, key=lambda s: abs(s[2] - time))


def block_mean(a, c):
    nx, ny = a.shape[0] // c, a.shape[1] // c
    return a[:nx * c, :ny * c].reshape(nx, c, ny, c).mean(axis=(1, 3))


def main():
    run = Run(sys.argv[3], sys.argv[1], sys.argv[2])
    out = Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    runtime = run.param("runtime")
    mass_ratio = run.param("mass_ratio")
    _, k_th = threshold(run.param("photon_energy"), mass_ratio)
    x0, x1, y0, y1 = run.extent()
    fields, particles, spectra = run.stamps("fields"), run.stamps("particles"), run.stamps("spectra")
    chosen = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        s = nearest(fields, frac * runtime)
        if s not in chosen:
            chosen.append(s)
    lines = [f"check 5: {run.name}; threshold K_th = {k_th:.4f}; coarse grid {COARSE} cells"]
    spec_rows = []
    saved = {}
    for k, (bp, step, time) in enumerate(chosen):
        v = run.fields(bp, ["fN_3", "fN_4"])
        pairs = v["fN_3"].astype(np.float64) + v["fN_4"].astype(np.float64)
        nx, ny = pairs.shape
        pbp, pstep, ptime = nearest(particles, time)
        p = run.read(pbp, ["pX1_2", "pX2_2", "pU1_2", "pU2_2", "pU3_2"])
        u2 = sum(p[n].astype(np.float64) ** 2 for n in ("pU1_2", "pU2_2", "pU3_2"))
        K = u2 / (np.sqrt(1.0 + u2) + 1.0)
        m = K > k_th
        hist, _, _ = np.histogram2d(p["pX1_2"][m], p["pX2_2"][m],
                                    bins=[nx // COARSE, ny // COARSE], range=[[x0, x1], [y0, y1]])
        coarse = block_mean(pairs, COARSE)
        both = (coarse.sum() > 0) and (hist.sum() > 0)
        r = float(np.corrcoef(coarse.ravel(), hist.ravel())[0, 1]) if both else float("nan")
        share = float(coarse[hist > 0].sum() / coarse.sum()) if coarse.sum() > 0 else float("nan")
        lines.append(f"  t = {time:.6g} (step {step}; ions of step {pstep}, t = {ptime:.6g}): "
                     f"ions written {K.size}, above threshold {int(m.sum())} "
                     f"({m.mean():.4f}); mean lepton density {pairs.mean():.4e}; "
                     f"correlation {r:.4f}; share of the pairs in cells with ions above threshold {share:.4f}; "
                     f"cells with such ions {np.mean(hist > 0):.4f} of the grid")
        sbp, sstep, stime = nearest(spectra, time)
        sv = run.read(sbp, ["sEbn"] + [f"sN_{s}" for s in (1, 2, 3, 4)])
        edges = sv["sEbn"].astype(np.float64)
        counts = {s: sv[f"sN_{s}"].astype(np.float64).reshape(2, -1).sum(axis=0) for s in (1, 2, 3, 4)}
        spec_rows.append((stime, edges, counts))
        lep = counts[3] + counts[4]
        centre = np.sqrt(edges[1:] * edges[:-1])
        if lep.sum() > 0:
            cum = np.cumsum(lep) / lep.sum()
            q = [float(centre[np.searchsorted(cum, x)]) for x in (0.5, 0.9, 0.99)]
            lines.append(f"    leptons of the pairs at t = {stime:.6g}: {lep.sum():.0f}; mean K "
                         f"{(lep * centre).sum() / lep.sum():.4g}; K at 50, 90, 99 percent "
                         f"{q[0]:.4g}, {q[1]:.4g}, {q[2]:.4g}")
        saved[f"pairs_{k}"] = coarse
        saved[f"ions_{k}"] = hist
        saved[f"time_{k}"] = np.array(time)
    np.savez_compressed(out / "check5_maps.npz", extent=np.array([x0, x1, y0, y1]),
                        k_th=np.array(k_th), n=np.array(len(chosen)), **saved)
    with open(out / "check5_spectra.txt", "w") as f:
        f.write("# K_lo K_hi then, for each time, the counts of species 1 2 3 4\n")
        f.write("# times: " + " ".join(f"{s[0]:.9g}" for s in spec_rows) + "\n")
        edges = spec_rows[0][1]
        for b in range(edges.size - 1):
            row = [f"{edges[b]:.6e}", f"{edges[b + 1]:.6e}"]
            for _, _, counts in spec_rows:
                row += [f"{counts[s][b]:.0f}" for s in (1, 2, 3, 4)]
            f.write(" ".join(row) + "\n")
    text = "\n".join(lines) + "\n"
    (out / "check5.txt").write_text(text)
    print(text, end="")
    try:
        import plot_check5
        plot_check5.plot(out)
        print("  figures written")
    except ImportError:
        print("  NO FIGURES here (no matplotlib): run plot_check5.py on the files of", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
