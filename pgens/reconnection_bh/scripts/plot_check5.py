#!/usr/bin/env python3
"""Figures of check 5 from the files written by check5_maps.py (matplotlib).

    plot_check5.py <directory with check5_maps.npz and check5_spectra.txt>
"""
import sys
from pathlib import Path

import numpy as np


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    out = Path(out)
    z = np.load(out / "check5_maps.npz")
    ext = list(z["extent"])
    for k in range(int(z["n"])):
        time = float(z[f"time_{k}"])
        fig, ax = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for a, data, title in ((ax[0], z[f"pairs_{k}"], "Bethe-Heitler leptons (N_3 + N_4)"),
                               (ax[1], z[f"ions_{k}"],
                                "ions above threshold (written particles per coarse cell)")):
            pos = data[data > 0]
            if pos.size:
                im = a.imshow(data.T, origin="lower", extent=ext, aspect="equal", cmap="viridis",
                              norm=LogNorm(vmin=max(pos.min(), pos.max() * 1e-4), vmax=pos.max()))
                fig.colorbar(im, ax=a, shrink=0.8)
            a.set_title(f"{title}, t = {time:.5g}")
            a.set_ylabel("y [d_e]")
        ax[1].set_xlabel("x [d_e]")
        fig.tight_layout()
        fig.savefig(out / f"check5_maps_{k}.png", dpi=130)
        plt.close(fig)
    lines = open(out / "check5_spectra.txt").read().splitlines()
    times = [float(x) for x in lines[1].split(":")[1].split()]
    data = np.array([[float(x) for x in line.split()] for line in lines[2:]])
    lo, hi = data[:, 0], data[:, 1]
    centre, dln = np.sqrt(lo * hi), np.log(hi / lo)
    fig, a = plt.subplots(figsize=(7.5, 5.5))
    for i, t in enumerate(times):
        lep = (data[:, 2 + 4 * i + 2] + data[:, 2 + 4 * i + 3]) / dln
        a.loglog(centre[1:-1], lep[1:-1], color=plt.cm.plasma(i / max(len(times) - 1, 1)),
                 label=f"pairs, t = {t:.5g}")
    i = len(times) - 1
    a.loglog(centre[1:-1], (data[:, 2 + 4 * i] / dln)[1:-1], "k--", lw=1, label="electrons, end")
    a.loglog(centre[1:-1], (data[:, 2 + 4 * i + 1] / dln)[1:-1], "k:", lw=1, label="ions, end")
    a.axvline(float(z["k_th"]), color="grey", lw=0.8)
    a.set_xlabel("K = gamma - 1 [m_e c^2]")
    a.set_ylabel("dN / d ln K")
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "check5_spectra.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    plot(sys.argv[1])
