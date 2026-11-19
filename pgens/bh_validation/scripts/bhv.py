"""Shared readers of the bh_validation diagnostics (standard library only)."""
import math
import os
import re


def read_rows(run_dir):
    rows = []
    for line in open(os.path.join(run_dir, "run.log")):
        if line.startswith("BHV step"):
            t = line.split()
            rows.append(dict(zip(t[1::2], (float(x) for x in t[2::2]))))
    return rows


def read_input(run_dir):
    text = open(os.path.join(run_dir, "input.toml")).read()

    def get(key, default=None):
        m = re.search(rf"^\s*{key}\s*=\s*(.+)$", text, re.M)
        return float(m.group(1)) if m else default

    return dict(tau=get("optical_depth"), a=get("pair_weight", 1.0), mu_r=get("mass_ratio"),
                gamma_p=get("gamma_p"), n_ions=get("n_ions"))


class Series:
    """<K>, Var K and pairs per emitter after n steps (t = n dt)."""

    def __init__(self, run_dir):
        rows = read_rows(run_dir)
        inp = read_input(run_dir)
        self.rows, self.inp = rows, inp
        self.K0 = rows[0].get("K_ref", inp["mu_r"] * (inp["gamma_p"] - 1.0))
        self.dt = (rows[1]["time"] - rows[0]["time"]) / (rows[1]["step"] - rows[0]["step"])
        self.N = rows[0]["ion_w"]
        self.n_emitters = rows[0]["ion_n"]
        N = self.N
        self.steps = [0.0] + [r["step"] + 1.0 for r in rows]
        self.t = [n * self.dt for n in self.steps]
        self.mean = [self.K0] + [self.K0 + r["ion_dK"] / r["ion_w"] for r in rows]
        self.var = [0.0] + [(r["ion_dK2"] / r["ion_w"] - (r["ion_dK"] / r["ion_w"]) ** 2) * N / (N - 1.0) for r in rows]
        # pairs created, in units of the emitters' weight (a pair of weight a counts a)
        self.pairs = [0.0] + [r["pairs"] / r["ion_n"] for r in rows]

    def se(self, i):
        return math.sqrt(self.var[i] / self.n_emitters)

    def first_below(self, f):
        for i, m in enumerate(self.mean):
            if m <= f * self.K0:
                return i
        return None

    def at_time(self, t):
        best = min(range(len(self.t)), key=lambda i: abs(self.t[i] - t))
        return best
