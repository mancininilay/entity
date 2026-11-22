"""Readers for the output of pgens/reconnection_bh (needs numpy).

A run directory holds the input file, `bh_counters.txt`, `bh_gauss.txt` and
the output directory of the engine, `<simulation.name>/{fields,particles,
spectra}/*.bp`. BP variables are extracted with tools/bp2raw into a cache
directory and read as raw arrays.
"""
import re
import subprocess
from pathlib import Path

import numpy as np

DTYPES = {
    "float": np.float32, "double": np.float64,
    "int32_t": np.int32, "uint32_t": np.uint32,
    "int64_t": np.int64, "uint64_t": np.uint64,
    "int16_t": np.int16, "uint16_t": np.uint16,
    "int8_t": np.int8, "uint8_t": np.uint8,
}


class Run:
    def __init__(self, run_dir, bp2raw, cache):
        self.dir = Path(run_dir)
        self.bp2raw = str(bp2raw)
        outs = [p.parent for p in self.dir.glob("*/fields") if p.is_dir()]
        if len(outs) != 1:
            raise SystemExit(f"{run_dir}: expected one output directory, found {len(outs)}")
        self.out = outs[0]
        self.name = self.out.name
        self.cache = Path(cache) / self.name
        inputs = sorted(self.dir.glob("*.toml"))
        if len(inputs) != 1:
            raise SystemExit(f"{run_dir}: expected one input file, found {len(inputs)}")
        self.input = inputs[0].read_text()

    # ------------------------------------------------------------------ input
    def param(self, key, cast=float, default=None):
        """Value of the first line `key = value` of the input file."""
        m = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.+?)\s*$", self.input, re.M)
        if m is None:
            if default is None:
                raise SystemExit(f"{self.name}: no `{key}` in the input")
            return default
        return cast(m.group(1).strip('"'))

    def extent(self):
        m = re.search(r"extent\s*=\s*\[\[\s*([-0-9.eE+]+)\s*,\s*([-0-9.eE+]+)\s*\]\s*,"
                      r"\s*\[\s*([-0-9.eE+]+)\s*,\s*([-0-9.eE+]+)\s*\]\]", self.input)
        return tuple(float(x) for x in m.groups())

    def sigma(self):
        return (self.param("skindepth0") / self.param("larmor0")) ** 2

    def emitting(self):
        return re.search(r'^\s*emission\s*=\s*"bethe_heitler"', self.input, re.M) is not None

    # ----------------------------------------------------------------- output
    def outputs(self, kind):
        return sorted((self.out / kind).glob("*.bp"))

    def read(self, bp, names):
        """Variables `names` of one BP file as a dict of arrays."""
        bp = Path(bp)
        d = self.cache / bp.parent.name / bp.name
        d.mkdir(parents=True, exist_ok=True)
        missing = [n for n in names if not (d / f"{n}.0.raw").exists()]
        index_file = d / "index.txt"
        lines = index_file.read_text().splitlines() if index_file.exists() else []
        if missing:
            tmp = d / "new"
            tmp.mkdir(exist_ok=True)
            subprocess.run([self.bp2raw, str(bp), str(tmp)] + missing, check=True)
            for n in missing:
                (tmp / f"{n}.0.raw").replace(d / f"{n}.0.raw")
            lines += (tmp / "index.txt").read_text().splitlines()
            index_file.write_text("\n".join(lines) + "\n")
        index = {}
        for line in lines:
            t = line.split()
            index[t[0]] = (t[2], [int(x) for x in t[4:]])
        out = {}
        for n in names:
            dtype, dims = index[n]
            a = np.fromfile(d / f"{n}.0.raw", dtype=DTYPES[dtype])
            out[n] = a.reshape(dims) if len(dims) > 1 else (a if dims else a[0])
        return out

    def fields(self, bp, names):
        """Field arrays of one output with the shape (cells in x1, cells in x2), and
        the cell centres X1, X2. A build whose arrays are column-major (CUDA) writes
        the fields with their dimensions reversed (writer.cpp, attribute LayoutRight
        = 0): such arrays are transposed here."""
        v = self.read(bp, ["X1", "X2"] + list(names))
        nx, ny = v["X1"].size, v["X2"].size
        if nx == ny:
            raise SystemExit("square grid: the layout of the field output cannot be told from its shape")
        for n in names:
            if v[n].shape == (ny, nx):
                v[n] = v[n].T
            elif v[n].shape != (nx, ny):
                raise SystemExit(f"{bp}: {n} has the shape {v[n].shape}, grid {nx} x {ny}")
        return v

    def stamps(self, kind):
        """(file, step, time) of every output of one kind."""
        out = []
        for bp in self.outputs(kind):
            v = self.read(bp, ["Step", "Time"])
            out.append((bp, int(v["Step"]), float(v["Time"])))
        return out

    # ------------------------------------------------------------ text files
    def _table(self, name):
        path = self.dir / name
        header = None
        for line in open(path):
            if line.startswith("#"):
                header = line[1:].split()
            else:
                break
        data = np.loadtxt(path, ndmin=2)
        if data.shape[1] != len(header):
            raise SystemExit(f"{path}: {data.shape[1]} columns, {len(header)} names")
        return {h: data[:, i] for i, h in enumerate(header)}

    def counters(self):
        return self._table("bh_counters.txt")

    def gauss(self):
        cols = self._table("bh_gauss.txt")
        first = open(self.dir / "bh_gauss.txt").readline()
        cols["kappa"] = float(re.search(r"kappa ([-0-9.eE+]+)", first).group(1))
        cols["passes"] = int(re.search(r"filter passes (\d+)", first).group(1))
        return cols


def threshold(eps0, mass_ratio):
    """Threshold Lorentz factor and kinetic energy for a photon of energy eps0."""
    gth = 1.0 / eps0 + eps0 / 4.0
    return gth, mass_ratio * (gth - 1.0)


def verdict(ok):
    return "PASS" if ok else "FAIL"
