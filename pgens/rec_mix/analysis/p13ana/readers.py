"""Readers for the rec_mix user outputs (Entity) and the p1.1 Tristan user spectra.

Entity rec_mix files live in ``<simulation.name>/usr/``:

* ``recmix.meta.json``   run description (species, roles, bins, moment names, CV).
* ``recmix.<step>.bin``  masked spectra and moments (one file per user output).
* ``cvbudget.bin``       control-volume records, appended (restarts append a
                         segment-start record).
* ``midline.<step>.bin`` full-resolution midplane rows (cell centres) and the
                         node line at x2 = y0.

* ``segments.jsonl``     one line per process start: run_id, start_label (the
                         label of the restored state; 0 for a fresh run).
* ``fields_manifest.jsonl`` (label, run_id) of every Entity field output that
                         contained the rec_mix custom fields.

All binary files are little-endian. Layouts (see pgens/rec_mix/README.md):

recmix v2: char[8] "P13RMX02" | int64 step | float64 time | uint64 run_id |
         int32 nspec, nbins, nmom, 0 | float64 ncells_masked, ncells_total |
         float64 hist[nspec][nbins] | float64 moms[nspec][nmom] |
         float64 fill[nspec] (max over ranks of npart/maxnpart)
         (v1 "P13RMX01": no run_id, no fill)
cvbudget record (float64): step, time, segment_start, run_id (0 before format 3),
         per species (Ncv, Icv, inE, outE, inN, outN), U_EM, int(Poynting) dt,
         int(field-side current work) dt (format 2+)
vdump (validation only, setup.validation_dump; one file per rank,
         vdump.<label>.r<rank>.bin): char[8] "P13VDP01" | int64 label | float64 time |
         uint64 run_id | int32 rank, nspec, sizeof(dx), sizeof(real), off1, off2, n1, n2,
         domain index, number of ranks |
         per species: int64 n | int32 has_ids, 0 | int32 i1[n], i2[n] (global cells) |
         dx dx1[n], dx2[n] | real ux1[n], ux2[n], ux3[n], w[n] | uint64 cv[n] |
         (has_ids) uint64 idx[n], dom[n]. The exact particle state of the post-step
         (cell index and float offset), unlike the rounded positions X = i + dx of
         Entity's particle output (Codex A0-2).
midline v2: char[8] "P13MID02" | int64 step | float64 time | uint64 run_id |
         int32 N1, nrows, ncomp, j_first | float32 rows[nrows][ncomp][N1] |
         float32 E3_node[N1] | float32 B2_node[N1]   (v1 "P13MID01": no run_id)

Restart provenance (Codex IC3). A file (or CV record) written by segment k at
label s is valid iff s <= start_label of every later segment. This rejects the
"old future" of a run resumed from an earlier checkpoint (also before the new
segment has caught up, and after a cadence change). Files without a run id
(format 1) are accepted only when the directory has a single segment or no
manifest, with a warning. Filenames and modification times are not used.
"""

from __future__ import annotations

import glob
import json
import os
import re
import warnings
from dataclasses import dataclass, field

import numpy as np

NSPEC = 9


# --------------------------------------------------------------------------
# Entity rec_mix
# --------------------------------------------------------------------------
def read_meta(usr_dir: str) -> dict:
    with open(os.path.join(usr_dir, "recmix.meta.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Restart provenance
# --------------------------------------------------------------------------
def read_segments(usr_dir: str) -> list[dict]:
    """Segments in the order they started (append order of segments.jsonl)."""
    path = os.path.join(usr_dir, "segments.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Provenance:
    """Validity of (run_id, label) pairs under the restart rule."""

    def __init__(self, segments: list[dict]):
        self.segments = segments
        self.index = {int(sg["run_id"]): k for k, sg in enumerate(segments)}
        self.starts = [int(sg["start_label"]) for sg in segments]

    def valid(self, run_id: int | None, label: int) -> bool:
        if not self.segments:
            return True  # no manifest: legacy single-run directory
        if run_id is None or run_id == 0:
            if len(self.segments) == 1:
                return True
            return False
        k = self.index.get(int(run_id))
        if k is None:
            return False  # written by a process that is not in the manifest
        return all(label <= st for st in self.starts[k + 1:])


def provenance(usr_dir: str) -> Provenance:
    segs = read_segments(usr_dir)
    if not segs:
        warnings.warn(f"{usr_dir}: no segments.jsonl; all files accepted (legacy output)")
    return Provenance(segs)


def bin_centres(meta: dict) -> np.ndarray:
    """Geometric bin centres of the log energy bins, as Tristan's 'ebins'."""
    sp = meta["spectra"]
    lmin, lmax, n = np.log(sp["min"]), np.log(sp["max"]), int(sp["num"])
    dl = (lmax - lmin) / n
    return np.exp(lmin + (np.arange(1, n + 1) - 0.5) * dl)


@dataclass
class RecmixOutput:
    step: int
    time: float
    ncells_masked: float
    ncells_total: float
    hist: np.ndarray          # [nspec, nbins]
    moms: np.ndarray          # [nspec, nmom]
    names: list = field(default_factory=list)
    run_id: int | None = None
    fill: np.ndarray | None = None  # max over ranks of npart/maxnpart

    def m(self, name: str) -> np.ndarray:
        """Moment `name` for all species (array of length nspec)."""
        return self.moms[:, self.names.index(name)]


def read_recmix(path: str, names: list | None = None) -> RecmixOutput:
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in (b"P13RMX01", b"P13RMX02"):
            raise ValueError(f"{path}: bad magic {magic!r}")
        v2 = magic == b"P13RMX02"
        step = int(np.frombuffer(f.read(8), "<i8")[0])
        time = float(np.frombuffer(f.read(8), "<f8")[0])
        run_id = int(np.frombuffer(f.read(8), "<u8")[0]) if v2 else None
        nspec, nbins, nmom, _ = np.frombuffer(f.read(16), "<i4")
        ncm, nct = np.frombuffer(f.read(16), "<f8")
        hist = np.frombuffer(f.read(8 * nspec * nbins), "<f8").reshape(nspec, nbins)
        moms = np.frombuffer(f.read(8 * nspec * nmom), "<f8").reshape(nspec, nmom)
        fill = np.frombuffer(f.read(8 * nspec), "<f8").copy() if v2 else None
    if names is None:
        names = [f"m{k}" for k in range(nmom)]
    return RecmixOutput(step, time, float(ncm), float(nct), hist.copy(), moms.copy(),
                        list(names), run_id, fill)


def _step_of(path: str) -> int:
    m = re.search(r"\.(\d+)\.bin$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_recmix(usr_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(usr_dir, "recmix.*.bin")), key=_step_of)


def read_all_recmix(usr_dir: str) -> tuple[dict, list[RecmixOutput]]:
    """All valid user outputs, in label order (stale files of an abandoned
    future are dropped by the provenance rule)."""
    meta = read_meta(usr_dir)
    prov = provenance(usr_dir)
    outs = []
    for p in list_recmix(usr_dir):
        o = read_recmix(p, meta["moments"])
        if prov.valid(o.run_id, o.step):
            outs.append(o)
    return meta, outs


def list_vdump_labels(usr_dir: str) -> list[int]:
    labs = set()
    for p in glob.glob(os.path.join(usr_dir, "vdump.*.r*.bin")):
        m = re.search(r"vdump\.(\d+)\.r\d+\.bin$", os.path.basename(p))
        if m:
            labs.add(int(m.group(1)))
    return sorted(labs)


def read_vdump(usr_dir: str, label: int, check_provenance: bool = True) -> dict:
    """Exact particle state at a user-output label, all ranks concatenated.
    Refuses an incomplete set (number of files against the ranks recorded in
    every header, and the rank tiles against the global grid of the meta).
    Returns {"label", "time", "run_id", "ranks": [{rank, dom, off1, off2, n1, n2}],
    "species": {s: {"i1", "i2" (int64 global cells), "dx1", "dx2" (float32),
    "u1", "u2", "u3", "w" (float32), "cv", "idx", "dom" (uint64), "rank" (int)}}}.
    Raises if the files disagree on run_id, or (check_provenance) if the label
    belongs to an abandoned future (IC3 rule)."""
    paths = sorted(glob.glob(os.path.join(usr_dir, f"vdump.{label:08d}.r*.bin")))
    if not paths:
        raise FileNotFoundError(f"no vdump files for label {label} in {usr_dir}")
    out = {"label": label, "ranks": [], "species": {}}
    parts = {s: [] for s in range(1, NSPEC + 1)}
    rid = None
    for p in paths:
        b = open(p, "rb").read()
        if b[:8] != b"P13VDP01":
            raise ValueError(f"{p}: not a P13VDP01 file")
        lab, tm, run_id = np.frombuffer(b, "<i8", 1, 8)[0], np.frombuffer(b, "<f8", 1, 16)[0], \
            int(np.frombuffer(b, "<u8", 1, 24)[0])
        hdr = np.frombuffer(b, "<i4", 10, 32)
        rank, nsp, sdx, sre, o1, o2, n1, n2, dom, nranks = (int(x) for x in hdr)
        if nranks != len(paths):
            raise ValueError(f"{p}: {nranks} ranks recorded, {len(paths)} files found (incomplete dump)")
        if lab != label or nsp != NSPEC:
            raise ValueError(f"{p}: label {lab} or nspec {nsp} unexpected")
        if rid is None:
            rid, out["time"], out["run_id"] = run_id, float(tm), run_id
        elif run_id != rid:
            raise ValueError(f"{p}: run_id {run_id} differs from {rid} (mixed writers)")
        out["ranks"].append({"rank": rank, "dom": dom, "off1": o1, "off2": o2, "n1": n1, "n2": n2})
        fdx = {4: "<f4", 8: "<f8"}[sdx]
        fre = {4: "<f4", 8: "<f8"}[sre]
        pos = 72
        for s in range(1, NSPEC + 1):
            n = int(np.frombuffer(b, "<i8", 1, pos)[0])
            has = int(np.frombuffer(b, "<i4", 1, pos + 8)[0])
            pos += 16
            d = {}
            for k, dt, sz in (("i1", "<i4", 4), ("i2", "<i4", 4), ("dx1", fdx, sdx), ("dx2", fdx, sdx),
                              ("u1", fre, sre), ("u2", fre, sre), ("u3", fre, sre), ("w", fre, sre),
                              ("cv", "<u8", 8)):
                d[k] = np.frombuffer(b, dt, n, pos).copy()
                pos += n * sz
            if has:
                for k in ("idx", "dom"):
                    d[k] = np.frombuffer(b, "<u8", n, pos).copy()
                    pos += n * 8
            d["rank"] = np.full(n, rank, dtype=np.int64)
            d["dom_now"] = np.full(n, dom, dtype=np.int64)
            parts[s].append(d)
        if pos != len(b):
            raise ValueError(f"{p}: {len(b) - pos} trailing bytes")
    try:
        d = read_meta(usr_dir)["derived"]
        area = sum(rk["n1"] * rk["n2"] for rk in out["ranks"])
        if area != d["N1"] * d["N2"] or len({rk["rank"] for rk in out["ranks"]}) != len(paths):
            raise ValueError(f"label {label}: rank tiles cover {area} of {d['N1'] * d['N2']} cells")
    except FileNotFoundError:
        pass
    if check_provenance and not provenance(usr_dir).valid(rid, label):
        raise ValueError(f"label {label} in {usr_dir} is not valid under the restart provenance")
    for s, lst in parts.items():
        keys = set().union(*[x.keys() for x in lst])
        out["species"][s] = {k: np.concatenate([x[k] for x in lst if k in x]) for k in keys}
        for k in ("i1", "i2"):
            out["species"][s][k] = out["species"][s][k].astype(np.int64)
    return out


def read_cvbudget(usr_dir: str, nspec: int = NSPEC, meta: dict | None = None) -> dict:
    """Return the CV records as arrays.

    The record size comes from the meta (format_version 1: 60 doubles, 2: 61).
    Restarts append a segment-start record; when a segment starts at step s,
    every earlier record with step >= s is stale (a resume from an older
    checkpoint) and is dropped."""
    path = os.path.join(usr_dir, "cvbudget.bin")
    if meta is None:
        meta = read_meta(usr_dir)
    nrec = int(meta.get("cv", {}).get("record_doubles", 4 + 6 * nspec + 2))
    raw = np.fromfile(path, "<f8")
    if raw.size % nrec:
        raise ValueError(f"{path}: size {raw.size} not a multiple of {nrec}")
    r = raw.reshape(-1, nrec)
    keep = np.ones(len(r), bool)
    for k in range(len(r)):
        if r[k, 2] > 0.5:  # segment start at step s: drop earlier records with step >= s
            keep[:k] &= ~(r[:k, 0] >= r[k, 0])
    prov = provenance(usr_dir)
    if prov.segments:  # format 3: the run id is in the fourth entry
        keep &= np.array([prov.valid(int(rid) if rid > 0 else None, int(st))
                          for st, rid in zip(r[:, 0], r[:, 3])], bool)
    r = r[keep]
    sp = r[:, 4:4 + 6 * nspec].reshape(-1, nspec, 6)
    tail = r[:, 4 + 6 * nspec:]
    out = {
        "step": r[:, 0].astype(np.int64),
        "time": r[:, 1],
        "segment_start": r[:, 2].astype(bool),
        "Ncv": sp[:, :, 0], "Icv": sp[:, :, 1],
        "inE": sp[:, :, 2], "outE": sp[:, :, 3],
        "inN": sp[:, :, 4], "outN": sp[:, :, 5],
        "U_EM": tail[:, 0], "flux_int": tail[:, 1],
    }
    if tail.shape[1] > 2:
        out["jwork_int"] = tail[:, 2]
    return out


@dataclass
class Midline:
    step: int
    time: float
    j_first: int
    rows: np.ndarray      # [nrows, ncomp, N1], components E1 E2 E3 B1 B2 B3
    e3_node: np.ndarray   # [N1] E3 at nodes (i, j0)
    b2_node: np.ndarray   # [N1] B2 at (i + 1/2, j0)
    run_id: int | None = None


def _midline_header(f):
    magic = f.read(8)
    if magic not in (b"P13MID01", b"P13MID02"):
        raise ValueError(f"bad magic {magic!r}")
    step = int(np.frombuffer(f.read(8), "<i8")[0])
    time = float(np.frombuffer(f.read(8), "<f8")[0])
    run_id = int(np.frombuffer(f.read(8), "<u8")[0]) if magic == b"P13MID02" else None
    return step, time, run_id


def read_midline(path: str) -> Midline:
    with open(path, "rb") as f:
        step, time, run_id = _midline_header(f)
        n1, nrows, ncomp, jf = (int(x) for x in np.frombuffer(f.read(16), "<i4"))
        rows = np.frombuffer(f.read(4 * nrows * ncomp * n1), "<f4").reshape(nrows, ncomp, n1)
        e3 = np.frombuffer(f.read(4 * n1), "<f4")
        b2 = np.frombuffer(f.read(4 * n1), "<f4")
    return Midline(step, time, jf, rows.copy(), e3.copy(), b2.copy(), run_id)


def list_midlines(usr_dir: str) -> list[str]:
    """Valid midline files in label order (provenance rule)."""
    prov = provenance(usr_dir)
    out = []
    for p in sorted(glob.glob(os.path.join(usr_dir, "midline.*.bin")), key=_step_of):
        with open(p, "rb") as f:
            step, _, run_id = _midline_header(f)
        if prov.valid(run_id, step):
            out.append(p)
    return out


# --------------------------------------------------------------------------
# Tristan p1.1 user spectra (output/usr/reconn_spec.NNNNN), as extractfi.py
# --------------------------------------------------------------------------
def tristan_spec_index(path: str) -> int:
    m = re.search(r"reconn_spec\.(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def list_tristan_spectra(usr_dir: str) -> list[str]:
    fs = glob.glob(os.path.join(usr_dir, "reconn_spec.*"))
    return sorted([p for p in fs if tristan_spec_index(p) >= 0], key=tristan_spec_index)


def read_tristan_spectra(path: str) -> dict:
    """Return {'ebins': ..., 'counts': {s: counts summed over space}}."""
    import h5py

    out = {"counts": {}}
    with h5py.File(path, "r") as f:
        fmt = "n%03d" if "n004" in f.keys() else "n%d"
        out["ebins"] = f["ebins"][:].astype(np.float64)
        for s in range(1, NSPEC + 1):
            key = fmt % s
            if key in f:
                out["counts"][s] = np.sum(f[key][:], axis=(1, 2, 3)).astype(np.float64)
    return out


# --------------------------------------------------------------------------
# Entity field output (BP files), arrays indexed [i1, i2] (x1 = outflow)
# --------------------------------------------------------------------------
def list_entity_fields(sim_dir: str, allow_unmanifested: bool = False) -> list[str]:
    """Valid Entity field files in label order. Entity's files carry no writer
    id; rec_mix records (label, run_id) in usr/fields_manifest.jsonl whenever a
    field output with its custom fields is written. The last manifest entry for
    a label identifies the writer of the file now on disk, and the provenance
    rule applies. With more than one segment, files without a manifest entry
    are refused (their writer is unknown). A single segment cannot hold stale
    files, so there unmanifested files are accepted (with a warning when the
    manifest is missing altogether, unless `allow_unmanifested`).
    The manifest is written only when the field output includes a rec_mix
    custom field (make_input.py always requests `mixing` and `mask`). Entity's
    particle, spectra and stats files are not covered: after a rollback
    restart, move the later ones aside before reading them."""
    fs = glob.glob(os.path.join(sim_dir, "fields", "fields.*.bp"))
    fs = sorted(fs, key=lambda p: int(re.search(r"fields\.(\d+)\.bp", p).group(1)))
    usr = os.path.join(sim_dir, "usr")
    prov = provenance(usr)
    writer = {}
    man = os.path.join(usr, "fields_manifest.jsonl")
    if os.path.exists(man):
        with open(man) as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    writer[int(e["label"])] = int(e["run_id"])
    out = []
    for p in fs:
        lab = int(re.search(r"fields\.(\d+)\.bp", p).group(1))
        if lab in writer:
            if prov.valid(writer[lab], lab):
                out.append(p)
        elif (allow_unmanifested or not writer) and len(prov.segments) <= 1:
            # a single segment cannot have stale files; without any manifest
            # (no rec_mix custom field in the output, or a legacy run) say so
            if not writer and not allow_unmanifested:
                warnings.warn(f"{p}: no fields manifest (single segment); accepted")
            out.append(p)
    return out


def read_entity_fields(path: str, names: list[str] | None = None) -> dict:
    """Read an Entity field output. Keys are the stored names without the leading
    'f' (e.g. 'E3', 'N_4_5_6', 'V2_6', 'mixing'), plus 'X1', 'X2', 'step', 'time'."""
    from adios2 import FileReader

    out = {}
    with FileReader(path) as r:
        avail = r.available_variables()
        if names is None:
            names = [k[1:] for k in avail if k.startswith("f")]
        for n in names:
            out[n] = np.asarray(r.read("f" + n))
        out["X1"] = np.asarray(r.read("X1"))
        out["X2"] = np.asarray(r.read("X2"))
        out["step"] = int(np.asarray(r.read("Step")))
        out["time"] = float(np.asarray(r.read("Time")))
    return out
