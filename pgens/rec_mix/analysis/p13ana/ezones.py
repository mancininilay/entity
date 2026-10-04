"""E-zone finder (SM S8-S9; P1.3 tech spec III.6, D31).

Zones are contiguous runs of pixels with |E_x3| > |B_x2| along a line in the
outflow direction (Entity x1). Lengths are measured between outer pixel edges
(n_pixels * pixel_size). Pixels within `trim` cells of either end are removed
before the search.

Open point (A1): the p1.1 finder `compute_ezone_lengths.py` was not provided;
`edge_policy` makes the treatment of zones touching the trimmed ends explicit:
"cut" keeps them truncated at the cut, "drop" discards them.
"""

from __future__ import annotations

import numpy as np


def zones_1d(cond: np.ndarray, pixel: float = 1.0, edge_policy: str = "cut") -> np.ndarray:
    c = np.asarray(cond, bool).astype(np.int8)
    if c.size == 0:
        return np.zeros(0)
    d = np.diff(np.concatenate(([0], c, [0])))
    starts = np.nonzero(d == 1)[0]
    stops = np.nonzero(d == -1)[0]
    L = (stops - starts) * pixel
    if edge_policy == "drop":
        keep = (starts > 0) & (stops < c.size)
        L = L[keep]
    elif edge_policy != "cut":
        raise ValueError(edge_policy)
    return L.astype(float)


def line_zones(e3, b2, trim_cells: float, pixel: float = 1.0, criterion: str = "EzBy",
               e2=None, b2sq=None, edge_policy: str = "cut") -> np.ndarray:
    """Zone lengths (cells) on one line. e3, b2: field samples with spacing `pixel`.
    criterion "EzBy": |E3| > |B2|; "E2B2": |E|^2 > |B|^2 (pass e2 = |E|^2, b2sq = |B|^2)."""
    e3 = np.asarray(e3, float)
    if criterion == "EzBy":
        cond = np.abs(e3) > np.abs(np.asarray(b2, float))
    elif criterion == "E2B2":
        cond = np.asarray(e2, float) > np.asarray(b2sq, float)
    else:
        raise ValueError(criterion)
    nt = int(round(trim_cells / pixel))
    if nt > 0:
        cond = cond[nt:-nt]
    return zones_1d(cond, pixel, edge_policy)


def midline_zones(ml, trim_cells: float = 160.0, rows=None, criterion="EzBy", edge_policy="cut"):
    """Primary P1.3 measurement on a rec_mix midline output: the two cell-centred
    rows adjacent to the midplane (j0 - 1 and j0) by default. Returns a dict of
    per-row zone-length arrays."""
    j0 = ml.j_first + ml.rows.shape[0] // 2
    if rows is None:
        rows = [j0 - 1, j0]
    out = {}
    for j in rows:
        r = ml.rows[j - ml.j_first]
        E1, E2, E3, B1, B2, B3 = r
        if criterion == "EzBy":
            out[j] = line_zones(E3, B2, trim_cells, 1.0, "EzBy", edge_policy=edge_policy)
        else:
            out[j] = line_zones(None, None, trim_cells, 1.0, "E2B2",
                                e2=E1**2 + E2**2 + E3**2, b2sq=B1**2 + B2**2 + B3**2,
                                edge_policy=edge_policy)
    return out


def node_line_zones(ml, trim_cells: float = 160.0, edge_policy="cut"):
    """Sensitivity: the exact midplane node line x2 = y0. E3 lives on nodes (i, j0)
    and B2 on (i + 1/2, j0); B2 is averaged to the nodes (periodic wrap ignored
    at the ends, which are trimmed anyway)."""
    b2n = 0.5 * (ml.b2_node + np.roll(ml.b2_node, 1))
    return line_zones(ml.e3_node, b2n, trim_cells, 1.0, "EzBy", edge_policy=edge_policy)


def subsample(line, stride: int = 8, offset: int = 0):
    """Continuity protocol: emulate a downsampled field output (sampling, not
    averaging) with the given stride and offset."""
    return np.asarray(line)[offset::stride]


def snapshot_mean(lengths_per_output) -> tuple[float, int]:
    """Mean of the per-output mean zone lengths (SM S9); outputs without zones
    are skipped. Returns (mean, number of outputs used)."""
    m = [np.mean(L) for L in lengths_per_output if len(L) > 0]
    return (float(np.mean(m)) if m else np.nan, len(m))


def de_cells(dup_cells: float, q: float, mu: float) -> float:
    """Lepton skin depth in cells, d_e = d_up (1 + q/(mu (2 - q)))^(1/2) (SM S3).
    Note: run_ezone_suite_trimmed.py hard-codes 5 * sqrt(1 + q/(mu (2 - q)))."""
    return dup_cells * np.sqrt(1.0 + (q / (mu * (2.0 - q)) if q > 0 else 0.0))
