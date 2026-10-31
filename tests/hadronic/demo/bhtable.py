"""Born loss table of a bh_proto run (hadronic_table.txt) with the kernel's own interpolation (bh_interpolate.h):
J, I, Dmean log-linear in y = log10(Gamma - Gamma_th); eps_mean linear in y; 0 below the table, the last node above."""
import json, math
import numpy as np

MU_R = 1836.15

class Table:
    def __init__(self, path):
        hdr, rows = {}, []
        for line in open(path):
            if line.startswith('#'):
                k, _, v = line[1:].partition(':')
                hdr[k.strip()] = v.strip()
            elif line.strip():
                rows.append([float(x) for x in line.split()])
        a = np.array(rows)
        cols = json.loads(hdr['columns'])
        self.hdr = hdr
        self.gamma_th = float(hdr['gamma_th']); self.gamma_min = float(hdr['gamma_min']); self.gamma_max = float(hdr['gamma_max'])
        self.y_min = float(hdr['y_min']); self.y_max = float(hdr['y_max'])
        self.gamma = a[:, cols.index('gamma_p')]; self.y = a[:, cols.index('y')]
        self.J = a[:, cols.index('J')]; self.I = a[:, cols.index('I')]; self.D = a[:, cols.index('Dmean')]
        self.eps = a[:, cols.index('eps_mean')]
        self.qk_source = hdr.get('qk_source', '"chodorowski-3.7"').strip('"')
        self.sha = hdr.get('data_sha256', '').strip('"')

    def _loglin(self, v, G):
        G = np.asarray(G, dtype=float)
        out = np.zeros_like(G)
        ok = (G > self.gamma_th) & (G >= self.gamma_min)
        yy = np.where(ok, np.log10(np.maximum(G - self.gamma_th, 1e-300)), self.y_min - 1.0)
        top = ok & ((G >= self.gamma_max) | (yy >= self.y_max))
        mid = ok & ~top & (yy >= self.y_min)
        out[top] = v[-1]
        if mid.any():
            hi = np.searchsorted(self.y, yy[mid], side='right'); hi = np.clip(hi, 1, len(self.y) - 1); lo = hi - 1
            t = (yy[mid] - self.y[lo]) / (self.y[hi] - self.y[lo])
            out[mid] = np.exp(np.log(v[lo]) + t * (np.log(v[hi]) - np.log(v[lo])))
        return out

    def Jf(self, G): return self._loglin(self.J, G)
    def If(self, G): return self._loglin(self.I, G)
    def Df(self, G):
        G = np.asarray(G, dtype=float)
        below = ~((G > self.gamma_th) & (G >= self.gamma_min))
        out = self._loglin(self.D, G)
        return np.where(below, self.D[0], out)

def K_of_gamma(G, mu=MU_R): return mu * (np.asarray(G) - 1.0)
def gamma_of_K(K, mu=MU_R): return 1.0 + np.asarray(K) / mu

# scalar fast path (pure Python, same interpolation) for the reference integrations
import bisect as _bisect
def _scalar_setup(self):
    self._ys = self.y.tolist(); self._lnI = np.log(self.I).tolist(); self._lnJ = np.log(self.J).tolist(); self._lnD = np.log(self.D).tolist()
def _scalar(self, lnv, last, G, below):
    if not (G > self.gamma_th) or not (G >= self.gamma_min): return below
    yy = math.log10(G - self.gamma_th)
    if G >= self.gamma_max or yy >= self.y_max: return last
    if yy < self.y_min: return below
    hi = _bisect.bisect_right(self._ys, yy); hi = min(max(hi, 1), len(self._ys) - 1); lo = hi - 1
    t = (yy - self._ys[lo]) / (self._ys[hi] - self._ys[lo])
    return math.exp(lnv[lo] + t * (lnv[hi] - lnv[lo]))
def _If_s(self, G):
    if not hasattr(self, '_ys'): _scalar_setup(self)
    return _scalar(self, self._lnI, self.I[-1], G, 0.0)
def _Jf_s(self, G):
    if not hasattr(self, '_ys'): _scalar_setup(self)
    return _scalar(self, self._lnJ, self.J[-1], G, 0.0)
def _Df_s(self, G):
    if not hasattr(self, '_ys'): _scalar_setup(self)
    return _scalar(self, self._lnD, self.D[-1], G, float(self.D[0]))
Table.If_s = _If_s; Table.Jf_s = _Jf_s; Table.Df_s = _Df_s
