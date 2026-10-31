"""Reduces one demonstration run directory (P2 step 8) to a compact reduced.npz + reduced.json, then (with --delete)
removes the large text dumps (hadronic_ions_1.txt, hadronic_pairs.txt). Analysis code, not a ctest.

Kept: per ledger row the 51 ledger columns; per per-ion block (one per ledger row): n_alive, mean/var of the stored K and
of K_eff = K - delta (delta = pending payload), mean delta, mean debt r, K quantiles, and the full K arrays at three
blocks (<K>/K0 nearest 0.9, 0.7 and the last block); from the pair dump: Gamma_p-binned sums (count, sum w, sum w^2,
sum w D, sum w D^2, sum D, sum D^2, sum w (E- + E+)), per-quantity histograms of E-, E+, D on the frozen bhref 0.2.0 bin
edges (counts, sum w, sum w^2) for the pairs whose parent Gamma_p lies within 0.5 % below the run's Gamma_p0, the
w_pair histogram (log bins), totals; from run.log the closure, array-identity and ledger lines.
Usage: reduce_run.py <run dir> <Gamma_p0> [--refdir tests/data/hadronic/bhref-0.2.0] [--delete]"""
import json, os, re, sys, time
import numpy as np

MU_R = 1836.15
GAMMA_TH = 2.125
# Gamma_p bins for figure 4: uniform in log10(Gamma - Gamma_th) from 2.14 to 10.5
G_EDGES = GAMMA_TH + np.logspace(np.log10(2.14 - GAMMA_TH), np.log10(10.5 - GAMMA_TH), 61)
W_EDGES = np.logspace(-3, 4, 141)

def parse_ledger(path):
    rows = [list(map(float, l.split())) for l in open(path) if l.strip() and not l.startswith('#')]
    return np.array(rows) if rows else np.zeros((0, 51))

def parse_log(path):
    out = {}
    txt = open(path, errors='replace').read()
    m = re.search(r'closure species 1 : sum w \(K_init - K_final\) = (\S+), E_drag = (\S+), difference = (\S+), relative = (\S+)', txt)
    if m: out['closure'] = dict(dK=float(m[1].rstrip(',')), E_drag=float(m[2].rstrip(',')), diff=float(m[3].rstrip(',')), rel=float(m[4]))
    m = re.search(r'array identity species 1 : sum_pairs w_s \(E- \+ E\+ - eps_s\) = (\S+) \((\d+) kernel-born pairs.*?rhs = .*? = (\S+), residual = (\S+), E_round \(ledger\) = (\S+), residual - E_round = (\S+), B_budget = (\S+), \|residual\|/B = (\S+)', txt)
    if m: out['array_identity'] = dict(pairs_net=float(m[1]), n_pairs=int(m[2]), rhs=float(m[3].rstrip(',')), residual=float(m[4].rstrip(',')),
                                       E_round=float(m[5].rstrip(',')), res_minus_Eround=float(m[6].rstrip(',')), B=float(m[7].rstrip(',')), rel_B=float(m[8]))
    m = re.search(r'bh_proto: ledger species 1 : (.*)', txt)
    if m:
        toks = m[1].split(); d = {}; i = 0
        while i < len(toks) - 1:
            k = toks[i]
            if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k):
                vals = []; j = i + 1
                while j < len(toks) and not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', toks[j]):
                    vals.append(float(toks[j])); j += 1
                d[k] = vals[0] if len(vals) == 1 else vals; i = j
            else:
                i += 1
        out['ledger_final'] = d
    m = re.search(r'restframe sampler species 1 : (.*)', txt)
    if m: out['sampler_line'] = m[1][:400]
    m = re.search(r'bh_proto: real_t = (\w+)', txt)
    if m: out['real_t'] = m[1]
    m = re.search(r'qk_source = (\w+)', txt)
    if m: out['qk_source'] = m[1]
    out['checksums'] = re.findall(r'bh_proto: checksum (.*)', txt)
    return out

def reduce_ions(path, K0):
    blocks = []; keepK = {}; cur = None; buf = []
    def flush():
        if cur is None or not buf: return
        a = np.array(b' '.join(buf).split(), dtype=np.float64).reshape(-1, 11)
        alive = a[:, 8] == 1
        K = a[alive, 3]; d = a[alive, 7]; r = a[alive, 9]; Ke = K - d
        q = np.quantile(K, [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
        blocks.append([cur[0], cur[1], K.size, K.mean(), K.var(ddof=1), Ke.mean(), Ke.var(ddof=1), d.mean(), r.mean(), *q])
        keepK[len(blocks) - 1] = K.astype(np.float64)
    with open(path, 'rb') as f:
        for line in f:
            if line.startswith(b'# step'):
                flush(); buf = []
                t = line.split(); cur = (float(t[2]), float(t[4]))
            elif line.startswith(b'#'):
                continue
            else:
                buf.append(line)
    flush()
    B = np.array(blocks)
    mk = B[:, 3] / K0
    sel = sorted({int(np.argmin(np.abs(mk - 0.9))), int(np.argmin(np.abs(mk - 0.7))), len(B) - 1})
    return B, {i: keepK[i] for i in sel}

def ref_edges(refdir, G0):
    if refdir is None: return None
    tag = {3.0: '3', 10.0: '10'}.get(G0)
    if tag is None: return None
    p = os.path.join(refdir, f'hist_mono05_G{tag}.txt')
    rows = np.array([list(map(float, l.split())) for l in open(p) if l.strip() and not l.startswith('#')])
    ed = {}
    for q, name in ((0, 'Eminus'), (1, 'Eplus'), (4, 'D')):
        r = rows[rows[:, 0] == q]
        ed[name] = np.concatenate([r[:, 2], r[-1:, 3]])
    return ed

def reduce_pairs(path, G0, edges):
    nG = len(G_EDGES) - 1
    acc = {k: np.zeros(nG) for k in ('n', 'w', 'w2', 'wD', 'wD2', 'D', 'D2', 'wE', 'weps')}
    wh = np.zeros(len(W_EDGES) - 1); w_lo = w_hi = 0
    hist = {}
    if edges:
        for k, e in edges.items():
            hist[k] = {'edges': e, 'n': np.zeros(len(e) - 1), 'w': np.zeros(len(e) - 1), 'w2': np.zeros(len(e) - 1)}
    tot = dict(n=0, w=0.0, wD=0.0, wE=0.0, weps=0.0, n_window=0, w_window=0.0, wmin=np.inf, wmax=0.0, Gp_min=np.inf, Gp_max=0.0)
    win_lo = G0 * (1 - 0.005)
    with open(path, 'rb') as f:
        hdr = [f.readline() for _ in range(3)]
        rest = b''
        while True:
            buf = f.read(256 << 20)
            if not buf and not rest: break
            buf = rest + buf
            cut = buf.rfind(b'\n') + 1
            if cut == 0: break
            rest = buf[cut:]
            a = np.array(buf[:cut].split(), dtype=np.float64).reshape(-1, 25)
            born = (a[:, 15] > 0) & (a[:, 13] == 1) & (a[:, 14] == 1)
            a = a[born]
            if a.size == 0:
                if not buf[cut:] and len(buf) < (256 << 20): pass
                continue
            w = a[:, 5]; gm = np.sqrt(1 + a[:, 7]**2 + a[:, 8]**2 + a[:, 9]**2); gp = np.sqrt(1 + a[:, 10]**2 + a[:, 11]**2 + a[:, 12]**2)
            eps = a[:, 17]; Gp = a[:, 16]; D = gm + gp - eps
            idx = np.searchsorted(G_EDGES, Gp, side='right') - 1; ok = (idx >= 0) & (idx < nG)
            for k, v in (('n', np.ones_like(w)), ('w', w), ('w2', w*w), ('wD', w*D), ('wD2', w*D*D), ('D', D), ('D2', D*D), ('wE', w*(gm+gp)), ('weps', w*eps)):
                acc[k] += np.bincount(idx[ok], weights=v[ok], minlength=nG)
            wh += np.histogram(w, bins=W_EDGES)[0]; w_lo += int((w < W_EDGES[0]).sum()); w_hi += int((w >= W_EDGES[-1]).sum())
            tot['n'] += w.size; tot['w'] += w.sum(); tot['wD'] += (w*D).sum(); tot['wE'] += (w*(gm+gp)).sum(); tot['weps'] += (w*eps).sum()
            tot['wmin'] = min(tot['wmin'], w.min()); tot['wmax'] = max(tot['wmax'], w.max()); tot['Gp_min'] = min(tot['Gp_min'], Gp.min()); tot['Gp_max'] = max(tot['Gp_max'], Gp.max())
            win = Gp >= win_lo
            tot['n_window'] += int(win.sum()); tot['w_window'] += float(w[win].sum())
            for k, x in (('Eminus', gm), ('Eplus', gp), ('D', D)):
                if k in hist:
                    h = hist[k]
                    h['n'] += np.histogram(x[win], bins=h['edges'])[0]
                    h['w'] += np.histogram(x[win], bins=h['edges'], weights=w[win])[0]
                    h['w2'] += np.histogram(x[win], bins=h['edges'], weights=w[win]**2)[0]
            if not buf[cut:] and len(buf) - len(rest) < (256 << 20) and not rest:
                pass
    return acc, wh, (w_lo, w_hi), hist, tot

def main():
    rd = sys.argv[1]; G0 = float(sys.argv[2])
    refdir = None; delete = '--delete' in sys.argv
    if '--refdir' in sys.argv: refdir = sys.argv[sys.argv.index('--refdir') + 1]
    t0 = time.time()
    K0 = MU_R * (G0 - 1.0)
    out = {'G0': G0, 'K0': K0, 'run': os.path.basename(rd.rstrip('/'))}
    npz = {}
    npz['ledger'] = parse_ledger(os.path.join(rd, 'hadronic_ledger_1.txt'))
    out['log'] = parse_log(os.path.join(rd, 'run.log'))
    ions = os.path.join(rd, 'hadronic_ions_1.txt')
    if os.path.exists(ions):
        B, keepK = reduce_ions(ions, K0)
        npz['blocks'] = B
        for i, K in keepK.items(): npz[f'K_block{i}'] = K
        out['blocks_columns'] = ['step', 'time', 'n_alive', 'meanK', 'varK', 'meanKeff', 'varKeff', 'mean_delta', 'mean_r',
                                 'q0.001', 'q0.01', 'q0.05', 'q0.25', 'q0.5', 'q0.75', 'q0.95', 'q0.99', 'q0.999']
        out['kept_blocks'] = sorted(keepK)
    pairs = os.path.join(rd, 'hadronic_pairs.txt')
    if os.path.exists(pairs) and os.path.getsize(pairs) > 1000:
        acc, wh, wout, hist, tot = reduce_pairs(pairs, G0, ref_edges(refdir, G0))
        npz['G_edges'] = G_EDGES
        for k, v in acc.items(): npz['Gbin_' + k] = v
        npz['W_edges'] = W_EDGES; npz['W_hist'] = wh
        out['w_outside_hist'] = list(wout)
        for k, h in hist.items():
            npz[f'hist_{k}_edges'] = h['edges']; npz[f'hist_{k}_n'] = h['n']; npz[f'hist_{k}_w'] = h['w']; npz[f'hist_{k}_w2'] = h['w2']
        out['pairs'] = {k: (float(v) if not isinstance(v, int) else v) for k, v in tot.items()}
    np.savez_compressed(os.path.join(rd, 'reduced.npz'), **npz)
    out['reduce_seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(rd, 'reduced.json'), 'w'), indent=1, default=float)
    if delete:
        for p in (ions, pairs):
            if os.path.exists(p): os.remove(p)
    print(f"reduced {rd}: {out['reduce_seconds']:.0f} s; blocks {len(npz.get('blocks', []))}; pairs {out.get('pairs', {}).get('n', 0)}")

if __name__ == '__main__':
    main()
