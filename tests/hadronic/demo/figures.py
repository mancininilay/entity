"""The eight demonstration figures (P2 step 8, plan §2) and numbers.md (every number shown on a figure, with its run).
Usage: figures.py <campaign output root> <figure dir> [--perf <perf_medians.csv>]
Analysis code (not a ctest). Reads the reduced runs written by run_demo.py; recomputes the references with refs.py."""
import glob, json, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bhtable import Table, MU_R, gamma_of_K
from refs import D2Fixture, plain_ode, corrected
import budget_lag

OUT = sys.argv[1]; FIG = sys.argv[2]
PERF = sys.argv[sys.argv.index('--perf') + 1] if '--perf' in sys.argv else None
os.makedirs(FIG, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
REF020 = os.path.join(HERE, '../../data/hadronic/bhref-0.2.0')
TAU0 = {6: 0.2564289235, 3: 7.862350556, 10: 0.09136836671}
N_ION = 100000
COL = {'none': '#7f7f7f', 'poisson': '#1f77b4', 'budget': '#ff7f0e', 'event': '#2ca02c'}
SPEC_COL = {'restframe': '#2ca02c', 'delta': '#9467bd'}
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 7.5, 'figure.dpi': 110,
                     'savefig.bbox': 'tight', 'axes.grid': True, 'grid.alpha': 0.25})
NUM = []   # (figure, quantity, value, run)

def num(fig, what, value, run, fmt='{:.4g}'):
    NUM.append((fig, what, fmt.format(value) if isinstance(value, (float, int, np.floating)) else str(value), run))

MANIFEST = json.load(open(os.path.join(OUT, 'manifest.json'))) if os.path.exists(os.path.join(OUT, 'manifest.json')) else {}
LOADED = {}
def load(name):
    # only runs the campaign manifest marks done (an aborted run leaves a partial reduction)
    if MANIFEST.get(name, {}).get('status') != 'done': return None
    if name in LOADED: return LOADED[name]
    d = os.path.join(OUT, name)
    if not os.path.exists(os.path.join(d, 'reduced.npz')): return None
    z = np.load(os.path.join(d, 'reduced.npz')); j = json.load(open(os.path.join(d, 'reduced.json')))
    LOADED[name] = {'name': name, 'z': z, 'j': j, 'dir': d, 'manifest': MANIFEST[name]}
    return LOADED[name]

def blocks(r):
    B = r['z']['blocks']; K0 = r['j']['K0']
    t = np.concatenate([[0.0], B[:, 1]]); mK = np.concatenate([[K0], B[:, 3]]); vK = np.concatenate([[0.0], B[:, 4]])
    mKe = np.concatenate([[K0], B[:, 5]]); n = np.concatenate([[B[0, 2]], B[:, 2]])
    return t, mK, vK, mKe, n

def save(fig, stem, caption):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'{stem}.{ext}'), dpi=200 if ext == 'png' else None)
    open(os.path.join(FIG, f'{stem}.caption.txt'), 'w').write(caption.strip() + '\n')
    plt.close(fig)

TABLES = {}
def table(prec):
    if prec not in TABLES: TABLES[prec] = Table(os.path.join(OUT, f'hadronic_table_{prec}.txt'))
    return TABLES[prec]
D2F = None
def d2fix():
    global D2F
    if D2F is None: D2F = D2Fixture(os.path.join(REF020, 'moments_mono05.txt'), table('double').gamma_th)
    return D2F

CFGS = ['none', 'poisson_delta', 'poisson_restframe', 'budget_delta', 'budget_restframe', 'event_delta', 'event_restframe']
REFCACHE = {}
PLAINCACHE = {}
def plain_cached(g, times):
    key = (g, len(times), float(times[-1]))
    if key not in PLAINCACHE: PLAINCACHE[key] = plain_ode(table('double'), TAU0[g], MU_R * (g - 1), times)
    return PLAINCACHE[key]
def event_ref(g, spec, prec, r):
    key = (g, spec, prec)
    if key not in REFCACHE:
        t, mK, vK, _, _ = blocks(r); T = table('double')
        d2 = (lambda G: d2fix().D2(G)) if spec == 'restframe' else (lambda G: T.Df(np.atleast_1d(G))**2)
        REFCACHE[key] = corrected(T, TAU0[g], r['j']['K0'], t[1:], vK[1:], d2)
    return REFCACHE[key]

# ------------------------------------------------------------------ figure 1: cooling curves (E2, T5)
def fig1():
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.2), sharex='col', gridspec_kw={'height_ratios': [2, 1.3]}, layout='constrained')
    for c, g in enumerate((6, 3)):
        targets = (0.9, 0.7, 0.5) if g == 6 else (0.9, 0.7, 0.65)
        T = table('double'); K0 = MU_R * (g - 1)
        ax, axr = axs[0, c], axs[1, c]
        ref_rf = {p: load(f'demo_g{g}_event_restframe_{p}') for p in ('single', 'double')}
        tt = None
        for prec, mk in (('double', 'o'), ('single', 'x')):
            ev_se = None
            if ref_rf[prec] is not None:
                t_, m_, v_, _, n_ = blocks(ref_rf[prec]); ev_se = np.sqrt(v_ / n_)
            for cfg in CFGS:
                r = load(f'demo_g{g}_{cfg}_{prec}')
                if r is None: continue
                t, mKs, vK, mK, n = blocks(r); tt = t   # mK: K_eff = K - delta, the energy the ledger has charged
                mode = cfg.split('_')[0]; spec = cfg.split('_')[1] if '_' in cfg else None
                if mode == 'event':
                    R = event_ref(g, spec, prec, r); kref = np.concatenate([[K0], R['K_corr']]); se = np.sqrt(vK / n)
                    lab = 'event: Jensen-corrected ODE'
                else:
                    kref = np.concatenate([[K0], plain_cached(g, t[1:])]); se = ev_se
                    lab = 'continuous: plain ODE at dt/16'
                if prec == 'double':
                    ax.plot(t / 1e3, mK / K0, color=COL[mode], ls='-' if spec != 'delta' else '--', lw=1.2 if mode != 'none' else 2.2, alpha=0.85,
                            label=f'{mode}' + (f' {spec}' if spec else ''))
                if se is None: continue
                z = (mK - kref) / np.where(se > 0, se, np.nan)
                axr.plot(t[1:] / 1e3, z[1:], mk, ms=2.5 if mk == 'o' else 3, color=COL[mode], alpha=0.7 if spec != 'delta' else 0.45,
                         mfc='none' if spec == 'delta' else COL[mode])
                for f in targets:
                    i = int(np.argmax(mK / K0 <= f)) if (mK / K0 <= f).any() else None
                    if i: num('1', f'Gamma_p0={g} {cfg} {prec}: E2 z at <K>/K0={f} (t={t[i]:g}) vs {lab}', z[i], f'demo_g{g}_{cfg}_{prec}', '{:+.2f}')
                num('1', f'Gamma_p0={g} {cfg} {prec}: max |z| over the run', np.nanmax(np.abs(z[1:])), f'demo_g{g}_{cfg}_{prec}', '{:.2f}')
                num('1', f'Gamma_p0={g} {cfg} {prec}: <K_eff>/K0 at the end (t={t[-1]:g})', mK[-1] / K0, f'demo_g{g}_{cfg}_{prec}', '{:.5f}')
                if mode != 'event':
                    num('1', f'Gamma_p0={g} {cfg} {prec}: (K_eff - plain ODE)/K0 at the end (deterministic: the explicit drag step at dt = 0.5)', (mK[-1] - kref[-1]) / K0, f'demo_g{g}_{cfg}_{prec}', '{:+.2e}')
                    num('1', f'Gamma_p0={g} {cfg} {prec}: max (stored K - K_eff)/K0 (the pending payload; all ions identical)', float(np.max(mKs - mK)) / K0, f'demo_g{g}_{cfg}_{prec}', '{:.2e}')
        if tt is not None:
            kp = plain_cached(g, tt[1:]); ax.plot(tt[1:] / 1e3, kp / K0, 'k:', lw=1, label='plain ODE (Born table)')
        for f in targets: ax.axhline(f, color='k', lw=0.5, alpha=0.4)
        axr.axhspan(-4, 4, color='#2ca02c', alpha=0.07); axr.axhline(0, color='k', lw=0.6)
        for s in (-4, 4): axr.axhline(s, color='#2ca02c', lw=0.8, ls='--')
        ax.set_title(f'$\\Gamma_{{p0}} = {g}$: $N_{{ion}} = 10^5$, $\\lambda = 0.05$ at $\\Gamma_{{p0}}$')
        ax.set_ylabel(r'$\langle K\rangle / K_0$'); axr.set_ylabel('residual / s.e.'); axr.set_xlabel(r'$t$ [$10^3\,l/c$]')
        axr.set_ylim(-6, 6)
        if c == 0:
            ax.legend(ncol=2, loc='upper right')
            from matplotlib.lines import Line2D
            axr.legend(handles=[Line2D([], [], ls='', marker='o', color='k', ms=3, label='double'),
                                Line2D([], [], ls='', marker='x', color='k', ms=3, label='single'),
                                Line2D([], [], ls='', marker='o', mfc='none', color='k', ms=3, label='open: delta spectrum')],
                       ncol=3, loc='lower left', fontsize=6.5)
    fig.suptitle('Figure 1. Cooling: every mode against its own mean equation (E2, T5); 4 s.e. band')
    save(fig, 'fig1_cooling', """
Figure 1. Cooling curves <K(t)>/K0 of 10^5 ions at Gamma_p0 = 6 and 3 for every mode (continuous none, poisson, budget;
event; delta and restframe spectra), bh_proto through entity.xc on the merged head e7ba238c, Born loss tables, single (x)
and double (o) precision. Residual panel: event mode against the Jensen-corrected ODE on the Born table with the run's own
Var K (the registered E2/E10 statement, 4 s.e. of the sample mean); continuous modes against the plain ODE (RK4 at dt/16)
in units of the event run's s.e. at the same time. The registered continuous E2 runs the continuous mode at dt/16 with this
scale; the demonstration runs it at the production dt = 0.5 (and eta_u = 1e-4), where the explicit drag step leaves a
smooth deterministic offset (the revision-1.0 history row, no verdict). All curves use K_eff = K - delta,
the energy the ledger has charged (the stored K of the continuous modes lags by the pending payload, up to eta_u K, applied
to all ions at once). The continuous modes share one trajectory (none, poisson and budget differ only in what the paid
energy becomes). Delta spectrum: open markers.
Tolerance: 4 s.e. at the targets 0.9/0.7/0.5 (Gamma_p0 = 6) and 0.9/0.7/0.65 (Gamma_p0 = 3). Numbers: numbers.md, figure 1.""")

# ------------------------------------------------------------------ figure 2: straggling (E3)
def fig2():
    fig, axs = plt.subplots(3, 2, figsize=(10, 8.8), gridspec_kw={'height_ratios': [2, 1.3, 1.6]}, layout='constrained')
    tol = 4 * np.sqrt(2 / N_ION)
    for c, g in enumerate((6, 3)):
        K0 = MU_R * (g - 1); ax, axr, axh = axs[0, c], axs[1, c], axs[2, c]
        for spec, ls in (('restframe', '-'), ('delta', '--')):
            for prec, mk in (('double', 'o'), ('single', 'x')):
                r = load(f'demo_g{g}_event_{spec}_{prec}')
                if r is None: continue
                t, mK, vK, mKe, n = blocks(r); R = event_ref(g, spec, prec, r)
                rel = vK[1:] / R['V_fb'] - 1; curv = R['V_fb_curv'] / R['V_fb'] - 1
                sig = np.sqrt(vK[1:]) / mK[1:]
                dom = (sig <= 0.02) & (np.abs(curv) <= 0.006) & ((K0 - mK[1:]) / K0 >= 0.01)
                if prec == 'double':
                    ax.plot(t[1:] / 1e3, vK[1:] / K0**2, ls, color=SPEC_COL[spec], lw=1.4, label=f'event {spec}: measured Var K')
                    ax.plot(t[1:] / 1e3, R['V_fb'] / K0**2, ':', color='k', lw=1, label='drift-feedback prediction' if spec == 'restframe' else None)
                axr.plot(t[1:][dom] / 1e3, rel[dom] * 100, mk, ms=2.5, color=SPEC_COL[spec],
                         mfc='none' if spec == 'delta' else None)
                axr.plot(t[1:][~dom] / 1e3, rel[~dom] * 100, mk, ms=2, color='#d62728', alpha=0.35)
                if dom.any():
                    ilast = np.where(dom)[0][-1]
                    num('2', f'Gamma_p0={g} event {spec} {prec}: E3 |V/V_fb - 1| at the last record in the registered domain (t={t[1:][ilast]:g}, dK/K={(K0-mK[1:][ilast])/K0:.3f}, curvature {curv[ilast]*100:.3f} %)',
                        abs(rel[ilast]) * 100, f'demo_g{g}_event_{spec}_{prec}', '{:.3f} %')
                    num('2', f'Gamma_p0={g} event {spec} {prec}: E3 max |V/V_fb - 1| over the registered domain', np.max(np.abs(rel[dom])) * 100,
                        f'demo_g{g}_event_{spec}_{prec}', '{:.3f} %')
                num('2', f'Gamma_p0={g} event {spec} {prec}: sigma_K/K at the end', sig[-1], f'demo_g{g}_event_{spec}_{prec}', '{:.4f}')
                if (~dom).any():
                    num('2', f'Gamma_p0={g} event {spec} {prec}: max |V/V_fb - 1| outside the registered domain (not a registered statement); largest curvature budget there',
                        f'{np.max(np.abs(rel[~dom])) * 100:.3f} %; {np.max(np.abs(curv[~dom])) * 100:.3f} %', f'demo_g{g}_event_{spec}_{prec}')
            rc = load(f'demo_g{g}_poisson_{spec}_double')
            if rc is not None:
                num('2', f'Gamma_p0={g} continuous poisson {spec} double: max Var K / K0^2 (deterministic: rounding only)',
                    float(np.max(rc['z']['blocks'][:, 4])) / K0**2, rc['name'], '{:.2e}')
        axr.axhspan(-tol * 100, tol * 100, color=COL['event'], alpha=0.08)
        for s in (-1, 1): axr.axhline(s * tol * 100, color=COL['event'], ls='--', lw=0.8)
        axr.set_ylabel(r'$V/V_{fb} - 1$ [%]'); axr.set_xlabel(r'$t$ [$10^3\,l/c$]'); axr.set_ylim(-5, 5)
        ax.set_ylabel(r'Var $K / K_0^2$'); ax.set_title(f'$\\Gamma_{{p0}} = {g}$: event mode, $a = 1$'); ax.legend(loc='lower right' if g == 6 else 'upper right')
        ax.set_xlabel(r'$t$ [$10^3\,l/c$]')
        # K distribution at three times (event restframe and delta, double); continuous: all ions identical
        for spec, col in (('restframe', SPEC_COL['restframe']), ('delta', SPEC_COL['delta'])):
            r = load(f'demo_g{g}_event_{spec}_double')
            if r is None: continue
            kept = [k for k in r['z'].files if k.startswith('K_block')]
            for k, alpha in zip(sorted(kept, key=lambda s: int(s[7:])), (0.9, 0.6, 0.35)):
                K = r['z'][k]
                axh.hist(K / K0, bins=90, histtype='step', color=col, alpha=alpha, lw=1.0,
                         label=f'{spec}, $\\langle K\\rangle/K_0$ = {K.mean()/K0:.2f}')
            rc = load(f'demo_g{g}_none_double')
            if rc is not None and spec == 'restframe':
                for k in sorted([k for k in rc['z'].files if k.startswith('K_block')], key=lambda s: int(s[7:])):
                    axh.axvline(rc['z'][k][0] / K0, color=COL['none'], lw=1)
        axh.set_xlabel(r'$K / K_0$'); axh.set_ylabel('ions per bin'); axh.set_yscale('log'); axh.legend(fontsize=6.5, ncol=2, loc='upper left')
    fig.suptitle('Figure 2. Straggling: the event mode carries the dispersion; the continuous modes have none (E3)')
    save(fig, 'fig2_straggling', f"""
Figure 2. Top: Var K(t)/K0^2 of the event mode (restframe solid, delta dashed; double precision) against the drift-feedback
prediction dV/dt = -2 tau0 (dI/dK) V + a R E[D^2] (E[D^2] from the frozen bhref 0.2.0 D2mean for restframe, <D>^2 for
delta), computed along the Jensen-corrected mean. Middle: relative deviation in % (o double, x single); red points lie
outside the registered domain of revision 1.1.1 (sigma_K/K <= 0.02, curvature budget <= 0.6 %, Delta K/K >= 1 %).
Tolerance: 4 sqrt(2/N) = {tol*100:.3f} % (E3). Bottom: the K distribution at three times (event restframe and delta,
double); the grey lines are the continuous-mode runs, where all ions share one K (Var K at rounding level). The delta
histograms are combs: with a deterministic loss <D>(K) per event every ion sits on the same jump lattice
K_(i+1) = K_i - <D>(K_i), and the bins between lattice points stay empty.
Numbers: numbers.md, figure 2.""")

# ------------------------------------------------------------------ figure 3: pair spectra (E4, A10)
def ref_hist(g):
    rows = np.array([list(map(float, l.split())) for l in open(os.path.join(REF020, f'hist_mono05_G{g}.txt')) if l.strip() and not l.startswith('#')])
    out = {}
    for q, name in ((0, 'Eminus'), (1, 'Eplus'), (4, 'D')):
        r = rows[rows[:, 0] == q]; out[name] = (np.concatenate([r[:, 2], r[-1:, 3]]), r[:, 4])
    return out

def fig3():
    fig, axs = plt.subplots(4, 3, figsize=(11, 9.2), gridspec_kw={'height_ratios': [2, 1, 2, 1]}, layout='constrained')
    for rr, g in enumerate((3, 10)):
        ref = ref_hist(g)
        for cc, q in enumerate(('Eminus', 'Eplus', 'D')):
            ax, axr = axs[2*rr, cc], axs[2*rr + 1, cc]
            e, cnt = ref[q]; w = np.diff(e); N = cnt.sum(); ctr = .5 * (e[1:] + e[:-1])
            dref = cnt / N / w; sref = np.sqrt(cnt) / N / w
            ax.stairs(dref, e, color='k', lw=1.2, label='bhref 0.2.0 (unweighted chain, $10^6$)')
            for src, colr, mk in (('poisson', COL['poisson'], 'o'), ('budget_r025', COL['budget'], 's'), ('event', COL['event'], '^')):
                r = load(f'spec_g{g}_{src}_double')
                if r is None or f'hist_{q}_n' not in r['z'].files: continue
                if src.startswith('budget'):
                    h = r['z'][f'hist_{q}_w']; h2 = r['z'][f'hist_{q}_w2']; S = r['j']['pairs']['w_window']
                    d = h / S / w; sd = np.sqrt(h2) / S / w; lab = 'budget: $\\sum w$ (w = r/D)'
                else:
                    h = r['z'][f'hist_{q}_n']; S = r['j']['pairs']['n_window']; d = h / S / w; sd = np.sqrt(h) / S / w
                    lab = f'{src}: counts (w = 1)'
                ok = (cnt >= 100) & (d > 0)
                pull = (d - dref) / np.sqrt(sd**2 + sref**2)
                chi2 = float(np.sum(pull[ok]**2)); ndf = int(ok.sum())
                ax.plot(ctr[ok], d[ok], mk, ms=2.2, color=colr, label=lab if cc == 0 else None)
                axr.plot(ctr[ok], pull[ok], mk, ms=2.2, color=colr)
                num('3', f'Gamma_p={g} {q} {src} (double): chi2/ndf of the density against bhref 0.2.0 (bins with >= 100 reference counts)',
                    f'{chi2:.1f}/{ndf} = {chi2/ndf:.3f}', r['name'])
                num('3', f'Gamma_p={g} {q} {src} (double): max |pull|', np.max(np.abs(pull[ok])), r['name'], '{:.2f}')
                num('3', f'Gamma_p={g} {src} (double): pairs in the window (parent Gamma_p within 0.5 % of {g})',
                    r['j']['pairs']['n_window'], r['name'], '{:.0f}')
            QL = {'Eminus': r'$E_-$', 'Eplus': r'$E_+$', 'D': r'$D = E_- + E_+ - \epsilon$'}[q]
            ax.set_yscale('log'); ax.set_title(f'$\\Gamma_p = {g}$: {QL}')
            ax.set_xlim(e[:-1][cnt > 0][0], e[1:][cnt > 0][-1])
            if g == 10: ax.set_xscale('log'); axr.set_xscale('log')
            axr.set_xlim(ax.get_xlim()); axr.axhspan(-3, 3, color='k', alpha=0.05); axr.set_ylim(-5, 5); axr.set_ylabel('pull')
            axr.set_xlabel(QL + r' [$m_e c^2$, lab]')
            if cc == 0: ax.legend(fontsize=6.5)
    fig.suptitle('Figure 3. Three ways to create pairs, one physical spectrum (E4, A10)')
    save(fig, 'fig3_spectra', """
Figure 3. Lab-frame lepton energies E-, E+ (gamma) and the ion loss D = E- + E+ - eps of the pairs created at nearly fixed
Gamma_p = 3 and 10 (spectrum runs: 2.5e5 ions, 40 steps, restframe spectrum, double precision; pairs whose parent Gamma_p
lies within 0.5 % of the nominal value) by the poisson source (counts, w = 1), the event mode (counts) and the budget
source (sum of the variable weights w = r/D: the D-weighted draw reweighted to the physical law), against the frozen
bhref 0.2.0 unweighted-chain histograms (10^6 draws; the sampler law is unchanged by the Born q_k). Pull panel: (run -
reference) / combined s.e. per bin, bins with >= 100 reference counts. The registered tests are E4 (calibrated
Anderson-Darling family) and A10; the chi2/ndf here is a display statistic. Numbers: numbers.md, figure 3.""")

# ------------------------------------------------------------------ figure 4: pairs per unit paid energy (A12, T10)
BLAG = {}
YIELD = {}   # (src, spec, g) -> (whole-run N / continuum - 1, expected), filled by figure 4
def budget_model(g):
    if g not in BLAG: BLAG[g] = budget_lag.model(OUT, float(g))
    return BLAG[g]

def whole_run_yield(src, spec, g, r):
    """Whole-run N_pairs / Q_paid against the continuum reference int dK/<D> over the paid interval, with the expected
    offset of each mode: event, the finite jump (each event costs <D> at its pre-event state: first order
    -(E[D^2] / 2<D>) dln<D>/dK along the path, plus the straggling spread of the end points); budget, the emission lag
    (one-ion model of the kernel's rule, budget_lag.py); poisson, none (a Poisson sample of the drag at the stored state)."""
    T = table('double'); z = r['z']; j = r['j']; lf = j['log']['ledger_final']; K0 = MU_R * (g - 1)
    if src == 'budget': Np, Q = j['pairs']['w'], j['pairs']['wD']        # emitted packets: Q = sum r, the paid interval [K0 - Q/N, K0]
    elif src == 'event': Np, Q = j['pairs']['n'], j['pairs']['wD']       # sum of the event losses D, exact
    else: Np, Q = j['pairs']['n'], lf['E_drag']                          # the drag the pairs sample
    K = np.linspace(K0 - Q / N_ION, K0, 4001); G = 1 + K / MU_R; D = T.Df(G); f = 1 / D
    ref = np.trapezoid(f, K) * N_ION; dev = Np / ref - 1
    D2 = D**2 if spec == 'delta' else d2fix().D2(G)
    if src == 'event':
        exp_ = -np.trapezoid(f * D2 / (2 * D) * np.gradient(np.log(D), K), K) / np.trapezoid(f, K)
        # end points: the ions stop at different K (Var K at the end); the path integral is convex in the paid energy,
        # F''(Q) = <D>'/<D>^2 at the bottom, so the ensemble sum exceeds N_ion F(Q_mean) by (1/2) F'' Var K
        _, _, vK, _, _ = blocks(r); dD = np.gradient(D, K)[0]
        endp = 0.5 * dD / D[0]**2 * vK[-1] / np.trapezoid(f, K); exp_ += endp
        se = np.sqrt(max(np.trapezoid(f * (D2 / D**2 - 1), K) / np.trapezoid(f, K), 0) / Np); what = 'finite jump, first order, plus the end-point spread'
        num('4', f'event {spec} Gamma_p0={g} (double): end-point spread term (1/2) <D>\'/<D>^2 Var K_end / int dK/<D>', f'{endp:+.4%}', r['name'])
    elif src == 'budget':
        M = budget_model(g); exp_ = M['bias']; se = 0.0; what = 'emission lag, one-ion model'
        if spec == 'restframe': se = np.sqrt(np.sum(z['Gbin_w2'])) / j['pairs']['w']
    else:
        exp_ = 0.0; se = np.sqrt(np.trapezoid(f * D2 / D**2, K) / np.trapezoid(f, K) / Np); what = 'none'
    zz = (dev - exp_) / se if se > 0 else float('nan')
    num('4', f'{src} {spec} Gamma_p0={g} (double): whole run N_pairs / int dK/<D> over the paid interval - 1; expected ({what}); (measured - expected)/s.e.',
        f'{dev:+.4%}; {exp_:+.4%}; ' + (f'{zz:+.2f}' if np.isfinite(zz) else f'deterministic (no sampling error), difference {dev - exp_:+.1e}'), r['name'])
    return dev, exp_

def fig4():
    WR = []
    T = table('double'); fig, (ax, axr, axe) = plt.subplots(3, 1, figsize=(7.2, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1.2, 1.2]}, layout='constrained')
    E = None; Gg = np.linspace(2.14, 6.0, 400); ax.plot(Gg, 1 / T.Df(Gg), 'k-', lw=1, label=r'$1/\langle D\rangle(\Gamma_p)$ (Born table)')
    for src, colr, mk in (('poisson', COL['poisson'], 'o'), ('budget', COL['budget'], 's'), ('event', COL['event'], '^')):
        for spec in ('restframe', 'delta'):
            for g in (6, 3):
                r = load(f'demo_g{g}_{src}_{spec}_double')
                if r is None or 'G_edges' not in r['z'].files: continue
                z = r['z']; E = z['G_edges']; c = .5 * (E[1:] + E[:-1]); K0 = MU_R * (g - 1)
                t, mK, vK, mKe, n = blocks(r); Gend = 1 + mK[-1] / MU_R
                if src == 'event':
                    Q = z['Gbin_wD']; N_ = z['Gbin_n']; sN = np.sqrt(np.maximum(z['Gbin_D2'] - z['Gbin_D']**2 / np.maximum(N_, 1), 0))
                    ratio = N_ / Q; sr = ratio * sN / np.maximum(Q, 1e-300)
                elif src == 'budget':
                    Q = z['Gbin_wD']; N_ = z['Gbin_w']; ratio = N_ / Q; sr = np.sqrt(z['Gbin_w2']) / np.maximum(Q, 1e-300)
                else:
                    # poisson: CUMULATIVE from Gamma_p0 down to each bin's lower edge. The pairs carry the STORED parent
                    # Gamma_p, which moves only when the pending loss is applied (quanta of order eta_u K per ion), so a
                    # narrow bin holds an integer number of quanta (+-0.5 quantum ~ +-0.3 % here); the cumulative ratio
                    # makes that a vanishing fraction. Q_cum = N mu_r (Gamma_p0 - edge), the drag down to the edge.
                    N_ = np.cumsum(z['Gbin_n'][::-1])[::-1]; lo = np.clip(E[:-1], Gend, g)
                    Q = N_ION * MU_R * (g - lo); ratio = N_ / np.where(Q > 0, Q, np.nan); sr = np.sqrt(N_) / np.where(Q > 0, Q, np.nan)
                    c = lo
                # reference: 1/<D> averaged uniformly in K over the part of the bin the ions crossed (poisson: over
                # [edge, Gamma_p0], the cumulative interval)
                refv = np.full(len(c), np.nan)
                for i in range(len(c)):
                    a, b = (max(E[i], Gend), min(E[i + 1], g)) if src != 'poisson' else (max(E[i], Gend), g)
                    if b > a: gg = np.linspace(a, b, 401 if src == 'poisson' else 41); refv[i] = np.mean(1 / T.Df(gg))
                ok = np.isfinite(ratio) & (N_ > 50) & np.isfinite(refv)
                if src == 'poisson':  # the edge bin only partly crossed at the bottom is dropped
                    ok &= (E[:-1] >= Gend) & (E[:-1] < g)
                else:
                    # the bottom bins the ensemble crossed only in part (event: the final K spread, 4 sigma_K; budget:
                    # the unpaid debt, up to r_th ~ 100) are dropped: their reference is not a uniform crossing
                    ok &= E[:-1] >= Gend + max(4 * np.sqrt(vK[-1]), 100.0) / MU_R
                lab = f'{src} {spec}' if g == 6 else None
                ax.plot(c[ok], ratio[ok], mk, ms=3, color=colr, mfc='none' if spec == 'delta' else colr, label=lab)
                if src == 'poisson' and spec == 'restframe':
                    # the poisson points are cumulative from Gamma_p0: their reference is the path average of 1/<D>
                    ax.plot(c[ok], refv[ok], '--', color=colr, lw=0.9,
                            label=r'poisson reference: $1/\langle D\rangle$ averaged from $\Gamma_{p0}$ down' if g == 6 else None)
                dev = (ratio / refv - 1) * 100; sdev = sr / refv * 100
                axr.errorbar(c[ok], dev[ok], yerr=None if (src != 'poisson' and spec == 'delta') else sdev[ok], fmt=mk, ms=2.5, color=colr,
                             mfc='none' if spec == 'delta' else colr, lw=0.6)
                zz = (ratio / refv - 1) / (sr / refv)
                if src != 'poisson' and spec == 'delta':
                    # deterministic: every event or packet costs exactly <D>(Gamma_p) and all ions follow the same path in K
                    # (event: the same jump lattice; budget: the same ~80 emission points), so a bin holds a fixed set of
                    # points and the deviation is their placement against the uniform bin average; no z is reported
                    num('4', f'{src} delta Gamma_p0={g} (double): max |N/Q * <D> - 1| over the bins (deterministic: point placement in the bin, no sampling error)',
                        f'{np.nanmax(np.abs(dev[ok])):.3f} %', r['name'])
                else:
                    num('4', f'{src} {spec} Gamma_p0={g} (double): max |N/Q * <D> - 1| over the bins (%), max |z|' + (' [cumulative from Gamma_p0]' if src == 'poisson' else ''),
                        f'{np.nanmax(np.abs(dev[ok])):.3f} %, {np.nanmax(np.abs(zz[ok])):.2f}', r['name'])
                WR.append((src, spec, g) + whole_run_yield(src, spec, g, r)); YIELD[(src, spec, g)] = WR[-1][3:]
                # energy yield: sum w (E- + E+) / Q vs 1 + eps/<D>
                wE = np.cumsum(z['Gbin_wE'][::-1])[::-1] if src == 'poisson' else z['Gbin_wE']
                ey = wE / np.where(Q > 0, Q, np.nan); eref = 1 + 0.5 * refv
                axe.plot(c[ok], (ey[ok] / eref[ok] - 1) * 100, mk, ms=2.5, color=colr, mfc='none' if spec == 'delta' else colr)
                num('4', f'{src} {spec} Gamma_p0={g} (double): max |energy yield / (1 + eps <1/D>) - 1| over the bins', np.nanmax(np.abs(ey[ok] / eref[ok] - 1)) * 100,
                    r['name'], '{:.3f} %')
    ax.set_ylabel(r'$N_{pairs} / Q_{paid}$  [$(m_e c^2)^{-1}$]'); ax.legend(ncol=2, fontsize=6.5); ax.set_yscale('log')
    # whole run against the continuum over the paid interval, with each mode's expected offset
    lines = ['whole run: $N / \\int dK/\\langle D\\rangle$ $-$ 1 (expected)']
    for src, spec, g, dev, ex in sorted(WR, key=lambda w: (w[0], -w[2], w[1])):
        lines.append(f'{src} {spec} $\\Gamma_{{p0}}$={g}: {dev*100:+.3f} % ({ex*100:+.3f} %)')
    ax.text(0.02, 0.02, '\n'.join(lines), transform=ax.transAxes, fontsize=5.6, va='bottom', ha='left', family='monospace',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
    for g in (6, 3):
        if load(f'demo_g{g}_budget_delta_double') is not None or load(f'demo_g{g}_budget_restframe_double') is not None:
            M = budget_model(g)
            num('4', f'budget Gamma_p0={g}: emission-lag offset of the whole-run number, one-ion model of the kernel rule (budget_lag.py): packets/ion, K0 triggers/ion, offset',
                f"{len(M['P'])}, {M['nk0']}, {M['bias']*100:+.4f} %", 'model')
    axr.set_ylabel(r'number yield $\times \langle D\rangle - 1$ [%]'); axr.axhline(0, color='k', lw=0.6); axr.set_ylim(-1, 1)
    axe.set_ylabel('energy yield / expected $- 1$ [%]'); axe.axhline(0, color='k', lw=0.6); axe.set_xlabel(r'parent $\Gamma_p$'); axe.set_ylim(-0.5, 0.5)
    fig.suptitle('Figure 4. Pairs per unit paid ion energy: the physical yield in every mode (A12, T10)')
    save(fig, 'fig4_yield', """
Figure 4. Number of physical pairs per unit of paid ion energy in bins of the parent Gamma_p (poisson: cumulative from
Gamma_p0 down to each point, because its pairs carry the stored parent Gamma_p, which moves in pending-application quanta
of order eta_u K; per narrow bin that quantisation alone gives +-0.3 %), from the demonstration runs
(Gamma_p0 = 6 and 3, double precision; restframe filled, delta open), against 1/<D>(Gamma_p) of the Born table averaged
over the part of each bin the ions crossed. Paid energy Q: budget, the packet energies r = w D (exact); event, the sum of
the event losses D (exact, a = 1); poisson, the continuous drag N_ion mu_r Delta Gamma_p over the bin (the pairs are a
Poisson sample of it). Middle: relative deviation with its statistical error (none for the deterministic delta event and
budget runs, where all ions follow the same path in K and a bin holds a fixed set of event or emission points); the
bottom bins the ensemble crossed only in part are dropped. Box: the whole run against the continuum integral of
dK/<D> over the paid interval, with each mode's expected offset in brackets: poisson none; event the finite jump (each
event costs <D> at its pre-event state) plus the spread of the end points; budget the emission lag (a packet pays the
energy accrued since the previous packet, but its pairs are drawn at the emission Gamma_p, the lowest of that interval,
so the per-bin yield is physical while the pairs come late: a one-ion model of the kernel's rule reproduces the run's
packets and the offset, budget_lag.py). Bottom: the energy yield
sum w (E- + E+) / Q against 1 + eps <1/D> (mono bath, eps = 0.5). The registered statements are A12 (1 % cost, 2 %
number and energy per node, N = 2e5, the sampler at fixed Gamma_p) and T10; neither sees the emission lag.
Numbers: numbers.md, figure 4.""")

# ------------------------------------------------------------------ figure 5: energy bookkeeping (E1, T11, ledger identity)
def fig5():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
    ax, axc, axs_ = axs
    from matplotlib.lines import Line2D
    XOFF = {(6, 'double'): -0.27, (6, 'single'): -0.09, (3, 'double'): 0.09, (3, 'single'): 0.27}
    nzero = 0; nrun = 0
    for prec, mk in (('double', 'o'), ('single', 'x')):
        for g in (6, 3):
            for cfg in CFGS:
                r = load(f'demo_g{g}_{cfg}_{prec}')
                if r is None: continue
                L = r['z']['ledger']; mode = cfg.split('_')[0]
                rel = np.abs(L[:, 22]) / np.maximum(np.abs(L[:, 2]), 1e-300); nrun += 1; nzero += int(not rel.any())
                ax.semilogy(L[:, 1] / L[-1, 1], np.maximum(rel, 1e-19), mk if prec == 'single' else '-', ms=1.5, lw=0.8, color=COL[mode], alpha=0.6)
                num('5', f'{cfg} Gamma_p0={g} {prec}: max over the rows of |ledger identity residual| / E_drag', float(rel.max()), r['name'], '{:.2e}')
                cl = r['j']['log'].get('closure', {}); ai = r['j']['log'].get('array_identity', {})
                x = CFGS.index(cfg) + XOFF[(g, prec)]; mk2 = 'o' if prec == 'double' else '^'
                if cl:
                    axc.semilogy(x, max(abs(cl['rel']), 1e-19), mk2, color=COL[mode], ms=4.5)
                    num('5', f'{cfg} Gamma_p0={g} {prec}: T11 ion closure |sum w (K_init - K_final) - E_drag| / E_drag', abs(cl['rel']), r['name'], '{:.2e}')
                if ai and cl:
                    v = abs(ai['res_minus_Eround']) / max(cl['E_drag'], 1e-300)
                    axc.semilogy(x, max(v, 1e-19), mk2, color=COL[mode], ms=6.5, mfc='none', mew=1.0)
                if ai and ai.get('B'):
                    num('5', f'{cfg} Gamma_p0={g} {prec}: array identity |residual| / B_budget (pairs from the stored arrays)', ai['rel_B'], r['name'], '{:.2e}')
                    num('5', f'{cfg} Gamma_p0={g} {prec}: array identity (residual - E_round) / E_drag', abs(ai['res_minus_Eround']) / max(cl.get('E_drag', 1), 1e-300), r['name'], '{:.2e}')
                lf = r['j']['log'].get('ledger_final', {})
                if mode == 'none' and lf:
                    num('5', f'none Gamma_p0={g} {prec}: E_lost_nopairs / E_drag (no pairs: nothing conserved in pairs)', lf['E_lost_nopairs'] / lf['E_drag'], r['name'], '{:.6f}')
    ax.set_xlabel('t / t_end'); ax.set_ylabel('|identity residual| / E_drag (exact 0 drawn at 1e-19)')
    ax.set_title(f'ledger identity per row: exactly 0 at every row in {nzero} of {nrun} runs')
    num('5', 'runs whose ledger identity residual is exactly 0 at every row', f'{nzero} of {nrun}', 'all demo_g{6,3}_* runs')
    ax.axhline(1e-12, color='k', ls='--', lw=0.8); ax.text(0.02, 4e-13, 'registered 1e-12', fontsize=7)
    axc.axhline(1e-12, color='k', ls='--', lw=0.8); axc.axhline(1e-5, color='k', ls=':', lw=0.8)
    axc.text(len(CFGS) - 0.55, 2e-13, '1e-12 (double)', fontsize=6.5, ha='right'); axc.text(len(CFGS) - 0.55, 2e-6, '1e-5 (single)', fontsize=6.5, ha='right')
    axc.set_xticks(range(len(CFGS))); axc.set_xticklabels([c.replace('_', '\n') for c in CFGS], fontsize=6.5)
    axc.set_title('from the particle arrays, independent of the ledger'); axc.set_ylabel('relative to E_drag (exact 0 drawn at 1e-19)')
    axc.legend(handles=[Line2D([], [], ls='', marker='o', color='k', ms=4, label='ions: |sum w dK - E_drag| (T11)'),
                        Line2D([], [], ls='', marker='o', mfc='none', color='k', ms=6, label='pairs: |sum w (E- + E+ - eps) - rhs - E_round|'),
                        Line2D([], [], ls='', marker='o', color='0.5', ms=4, label='double (left of each pair of points)'),
                        Line2D([], [], ls='', marker='^', color='0.5', ms=4, label='single (right); Gamma_p0 = 6 left, 3 right')],
               fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 0.9))
    for g in (6, 3):
        for spec, ls in (('restframe', '-'), ('delta', '--')):
            for prec in ('double', 'single'):
                r = load(f'demo_g{g}_poisson_{spec}_{prec}')
                if r is None or 'Gbin_D2' not in r['z'].files: continue
                L = r['z']['ledger']; z = r['z']; ED2 = z['Gbin_D2'].sum() / max(z['Gbin_n'].sum(), 1)
                sig = np.sqrt(np.maximum(L[:, 23], 1) * ED2)
                if prec == 'double':
                    axs_.plot(L[:, 1] / L[-1, 1], L[:, 7] / sig, ls, color=COL['poisson'] if g == 6 else '#17becf', lw=1, label=f'$\\Gamma_{{p0}}$={g} {spec}')
                num('5', f'poisson {spec} Gamma_p0={g} {prec}: E_stochastic / E_drag at the end (signed)', L[-1, 7] / L[-1, 2], r['name'], '{:+.2e}')
                num('5', f'poisson {spec} Gamma_p0={g} {prec}: E_stochastic / sqrt(n_events E[D^2]) at the end', L[-1, 7] / sig[-1], r['name'], '{:+.2f}')
    axs_.axhspan(-3, 3, color=COL['poisson'], alpha=0.06); axs_.axhline(0, color='k', lw=0.6)
    axs_.set_title(r'poisson: $E_{stochastic} / \sqrt{n_{events} E[D^2]}$'); axs_.set_xlabel('t / t_end'); axs_.legend(fontsize=6.5)
    fig.suptitle('Figure 5. What each mode conserves: exactly (budget, event), in expectation (poisson), not at all (none)')
    save(fig, 'fig5_energy', """
Figure 5. Left: the ledger identity E_drag = (E_pairs - E_absorbed) + E_round + E_stochastic + debt + E_exported +
E_lost_nopairs, residual relative to E_drag at every ledger row of the 28 runs (double lines, single crosses; colour by
mode); registered 1e-12. Middle: two checks from the particle arrays, independent of the ledger: the ions,
|sum w (K_init - K_final) - E_drag| / E_drag (T11), and the pairs, |sum_pairs w (E- + E+ - eps) - rhs - E_round| / E_drag
with rhs = sum w dK - sum w r - E_exported - E_lost_nopairs - E_stochastic + debt_initial: the ledger's E_round accounts
for the whole float rounding of the pair energies (the rounding itself, the E1 statement, is in figure 6). Per mode,
Gamma_p0 = 6 left and 3 right, double (o) and single (triangle). Right: the poisson source's E_stochastic = E_drag - sum W D, the signed random walk of a Poisson sample
around the paid energy, in units of sqrt(n_events E[D^2]). The none source loses the energy (E_lost_nopairs = E_drag).
Numbers: numbers.md, figure 5.""")

# ------------------------------------------------------------------ figure 6: single versus double (T5b, E1 exception)
def fig6():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained'); ax, axu, axe = axs
    for prec, col in (('double', 'k'), ('single', '#d62728')):
        r = load(f'stair_g6_none_{prec}')
        if r is None: continue
        t, mK, vK, mKe, n = blocks(r); K0 = r['j']['K0']
        ax.step(t, mK - K0, where='post', color=col, lw=1.2, label=f'stored K ({prec})')
        ax.plot(t, mKe - K0, '--', color=col, lw=0.9, label=f'K_eff = K - delta ({prec})')
        num('6', f'staircase run {prec}: K - K0 at the end (t={t[-1]:g}): stored, K_eff', f'{mK[-1]-K0:.6g}, {mKe[-1]-K0:.6g}', r['name'])
    rs = load('stair_g6_none_single'); rd = load('stair_g6_none_double')
    if rs is not None:
        T = table('single'); t, mK, vK, mKe, n = blocks(rs); K0 = rs['j']['K0']; tau = 2.564289235e-4
        kp = plain_ode(T, tau, K0, t[1:]); ax.plot(t[1:], kp - K0, ':', color='#1f77b4', lw=1.2, label='plain ODE')
        # a naive float32 update u <- fl(u - du) with the same per-step loss: the loss is below half an ulp of u
        u = np.float32(np.sqrt(K0 * (K0 + 2.0))); du = tau * float(T.If([6.0])[0]) * 0.5 * float(np.sqrt(1 + u**2)) / float(u)
        un = u
        nsteps = int(round(t[-1] / 0.5))
        for _ in range(nsteps): un = np.float32(un - np.float32(du))
        Kn = float(np.sqrt(1 + np.float64(un)**2) - 1)
        ax.axhline(Kn - K0, color='#9467bd', lw=1, ls='-.', label='naive float update (no pending)')
        num('6', 'staircase: per-step loss over the single ulp-energy at K = 9180', (tau * float(T.If([6.0])[0]) * 0.5) / (float(np.spacing(np.float32(u))) * float(u) / float(np.sqrt(1 + u**2))), 'stair_g6_none_single', '{:.3f}')
        num('6', f'naive float update: K - K0 after {nsteps} steps (the float32 representation of the initial u only)', Kn - K0, 'computed (numpy float32)', '{:.4g}')
        napp = int(np.sum(np.diff(mK) < 0))
        num('6', 'staircase single: number of pending applications (drops of the stored K) over the run; drop size / K0 (eta_u)',
            f'{napp}; {float(np.median(-np.diff(mK)[np.diff(mK) < 0])) / K0:.3e}', 'stair_g6_none_single')
        if rd is not None:
            t2, mK2, _, mKe2, _ = blocks(rd)
            num('6', 'staircase: max |K_eff(single) - K_eff(double)| / K0 over the records', float(np.max(np.abs(mKe - mKe2))) / K0, 'stair_g6_none_{single,double}', '{:.2e}')
    for g in (6, 3):
        a_, b_ = load(f'demo_g{g}_none_single'), load(f'demo_g{g}_none_double')
        if a_ is not None and b_ is not None:
            _, ms, _, mes, _ = blocks(a_); _, md, _, med, _ = blocks(b_); K0g = MU_R * (g - 1)
            num('6', f'Gamma_p0={g} continuous (none) single vs double: max |stored K_s - K_d| / K0; max |K_eff,s - K_eff,d| / K0 over the records',
                f'{np.max(np.abs(ms - md)) / K0g:.2e}; {np.max(np.abs(mes - med)) / K0g:.2e}', f'demo_g{g}_none_{{single,double}}')
    ax.set_xlabel(r'$t$ [$l/c$]'); ax.set_ylabel(r'$K - K_0$ [$m_e c^2$]'); ax.set_title(r'one ion, $\Gamma_p = 6$, $\lambda = 5\times10^{-5}$'); ax.legend(fontsize=6.5)
    # ulp-energy ratio dK_step / dK_ulp versus Gamma_p, single and double, at the demo tau0 and at the staircase tau0
    T = table('double'); G = np.linspace(2.2, 10, 300); K = MU_R * (G - 1); u = np.sqrt(K * (K + 2)); gd = np.sqrt(1 + u**2)
    for tau, lab, ls in ((TAU0[6], r'demo $\Gamma_{p0}=6$ setting', '-'), (TAU0[3], r'demo $\Gamma_{p0}=3$ setting', '--'), (2.564289235e-4, 'staircase setting', ':')):
        dK = tau * T.If(G) * 0.5
        for prec, col, dt_ in (('single', '#d62728', np.float32), ('double', 'k', np.float64)):
            ulpE = np.spacing(u.astype(dt_)).astype(float) * u / gd
            axu.semilogy(G, dK / ulpE, ls, color=col, lw=1, label=f'{lab}, {prec}')
    axu.axhline(1, color='k', lw=0.5); axu.set_xlabel(r'$\Gamma_p$'); axu.set_ylabel(r'$dK_{step} / dK_{ulp}$'); axu.legend(fontsize=6)
    axu.set_title('loss per step over the ulp-energy of the stored u')
    # E1: pairs from the arrays against the ions from the arrays, |residual| / E_drag (the float rounding of the pair
    # energies, recorded by the ledger as E_round), single vs double, by mode
    for i, cfg in enumerate(CFGS[1:]):
        for g, dx in ((6, -0.15), (3, 0.15)):
            for prec, mk, col in (('double', 'o', 'k'), ('single', 'x', '#d62728')):
                r = load(f'demo_g{g}_{cfg}_{prec}')
                if r is None: continue
                ai = r['j']['log'].get('array_identity'); cl = r['j']['log'].get('closure')
                if not ai or not cl: continue
                v = abs(ai['residual']) / cl['E_drag']
                axe.semilogy(i + dx, max(v, 1e-19), mk, color=col, ms=5)
                num('6', f'{cfg} Gamma_p0={g} {prec}: E1 |sum_pairs w (E- + E+ - eps) - rhs| / E_drag (pair-energy rounding)', v, r['name'], '{:.2e}')
                lf = r['j']['log'].get('ledger_final', {})
                if lf: num('6', f'{cfg} Gamma_p0={g} {prec}: |E_round| / E_drag (ledger)', abs(lf['E_round']) / lf['E_drag'], r['name'], '{:.2e}')
    axe.axhline(1e-5, color='#d62728', ls=':', lw=0.8); axe.axhline(1e-12, color='k', ls='--', lw=0.8)
    axe.text(-0.4, 1.5e-5, 'E1 single 1e-5', fontsize=6.5, color='#d62728'); axe.text(-0.4, 1.5e-12, 'E1 double 1e-12', fontsize=6.5)
    ied = CFGS[1:].index('event_delta')
    axe.plot(ied, 1.83e-5, '_', color='#d62728', ms=14, mew=1.5)
    axe.annotate('registered E1 delta single row\n1.83e-5 (exception, rev. 1.1)', (ied, 1.83e-5), (ied - 1.6, 3e-3), fontsize=6,
                 arrowprops=dict(arrowstyle='->', lw=0.6))
    axe.set_xticks(range(len(CFGS) - 1)); axe.set_xticklabels([c.replace('_', '\n') for c in CFGS[1:]], fontsize=6.5)
    axe.set_ylabel('E1: |pairs - ions| / E_drag (0 drawn at 1e-19)'); axe.set_title(r'E1 pair-energy rounding: single (x) vs double (o); $\Gamma_{p0}$ = 6 left, 3 right')
    fig.suptitle('Figure 6. Single versus double precision: the pending-loss mechanism and what float costs (T5b, E1)')
    save(fig, 'fig6_precision', """
Figure 6. Left: one ion at Gamma_p = 6 with a loss per step of 0.43 single-precision ulp-energies (staircase run, lambda =
5e-5, 20000 steps): the loss accumulates in the pending payload delta and is applied to the stored momentum when it
reaches eta_u K (0.92 here, above the 100-ulp floor in both precisions), so the stored K drops in steps while
K_eff = K - delta follows the plain ODE, identically in single and double; a naive float update u <- fl(u - du) would
not cool at all (T5b, the pending mechanism of III.3). Middle: the loss per step over the ulp-energy of the stored momentum, versus Gamma_p, for the demonstration
settings and the staircase setting, single and double. Right: E1, the energy of the pairs from the stored pair arrays
against the energy the ions lost from the stored ion arrays (E1 for the event mode; for the continuous sources the same
identity with E_stochastic, the debt and the pending payload on the right-hand side), |residual| / E_drag, single (red x)
and double (black o), against 1e-5 (single) and 1e-12 (double). The residual is the float rounding of the pair energies;
the ledger records it as E_round (figure 5). The event-mode delta spectrum in single precision is the development-only
exception of revision 1.1 (a constant recoil rounded the same way at every event; its registered E1 row was 1.83e-5).
Numbers: numbers.md, figure 6.""")

# ------------------------------------------------------------------ figure 7: budget weights (T18-lite)
def fig7():
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained'); axw, axp, axd = axs
    for g in (6, 3):
        for spec, ls in (('restframe', '-'), ('delta', '--')):
            r = load(f'demo_g{g}_budget_{spec}_double')
            if r is None or 'W_hist' not in r['z'].files: continue
            z = r['z']; W = z['W_edges']; h = z['W_hist']
            axw.stairs(h / h.sum(), W, color=COL['budget'] if g == 6 else '#8c564b', ls=ls, label=f'$\\Gamma_{{p0}}$={g} {spec}')
            p = r['j']['pairs']
            num('7', f'budget {spec} Gamma_p0={g} (double): w_pair min / mean / max', f"{p['wmin']:.4g} / {p['w']/p['n']:.4g} / {p['wmax']:.4g}", r['name'])
            L = z['ledger']; K0 = MU_R * (g - 1)
            axp.plot(L[:, 1], L[:, 24] / N_ION, ls, color=COL['budget'] if g == 6 else '#8c564b', label=f'$\\Gamma_{{p0}}$={g} {spec}')
            axd.plot(L[:, 1], L[:, 20] / (N_ION * K0), ls, color=COL['budget'] if g == 6 else '#8c564b')
            num('7', f'budget {spec} Gamma_p0={g} (double): packets per ion at the end', L[-1, 24] / N_ION, r['name'], '{:.3f}')
            num('7', f'budget {spec} Gamma_p0={g} (double): max outstanding debt / (N_ion K0)', float(np.max(L[:, 20])) / (N_ION * K0), r['name'], '{:.3e}')
            rp = load(f'demo_g{g}_poisson_{spec}_double')
            if rp is not None:
                Lp = rp['z']['ledger']
                num('7', f'Gamma_p0={g} {spec}: macro-pairs poisson / budget at the end (what the weights buy)', Lp[-1, 24] / max(L[-1, 24], 1), r['name'] + ' vs ' + rp['name'], '{:.2f}')
                axp.plot(Lp[:, 1], Lp[:, 24] / N_ION, ls, color=COL['poisson'] if g == 6 else '#17becf', lw=0.8, alpha=0.6, label=f'poisson $\\Gamma_{{p0}}$={g} {spec}' if spec == 'restframe' else None)
    axw.set_xscale('log'); axw.set_xlabel(r'$w_{pair} / w_{ion}$'); axw.set_ylabel('fraction of packets'); axw.legend(fontsize=6.5); axw.set_title('budget pair weights')
    axp.set_yscale('log'); axp.set_xscale('log'); axp.set_xlabel(r'$t$ [$l/c$]'); axp.set_ylabel('macro-pairs per ion'); axp.legend(fontsize=6); axp.set_title('pairs created (budget vs poisson)')
    axd.set_xscale('log'); axd.set_xlabel(r'$t$ [$l/c$]'); axd.set_ylabel(r'outstanding debt / $(N_{ion} K_0)$'); axd.set_title('energy owed to pairs not yet emitted')
    fig.suptitle('Figure 7. What variable weights cost and what they buy (budget source, T18-lite)')
    save(fig, 'fig7_weights', """
Figure 7. The budget source (t_latency = 0, r_th = min(10 <D>, 0.02 K), double precision): left, the distribution of the
pair weights w = r/D (restframe: spread by the sampled D; delta: narrow); middle, macro-pairs per ion over the run against
the poisson source with a = 1 (what the weights buy: fewer, heavier macro-pairs carrying the same physical number); right,
the outstanding debt, the energy already removed from the ions and not yet emitted as pairs (what they cost: a lag of order
r_th per ion, and with it a number yield above the physical one by the emission lag, figure 4). Numbers: numbers.md,
figure 7.""")

# ------------------------------------------------------------------ figure 8: mode table
def fig8(status_rows):
    perf = {}
    if PERF and os.path.exists(PERF):
        for l in open(PERF).read().splitlines()[1:]:
            f = l.split(',')
            m = re.match(r'm1_(\w+)_warm_(\d+)', f[1])
            if m: perf[m[1]] = float(f[4])
    rows = []; seen = set()
    for cfg in CFGS:
        mode = cfg.split('_')[0]; r6 = load(f'demo_g6_{cfg}_double'); r3 = load(f'demo_g3_{cfg}_double')
        lf6 = r6['j']['log'].get('ledger_final', {}) if r6 else {}
        rmax = 0.0
        for g in (6, 3):
            for prec in ('single', 'double'):
                r = load(f'demo_g{g}_{cfg}_{prec}')
                if r: L = r['z']['ledger']; rmax = max(rmax, float(np.max(np.abs(L[:, 22]) / np.maximum(np.abs(L[:, 2]), 1e-300))))
        cost = perf.get(mode)
        yv = YIELD.get((mode, cfg.split('_')[1] if '_' in cfg else None, 6))
        rows.append([cfg.replace('_', ' '), f'{cost:.0f}' if cost else '-', 'pending',
                     f"{lf6.get('n_spawned', 0)/N_ION:.1f}" if lf6 else '-', f"{lf6.get('n_rng', 0)/N_ION:.0f}" if lf6 else '-',
                     f'{rmax:.1e}', f"{lf6.get('E_stochastic', 0)/max(lf6.get('E_drag', 1), 1e-300):+.1e}" if lf6 else '-',
                     (f'{yv[0]*100:+.3f} % ({yv[1]*100:+.3f} %)' if yv else ('no pairs' if mode == 'none' else '-'))])
        if cost and mode not in seen:
            seen.add(mode); num('8', f'{mode}: Mac CPU smoke cost (ns per ion-step, 2^16 ions, delta spectrum, perf_smoke on 5e1da1d5)', cost, os.path.basename(PERF))
        if lf6: num('8', f'{cfg}: macro-pairs per ion, RNG draws per ion (Gamma_p0 = 6, double)', f"{lf6.get('n_spawned', 0)/N_ION:.2f}, {lf6.get('n_rng', 0)/N_ION:.1f}", f'demo_g6_{cfg}_double')
    fig = plt.figure(figsize=(12, 6.6))
    ax = fig.add_axes([0.01, 0.60, 0.98, 0.35]); ax.axis('off')
    tb = ax.table(cellText=rows, colLabels=['mode', 'Mac CPU smoke\nns / ion-step', 'A100\nns / ion-step', 'macro-pairs\nper ion (G6)',
                                             'RNG draws\nper ion (G6)', 'max |identity|\n/ E_drag (28 runs)', 'E_stochastic\n/ E_drag (G6)',
                                             'pair number vs physical\n(G6, expected)'],
                  loc='upper center', cellLoc='center')
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.55)
    ax2 = fig.add_axes([0.01, 0.0, 0.98, 0.56]); ax2.axis('off')
    tb2 = ax2.table(cellText=status_rows, colLabels=['registered test (merged head e7ba238c)', 'what it checks', 'status', 'source'],
                    loc='upper center', cellLoc='left', colWidths=[0.21, 0.33, 0.22, 0.24])
    tb2.auto_set_font_size(False); tb2.set_fontsize(7); tb2.scale(1, 1.32)
    fig.suptitle('Figure 8. Mode comparison and the registered tests on the merged head')
    save(fig, 'fig8_modes', """
Figure 8. Top: per mode, the stage cost per ion-step from the Mac CPU smoke of the scaling harness (OpenMP, 2^16 ions,
delta spectrum, rebased head 5e1da1d5; a smoke of the mechanics, not a performance measurement; the A100 column is filled
when the scaling job returns), macro-pairs and RNG draws per ion over the Gamma_p0 = 6 demonstration run, the largest
ledger-identity residual over all rows of the 28 runs, E_stochastic / E_drag, and the whole-run pair number against the
continuum integral of dK/<D> over the paid interval with each mode's expected offset in brackets (figure 4: poisson none;
budget the emission lag; event the finite jump). Bottom: the registered tests and their
status on the merged head, with the source of each verdict. Numbers: numbers.md, figure 8.""")

def status_table():
    e7 = {}
    for prec in ('single', 'double'):
        p = os.path.join(OUT, f'e7_born_{prec}', 'run.log')
        if os.path.exists(p):
            txt = open(p).read(); m = re.search(r'bh_event summary: (\d+) passed, (\d+) failed', txt)
            if m: e7[prec] = f'{m[1]}/{int(m[1]) + int(m[2])}'
    e7s = ('pass: ' + ', '.join(f'{v} {k}' for k, v in e7.items())) if e7 else 'running'
    return [
        ['T5, T5fine, T5b', 'continuous cooling vs RK4; dt convergence; float design', 'pass', 'integration ctest (born report §7.4)'],
        ['T7a, T7b, T11, T12, T13', 'fields; massive accounting; capacity; Gauss', 'pass', 'integration ctest'],
        ['T8 (Serial)', 'bit identity; explicit ledger snapshots', 'pass', 'Codex 1 reclosure R2'],
        ['A7 Born (unweighted, all-node weighted)', '<D> per node, 4 s.e., no floor, 2169 nodes', 'pass (3.63 / 3.80 s.e.)', 'Codex 1 closure C3'],
        ['A12 Born', 'cost 1 %, number and energy 2 % per node', 'pass (0.78 %)', 'Codex 1 closure C7'],
        ['E1 restframe / delta', 'pairs vs ions from the arrays (single 1e-5)', 'pass / single delta: exception', 'revision 1.1 decision; fig. 6'],
        ['E2 delta, E10 (restframe on Born)', 'event mean vs Jensen-corrected ODE, 4 s.e.', 'pass (<= 2.4 s.e.)', 'Codex 1 closure C6'],
        ['E3 (domain of rev. 1.1.1)', 'drift-feedback variance, 4 sqrt(2/N)', 'Born: pass; fit G3 single: R4', 'R4 diagnostic: Nilay disposes'],
        ['E4, E5, E6, E8', 'spectra (AD); mu_d; dt; failure paths', 'pass', 'event round 2, closure'],
        ['E7 on Born (a = 1, 2, 4)', 'accelerated event mode, E2/E3', e7s, 'this step (e7_born)'],
        ['E9 (MPI)', 'ledger over ranks', 'deferred (no MPI on Mac)', 'event contract'],
        ['M6 stride n = 16 (harness)', 'frozen-hazard check at lambda = 0.8', 'observed exceedance', 'harness report §7'],
    ]

def write_numbers():
    with open(os.path.join(FIG, 'numbers.md'), 'w') as f:
        f.write('# Demonstration numbers (P2 step 8; every number shown on a figure, with the run it came from)\n\n')
        f.write('Runs: `Entity Local Experiments/p2-demo-20261031/<run>/` (reduced.json, reduced.npz, run.log, ledger); merged head e7ba238c, '
                'branch dev/hadronic-demo; generated by `tests/hadronic/demo/figures.py`. For Codex 1\'s check.\n\n')
        cur = None
        for fig, what, val, run in NUM:
            if fig != cur:
                f.write(f'\n## Figure {fig}\n\n| quantity | value | run |\n| --- | --- | --- |\n'); cur = fig
            f.write(f'| {what} | {val} | `{run}` |\n')

if __name__ == '__main__':
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        try:
            f(); print('done', f.__name__, flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print('FAILED', f.__name__, e, flush=True)
    fig8(status_table()); print('done fig8')
    write_numbers(); print('numbers.md:', len(NUM), 'entries')
