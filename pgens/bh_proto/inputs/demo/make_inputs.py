"""Generates the demonstration inputs (P2 step 8) from demo_template.toml.
The 28 runs of plan §1 are 14 inputs x 2 precisions (the precision is the build): Gamma_p0 in {3, 6} x
{continuous none; continuous poisson a = 1 (delta, restframe); continuous budget (delta, restframe); event a = 1
(delta, restframe)}. N_ion = 1e5; tau0 so that lambda = R dt / a = 0.05 at Gamma_p0 (the E2/E10 setting, coeff from the
Born table: 0.2564289235 at 6, 7.862350556 at 3); runtime to <K>/K0 = 0.5 (6) and 0.65 (3) with margin; 100 ledger
rows (and per-ion dumps) per run. Budget: t_latency = 0 (latency cap off, as in the registered T10/A12 tests; the
production auto cap would emit a packet per ion every ~13 steps here). Pair capacity from the integral of dK/<D>
(poisson, event) or dK/r_th (budget) along the plain ODE, x 1.15 + 1e5; on_capacity_estimate = "warn" because the
startup estimate (III.8) assumes the initial rate for the whole run (no cooling) and would refuse the full-length runs;
the runtime capacity policy stays "error" (an actual overflow aborts the run).
Extra runs (stated in the report): spectrum runs at nearly fixed Gamma_p in {3, 10} for poisson, budget (r_th_factor 0.25)
and event, restframe, N_ion = 2.5e5, 40 steps (cooling 0.4 % at 3, 0.14 % at 10); a staircase run (continuous none, 4 ions,
lambda = 5e-5, 4000 steps, dumps every 10 steps) for figure 6."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
TPL = open(os.path.join(HERE, 'demo_template.toml')).read()
MU_R = 1836.15
TAU0 = {6.0: '0.2564289235', 3.0: '7.862350556', 10.0: '0.09136836671', 'stair': '2.564289235e-4'}
PAIRS_PER_ION = {6.0: 647.5, 3.0: 279.7}
BUDGET_PER_ION = {6.0: 64.75, 3.0: 27.97}
RUNTIME = {6.0: 19800.0, 3.0: 65000.0}       # steps 39600 and 130000 (dt = 0.5)
LEDGER_EVERY = {6.0: 396, 3.0: 1300}         # 100 rows

def mode_block(cfg, spec):
    return _mode_block(cfg, spec) + ('' if cfg == 'none' else '\n  on_capacity_estimate = "warn"')

def _mode_block(cfg, spec):
    if cfg == 'none':
        return '  mode                = "continuous"\n  source              = "none"'
    if cfg == 'poisson':
        return f'  mode                = "continuous"\n  source              = "poisson"\n  spectrum            = "{spec}"\n  pair_weight         = 1.0'
    if cfg == 'budget':
        return f'  mode                = "continuous"\n  source              = "budget"\n  spectrum            = "{spec}"\n  t_latency           = 0.0'
    if cfg == 'budget_r025':
        return f'  mode                = "continuous"\n  source              = "budget"\n  spectrum            = "{spec}"\n  t_latency           = 0.0\n  r_th_factor         = 0.25'
    if cfg == 'event':
        return f'  mode                = "event"\n  spectrum            = "{spec}"\n  pair_weight         = 1.0\n  f_max               = 0.01'
    if cfg == 'event_f02':  # Gamma_p = 10: a_max = 0.84 at f_max = 0.01; f_max = 0.02 as in the registered E4 at Gamma_p = 10
        return f'  mode                = "event"\n  spectrum            = "{spec}"\n  pair_weight         = 1.0\n  f_max               = 0.02'
    raise ValueError(cfg)

def write(name, G0, cfg, spec, nion, runtime, ledger_every, maxpair, seed, comment, diag=100000000, tau0=None):
    K = MU_R * (G0 - 1.0)
    s = TPL
    for k, v in {'@NAME@': name, '@RUNTIME@': f'{runtime:.1f}', '@SEED@': str(seed), '@MAXNPART_ION@': f'{nion * 1.05 + 64:.1f}',
                 '@MAXNPART_PAIR@': f'{maxpair:.1f}', '@TAU0@': tau0 or TAU0[G0], '@GAMMA_P0@': f'{G0:.1f}', '@KION@': f'{K:.10g}',
                 '@NION@': str(nion), '@LEDGER_EVERY@': str(ledger_every), '@MODE_BLOCK@': mode_block(cfg, spec),
                 '@DIAG@': str(diag), '@COMMENT@': comment}.items():
        s = s.replace(k, v)
    open(os.path.join(HERE, name + '.toml'), 'w').write(s)
    return name

names = []
seed = 20261031
for G0 in (6.0, 3.0):
    g = int(G0)
    for cfg, spec in (('none', None), ('poisson', 'delta'), ('poisson', 'restframe'), ('budget', 'delta'), ('budget', 'restframe'),
                      ('event', 'delta'), ('event', 'restframe')):
        nion = 100000
        if cfg in ('poisson', 'event'):
            maxpair = PAIRS_PER_ION[G0] * nion * 1.15 + 1e5
        elif cfg == 'budget':
            # x 2.5: the integral of dK/r_th misses the K0 trigger (a packet each time the ion has cooled by f_r K0 since
            # the last one), which supplied half of the packets in the first Gamma_p0 = 3 budget run (aborted on
            # capacity at step 33174, n_k0_trigger 1616550 of n_spawned 3316550)
            maxpair = BUDGET_PER_ION[G0] * nion * 2.5 + 1e5
        else:
            maxpair = 1024
        name = f'demo_g{g}_{cfg}' + (f'_{spec}' if spec else '')
        seed += 1
        names.append(write(name, G0, cfg, spec, nion, RUNTIME[G0], LEDGER_EVERY[G0], maxpair, seed,
                           f'Gamma_p0 = {G0:g}, {cfg}' + (f', {spec}' if spec else '') + ', N_ion = 1e5, lambda = 0.05 at Gamma_p0'))
# spectrum runs: nearly fixed Gamma_p (40 steps), restframe
for G0 in (3.0, 10.0):
    g = int(G0)
    for cfg in ('poisson', 'budget_r025', 'event'):
        seed += 1
        cfg_mode = 'event_f02' if (cfg == 'event' and G0 == 10.0) else cfg
        names.append(write(f'spec_g{g}_{cfg}', G0, cfg_mode, 'restframe', 250000, 20.0, 40, 3.0e6, seed,
                           f'spectrum run at Gamma_p0 = {G0:g}, {cfg}, restframe, 40 steps, N_ion = 2.5e5'))
# staircase run (figure 6): continuous none, 4 ions at Gamma_p0 = 6, lambda = 5e-5 (tau0 / 1000): the loss per step
# (4.2e-4) is 0.43 of the single-precision ulp-energy at K = 9180, so a plain float update would not cool at all; the
# pending payload accumulates the loss and applies it when it reaches eta_u K = 0.92 (in both precisions: the 100-ulp
# floor is below it); 20000 steps (about nine applications), ledger and per-ion dump every 50 steps
seed += 1
names.append(write('stair_g6_none', 6.0, 'none', None, 4, 10000.0, 50, 1024, seed,
                   'staircase run: continuous none, 4 ions at Gamma_p0 = 6, lambda = 5e-5, 20000 steps, dumps every 50 steps',
                   tau0=TAU0['stair']))
print('\n'.join(names))
