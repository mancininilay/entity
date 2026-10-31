"""Writes jobs.json for run_demo.py: the 28 demonstration runs (14 inputs x single/double), the 6 spectrum runs and the
staircase run in both precisions, and the two e7_born ctest-equivalent runs (bh_event.xc e7 with BH_QK_SOURCE=born).
Longest first. Usage: make_jobs.py <worktree> <builds dir> > jobs.json"""
import json, os, sys
wt, builds = sys.argv[1], sys.argv[2]
inp = os.path.join(wt, 'pgens/bh_proto/inputs/demo'); refdir = os.path.join(wt, 'tests/data/hadronic/bhref-0.2.0')
PAIRS = {6: 647.5e5 * 1.15 + 1e5, 3: 279.7e5 * 1.15 + 1e5}; BUDGET = {6: 64.75e5 * 2.5 + 1e5, 3: 27.97e5 * 2.5 + 1e5}
jobs = []
for prec in ('single', 'double'):
    exe = os.path.join(builds, f'demo-{prec}/src/entity.xc'); bpp = 64 if prec == 'double' else 46
    for g in (3, 6):
        for cfg in ('none', 'poisson_delta', 'poisson_restframe', 'budget_delta', 'budget_restframe', 'event_delta', 'event_restframe'):
            name = f'demo_g{g}_{cfg}'
            cap = PAIRS[g] if cfg.startswith(('poisson', 'event')) else (BUDGET[g] if cfg.startswith('budget') else 1024)
            jobs.append(dict(name=f'{name}_{prec}', input=os.path.join(inp, name + '.toml'), exe=exe, build=prec, G0=float(g), refdir=refdir,
                             threads=2, mem_gb=0.4 + 2 * cap * bpp / 1e9, order=(0 if g == 3 else 1)))
    for g in (3, 10):
        for cfg in ('poisson', 'budget_r025', 'event'):
            name = f'spec_g{g}_{cfg}'
            jobs.append(dict(name=f'{name}_{prec}', input=os.path.join(inp, name + '.toml'), exe=exe, build=prec, G0=float(g), refdir=refdir,
                             threads=2, mem_gb=0.4 + 2 * 3e6 * bpp / 1e9, order=3))
    jobs.append(dict(name=f'stair_g6_none_{prec}', input=os.path.join(inp, 'stair_g6_none.toml'), exe=exe, build=prec, G0=6.0, refdir=refdir,
                     threads=1, mem_gb=0.3, order=3))
    ev = os.path.join(builds, f'demo-{prec}/tests/kernels/test-kernels-bh_event.xc')
    jobs.append(dict(name=f'e7_born_{prec}', kind='cmd', cmd=f'"{ev}" e7', exe=ev, build=prec, threads=2, mem_gb=1.0,
                     env={'BH_QK_SOURCE': 'born'}, order=0))
jobs.sort(key=lambda j: (j['order'], -j['mem_gb']))
print(json.dumps(jobs, indent=1))
