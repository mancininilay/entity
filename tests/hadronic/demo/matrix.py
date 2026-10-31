"""The run matrix of the demonstration campaign as a Markdown table (for the step-8 report): run, build, executable and
input hashes, wall time, pairs, status. Usage: matrix.py <campaign output root>. Analysis code, not a ctest."""
import hashlib, json, os, sys

OUT = sys.argv[1]
M = json.load(open(os.path.join(OUT, 'manifest.json')))
J = {j['name']: j for j in json.load(open(os.path.join(OUT, 'jobs.json')))}

def sha(path):
    try:
        return hashlib.sha256(open(path, 'rb').read()).hexdigest()
    except OSError:
        return None

exes = {}
print('| run | build | input sha256 (12) | wall [s] | macro-pairs | status |')
print('| --- | --- | --- | --- | --- | --- |')
order = sorted([k for k in M if not k.startswith('_')], key=lambda k: (not k.startswith('demo'), k.startswith('e7') or k.startswith('r4'), k))
for k in order:
    m = M[k]
    if m.get('exe_sha256'): exes[(m['build'], m['exe'])] = m['exe_sha256']
    inp = m.get('input'); h = sha(inp)[:12] if inp and sha(inp) else '(command)'
    used = os.path.join(OUT, k, 'input.toml')
    if inp and os.path.exists(used) and sha(used) != sha(inp): h += ' (CHANGED since the run)'
    pairs = '-'
    red = os.path.join(OUT, k, 'reduced.json')
    if m.get('status') == 'done' and os.path.exists(red):
        p = json.load(open(red)).get('pairs', {})
        if p: pairs = f"{p.get('n', 0):,}"
    print(f"| `{k}` | {m.get('build', '-')} | {h} | {m.get('wall_s', '-')} | {pairs} | {m.get('status')} |")
print()
for (b, exe), h in sorted(exes.items()):
    print(f'- {b}: `{os.path.relpath(exe, os.path.join(OUT, ".."))}` sha256 `{h}`')
for k, e in M.get('_superseded', {}).items():
    print(f"- superseded: `{k}` ({e.get('status')}), kept in `superseded/`")
tabs = sorted({(M[k].get('build'), M[k].get('table_sha256')) for k in M if not k.startswith('_') and M[k].get('table_sha256')})
for b, h in tabs:
    print(f'- loss/pair table written by the {b} build (`hadronic_table_{b}.txt`): sha256 `{h}`')
