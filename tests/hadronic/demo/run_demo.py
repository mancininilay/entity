"""Runs the demonstration campaign (P2 step 8): every job runs entity.xc in its own directory under a LOCAL scratch root
(the master folder is iCloud-synced: the multi-GB text dumps never go there), then reduce_run.py --delete, then copies
the compact files (input.toml, run.log, hadronic_ledger_1.txt, reduced.npz, reduced.json; the table file once per build)
to the output root. Jobs run concurrently within a slot limit and a resident-memory budget estimated from the pair
capacity (64 bytes per pair particle in double, 46 in single, measured on the sizing run); OMP_NUM_THREADS per job.
A manifest (manifest.json) records per job: build, binary sha256, commit, threads, start/end, wall seconds, exit code.
Usage: run_demo.py <jobs.json> <scratch root> <output root> [--slots 5] [--mem-gb 26]"""
import hashlib, json, os, shutil, subprocess, sys, time

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def main():
    jobs = json.load(open(sys.argv[1])); scratch = sys.argv[2]; outroot = sys.argv[3]
    slots = int(sys.argv[sys.argv.index('--slots') + 1]) if '--slots' in sys.argv else 5
    memcap = float(sys.argv[sys.argv.index('--mem-gb') + 1]) if '--mem-gb' in sys.argv else 26.0
    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable
    os.makedirs(scratch, exist_ok=True); os.makedirs(outroot, exist_ok=True)
    manpath = os.path.join(outroot, 'manifest.json')
    manifest = json.load(open(manpath)) if os.path.exists(manpath) else {}
    pending = [j for j in jobs if manifest.get(j['name'], {}).get('status') != 'done']
    running = {}
    def launch(j):
        rd = os.path.join(scratch, j['name']); shutil.rmtree(rd, ignore_errors=True); os.makedirs(rd)
        if j.get('kind', 'entity') == 'entity':
            shutil.copy(j['input'], os.path.join(rd, 'input.toml'))
            cmd = f'"{j["exe"]}" -input input.toml > run.log 2>&1'
        else:
            cmd = j['cmd'] + ' > run.log 2>&1'
        env = dict(os.environ, OMP_NUM_THREADS=str(j.get('threads', 2)), OMP_PROC_BIND='false')
        env.pop('BH_QK_SOURCE', None)
        env.update(j.get('env', {}))
        p = subprocess.Popen(['bash', '-c', cmd], cwd=rd, env=env)
        manifest[j['name']] = dict(status='running', build=j.get('build'), exe=j.get('exe'), exe_sha256=sha256(j['exe']) if j.get('exe') else None,
                                   threads=j.get('threads', 2), start=time.strftime('%Y-%m-%d %H:%M:%S'), t0=time.time(), G0=j.get('G0'),
                                   input=j.get('input'), env=j.get('env', {}))
        json.dump(manifest, open(manpath, 'w'), indent=1)
        print(f"[{time.strftime('%H:%M:%S')}] start {j['name']} (mem {j.get('mem_gb', 0.5):.1f} GB, {j.get('threads', 2)} threads)", flush=True)
        return p
    def finish(j, p):
        rd = os.path.join(scratch, j['name']); m = manifest[j['name']]
        m['exit'] = p.returncode; m['end'] = time.strftime('%Y-%m-%d %H:%M:%S'); m['wall_s'] = round(time.time() - m['t0'], 1)
        od = os.path.join(outroot, j['name']); os.makedirs(od, exist_ok=True)
        if j.get('kind', 'entity') == 'entity':
            r = subprocess.run([py, os.path.join(here, 'reduce_run.py'), rd, str(j['G0']), '--refdir', j['refdir'], '--delete'],
                               capture_output=True, text=True)
            m['reduce'] = (r.stdout + r.stderr).strip()[-400:]
            for f in ('input.toml', 'run.log', 'hadronic_ledger_1.txt', 'reduced.npz', 'reduced.json'):
                if os.path.exists(os.path.join(rd, f)): shutil.copy(os.path.join(rd, f), od)
            tf = os.path.join(rd, 'hadronic_table.txt')
            if os.path.exists(tf):
                m['table_sha256'] = sha256(tf)
                dst = os.path.join(outroot, f"hadronic_table_{j['build']}.txt")
                if not os.path.exists(dst): shutil.copy(tf, dst)
            m['status'] = 'done' if (p.returncode == 0 and r.returncode == 0) else 'failed'
        else:
            shutil.copy(os.path.join(rd, 'run.log'), od)
            m['status'] = 'done' if p.returncode == 0 else 'failed'
        json.dump(manifest, open(manpath, 'w'), indent=1)
        print(f"[{time.strftime('%H:%M:%S')}] end {j['name']}: exit {p.returncode}, {m['wall_s']:.0f} s, {m['status']} {m.get('reduce', '')[-120:]}", flush=True)
    while pending or running:
        used = sum(j.get('mem_gb', 0.5) for j, _ in running.values())
        for j in list(pending):
            if len(running) >= slots: break
            if used + j.get('mem_gb', 0.5) > memcap and running: continue
            running[j['name']] = (j, launch(j)); used += j.get('mem_gb', 0.5); pending.remove(j)
        time.sleep(10)
        for name, (j, p) in list(running.items()):
            if p.poll() is not None:
                del running[name]; finish(j, p)
    print('campaign done', flush=True)

if __name__ == '__main__':
    main()
