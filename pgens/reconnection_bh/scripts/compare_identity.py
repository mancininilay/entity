#!/usr/bin/env python3
"""Bit-for-bit comparison of the output of two runs (standard library only).

    compare_identity.py <bp2raw> <output dir of run A> <output dir of run B> <work dir>

Every variable of every BP file under fields/, particles/ and spectra/ of
run A is extracted with bp2raw from both runs and compared byte by byte,
together with the dimensions; the statistics file (*_stats.csv) is compared
as text. Exit code 0 only if nothing differs and nothing is missing.
"""
import hashlib
import subprocess
import sys
from pathlib import Path


def extract(bp2raw, bp, out):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([bp2raw, str(bp), str(out)], check=True)
    index = {}
    for line in (out / "index.txt").read_text().splitlines():
        t = line.split()
        index[(t[0], t[1])] = (t[2], tuple(t[3:]))
    return index


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest(), path.stat().st_size


def main():
    bp2raw, a, b, work = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    n_var = n_bytes = n_diff = 0
    for kind in ("fields", "particles", "spectra"):
        files = sorted((a / kind).glob("*.bp"))
        if not files:
            print(f"MISSING: no {kind} output in {a}")
            n_diff += 1
        for fa in files:
            fb = b / kind / fa.name
            if not fb.exists():
                print(f"MISSING in B: {kind}/{fa.name}")
                n_diff += 1
                continue
            ia = extract(bp2raw, fa, work / "a" / kind / fa.name)
            ib = extract(bp2raw, fb, work / "b" / kind / fa.name)
            for key in sorted(set(ia) | set(ib)):
                if key not in ia or key not in ib or ia[key] != ib[key]:
                    print(f"DIFFERENT layout: {kind}/{fa.name} {key}: {ia.get(key)} vs {ib.get(key)}")
                    n_diff += 1
                    continue
                name = f"{key[0]}.{key[1]}.raw"
                da, size = digest(work / "a" / kind / fa.name / name)
                db, _ = digest(work / "b" / kind / fa.name / name)
                n_var += 1
                n_bytes += size
                if da != db:
                    print(f"DIFFERENT data: {kind}/{fa.name} {key[0]} ({ia[key][0]}, {size} bytes)")
                    n_diff += 1
            steps = (work / "a" / kind / fa.name / "Step.0.raw").read_bytes()
            print(f"{kind}/{fa.name}: step {int.from_bytes(steps, 'little')}, {len(ia)} variables")
    sa, sb = sorted(a.glob("*_stats.csv")), sorted(b.glob("*_stats.csv"))
    if len(sa) != 1 or len(sb) != 1 or sa[0].read_bytes() != sb[0].read_bytes():
        print("DIFFERENT statistics file")
        n_diff += 1
    else:
        print(f"statistics file: identical ({len(sa[0].read_text().splitlines()) - 1} rows)")
    print(f"compared {n_var} variables, {n_bytes} bytes: "
          + ("IDENTICAL" if n_diff == 0 else f"{n_diff} DIFFERENCES"))
    return 0 if n_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
