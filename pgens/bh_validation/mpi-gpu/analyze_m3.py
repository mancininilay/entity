#!/usr/bin/env python3
"""Verdicts of M3 (PREREGISTRATION.md of this folder). Standard library only.

    analyze_m3.py <output root> [groups] [precisions]
Runs: <root>/<group>/<precision>/r<ranks>/full/ with run.log and rc (the exit status of
the launcher); groups m3 and m3-bounds (the build with bounds-checked views).
"""
import re
import sys
from pathlib import Path


def lines(path, key):
    out = []
    for line in open(path, errors="replace"):
        if line.startswith(key + " step"):
            t = line.split()
            out.append({t[i]: float(t[i + 1]) for i in range(1, len(t) - 1, 2)})
    return out


def main():
    root = Path(sys.argv[1])
    groups = sys.argv[2].split() if len(sys.argv) > 2 else ["m3", "m3-bounds"]
    precs = sys.argv[3].split() if len(sys.argv) > 3 else ["single", "double"]
    n_fail = n_verdict = 0
    for grp in groups:
        for prec in precs:
            for ranks in (2, 4):
                d = root / grp / prec / f"r{ranks}" / "full"
                if not (d / "run.log").exists():
                    continue
                n_verdict += 1
                log = open(d / "run.log", errors="replace").read()
                rc = (d / "rc").read_text().strip() if (d / "rc").exists() else "?"
                steps = 10                       # registered
                bhv, rl = lines(d / "run.log", "BHV"), lines(d / "run.log", "BHVR")
                warn = "Bethe-Heitler pairs are dropped" in log
                engine = "Too many particles to receive" in log
                bounds = re.search(r"(out of bounds|View bounds error|Kokkos::View ERROR)[^\n]*", log)
                last = max((int(x["step"]) for x in rl), default=-1)
                print(f"M3 {grp} {prec} {ranks} ranks: launcher exit status {rc}; diagnostic lines after {len(bhv)} of "
                      f"{steps} steps")
                for x in rl:
                    if int(x["step"]) == last:
                        print(f"    after step {last + 1}, rank {int(x['rank'])}: emitters {x['ion_n']:.0f}, pairs "
                              f"{x['pairs']:.0f}, dropped {x['dropped']:.0f}, lepton arrays {x['el_fill']:.0f} and "
                              f"{x['po_fill']:.0f} of {x['el_cap']:.0f}")
                print(f"    policy's warning that pairs are dropped: {'yes' if warn else 'no'}; engine's error "
                      f"'Too many particles to receive': {'yes' if engine else 'no'}; out-of-bounds report: "
                      f"{bounds.group(0) if bounds else 'none'}")
                complete = rc == "0" and len(bhv) == steps
                ok = complete and not bounds
                if complete:
                    row = bhv[-1]
                    full = [x for x in rl if int(x["step"]) == last and x["el_fill"] == x["el_cap"]]
                    others = [x for x in rl if int(x["step"]) == last and x["el_fill"] < x["el_cap"]]
                    ok = (ok and len(full) == 1 and int(full[0]["rank"]) == 0 and full[0]["dropped"] > 0
                          and all(x["dropped"] == 0 for x in others)
                          and row["pairs"] == row["el_n"] == row["po_n"]
                          and row["dropped"] == full[0]["dropped"])
                print(f"    registered criterion (the run completes; rank 0 full, skips and counts; the other ranks go "
                      f"on; counters consistent; nothing out of bounds): {'PASS' if ok else 'FAIL'}")
                n_fail += not ok
    print(f"M3 failures: {n_fail} of {n_verdict}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
