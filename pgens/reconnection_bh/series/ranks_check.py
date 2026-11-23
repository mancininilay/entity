#!/usr/bin/env python3
"""Check 1, the part per rank, for a run on several ranks (Claude 1, 2026-11-23; standard library).

    ranks_check.py <run directory>

From bh_ranks.txt (one line per rank every `rank_interval` steps: npart and maxnpart of every
species, then the pairs and the pairs dropped for capacity of that rank) and bh_counters.txt:
  - at the last step written in both files, the pairs of the ranks sum to the pairs of the
    counters, and the dropped pairs likewise;
  - at no line does a species hold more than 90 percent of its capacity on its rank.
Reported: the largest fill of every species over the run, with its rank and time.
Exit 0 pass, 1 fail.
"""
import sys
from pathlib import Path

FILL = 0.9


def main():
    run = Path(sys.argv[1])
    rows = [l.split() for l in (run / "bh_ranks.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    head = (run / "bh_counters.txt").read_text().splitlines()[0][1:].split()
    cnt = [l.split() for l in (run / "bh_counters.txt").read_text().splitlines()[1:] if l.strip()]
    nspec = (len(rows[0]) - 5) // 2
    worst = {s: (0.0, None, None) for s in range(1, nspec + 1)}
    over = []
    for r in rows:
        for s in range(1, nspec + 1):
            n, cap = float(r[3 + 2 * (s - 1)]), float(r[4 + 2 * (s - 1)])
            f = n / cap if cap > 0 else 0.0
            if f > worst[s][0]:
                worst[s] = (f, r[2], float(r[1]))
            if f > FILL:
                over.append((s, r[2], r[1], f))
    steps_r = {r[0] for r in rows}
    steps_c = {c[0]: c for c in cnt}
    common = sorted(steps_r & set(steps_c), key=float)
    ok = True
    print(f"ranks check: {run.name}; {nspec} species; {len(rows)} lines of bh_ranks.txt")
    if not common:
        print("  no step written in both files  FAIL")
        ok = False
    else:
        last = common[-1]
        per = [r for r in rows if r[0] == last]
        pairs = sum(float(r[-2]) for r in per)
        dropped = sum(float(r[-1]) for r in per)
        c = steps_c[last]
        cp, cd = float(c[head.index("pairs")]), float(c[head.index("capacity_dropped")])
        same = pairs == cp and dropped == cd
        ok &= same
        print(f"  step {last}: pairs of the {len(per)} ranks {pairs:.0f} (" + " + ".join(f"{float(r[-2]):.0f}" for r in per)
              + f"), counters {cp:.0f}; dropped {dropped:.0f} against {cd:.0f}  {'pass' if same else 'FAIL'}")
    for s in range(1, nspec + 1):
        f, rank, t = worst[s]
        print(f"  species {s}: largest fill {f:.4f} (rank {rank}, t = {t})")
    if over:
        ok = False
        print(f"  {len(over)} lines above {FILL:g} of the capacity, the first: species {over[0][0]}, rank {over[0][1]}, "
              f"time {over[0][2]}, fill {over[0][3]:.4f}  FAIL")
    else:
        print(f"  no species above {FILL:g} of its capacity on any rank  pass")
    print(f"ranks check verdict: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
