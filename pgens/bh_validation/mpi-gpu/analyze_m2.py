#!/usr/bin/env python3
"""Verdicts of M2 (PREREGISTRATION.md of this folder). Standard library only.

    analyze_m2.py <output root> [precisions]
Runs: <root>/m2/<precision>/r<ranks>/<name>/run.log, ranks 2 and 4. r(t) is the
largest |div E - kappa rho| over the nodes (V1); the control is the same run with
the policy off.
"""
import re
import sys
from pathlib import Path


ANSI = re.compile(r"^(?:\x1b\[[0-9;]*[A-Za-z])+")


def lines(path, key):
    out = []
    if not path.exists():
        return out
    for line in open(path, errors="replace"):
        # the engine's terminal control codes can precede the first line a run prints
        line = ANSI.sub("", line)
        if line.startswith(key + " step"):
            t = line.split()
            out.append({t[i]: float(t[i + 1]) for i in range(1, len(t) - 1, 2)})
    return out


# name: (control, kind, registered)
RUNS = {
    "A_on": ("A_off", "pass", True), "B_on": ("B_off", "pass", True),
    "A_on_stale": ("A_off", "pass", False), "B_on_stale": ("B_off", "pass", False),
    "A_neg_stale": ("A_off", "negative", None), "B_neg_stale": ("B_off", "cannot_fail", True),
    "B_birth": ("B_birth_off", "birth", True), "A_birth": ("A_birth_off", "no_birth", True),
}


def main():
    root = Path(sys.argv[1])
    precs = sys.argv[2].split() if len(sys.argv) > 2 else ["single", "double"]
    n_fail = n_verdict = 0
    print(f"{'prec':6s} {'ranks':>5s} {'run':12s} {'pairs':>8s} {'max_t r':>11s} {'max_t r_off':>11s} {'ratio':>10s}  verdict")
    for prec in precs:
        for ranks in (2, 4):
            base = root / "m2" / prec / f"r{ranks}"
            for name, (control, kind, registered) in RUNS.items():
                rows = lines(base / name / "run.log", "BHV")
                rows_off = lines(base / control / "run.log", "BHV")
                if kind == "negative":
                    registered = ranks == 2      # the registered negative control is on 2 ranks
                tag = "" if registered else "  [additional, not registered]"
                # a V1 run has 20 lines; the birth runs one more, after the first step
                want = 21 if "birth" in name else 20
                if len(rows) != want or len(rows_off) != want:
                    print(f"{prec:6s} {ranks:5d} {name:12s} incomplete: {len(rows)} and {len(rows_off)} lines of {want}  FAIL{tag}")
                    n_verdict += bool(registered)
                    n_fail += bool(registered)
                    continue
                r_off = max(r["gauss_r"] for r in rows_off)
                r_max = max(r["gauss_r"] for r in rows)
                pairs = rows[-1]["pairs"]
                ratio = r_max / r_off
                extra = ""
                if kind == "negative":
                    ok = ratio > 1000.0
                    text = "PASS (violates)" if ok else "FAIL (negative control too weak)"
                elif kind == "cannot_fail":
                    ok = ratio <= 10.0 and pairs >= 1.0e4
                    text = ("PASS (as predicted: ions first cannot fail on the missing write)" if ok
                            else "FAIL (prediction wrong: ions first does fail on the missing write)")
                else:
                    ok = ratio <= 10.0 and pairs >= 1.0e4
                    text = "PASS" if ok else "FAIL"
                if kind in ("birth", "no_birth"):
                    first = [x for x in lines(base / name / "run.log", "BHVR") if int(x["step"]) == 0]
                    away = sum(x["po_n"] for x in first if int(x["rank"]) != 0)
                    born = sum(x["pairs"] for x in first)
                    born_away = sum(x["pairs"] for x in first if int(x["rank"]) != 0)
                    extra = (f"; after the first step: {born:.0f} pairs, all on rank 0 ({born_away:.0f} elsewhere), "
                             f"{away:.0f} positrons on the other ranks")
                    if kind == "birth":
                        ok = ok and away >= 5 and born_away == 0
                    else:
                        ok = ok and away == 0 and born_away == 0
                    text = "PASS" if ok else "FAIL"
                if registered:
                    n_verdict += 1
                    n_fail += not ok
                print(f"{prec:6s} {ranks:5d} {name:12s} {pairs:8.0f} {r_max:11.3e} {r_off:11.3e} {ratio:10.3g}  {text}{extra}{tag}")
            # reported: where the leptons are at the end, against where they were born
            for name in ("A_on", "B_on"):
                rl = lines(base / name / "run.log", "BHVR")
                if rl:
                    last = max(int(x["step"]) for x in rl)
                    net = ", ".join(f"rank {int(x['rank'])}: {x['pairs']:.0f} born, {x['po_n']:.0f} present"
                                    for x in rl if int(x["step"]) == last)
                    print(f"{prec:6s} {ranks:5d} {name:12s} positrons at the end: {net}  [reported]")
    print(f"M2 failures: {n_fail} of {n_verdict}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
