#!/usr/bin/env python3
"""Verdicts of M1 (PREREGISTRATION.md of this folder). Standard library only.

    analyze_m1.py <output root> <reference/points.txt> [precisions]
Runs: <root>/m1/<precision>/r<ranks>/<case>_{on,off}/ with run.log and input.toml;
cases H (0.5 bath, Gamma_p = 6) and L (1.8 bath, Gamma_p = 1.2); ranks 1, 2, 4.
"""
import math
import re
import sys
from pathlib import Path

MU_R = 1836.15
RANKS = (1, 2, 4)
CASES = ("H", "L")


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


def one(root, prec, ranks, case, ref):
    d = root / "m1" / prec / f"r{ranks}"
    on, off = lines(d / f"{case}_on" / "run.log", "BHV"), lines(d / f"{case}_off" / "run.log", "BHV")
    if not (d / f"{case}_on" / "input.toml").exists():
        return None
    text = open(d / f"{case}_on" / "input.toml").read()
    tau = float(re.search(r"optical_depth\s*=\s*([0-9.eE+-]+)", text).group(1))
    eps0 = float(re.search(r"photon_energy\s*=\s*([0-9.eE+-]+)", text).group(1))
    runtime = float(re.search(r"runtime\s*=\s*([0-9.eE+-]+)", text).group(1))
    if len(on) < 2 or len(off) < 2:
        return None
    dt = (on[-1]["time"] - on[0]["time"]) / (on[-1]["step"] - on[0]["step"])
    n = int(runtime / dt)                       # the number of steps the engine takes
    row, row_off = on[-1], off[-1]
    complete = int(row["step"]) == n - 1 and int(row_off["step"]) == n - 1
    N = row["ion_n"]
    r = ref[case]
    p = tau * r["J"] * dt
    s = r["slope"] * r["D"] / MU_R
    expected = N * p * (n - s * p * n * (n - 1) / 2.0)
    pairs = row["pairs"]
    z_n = (pairs - expected) / math.sqrt(expected)
    loss = row_off["ion_K"] - row["ion_K"]
    d_ion = loss / pairs
    # sum over the emitters of (their loss)^2, from the sums of (K - K_ref) and (K - K_ref)^2
    sd2 = row["ion_dK2"] - row_off["ion_dK2"] + 2.0 * (row_off["ion_dK"] / N) * loss
    cross = N * n * (n - 1) * p * p * r["D"] ** 2    # emitters that interacted twice
    var_d = (sd2 - cross) / pairs - d_ion ** 2
    se_d = math.sqrt(var_d / pairs)
    z_d = (d_ion - r["D"]) / se_d
    closed = (row["el_n"] == pairs and row["po_n"] == pairs and row["dropped"] == 0 and
              row["above"] == 0 and row["pexc"] == 0 and row["bounded"] == 0 and row["ion_n"] == row_off["ion_n"])
    off_zero = all(row_off[k] == 0 for k in ("pairs", "el_n", "po_n", "absorbed"))
    book = (row["el_E"] + row["po_E"] - eps0 * pairs) / loss - 1.0
    # per rank: pairs against p times the emitters present before each step
    rl = lines(d / f"{case}_on" / "run.log", "BHVR")
    per_rank = []
    for k in range(ranks):
        mine = [x for x in rl if int(x["rank"]) == k]
        if len(mine) != n:
            per_rank = None
            break
        # emitters on the rank before step j: those after step j - 1 (after step 0 for j = 0)
        exposure = mine[0]["ion_n"] + sum(x["ion_n"] for x in mine[:-1])
        e_k = p * exposure
        per_rank.append((mine[-1]["pairs"], e_k, (mine[-1]["pairs"] - e_k) / math.sqrt(e_k),
                         mine[-1]["el_n"], mine[-1]["po_n"], mine[-1]["dropped"], mine[-1]["ion_n"]))
    return dict(prec=prec, ranks=ranks, case=case, n=n, dt=dt, tau=tau, p=p, N=N, complete=complete,
                pairs=pairs, expected=expected, z_n=z_n, d_ion=d_ion, D=r["D"], se_d=se_d, z_d=z_d,
                std_d=math.sqrt(var_d), closed=closed, off_zero=off_zero, book=book,
                absorbed=row["absorbed"] / pairs, per_rank=per_rank, row=row)


def main():
    root, ref_path = Path(sys.argv[1]), Path(sys.argv[2])
    precs = sys.argv[3].split() if len(sys.argv) > 3 else ["single", "double"]
    ref = {}
    for line in open(ref_path):
        if line.startswith("#"):
            continue
        t = line.split()
        ref[t[0]] = dict(eps0=float(t[1]), g=float(t[2]), J=float(t[4]), D=float(t[6]), slope=float(t[7]))
    n_fail = n_verdict = 0

    def verdict(ok):
        nonlocal n_fail, n_verdict
        n_verdict += 1
        n_fail += not ok
        return "PASS" if ok else "FAIL"

    for prec in precs:
        for case in CASES:
            res = {}
            for ranks in RANKS:
                x = one(root, prec, ranks, case, ref)
                tag = f"M1 {case} (eps0 {ref[case]['eps0']}, Gamma_p {ref[case]['g']}) {prec} {ranks} rank(s)"
                if x is None:
                    print(f"{tag}: no output  {verdict(False)}")
                    continue
                res[ranks] = x
                print(f"{tag}: steps {x['n']} dt {x['dt']:.9g} tau {x['tau']:.9g} p/step {x['p']:.6g} emitters {x['N']:.0f}")
                print(f"  run complete (last line after step {x['n']}): {verdict(x['complete'])}")
                print(f"  pairs: observed {x['pairs']:.0f} expected {x['expected']:.1f} "
                      f"(observed/expected - 1 = {x['pairs'] / x['expected'] - 1:+.3e}) z = {x['z_n']:+.2f}  "
                      f"{verdict(abs(x['z_n']) <= 4.0 and x['pairs'] >= 1e5)}")
                print(f"  mean loss: observed {x['d_ion']:.6f} (ion side) bhref {x['D']:.6f} "
                      f"(observed/bhref - 1 = {x['d_ion'] / x['D'] - 1:+.3e}) s.e. {x['se_d']:.2e} "
                      f"z = {x['z_d']:+.2f}  {verdict(abs(x['z_d']) <= 4.0)}")
                print(f"  closed box: pairs = electrons = positrons present ({x['row']['el_n']:.0f}, "
                      f"{x['row']['po_n']:.0f}), above {x['row']['above']:.0f} pexc {x['row']['pexc']:.0f} "
                      f"bounded {x['row']['bounded']:.0f} dropped {x['row']['dropped']:.0f}; emitters conserved: "
                      f"{verdict(x['closed'])}; run without the policy all zero: {verdict(x['off_zero'])}")
                if x["per_rank"] is None:
                    print(f"  per rank: lines missing  {verdict(False)}")
                else:
                    tot = sum(v[0] for v in x["per_rank"])
                    print(f"  per rank: sum of the pairs of the ranks {tot:.0f} = total: {verdict(tot == x['pairs'])}")
                    for k, v in enumerate(x["per_rank"]):
                        print(f"    rank {k}: pairs {v[0]:.0f} expected {v[1]:.1f} z = {v[2]:+.2f}  "
                              f"{verdict(abs(v[2]) <= 4.0)}; at the end emitters {v[6]:.0f}, electrons {v[3]:.0f}, "
                              f"positrons {v[4]:.0f}, dropped {v[5]:.0f}")
                print(f"  reported: std of D {x['std_d']:.4f}; lepton-side sum of D / ion loss - 1 = {x['book']:+.2e}; "
                      f"absorbed/pair {x['absorbed']:.9f}")
            for a, b in ((1, 2), (1, 4), (2, 4)):
                if a in res and b in res:
                    za = (res[a]["pairs"] - res[b]["pairs"]) / math.sqrt(res[a]["expected"] + res[b]["expected"])
                    zd = (res[a]["d_ion"] - res[b]["d_ion"]) / math.hypot(res[a]["se_d"], res[b]["se_d"])
                    print(f"M1 {case} {prec} {a} against {b} ranks: pairs z = {za:+.2f}  {verdict(abs(za) <= 4.0)}; "
                          f"mean loss z = {zd:+.2f}  {verdict(abs(zd) <= 4.0)}")
    print(f"M1 failures: {n_fail} of {n_verdict}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
