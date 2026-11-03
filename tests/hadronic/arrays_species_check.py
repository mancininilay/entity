#!/usr/bin/env python3
"""D2 fix (2026-11-03): checks one bh_proto run of arrays_species.sh from its log and its complete per-species dumps.

Independent of the generator's code path: the dumps are read here and summed with math.fsum, species by species, without
the generator's double-double accumulators. Checks, each a PASS or FAIL line:
  dump      every rank's file holds max(npart-, npart+) rows and each side's slots up to its own count (tag -1 beyond it);
            the global counts equal the ledger's npart_minus and npart_plus
  sums      sum w gamma and the photon term of the printed "array identity" line equal the sums recomputed from the dumps
  photon    half the e- payload sum plus half the e+ payload sum equals each species' full sum (every pair complete) and
            the ledger's E_absorbed: the photon energy is counted exactly once
  partners  the alive born e- and e+ multisets of (birth, parent Gamma_p, eps, w) are equal: every lepton has its partner
            somewhere, with the same stored weight and eps
  complete  n- = n+ = n_spawned, and the "array leptons" line says so
  closure   |residual - E_round| <= tol max(|E_drag|, 1), tol 1e-5 single, 1e-12 double (Codex 1's array E1 gate); the raw
            |residual| likewise, except single precision delta (the registered revision 1.1 exception)
  old       the old index-paired sum (slot k of e- with slot k of e+, both alive and born; pgen.hpp at 1a3c3fd2) against the
            same right-hand side: --old pass or fail states what it must do (the negative control)
Absorbing box (--box absorbing, one rank, slot k of the two species is a true pair): the line must report incomplete
pairs, at least one lone e- and one lone e+ must exist, and the printed photon term must equal
sum_both w eps + 1/2 sum_lone w eps: a lone survivor keeps its own half and nothing is counted twice.
Exit: 0 all passed, 1 a check failed, 2 a file or line could not be read. Python 3.6 compatible.
"""
import argparse
import glob
import math
import os
import re
import sys

COLS = 25
fsum = math.fsum
n_pass = n_fail = 0


def verdict(ok, name, what):
    global n_pass, n_fail
    print("{}: {}: {}".format("PASS" if ok else "FAIL", name, what))
    if ok:
        n_pass += 1
    else:
        n_fail += 1


def gamma(u):
    return math.sqrt(1.0 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2])


def read_dumps(d, ranks):
    """Per rank: (n_minus, n_plus, rows); rows are lists of 25 floats."""
    paths = [os.path.join(d, "hadronic_pairs.txt")] if ranks == 1 else sorted(glob.glob(os.path.join(d, "hadronic_pairs_rank*.txt")))
    if len(paths) != ranks:
        raise RuntimeError("expected {} dump file(s), found {}".format(ranks, len(paths)))
    out = []
    for p in paths:
        nm = npl = n = None
        rows = []
        with open(p) as f:
            for line in f:
                if line.startswith("#"):
                    m = re.match(r"# n (\d+) npart_minus (\d+) npart_plus (\d+)", line)
                    if m:
                        n, nm, npl = (int(x) for x in m.groups())
                    continue
                v = [float(x) for x in line.split()]
                if len(v) != COLS:
                    raise RuntimeError("{}: a row of {} columns".format(p, len(v)))
                rows.append(v)
        if n is None:
            raise RuntimeError("{}: no count header".format(p))
        out.append((n, nm, npl, rows))
    return out


def side(row, sign):
    """(w, u, tag, birth, Gp, eps) of one side of a dump row; sign -1 for e-, +1 for e+."""
    if sign < 0:
        return row[5], row[7:10], row[13], row[15], row[16], row[17]
    return row[6], row[10:13], row[14], row[18], row[19], row[20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--precision", choices=("single", "double"), required=True)
    ap.add_argument("--spectrum", choices=("delta", "restframe"), required=True)
    ap.add_argument("--ranks", type=int, required=True)
    ap.add_argument("--box", choices=("periodic", "absorbing"), default="periodic")
    ap.add_argument("--old", choices=("pass", "fail", "none"), default="none")
    a = ap.parse_args()
    name = a.name
    tol = 1e-5 if a.precision == "single" else 1e-12
    log = open(os.path.join(a.dir, "run.log"), errors="replace").read()
    m = re.search(r"array identity species 1 : sum_pairs w_s \(E- \+ E\+ - eps_s\) = (\S+) \((\d+) kernel-born pairs, "
                  r"sum w_pair = (\S+), sum w_s E = (\S+), sum w_s eps = (\S+)\), rhs = .*? = (\S+), residual = (\S+), "
                  r"E_round \(ledger\) = (\S+), residual - E_round = (\S+),", log)
    lep = re.search(r"array leptons species 1 : alive kernel-born e- (\d+) e\+ (\d+), n_spawned (\d+): (.*)", log)
    led = re.search(r"ledger species 1 : E_drag (\S+) E_pairs (\S+) E_absorbed (\S+) .*? n_spawned (\d+) .*? "
                    r"npart_minus (\d+) npart_plus (\d+)", log)
    if not (m and lep and led):
        raise RuntimeError("missing array identity, array leptons or ledger line")
    net, npairs, sw, sE, seps, rhs, res, er, corr = (float(x) for x in m.groups())
    n_m, n_p, n_sp = (int(x) for x in lep.groups()[:3])
    e_drag, e_pairs, e_abs = (float(x) for x in led.groups()[:3])
    gm_np, gp_np = int(led.group(5)), int(led.group(6))
    scale = max(abs(e_drag), 1.0)
    dumps = read_dumps(a.dir, a.ranks)

    # dump: every slot of each species with its own count
    ok = True
    for n, nm, npl, rows in dumps:
        ok = ok and n == max(nm, npl) == len(rows)
        for k, r in enumerate(rows):
            ok = ok and (r[13] != -1) == (k < nm) and (r[14] != -1) == (k < npl)
    verdict(ok, name, "dump: rows = max(npart-, npart+) in every file, each side complete to its own count ({})".format(
        ", ".join("{}/{}".format(x[1], x[2]) for x in dumps)))
    verdict(sum(x[1] for x in dumps) == gm_np and sum(x[2] for x in dumps) == gp_np, name,
            "dump: slots over the ranks {} / {} = the ledger's npart {} / {}".format(sum(x[1] for x in dumps), sum(x[2] for x in dumps), gm_np, gp_np))

    # species-wise sums over every alive born lepton of every rank
    lep_rows = {-1: [], 1: []}
    for n, nm, npl, rows in dumps:
        for sign, count in ((-1, nm), (1, npl)):
            for r in rows[:count]:
                w, u, tag, birth, Gp, eps = side(r, sign)
                if tag == 1 and birth > 0.0:
                    lep_rows[sign].append((birth, Gp, eps, w, gamma(u)))
    n_m_d, n_p_d = len(lep_rows[-1]), len(lep_rows[1])
    wE = fsum(x[3] * x[4] for s in (-1, 1) for x in lep_rows[s])
    wepsm = fsum(x[3] * x[2] for x in lep_rows[-1])
    wepsp = fsum(x[3] * x[2] for x in lep_rows[1])
    half = 0.5 * fsum([wepsm, wepsp])
    close = lambda x, y: abs(x - y) <= 1e-13 * max(abs(x), abs(y), 1.0)
    verdict(n_m_d == n_m and n_p_d == n_p, name, "sums: alive born leptons in the dumps {} / {} = the printed {} / {}".format(n_m_d, n_p_d, n_m, n_p))
    verdict(close(wE, sE), name, "sums: sum w gamma from the dumps {:.17g} = the printed {:.17g}".format(wE, sE))
    verdict(close(half, seps), name, "sums: (1/2) sum w eps over both species {:.17g} = the printed {:.17g}".format(half, seps))
    verdict(close(wE - half, net), name, "sums: the species-wise net {:.17g} = the printed sum_pairs {:.17g}".format(wE - half, net))

    # the old index-paired sum on the same slots (pgen.hpp at 1a3c3fd2)
    o_net = []
    o_n = 0
    for n, nm, npl, rows in dumps:
        for r in rows[:min(nm, npl)]:
            wm, um, tm, bm, _, em = side(r, -1)
            wp, up, tp, bp, _, _ = side(r, 1)
            if tm == 1 and tp == 1 and bm > 0.0 and bp > 0.0:
                o_net.append(wm * (gamma(um) + gamma(up) - em))
                o_n += 1
    o_net = fsum(o_net)
    o_corr = rhs - o_net - er

    if a.box == "absorbing":
        both = lone_m = lone_p = 0
        terms = []
        n, nm, npl, rows = dumps[0]
        for r in rows[:min(nm, npl)]:
            wm, _, tm, bm, _, em = side(r, -1)
            wp, _, tp, bp, _, ep = side(r, 1)
            am, ap_ = tm == 1 and bm > 0.0, tp == 1 and bp > 0.0
            if am and ap_:
                both += 1
                terms.append(wm * em)
            elif am:
                lone_m += 1
                terms.append(0.5 * wm * em)
            elif ap_:
                lone_p += 1
                terms.append(0.5 * wp * ep)
        verdict(lone_m > 0 and lone_p > 0, name, "lone survivors exist in both species: {} complete, {} lone e-, {} lone e+ (not vacuous)".format(both, lone_m, lone_p))
        verdict(close(fsum(terms), seps), name, "photon: complete pairs once, a lone survivor its own half: {:.17g} = the printed {:.17g}".format(fsum(terms), seps))
        verdict(lep.group(4).startswith("incomplete pairs"), name, "the array leptons line reports incomplete pairs ({} / {} of {} spawned)".format(n_m, n_p, n_sp))
        verdict(o_n == both, name, "(the old sum dropped every lone survivor: {} rows)".format(o_n))
    else:
        verdict(close(wepsm, wepsp) and close(wepsm, half) and close(half, e_abs), name,
                "photon: sum w eps over e- {:.17g}, over e+ {:.17g}, the half sums {:.17g}, the ledger's E_absorbed {:.17g}: counted once".format(wepsm, wepsp, half, e_abs))
        same = sorted(x[:4] for x in lep_rows[-1]) == sorted(x[:4] for x in lep_rows[1])
        verdict(same, name, "partners: the e- and e+ multisets of (birth, parent Gamma_p, eps, w) are equal ({} leptons each)".format(n_m_d))
        verdict(n_m == n_p == n_sp and lep.group(4).startswith("every pair complete"), name,
                "complete: e- {} = e+ {} = n_spawned {}".format(n_m, n_p, n_sp))
        verdict(n_sp > 100, name, "not vacuous: {} pairs".format(n_sp))
        verdict(abs(corr) <= tol * scale, name, "closure: |residual - E_round| / max(|E_drag|, 1) = {:.3e} <= {:.0e}".format(abs(corr) / scale, tol))
        if a.precision == "single" and a.spectrum == "delta":
            print("HISTORY: {}: raw |residual| / max(|E_drag|, 1) = {:.3e} (the revision 1.1 single-delta exception)".format(name, abs(res) / scale))
        else:
            verdict(abs(res) <= tol * scale, name, "closure: raw |residual| / max(|E_drag|, 1) = {:.3e} <= {:.0e}".format(abs(res) / scale, tol))
        rel = abs(o_corr) / scale
        print("(diag) {}: old index-paired sum: {} of {} pairs, net {:.17g}, |residual - E_round| / max(|E_drag|, 1) = {:.6e}".format(name, o_n, n_sp, o_net, rel))
        if a.old == "pass":
            verdict(rel <= tol and o_n == n_sp, name, "negative control: the old index-paired sum closes here ({} of {} pairs, {:.3e})".format(o_n, n_sp, rel))
        elif a.old == "fail":
            verdict(rel > tol and o_n < n_sp, name, "negative control: the old index-paired sum FAILS here as it must ({} of {} pairs, {:.3e} > {:.0e})".format(o_n, n_sp, rel, tol))
    print("arrays_species_check {}: {} passed, {} failed".format(name, n_pass, n_fail))
    return 1 if n_fail else 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:  # unreadable input is infrastructure, never Python's exit 1
        print("INFRA: arrays_species_check.py: {}: {}".format(type(e).__name__, e))
        rc = 2
    sys.exit(rc)
