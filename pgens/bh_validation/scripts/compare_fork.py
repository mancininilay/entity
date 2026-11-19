#!/usr/bin/env python3
"""V8: the emission policy against the event stage of the fork (dev/hadronic 72427ff6).

    compare_fork.py <policy run dir> <log of `test-kernels-bh_event.xc e10` with BH_QK_SOURCE=born>

The fork's test runs N = 1e5 emitters at Gamma_p0 = 6 with a Poisson number of
interactions per step of mean 0.05 at Gamma_p0 (restframe spectrum, Born tables)
and prints <K> at its three targets, Var K at its three points and the number
of pairs. The policy run has the same probability per step, so that both are
compared after the same number of steps:
  - <K> within 4 combined s.e.;
  - Var K within 4 combined s.e. after adding to the policy's variance the
    predicted effect of one draw per step instead of a Poisson count, the
    solution of dV/dt = -2 tau (dI/dK) V + a R p <D>^2 along the run's <K>(t);
  - pairs per emitter within 4 combined s.e. (s.e. = sqrt(pairs per emitter / N)).
A consistency check by the same author, not an independent reference.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_cooling import Tables
from bhv import Series


def main():
    run_dir, fork_log = sys.argv[1], sys.argv[2]
    S = Series(run_dir)
    T = Tables(os.path.join(run_dir, "bhv_tables.txt"))
    tau, a, mu_r = S.inp["tau"], S.inp["a"], S.inp["mu_r"]
    text = open(fork_log).read()
    head = re.search(r"N = (\d+), K0 = ([\d.eE+-]+), dt = ([\d.eE+-]+), coeff = ([\d.eE+-]+), a = ([\d.eE+-]+), (\d+) launches.*?n_spawned (\d+)", text)
    N_f, dt_f, launches, spawned = float(head.group(1)), float(head.group(3)), int(head.group(6)), float(head.group(7))
    e2 = re.findall(r"E2 at <K>/K0 = ([\d.]+) \(t = ([\d.eE+-]+), <K> = ([\d.eE+-]+)\).*?s\.e\. = ([\d.eE+-]+)\)", text)
    e3 = re.findall(r"E3 \[registered.*?\] at (.*?) \(t = ([\d.eE+-]+), .*?Var_meas = ([\d.eE+-]+),", text)

    I_of_K = lambda K: T.I(1.0 + K / mu_r)
    J_of_K = lambda K: T.J(1.0 + K / mu_r)

    def Ip(K):
        h = 0.02 * K
        return (I_of_K(K + h) - I_of_K(K - h)) / (2.0 * h)

    # predicted Var_Poisson - Var_one_draw along the policy's measured mean
    delta = [0.0]
    for i in range(1, len(S.t)):
        hh = S.t[i] - S.t[i - 1]
        Km = 0.5 * (S.mean[i - 1] + S.mean[i])
        J = J_of_K(Km)
        Dm = I_of_K(Km) / J
        p = tau * J * S.dt / a
        g = a * tau * J * p * Dm * Dm
        c = tau * Ip(Km)
        ex = math.exp(-2.0 * c * hh)
        delta.append(delta[-1] * ex + g * (1.0 - ex) / (2.0 * c))

    def record(n_steps):
        for i, n in enumerate(S.steps):
            if abs(n - n_steps) < 0.5:
                return i
        return None

    n_fail = 0
    print(f"fork: N = {N_f:.0f}, {launches} launches, {spawned:.0f} pairs; policy: N = {S.n_emitters:.0f}, "
          f"{S.steps[-1]:.0f} steps, records every {S.steps[1]:.0f} steps")
    for f, t, K, se in e2:
        n = float(t) / dt_f
        i = record(n)
        if i is None:
            print(f"FAIL V8 <K> at the fork's target {f}: no policy record after {n:.0f} steps")
            n_fail += 1
            continue
        s = math.sqrt(float(se) ** 2 + S.se(i) ** 2)
        d = (S.mean[i] - float(K)) / s
        ok = abs(d) <= 4.0
        n_fail += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} V8 <K> after {n:.0f} steps (fork target {f}): policy {S.mean[i]:.4f}, fork {float(K):.6g}, "
              f"difference {d:+.2f} combined s.e. (limit 4)")
    for name, t, V in e3:
        n = float(t) / dt_f
        i = record(n)
        if i is None:
            print(f"FAIL V8 Var K at the fork's point {name}: no policy record after {n:.0f} steps")
            n_fail += 1
            continue
        Vf, Vp = float(V), S.var[i] + delta[i]
        s = math.sqrt(2.0 / (N_f - 1.0) * Vf * Vf + 2.0 / (S.n_emitters - 1.0) * S.var[i] ** 2)
        d = (Vp - Vf) / s
        ok = abs(d) <= 4.0
        n_fail += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} V8 Var K after {n:.0f} steps ({name}): policy {S.var[i]:.6g} + one-draw term {delta[i]:.5g} = {Vp:.6g}, "
              f"fork {Vf:.6g}, difference {d:+.2f} combined s.e. (limit 4); without the term {(S.var[i] - Vf) / s:+.2f}")
    i = record(float(launches))
    if i is None:
        print(f"FAIL V8 pairs: no policy record after {launches} steps")
        n_fail += 1
    else:
        pf, pp = spawned / N_f, S.pairs[i]
        s = math.sqrt(pf / N_f + pp / S.n_emitters)
        d = (pp - pf) / s
        ok = abs(d) <= 4.0
        n_fail += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} V8 pairs per emitter after {launches} steps: policy {pp:.4f}, fork {pf:.4f}, "
              f"difference {d:+.2f} combined s.e. (limit 4)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
