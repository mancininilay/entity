#!/usr/bin/env python3
"""Step budgets of the A100 perf campaign from measured costs (perf round 4, Claude 1, 2026-11-03).

Every input below is a MEASUREMENT of job B7398318 (one A100-SXM4-80GB, 8 CPUs, perf head 6e0c2cf7; Codex 1's readback
and its evidence `Entity Local Experiments/p2-perf-readback-codex1-20261103-7398318/`) or of job A7397866 (CUDA builds),
except the two PLANNING assumptions marked as such (the engine step slope above 2^18 and the O(N) part of the process
startup), which nothing measured so far bounds. The script prints the per-process cost model, the engine campaign per N,
the kernel-level and F8 steps, and the partition of the campaign into one-hour jobs (`--parts`), and asserts that every
part's ceilings fit its hour with the registered margin. Python 3.6 compatible; no input files.

Usage: perf_budget.py [--beta-ns 2.0] [--qk fit|born] [--table]   the model and the campaign per N
       perf_budget.py --emit-table      the per-process ceilings embedded in p2-perf-a100.sbatch (its self-test compares them)
       perf_budget.py --plan            the one-hour parts covering the registered campaign; exit 1 unless every part fits
                                        and the parts cover the 143 registered processes exactly once
       perf_budget.py --plan-env        the same parts as name|part|N|first|last|m1|modes|variants|m5 (the job's self-test
                                        dry-runs every one of them through the job's own guard)
"""
import argparse
import sys

# ---- measured, B7398318 (engine at N = 2^18 = 262144, Born tables, 551 steps per process, OMP_NUM_THREADS=1) ----------
N0 = 262144
STEPS = 550           # 50 warm-up + 500 measured; the 551st step is finish() (ledger row, per-ion and pair text dumps)
# per configuration: (stepping s for 550 steps, finish s of the last step), the MAXIMUM over the three repeats of
# "loop elapsed - last-step Custom" and "last-step Custom" read from each run.log (Entity's Elapsed line and the final
# diagnostics block); m5poisson_pic from its partial run (323 steps in 7.12 s) with the source run's finish;
# m5budget_pic (not reached) as m5budget
MEASURED = {
    ("m1", "none", "cold"): (6.04, 3.88),
    ("m1", "none", "warm"): (6.13, 5.04),
    ("m1", "poisson", "cold"): (6.32, 4.12),
    ("m1", "poisson", "warm"): (7.51, 9.95),
    ("m1", "budget", "cold"): (5.90, 3.63),
    ("m1", "budget", "warm"): (6.41, 5.69),
    ("m1", "event", "cold"): (5.77, 3.68),
    ("m1", "event", "warm"): (6.86, 8.05),
    ("m5pic", "none", "warm"): (8.76, 3.75),
    ("m5poisson", "poisson", "warm"): (12.39, 10.07),
    ("m5budget", "budget", "warm"): (8.95, 5.43),
    ("m5poisson_pic", "poisson", "warm"): (12.13, 10.07),
    ("m5budget_pic", "budget", "warm"): (8.95, 5.43),
}
# leptons (e- + e+) at the end of the run per ion: poisson and event warm 365750 / 365608 per 262144 (B7398318 ledger rows);
# the restart of the poisson end population carries them from its first step
LEPTONS_END = {("m1", "poisson", "warm"): 1.395, ("m1", "event", "warm"): 1.395, ("m5poisson", "poisson", "warm"): 1.395,
               ("m5poisson_pic", "poisson", "warm"): 1.395}
BORN_S = 7.20         # Born node table per hadronic-on process, host, one thread: 7.158-7.196 s (26 processes)
INIT_S = 3.0          # per-process residual at 2^18: (600 - 186.5 Born - 322.4 loop - 7.1 partial) / 28 processes
                      # (mpirun, CUDA/Kokkos/MPI init, injection, driver awk, bh_check, checkpoint copy)
# ---- planning assumptions (NOT measured: no engine step above 2^18 exists on the A100) --------------------------------
BETA_NS = 2.0         # engine step cost per particle-step above 2^18 [ns]; basis: the stage itself costs 0.30 (continuous)
                      # and 0.80 (event) ns per ion-step at 2^22 (M3, B7398318), the reducer 0.11 ns (M4); the pusher,
                      # boundaries and the one-rank communication are a few bandwidth-bound passes over ~100 B per particle.
                      # At 2^18 the step (9-12 ms) is dominated by per-step fixed costs (Communications 6-8 ms, one rank),
                      # so a linear extrapolation of the 2^18 step would overstate large N by an order of magnitude; the
                      # sensitivity to BETA_NS is printed (--beta-ns)
INIT_NS_PER_ION = 1000.0  # O(N) part of the startup [ns per ion]: host injection and the stored-array sum (binary passes),
                          # an upper estimate well above a binary host pass and well below the text dump's 12-17 us/particle
FINISH_LINEAR = True  # FACT (pgens/bh_proto/pgen.hpp write_ions, write_pairs): the finish dump is a serial host loop over the
                      # ions and the pairs with text formatting: linear in the particle count, measured 12-17 us/particle

# ---- measured, steps other than the engine (B7398318 steps.tsv; A7397866 builds) ---------------------------------------
# name: (measured s, source, ceiling s)
FIXED_STEPS = [
    ("step0-checkout", 25, "B7398318", 120),
    ("step1-build (single: 12 targets, 79 Ninja steps, -j 8)", 654, "B7398318", 780),
    ("step3-stride (M6, 4M ions)", 96, "B7398318", 150),
    ("step3-variants (M3, 4M ions)", 62, "B7398318", 100),
    ("step4-reduction (M4)", 3, "B7398318", 30),
    ("step5-nsys x4 (each <= 32 s incl. the failed ncu attempt)", 4 * 32, "B7398318", 4 * 50),
    ("step5-rngpool", 2, "B7398318", 30),
    ("step6-f8-single (8 entries)", 27, "B7398318 (injector aborted early)", 120),
    ("step7-f8-double-build (8 targets, 69 Ninja steps, -j 8)", 1000,
     "A7397866: 27 targets, 109 steps in 1003 s at -j 12; B7398318: 49/69 steps at 480 s", 1000),
    ("step7-f8-double (8 entries)", 27, "as single", 120),
]
NCU_STEP_S = 50       # per configuration, OPTIONAL and off by default (Rusty's driver refused the counters, B3); unmeasured
MARGIN = 0.80         # a part's ceilings may use at most 80 % of the hour (kill grace, environment, write_status)
HOUR = 3600
PREAMBLE_S = 120 + 780   # every part checks out and builds the single precision (the gate)


def processes(n, reps):
    """The driver's process list at one N: (tag, mode, variant) per process, in run order."""
    out = []
    for mode in ("none", "poisson", "budget", "event"):
        for var in ("warm", "cold"):
            out += [("m1", mode, var)] * reps
    for _ in range(reps):
        out += [("m5pic", "none", "warm"), ("m5poisson", "poisson", "warm"), ("m5budget", "budget", "warm"),
                ("m5poisson_pic", "poisson", "warm"), ("m5budget_pic", "budget", "warm")]
    return out


def hadronic_on(key):
    return key[0] in ("m1", "m5poisson", "m5budget")


def process_cost(key, n, qk, beta_ns):
    """(init, born, stepping, finish) in seconds for one process at N ions."""
    step0, fin0 = MEASURED[key]
    s = n / float(N0)
    lep = LEPTONS_END.get(key, 0.0)
    # particles per step above the 2^18 run: ions plus, for the pair-loaded runs, the mean lepton count (half the end
    # count for a growing population, the full count for the restarted end population)
    mean_lep = lep if key[0].endswith("_pic") else 0.5 * lep
    stepping = step0 + STEPS * beta_ns * 1e-9 * (n - N0) * (1.0 + mean_lep)
    finish = fin0 * s if FINISH_LINEAR else fin0
    init = INIT_S + INIT_NS_PER_ION * 1e-9 * (n - N0)
    born = BORN_S if (qk == "born" and hadronic_on(key)) else 0.0
    return init, born, stepping, finish


def engine_per_n(n, reps, qk, beta_ns):
    tot = [0.0, 0.0, 0.0, 0.0]
    procs = processes(n, reps)
    for key in procs:
        for i, v in enumerate(process_cost(key, n, qk, beta_ns)):
            tot[i] += v
    return len(procs), tot


REG_REPS = {262144: 3, 1048576: 3, 4194304: 3, 16777216: 2}   # registered repeats (the F3 cut: 2 at 2^24)
CEIL_FACTOR = 1.25
M5_GROUPS = {"pic": ["m5pic"], "poisson": ["m5poisson", "m5poisson_pic"], "budget": ["m5budget", "m5budget_pic"]}
MODES = ("none", "poisson", "budget", "event")
VARIANTS = ("warm", "cold")


def process_ceiling(key, n, qk="fit", beta_ns=BETA_NS):
    """The ceiling of one process [s]: CEIL_FACTOR x the model cost, rounded up."""
    return int(-(-CEIL_FACTOR * sum(process_cost(key, n, qk, beta_ns)) // 1))


def slice_processes(n, first, last, m1, modes, variants, m5_groups):
    """The (tag, mode, variant, N, repeat) processes of one driver invocation at one N (perf_driver.sh's loops)."""
    out = []
    if m1:
        for mode in modes:
            for var in variants:
                out += [("m1", mode, var, n, r) for r in range(first, last + 1)]
    for r in range(first, last + 1):
        for g in ("pic", "poisson", "budget"):
            if g in m5_groups:
                out += [(t, "none" if t == "m5pic" else g, "warm", n, r) for t in M5_GROUPS[g]]
    return out


# the parts: (name, PERF_N, first, last, PERF_M1, PERF_MODES, PERF_VARIANTS, PERF_M5_SETS); "kernel" has no engine slice
PLAN = [("K", None, 0, 0, 0, (), (), ())]
PLAN += [("E01", 262144, 1, 3, 1, MODES, VARIANTS, ("pic", "poisson", "budget")),
         ("E02", 1048576, 1, 3, 1, MODES, VARIANTS, ("pic", "poisson", "budget"))]
PLAN += [("E%02d" % (3 + r - 1), 4194304, r, r, 1, MODES, VARIANTS, ("pic", "poisson", "budget")) for r in (1, 2, 3)]
for r in (1, 2):
    PLAN += [("E%02d" % (4 + 2 * r), 16777216, r, r, 1, ("poisson", "event"), ("warm",), ()),
             ("E%02d" % (5 + 2 * r), 16777216, r, r, 1, MODES, ("cold",), ())]
PLAN += [("E10", 16777216, 1, 2, 1, ("none", "budget"), ("warm",), ())]
PLAN += [("E%02d" % (10 + r), 16777216, r, r, 0, (), (), ("poisson",)) for r in (1, 2)]
PLAN += [("E%02d" % (12 + r), 16777216, r, r, 0, (), (), ("pic", "budget")) for r in (1, 2)]


def kernel_part_ceilings(ncu):
    fixed = {name.split(" ")[0]: c for name, m, src, c in FIXED_STEPS}
    return sum(fixed.values()) + (4 * NCU_STEP_S if ncu else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-ns", type=float, default=BETA_NS)
    ap.add_argument("--qk", choices=("fit", "born"), default="fit")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--emit-table", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--plan-env", action="store_true", help="one line per part: name|part|N|first|last|m1|modes|variants|m5")
    a = ap.parse_args()
    if a.plan_env:
        for name, n, first, last, m1, modes, variants, m5 in PLAN:
            if n is None:
                print("{}|kernel||||||".format(name))
            else:
                print("{}|engine|{}|{}|{}|{}|{}|{}|{}".format(name, n, first, last, m1, " ".join(modes), " ".join(variants), " ".join(m5)))
        return 0
    if a.emit_table:
        # N tag mode variant ceiling_s, for the decided fit engine runs and the default planning slope (the job's table)
        for n in sorted(REG_REPS):
            for key in sorted(MEASURED):
                print("{} {} {} {} {}".format(n, key[0], key[1], key[2], process_ceiling(key, n)))
        return 0
    grid = sorted(REG_REPS.items())
    print("engine campaign, qk_source = {}, planning slope {:.2f} ns per particle-step above 2^18".format(a.qk, a.beta_ns))
    print("| N | processes | init s | Born s | stepping s | finish (dumps) s | total s | ceiling s (x{}) |".format(CEIL_FACTOR))
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    eng = {}
    for n, reps in grid:
        np_, (i, b, st, f) = engine_per_n(n, reps, a.qk, a.beta_ns)
        tot = i + b + st + f
        eng[n] = (np_, tot, f)
        print("| {} | {} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} | {:.0f} |".format(n, np_, i, b, st, f, tot, CEIL_FACTOR * tot))
    T = sum(v[1] for v in eng.values())
    F = sum(v[2] for v in eng.values())
    print("all four N: {} processes, {:.0f} s ({:.2f} h), of which the end-of-run text dumps {:.0f} s ({:.2f} h); "
          "without the dumps {:.0f} s ({:.2f} h)".format(sum(v[0] for v in eng.values()), T, T / 3600.0, F, F / 3600.0,
                                                        T - F, (T - F) / 3600.0))
    if a.table:
        print("\nper-process at 2^24 (s): init, Born, stepping, finish")
        for key in sorted(MEASURED):
            print("  {:<36} {}".format("/".join(key), ", ".join("{:.0f}".format(v) for v in process_cost(key, 16777216, a.qk, a.beta_ns))))
    print("\nother steps (measured -> ceiling, s)")
    for name, m, src, c in FIXED_STEPS:
        print("  {:<58} {:>5} -> {:>5}   [{}]".format(name, m, c, src))
    print("  {:<58} {:>5} -> {:>5}   [optional, off by default]".format("step5-ncu x4", "-", 4 * NCU_STEP_S))
    if not a.plan:
        return 0
    fixed = {name.split(" ")[0]: c for name, m, src, c in FIXED_STEPS}
    gate = fixed["step0-checkout"] + fixed["step1-build"]
    cap = int(MARGIN * HOUR)
    print("\none-hour parts: every part has its own checkout and single-precision build gate ({} s); ceilings at most {} s "
          "({:.0f} % of the hour)".format(gate, cap, 100 * MARGIN))
    print("| part | PERF_N | repeats | PERF_M1 | PERF_MODES | PERF_VARIANTS | PERF_M5_SETS | processes | ceilings s |")
    print("| --- | ---: | --- | ---: | --- | --- | --- | ---: | ---: |")
    ok = True
    seen = {}
    for name, n, first, last, m1, modes, variants, m5 in PLAN:
        if n is None:
            total = kernel_part_ceilings(ncu=False)
            total_ncu = kernel_part_ceilings(ncu=True)
            fits = total <= cap and total_ncu <= cap
            print("| {} | (kernel: M3, M4, M6, nsys, RNG, F8 single and double) | | | | | | | {} ({} with ncu) |".format(
                name, total, total_ncu))
        else:
            procs = slice_processes(n, first, last, m1, modes, variants, m5)
            for p in procs:
                seen[p] = seen.get(p, 0) + 1
            eng_c = sum(process_ceiling(p[:3], n) for p in procs)
            total = gate + eng_c
            fits = total <= cap
            print("| {} | {} | {}-{} | {} | {} | {} | {} | {} | {} |".format(name, n, first, last, m1, " ".join(modes) or "-",
                  " ".join(variants) or "-", " ".join(m5) or "-", len(procs), total))
        ok = ok and fits
        if not fits:
            print("  PART {} DOES NOT FIT".format(name))
    registered = set()
    for n, reps in grid:
        registered.update(slice_processes(n, 1, reps, 1, MODES, VARIANTS, ("pic", "poisson", "budget")))
    dup = [p for p, c in seen.items() if c > 1]
    missing = sorted(registered - set(seen))
    extra = sorted(set(seen) - registered)
    print("coverage: {} registered processes, {} in the parts, {} duplicated, {} missing, {} extra".format(
        len(registered), len(seen), len(dup), len(missing), len(extra)))
    ok = ok and not dup and not missing and not extra and len(registered) == 143
    print("plan: {} parts ({} A100-hours reserved), {}".format(len(PLAN), len(PLAN), "every part fits and the coverage is exact" if ok
                                                                  else "INVALID"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
