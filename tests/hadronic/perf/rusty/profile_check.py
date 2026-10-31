#!/usr/bin/env python3
"""Content assertions of the M3 profiles of p2-perf-a100.sbatch (Codex 1's closure C4, 2026-10-31). Exit 0 when the
profile holds what the configuration requires, 4 otherwise (an infrastructure failure of the profile step).

  log  <program log> --n N --mode M --kernel K
       the profiled program's own lines: "bh_perf: BH_PERF_N = N " with N the authorisation's fixed value (never an
       overridable variable), the configuration filter "mode 'M', kernel 'K'", and exactly one configuration of
       20 warm-up + 100 timed launches selected
  nsys <nvtx_sum csv> --mode M --kernel K
       the NVTX ranges bh_perf_restore_and_reseed and bh_perf_finish_readback and the launch range of (M, K), and no
       launch range of another configuration; when the report has an Instances column, the launch range has 120
  ncu  <raw page csv> --kernel K --min 12
       counts UNIQUE profiled launches by the profiler's result ID (a raw or details page may hold several rows per
       launch: one launch with 24 metric rows is one launch), every row's kernel must be the configuration's functor type
       (production: BetheHeitler_kernel<, not the variant; const: PerfTablesConst; float: PerfTablesFloat), and at least
       --min unique launches
Python 3.6 compatible (Rusty's system python3).
"""
import argparse
import csv
import re
import sys

LAUNCH_LABEL = {"production": "bh_perf_production_{m}", "const": "bh_perf_variant_const_{m}", "float": "bh_perf_variant_float_{m}"}
KERNELS = ("production", "const", "float")
MODES = ("event", "continuous")


def fail(msg):
    print("PROFILE CHECK FAILED: " + msg)
    return 4


def check_log(a):
    try:
        text = open(a.file, errors="replace").read()
    except OSError as e:
        return fail("cannot read {}: {}".format(a.file, e))
    n = re.findall(r"^bh_perf: BH_PERF_N = (\d+) ", text, re.M)
    if n != [str(a.n)]:
        return fail("{}: BH_PERF_N lines {} (expected exactly one, = {}: the authorised fixed N)".format(a.file, n, a.n))
    flt = "bh_perf: configuration filter: mode '{}', kernel '{}'".format(a.mode, a.kernel)
    if flt not in text:
        return fail("{}: the configuration filter line {!r} is missing".format(a.file, flt))
    m = re.findall(r"^\(perf\) M3 launches per configuration: (\d+) warm-up \+ (\d+) timed = (\d+) parallel_reduce launches of the kernel, "
                   r"(\d+) configurations selected", text, re.M)
    if m != [("20", "100", "120", "1")]:
        return fail("{}: launches/configurations line {} (expected 20 + 100 = 120 launches, 1 configuration)".format(a.file, m))
    print("profile log {}: BH_PERF_N = {}, configuration ({}, {}), 120 launches of 1 configuration".format(a.file, a.n, a.mode, a.kernel))
    return 0


def read_rows(path):
    with open(path, newline="", errors="replace") as f:
        return [r for r in csv.reader(f)]


def range_name(cell):
    # nsys may prefix a push/pop range with its domain (":name" or "domain:name")
    return cell.strip().split(":")[-1].strip()


def check_nsys(a):
    try:
        rows = read_rows(a.file)
    except OSError as e:
        return fail("cannot read {}: {}".format(a.file, e))
    names = {}
    header = rows[0] if rows else []
    inst_col = next((i for i, h in enumerate(header) if h.strip() == "Instances"), None)
    for r in rows[1:]:
        for c in r:
            names.setdefault(range_name(c), r)
    want = LAUNCH_LABEL[a.kernel].format(m=a.mode)
    for need in ("bh_perf_restore_and_reseed", "bh_perf_finish_readback", want):
        if need not in names:
            return fail("{}: NVTX range {} missing".format(a.file, need))
    others = [LAUNCH_LABEL[k].format(m=m) for k in KERNELS for m in MODES if (k, m) != (a.kernel, a.mode)]
    present = [o for o in others if o in names]
    if present:
        return fail("{}: launch ranges of other configurations present: {}".format(a.file, present))
    note = "Instances column absent: instance count not checked"
    if inst_col is not None:
        row = names[want]
        try:
            inst = int(float(row[inst_col]))
        except (IndexError, ValueError):
            return fail("{}: unreadable Instances of {}".format(a.file, want))
        if inst != 120:
            return fail("{}: {} has {} instances, expected 120".format(a.file, want, inst))
        note = "{} instances of {}".format(inst, want)
    print("nsys {}: ranges restore/reseed, finish readback and {} present, no other configuration; {}".format(a.file, want, note))
    return 0


def kernel_matches(kernel, text):
    if kernel == "production":
        return "BetheHeitler_kernel<" in text and "BetheHeitler_kernel_variant" not in text
    return ("PerfTablesConst" if kernel == "const" else "PerfTablesFloat") in text


def check_ncu(a):
    try:
        rows = read_rows(a.file)
    except OSError as e:
        return fail("cannot read {}: {}".format(a.file, e))
    hdr_i = next((i for i, r in enumerate(rows) if r and r[0].strip() == "ID"), None)
    if hdr_i is None:
        hdr_i = next((i for i, r in enumerate(rows) if "ID" in [c.strip() for c in r]), None)
    if hdr_i is None:
        return fail("{}: no header with an ID column (unknown raw-page format)".format(a.file))
    id_col = [c.strip() for c in rows[hdr_i]].index("ID")
    ids, n_rows, wrong = set(), 0, 0
    for r in rows[hdr_i + 1:]:
        if len(r) <= id_col or not r[id_col].strip().isdigit():
            continue  # the units row and blank lines carry no launch ID
        text = ",".join(r)
        if "BetheHeitler" not in text:
            continue  # another kernel (none expected under the name filter)
        n_rows += 1
        if not kernel_matches(a.kernel, text):
            wrong += 1
            continue
        ids.add(r[id_col].strip())
    print("ncu {}: {} unique launch IDs of the {} kernel over {} BetheHeitler rows; {} rows of another configuration".format(
        a.file, len(ids), a.kernel, n_rows, wrong))
    if wrong:
        return fail("{}: {} rows belong to another kernel configuration".format(a.file, wrong))
    if len(ids) < a.min:
        return fail("{}: {} unique launches captured, expected at least {}".format(a.file, len(ids), a.min))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("log"); p.add_argument("file"); p.add_argument("--n", type=int, required=True)
    p.add_argument("--mode", choices=MODES, required=True); p.add_argument("--kernel", choices=KERNELS, required=True)
    p = sub.add_parser("nsys"); p.add_argument("file")
    p.add_argument("--mode", choices=MODES, required=True); p.add_argument("--kernel", choices=KERNELS, required=True)
    p = sub.add_parser("ncu"); p.add_argument("file"); p.add_argument("--kernel", choices=KERNELS, required=True)
    p.add_argument("--min", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "log":
        return check_log(a)
    if a.cmd == "nsys":
        return check_nsys(a)
    if a.cmd == "ncu":
        return check_ncu(a)
    ap.print_help()
    return 4


if __name__ == "__main__":
    sys.exit(main())
