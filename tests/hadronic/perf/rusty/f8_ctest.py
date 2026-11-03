#!/usr/bin/env python3
"""F8 step of p2-perf-a100.sbatch (Codex 1's closure C1, 2026-10-31): run the named CTest entries of one build, each on its
own, through Rusty's validated launcher plan, and classify every entry from CTest's STRUCTURED result (JUnit XML), so that
a scientific failure means an executed check failed and never that its executable did not run.

Launcher plan (Codex 1's G3 `prepare-ctest-launchers.py`, 2026-10-26, reused): the build's own CTest plan
(`ctest --show-only=json-v1`) is copied into a separate directory with every DIRECT command prefixed by the launcher
(`mpirun -n 1`: a direct test that calls MPI_Init, e.g. FRAMEWORK::f8_tracers through GlobalInitialize, runs as one
MPI rank) and every command that already starts with mpirun / mpiexec / orterun / srun (the stock two-rank entries)
unchanged; the test properties (WORKING_DIRECTORY, ...) are kept. Only the named entries are written. An entry whose
executable does not exist is written with a non-existent command so that CTest itself reports it Not Run.

Classification of an entry (CTest JUnit status and, for a failed entry, the NATIVE exit value from CTest's Test.xml
`Exit Value` measurement, recorded by running each entry with `-T Test`: JUnit's "Failed" does not distinguish exit 1
from exit 2; Codex 1's reclosure R1, 2026-10-31). Scientific needs POSITIVE evidence of an executed check that failed;
everything else is infrastructure (a stock test's non-zero exit is NOT presumed to be a check failure):
  status run                                     -> passed
  status notrun / disabled ("Unable to find executable", ...)                -> INFRASTRUCTURE (not run)
  status fail, message Timeout / SEGFAULT / Subprocess aborted / ILLEGAL / ... -> INFRASTRUCTURE (timeout, exception)
  status fail, message Failed (the process ran to completion and returned non-zero):
      the output carries an MPI launcher / initialisation failure marker      -> INFRASTRUCTURE
      native exit value not recorded                                          -> INFRASTRUCTURE
      native exit value != 1 (e.g. 2: the caught-exception exit of f8_tracers, injectors_global_weights and
        reduced_stats_weights; MPI_Abort's error code)                        -> INFRASTRUCTURE
      native exit value 1 AND the entry's anchored check-failure evidence (CHECK_EVIDENCE below):
        f8_tracers, injectors_global_weights, reduced_stats_weights: >= 1 verdict line starting "FAIL: ", the
          program's summary line "<prog> summary: P passed, F failed" with F >= 1, no "<prog>: exception:" line
        deposit: one of its four check messages "DepositCurrents_kernel::SumDivJ != 0" / "SumJ{x,y,z} mismatch"
        particles_sort, checkpoint-data, checkpoint-extents, spectra: a raise::Error message raised at a line of the
          test's OWN source file ("<message> .../tests/framework/<test>.cpp:<line> @ <func>"), other than the
          two-rank launch precondition                                       -> SCIENTIFIC
      native exit value 1 without that evidence (no parsed check result)      -> INFRASTRUCTURE
  no JUnit record / unknown status                                            -> INFRASTRUCTURE
  An infrastructure entry whose output nevertheless carries the anchored verdict lines of executed failed checks (an
  exception after a failed check) also prints a "FAIL: " line, so that the job records <step>-scientific beside the
  infrastructure class, as its run() does for any step. In an MPI build raise::Error calls MPI_Abort before the test's
  catch, so a failed raise check of the four framework tests exits with MPI_Abort's code and is reported as
  infrastructure with its log (never presumed scientific); its message is then read from the log by hand.
An entry not registered in the build is the selection gate (exit 3).

Output: one line per entry on stdout ("F8 entry <name>: passed", "FAIL: F8 entry <name>: ..." for a scientific failure,
"INFRA: F8 entry <name>: ..." for an infrastructure failure), the CTest console log and JUnit file of every entry in the
destination directory, and a summary line.
Evidence retention (perf round 4, Codex 1's readback of B7398318: the sequential `-T Test` invocations shared one Testing/
directory, so each overwrote the previous entry's Test.xml and LastTest log, and CTest kept only 1024 bytes of a passed
entry's output): every entry runs in its OWN CTest directory `entry-NN/` (a one-entry copy of the launcher plan), so its
Testing/<tag>/Test.xml and Testing/Temporary/LastTest_*.log are never overwritten; CTest's stored-output limits are raised
(`--test-output-size-passed/failed`, default 16 MiB, options below) and the entry's output as CTest stored it is written to
`output-NN.txt`. A stored output that still carries CTest's truncation marker is reported on a "NOTE: " line and counted
in the summary; the classification above is unchanged. Exit: 0 all passed; 1 scientific failures only; 3 selection gate; 4 any
infrastructure failure, including an exception raised in this tool itself (never exit 1 through a traceback). Python 3.6
compatible (Rusty's system python3).
"""
import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

LAUNCHERS = {"mpirun", "mpiexec", "orterun", "srun", "prterun"}
MPI_INFRA_MARKERS = (
    "not enough slots available",
    "An error occurred in MPI_Init",
    "orte_ess_init failed",
    "ORTE_ERROR_LOG",
    "PMIX ERROR",
    "mpirun was unable to",
    "mpirun noticed that process rank",
    "MPI_ERRORS_ARE_FATAL",
)
# verdict programs: the prefix of their summary and exception lines
VERDICT_PROGRAMS = {
    "FRAMEWORK::f8_tracers": "f8_tracers",
    "KERNELS::injectors_global_weights": "injectors_global_weights",
    "KERNELS::reduced_stats_weights": "reduced_stats_weights",
}
# raise-style stock tests: the check-failure line printed by the test's own catch (anchored, multiline)
CHECK_EVIDENCE = {
    "KERNELS::deposit": re.compile(r"^DepositCurrents_kernel::(SumDivJ != 0|SumJ[xyz] mismatch)\s*$", re.M),
}
for _n, _f in (("FRAMEWORK::particles_sort", "particles_sort"), ("FRAMEWORK::checkpoint-data", "checkpoint-data"),
               ("FRAMEWORK::checkpoint-extents", "checkpoint-extents"), ("FRAMEWORK::spectra", "spectra")):
    CHECK_EVIDENCE[_n] = re.compile(r"^(?!(Error: )?this test requires exactly 2 MPI ranks )(Error: )?\S.* \S*tests/framework/"
                                    + re.escape(_f) + r"\.cpp:\d+ @ \S.*$", re.M)


def quote(value):
    # CMake bracket arguments suppress variable expansion and preserve semicolons (Codex 1's quoting)
    text = str(value)
    equals = "="
    while "]" + equals + "]" in text:
        equals += "="
    return "[" + equals + "[" + text + "]" + equals + "]"


def executable_of(command, workdir):
    """The executable an entry needs: command[0] for a direct command; for a launcher command (the gen_test pattern
    `mpiexec -n N <exe>`) its last element, resolved against the working directory."""
    exe = command[0] if os.path.basename(command[0]) not in LAUNCHERS else command[-1]
    if not os.path.isabs(exe) and workdir:
        exe = os.path.join(workdir, exe)
    return exe


TRUNCATION_MARKER = "[This part of the test output was removed since it exceeds the threshold of"


def entry_dir(dest, i):
    return os.path.join(dest, "entry-{:02d}".format(i))


def stored_output(dest, name):
    """The entry's output as CTest stored it in its Test.xml (Measurement Value), or None."""
    try:
        with open(os.path.join(dest, "Testing", "TAG")) as f:
            tag = f.readline().strip()
        root = ET.parse(os.path.join(dest, "Testing", tag, "Test.xml")).getroot()
    except (OSError, ET.ParseError):
        return None
    for test in root.iter("Test"):
        if test.findtext("Name") == name:
            for m in test.iter("Measurement"):
                return m.findtext("Value") or ""
    return None


def write_plan(build, dest, launcher, names):
    plan = json.loads(subprocess.check_output(["ctest", "--test-dir", build, "--show-only=json-v1"], universal_newlines=True))
    by_name = {t["name"]: t for t in plan.get("tests", [])}
    missing = [n for n in names if n not in by_name]
    if missing:
        for n in missing:
            print("F8 selection gate: entry {} is not registered in {}".format(n, build))
        return None
    os.makedirs(dest)
    header = "# Generated by f8_ctest.py from the unchanged CTest plan of {} (launcher: {!r}).".format(build, launcher)
    lines = [header]
    notes = {}
    for i, n in enumerate(names):
        first = len(lines)
        t = by_name[n]
        props = t.get("properties", [])
        workdir = next((p["value"] for p in props if p["name"] == "WORKING_DIRECTORY"), "")
        command = t.get("command")
        if not command:
            command = [os.path.join(build, "F8-NOT-BUILT", n.replace(":", "_"))]
            notes[n] = "no command in the CTest plan (target not built)"
        elif not (os.path.isfile(executable_of(command, workdir)) and os.access(executable_of(command, workdir), os.X_OK)):
            notes[n] = "executable {} missing".format(executable_of(command, workdir))
            command = [os.path.join(build, "F8-NOT-BUILT", n.replace(":", "_"))]
        elif os.path.basename(command[0]) not in LAUNCHERS and launcher:
            command = shlex.split(launcher) + command
        lines.append("add_test(" + " ".join(quote(v) for v in [n] + command) + ")")
        for p in props:
            value = p["value"]
            if isinstance(value, bool):
                value = "TRUE" if value else "FALSE"
            elif isinstance(value, list):
                value = ";".join(str(v) for v in value)
            lines.append("set_tests_properties(" + quote(n) + " PROPERTIES " + quote(p["name"]) + " " + quote(value) + ")")
        # the entry's own CTest directory (round 4): its Testing/ is never shared with another entry
        os.makedirs(entry_dir(dest, i))
        with open(os.path.join(entry_dir(dest, i), "CTestTestfile.cmake"), "w") as f:
            f.write("\n".join([header + " Entry {} only.".format(n)] + lines[first:]) + "\n")
    with open(os.path.join(dest, "CTestTestfile.cmake"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(dest, "original-test-plan.json"), "w") as f:
        json.dump([by_name[n] for n in names], f, indent=2)
    return notes


def junit_result(path):
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None, None, ""
    for case in root.iter("testcase"):
        status = case.get("status")
        message = ""
        for child in case:
            if child.tag in ("failure", "skipped", "error"):
                message = child.get("message", "")
        out = "".join(e.text or "" for e in case.iter("system-out"))
        return status, message, out
    return None, None, ""


def native_exit_value(dest, name):
    """The entry's native exit value from the Test.xml that `ctest -T Test` wrote for this invocation (None if absent)."""
    try:
        with open(os.path.join(dest, "Testing", "TAG")) as f:
            tag = f.readline().strip()
        paths = [os.path.join(dest, "Testing", tag, "Test.xml")]
    except OSError:
        paths = sorted(glob.glob(os.path.join(dest, "Testing", "*", "Test.xml")))
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        for test in root.iter("Test"):
            if test.findtext("Name") != name:
                continue
            for m in test.iter("NamedMeasurement"):
                if m.get("name") == "Exit Value":
                    try:
                        return int((m.findtext("Value") or "").strip())
                    except ValueError:
                        return None
    return None


def executed_check_failure(name, text):
    """The anchored evidence that an executed check of this entry failed (a description), or None."""
    prog = VERDICT_PROGRAMS.get(name)
    if prog:
        lines = text.splitlines()
        fails = [l for l in lines if l.startswith("FAIL: ") and "exception" not in l]
        n_failed = None
        for l in lines:
            m = re.match(r"^" + re.escape(prog) + r" summary: (\d+) passed, (\d+) failed\s*$", l)
            if m:
                n_failed = int(m.group(2))
        if any(l.startswith(prog + ": exception:") or l.startswith("EXCEPTION:") for l in lines):
            return None
        if fails and n_failed is not None and n_failed >= 1:
            return "executed check failure ({} FAIL line(s), summary failed count {})".format(len(fails), n_failed)
        return None
    pattern = CHECK_EVIDENCE.get(name)
    if pattern:
        m = pattern.search(text)
        if m:
            return "executed check failure ({!r})".format(m.group(0).strip())
    return None


def failed_verdicts(name, text):
    """FAIL verdict lines of a verdict program in an output that is infrastructure for another reason."""
    if name not in VERDICT_PROGRAMS:
        return 0
    return len([l for l in text.splitlines() if l.startswith("FAIL: ") and "exception" not in l])


def classify(name, status, message, exit_value, text):
    if status == "run":
        return "passed", "passed"
    if status is None:
        return "infra", "no CTest JUnit record"
    if status in ("notrun", "disabled"):
        return "infra", "not run ({})".format(message or status)
    if status == "fail" and message != "Failed":
        return "infra", "{} (timeout or exception, CTest status)".format(message or "unknown failure")
    if status == "fail":
        marker = next((m for m in MPI_INFRA_MARKERS if m in text), None)
        if marker:
            return "infra", "returned non-zero with an MPI launcher/initialisation marker ({!r})".format(marker)
        if exit_value is None:
            return "infra", "returned non-zero; native exit value not recorded (no Test.xml Exit Value)"
        if exit_value != 1:
            return "infra", "exited {} (a caught exception or another non-check exit), not an executed check failure".format(exit_value)
        evidence = executed_check_failure(name, text)
        if evidence:
            return "sci", evidence
        return "infra", "exited 1 without a parsed check result (no anchored check-failure evidence for this entry)"
    return "infra", "unknown CTest status {!r}".format(status)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--dest", required=True, help="new directory for the launcher plan and the per-entry logs")
    ap.add_argument("--launcher", default="mpirun -n 1", help="prefix of direct commands ('' for none)")
    ap.add_argument("--timeout", type=int, default=80, help="per-entry CTest timeout (s)")
    ap.add_argument("--passed-output-size", type=int, default=16777216,
                    help="CTest's stored-output limit of a passed entry, bytes (CTest's own default is 1024)")
    ap.add_argument("--failed-output-size", type=int, default=16777216, help="the same for a failed entry, bytes")
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    notes = write_plan(a.build, a.dest, a.launcher, a.names)
    if notes is None:
        print("F8: selection gate failed")
        return 3
    n_pass = n_sci = n_infra = n_trunc = 0
    for i, name in enumerate(a.names):
        junit = os.path.join(a.dest, "junit-{:02d}.xml".format(i))
        log = os.path.join(a.dest, "ctest-{:02d}.log".format(i))
        edir = entry_dir(a.dest, i)
        with open(log, "w") as f:
            # -T Test: CTest's Test.xml carries the native exit value ("Exit Value"), which JUnit's "Failed" does not; the
            # entry's own directory keeps its Test.xml; the raised limits keep its whole output
            subprocess.call(["ctest", "--test-dir", edir, "-T", "Test", "-R", "^" + name + "$", "--output-on-failure",
                             "--timeout", str(a.timeout), "--output-junit", junit,
                             "--test-output-size-passed", str(a.passed_output_size),
                             "--test-output-size-failed", str(a.failed_output_size)], stdout=f, stderr=subprocess.STDOUT)
        status, message, out = junit_result(junit)
        exit_value = native_exit_value(edir, name) if status == "fail" else None
        stored = stored_output(edir, name)
        with open(os.path.join(a.dest, "output-{:02d}.txt".format(i)), "w") as f:
            f.write(stored if stored is not None else out)
        if TRUNCATION_MARKER in (stored if stored is not None else out):
            n_trunc += 1
            print("NOTE: F8 entry {}: CTest truncated the stored output (limit {} bytes passed, {} failed) [output-{:02d}.txt]".format(
                name, a.passed_output_size, a.failed_output_size, i))
        with open(log) as f:
            # the entry's output: JUnit's system-out, or the console log (which repeats it) when JUnit has none
            text = out if out.strip() else f.read()
        cls, why = classify(name, status, message, exit_value, text)
        if name in notes:
            why += "; " + notes[name]
        if cls == "passed":
            n_pass += 1
            print("F8 entry {}: passed".format(name))
        elif cls == "sci":
            n_sci += 1
            print("FAIL: F8 entry {}: {} [log {}]".format(name, why, log))
        else:
            n_infra += 1
            print("INFRA: F8 entry {}: {} [log {}]".format(name, why, log))
            n_verdicts = failed_verdicts(name, text)
            if n_verdicts:
                n_sci += 1
                print("FAIL: F8 entry {}: its output also carries {} FAIL verdict line(s) of executed checks before the "
                      "infrastructure fault [log {}]".format(name, n_verdicts, log))
    print("F8: {} entries, {} passed, {} scientific failures, {} infrastructure failures, {} truncated stored outputs".format(
        len(a.names), n_pass, n_sci, n_infra, n_trunc))
    if n_infra:
        return 4
    return 1 if n_sci else 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:  # an exception in this tool is infrastructure, never Python's exit 1
        print("INFRA: f8_ctest.py exception: {}: {}".format(type(e).__name__, e))
        rc = 4
    sys.exit(rc)
