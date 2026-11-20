#!/usr/bin/env python3
"""Compile one standalone source against the libraries of an Entity build tree,
with the compile flags of that tree (taken from its compile_commands.json, the
entry of tests/framework/particles_sort.cpp) and its libraries.

    build_repro.py <build tree> <source.cpp> <output>
"""
import json
import os
import shlex
import subprocess
import sys

tree, src, out = sys.argv[1:4]
cc = json.load(open(os.path.join(tree, "compile_commands.json")))
entry = next(e for e in cc if e["file"].endswith("tests/framework/particles_sort.cpp"))
args = shlex.split(entry["command"])
cmd = []
skip = False
for a in args:
    if skip:
        skip = False
        continue
    if a in ("-o", "-c"):
        skip = a == "-o"
        continue
    if a.endswith("particles_sort.cpp"):
        continue
    cmd.append(a)
kokkos = os.path.expanduser("~/.entity/kokkos/5.0.1/openmp/lib")
cmd += [src, "-o", out,
        os.path.join(tree, "framework/libntt_framework.a"),
        os.path.join(tree, "global/libntt_global.a"),
        "-L" + kokkos, "-lkokkoscore", "-lkokkoscontainers", "-lkokkossimd",
        "-L/opt/homebrew/opt/libomp/lib", "-lomp"]
defs = sorted(a for a in cmd if a.startswith("-D"))
print("definitions:", " ".join(defs))
r = subprocess.run(cmd, cwd=entry["directory"], capture_output=True, text=True)
err = "\n".join(l for l in r.stderr.splitlines() if "duplicate libraries" not in l)
if err.strip():
    print(err[-3000:])
sys.exit(r.returncode)
