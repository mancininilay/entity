#!/usr/bin/env python3
"""Compile and link one copy of the unit test with the commands of a build tree of the pull-request
branch (its compile_commands.json and the link line of ninja), with an include overlay in front.

    build_mutant.py <build tree> <overlay directory | none> <test source> <output executable>
"""
import json
import os
import shlex
import subprocess
import sys

tree, overlay, src, out = sys.argv[1:5]
cc = json.load(open(os.path.join(tree, "compile_commands.json")))
entry = next(e for e in cc if e["file"].endswith("tests/archetypes/bethe_heitler.cpp"))
args = shlex.split(entry["command"])
obj = out + ".o"
cmd, skip = [], False
for a in args:
    if skip:
        skip = False
        continue
    if a == "-o":
        skip = True
        continue
    if a == "-c" or a.endswith("tests/archetypes/bethe_heitler.cpp"):
        continue
    cmd.append(a)
if overlay != "none":
    cmd.insert(1, "-I" + overlay)
cmd += ["-c", src, "-o", obj]
r = subprocess.run(cmd, cwd=entry["directory"], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-3000:])
    sys.exit(r.returncode)
link = subprocess.run(["ninja", "-C", tree, "-t", "commands", "tests/archetypes/test-archetypes-bethe_heitler.xc"],
                      capture_output=True, text=True).stdout.strip().splitlines()[-1]
largs = shlex.split(link)
largs = [a for a in largs if a not in (":", "&&")]
i = largs.index("-o")
largs[i + 1] = out
largs = [obj if a.endswith("bethe_heitler.cpp.o") else a for a in largs]
r = subprocess.run(largs, cwd=tree, capture_output=True, text=True)
err = "\n".join(l for l in r.stderr.splitlines() if "duplicate libraries" not in l)
if err.strip():
    print(err[-3000:])
os.remove(obj)
sys.exit(r.returncode)
