#!/usr/bin/env python3
"""Momenta along x1 of the pairs created in the first step of a V4 run, as raw
float64 (electrons, then positrons): a compact key for the event-by-event
comparison with the replay of the generator streams on a CPU.

    extract_block0.py <run dir> <electron species> <positron species> <out.bin>
"""
import array
import struct
import sys

run, e_idx, p_idx, out = sys.argv[1:5]
n0 = None
for line in open(run + "/run.log"):
    if line.startswith("BHV step 0 "):
        t = line.split()
        n0 = int(float(dict(zip(t[1::2], t[2::2]))["pairs"]))
        break
with open(out, "wb") as g:
    g.write(struct.pack("<Q", n0))
    for idx in (e_idx, p_idx):
        with open(f"{run}/bhv_species_{idx}_rank_0.bin", "rb") as f:
            n, size = struct.unpack("<QQ", f.read(16))
            a = array.array("f" if size == 4 else "d")
            a.frombytes(f.read(n0 * size))
            array.array("d", a).tofile(g)
print(out, n0, "pairs")
