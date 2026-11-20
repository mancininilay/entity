#!/usr/bin/env python3
"""Event-by-event comparison of the pairs created on the GPU in the first step
with the outcomes computed on a CPU from the same generator streams.

    match_block0.py <block0_ux1.bin (extract_block0.py)> <replay.bin (bhv-replay-pool.xc)>

Every pair of the run is looked up, by the electron momentum along the
emitter, among the replayed outcomes; the positron momentum of the nearest
one is compared too.
"""
import array
import bisect
import collections
import struct
import sys

run_file, replay_file = sys.argv[1:3]
with open(run_file, "rb") as f:
    (n0,) = struct.unpack("<Q", f.read(8))
    e = array.array("d"); e.frombytes(f.read(8 * n0))
    p = array.array("d"); p.frombytes(f.read(8 * n0))
with open(replay_file, "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    data = f.read()
rec = struct.Struct("<IIddd")
events = [rec.unpack_from(data, i * rec.size) for i in range(n)]
events.sort(key=lambda r: r[2])
keys = [r[2] for r in events]
print(f"run: {n0} pairs of the first step; replay: {n} outcomes")
worst_e = worst_p = 0.0
bins = collections.Counter()
trials = collections.Counter()
used = set()
unmatched = 0
duplicates = 0
for a, b in zip(e, p):
    j = bisect.bisect_left(keys, a)
    best = None
    for k in (j - 1, j):
        if 0 <= k < n and (best is None or abs(keys[k] - a) < abs(keys[best] - a)):
            best = k
    de = abs(keys[best] - a) / abs(a)
    dp = abs(events[best][3] - b) / abs(b)
    d = max(de, dp)
    if d > 1.0e-6:
        unmatched += 1
        continue
    if best in used:
        duplicates += 1
    used.add(best)
    worst_e, worst_p = max(worst_e, de), max(worst_p, dp)
    bins["<= 1e-15" if d <= 1e-15 else "<= 1e-13" if d <= 1e-13 else "<= 1e-11" if d <= 1e-11 else "<= 1e-9" if d <= 1e-9 else "<= 1e-6"] += 1
    trials[events[best][1]] += 1
print(f"matched within 1e-6 (relative, both leptons): {n0 - unmatched} of {n0}; not matched: {unmatched}; replayed outcome matched twice: {duplicates}")
print("largest relative difference among the matched: electron %.3e, positron %.3e" % (worst_e, worst_p))
print("relative difference of the matched:", dict(bins))
exact = sum(1 for a, b in zip(e, p) if (lambda j: j < n and keys[j] == a and events[j][3] == b)(bisect.bisect_left(keys, a)))
print(f"bit-identical in both momenta: {exact}")
t = sorted(trials.items())
print("trial index of the matched outcomes (uses of a generator before this one): min %d, max %d; counts by index: %s" % (t[0][0], t[-1][0], " ".join(f"{k}:{v}" for k, v in t)))
states = collections.Counter(events[k][0] for k in used)
print(f"generators that produced the matched pairs: {len(states)} of 221184")
