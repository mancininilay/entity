#!/usr/bin/env python3
"""V4 samples step by step (standard library only).

    v4_blocks.py <run dir> <electron species> <positron species> <tabulated mean D>

The pairs created in one step occupy a contiguous block of slots (slot =
offset + ticket, no sorting). For every block and for the whole sample:
mean(E- - E+), which is zero by the symmetry of the Born cross section, and
mean(E- + E+ - eps0) - <D>, each with its standard error from the sample.
"""
import array
import math
import struct
import sys


def read(path):
    with open(path, "rb") as f:
        n, size = struct.unpack("<QQ", f.read(16))
        code = "f" if size == 4 else "d"
        u = []
        for _ in range(3):
            a = array.array(code)
            a.frombytes(f.read(n * size))
            u.append(a)
    return n, u


def main():
    run, e_idx, p_idx, d_tab = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    eps0 = 0.5
    counts = [0]
    for line in open(run + "/run.log"):
        if line.startswith("BHV step"):
            t = line.split()
            row = dict(zip(t[1::2], t[2::2]))
            counts.append(int(float(row["pairs"])))
    n, ue = read(f"{run}/bhv_species_{e_idx}_rank_0.bin")
    n2, up = read(f"{run}/bhv_species_{p_idx}_rank_0.bin")
    assert n == n2 == counts[-1], (n, n2, counts)
    tot = [0, 0.0, 0.0, 0.0, 0.0]
    print(f"{run}: {n} pairs, steps {len(counts) - 1}, tabulated <D> = {d_tab}")
    for s in range(len(counts) - 1):
        a, b = counts[s], counts[s + 1]
        m = b - a
        sa = sa2 = sd = sd2 = 0.0
        for i in range(a, b):
            em = math.sqrt(1.0 + ue[0][i] ** 2 + ue[1][i] ** 2 + ue[2][i] ** 2)
            ep = math.sqrt(1.0 + up[0][i] ** 2 + up[1][i] ** 2 + up[2][i] ** 2)
            x = em - ep
            d = em + ep - eps0 - d_tab
            sa += x
            sa2 += x * x
            sd += d
            sd2 += d * d
        for k, v in enumerate((m, sa, sa2, sd, sd2)):
            tot[k] += v
        report(f"  step {s}", m, sa, sa2, sd, sd2)
    report("  all   ", *tot)


def report(label, m, sa, sa2, sd, sd2):
    ma, md = sa / m, sd / m
    ea = math.sqrt((sa2 / m - ma * ma) / m)
    ed = math.sqrt((sd2 / m - md * md) / m)
    print(f"{label}: {m:8d} pairs; mean(E- - E+) = {ma:+.6f} +- {ea:.6f} ({ma / ea:+.2f} s.e.); mean D - <D> = {md:+.6f} +- {ed:.6f} ({md / ed:+.2f} s.e.)")


if __name__ == "__main__":
    main()
