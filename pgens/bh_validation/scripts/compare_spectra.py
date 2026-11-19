#!/usr/bin/env python3
"""V9 for the spectra: two outputs of bhv_spectra (two species orders) against each other.

    compare_spectra.py <output A> <output B>

From the STAT lines: number of pairs within 4 sqrt(n_A + n_B), and the means
of E-, E+, cos-, cos+ and D within 4 combined standard errors.
"""
import math
import sys


def stats(path):
    out = {}
    for line in open(path):
        if line.startswith("STAT "):
            t = line.split()
            out[t[1]] = (float(t[2]), [(float(t[3 + 2 * q]), float(t[4 + 2 * q])) for q in range(5)])
    return out


def main():
    A, B = stats(sys.argv[1]), stats(sys.argv[2])
    names = ["E-", "E+", "cos-", "cos+", "D"]
    n_fail = 0
    for g in sorted(A, key=float):
        (na, qa), (nb, qb) = A[g], B[g]
        dn = (na - nb) / math.sqrt(na + nb)
        cells = [f"pairs {na:.0f} vs {nb:.0f} ({dn:+.2f})"]
        ok = abs(dn) <= 4.0
        for name, (ma, sa), (mb, sb) in zip(names, qa, qb):
            d = (ma - mb) / math.sqrt(sa * sa + sb * sb)
            ok = ok and abs(d) <= 4.0
            cells.append(f"<{name}> {ma:.6g} vs {mb:.6g} ({d:+.2f})")
        n_fail += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} V9 spectra at Gamma_p = {g}: " + "; ".join(cells) + " [differences in combined s.e., limit 4]")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
