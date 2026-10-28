#!/usr/bin/env python3
"""
Development-time generator of the threshold series of the two rest-frame
Bethe-Heitler laws (P2 step 4, round 2, finding 1). Not used by the build or by
any test (the test suite contains no Python, D18); its output,
`src/archetypes/hadronic/bh_threshold_series.inc`, is committed.

Usage: python3 tools/bh_threshold_series.py > src/archetypes/hadronic/bh_threshold_series.inc

Exact rational arithmetic (fractions.Fraction) on truncated power series.

1. Energy law (MOK 1969 3D-1000): dsigma/dE+ = p+ p- B / k^3 with B the bracket
   of the formula. With a = p+^2, b = p-^2, the logarithm is
   L = 2 ln((E+E- + p+p- + 1)/k) = 2 artanh(p+p-/(1 + E+E-)) (because
   (1 + E+E-)^2 - p+^2 p-^2 = k^2), odd in p+ and in p-, so B is a function of
   (a, b). Q = B a b is computed as a series in (a, b) to total degree D + 2;
   every term with a^0 or b^0 vanishes identically (asserted) and
   B = sum_{i,j} c_ij a^i b^j, total degree <= D.

2. Single-lepton energy-angle law (MOK 1969 3D-2000 = Blumenthal 1970 eq. 10):
   d^2sigma/(dE+ dcos) = -p+ p- K / (4 k^3) with K the bracket. K p+^2 is
   computed as a series in (p, b, c) = (p+, p-^2, cos theta) truncated at
   weight i + 2j <= W + 2 (c is not a small quantity: its powers are kept
   exactly at each order); every term with p^0 or p^1 vanishes identically
   (asserted) and K = sum c_ijl p^i b^j c^l, weight i + 2j <= W.

Consistency checks printed to stderr: the leading terms B -> (4/3)(a + b) and
K -> -(8/3) b - 4 p^2 (1 - c^2), whose angular integral is -4 B (the angular
law integrates to the energy law).
"""
import sys
from fractions import Fraction as Fr
from math import factorial

D_ENERGY = 16  # total degree in (a, b) of the energy-law bracket B
W_ANGLE = 20   # weight i + 2j in (p+, p-^2) of the angular-law bracket K


def binom_half(n):
    r = Fr(1)
    for k in range(n):
        r *= (Fr(1, 2) - k) / (k + 1)
    return r


def asinh_over(n):  # coefficient of x^(2n) in asinh(x)/x
    return Fr((-1) ** n * factorial(2 * n), 4 ** n * factorial(n) ** 2 * (2 * n + 1))


class Series:
    """truncated multivariate power series; trunc(key) -> True if the monomial is kept"""

    def __init__(self, c, trunc, zero):
        self.c = c
        self.trunc = trunc
        self.zero = zero

    def new(self, c):
        return Series(c, self.trunc, self.zero)

    def const(self, x):
        return self.new({self.zero: Fr(x)} if x else {})

    def __add__(self, o):
        if not isinstance(o, Series):
            o = self.const(o)
        r = dict(self.c)
        for k, v in o.c.items():
            nv = r.get(k, 0) + v
            if nv:
                r[k] = nv
            elif k in r:
                del r[k]
        return self.new(r)

    __radd__ = __add__

    def __neg__(self):
        return self.new({k: -v for k, v in self.c.items()})

    def __sub__(self, o):
        return self + (-o if isinstance(o, Series) else self.const(-o))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, Series):
            o = Fr(o)
            return self.new({k: v * o for k, v in self.c.items()} if o else {})
        r = {}
        for k1, v1 in self.c.items():
            for k2, v2 in o.c.items():
                key = tuple(x + y for x, y in zip(k1, k2))
                if self.trunc(key):
                    r[key] = r.get(key, 0) + v1 * v2
        return self.new({k: v for k, v in r.items() if v})

    __rmul__ = __mul__

    def compose(self, coeffs):
        """sum_n coeffs[n] self^n (self must have no constant term)"""
        assert self.zero not in self.c
        r = self.const(coeffs[0])
        p = self.const(1)
        for n in range(1, len(coeffs)):
            p = p * self
            if not p.c:
                break
            r = r + p * coeffs[n]
        return r


def make(nvar, trunc):
    zero = (0,) * nvar
    base = Series({}, trunc, zero)
    var = [Series({tuple(1 if i == j else 0 for i in range(nvar)): Fr(1)}, trunc, zero) for j in range(nvar)]
    return base, var


NN = 64
SQRT1P = [binom_half(n) for n in range(NN)]
ASINH_OVER = [asinh_over(n) for n in range(NN)]
ATANH_OVER = [Fr(1, 2 * n + 1) for n in range(NN)]
INV1P = [Fr((-1) ** n) for n in range(NN)]


def energy_series(D):
    base, (a, b) = make(2, lambda k: k[0] + k[1] <= D + 2)
    Ep = a.compose(SQRT1P)
    Em = b.compose(SQRT1P)
    Rp = a.compose(ASINH_OVER) * 2  # eps+/p+
    Rm = b.compose(ASINH_OVER) * 2
    EE = Ep * Em
    k = Ep + Em
    k2 = k * k
    inv1pEE = ((EE - 1) * Fr(1, 2)).compose(INV1P) * Fr(1, 2)  # 1/(1 + E+E-)
    M = (a * b * inv1pEE * inv1pEE).compose(ATANH_OVER) * inv1pEE * 2  # L/(p+ p-)
    N = (k2 * (EE * EE + a * b - EE) - EE * a * b * Fr(8, 3)
         - k * Fr(1, 2) * ((EE - a) * Rp * b + (EE - b) * Rm * a))
    Q = a * b * Fr(-4, 3) - EE * (a + b) * 2 + Em * Rp * b + Ep * Rm * a - Rp * Rm * a * b + M * N
    bad = [k for k in Q.c if k[0] == 0 or k[1] == 0]
    assert not bad, "energy law: nonvanishing a^0 or b^0 terms %s" % bad[:4]
    B = {(i - 1, j - 1): v for (i, j), v in Q.c.items() if (i - 1) + (j - 1) <= D}
    assert B.get((1, 0)) == Fr(4, 3) and B.get((0, 1)) == Fr(4, 3) and (0, 0) not in B
    assert all(B.get((i, j), 0) == B.get((j, i), 0) for (i, j) in B)
    return B


def angle_series(W):
    base, (p, b, c) = make(3, lambda k: k[0] + 2 * k[1] <= W + 2)
    p2 = p * p
    Ep = p2.compose(SQRT1P)
    Em = b.compose(SQRT1P)
    k = Ep + Em
    Ep2, Em2, k2 = Ep * Ep, Em * Em, k * k
    sp2 = 1 - c * c
    Dl = Ep - p * c
    invD = (Dl - 1).compose(INV1P)
    invD2 = invD * invD
    invD4 = invD2 * invD2
    aa = Ep - p
    Q2 = (Em + aa) * (Em + aa) + k * p * (1 - c) * 2
    invQ2 = ((Q2 - 4) * Fr(1, 4)).compose(INV1P) * Fr(1, 4)
    eQ_over = invQ2 * (b * invQ2).compose(ATANH_OVER) * 2  # eps^Q/(p- Q)
    EE = Ep * Em
    inv1pEE = ((EE - 1) * Fr(1, 2)).compose(INV1P) * Fr(1, 2)
    M = (p2 * b * inv1pEE * inv1pEE).compose(ATANH_OVER) * inv1pEE * 2  # L/(p+ p-)
    Rm = b.compose(ASINH_OVER) * 2  # eps-/p-
    T1 = sp2 * (Ep2 * 2 + 1) * invD4 * 8
    T2 = (Ep2 * 5 - Em * Ep * 2 + 3) * invD2 * (-2)
    T3 = p2 * (p2 - k2) * invQ2 * invD2 * (-2)
    T4 = Em * invD * (-4)
    T5 = -(M * (Ep * sp2 * (k * 3 + p2 * Em) * invD4 * 4 + Ep2 * (Ep2 + Em2) * invD2 * 4
                + (2 - (Ep2 * 7 + Em * Ep * 3 + Em2) * 2) * invD2 + k * (Ep2 - Ep * Em - 1) * invD * 2))
    T6 = Rm * p2 * invD * 4
    T7 = eQ_over * p2 * (invD2 * 4 - k * invD * 6 - k * (p2 - k2) * invQ2 * invD * 2)
    Kp2 = T1 + T2 + T3 + T4 + T5 + T6 + T7
    bad = [k for k in Kp2.c if k[0] < 2]
    assert not bad, "angular law: nonvanishing p^0 or p^1 terms %s" % bad[:4]
    K = {(i - 2, j, l): v for (i, j, l), v in Kp2.c.items() if (i - 2) + 2 * j <= W}
    assert K.get((0, 1, 0)) == Fr(-8, 3) and K.get((2, 0, 0)) == Fr(-4) and K.get((2, 0, 2)) == Fr(4)
    assert not [k for k in K if k[0] + 2 * k[1] == 0]
    return K


def lit(v):
    # 25 significant digits: the long double instantiation gets the coefficient to its precision
    if v == 0:
        return "0.0L"
    from decimal import Decimal, getcontext
    getcontext().prec = 40
    d = Decimal(v.numerator) / Decimal(v.denominator)
    return format(d, ".24e") + "L"


def main():
    B = energy_series(D_ENERGY)
    K = angle_series(W_ANGLE)
    sys.stderr.write("energy law: %d coefficients (degree <= %d); angular law: %d coefficients (weight <= %d)\n"
                     % (len(B), D_ENERGY, len(K), W_ANGLE))
    out = []
    out.append("// Generated by tools/bh_threshold_series.py (exact rational arithmetic); do not edit.")
    out.append("// Energy law (MOK 3D-1000): dsigma/dE+ = p+ p- B(a, b)/k^3, a = p+^2, b = p-^2,")
    out.append("//   B = sum_{i+j<=%d} BS_C[i][j] a^i b^j (B is symmetric; BS_C[1][0] = BS_C[0][1] = 4/3)." % D_ENERGY)
    out.append("// Angular law (MOK 3D-2000 = Blumenthal eq. 10): d^2sigma/(dE+ dcos) = -p+ p- K/(4 k^3),")
    out.append("//   K = sum_n KS_C[n] p+^KS_I[n] (p-^2)^KS_J[n] cos^KS_L[n], weight KS_I + 2 KS_J <= %d," % W_ANGLE)
    out.append("//   sorted by (i, j, l).")
    out.append("inline constexpr int BS_DEG = %d;" % D_ENERGY)
    out.append("inline constexpr long double BS_C[%d][%d] = {" % (D_ENERGY + 1, D_ENERGY + 1))
    for i in range(D_ENERGY + 1):
        vals = [lit(B.get((i, j), Fr(0))) if i + j <= D_ENERGY else "0.0L" for j in range(D_ENERGY + 1)]
        out.append("  { " + ", ".join(vals) + " },")
    out.append("};")
    keys = sorted(K)
    out.append("inline constexpr int KS_W = %d;" % W_ANGLE)
    out.append("inline constexpr int KS_N = %d;" % len(keys))
    out.append("inline constexpr int KS_MAXI = %d;" % max(k[0] for k in keys))
    out.append("inline constexpr int KS_MAXJ = %d;" % max(k[1] for k in keys))
    out.append("inline constexpr int KS_MAXL = %d;" % max(k[2] for k in keys))
    out.append("inline constexpr unsigned char KS_I[KS_N] = {" + ", ".join(str(k[0]) for k in keys) + "};")
    out.append("inline constexpr unsigned char KS_J[KS_N] = {" + ", ".join(str(k[1]) for k in keys) + "};")
    out.append("inline constexpr unsigned char KS_L[KS_N] = {" + ", ".join(str(k[2]) for k in keys) + "};")
    out.append("inline constexpr long double KS_C[KS_N] = {")
    for n in range(0, len(keys), 4):
        out.append("  " + ", ".join(lit(K[k]) for k in keys[n:n + 4]) + ",")
    out.append("};")
    print("\n".join(out))


if __name__ == "__main__":
    main()
