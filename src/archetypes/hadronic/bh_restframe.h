/**
 * @file archetypes/hadronic/bh_restframe.h
 * @brief Bethe-Heitler pair production on a static point source (Z = 1,
 *        unscreened, Born approximation): the rest-frame laws of the pair
 *        sampler (P2 step 4, contract revision 1.0 section 1, spec III.6)
 *
 * Host-only code. The two closed forms cancel by many orders near threshold
 * and at the end points of the energy range; since round 2 they are evaluated
 * with a representation that is stable in double (threshold series, run-time
 * Taylor series for a slow lepton, stable sub-expressions elsewhere; see the
 * namespace `stable`), templated on the floating type; production uses
 * double. Nothing here depends on Kokkos, the problem generator or the input
 * parser.
 *
 * Units: energies in m_e c^2, momenta in m_e c, cross sections in alpha r_e^2.
 * Primed quantities of the contract (proton rest frame) are unprimed here:
 * k is the photon energy (eps'), E+ and E- the lepton total energies
 * (E+ + E- = k, no recoil energy), p = sqrt(E^2 - 1), theta the angle of a
 * lepton to the photon direction.
 *
 * Laws implemented and their sources (identifications as corrected in round
 * 2; the algebra, signs and normalisation of the three laws were checked
 * against the PDFs by the referee, Codex 1, review of 2026-10-27: they are the
 * papers'; the numerical anchors A1 to A5 and the closed-vs-integral check of
 * the test suite verify them independently):
 *
 *   1. dsigma/dE+ (the energy law): Bethe-Heitler Born cross section
 *      integrated over both lepton directions, Motz, Olsen and Koch 1969,
 *      Rev. Mod. Phys. 41, 581, formula 3D-1000 (Z = 1, no screening), which
 *      cites Heitler 1954, p. 258, eq. (8). Its integral over E+ reproduces
 *      Maximon's sigma~(k) (anchor A1). g(E+ | k) = (1/sigma~) dsigma/dE+.
 *
 *   2. d^5 sigma/(dE+ dOmega+ dOmega-) (the fully differential law): Bethe
 *      and Heitler 1934, MOK 1969 formula 3D-3000 (Z = 1, no screening),
 *      Heitler 1954, p. 257, eq. (6). It is the parent of
 *      both laws above and below; here it is integrated numerically over the
 *      direction of the second lepton (`dsigma_dE_dcos_numeric`) on the
 *      sphere of the momentum transfer, which makes the 1/q^4 structure
 *      analytic (log grid in q^2) and leaves the forward peak of the second
 *      lepton (1/D-^2, D- = E- - p- cos theta-) as a located bump handled by
 *      graded panels.
 *
 *   3. d^2 sigma/(dE+ dcos theta+) (the single-lepton energy-angle law): the
 *      Sauter-Gluckstern-Hull expression, MOK 1969 formula 3D-2000 (Z = 1,
 *      no screening) = Blumenthal 1970, PRD 1, 1596, eq. (10) (the
 *      definitions of its symbols in his eq. (11)), i.e. the law 2 integrated
 *      over the other lepton's direction in closed form
 *      (`dsigma_dE_dcos_closed`). History (round 1): it was first written as
 *      the pair analogue of the bremsstrahlung formula 2BN of Koch and Motz
 *      1959 (substitution E0 -> -E+, E -> E-, k -> -k), and the signs of the
 *      three logarithmic groups and the overall factor were fixed by agreement
 *      with the numerical integral of law 2; the referee confirmed that the
 *      signs so fixed are the paper's. The test suite re-verifies the
 *      agreement with law 2 at every run. h(E+, cos theta+ | k) = (1/sigma~)
 *      d^2 sigma.
 *      By the e-/e+ symmetry of the Born cross section the same law holds for
 *      the electron with E+ -> E-.
 *
 *   4. The recoil moment q_k(k) = k - 2 <p cos theta>_h (contract 1.3), by
 *      composite Gauss-Legendre quadrature over E+ and over cos theta+ of the
 *      law 3, with the stated convergence (`RestFrameQuadrature::refined`).
 *
 * @namespaces:
 *   - arch::bh::rf::
 */

#ifndef ARCHETYPES_HADRONIC_BH_RESTFRAME_H
#define ARCHETYPES_HADRONIC_BH_RESTFRAME_H

#include "archetypes/hadronic/bh_physics.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace arch::bh::rf {

  // ------------------------------------------------------------------------ //
  // Kinematic helpers
  // ------------------------------------------------------------------------ //

  /// p = sqrt(E^2 - 1) without cancellation
  inline double momentum(double E) {
    return E > 1.0 ? std::sqrt((E - 1.0) * (E + 1.0)) : 0.0;
  }

  /// E - p = 1 / (E + p)
  inline double energy_minus_momentum(double E, double p) {
    return 1.0 / (E + p);
  }

  // ------------------------------------------------------------------------ //
  // Stable evaluation in any floating type (round 2, finding 1)
  //
  // The two closed forms (law 1 and law 3) are sums of terms of size 1/p^4 and
  // 1/p^2 that cancel to an O(p^2) result as the lepton momenta go to zero:
  // the relative cancellation is ~p^6 at threshold (both momenta small) and
  // ~p+^2 or worse at an end point of the energy range (one momentum small).
  // Round 1 evaluated them in long double, which has 64 bits on x86 and 53 on
  // Apple arm64 (where native A2 failed). Round 2 uses a representation that
  // is stable in double, in three regions of (p+, p-):
  //
  //  - threshold region, both momenta <= P_SW = 0.15: exact-rational power
  //    series generated by tools/bh_threshold_series.py (bh_threshold_series.inc):
  //    law 1 as B(a, b), a = p+^2, b = p-^2, total degree 16 (152 coefficients;
  //    the logarithm is 2 artanh(p+p-/(1 + E+E-)), odd in each momentum, so B
  //    is a function of (a, b)); law 3 as K(p+, p-^2, cos theta), weight
  //    i + 2j <= 20 (503 coefficients). Every term that the cancellation
  //    removes vanishes identically in rational arithmetic (asserted by the
  //    generator). Truncation: relative <= ~1e-14 at p = P_SW (measured
  //    against __float128 in the round-2 scratch run, report R2 stage 1);
  //  - slow region, the lepton whose factor 1/p^2 appears (p+ for law 3, the
  //    smaller momentum for law 1) below P_SLOW = 0.1 and the other above
  //    P_SW: a truncated Taylor series in that momentum, computed at run time
  //    in the floating type T (`Tps`, 20 terms); the coefficients that the
  //    cancellation sets to zero (orders 0 and 1) are dropped exactly, the
  //    others are computed from O(1) quantities. Truncation <= P_SLOW^18;
  //  - general region: the closed forms with stable sub-expressions (no
  //    logarithm of a number near 1): eps = 2 asinh(p), L = 2 artanh(p+p-/
  //    (1 + E+E-)), eps^Q = 2 artanh(p-/Q). The remaining amplification is at
  //    most ~1/(P_SLOW^2 P_SW^2) ~ 4e3 (measured: <= 2e-11 relative).
  //
  // Every region is evaluated in the template type T; nothing relies on the
  // width of long double.
  // ------------------------------------------------------------------------ //
  namespace stable {

    inline constexpr double P_SW   = 0.15; // threshold region: both momenta <= P_SW
    inline constexpr double P_SLOW = 0.1;  // slow region: the critical momentum < P_SLOW
    inline constexpr int    TPS_N  = 20;   // terms of the run-time Taylor series

#include "archetypes/hadronic/bh_threshold_series.inc"

    enum class Region : int {
      Threshold = 0,
      Slow      = 1,
      General   = 2
    };

    /// truncated univariate power series sum_{n<N} c[n] u^n in the type T
    template <class T, int N>
    struct Tps {
      T c[N];

      Tps() {
        for (int n = 0; n < N; ++n) {
          c[n] = T(0);
        }
      }

      explicit Tps(T v) : Tps() {
        c[0] = v;
      }

      static Tps var() {
        Tps r;
        if (N > 1) {
          r.c[1] = T(1);
        }
        return r;
      }

      friend Tps operator+(const Tps& x, const Tps& y) {
        Tps r;
        for (int n = 0; n < N; ++n) {
          r.c[n] = x.c[n] + y.c[n];
        }
        return r;
      }

      friend Tps operator-(const Tps& x, const Tps& y) {
        Tps r;
        for (int n = 0; n < N; ++n) {
          r.c[n] = x.c[n] - y.c[n];
        }
        return r;
      }

      friend Tps operator-(const Tps& x) {
        Tps r;
        for (int n = 0; n < N; ++n) {
          r.c[n] = -x.c[n];
        }
        return r;
      }

      friend Tps operator+(const Tps& x, T v) {
        Tps r = x;
        r.c[0] += v;
        return r;
      }

      friend Tps operator-(const Tps& x, T v) {
        Tps r = x;
        r.c[0] -= v;
        return r;
      }

      friend Tps operator*(const Tps& x, T v) {
        Tps r;
        for (int n = 0; n < N; ++n) {
          r.c[n] = x.c[n] * v;
        }
        return r;
      }

      friend Tps operator*(const Tps& x, const Tps& y) {
        Tps r;
        for (int n = 0; n < N; ++n) {
          T s = T(0);
          for (int m = 0; m <= n; ++m) {
            s += x.c[m] * y.c[n - m];
          }
          r.c[n] = s;
        }
        return r;
      }
    };

    template <class T, int N>
    inline Tps<T, N> recip(const Tps<T, N>& x) {
      Tps<T, N> r;
      const T   r0 = T(1) / x.c[0];
      r.c[0]       = r0;
      for (int n = 1; n < N; ++n) {
        T s = T(0);
        for (int m = 1; m <= n; ++m) {
          s += x.c[m] * r.c[n - m];
        }
        r.c[n] = -r0 * s;
      }
      return r;
    }

    template <class T, int N>
    inline Tps<T, N> sqrt(const Tps<T, N>& x) {
      Tps<T, N> r;
      r.c[0] = std::sqrt(x.c[0]);
      for (int n = 1; n < N; ++n) {
        T s = x.c[n];
        for (int m = 1; m < n; ++m) {
          s -= r.c[m] * r.c[n - m];
        }
        r.c[n] = s / (T(2) * r.c[0]);
      }
      return r;
    }

    /// artanh of a series with |x0| < 1: f' = x'/(1 - x^2)
    template <class T, int N>
    inline Tps<T, N> atanh(const Tps<T, N>& x) {
      Tps<T, N> d;
      for (int n = 0; n + 1 < N; ++n) {
        d.c[n] = T(n + 1) * x.c[n + 1];
      }
      const Tps<T, N> q = d * recip(Tps<T, N>(T(1)) - x * x);
      Tps<T, N>       r;
      r.c[0] = std::atanh(x.c[0]);
      for (int n = 1; n < N; ++n) {
        r.c[n] = q.c[n - 1] / T(n);
      }
      return r;
    }

    /// sum_n coef(n) x^n for a series x without constant term (Horner)
    template <class T, int N, class C>
    inline Tps<T, N> compose(const Tps<T, N>& x, C&& coef) {
      Tps<T, N> r(coef(N - 1));
      for (int n = N - 2; n >= 0; --n) {
        r = r * x + coef(n);
      }
      return r;
    }

    /// coefficient of x^n in asinh(sqrt(x))/sqrt(x): (-1)^n (2n-1)!!/((2n)!! (2n+1))
    template <class T>
    inline T asinh_over_coef(int n) {
      T c = T(1);
      for (int m = 1; m <= n; ++m) {
        c *= -T(2 * m - 1) / T(2 * m);
      }
      return c / T(2 * n + 1);
    }

    /// 2 asinh(p)/p for p >= 0 (p^2 = a), stable at p -> 0
    template <class T>
    inline T two_asinh_over(T p, T a) {
      if (p < T(0.05)) {
        T s = T(0);
        for (int n = 12; n >= 0; --n) {
          s = s * a + asinh_over_coef<T>(n);
        }
        return T(2) * s;
      }
      return T(2) * std::asinh(p) / p;
    }

    /// 2 artanh(t)/t for |t| < 1, stable at t -> 0
    template <class T>
    inline T two_atanh_over(T t) {
      const T t2 = t * t;
      if (t2 < T(0.0025)) {
        T s = T(0);
        for (int n = 14; n >= 0; --n) {
          s = s * t2 + T(1) / T(2 * n + 1);
        }
        return T(2) * s;
      }
      return T(2) * std::atanh(t) / t;
    }

    // ---------------------------------------------------------------------- //
    // the threshold series
    // ---------------------------------------------------------------------- //

    /// B(a, b) of law 1 (both momenta <= P_SW)
    template <class T>
    inline T series_B(T a, T b) {
      T s = T(0);
      for (int i = BS_DEG; i >= 0; --i) {
        T r = T(0);
        for (int j = BS_DEG - i; j >= 0; --j) {
          r = r * b + static_cast<T>(BS_C[i][j]);
        }
        s = s * a + r;
      }
      return s;
    }

    /// K(p+, b = p-^2, c = cos theta) of law 3 (both momenta <= P_SW)
    template <class T>
    inline T series_K(T p, T b, T c) {
      T pw_p[KS_MAXI + 1], pw_b[KS_MAXJ + 1], pw_c[KS_MAXL + 1];
      pw_p[0] = pw_b[0] = pw_c[0] = T(1);
      for (int n = 1; n <= KS_MAXI; ++n) {
        pw_p[n] = pw_p[n - 1] * p;
      }
      for (int n = 1; n <= KS_MAXJ; ++n) {
        pw_b[n] = pw_b[n - 1] * b;
      }
      for (int n = 1; n <= KS_MAXL; ++n) {
        pw_c[n] = pw_c[n - 1] * c;
      }
      // highest weights first (smallest terms first)
      T s = T(0);
      for (int n = KS_N - 1; n >= 0; --n) {
        s += static_cast<T>(KS_C[n]) * pw_p[KS_I[n]] * pw_b[KS_J[n]] * pw_c[KS_L[n]];
      }
      return s;
    }

    // ---------------------------------------------------------------------- //
    // law 1: the bracket times a b in the general region, and the slow series
    // ---------------------------------------------------------------------- //

    /**
     * @brief Q = B a b of law 1 with stable sub-expressions:
     *        Q = -(4/3) a b - 2 E+E- (a + b) + E- R+ b + E+ R- a - R+ R- a b + M N,
     *        R = eps/p = 2 asinh(p)/p, M = L/(p+ p-) = 2 artanh(t)/t /(1 + E+E-)
     *        with t = p+ p-/(1 + E+E-), N = k^2 (E+^2 E-^2 + a b - E+E-) - (8/3)
     *        E+E- a b - (k/2) [(E+E- - a) R+ b + (E+E- - b) R- a]
     */
    template <class T>
    inline T energy_Q_general(T Ep, T Em, T pp, T pm, T a, T b) {
      const T k   = Ep + Em;
      const T EE  = Ep * Em;
      const T Rp  = two_asinh_over(pp, a);
      const T Rm  = two_asinh_over(pm, b);
      const T i1  = T(1) / (T(1) + EE);
      const T M   = two_atanh_over(pp * pm * i1) * i1;
      const T ab  = a * b;
      const T Nn  = k * k * (EE * EE + ab - EE) - (T(8) / T(3)) * EE * ab -
                   T(0.5) * k * ((EE - a) * Rp * b + (EE - b) * Rm * a);
      return -(T(4) / T(3)) * ab - T(2) * EE * (a + b) + Em * Rp * b + Ep * Rm * a - Rp * Rm * ab + M * Nn;
    }

    /**
     * @brief B of law 1 for p+ < P_SLOW (p+ the smaller momentum): the same Q as
     *        a series in u = p+, orders 0 and 1 dropped (zero identically: B is
     *        finite at p+ = 0), B = sum_{n>=2} Q_n p+^(n-2) / b
     */
    template <class T>
    inline T energy_B_slow(T pp, T pm, T Em, T b) {
      using X       = Tps<T, TPS_N>;
      const X u     = X::var();
      const X a     = u * u;
      const X Ep    = sqrt(a + T(1));
      const X Rp    = compose(a, [](int n) { return T(2) * asinh_over_coef<T>(n); });
      const T Rm    = two_asinh_over(pm, b);
      const X EE    = Ep * Em;
      const X k     = Ep + Em;
      const X i1    = recip(EE + T(1));
      const X t2    = a * i1 * i1 * b;
      const X M     = compose(t2, [](int n) { return T(1) / T(2 * n + 1); }) * i1 * T(2);
      const X ab    = a * b;
      const X Nn    = k * k * (EE * EE + ab - EE) - EE * ab * (T(8) / T(3)) -
                   k * T(0.5) * ((EE - a) * Rp * b + (EE - b) * a * Rm);
      const X Q     = ab * (-T(4) / T(3)) - EE * (a + b) * T(2) + Rp * (Em * b) + Ep * a * Rm - Rp * ab * Rm + M * Nn;
      T       s     = T(0);
      for (int n = TPS_N - 1; n >= 2; --n) {
        s = s * pp + Q.c[n];
      }
      return s / b;
    }

    /// the region of law 1 for momenta pp <= pm (canonical order)
    template <class T>
    inline Region energy_region(T pp, T pm) {
      if (pm <= T(P_SW)) {
        return Region::Threshold;
      }
      if (pp < T(P_SLOW)) {
        return Region::Slow;
      }
      return Region::General;
    }

  } // namespace stable

  // ------------------------------------------------------------------------ //
  // Law 1: the energy distribution, Motz, Olsen and Koch 1969 3D-1000
  // (Heitler 1954, p. 258, eq. 8)
  // ------------------------------------------------------------------------ //

  /**
   * @brief dsigma/dE+ in alpha r_e^2 per m_e c^2 (MOK 1969 formula 3D-1000,
   *        Z = 1, unscreened; Heitler 1954 p. 258 eq. 8), evaluated in the type
   *        T with the stable representation of namespace `stable`. Zero outside
   *        1 < E+ < k - 1. dsigma/dE+ = p+ p- B / k^3.
   */
  template <class T>
  inline T dsigma_dE_T(T k, T Ep) {
    T Em = k - Ep;
    if (!(Ep > T(1)) || !(Em > T(1))) {
      return T(0);
    }
    // the law is symmetric under E+ <-> E-: evaluate with the operands in a
    // canonical order so that g(E) and g(k - E) are bitwise equal whenever the
    // two energies are exactly complementary
    if (Ep > Em) {
      std::swap(Ep, Em);
    }
    const T xp = Ep - T(1), xm = Em - T(1);
    const T a = xp * (xp + T(2)), b = xm * (xm + T(2));
    const T pp = std::sqrt(a), pm = std::sqrt(b);
    T       B;
    switch (stable::energy_region(pp, pm)) {
      case stable::Region::Threshold:
        B = stable::series_B(a, b);
        break;
      case stable::Region::Slow:
        B = stable::energy_B_slow(pp, pm, Em, b);
        break;
      default:
        B = stable::energy_Q_general(Ep, Em, pp, pm, a, b) / (a * b);
        break;
    }
    return pp * pm * B / (k * k * k);
  }

  /**
   * @brief The same in the round-1 form (direct closed form, logarithms of
   *        numbers near 1), kept only as the reference that exhibits finding 1
   *        in the tests; not used by any production path
   */
  template <class T>
  inline T dsigma_dE_round1_T(T k, T Ep) {
    T Em = k - Ep;
    if (!(Ep > T(1)) || !(Em > T(1))) {
      return T(0);
    }
    if (Ep > Em) {
      std::swap(Ep, Em);
    }
    const T pp  = std::sqrt((Ep - T(1)) * (Ep + T(1)));
    const T pm  = std::sqrt((Em - T(1)) * (Em + T(1)));
    const T ep  = T(2) * std::log(Ep + pp);
    const T em  = T(2) * std::log(Em + pm);
    const T L   = T(2) * std::log((Ep * Em + pp * pm + T(1)) / k);
    const T pp2 = pp * pp, pm2 = pm * pm;
    const T pp3 = pp2 * pp, pm3 = pm2 * pm;
    const T EE  = Ep * Em;
    const T br  = -T(4) / T(3) - T(2) * EE * (pp2 + pm2) / (pp2 * pm2) + ep * Em / pp3 + em * Ep / pm3 -
                 ep * em / (pp * pm) +
                 L * (k * k * (EE * EE + pp2 * pm2) / (pp3 * pm3) - (T(8) / T(3)) * EE / (pp * pm) -
                      (k / (T(2) * pp * pm)) *
                        ((EE - pp2) * ep / pp3 + (EE - pm2) * em / pm3 + T(2) * k * EE / (pp2 * pm2)));
    return pp * pm * br / (k * k * k);
  }

  /// law 1 in the type T, as double (the quadratures are in double)
  template <class T>
  inline double dsigma_dE_as(double k, double Ep) {
    return static_cast<double>(dsigma_dE_T<T>(static_cast<T>(k), static_cast<T>(Ep)));
  }

  /// the production evaluation: double
  inline double dsigma_dE(double k, double Ep) {
    return dsigma_dE_as<double>(k, Ep);
  }

  /// g(E+ | k) = dsigma/dE+ / sigma~(k) (Maximon's total)
  inline double g(double k, double Ep) {
    const double s = sigma_tilde(k);
    return s > 0.0 ? dsigma_dE(k, Ep) / s : 0.0;
  }

  // ------------------------------------------------------------------------ //
  // Law 2: the fully differential Bethe-Heitler law (MOK 3D-3000, Heitler 1954
  // p. 257 eq. 6) and
  // its numerical integration over the second lepton's direction
  // ------------------------------------------------------------------------ //

  /**
   * @brief Quadrature parameters of the integration over the second lepton's
   *        direction (q-sphere parametrisation: u = ln q^2, azimuth psi)
   */
  struct AngularQuadrature {
    int    gl_order      = 12;   // Gauss-Legendre order per panel
    double max_panel_u   = 1.0;  // largest panel in u = ln q^2
    double max_panel_psi = 0.8;  // largest panel in psi (radians)
    double grading       = 2.0;  // panel growth factor away from the peak

    AngularQuadrature refined() const {
      AngularQuadrature a = *this;
      a.gl_order          = std::min(64, gl_order + 4);
      a.max_panel_u      *= 0.5;
      a.max_panel_psi    *= 0.5;
      return a;
    }
  };

  /**
   * @brief Panel edges on [a, b] graded away from the interval [c_lo, c_hi]
   *        (the peak), growing by `grading` up to `max_width`
   */
  inline void graded_panels(double               a,
                            double               b,
                            double               c_lo,
                            double               c_hi,
                            double               grading,
                            double               max_width,
                            std::vector<double>& edges) {
    edges.clear();
    if (!(b > a)) {
      return;
    }
    c_lo = std::max(a, std::min(b, c_lo));
    c_hi = std::max(c_lo, std::min(b, c_hi));
    // the peak interval itself, split if wider than max_width
    std::vector<double> left, right;
    {
      const double w    = c_hi - c_lo;
      const int    npan = std::max(1, static_cast<int>(std::ceil(w / max_width)));
      for (int i = 0; i <= npan; ++i) {
        right.push_back(c_lo + w * static_cast<double>(i) / npan);
      }
    }
    double width = std::max(c_hi - c_lo, 1.0e-300);
    // to the right of the peak
    {
      double x = c_hi;
      while (x < b) {
        width = std::min(width * grading, max_width);
        x     = std::min(b, x + width);
        if (b - x < 0.25 * width) {
          x = b;
        }
        right.push_back(x);
      }
    }
    width = std::max(c_hi - c_lo, 1.0e-300);
    {
      double x = c_lo;
      while (x > a) {
        width = std::min(width * grading, max_width);
        x     = std::max(a, x - width);
        if (x - a < 0.25 * width) {
          x = a;
        }
        left.push_back(x);
      }
    }
    edges.reserve(left.size() + right.size());
    for (auto it = left.rbegin(); it != left.rend(); ++it) {
      edges.push_back(*it);
    }
    for (const double x : right) {
      edges.push_back(x);
    }
    // remove accidental duplicates
    edges.erase(std::unique(edges.begin(),
                            edges.end(),
                            [](double x, double y) {
                              return !(y > x);
                            }),
                edges.end());
  }

  /**
   * @brief d^2 sigma/(dE+ dcos theta+) in alpha r_e^2 by numerical integration
   *        of the fully differential Bethe-Heitler law over the direction of
   *        the second lepton.
   * @param k   photon energy
   * @param Ep  energy of the lepton whose angle is fixed
   * @param omc 1 - cos theta+ (passed directly to resolve the forward peak)
   *
   * Parametrisation: with Q = k z^ - p+ (the momentum transfer when the second
   * lepton is absent), the second lepton's momentum p- lies on the sphere
   * |p-| = p- and the momentum transfer q = Q - p- on the sphere of radius p-
   * about Q; with alpha the polar angle of p-^ about Q^ and psi its azimuth,
   * q^2 = Q^2 + p-^2 - 2 Q p- cos alpha and dOmega- = dq^2 dpsi / (2 Q p-).
   * The law is written as d^5 sigma = (1/(4 pi^2)) (p+ p-/k^3) F dE+ dOmega+ dOmega-/q^4
   * with F the bracket of MOK 3D-3000 (Heitler 1954 p. 257 eq. 6); the normalisation 1/(4 pi^2) is the one
   * for which the angular integral reproduces the energy law 1 (checked by A2;
   * a 1/(2 pi) prefactor gives 2 pi times the energy law, fact of the scratch
   * run of 2026-10-26). The integrand F/q^4 (MOK 3D-3000) becomes F/q^2 in u = ln q^2; the
   * remaining peak is the forward direction of the second lepton
   * (1/D-^2), at alpha = theta_Q, psi = 0, of angular width
   * delta_theta = sqrt(2 (E- - p-)/p-), covered by graded panels in u and in
   * psi. The integrand is even in psi: the range [0, pi] is doubled.
   */
  inline double dsigma_dE_dcos_numeric(double                   k,
                                       double                   Ep,
                                       double                   omc,
                                       const AngularQuadrature& aq) {
    const double Em = k - Ep;
    if (!(Ep > 1.0) || !(Em > 1.0) || !(omc >= 0.0) || !(omc <= 2.0)) {
      return 0.0;
    }
    const double pp  = momentum(Ep), pm = momentum(Em);
    const double a   = energy_minus_momentum(Ep, pp); // E+ - p+
    const double b   = energy_minus_momentum(Em, pm); // E- - p-
    const double sp2 = omc * (2.0 - omc);
    const double sp  = std::sqrt(std::max(0.0, sp2));
    const double Dp  = a + pp * omc; // E+ - p+ cos theta+
    // Q = k z^ - p+ = (-p+ sin theta+, 0, k - p+ cos theta+)
    const double Qx = -pp * sp;
    const double Qz = Em + a + pp * omc; // = k - p+ cos theta+, cancellation-free
    const double Q2 = Qx * Qx + Qz * Qz;
    const double Q  = std::sqrt(Q2);
    // Q^2 - p-^2 = 1 + 2 a E- + a^2 + 2 k p+ omc  (> 0)
    const double Q2mpm2 = 1.0 + 2.0 * a * Em + a * a + 2.0 * k * pp * omc;
    const double qmin   = Q2mpm2 / (Q + pm);
    const double qmax   = Q + pm;
    const double q2min  = qmin * qmin, q2max = qmax * qmax;
    const double umin   = std::log(q2min), umax = std::log(q2max);
    if (!(umax > umin)) {
      return 0.0;
    }
    // frame about Q^: e1 in the (Q^, z^) plane toward +z, e2 = y^
    const double Qhx = Qx / Q, Qhz = Qz / Q;
    const double e1x = Qz / Q, e1z = -Qx / Q; // e1 . z^ = sin theta_Q >= 0
    const double sinTQ = -Qx / Q, cosTQ = Qz / Q;
    const double thetaQ = std::atan2(sinTQ, cosTQ);
    // the forward peak of the second lepton: p-^ = z^, i.e. alpha = theta_Q,
    // psi = 0; angular half-width delta_theta
    const double dtheta = std::min(constants::PI, std::sqrt(2.0 * b / pm));
    auto         u_of_alpha = [&](double alpha) {
      const double q2 = Q2 + pm * pm - 2.0 * Q * pm * std::cos(alpha);
      return std::log(std::max(q2min, std::min(q2max, q2)));
    };
    double u_lo = u_of_alpha(std::max(0.0, thetaQ - dtheta));
    double u_hi = u_of_alpha(std::min(constants::PI, thetaQ + dtheta));
    if (thetaQ - dtheta <= 0.0) {
      u_lo = umin;
    }
    if (thetaQ + dtheta >= constants::PI) {
      u_hi = umax;
    }
    const GaussLegendre gl(aq.gl_order);
    std::vector<double> ue, pe;
    graded_panels(umin, umax, u_lo, u_hi, aq.grading, aq.max_panel_u, ue);

    const double k2    = k * k;
    const double Ep2   = Ep * Ep, Em2 = Em * Em;
    const double pt_p2 = pp * pp * sp2; // |p+_perp|^2
    double       total = 0.0;
    for (std::size_t iu = 0; iu + 1 < ue.size(); ++iu) {
      const double um = 0.5 * (ue[iu] + ue[iu + 1]);
      const double uh = 0.5 * (ue[iu + 1] - ue[iu]);
      for (int i = 0; i < gl.order(); ++i) {
        const double u  = um + uh * gl.x[i];
        const double wu = uh * gl.w[i];
        const double q2 = std::exp(u);
        // alpha from q^2, cancellation-free at both ends
        const double omca = (q2 - q2min) / (2.0 * Q * pm); // 1 - cos alpha
        const double opca = (q2max - q2) / (2.0 * Q * pm); // 1 + cos alpha
        const double cosa = 1.0 - omca;
        const double sina = std::sqrt(std::max(0.0, omca * opca));
        // psi grading: the 1/D-^2 peak at psi = 0 has width
        // delta_psi = delta_theta / sqrt(sin alpha sin theta_Q)
        double dpsi = constants::PI;
        if (sina * sinTQ > 0.0) {
          dpsi = std::min(constants::PI, dtheta / std::sqrt(sina * sinTQ));
        }
        graded_panels(0.0, constants::PI, 0.0, dpsi, aq.grading, aq.max_panel_psi, pe);
        double spsi = 0.0;
        for (std::size_t ip = 0; ip + 1 < pe.size(); ++ip) {
          const double pm_ = 0.5 * (pe[ip] + pe[ip + 1]);
          const double ph  = 0.5 * (pe[ip + 1] - pe[ip]);
          for (int j = 0; j < gl.order(); ++j) {
            const double psi  = pm_ + ph * gl.x[j];
            const double wpsi = ph * gl.w[j];
            const double cpsi = std::cos(psi), spsi_ = std::sin(psi);
            // p- = p- [cos alpha Q^ + sin alpha (cos psi e1 + sin psi e2)]
            const double px = pm * (cosa * Qhx + sina * cpsi * e1x);
            const double py = pm * (sina * spsi_);
            const double pz = pm * (cosa * Qhz + sina * cpsi * e1z);
            const double pt2 = px * px + py * py; // |p-_perp|^2
            // 1 - cos theta- = pt^2 / (p- (p- + pz)), cancellation-free
            const double den = pm + pz;
            const double omcm = den > 1.0e-300 * pm ? pt2 / (pm * den) : 2.0;
            const double Dm   = b + pm * omcm; // E- - p- cos theta-
            const double F =
              -pt_p2 * (4.0 * Em2 - q2) / (Dp * Dp) -
              pt2 * (4.0 * Ep2 - q2) / (Dm * Dm) +
              2.0 * k2 * (pt_p2 + pt2) / (Dp * Dm) +
              2.0 * (pp * sp * px) * (2.0 * Ep2 + 2.0 * Em2 - q2) / (Dp * Dm);
            spsi += wpsi * F;
          }
        }
        total += wu * spsi / q2;
      }
    }
    // d^2 sigma/(dE+ dcos theta+) = (p+ / (2 pi Q k^3)) int du int_0^pi dpsi F/q^2
    return pp / (2.0 * constants::PI * Q * k2 * k) * total;
  }

  // ------------------------------------------------------------------------ //
  // Law 3: the Sauter-Gluckstern-Hull closed form (MOK 1969 3D-2000 =
  // Blumenthal 1970 eq. 10, definitions in his eq. 11)
  // ------------------------------------------------------------------------ //

  namespace stable {

    /**
     * @brief The bracket K of law 3 (d^2sigma/(dE+ dcos) = s0 p+ p- K/(4 k^3))
     *        in the general region, with explicit signs of the logarithmic
     *        groups and stable sub-expressions: L = 2 artanh(p+p-/(1 + E+E-)),
     *        eps- = 2 asinh(p-), eps^Q = 2 artanh(p-/Q)
     */
    template <class T>
    inline T angle_K_general(T Ep, T Em, T pp, T pm, T omc, T sL, T se, T sQ) {
      const T k   = Ep + Em;
      const T a   = T(1) / (Ep + pp); // E+ - p+
      const T sp2 = omc * (T(2) - omc);
      const T D   = a + pp * omc; // Delta+ = E+ - p+ cos theta+
      const T D2 = D * D, D4 = D2 * D2;
      const T pp2 = pp * pp;
      const T k2  = k * k;
      const T Ep2 = Ep * Ep, Em2 = Em * Em;
      // Q^2 = |k z^ - p+|^2 = (E- + a)^2 + 2 k p+ omc
      const T Q2 = (Em + a) * (Em + a) + T(2) * k * pp * omc;
      const T Q  = std::sqrt(Q2);
      const T i1 = T(1) / (T(1) + Ep * Em);
      const T M  = two_atanh_over(pp * pm * i1) * i1;      // L/(p+ p-)
      const T Rm = two_asinh_over(pm, pm * pm);             // eps-/p-
      const T eQ = two_atanh_over(pm / Q) / (Q * Q);        // eps^Q/(p- Q)
      const T t1 = T(8) * sp2 * (T(2) * Ep2 + T(1)) / (pp2 * D4);
      const T t2 = -T(2) * (T(5) * Ep2 - T(2) * Em * Ep + T(3)) / (pp2 * D2);
      const T t3 = -T(2) * (pp2 - k2) / (Q2 * D2);
      const T t4 = -T(4) * Em / (pp2 * D);
      const T t5 = M * (T(4) * Ep * sp2 * (T(3) * k + pp2 * Em) / (pp2 * D4) + T(4) * Ep2 * (Ep2 + Em2) / (pp2 * D2) +
                        (T(2) - T(2) * (T(7) * Ep2 + T(3) * Em * Ep + Em2)) / (pp2 * D2) +
                        T(2) * k * (Ep2 - Ep * Em - T(1)) / (pp2 * D));
      const T t6 = T(4) * Rm / D;
      const T t7 = eQ * (T(4) / D2 - T(6) * k / D - T(2) * k * (pp2 - k2) / (Q2 * D));
      return t1 + t2 + t3 + t4 + sL * t5 + se * t6 + sQ * t7;
    }

    /**
     * @brief K of law 3 for p+ < P_SLOW (p- > P_SW): K p+^2 as a series in
     *        u = p+, orders 0 and 1 dropped (zero identically: K is finite at
     *        p+ = 0), K = sum_{n>=2} (K p+^2)_n p+^(n-2); the paper's signs
     */
    template <class T>
    inline T angle_K_slow(T pp, T pm, T Em, T omc) {
      using X        = Tps<T, TPS_N>;
      const X u      = X::var();
      const X p2     = u * u;
      const X Ep     = sqrt(p2 + T(1));
      const X k      = Ep + Em;
      const X Ep2    = Ep * Ep;
      const T Em2    = Em * Em;
      const X k2     = k * k;
      const T sp2    = omc * (T(2) - omc);
      const X aa     = Ep - u;               // E+ - p+
      const X Dl     = aa + u * omc;         // E+ - p+ cos theta
      const X invD   = recip(Dl);
      const X invD2  = invD * invD;
      const X invD4  = invD2 * invD2;
      const X Q2     = (aa + Em) * (aa + Em) + k * u * (T(2) * omc);
      const X invQ2  = recip(Q2);
      const X invQ   = recip(sqrt(Q2));
      const X eQ     = atanh(invQ * pm) * invQ * (T(2) / pm); // eps^Q/(p- Q)
      const X EE     = Ep * Em;
      const X i1     = recip(EE + T(1));
      const T b      = pm * pm;
      const X M      = compose(p2 * i1 * i1 * b, [](int n) { return T(1) / T(2 * n + 1); }) * i1 * T(2);
      const T Rm     = two_asinh_over(pm, b);
      const X T1     = (Ep2 * T(2) + T(1)) * invD4 * (T(8) * sp2);
      const X T2     = (Ep2 * T(5) - Ep * (T(2) * Em) + T(3)) * invD2 * T(-2);
      const X T3     = p2 * (p2 - k2) * invQ2 * invD2 * T(-2);
      const X T4     = invD * (T(-4) * Em);
      const X T5     = -(M * (Ep * (k * T(3) + p2 * Em) * invD4 * (T(4) * sp2) + Ep2 * (Ep2 + Em2) * invD2 * T(4) +
                         ((Ep2 * T(7) + Ep * (T(3) * Em) + Em2) * T(-2) + T(2)) * invD2 +
                         k * (Ep2 - Ep * Em - T(1)) * invD * T(2)));
      const X T6     = p2 * invD * (T(4) * Rm);
      const X T7     = eQ * p2 * (invD2 * T(4) - k * invD * T(6) - k * (p2 - k2) * invQ2 * invD * T(2));
      const X Kp2    = T1 + T2 + T3 + T4 + T5 + T6 + T7;
      T       s      = T(0);
      for (int n = TPS_N - 1; n >= 2; --n) {
        s = s * pp + Kp2.c[n];
      }
      return s;
    }

    /// the region of law 3 (p+ the momentum of the lepton whose angle is given)
    template <class T>
    inline Region angle_region(T pp, T pm) {
      if (pp <= T(P_SW) && pm <= T(P_SW)) {
        return Region::Threshold;
      }
      if (pp < T(P_SLOW)) {
        return Region::Slow;
      }
      return Region::General;
    }

  } // namespace stable

  /**
   * @brief Signs of the logarithmic groups and the overall factor of the
   *        closed form (see the file header): fixed against the numerical
   *        integral of law 2; the referee confirmed they are the paper's.
   */
  struct ClosedFormSigns {
    double s0 = 1.0; // overall
    double sL = 1.0; // the L group
    double se = 1.0; // the eps- group
    double sQ = 1.0; // the eps^Q group
  };

  /**
   * @brief d^2 sigma/(dE+ dcos theta+) in alpha r_e^2, the closed form with
   *        explicit signs, general-region evaluation only (used by the sign
   *        verification against law 2 at moderate momenta)
   */
  template <class T>
  inline T dsigma_dE_dcos_closed_signed_T(T k, T Ep, T omc, const ClosedFormSigns& s) {
    const T Em = k - Ep;
    if (!(Ep > T(1)) || !(Em > T(1)) || !(omc >= T(0)) || !(omc <= T(2))) {
      return T(0);
    }
    const T xp = Ep - T(1), xm = Em - T(1);
    const T pp = std::sqrt(xp * (xp + T(2))), pm = std::sqrt(xm * (xm + T(2)));
    const T K  = stable::angle_K_general(Ep, Em, pp, pm, omc, T(s.sL), T(s.se), T(s.sQ));
    // d^2 sigma/(dE+ dOmega+) = s0 (p+ p-/(8 pi k^3)) K;  x 2 pi for dcos theta+
    return T(s.s0) * pp * pm * K / (T(4) * k * k * k);
  }

  inline double dsigma_dE_dcos_closed_signed(double k, double Ep, double omc, const ClosedFormSigns& s) {
    return dsigma_dE_dcos_closed_signed_T<double>(k, Ep, omc, s);
  }

  /**
   * @brief The signs fixed by the comparison with the numerical integral of
   *        law 2 (fact, scratch run of 2026-10-26 at 12 kinematic points over
   *        k in [2.5, 1000], E+ from 1.2 to 900, 1 - cos theta+ from 1e-6 to
   *        1.7): with s0 = -1, sL = -1, se = +1, sQ = +1 the ratio closed/numeric
   *        is the same constant at every point to 6 digits, and the constant
   *        fixes the normalisation to d^2 sigma/(dE+ dOmega+) =
   *        -(p+ p-/(8 pi k^3)) bracket (the prefactor of Koch and Motz 2BN);
   *        every other combination varies by factors of order one between the
   *        points. Round 2: the referee confirmed these are the paper's signs.
   */
  inline const ClosedFormSigns& closed_form_signs() {
    static const ClosedFormSigns s { -1.0, -1.0, 1.0, 1.0 };
    return s;
  }

  /**
   * @brief d^2 sigma/(dE+ dcos theta+) in alpha r_e^2 (MOK 3D-2000 =
   *        Blumenthal eq. 10 with the paper's signs), evaluated in the type T
   *        with the stable representation of namespace `stable`; omc = 1 - cos
   *        theta+. d^2sigma/(dE+ dcos) = -p+ p- K / (4 k^3).
   */
  template <class T>
  inline T dsigma_dE_dcos_T(T k, T Ep, T omc) {
    const T Em = k - Ep;
    if (!(Ep > T(1)) || !(Em > T(1)) || !(omc >= T(0)) || !(omc <= T(2))) {
      return T(0);
    }
    const T xp = Ep - T(1), xm = Em - T(1);
    const T pp = std::sqrt(xp * (xp + T(2))), pm = std::sqrt(xm * (xm + T(2)));
    T       K;
    switch (stable::angle_region(pp, pm)) {
      case stable::Region::Threshold:
        K = stable::series_K(pp, pm * pm, T(1) - omc);
        break;
      case stable::Region::Slow:
        K = stable::angle_K_slow(pp, pm, Em, omc);
        break;
      default:
        K = stable::angle_K_general(Ep, Em, pp, pm, omc, T(-1), T(1), T(1));
        break;
    }
    return -pp * pm * K / (T(4) * k * k * k);
  }

  /**
   * @brief The same in the round-1 form (long-double-free direct closed form
   *        with logarithms of numbers near 1), kept only as the reference that
   *        exhibits finding 1 in the tests
   */
  template <class T>
  inline T dsigma_dE_dcos_round1_T(T k, T Ep, T omc) {
    const T Em = k - Ep;
    if (!(Ep > T(1)) || !(Em > T(1)) || !(omc >= T(0)) || !(omc <= T(2))) {
      return T(0);
    }
    const T pp  = std::sqrt((Ep - T(1)) * (Ep + T(1)));
    const T pm  = std::sqrt((Em - T(1)) * (Em + T(1)));
    const T a   = T(1) / (Ep + pp);
    const T sp2 = omc * (T(2) - omc);
    const T D   = a + pp * omc;
    const T D2 = D * D, D4 = D2 * D2;
    const T pp2 = pp * pp;
    const T k2  = k * k;
    const T Ep2 = Ep * Ep, Em2 = Em * Em;
    const T Q2     = (Em + a) * (Em + a) + T(2) * k * pp * omc;
    const T Q      = std::sqrt(Q2);
    const T Q2mpm2 = T(1) + T(2) * a * Em + a * a + T(2) * k * pp * omc;
    const T L      = T(2) * std::log((Ep * Em + T(1) + pp * pm) / k);
    const T em     = T(2) * std::log(Em + pm);
    const T eQ     = T(2) * std::log(Q + pm) - std::log(Q2mpm2);
    const T t1     = T(8) * sp2 * (T(2) * Ep2 + T(1)) / (pp2 * D4);
    const T t2     = -T(2) * (T(5) * Ep2 - T(2) * Em * Ep + T(3)) / (pp2 * D2);
    const T t3     = -T(2) * (pp2 - k2) / (Q2 * D2);
    const T t4     = -T(4) * Em / (pp2 * D);
    const T t5     = (L / (pp * pm)) * (T(4) * Ep * sp2 * (T(3) * k + pp2 * Em) / (pp2 * D4) +
                                    T(4) * Ep2 * (Ep2 + Em2) / (pp2 * D2) +
                                    (T(2) - T(2) * (T(7) * Ep2 + T(3) * Em * Ep + Em2)) / (pp2 * D2) +
                                    T(2) * k * (Ep2 - Ep * Em - T(1)) / (pp2 * D));
    const T t6     = T(4) * em / (pm * D);
    const T t7     = (eQ / (pm * Q)) * (T(4) / D2 - T(6) * k / D - T(2) * k * (pp2 - k2) / (Q2 * D));
    return -pp * pm * (t1 + t2 + t3 + t4 - t5 + t6 + t7) / (T(4) * k2 * k);
  }

  /// law 3 in the type T, as double
  template <class T>
  inline double dsigma_dE_dcos_as(double k, double Ep, double omc) {
    return static_cast<double>(dsigma_dE_dcos_T<T>(static_cast<T>(k), static_cast<T>(Ep), static_cast<T>(omc)));
  }

  /// the production evaluation: double
  inline double dsigma_dE_dcos_closed(double k, double Ep, double omc) {
    return dsigma_dE_dcos_as<double>(k, Ep, omc);
  }

  /// h(E+, cos theta+ | k) = d^2 sigma/(dE+ dcos theta+) / sigma~(k), omc = 1 - cos theta+
  inline double h(double k, double Ep, double omc) {
    const double s = sigma_tilde(k);
    return s > 0.0 ? dsigma_dE_dcos_closed(k, Ep, omc) / s : 0.0;
  }

  // ------------------------------------------------------------------------ //
  // Quadratures over E+ and over cos theta+
  // ------------------------------------------------------------------------ //

  struct RestFrameQuadrature {
    int    gl_order      = 16;  // Gauss-Legendre order per panel
    int    energy_panels = 16;  // panels in s, E+ = 1 + (k - 2) sin^2(pi s/2)
    double h_t_angle     = 0.5; // panel width in t = ln(omc + w), w = (E+ - p+)/p+
    AngularQuadrature angular {};

    RestFrameQuadrature refined() const {
      RestFrameQuadrature q = *this;
      q.gl_order            = std::min(64, gl_order + 8);
      q.energy_panels      *= 2;
      q.h_t_angle          *= 0.5;
      q.angular             = angular.refined();
      return q;
    }
  };

  /**
   * @brief int_1^{k-1} f(E+) dE+ with E+ = 1 + (k - 2) sin^2(pi s/2), which
   *        removes the sqrt end-point behaviour of p+ p-
   */
  template <class F>
  inline double integrate_energy(double k, F&& f, const RestFrameQuadrature& q) {
    if (!(k > 2.0)) {
      return 0.0;
    }
    const GaussLegendre gl(q.gl_order);
    const double        d  = k - 2.0;
    const int           np = std::max(1, q.energy_panels);
    double              sum = 0.0;
    for (int p = 0; p < np; ++p) {
      const double s0 = static_cast<double>(p) / np, s1 = static_cast<double>(p + 1) / np;
      const double sm = 0.5 * (s0 + s1), sh = 0.5 * (s1 - s0);
      for (int i = 0; i < gl.order(); ++i) {
        const double s   = sm + sh * gl.x[i];
        const double sn  = std::sin(0.5 * constants::PI * s);
        const double E   = 1.0 + d * sn * sn;
        const double jac = d * 0.5 * constants::PI * std::sin(constants::PI * s);
        sum += sh * gl.w[i] * jac * f(E);
      }
    }
    return sum;
  }

  /// the forward-peak scale of the law 3 in omc = 1 - cos theta+: (E+ - p+)/p+
  inline double angular_scale(double Ep) {
    const double pp = momentum(Ep);
    return pp > 0.0 ? energy_minus_momentum(Ep, pp) / pp : 2.0;
  }

  /**
   * @brief int_0^2 f(omc) d(omc) (omc = 1 - cos theta+) with t = ln(omc + w),
   *        w = (E+ - p+)/p+, panels of width <= h_t_angle
   */
  template <class F>
  inline double integrate_angle(double Ep, F&& f, const RestFrameQuadrature& q) {
    const double        w  = angular_scale(Ep);
    const double        t0 = std::log(w), t1 = std::log(2.0 + w);
    const GaussLegendre gl(q.gl_order);
    const int           np = std::max(1, static_cast<int>(std::ceil((t1 - t0) / q.h_t_angle)));
    double              sum = 0.0;
    for (int p = 0; p < np; ++p) {
      const double a  = t0 + (t1 - t0) * static_cast<double>(p) / np;
      const double b  = t0 + (t1 - t0) * static_cast<double>(p + 1) / np;
      const double tm = 0.5 * (a + b), th = 0.5 * (b - a);
      for (int i = 0; i < gl.order(); ++i) {
        const double t   = tm + th * gl.x[i];
        const double et  = std::exp(t);
        const double omc = std::min(2.0, std::max(0.0, et - w));
        sum += th * gl.w[i] * et * f(omc);
      }
    }
    return sum;
  }

  /// A1 integrand: int dsigma/dE+ dE+ (must equal sigma~ Maximon); the law in the type T
  template <class T = double>
  inline double sigma_from_energy_law(double k, const RestFrameQuadrature& q) {
    return integrate_energy(k, [k](double E) { return dsigma_dE_as<T>(k, E); }, q);
  }

  /// A2 integrand: int d^2 sigma/(dE+ dcos) dcos at fixed E+ (closed form, type T)
  template <class T = double>
  inline double dsigma_dE_from_angular_law(double k, double Ep, const RestFrameQuadrature& q) {
    return integrate_angle(Ep, [k, Ep](double omc) { return dsigma_dE_dcos_as<T>(k, Ep, omc); }, q);
  }

  /// the same with the numerical integral of law 2 (slow; verification only)
  inline double dsigma_dE_from_angular_law_numeric(double k, double Ep, const RestFrameQuadrature& q) {
    return integrate_angle(
      Ep,
      [k, Ep, &q](double omc) { return dsigma_dE_dcos_numeric(k, Ep, omc, q.angular); },
      q);
  }

  // ------------------------------------------------------------------------ //
  // Moments and the recoil
  // ------------------------------------------------------------------------ //

  struct Moments {
    double k { 0.0 };
    double sigma_g { 0.0 };       // int dsigma/dE+ dE+ (A1 value)
    double sigma_maximon { 0.0 }; // sigma~(k)
    double sigma_h { 0.0 };       // int int d^2 sigma dcos dE+ (closed form)
    double mean_E { 0.0 };        // <E+> (= <E-> = k/2 by symmetry)
    double mean_pcos { 0.0 };     // <p+ cos theta+>_h
    double q_k { 0.0 };           // k - 2 <p cos theta> = 2 <(E - p) + p (1 - cos theta)>
  };

  /**
   * @brief The rest-frame moments at photon energy k from the laws 1 and 3
   *        evaluated in the type T (contract 1.3: q_k = k - 2 <p cos>)
   */
  template <class T = double>
  inline Moments moments(double k, const RestFrameQuadrature& q) {
    Moments m;
    m.k             = k;
    m.sigma_maximon = sigma_tilde(k);
    if (!(k > 2.0) || !(m.sigma_maximon > 0.0)) {
      m.q_k = 2.0;
      return m;
    }
    // the moments are normalised by the laws' own integrals (the Born cross
    // section), not by Maximon's series, whose truncation (1.5e-4 at k = 4)
    // would otherwise enter them; sigma_g and sigma_h are reported alongside
    m.sigma_g = integrate_energy(k, [k](double E) { return dsigma_dE_as<T>(k, E); }, q);
    m.mean_E  = integrate_energy(k, [k](double E) { return E * dsigma_dE_as<T>(k, E); }, q) /
               m.sigma_g;
    // <p+ cos theta+> and the angular-law total from the law 3
    m.sigma_h = integrate_energy(
      k,
      [&](double E) { return dsigma_dE_from_angular_law<T>(k, E, q); },
      q);
    m.mean_pcos = integrate_energy(
                    k,
                    [&](double E) {
                      const double pp = momentum(E);
                      return pp * integrate_angle(
                                    E,
                                    [&](double omc) {
                                      return dsigma_dE_dcos_as<T>(k, E, omc) * (1.0 - omc);
                                    },
                                    q);
                    },
                    q) /
                  m.sigma_h;
    // q_k = k - 2 <p cos theta> written without cancellation (round 2): with
    // k - 2 p cos = (k - 2E) + 2 (E - p) + 2 p (1 - cos) and <k - 2E> = 0
    // exactly by the e-/e+ symmetry of the energy law (verified by A2 and the
    // symmetry checks), q_k = 2 <(E - p) + p (1 - cos theta)>, a mean of
    // positive terms; the direct difference loses ~1e-9 relative at k ~ 1000
    // in double (q_k ~ 0.04 from 2 <p cos> ~ 1000)
    m.q_k = 2.0 *
            integrate_energy(
              k,
              [&](double E) {
                const double pp  = momentum(E);
                const double emp = energy_minus_momentum(E, pp);
                return integrate_angle(
                  E,
                  [&](double omc) { return dsigma_dE_dcos_as<T>(k, E, omc) * (emp + pp * omc); },
                  q);
              },
              q) /
            m.sigma_h;
    return m;
  }

  // ------------------------------------------------------------------------ //
  // The eps' grid of the fixture (contract section 4 item 1)
  // ------------------------------------------------------------------------ //

  /**
   * @brief 64 nodes per decade in log10(k - 2) from k - 2 = 1e-4 while
   *        k <= 1000, then k = 1000 exactly (convention: 448 regular nodes,
   *        delta_i = 10^(-4 + i/64), i = 0..447, plus the kernel cap)
   */
  inline std::vector<double> qk_grid() {
    std::vector<double> k;
    for (int i = 0;; ++i) {
      const double delta = std::pow(10.0, -4.0 + static_cast<double>(i) / 64.0);
      if (2.0 + delta > constants::K_MAX) {
        break;
      }
      k.push_back(2.0 + delta);
    }
    k.push_back(constants::K_MAX);
    return k;
  }

  // ------------------------------------------------------------------------ //
  // A small host thread pool for the per-node work
  // ------------------------------------------------------------------------ //

  template <class F>
  inline void parallel_for_index(std::size_t n, F&& f, unsigned n_threads = 0) {
    if (n == 0) {
      return;
    }
    if (n_threads == 0) {
      n_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    n_threads = static_cast<unsigned>(std::min<std::size_t>(n_threads, n));
    if (n_threads <= 1) {
      for (std::size_t i = 0; i < n; ++i) {
        f(i);
      }
      return;
    }
    std::atomic<std::size_t>  next { 0 };
    std::vector<std::thread>  pool;
    std::vector<std::exception_ptr> errors(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
      pool.emplace_back([&, t]() {
        try {
          for (;;) {
            const std::size_t i = next.fetch_add(1);
            if (i >= n) {
              break;
            }
            f(i);
          }
        } catch (...) {
          errors[t] = std::current_exception();
        }
      });
    }
    for (auto& th : pool) {
      th.join();
    }
    for (auto& e : errors) {
      if (e) {
        std::rethrow_exception(e);
      }
    }
  }

} // namespace arch::bh::rf

#endif // ARCHETYPES_HADRONIC_BH_RESTFRAME_H
