/**
 * @file archetypes/hadronic/bh_restframe.h
 * @brief Bethe-Heitler pair production on a static point source (Z = 1,
 *        unscreened, Born approximation): the rest-frame laws of the pair
 *        sampler (P2 step 4, contract revision 1.0 section 1, spec III.6)
 *
 * Host-only code, double precision (long double inside the energy law, where
 * the Born formula cancels by many orders near threshold). Nothing here
 * depends on Kokkos, the problem generator or the input parser.
 *
 * Units: energies in m_e c^2, momenta in m_e c, cross sections in alpha r_e^2.
 * Primed quantities of the contract (proton rest frame) are unprimed here:
 * k is the photon energy (eps'), E+ and E- the lepton total energies
 * (E+ + E- = k, no recoil energy), p = sqrt(E^2 - 1), theta the angle of a
 * lepton to the photon direction.
 *
 * Laws implemented (formula numbers as stated in the brief; the literature
 * PDFs were not available in the build container, every formula below is
 * transcribed from the author's knowledge of the papers and verified by the
 * numerical anchors A1 to A5 of the test suite, see the step-4 report):
 *
 *   1. dsigma/dE+ (the energy law): Bethe-Heitler Born cross section
 *      integrated over both lepton directions, as written by Blumenthal
 *      1970, PRD 1, 1596, eq. (10) (= Heitler 1954 section 26 eq. (14) =
 *      Motz, Olsen and Koch 1969 formula 3D-1000 with Z = 1, no screening).
 *      Its integral over E+ reproduces Maximon's sigma~(k) (anchor A1).
 *      g(E+ | k) = (1/sigma~) dsigma/dE+.
 *
 *   2. d^5 sigma/(dE+ dOmega+ dOmega-) (the fully differential law): Bethe
 *      and Heitler 1934, as written by Heitler 1954 section 26 eq. (6)
 *      (= MOK 1969 formula 3D-3000, Z = 1, no screening). It is the parent of
 *      both laws above and below; here it is integrated numerically over the
 *      direction of the second lepton (`dsigma_dE_dcos_numeric`) on the
 *      sphere of the momentum transfer, which makes the 1/q^4 structure
 *      analytic (log grid in q^2) and leaves the forward peak of the second
 *      lepton (1/D-^2, D- = E- - p- cos theta-) as a located bump handled by
 *      graded panels.
 *
 *   3. d^2 sigma/(dE+ dcos theta+) (the single-lepton energy-angle law): the
 *      Sauter-Gluckstern-Hull expression, MOK 1969 formula 3D-2000 (Z = 1,
 *      no screening), i.e. the law 2 integrated over the other lepton's
 *      direction in closed form (`dsigma_dE_dcos_closed`). It is the pair
 *      analogue of the bremsstrahlung formula 2BN of Koch and Motz 1959,
 *      obtained from it by the substitution E0 -> -E+, E -> E-, k -> -k; the
 *      signs of the three logarithmic groups and the overall factor, which the
 *      substitution leaves ambiguous when the PDF is not at hand, were fixed
 *      by requiring agreement with the numerical integral of law 2 (the test
 *      suite re-verifies this agreement at every run: it is the A2-level check
 *      of the transcription). h(E+, cos theta+ | k) = (1/sigma~) d^2 sigma.
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
  // Law 1: the energy distribution, Blumenthal 1970 eq. (10)
  // ------------------------------------------------------------------------ //

  /**
   * @brief dsigma/dE+ in alpha r_e^2 per m_e c^2 (Blumenthal 1970 eq. 10).
   *        Zero outside 1 < E+ < k - 1.
   *
   * Evaluated in long double: the terms of the bracket are O(1/p^4) near
   * threshold and cancel to the O((k - 2)^3) total.
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
    const T pp  = std::sqrt((Ep - T(1)) * (Ep + T(1)));
    const T pm  = std::sqrt((Em - T(1)) * (Em + T(1)));
    const T ep  = T(2) * std::log(Ep + pp); // ln((E+ + p+)/(E+ - p+))
    const T em  = T(2) * std::log(Em + pm);
    const T L   = T(2) * std::log((Ep * Em + pp * pm + T(1)) / k);
    const T pp2 = pp * pp, pm2 = pm * pm;
    const T pp3 = pp2 * pp, pm3 = pm2 * pm;
    const T EE  = Ep * Em;
    const T br  = -T(4) / T(3) - T(2) * EE * (pp2 + pm2) / (pp2 * pm2) +
                 ep * Em / pp3 + em * Ep / pm3 - ep * em / (pp * pm) +
                 L * (k * k * (EE * EE + pp2 * pm2) / (pp3 * pm3) -
                      (T(8) / T(3)) * EE / (pp * pm) -
                      (k / (T(2) * pp * pm)) *
                        ((EE - pp2) * ep / pp3 + (EE - pm2) * em / pm3 +
                         T(2) * k * EE / (pp2 * pm2)));
    return pp * pm * br / (k * k * k);
  }

  inline double dsigma_dE(double k, double Ep) {
    return static_cast<double>(
      dsigma_dE_T<long double>(static_cast<long double>(k),
                               static_cast<long double>(Ep)));
  }

  /// g(E+ | k) = dsigma/dE+ / sigma~(k) (Maximon's total)
  inline double g(double k, double Ep) {
    const double s = sigma_tilde(k);
    return s > 0.0 ? dsigma_dE(k, Ep) / s : 0.0;
  }

  // ------------------------------------------------------------------------ //
  // Law 2: the fully differential Bethe-Heitler law (Heitler 1954, 26.6) and
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
   * with F the bracket of Heitler 26.6; the normalisation 1/(4 pi^2) is the one
   * for which the angular integral reproduces the energy law 1 (checked by A2;
   * a 1/(2 pi) prefactor gives 2 pi times the energy law, fact of the scratch
   * run of 2026-10-26). The integrand F/q^4 (Heitler 26.6) becomes F/q^2 in u = ln q^2; the
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
  // Law 3: the Sauter-Gluckstern-Hull closed form (MOK 1969, 3D-2000)
  // ------------------------------------------------------------------------ //

  /**
   * @brief Signs of the logarithmic groups and the overall factor of the
   *        closed form (see the file header): fixed against the numerical
   *        integral of law 2.
   */
  struct ClosedFormSigns {
    double s0 = 1.0; // overall
    double sL = 1.0; // the L group
    double se = 1.0; // the eps- group
    double sQ = 1.0; // the eps^Q group
  };

  /**
   * @brief d^2 sigma/(dE+ dcos theta+) in alpha r_e^2, closed form, with
   *        explicit signs (used by the verification against law 2)
   */
  template <class T>
  inline T dsigma_dE_dcos_closed_signed_T(T k, T Ep, T omc, const ClosedFormSigns& s) {
    const T Em = k - Ep;
    if (!(Ep > T(1)) || !(Em > T(1)) || !(omc >= T(0)) || !(omc <= T(2))) {
      return T(0);
    }
    const T pp  = std::sqrt((Ep - T(1)) * (Ep + T(1)));
    const T pm  = std::sqrt((Em - T(1)) * (Em + T(1)));
    const T a   = T(1) / (Ep + pp); // E+ - p+
    const T sp2 = omc * (T(2) - omc);
    const T D   = a + pp * omc; // Delta+ = E+ - p+ cos theta+
    const T D2  = D * D, D4 = D2 * D2;
    const T pp2 = pp * pp, pm2 = pm * pm;
    const T k2  = k * k;
    const T Ep2 = Ep * Ep, Em2 = Em * Em;
    // Q^2 = |k z^ - p+|^2 = (E- + a)^2 + 2 k p+ omc
    const T Q2     = (Em + a) * (Em + a) + T(2) * k * pp * omc;
    const T Q      = std::sqrt(Q2);
    const T Q2mpm2 = T(1) + T(2) * a * Em + a * a + T(2) * k * pp * omc; // Q^2 - p-^2
    // logarithms
    const T L  = T(2) * std::log((Ep * Em + T(1) + pp * pm) / k);
    const T em = T(2) * std::log(Em + pm);
    const T eQ = T(2) * std::log(Q + pm) - std::log(Q2mpm2); // ln((Q+p-)/(Q-p-))

    const T t1 = T(8) * sp2 * (T(2) * Ep2 + T(1)) / (pp2 * D4);
    const T t2 = -T(2) * (T(5) * Ep2 - T(2) * Em * Ep + T(3)) / (pp2 * D2);
    const T t3 = -T(2) * (pp2 - k2) / (Q2 * D2);
    const T t4 = -T(4) * Em / (pp2 * D);
    const T t5 = (L / (pp * pm)) *
                 (T(4) * Ep * sp2 * (T(3) * k + pp2 * Em) / (pp2 * D4) +
                  T(4) * Ep2 * (Ep2 + Em2) / (pp2 * D2) +
                  (T(2) - T(2) * (T(7) * Ep2 + T(3) * Em * Ep + Em2)) / (pp2 * D2) +
                  T(2) * k * (Ep2 - Ep * Em - T(1)) / (pp2 * D));
    const T t6 = T(4) * em / (pm * D);
    const T t7 = (eQ / (pm * Q)) *
                 (T(4) / D2 - T(6) * k / D - T(2) * k * (pp2 - k2) / (Q2 * D));
    const T bracket = t1 + t2 + t3 + t4 + T(s.sL) * t5 + T(s.se) * t6 + T(s.sQ) * t7;
    // d^2 sigma/(dE+ dOmega+) = s0 (p+ p-/(8 pi k^3)) bracket;  x 2 pi for dcos theta+
    return T(s.s0) * pp * pm * bracket / (T(4) * k2 * k);
  }

  /// evaluated in long double (the bracket cancels by ~1/p+^2 as p+ -> 0)
  inline double dsigma_dE_dcos_closed_signed(double                 k,
                                             double                 Ep,
                                             double                 omc,
                                             const ClosedFormSigns& s) {
    return static_cast<double>(
      dsigma_dE_dcos_closed_signed_T<long double>(static_cast<long double>(k),
                                                  static_cast<long double>(Ep),
                                                  static_cast<long double>(omc),
                                                  s));
  }

  /**
   * @brief The signs fixed by the comparison with the numerical integral of
   *        law 2 (fact, scratch run of 2026-10-26 at 12 kinematic points over
   *        k in [2.5, 1000], E+ from 1.2 to 900, 1 - cos theta+ from 1e-6 to
   *        1.7): with s0 = -1, sL = -1, se = +1, sQ = +1 the ratio closed/numeric
   *        is the same constant at every point to 6 digits, and the constant
   *        fixes the normalisation to d^2 sigma/(dE+ dOmega+) =
   *        -(p+ p-/(8 pi k^3)) bracket (the prefactor of Koch and Motz 2BN); every other combination varies by factors
   *        of order one between the points. The test suite re-checks the
   *        agreement at every run.
   */
  inline const ClosedFormSigns& closed_form_signs() {
    static const ClosedFormSigns s { -1.0, -1.0, 1.0, 1.0 };
    return s;
  }

  inline double dsigma_dE_dcos_closed(double k, double Ep, double omc) {
    return dsigma_dE_dcos_closed_signed(k, Ep, omc, closed_form_signs());
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

  /// A1 integrand: int dsigma/dE+ dE+ (must equal sigma~ Maximon)
  inline double sigma_from_energy_law(double k, const RestFrameQuadrature& q) {
    return integrate_energy(k, [k](double E) { return dsigma_dE(k, E); }, q);
  }

  /// A2 integrand: int d^2 sigma/(dE+ dcos) dcos at fixed E+ (closed form)
  inline double dsigma_dE_from_angular_law(double k, double Ep, const RestFrameQuadrature& q) {
    return integrate_angle(Ep, [k, Ep](double omc) { return dsigma_dE_dcos_closed(k, Ep, omc); }, q);
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
    double q_k { 0.0 };           // k - 2 <p cos theta>
  };

  /**
   * @brief The rest-frame moments at photon energy k from the laws 1 and 3,
   *        normalised with Maximon's sigma~ (contract 1.3: q_k = k - 2 <p cos>)
   */
  inline Moments moments(double k, const RestFrameQuadrature& q) {
    Moments m;
    m.k             = k;
    m.sigma_maximon = sigma_tilde(k);
    if (!(k > 2.0) || !(m.sigma_maximon > 0.0)) {
      m.q_k = 2.0;
      return m;
    }
    m.sigma_g = integrate_energy(k, [k](double E) { return dsigma_dE(k, E); }, q);
    m.mean_E  = integrate_energy(k, [k](double E) { return E * dsigma_dE(k, E); }, q) /
               m.sigma_maximon;
    // <p+ cos theta+> and the angular-law total from the law 3
    m.mean_pcos = integrate_energy(
                    k,
                    [&](double E) {
                      const double pp = momentum(E);
                      return pp * integrate_angle(
                                    E,
                                    [&](double omc) {
                                      return dsigma_dE_dcos_closed(k, E, omc) * (1.0 - omc);
                                    },
                                    q);
                    },
                    q) /
                  m.sigma_maximon;
    m.sigma_h = integrate_energy(
      k,
      [&](double E) { return dsigma_dE_from_angular_law(k, E, q); },
      q);
    m.q_k = k - 2.0 * m.mean_pcos;
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
