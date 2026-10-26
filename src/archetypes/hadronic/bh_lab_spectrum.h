/**
 * @file archetypes/hadronic/bh_lab_spectrum.h
 * @brief The lab-frame electron spectrum of Bethe-Heitler pairs produced by an
 *        ultra-relativistic ion in a monoenergetic isotropic photon field, by
 *        direct quadrature of the rest-frame laws and the exact boost (the
 *        anchor A8 of the step-4 contract, Kelner and Aharonian 2008 eqs. (58)
 *        to (64))
 *
 * Kelner and Aharonian 2008, PRD 78, 034013, section IV, obtain the electron
 * spectrum of the Bethe-Heitler process from the energy-angle distribution of
 * one lepton in the proton rest frame (their eq. (59), the
 * Sauter-Gluckstern-Hull law as written by Blumenthal 1970 eq. (11): the same
 * law as `dsigma_dE_dcos_closed` of bh_restframe.h, whose sign pattern,
 * -4 sin^2 theta (2E^2 + 1)/(p^2 Delta^4) + (5E^2 - 2E E_other + 3)/(p^2 Delta^2)
 * + (p^2 - k^2)/(T^2 Delta^2) + 2 E_other/(p^2 Delta) + Y(...) - delta^T(...)
 * - 2 y/Delta, is the one that the numerical integral of the fully
 * differential law selected in stage 1), the photon energy in the proton rest
 * frame omega = Gamma eps (1 - beta cos theta) for a photon at angle theta to
 * the ion velocity, and the Lorentz transformation of the lepton energy to the
 * lab, E = Gamma (E' + beta p' cos theta'_u), theta'_u the lepton's rest-frame
 * angle to the ion velocity; the spectrum is the integral over the photon
 * angle (equivalently over omega), the lepton rest-frame energy and the lepton
 * angles with the delta function of the energy transformation (their eqs.
 * (62) to (64) with the integration limits). The literature PDF was not
 * available in the build container: the equation numbers are quoted from the
 * brief and the structure from memory; this file implements the same chain
 * without the ultra-relativistic approximations of the paper (exact beta,
 * exact p' = sqrt(E'^2 - 1)), as the cumulative distribution of the lab
 * electron energy:
 *
 *   F(E) = (1/Phi(eps'_hi)) int_2^{eps'_hi} deps' eps' int dE' int dcos theta'
 *          d^2sigma/(dE' dcos theta') (eps', E', cos theta') P_phi(E_lab <= E),
 *
 * with P_phi the fraction of the uniform azimuth phi about the photon
 * direction for which Gamma (E' + beta p' (a - b cos phi)) <= E, where
 * a = cos theta' cos theta'_k, b = sin theta' sin theta'_k and cos theta'_k =
 * (eps/(Gamma eps') - 1)/beta is the photon's rest-frame direction relative
 * to the ion velocity: P_phi = arccos(clamp((a - c)/b))/pi with c =
 * (E/Gamma - E')/(beta p'). (eps' sigma~ is the eps' marginal of the
 * unweighted chain, g and h its conditionals: eps' g h = eps' d^2sigma/sigma~
 * ... times sigma~, so the weight is eps' d^2sigma and the normalisation is
 * Phi(eps'_hi) = int eps' sigma~ deps'.) Both the sampler and this quadrature
 * use the static-source Born law; the comparison tests the sampling chain, not
 * the regime of validity of the static-source model (see the report for the
 * applicability condition eps' Gamma << mu_r at the two anchor points).
 *
 * Host-only, double precision, Kokkos-free.
 *
 * @namespaces:
 *   - arch::bh::rf::
 */

#ifndef ARCHETYPES_HADRONIC_BH_LAB_SPECTRUM_H
#define ARCHETYPES_HADRONIC_BH_LAB_SPECTRUM_H

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_restframe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace arch::bh::rf {

  struct LabSpectrumQuadrature {
    int    gl_order     = 10;   // Gauss-Legendre order per panel
    double h_t_k        = 0.125; // panel width in t = ln(eps' - 2)
    int    s_panels     = 24;   // panels in s (lepton energy)
    double h_t_angle    = 0.5;  // panel width in t = ln(1 - cos theta' + w)
    double delta_min    = 1.0e-6; // lower end of the eps' range, eps' - 2

    LabSpectrumQuadrature refined() const {
      LabSpectrumQuadrature q = *this;
      q.gl_order              = std::min(64, gl_order + 4);
      q.h_t_k                *= 0.5;
      q.s_panels             *= 2;
      q.h_t_angle            *= 0.5;
      return q;
    }
  };

  /**
   * @brief The cumulative distribution of the lab electron energy at the given
   *        edges (increasing), for an ion of Lorentz factor gamma in a mono
   *        bath of photon energy eps0; also the mean lab electron energy.
   * @param edges increasing lab energies (m_e c^2)
   * @param mean_out if given, receives <E_lab>
   * @return F(edges[i])
   */
  inline std::vector<double> lab_electron_cdf(double                        gamma,
                                              double                        eps0,
                                              const std::vector<double>&    edges,
                                              const LabSpectrumQuadrature&  q,
                                              double*                       mean_out = nullptr) {
    if (!(gamma > 1.0) || !(eps0 > 0.0)) {
      throw std::invalid_argument("lab_electron_cdf: gamma > 1 and eps0 > 0 required");
    }
    for (std::size_t i = 1; i < edges.size(); ++i) {
      if (!(edges[i] > edges[i - 1])) {
        throw std::invalid_argument("lab_electron_cdf: edges must increase");
      }
    }
    const double gb   = std::sqrt((gamma - 1.0) * (gamma + 1.0));
    const double beta = gb / gamma;
    const double A    = gamma + gb;
    const double khi  = eps0 * A; // eps'_hi
    if (!(khi > 2.0 + q.delta_min)) {
      throw std::invalid_argument("lab_electron_cdf: below threshold");
    }
    if (khi > constants::K_MAX * (1.0 + 1.0e-12)) {
      throw std::invalid_argument("lab_electron_cdf: eps'_hi above the kernel cap 1000");
    }
    const GaussLegendre G(q.gl_order);
    std::vector<double> F(edges.size(), 0.0);
    double              norm = 0.0, mean = 0.0;
    // eps' in t = ln(eps' - 2)
    const double tk0 = std::log(q.delta_min), tk1 = std::log(khi - 2.0);
    const int    nk  = std::max(1, static_cast<int>(std::ceil((tk1 - tk0) / q.h_t_k)));
    for (int ik = 0; ik < nk; ++ik) {
      const double a0 = tk0 + (tk1 - tk0) * ik / nk, a1 = tk0 + (tk1 - tk0) * (ik + 1) / nk;
      for (int gk = 0; gk < G.order(); ++gk) {
        const double tk    = 0.5 * (a0 + a1) + 0.5 * (a1 - a0) * G.x[gk];
        const double delta = std::exp(tk);
        const double k     = 2.0 + delta;
        const double wk    = 0.5 * (a1 - a0) * G.w[gk] * delta * k; // deps' = delta dt, times eps'
        // the photon's rest-frame direction relative to u^
        double cos_k = (eps0 / (gamma * k) - 1.0) / beta;
        cos_k        = std::min(1.0, std::max(-1.0, cos_k));
        const double sin_k = std::sqrt(std::max(0.0, (1.0 - cos_k) * (1.0 + cos_k)));
        // lepton energy in s
        for (int js = 0; js < q.s_panels; ++js) {
          const double s0 = static_cast<double>(js) / q.s_panels, s1 = static_cast<double>(js + 1) / q.s_panels;
          for (int gs = 0; gs < G.order(); ++gs) {
            const double s   = 0.5 * (s0 + s1) + 0.5 * (s1 - s0) * G.x[gs];
            const double sn  = std::sin(0.5 * constants::PI * s);
            const double E   = 1.0 + (k - 2.0) * sn * sn;
            const double jac = (k - 2.0) * 0.5 * constants::PI * std::sin(constants::PI * s);
            const double ws  = 0.5 * (s1 - s0) * G.w[gs] * jac;
            const double p   = momentum(E);
            if (!(p > 0.0)) {
              continue;
            }
            // lepton angle to k^' in t = ln(omc + w)
            const double w  = angular_scale(E);
            const double t0 = std::log(w), t1 = std::log(2.0 + w);
            const int    na = std::max(1, static_cast<int>(std::ceil((t1 - t0) / q.h_t_angle)));
            for (int ja = 0; ja < na; ++ja) {
              const double b0 = t0 + (t1 - t0) * ja / na, b1 = t0 + (t1 - t0) * (ja + 1) / na;
              for (int ga = 0; ga < G.order(); ++ga) {
                const double t   = 0.5 * (b0 + b1) + 0.5 * (b1 - b0) * G.x[ga];
                const double et  = std::exp(t);
                const double omc = std::min(2.0, std::max(0.0, et - w));
                const double wa  = 0.5 * (b1 - b0) * G.w[ga] * et;
                const double law = dsigma_dE_dcos_closed(k, E, omc);
                if (!(law > 0.0)) {
                  continue;
                }
                const double wt  = wk * ws * wa * law;
                const double ct  = 1.0 - omc, st = std::sqrt(std::max(0.0, omc * (2.0 - omc)));
                const double a   = ct * cos_k, b = st * sin_k;
                // E_lab(phi) = gamma (E + beta p (a - b cos phi)): range [lo, hi], mean gamma (E + beta p a)
                const double Emid = gamma * (E + beta * p * a);
                const double half = gamma * beta * p * b;
                const double lo = Emid - half, hi = Emid + half;
                norm += wt;
                mean += wt * Emid;
                // CDF contributions: P_phi(E_lab <= edge) = arccos((a - c)/b)/pi, c = (edge/gamma - E)/(beta p)
                // edges below lo contribute 0, above hi contribute 1
                auto it_lo = std::lower_bound(edges.begin(), edges.end(), lo);
                auto it_hi = std::lower_bound(edges.begin(), edges.end(), hi);
                for (auto it = it_lo; it != it_hi; ++it) {
                  double x = half > 0.0 ? (Emid - *it) / half : 0.0; // = (a - c)/b
                  x        = std::min(1.0, std::max(-1.0, x));
                  F[static_cast<std::size_t>(it - edges.begin())] += wt * std::acos(x) / constants::PI;
                }
                for (auto it = it_hi; it != edges.end(); ++it) {
                  F[static_cast<std::size_t>(it - edges.begin())] += wt;
                }
              }
            }
          }
        }
      }
    }
    if (!(norm > 0.0)) {
      throw std::runtime_error("lab_electron_cdf: zero normalisation");
    }
    for (double& f : F) {
      f = std::min(1.0, f / norm);
    }
    if (mean_out) {
      *mean_out = mean / norm;
    }
    return F;
  }

  /// the lab energy range of the leptons for a mono bath: [1, Gamma (E'_max + beta p'_max)]
  inline void lab_energy_range(double gamma, double eps0, double& E_lo, double& E_hi) {
    const double gb  = std::sqrt((gamma - 1.0) * (gamma + 1.0));
    const double khi = eps0 * (gamma + gb);
    const double Em  = khi - 1.0;
    E_lo             = 1.0;
    E_hi             = gamma * Em + gb * momentum(Em);
  }

} // namespace arch::bh::rf

#endif // ARCHETYPES_HADRONIC_BH_LAB_SPECTRUM_H
