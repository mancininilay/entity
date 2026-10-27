/**
 * @file archetypes/hadronic/bh_exact_chain_cdf.h
 * @brief The exact-chain self-check of the rest-frame sampler (round 2,
 *        finding 2; round 1's `bh_lab_spectrum.h`): the CDF of the lab
 *        electron energy of the sampler's own chain, for an ion of Lorentz
 *        factor Gamma in a monoenergetic isotropic photon field, by direct
 *        quadrature of the rest-frame laws with the exact boost and the exact
 *        photon direction. It is an internal test of the sampling chain, NOT
 *        the external anchor A8: the paper's CDF (Kelner and Aharonian 2008
 *        eqs. (58) to (64)) uses the head-on approximation, and A8 compares
 *        with the reference's transcription of it (`ka2008_*.txt` of the bhref
 *        fixtures). The head-on map is available here too (`LabMap::HeadOn`)
 *        for a deterministic comparison with that transcription.
 *
 * The chain: the photon energy in the proton rest frame eps' = Gamma eps (1 -
 * beta cos theta) for a photon at angle theta to the ion velocity (eps'
 * sigma~ is the eps' marginal of the unweighted chain), the single-lepton
 * energy-angle law (MOK 1969 3D-2000 = Blumenthal 1970 eq. (10), definitions
 * in his eq. (11); `dsigma_dE_dcos_closed` of bh_restframe.h, signs and
 * normalisation verified against the PDFs by the referee, Codex 1,
 * 2026-10-27), and the Lorentz transformation of the lepton energy to the lab:
 *
 *   LabMap::Exact:  E = Gamma (E' + beta p' cos theta'_u), theta'_u the lepton's
 *                   rest-frame angle to the ion velocity, with the photon's
 *                   rest-frame direction cos theta'_k = (eps/(Gamma eps') - 1)/beta
 *                   and a uniform azimuth about it (exact beta, exact p');
 *   LabMap::HeadOn: E = Gamma (E' - p' cos theta'_gamma), theta'_gamma the
 *                   lepton's angle to the photon direction, the photon taken
 *                   head-on (the paper's approximation; E < 1 is possible and
 *                   is counted below the first edge).
 *
 *   F(E) = (1/Phi(eps'_hi)) int_2^{eps'_hi} deps' eps' int dE' int dcos theta'
 *          d^2sigma/(dE' dcos theta') (eps', E', cos theta') P(E_lab <= E),
 *
 * with, for the exact map, P the fraction of the uniform azimuth phi about the
 * photon direction for which Gamma (E' + beta p' (a - b cos phi)) <= E,
 * a = cos theta' cos theta'_k, b = sin theta' sin theta'_k: P =
 * arccos(clamp((a - c)/b))/pi with c = (E/Gamma - E')/(beta p'). The mean
 * difference of the two maps is eps <p' cos theta'_gamma / eps'>, which tends
 * to eps/2 when eps' >> 1 (the mechanism of the failed reading of round 1).
 * Both the sampler and this quadrature use the static-source Born law; the
 * applicability of that model to a finite ion mass is eps' << mu_r,
 * equivalently eps Gamma << mu_r (corrected in round 2).
 *
 * Host-only, double precision, Kokkos-free.
 *
 * @namespaces:
 *   - arch::bh::rf::
 */

#ifndef ARCHETYPES_HADRONIC_BH_EXACT_CHAIN_CDF_H
#define ARCHETYPES_HADRONIC_BH_EXACT_CHAIN_CDF_H

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_restframe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace arch::bh::rf {

  struct ExactChainQuadrature {
    int    gl_order     = 10;   // Gauss-Legendre order per panel
    double h_t_k        = 0.125; // panel width in t = ln(eps' - 2)
    int    s_panels     = 24;   // panels in s (lepton energy)
    double h_t_angle    = 0.5;  // panel width in t = ln(1 - cos theta' + w)
    double delta_min    = 1.0e-6; // lower end of the eps' range, eps' - 2

    ExactChainQuadrature refined() const {
      ExactChainQuadrature q = *this;
      q.gl_order              = std::min(64, gl_order + 4);
      q.h_t_k                *= 0.5;
      q.s_panels             *= 2;
      q.h_t_angle            *= 0.5;
      return q;
    }
  };

  enum class LabMap {
    Exact,
    HeadOn
  };

  /**
   * @brief The cumulative distribution of the lab electron energy at the given
   *        edges (increasing), for an ion of Lorentz factor gamma in a mono
   *        bath of photon energy eps0, with the exact or the head-on map;
   *        also the mean lab electron energy.
   * @param edges increasing lab energies (m_e c^2)
   * @param mean_out if given, receives <E_lab>
   * @return F(edges[i])
   */
  inline std::vector<double> exact_chain_lab_cdf(double                      gamma,
                                                 double                      eps0,
                                                 const std::vector<double>&  edges,
                                                 const ExactChainQuadrature& q,
                                                 double*                     mean_out = nullptr,
                                                 LabMap                      map      = LabMap::Exact) {
    if (!(gamma > 1.0) || !(eps0 > 0.0)) {
      throw std::invalid_argument("exact_chain_lab_cdf: gamma > 1 and eps0 > 0 required");
    }
    for (std::size_t i = 1; i < edges.size(); ++i) {
      if (!(edges[i] > edges[i - 1])) {
        throw std::invalid_argument("exact_chain_lab_cdf: edges must increase");
      }
    }
    const double gb   = std::sqrt((gamma - 1.0) * (gamma + 1.0));
    const double beta = gb / gamma;
    const double A    = gamma + gb;
    const double khi  = eps0 * A; // eps'_hi
    if (!(khi > 2.0 + q.delta_min)) {
      throw std::invalid_argument("exact_chain_lab_cdf: below threshold");
    }
    if (khi > constants::K_MAX * (1.0 + 1.0e-12)) {
      throw std::invalid_argument("exact_chain_lab_cdf: eps'_hi above the kernel cap 1000");
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
                if (map == LabMap::HeadOn) {
                  const double Eho = gamma * (E - p * ct);
                  norm += wt;
                  mean += wt * Eho;
                  for (auto it = std::lower_bound(edges.begin(), edges.end(), Eho); it != edges.end(); ++it) {
                    F[static_cast<std::size_t>(it - edges.begin())] += wt;
                  }
                  continue;
                }
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
      throw std::runtime_error("exact_chain_lab_cdf: zero normalisation");
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

#endif // ARCHETYPES_HADRONIC_BH_EXACT_CHAIN_CDF_H
