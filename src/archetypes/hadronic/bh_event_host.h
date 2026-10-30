/**
 * @file archetypes/hadronic/bh_event_host.h
 * @brief Host-side numbers of the event (Monte Carlo) loss mode, `mode =
 *        "event"` (P2 step 5; contract 2026-10-29 sections 2 and 3): the
 *        closed-form upper bound of the unweighted ion loss per interaction,
 *        the admissible macro-pair weight a_max, the maximal Poisson mean over
 *        the table and the ulp-energy ratio. Kokkos-free, templated on the
 *        floating type where the stored momentum's resolution enters, so that
 *        the host-only parser test can exercise both instantiations (`long
 *        double` is `double` on Apple arm64: never rely on it).
 *
 * Conventions (spec III.2, contract section 2):
 *   D = E- + E+ - eps (lab), the ion's loss per interaction (C15). With
 *   p'_z in [-sqrt(eps'^2 - 4), sqrt(eps'^2 - 4)] and eps' in [eps'_lo,
 *   eps'_hi], eps'_hi = Gamma eps (1 + beta):
 *     D_max(eps, eps') = Gamma eps' - eps + Gamma beta sqrt(eps'^2 - 4),
 *   increasing in eps' and in eps, so over the bath the bound is attained at
 *   eps = eps_max of the retained spectrum and eps' = eps'_hi(eps_max):
 *     D_max,unweighted(Gamma) = Gamma eps'_hi - eps_max
 *                               + Gamma beta sqrt(eps'_hi^2 - 4).
 *   a_max(Gamma) = f_max K / D_max,unweighted(Gamma), K = mu_r (Gamma - 1):
 *   the largest macro-pair weight for which no event can violate the
 *   per-event bound a D <= f_max K at that Gamma (fact: D <= D_max for every
 *   outcome). Fact (parser test): for eps_max <= 1 a_max decreases
 *   monotonically with Gamma above the threshold (K ~ Gamma - 1, D_max ~
 *   Gamma^2), so an ion that cools only becomes safer; for a hard bath
 *   (eps_max = 1.9, Gamma_th = 1.0013) a_max first rises from the threshold
 *   to a maximum near Gamma = 1.33 and decreases beyond; it decreases for
 *   Gamma >= 2 in every case tested. The active-band refusal of the pgen
 *   therefore scans every table node up to the top of the band and assumes
 *   no monotonicity.
 *
 * @namespaces:
 *   - arch::bh::event::
 */

#ifndef ARCHETYPES_HADRONIC_BH_EVENT_HOST_H
#define ARCHETYPES_HADRONIC_BH_EVENT_HOST_H

#include "archetypes/hadronic/bh_physics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace arch::bh::event {

  /**
   * @brief Gamma beta from Gamma (no cancellation for Gamma close to 1)
   */
  inline auto gamma_beta(double Gamma) -> double {
    return std::sqrt((Gamma - 1.0) * (Gamma + 1.0));
  }

  /**
   * @brief D_max(eps, eps') of spec III.2 at a given (eps, eps'); 0 below the
   *        pair threshold eps' < 2 (no interaction)
   */
  inline auto dmax_at(double Gamma, double eps, double eps_prime) -> double {
    if (!(eps_prime >= 2.0)) {
      return 0.0;
    }
    const double gb = gamma_beta(Gamma);
    return Gamma * eps_prime - eps + gb * std::sqrt((eps_prime - 2.0) * (eps_prime + 2.0));
  }

  /**
   * @brief The upper edge eps'_hi = Gamma eps (1 + beta) of the proton-frame
   *        photon energy for a lab photon of energy eps
   */
  inline auto eps_prime_hi(double Gamma, double eps) -> double {
    return Gamma * eps + gamma_beta(Gamma) * eps;
  }

  /**
   * @brief D_max,unweighted(Gamma): the bound over every outcome of the
   *        unweighted law for a bath whose retained spectrum ends at eps_max
   *        (eps0 for a mono bath). 0 when eps'_hi(eps_max) < 2 (below the
   *        threshold of the hardest photon: no interaction).
   */
  inline auto dmax_unweighted(double Gamma, double eps_max) -> double {
    return dmax_at(Gamma, eps_max, eps_prime_hi(Gamma, eps_max));
  }

  /**
   * @brief a_max(Gamma) = f_max K / D_max,unweighted(Gamma) with K = mu_r
   *        (Gamma - 1); +inf when D_max = 0 (no interaction possible: no
   *        bound can be violated)
   */
  inline auto a_max(double Gamma, double eps_max, double mu_r, double f_max) -> double {
    const double D = dmax_unweighted(Gamma, eps_max);
    if (!(D > 0.0)) {
      return std::numeric_limits<double>::infinity();
    }
    return f_max * mu_r * (Gamma - 1.0) / D;
  }

  /**
   * @brief Energy resolution of the stored momentum of an ion of mass mu_d
   *        at kinetic energy K: dK_ulp = ulp(u) mu_d u / gamma_d with u the
   *        magnitude of the stored (type R) four-velocity, ulp(u) =
   *        nextafter(u, +inf) - u in R. The same expression as the kernel's
   *        pending floor (kernels/hadronic/bethe_heitler.hpp, ulp_real), on
   *        the host.
   */
  template <class R>
  inline auto ulp_energy(double K, double mu_d) -> double {
    static_assert(std::numeric_limits<R>::is_iec559, "ulp_energy needs an IEEE floating type");
    if (!(K > 0.0) || !(mu_d > 0.0)) {
      return 0.0;
    }
    const double u  = std::sqrt(K * (K + 2.0 * mu_d)) / mu_d;
    const double gd = std::sqrt(1.0 + u * u);
    const R      ur = static_cast<R>(u);
    const R      up = std::nextafter(ur, std::numeric_limits<R>::infinity());
    const double ulp = static_cast<double>(up) - static_cast<double>(ur);
    return ulp * mu_d * u / gd;
  }

  /**
   * @brief Startup numbers over the table (contract section 3), computed on
   *        the host table of step 1 at its nodes: the maximal Poisson mean
   *        lambda_max = max_i coeff J_i dt / a and the node of the maximum;
   *        the minimal ratio of the smallest energy removed by one event, a
   *        <D>_i, to the energy resolution of the stored momentum of an ion
   *        of mass mu_d at that node, and its node. Nodes with J = 0 or <D> =
   *        0 (none inside the table by construction, but guarded) are skipped.
   */
  struct TableNumbers {
    double      lambda_max { 0.0 };
    double      gamma_at_lambda_max { 0.0 };
    double      ulp_ratio_min { std::numeric_limits<double>::infinity() };
    double      gamma_at_ulp_ratio_min { 0.0 };
    double      dK_ulp_at_min { 0.0 };
    double      dmean_at_min { 0.0 };
    std::size_t nodes { 0 };
  };

  template <class R>
  inline auto table_numbers(const HadronicTablesHost& T,
                            double                    coeff,
                            double                    dt,
                            double                    a,
                            double                    mu_d) -> TableNumbers {
    TableNumbers n;
    for (std::size_t i = 0; i < T.size(); ++i) {
      const double G = T.gamma[i];
      const double J = T.J[i];
      const double D = T.Dmean[i];
      if (!(G > 1.0)) {
        continue;
      }
      ++n.nodes;
      if (J > 0.0 && a > 0.0) {
        const double lam = coeff * J * dt / a;
        if (lam > n.lambda_max) {
          n.lambda_max          = lam;
          n.gamma_at_lambda_max = G;
        }
      }
      if (D > 0.0) {
        const double K   = T.mu_r * (G - 1.0);
        const double res = ulp_energy<R>(K, mu_d);
        if (res > 0.0) {
          const double ratio = a * D / res;
          if (ratio < n.ulp_ratio_min) {
            n.ulp_ratio_min          = ratio;
            n.gamma_at_ulp_ratio_min = G;
            n.dK_ulp_at_min          = res;
            n.dmean_at_min           = D;
          }
        }
      }
    }
    return n;
  }

  /**
   * @brief The registered lower bound of the ulp-energy ratio (contract
   *        section 2 e, [proposal]): a <D> must exceed 16 ulp-energies
   *        everywhere in the table so that the per-event rounding residual
   *        is a small random-sign fraction of the removed energy
   */
  constexpr double ULP_RATIO_MIN = 16.0;

  /**
   * @brief The Poisson mean above which the frozen-hazard regime is left
   *        (contract section 3, [proposal]): lambda_max > 0.1 is reported
   *        under the on_max_events policy; lambda_max >= 1 is outside the
   *        intended regime
   */
  constexpr double LAMBDA_WARN = 0.1;
  constexpr double LAMBDA_MAX  = 1.0;

} // namespace arch::bh::event

#endif // ARCHETYPES_HADRONIC_BH_EVENT_HOST_H
