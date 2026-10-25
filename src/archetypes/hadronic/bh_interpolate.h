/**
 * @file archetypes/hadronic/bh_interpolate.h
 * @brief Host-side (Kokkos-free) reference of the table interpolation used by
 *        the device lookup: linear in y = log10(Gamma - Gamma_th) of ln I, 0
 *        below the lower edge, the last node above the upper edge. Shared by
 *        the device-table unit test, the problem generator's summary and the
 *        run-based check tool.
 *
 * @namespaces:
 *   - arch::bh::
 */

#ifndef ARCHETYPES_HADRONIC_BH_INTERPOLATE_H
#define ARCHETYPES_HADRONIC_BH_INTERPOLATE_H

#include "archetypes/hadronic/bh_physics.h"

#include <cmath>
#include <cstddef>

namespace arch::bh {

  /**
   * @brief Host-side reference of the same interpolation (for tests and the
   *        reference integrators): linear in y of ln I on the host table
   */
  inline auto interpolate_loss_host(const HadronicTablesHost& T,
                                    double                    Gamma) -> double {
    if (!(Gamma > T.gamma_th) || !(Gamma >= T.gamma_min)) {
      return 0.0;
    }
    if (Gamma >= T.gamma_max) {
      return T.I.back();
    }
    const double yy = std::log10(Gamma - T.gamma_th);
    if (yy < T.y_min) {
      return 0.0;
    }
    if (yy >= T.y_max) {
      return T.I.back();
    }
    // bracket by bisection on the stored nodes
    std::size_t lo = 0, hi = T.size() - 1;
    while (hi - lo > 1) {
      const std::size_t mid = (lo + hi) / 2;
      if (T.y[mid] <= yy) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const double t = (yy - T.y[lo]) / (T.y[hi] - T.y[lo]);
    return std::exp(std::log(T.I[lo]) + t * (std::log(T.I[hi]) - std::log(T.I[lo])));
  }

  /**
   * @brief Host reference of the device Dmean / EpsMean (bh_tables_device.h):
   *        <D> log-linear in y of the node values, eps_mean linear in y; the
   *        first node value below the lower edge, the last node value at and
   *        above the upper edge
   */
  inline auto interpolate_outcome_host(const HadronicTablesHost& T,
                                       double                    Gamma,
                                       double&                   Dmean,
                                       double&                   eps_mean) -> void {
    const std::size_t last = T.size() - 1;
    if (!(Gamma > T.gamma_th) || !(Gamma >= T.gamma_min)) {
      Dmean    = T.Dmean[0];
      eps_mean = T.eps_mean[0];
      return;
    }
    if (Gamma >= T.gamma_max) {
      Dmean    = T.Dmean[last];
      eps_mean = T.eps_mean[last];
      return;
    }
    const double yy = std::log10(Gamma - T.gamma_th);
    if (yy < T.y_min) {
      Dmean    = T.Dmean[0];
      eps_mean = T.eps_mean[0];
      return;
    }
    if (yy >= T.y_max) {
      Dmean    = T.Dmean[last];
      eps_mean = T.eps_mean[last];
      return;
    }
    std::size_t lo = 0, hi = last;
    while (hi - lo > 1) {
      const std::size_t mid = (lo + hi) / 2;
      if (T.y[mid] <= yy) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const double t = (yy - T.y[lo]) / (T.y[hi] - T.y[lo]);
    Dmean    = std::exp(std::log(T.Dmean[lo]) + t * (std::log(T.Dmean[hi]) - std::log(T.Dmean[lo])));
    eps_mean = T.eps_mean[lo] + t * (T.eps_mean[hi] - T.eps_mean[lo]);
  }

  /**
   * @brief Host reference of the device J interpolation (log-linear in y)
   */
  inline auto interpolate_rate_host(const HadronicTablesHost& T, double Gamma) -> double {
    if (!(Gamma > T.gamma_th) || !(Gamma >= T.gamma_min)) {
      return 0.0;
    }
    if (Gamma >= T.gamma_max) {
      return T.J.back();
    }
    const double yy = std::log10(Gamma - T.gamma_th);
    if (yy < T.y_min) {
      return 0.0;
    }
    if (yy >= T.y_max) {
      return T.J.back();
    }
    std::size_t lo = 0, hi = T.size() - 1;
    while (hi - lo > 1) {
      const std::size_t mid = (lo + hi) / 2;
      if (T.y[mid] <= yy) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const double t = (yy - T.y[lo]) / (T.y[hi] - T.y[lo]);
    return std::exp(std::log(T.J[lo]) + t * (std::log(T.J[hi]) - std::log(T.J[lo])));
  }

} // namespace arch::bh

#endif // ARCHETYPES_HADRONIC_BH_INTERPOLATE_H
