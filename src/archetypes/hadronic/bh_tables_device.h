/**
 * @file archetypes/hadronic/bh_tables_device.h
 * @brief Device mirror of the Bethe-Heitler startup tables (spec v3.1.3 part
 *        IV, `HadronicTables`) and the device-side lookup used by the kernel.
 *
 * The host tables (`arch::bh::HadronicTablesHost`, bh_physics.h) are built in
 * double precision on the host; `mirror_tables_to_device` copies the node
 * arrays into `Kokkos::View<double*>` and records the grid descriptors. The
 * lookup is linear in y = log10(Gamma - Gamma_th) of log I (and log J),
 * returning 0 below the lower edge and the edge value above the upper edge
 * (the caller counts `n_above_table`).
 *
 * Node placement (contract 1.4, step-1 report section 3.3): y_min + i/256 for
 * i = 0..256 (the first decade, `n_fine` nodes including both ends), then
 * y_min + 1 + j/64 for j >= 1 while below y_max - 1e-9, then y_max appended
 * with Gamma set exactly to Gamma_max. The analytic index is used to locate
 * the cell and the stored node positions to interpolate, so the shorter last
 * interval and short tables (ending inside the first decade) are handled by
 * the same code.
 *
 * All arithmetic is in double.
 *
 * @namespaces:
 *   - arch::bh::
 */

#ifndef ARCHETYPES_HADRONIC_BH_TABLES_DEVICE_H
#define ARCHETYPES_HADRONIC_BH_TABLES_DEVICE_H

#include "arch/kokkos_aliases.h"

#include "archetypes/hadronic/bh_interpolate.h"
#include "archetypes/hadronic/bh_physics.h"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace arch::bh {

  /**
   * @brief Device views of the startup tables plus the grid descriptors
   *
   * `log_loss`/`log_rate` hold ln I and ln J at the nodes (interpolated
   * linearly in y); `loss`, `rate`, `dmean`, `eps_mean` are the node values.
   */
  struct HadronicTables {
    double gamma_th { 0.0 };  // threshold Lorentz factor (grid variable y = log10(Gamma - gamma_th))
    double gamma_min { 0.0 }; // lower edge: I = 0 below it
    double gamma_max { 0.0 }; // upper edge: the edge value is returned above it
    double y_min { 0.0 };
    double y_max { 0.0 };
    double dy_fine { 1.0 / 256.0 };
    double dy_coarse { 1.0 / 64.0 };
    int    n_fine { 0 }; // nodes in the first decade (including both ends)
    int    n { 0 };      // total node count

    array_t<double*> y;        // node positions in y
    array_t<double*> gamma;    // node Gamma
    array_t<double*> loss;     // I(Gamma)
    array_t<double*> rate;     // J(Gamma)
    array_t<double*> dmean;    // <D>(Gamma)
    array_t<double*> eps_mean; // interaction-weighted mean photon energy
    array_t<double*> log_loss; // ln I
    array_t<double*> log_rate; // ln J
    array_t<double*> log_dmean; // ln <D> (= ln I - ln J at the nodes)

    /**
     * @brief Index of the cell containing y (0 <= idx <= n - 2), for
     *        y_min <= y < y_max
     */
    Inline auto cell_index(double yy) const -> int {
      int idx;
      if (yy < y_min + 1.0 || n_fine < 257) {
        idx = static_cast<int>((yy - y_min) / dy_fine);
      } else {
        idx = (n_fine - 1) + static_cast<int>((yy - y_min - 1.0) / dy_coarse);
      }
      if (idx < 0) {
        idx = 0;
      }
      if (idx > n - 2) {
        idx = n - 2;
      }
      // guard against the rounding of the analytic index: at most one step
      if (idx + 1 < n - 1 && yy >= y(idx + 1)) {
        ++idx;
      }
      if (idx > 0 && yy < y(idx)) {
        --idx;
      }
      return idx;
    }

    /**
     * @brief Linear interpolation in y of a log-valued node array
     * @param logv ln of the node values, linv the node values themselves
     * @param above set to true if Gamma lies above the upper edge (edge value
     *        returned)
     * @return the interpolated value, 0 below the lower edge
     */
    Inline auto interpolate_log(const array_t<double*>& logv,
                                const array_t<double*>& linv,
                                double                  Gamma,
                                bool&                   above) const -> double {
      above = false;
      if (!(Gamma > gamma_th) || !(Gamma >= gamma_min)) {
        return 0.0;
      }
      if (Gamma >= gamma_max) {
        above = Gamma > gamma_max;
        return linv(n - 1); // the edge value (the last node exactly)
      }
      const double yy = math::log10(Gamma - gamma_th);
      if (yy < y_min) {
        return 0.0;
      }
      if (yy >= y_max) {
        return linv(n - 1);
      }
      const int    i  = cell_index(yy);
      const double y0 = y(i), y1 = y(i + 1);
      const double t  = (yy - y0) / (y1 - y0);
      return math::exp(logv(i) + t * (logv(i + 1) - logv(i)));
    }

    /**
     * @brief Energy-loss rate I(Gamma) per unit photon density (m_e c^2 alpha r_e^2 c)
     */
    Inline auto I(double Gamma, bool& above) const -> double {
      return interpolate_log(log_loss, loss, Gamma, above);
    }

    Inline auto I(double Gamma) const -> double {
      bool above;
      return interpolate_log(log_loss, loss, Gamma, above);
    }

    /**
     * @brief Interaction rate J(Gamma) per unit photon density (alpha r_e^2 c)
     */
    Inline auto J(double Gamma, bool& above) const -> double {
      return interpolate_log(log_rate, rate, Gamma, above);
    }

    Inline auto J(double Gamma) const -> double {
      bool above;
      return interpolate_log(log_rate, rate, Gamma, above);
    }

    /**
     * @brief Mean ion loss per interaction <D>(Gamma) = I/J (m_e c^2), the
     *        log-linear interpolation of the node ratios (identical, to double
     *        rounding, to the ratio of the interpolated I and J). Edge
     *        conventions for the outcome law (step 3 / closing packets): the
     *        value at the first node below the lower edge, the last node value
     *        at and above the upper edge.
     */
    Inline auto Dmean(double Gamma) const -> double {
      if (!(Gamma > gamma_th) || !(Gamma >= gamma_min)) {
        return dmean(0);
      }
      if (Gamma >= gamma_max) {
        return dmean(n - 1);
      }
      const double yy = math::log10(Gamma - gamma_th);
      if (yy < y_min) {
        return dmean(0);
      }
      if (yy >= y_max) {
        return dmean(n - 1);
      }
      const int    i  = cell_index(yy);
      const double y0 = y(i), y1 = y(i + 1);
      const double t  = (yy - y0) / (y1 - y0);
      return math::exp(log_dmean(i) + t * (log_dmean(i + 1) - log_dmean(i)));
    }

    /**
     * @brief Interaction-weighted mean photon energy eps_mean(Gamma) (m_e c^2),
     *        linear in y of the node values, same edge conventions as Dmean
     */
    Inline auto EpsMean(double Gamma) const -> double {
      if (!(Gamma > gamma_th) || !(Gamma >= gamma_min)) {
        return eps_mean(0);
      }
      if (Gamma >= gamma_max) {
        return eps_mean(n - 1);
      }
      const double yy = math::log10(Gamma - gamma_th);
      if (yy < y_min) {
        return eps_mean(0);
      }
      if (yy >= y_max) {
        return eps_mean(n - 1);
      }
      const int    i  = cell_index(yy);
      const double y0 = y(i), y1 = y(i + 1);
      const double t  = (yy - y0) / (y1 - y0);
      return eps_mean(i) + t * (eps_mean(i + 1) - eps_mean(i));
    }
  };

  /**
   * @brief Copy the host tables to device views
   * @throws std::invalid_argument on an empty or inconsistent host table
   */
  inline auto mirror_tables_to_device(const HadronicTablesHost& T) -> HadronicTables {
    if (T.size() < 2) {
      throw std::invalid_argument(
        "mirror_tables_to_device: fewer than two nodes in the host table");
    }
    if (T.y.size() != T.size() || T.I.size() != T.size() ||
        T.J.size() != T.size() || T.Dmean.size() != T.size() ||
        T.eps_mean.size() != T.size()) {
      throw std::invalid_argument(
        "mirror_tables_to_device: inconsistent node array sizes");
    }
    if (T.n_fine < 2 || T.n_fine > T.size()) {
      throw std::invalid_argument(
        "mirror_tables_to_device: n_fine = " + std::to_string(T.n_fine) +
        " inconsistent with " + std::to_string(T.size()) + " nodes");
    }
    HadronicTables D;
    D.gamma_th  = T.gamma_th;
    D.gamma_min = T.gamma_min;
    D.gamma_max = T.gamma_max;
    D.y_min     = T.y_min;
    D.y_max     = T.y_max;
    D.dy_fine   = 1.0 / static_cast<double>(T.nodes_per_decade_fine);
    D.dy_coarse = 1.0 / static_cast<double>(T.nodes_per_decade_coarse);
    D.n_fine    = static_cast<int>(T.n_fine);
    D.n         = static_cast<int>(T.size());

    const auto n = T.size();
    D.y          = array_t<double*> { "bh_y", n };
    D.gamma      = array_t<double*> { "bh_gamma", n };
    D.loss       = array_t<double*> { "bh_loss", n };
    D.rate       = array_t<double*> { "bh_rate", n };
    D.dmean      = array_t<double*> { "bh_dmean", n };
    D.eps_mean   = array_t<double*> { "bh_eps_mean", n };
    D.log_loss   = array_t<double*> { "bh_log_loss", n };
    D.log_rate   = array_t<double*> { "bh_log_rate", n };
    D.log_dmean  = array_t<double*> { "bh_log_dmean", n };

    auto h_y   = Kokkos::create_mirror_view(D.y);
    auto h_g   = Kokkos::create_mirror_view(D.gamma);
    auto h_I   = Kokkos::create_mirror_view(D.loss);
    auto h_J   = Kokkos::create_mirror_view(D.rate);
    auto h_D   = Kokkos::create_mirror_view(D.dmean);
    auto h_e   = Kokkos::create_mirror_view(D.eps_mean);
    auto h_lI  = Kokkos::create_mirror_view(D.log_loss);
    auto h_lJ  = Kokkos::create_mirror_view(D.log_rate);
    auto h_lD  = Kokkos::create_mirror_view(D.log_dmean);
    for (std::size_t i = 0; i < n; ++i) {
      if (!(T.I[i] > 0.0) || !(T.J[i] > 0.0) || !std::isfinite(T.I[i]) ||
          !std::isfinite(T.J[i])) {
        throw std::invalid_argument(
          "mirror_tables_to_device: I or J not positive and finite at node " +
          std::to_string(i));
      }
      h_y(i)  = T.y[i];
      h_g(i)  = T.gamma[i];
      h_I(i)  = T.I[i];
      h_J(i)  = T.J[i];
      h_D(i)  = T.Dmean[i];
      h_e(i)  = T.eps_mean[i];
      h_lI(i) = std::log(T.I[i]);
      h_lJ(i) = std::log(T.J[i]);
      if (!(T.Dmean[i] > 0.0) || !std::isfinite(T.Dmean[i]) ||
          !(T.eps_mean[i] > 0.0) || !std::isfinite(T.eps_mean[i])) {
        throw std::invalid_argument(
          "mirror_tables_to_device: <D> or eps_mean not positive and finite at node " +
          std::to_string(i));
      }
      h_lD(i) = std::log(T.Dmean[i]);
    }
    Kokkos::deep_copy(D.y, h_y);
    Kokkos::deep_copy(D.gamma, h_g);
    Kokkos::deep_copy(D.loss, h_I);
    Kokkos::deep_copy(D.rate, h_J);
    Kokkos::deep_copy(D.dmean, h_D);
    Kokkos::deep_copy(D.eps_mean, h_e);
    Kokkos::deep_copy(D.log_loss, h_lI);
    Kokkos::deep_copy(D.log_rate, h_lJ);
    Kokkos::deep_copy(D.log_dmean, h_lD);
    return D;
  }

} // namespace arch::bh

#endif // ARCHETYPES_HADRONIC_BH_TABLES_DEVICE_H
