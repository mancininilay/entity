/**
 * @file kernels/hadronic/bethe_heitler.hpp
 * @brief Bethe-Heitler ion interaction kernel, loss-only subset (spec v3.1.3
 *        II.2 steps 0 to 3 with `source = none`): continuous ion energy loss
 *        read from the startup tables with the precision-safe pending-loss
 *        application of III.3. No debt, packets, pairs or RNG are built here;
 *        the ledger carries every field of II.2 step 6 with the pair-related
 *        ones identically zero.
 *
 * Conventions (all energy bookkeeping in double):
 *   K     = mu_d u^2 / (gamma_d + 1)            ion kinetic energy (m_e c^2)
 *   Gamma_p = 1 + K / mu_r                      physical proton Lorentz factor
 *   dK    = (tau0 / skindepth0) g(x) I(Gamma_p) dt   requested loss per step
 *   delta = pld_r(p, pending)                   pending (unapplied) loss request
 *   delta_th = max(pending_floor, eta_u K),  pending_floor = pending_floor_ulp
 *              * ulp(|u| as real_t) * mu_d u / gamma_d
 *   application: K_new = max(K - delta, K_floor), u_new = sqrt(K_new (K_new +
 *   2 mu_d)) / mu_d along u-hat, stored as real_t; dK_applied = K(u_old) -
 *   K(u_new_stored) from the two stored values; delta -= dK_applied.
 *
 * The functor is used with `Kokkos::parallel_reduce` and the `LossLedger`
 * value type (a struct of doubles and 64-bit counters reduced with
 * `Kokkos::Sum`); no real_t atomics are used anywhere.
 *
 * @implements
 *   - kernel::hadronic::HadronicParams
 *   - kernel::hadronic::LossLedger
 *   - kernel::hadronic::BetheHeitler_kernel<>
 * @namespaces:
 *   - kernel::hadronic::
 */

#ifndef KERNELS_HADRONIC_BETHE_HEITLER_HPP
#define KERNELS_HADRONIC_BETHE_HEITLER_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "archetypes/hadronic/bh_tables_device.h"
#include "framework/containers/particles.h"

#include <Kokkos_Core.hpp>

#include <limits>

namespace kernel::hadronic {
  using namespace ntt;

  /**
   * @brief Spatial profile g(x) of the photon bath (D5); only `uniform` in
   *        this step
   */
  enum class BathProfile : int {
    Uniform = 0
  };

  /**
   * @brief Scalar parameters of the kernel (II.2), loss-only subset
   */
  struct HadronicParams {
    double tau0_over_skindepth0 { 0.0 };
    double mu_r { 1836.15 };
    double eta_u { 1.0e-4 };
    double pending_floor_ulp { 100.0 };
    double f_max { 0.1 };
    int    max_substeps { 64 };
    // payload indices into pld_r of the ion species
    unsigned short pld_pending { 0 };
    unsigned short pld_debt { 1 };
    unsigned short pld_debt_k0 { 2 };
    double         dt { 0.0 };
    double         time { 0.0 };
    int            profile { static_cast<int>(BathProfile::Uniform) };
    // "design off" (T5b only): the table dK is applied to u every step with no
    // pending accumulation (the remainder of each application is discarded)
    bool unsafe_application { false };
  };

  /**
   * @brief Per-launch ledger (II.2 step 6 and part IV), reduced in double
   *
   * Cumulative entries are summed by the host over launches; `debt_outstanding`,
   * `pending_outstanding`, `sum_wK`, `n_alive` are snapshots of the launch.
   */
  struct LossLedger {
    double E_drag { 0.0 };
    double E_pairs { 0.0 };
    double E_absorbed { 0.0 };
    double E_exported { 0.0 };
    double E_round { 0.0 };
    double E_lost_nopairs { 0.0 };
    double pending_discarded { 0.0 };
    double P_drag[3] { 0.0, 0.0, 0.0 };
    double P_pairs[3] { 0.0, 0.0, 0.0 };
    double P_absorbed[3] { 0.0, 0.0, 0.0 };
    // snapshots
    double debt_outstanding { 0.0 };
    double pending_outstanding { 0.0 };
    double sum_wK { 0.0 }; // sum of w K over alive and send-tagged ions after the launch
    // counters
    unsigned long long n_events { 0 };
    unsigned long long n_spawned { 0 };
    unsigned long long n_dropped { 0 };
    unsigned long long n_substeps { 0 };
    unsigned long long n_bounded { 0 };
    unsigned long long n_capped_loss { 0 }; // steps on which the f_max K cap changed the applied loss
    unsigned long long n_closing { 0 };
    unsigned long long n_capped { 0 };
    unsigned long long n_above_table { 0 };
    unsigned long long n_latency { 0 };
    unsigned long long n_resolution { 0 };
    unsigned long long n_processed { 0 }; // ions whose loss was evaluated from the table
    unsigned long long n_alive { 0 };     // snapshot: alive and send-tagged ions
    unsigned long long n_applied { 0 };   // applications to u this launch

    KOKKOS_INLINE_FUNCTION
    LossLedger() = default;

    KOKKOS_INLINE_FUNCTION
    auto operator+=(const LossLedger& o) -> LossLedger& {
      E_drag            += o.E_drag;
      E_pairs           += o.E_pairs;
      E_absorbed        += o.E_absorbed;
      E_exported        += o.E_exported;
      E_round           += o.E_round;
      E_lost_nopairs    += o.E_lost_nopairs;
      pending_discarded += o.pending_discarded;
      for (int d = 0; d < 3; ++d) {
        P_drag[d]     += o.P_drag[d];
        P_pairs[d]    += o.P_pairs[d];
        P_absorbed[d] += o.P_absorbed[d];
      }
      debt_outstanding    += o.debt_outstanding;
      pending_outstanding += o.pending_outstanding;
      sum_wK              += o.sum_wK;
      n_events            += o.n_events;
      n_spawned           += o.n_spawned;
      n_dropped           += o.n_dropped;
      n_substeps          += o.n_substeps;
      n_bounded           += o.n_bounded;
      n_capped_loss       += o.n_capped_loss;
      n_closing           += o.n_closing;
      n_capped            += o.n_capped;
      n_above_table       += o.n_above_table;
      n_latency           += o.n_latency;
      n_resolution        += o.n_resolution;
      n_processed         += o.n_processed;
      n_alive             += o.n_alive;
      n_applied           += o.n_applied;
      return *this;
    }
  };

  /**
   * @brief Ion kinetic energy in double from three stored momentum components
   *        (the expression shared by the kernel and the array-based closure
   *        checks: K = mu u^2 / (gamma + 1), no cancellation)
   */
  Inline auto kinetic_energy(double mu, double u1, double u2, double u3) -> double {
    const double u2s = u1 * u1 + u2 * u2 + u3 * u3;
    const double g   = math::sqrt(1.0 + u2s);
    return mu * u2s / (g + 1.0);
  }

  /**
   * @brief ulp of a real_t value: nextafter(x, +inf) - x (x >= 0)
   */
  Inline auto ulp_real(real_t x) -> double {
    if (!math::isfinite(x)) {
      x = std::numeric_limits<real_t>::max();
    }
    if (x == std::numeric_limits<real_t>::max()) {
      // the spacing just below the largest finite value
      const real_t down = math::nextafter(x, static_cast<real_t>(0));
      return static_cast<double>(x) - static_cast<double>(down);
    }
    const real_t up = math::nextafter(x, std::numeric_limits<real_t>::infinity());
    return static_cast<double>(up) - static_cast<double>(x);
  }

  /**
   * @brief Loss-only Bethe-Heitler kernel over the ions of one species
   * @tparam M Metric (template as the other particle kernels; unused in the
   *           uniform-profile subset)
   */
  template <class M>
  struct BetheHeitler_kernel {
    static constexpr auto D = M::Dim;

    ParticleArrays          prtls;
    const double            mu_d;
    const arch::bh::HadronicTables tables;
    const HadronicParams    prm;

    BetheHeitler_kernel(const ParticleArrays&          arrays,
                        double                         ion_mass,
                        const arch::bh::HadronicTables& tbl,
                        const HadronicParams&          params)
      : prtls { arrays }
      , mu_d { ion_mass }
      , tables { tbl }
      , prm { params } {}

    /**
     * @brief I(Gamma_p(K)) with the edge conventions (0 below, edge value above)
     */
    Inline auto loss_rate_at(double K) const -> double {
      const double Gp = 1.0 + K / prm.mu_r;
      return tables.I(Gp);
    }

    Inline void operator()(prtlidx_t p, LossLedger& L) const {
      const short tag = prtls.tag(p);
      const double w  = static_cast<double>(prtls.weight(p));
      double delta    = static_cast<double>(prtls.pld_r(p, prm.pld_pending));

      if (tag == ParticleTag::dead) {
        // absorbed this step: the request is never fulfilled
        if (delta != 0.0) {
          L.pending_discarded        += w * delta;
          prtls.pld_r(p, prm.pld_pending) = ZERO;
        }
        return;
      }
      // alive or send-tagged: process

      // step 0: energies
      const real_t u1r = prtls.ux1(p), u2r = prtls.ux2(p), u3r = prtls.ux3(p);
      const double u1 = static_cast<double>(u1r);
      const double u2 = static_cast<double>(u2r);
      const double u3 = static_cast<double>(u3r);
      const double u2s = u1 * u1 + u2 * u2 + u3 * u3;
      const double u   = math::sqrt(u2s);
      const double gd  = math::sqrt(1.0 + u2s);
      const double K   = mu_d * u2s / (gd + 1.0);
      double       Gp  = 1.0 + K / prm.mu_r;

      if (Gp > tables.gamma_max) {
        Gp = tables.gamma_max; // clamp to the edge: edge value of the table
        L.n_above_table += 1;
      }
      if (Gp < tables.gamma_min) {
        // below the lower edge: no loss, no debt in this step (no closing packet)
        if (delta != 0.0) {
          L.pending_discarded        += w * delta;
          prtls.pld_r(p, prm.pld_pending) = ZERO;
        }
        L.n_alive += 1;
        L.sum_wK  += w * K;
        return;
      }

      // step 1: requested loss
      const double g_x   = 1.0; // uniform profile (D5)
      const double coeff = prm.tau0_over_skindepth0 * g_x;
      double       dK    = coeff * tables.I(Gp) * prm.dt;
      const double K_floor = prm.mu_r * (tables.gamma_min - 1.0);
      // a request that is not finite (overflowing coefficient) or exceeds
      // f_max K is sub-cycled; the substep count is compared in double before
      // any conversion (R2: no undefined double-to-int conversion)
      const bool finite_request = math::isfinite(dK);
      if (K > 0.0 && (!finite_request || dK > prm.f_max * K)) {
        // sub-cycle: RK2 (midpoint) on dK/dt = -coeff I(Gamma_p(K))
        const double n_req = finite_request ? math::ceil(dK / (prm.f_max * K))
                                            : std::numeric_limits<double>::infinity();
        int  n       = prm.max_substeps;
        bool bounded = true;
        if (finite_request && n_req <= static_cast<double>(prm.max_substeps)) {
          n       = static_cast<int>(n_req);
          bounded = false;
        }
        if (n < 1) {
          n = 1;
        }
        const double h  = prm.dt / static_cast<double>(n);
        double       Kc = K;
        for (int s = 0; s < n; ++s) {
          const double k1 = -coeff * loss_rate_at(Kc);
          double       Km = Kc + 0.5 * h * k1;
          if (Km < K_floor) {
            Km = K_floor;
          }
          const double k2 = -coeff * loss_rate_at(Km);
          Kc             += h * k2;
          if (Kc < K_floor) {
            Kc = K_floor;
          }
        }
        dK = K - Kc;
        if (!math::isfinite(dK)) {
          dK = K - K_floor; // the integrator clamps at the floor; never NaN
        }
        if (bounded) {
          // explicit clamp reading (decision 2026-10-25 (c), III.7): the loss
          // of max_substeps RK2 substeps is capped at f_max K; n_bounded counts
          // the exhaustion, n_capped_loss the steps on which the cap changed
          // the applied value
          if (dK > prm.f_max * K) {
            dK = prm.f_max * K;
            L.n_capped_loss += 1;
          }
          L.n_bounded += 1;
        }
        L.n_substeps += static_cast<unsigned long long>(n);
      }
      delta += dK;
      L.n_processed += 1;

      // step 2: precision-safe application. The ulp of the stored momentum
      // norm is taken from the double norm u rounded to real_t (R4: a real_t
      // sum of squares overflows for |u| ~ 1e20); a norm that is not
      // representable in real_t takes the ulp of the largest finite real_t
      const real_t ur = math::isfinite(static_cast<real_t>(u))
                          ? static_cast<real_t>(u)
                          : std::numeric_limits<real_t>::max();
      const double pending_floor = prm.pending_floor_ulp * ulp_real(ur) * mu_d *
                                   u / gd;
      const double delta_th = (pending_floor > prm.eta_u * K) ? pending_floor
                                                              : prm.eta_u * K;
      if (delta > 0.0 && delta >= delta_th && u > 0.0) {
        double K_new = K - delta;
        if (K_new < K_floor) {
          K_new = K_floor;
        }
        if (K_new > K) {
          K_new = K;
        }
        const double u_new = math::sqrt(K_new * (K_new + 2.0 * mu_d)) / mu_d;
        const double scale = u_new / u;
        const real_t n1 = static_cast<real_t>(scale * u1);
        const real_t n2 = static_cast<real_t>(scale * u2);
        const real_t n3 = static_cast<real_t>(scale * u3);
        prtls.ux1(p) = n1;
        prtls.ux2(p) = n2;
        prtls.ux3(p) = n3;
        const double K_stored = kinetic_energy(mu_d,
                                               static_cast<double>(n1),
                                               static_cast<double>(n2),
                                               static_cast<double>(n3));
        const double dK_applied = K - K_stored;
        delta -= dK_applied;
        if (prm.unsafe_application) {
          delta = 0.0; // design off: no pending accumulation
        }
        L.E_drag         += w * dK_applied;
        L.E_lost_nopairs += w * dK_applied;
        L.P_drag[0]      += w * mu_d * (u1 - static_cast<double>(n1));
        L.P_drag[1]      += w * mu_d * (u2 - static_cast<double>(n2));
        L.P_drag[2]      += w * mu_d * (u3 - static_cast<double>(n3));
        L.n_applied      += 1;
        L.sum_wK         += w * K_stored;
      } else {
        if (prm.unsafe_application) {
          delta = 0.0;
        }
        L.sum_wK += w * K;
      }
      const real_t delta_r = static_cast<real_t>(delta);
      prtls.pld_r(p, prm.pld_pending) = delta_r;

      // step 3: snapshots (debt-related entries stay zero: source = none)
      L.pending_outstanding += w * static_cast<double>(delta_r);
      L.n_alive             += 1;
    }
  };

} // namespace kernel::hadronic

namespace Kokkos {
  template <>
  struct reduction_identity<kernel::hadronic::LossLedger> {
    KOKKOS_FORCEINLINE_FUNCTION static kernel::hadronic::LossLedger sum() {
      return kernel::hadronic::LossLedger {};
    }
  };
} // namespace Kokkos

#endif // KERNELS_HADRONIC_BETHE_HEITLER_HPP
