/**
 * @file tests/kernels/perf/bh_perf_tables.h
 * @brief P2 step 7, M3 (scaling-harness contract 2026-10-30 §1): the two harness table variants that replace
 *        the production lookup (arch::bh::HadronicTables: log-linear interpolation in y = log10(Gamma - Gamma_th)
 *        of ln I, ln J, ln <D>, with log10 and exp per lookup) in the kernel copy bh_perf_kernel_variant.hpp:
 *   (i)  PerfTablesConst: the lookup removed; I, J, <D>, eps_mean are constants (the production values at the
 *        reference Gamma_p); the edge tests (lower edge, upper edge) are kept so that the control flow of the
 *        kernel is unchanged.
 *   (ii) PerfTablesFloat: single-precision tables on a uniform grid in Gamma_p over [gamma_min, gamma_max] with
 *        the production node count, linear interpolation of the values themselves (no log, no exp); built on
 *        the host from the production host table.
 *        Same API as the production table as used by the kernel: gamma_min, gamma_max, I(G), I(G, above),
 *        J(G), J(G, above), Dmean(G), EpsMean(G). PERF BUILD ONLY (BH_PERF_VARIANT): never a physics number.
 */
#ifndef TESTS_KERNELS_PERF_BH_PERF_TABLES_H
#define TESTS_KERNELS_PERF_BH_PERF_TABLES_H

#ifndef BH_PERF_VARIANT
  #error "bh_perf_tables.h is perf-harness code: compile with BH_PERF_VARIANT (test build only)"
#endif

#include "global.h"

#include "arch/kokkos_aliases.h"

#include "archetypes/hadronic/bh_interpolate.h"
#include "archetypes/hadronic/bh_table_io.h"

#include <Kokkos_Core.hpp>

#include <cmath>

namespace kernel::hadronic_perf {
  using namespace ntt;

  struct PerfTablesConst {
    double gamma_th { 0.0 }, gamma_min { 0.0 }, gamma_max { 0.0 };
    double I0 { 0.0 }, J0 { 0.0 }, D0 { 0.0 }, eps0 { 0.0 };

    Inline auto I(double Gamma, bool& above) const -> double {
      above = false;
      if (!(Gamma > gamma_th) || !(Gamma >= gamma_min)) {
        return 0.0;
      }
      if (Gamma >= gamma_max) {
        above = Gamma > gamma_max;
        return I0;
      }
      return I0;
    }

    Inline auto I(double Gamma) const -> double {
      bool above;
      return I(Gamma, above);
    }

    Inline auto J(double Gamma, bool& above) const -> double {
      above = false;
      if (!(Gamma > gamma_th) || !(Gamma >= gamma_min)) {
        return 0.0;
      }
      if (Gamma >= gamma_max) {
        above = Gamma > gamma_max;
        return J0;
      }
      return J0;
    }

    Inline auto J(double Gamma) const -> double {
      bool above;
      return J(Gamma, above);
    }

    Inline auto Dmean(double /*Gamma*/) const -> double {
      return D0;
    }

    Inline auto EpsMean(double /*Gamma*/) const -> double {
      return eps0;
    }
  };

  struct PerfTablesFloat {
    double          gamma_th { 0.0 }, gamma_min { 0.0 }, gamma_max { 0.0 };
    float           g_min_f { 0.0f }, inv_dg_f { 0.0f };
    int             n { 0 };
    array_t<float*> I_f, J_f, D_f, eps_f; // node values at gamma_min + k dg, k = 0 .. n - 1

    Inline auto lerp(const array_t<float*>& v, double Gamma, bool& above) const -> double {
      above = false;
      if (!(Gamma > gamma_th) || !(Gamma >= gamma_min)) {
        return 0.0;
      }
      if (Gamma >= gamma_max) {
        above = Gamma > gamma_max;
        return static_cast<double>(v(n - 1));
      }
      const float x = (static_cast<float>(Gamma) - g_min_f) * inv_dg_f;
      int         i = static_cast<int>(x);
      if (i < 0) {
        i = 0;
      }
      if (i > n - 2) {
        i = n - 2;
      }
      const float t = x - static_cast<float>(i);
      return static_cast<double>(v(i) + t * (v(i + 1) - v(i)));
    }

    Inline auto I(double Gamma, bool& above) const -> double {
      return lerp(I_f, Gamma, above);
    }

    Inline auto I(double Gamma) const -> double {
      bool above;
      return lerp(I_f, Gamma, above);
    }

    Inline auto J(double Gamma, bool& above) const -> double {
      return lerp(J_f, Gamma, above);
    }

    Inline auto J(double Gamma) const -> double {
      bool above;
      return lerp(J_f, Gamma, above);
    }

    Inline auto Dmean(double Gamma) const -> double {
      bool         above;
      const double d = lerp(D_f, Gamma, above);
      return d > 0.0 ? d : static_cast<double>(D_f(0));
    }

    Inline auto EpsMean(double Gamma) const -> double {
      bool         above;
      const double e = lerp(eps_f, Gamma, above);
      return e > 0.0 ? e : static_cast<double>(eps_f(0));
    }
  };

  // host builders from the production host table
  inline auto make_const_tables(const arch::bh::HadronicTablesHost& H, double Gamma_ref) -> PerfTablesConst {
    PerfTablesConst T;
    T.gamma_th  = H.gamma_th;
    T.gamma_min = H.gamma_min;
    T.gamma_max = H.gamma_max;
    T.I0        = arch::bh::interpolate_loss_host(H, Gamma_ref);
    T.J0        = arch::bh::interpolate_rate_host(H, Gamma_ref);
    double D, e;
    arch::bh::interpolate_outcome_host(H, Gamma_ref, D, e);
    T.D0   = D;
    T.eps0 = e;
    return T;
  }

  inline auto make_float_tables(const arch::bh::HadronicTablesHost& H) -> PerfTablesFloat {
    PerfTablesFloat T;
    T.gamma_th  = H.gamma_th;
    T.gamma_min = H.gamma_min;
    T.gamma_max = H.gamma_max;
    T.n         = static_cast<int>(H.gamma.size());
    const double dg = (H.gamma_max - H.gamma_min) / static_cast<double>(T.n - 1);
    T.g_min_f       = static_cast<float>(H.gamma_min);
    T.inv_dg_f      = static_cast<float>(1.0 / dg);
    T.I_f           = array_t<float*> { "perf_I_f", static_cast<std::size_t>(T.n) };
    T.J_f           = array_t<float*> { "perf_J_f", static_cast<std::size_t>(T.n) };
    T.D_f           = array_t<float*> { "perf_D_f", static_cast<std::size_t>(T.n) };
    T.eps_f         = array_t<float*> { "perf_eps_f", static_cast<std::size_t>(T.n) };
    auto hI = Kokkos::create_mirror_view(T.I_f), hJ = Kokkos::create_mirror_view(T.J_f), hD = Kokkos::create_mirror_view(T.D_f),
         he = Kokkos::create_mirror_view(T.eps_f);
    for (int k = 0; k < T.n; ++k) {
      const double G = std::min(H.gamma_max, H.gamma_min + dg * k);
      hI(k)          = static_cast<float>(arch::bh::interpolate_loss_host(H, G));
      hJ(k)          = static_cast<float>(arch::bh::interpolate_rate_host(H, G));
      double D, e;
      arch::bh::interpolate_outcome_host(H, G, D, e);
      hD(k) = static_cast<float>(D);
      he(k) = static_cast<float>(e);
    }
    Kokkos::deep_copy(T.I_f, hI);
    Kokkos::deep_copy(T.J_f, hJ);
    Kokkos::deep_copy(T.D_f, hD);
    Kokkos::deep_copy(T.eps_f, he);
    return T;
  }

} // namespace kernel::hadronic_perf

#endif // TESTS_KERNELS_PERF_BH_PERF_TABLES_H
