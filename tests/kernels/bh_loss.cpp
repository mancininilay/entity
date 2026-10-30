/**
 * @file tests/kernels/bh_loss.cpp
 * @brief Pre-registered tests of the loss-only Bethe-Heitler kernel (P2 step 2
 *        brief, section 2): device tables (T3-like), T5 (cooling curve and
 *        dt-convergence), T5b single-build part (ulp-ratio scan, design on and
 *        off), T6 (mu_d independence), T9 (threshold), T11-lite (closure from
 *        the arrays), sub-cycling, above-table. The reference for every cooling
 *        curve is RK4 in double of dK/dt = -(tau0/skindepth0) I(Gamma_p(K)) with
 *        the exact rate `BetheHeitlerRates::at_gamma` (and, separately, with
 *        the interpolated table), computed here.
 *
 * Conventions (declared before any run):
 *   - bath mono eps0 = 0.5 (Gamma_th = 2.125), mu_r = 1836.15, skindepth0 = 1
 *   - "relative error" = |K_sim - K_ref| / K_ref with K_sim from the stored u
 *     (double from the real_t components); K_eff = K_sim - delta (the stored
 *     energy minus the pending request) is reported alongside
 *   - re-registered 2026-10-25 (decision (a)): the dt-convergence order is
 *     judged on K_eff against the interpolated-table RK4 with eta_u scaled as
 *     dt (1e-4 at the base dt, halved with dt), pass 1.0 +- 0.3, with the ratio
 *     eta_u K / pending_floor printed at every dt; the round-1 fixed-eta_u
 *     stored-K orders are printed as (diag); T6 is judged on K_eff (<= 1e-6);
 *     every run checks the stored-K envelope |K - K_eff| <= eta_u K +
 *     pending_floor at every output time; f_max defaults to 0.01
 *   - the ulp ratio of T5b is dK / dK_ulp with dK_ulp = ulp(|u|) mu_d u /
 *     gamma_d (the energy resolution of the stored momentum; equal to
 *     dK/ulp(u) for mu_d = 1, u >> 1)
 *
 * Verdict lines: "PASS" / "FAIL" with the measured number and the tolerance;
 * "EXPECTED-FAIL" for the T5b design-off runs; "[DIAG]" lines are reported only.
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "metrics/minkowski.h"

#include "archetypes/hadronic/bh_born_qk.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_tables_device.h"
#include "framework/containers/particles.h"
#include "kernels/hadronic/bethe_heitler.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;
namespace bh = arch::bh;
namespace kh = kernel::hadronic;

namespace {

  int n_pass = 0, n_fail = 0, n_expected_fail = 0;

  void verdict(bool ok, const std::string& what, double value, double tol,
               bool expected_fail = false) {
    if (expected_fail) {
      std::printf("%s (expected failure): %s : %.6e (tolerance %.3e)\n",
                  ok ? "PASS" : "EXPECTED-FAIL", what.c_str(), value, tol);
      if (!ok) {
        ++n_expected_fail;
      } else {
        ++n_pass;
      }
      return;
    }
    std::printf("%s: %s : %.6e (tolerance %.3e)\n", ok ? "PASS" : "FAIL",
                what.c_str(), value, tol);
    if (ok) {
      ++n_pass;
    } else {
      ++n_fail;
    }
  }

  void diag(const std::string& s) {
    std::printf("(diag) %s\n", s.c_str());
  }

  constexpr double MU_R = 1836.15;
  constexpr double EPS0 = 0.5;

  struct Tables {
    std::shared_ptr<const bh::PhotonBath> bath;
    std::unique_ptr<bh::BetheHeitlerRates> rates;
    bh::HadronicTablesHost                 host;
    bh::HadronicTables                     dev;
  };

  Tables make_tables(double mu_r = MU_R, double eps0 = EPS0) {
    Tables T;
    bh::BathSpec spec;
    spec.type = "mono";
    spec.eps0 = eps0;
    T.bath    = bh::make_bath(spec);
    bh::Quadrature q;
    if (const char* e = std::getenv("BH_QK_SOURCE")) { // step 6: BH_QK_SOURCE=born runs the group on the Born tables
      if (std::string(e) == "born") {
        q.qk_source = bh::QkSource::Born;
      }
    }
    T.rates   = std::make_unique<bh::BetheHeitlerRates>(T.bath, q);
    T.host    = bh::build_tables(*T.rates, mu_r);
    bh::validate_tables(T.host);
    T.dev = bh::mirror_tables_to_device(T.host);
    return T;
  }

  /**
   * @brief exact loss rate I(Gamma_p(K)) (0 at or below the threshold)
   */
  double exact_rate_K(const Tables& T, double mu_r, double K) {
    const double G = 1.0 + K / mu_r;
    if (!(G > T.host.gamma_th)) {
      return 0.0;
    }
    return T.rates->at_gamma(G).I;
  }

  double table_rate_K(const Tables& T, double mu_r, double K) {
    return bh::interpolate_loss_host(T.host, 1.0 + K / mu_r);
  }

  /**
   * @brief RK4 of dK/dt = -coeff f(K) [+ gain(K)] from t = 0 to t_end with
   *        fixed step h (the last step shortened), returning K(t_end)
   */
  double rk4(double K0, double t_end, double h,
             const std::function<double(double)>& dKdt) {
    double K = K0, t = 0.0;
    while (t < t_end - 1e-15 * t_end) {
      const double hh = std::min(h, t_end - t);
      const double k1 = dKdt(K);
      const double k2 = dKdt(K + 0.5 * hh * k1);
      const double k3 = dKdt(K + 0.5 * hh * k2);
      const double k4 = dKdt(K + hh * k3);
      K += hh / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
      t += hh;
    }
    return K;
  }

  /**
   * @brief Reference curve sampled at the given times (RK4 continued between
   *        samples)
   */
  std::vector<double> rk4_curve(double K0, const std::vector<double>& times, double h,
                                const std::function<double(double)>& dKdt) {
    std::vector<double> out;
    double K = K0, t = 0.0;
    for (const double tn : times) {
      const double seg = tn - t;
      if (seg > 0.0) {
        K = rk4(K, seg, h, dKdt);
      }
      t = tn;
      out.push_back(K);
    }
    return out;
  }

  /* ------------------------------------------------------------------------ */
  /* ion set on the device                                                      */
  /* ------------------------------------------------------------------------ */
  struct Ions {
    ParticleArrays arr;
    npart_t        n { 0 };
    double         mu_d { 1.0 };

    Ions(npart_t n_, double mu_d_) : n { n_ }, mu_d { mu_d_ } {
      arr.i1     = array_t<int*> { "i1", n };
      arr.i2     = array_t<int*> { "i2", n };
      arr.i3     = array_t<int*> { "i3", n };
      arr.dx1    = array_t<prtldx_t*> { "dx1", n };
      arr.dx2    = array_t<prtldx_t*> { "dx2", n };
      arr.dx3    = array_t<prtldx_t*> { "dx3", n };
      arr.ux1    = array_t<real_t*> { "ux1", n };
      arr.ux2    = array_t<real_t*> { "ux2", n };
      arr.ux3    = array_t<real_t*> { "ux3", n };
      arr.weight = array_t<real_t*> { "weight", n };
      arr.tag    = array_t<short*> { "tag", n };
      arr.pld_r  = array_t<real_t**> { "pld_r", n, 3 };
    }

    static double u_of_K(double K, double mu) {
      return std::sqrt(K * (K + 2.0 * mu)) / mu;
    }

    /**
     * @brief set ion p with kinetic energy K along the unit vector (c1, c2, c3)
     */
    void set(npart_t p, double K, double c1 = 1.0, double c2 = 0.0, double c3 = 0.0,
             double w = 1.0, short tag = ParticleTag::alive) {
      auto h1 = Kokkos::create_mirror_view(arr.ux1);
      auto h2 = Kokkos::create_mirror_view(arr.ux2);
      auto h3 = Kokkos::create_mirror_view(arr.ux3);
      auto hw = Kokkos::create_mirror_view(arr.weight);
      auto ht = Kokkos::create_mirror_view(arr.tag);
      auto hp = Kokkos::create_mirror_view(arr.pld_r);
      Kokkos::deep_copy(h1, arr.ux1);
      Kokkos::deep_copy(h2, arr.ux2);
      Kokkos::deep_copy(h3, arr.ux3);
      Kokkos::deep_copy(hw, arr.weight);
      Kokkos::deep_copy(ht, arr.tag);
      Kokkos::deep_copy(hp, arr.pld_r);
      const double u = u_of_K(K, mu_d);
      h1(p)          = static_cast<real_t>(u * c1);
      h2(p)          = static_cast<real_t>(u * c2);
      h3(p)          = static_cast<real_t>(u * c3);
      hw(p)          = static_cast<real_t>(w);
      ht(p)          = tag;
      for (int c = 0; c < 3; ++c) {
        hp(p, c) = ZERO;
      }
      Kokkos::deep_copy(arr.ux1, h1);
      Kokkos::deep_copy(arr.ux2, h2);
      Kokkos::deep_copy(arr.ux3, h3);
      Kokkos::deep_copy(arr.weight, hw);
      Kokkos::deep_copy(arr.tag, ht);
      Kokkos::deep_copy(arr.pld_r, hp);
    }

    std::vector<double> K() const {
      auto h1 = Kokkos::create_mirror_view(arr.ux1);
      auto h2 = Kokkos::create_mirror_view(arr.ux2);
      auto h3 = Kokkos::create_mirror_view(arr.ux3);
      Kokkos::deep_copy(h1, arr.ux1);
      Kokkos::deep_copy(h2, arr.ux2);
      Kokkos::deep_copy(h3, arr.ux3);
      std::vector<double> out(n);
      for (npart_t p = 0; p < n; ++p) {
        out[p] = kh::kinetic_energy(mu_d, h1(p), h2(p), h3(p));
      }
      return out;
    }

    std::vector<real_t> u1() const {
      auto h1 = Kokkos::create_mirror_view(arr.ux1);
      Kokkos::deep_copy(h1, arr.ux1);
      return std::vector<real_t>(h1.data(), h1.data() + n);
    }

    std::vector<double> delta() const {
      auto hp = Kokkos::create_mirror_view(arr.pld_r);
      Kokkos::deep_copy(hp, arr.pld_r);
      std::vector<double> out(n);
      for (npart_t p = 0; p < n; ++p) {
        out[p] = static_cast<double>(hp(p, 0));
      }
      return out;
    }

    std::vector<double> w() const {
      auto hw = Kokkos::create_mirror_view(arr.weight);
      Kokkos::deep_copy(hw, arr.weight);
      std::vector<double> out(n);
      for (npart_t p = 0; p < n; ++p) {
        out[p] = static_cast<double>(hw(p));
      }
      return out;
    }

    double sum_wK() const {
      const auto k = K();
      const auto ww = w();
      double     s = 0.0;
      for (npart_t p = 0; p < n; ++p) {
        s += ww[p] * k[p];
      }
      return s;
    }
  };

  /**
   * @brief One launch of the kernel over all ions, returning the per-launch ledger
   */
  kh::LossLedger launch(Ions& ions, const Tables& T, const kh::HadronicParams& prm) {
    kh::LossLedger L;
    Kokkos::parallel_reduce(
      "bh_loss",
      CreateParticleRangePolicy<Dim::_1D>({ 0u }, { ions.n }),
      kh::BetheHeitler_kernel<Minkowski<Dim::_1D>> { ions.arr, ions.mu_d, T.dev, prm },
      Kokkos::Sum<kh::LossLedger>(L));
    Kokkos::fence();
    return L;
  }

  struct HostLedger {
    kh::LossLedger C {};

    void add(const kh::LossLedger& L) {
      const auto po = L.pending_outstanding;
      const auto sw = L.sum_wK;
      const auto na = L.n_alive;
      C += L;
      C.pending_outstanding = po;
      C.sum_wK              = sw;
      C.n_alive             = na;
    }
  };

  kh::HadronicParams default_params(double coeff, double dt) {
    kh::HadronicParams prm;
    prm.tau0_over_skindepth0 = coeff;
    prm.mu_r                 = MU_R;
    prm.eta_u                = 1.0e-4;
    prm.pending_floor_ulp    = 100.0;
    prm.f_max                = 0.01;
    prm.max_substeps         = 64;
    prm.dt                   = dt;
    return prm;
  }

  /**
   * @brief single-precision ulp of x, whatever the build (the T5b ratio is
   *        defined on the float resolution, R6)
   */
  double ulp_float_of(double x) {
    const float xf = static_cast<float>(x);
    return static_cast<double>(std::nextafter(xf, std::numeric_limits<float>::infinity())) -
           static_cast<double>(xf);
  }

  /**
   * @brief the kernel's pending floor at K (build precision)
   */
  double pending_floor_of(double K, double mu_d, double floor_ulp) {
    const double u  = Ions::u_of_K(K, mu_d);
    const double gd = std::sqrt(1.0 + u * u);
    return floor_ulp * kh::ulp_real(static_cast<real_t>(u)) * mu_d * u / gd;
  }

  /**
   * @brief stored-K envelope (decision (a)): |K_stored - K_eff| = |delta| <=
   *        eta_u K + pending_floor at every output time
   */
  void envelope_check(const std::string& name, const std::vector<double>& Ks,
                      const std::vector<double>& deltas, double mu_d, double eta_u, double floor_ulp) {
    double worst = 0.0;
    for (std::size_t i = 0; i < Ks.size(); ++i) {
      const double bound = eta_u * Ks[i] + pending_floor_of(Ks[i], mu_d, floor_ulp);
      worst              = std::max(worst, std::abs(deltas[i]) / bound);
    }
    verdict(worst <= 1.0, "envelope [" + name + "]: max |K_stored - K_eff| / (eta_u K + pending_floor) over the output times",
            worst, 1.0);
  }

  double dK_ulp_of(double K, double mu_d) {
    const double u  = Ions::u_of_K(K, mu_d);
    const double gd = std::sqrt(1.0 + u * u);
    return ulp_float_of(u) * mu_d * u / gd; // single-precision ulp whatever the build (R6)
  }

  /**
   * @brief T11-lite closure check on a finished run
   */
  void closure_check(const std::string& name, double sum_wK_init, const Ions& ions,
                     const HostLedger& H) {
    const double arrays = sum_wK_init - ions.sum_wK();
    const double E_drag = H.C.E_drag.value();
    const double rel    = E_drag != 0.0 ? std::abs(arrays - E_drag) / std::abs(E_drag)
                                        : std::abs(arrays - E_drag);
    verdict(rel <= 1.0e-9,
            "T11-lite closure [" + name + "]: |sum w (K_init - K_final) - E_drag| / E_drag",
            rel, 1.0e-9);
    diag("  " + name + ": arrays " + std::to_string(arrays) + " E_drag " +
         std::to_string(E_drag) + " E_lost_nopairs " + std::to_string(H.C.E_lost_nopairs.value()));
  }

  /* ------------------------------------------------------------------------ */
  /* device tables (T3-like)                                                    */
  /* ------------------------------------------------------------------------ */
  void test_device_tables(const Tables& T) {
    std::printf("---- device tables (lookup vs at_gamma; cell midpoints) ----\n");
    const int           N = 1000;
    std::vector<double> gammas(N);
    std::uint64_t       s = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < N; ++i) {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      const double r = static_cast<double>(s >> 11) * 0x1.0p-53;
      // log-uniform in Gamma - Gamma_th between the edges
      const double y = T.host.y_min + r * (T.host.y_max - T.host.y_min);
      gammas[i]      = T.host.gamma_th + std::pow(10.0, y);
    }
    array_t<double*> g_dev { "g", N }, I_dev { "I", N }, J_dev { "J", N };
    array_t<int*>    above_dev { "above", N };
    {
      auto h = Kokkos::create_mirror_view(g_dev);
      for (int i = 0; i < N; ++i) {
        h(i) = gammas[i];
      }
      Kokkos::deep_copy(g_dev, h);
    }
    auto tab = T.dev;
    Kokkos::parallel_for(
      "lookup", N, Lambda(const int i) {
        bool a = false;
        I_dev(i)     = tab.I(g_dev(i), a);
        above_dev(i) = a ? 1 : 0;
        J_dev(i)     = tab.J(g_dev(i));
      });
    Kokkos::fence();
    auto hI = Kokkos::create_mirror_view(I_dev);
    auto hJ = Kokkos::create_mirror_view(J_dev);
    auto ha = Kokkos::create_mirror_view(above_dev);
    Kokkos::deep_copy(hI, I_dev);
    Kokkos::deep_copy(hJ, J_dev);
    Kokkos::deep_copy(ha, above_dev);
    double max_rel_I = 0.0, max_rel_J = 0.0;
    int    n_checked = 0, n_above = 0;
    for (int i = 0; i < N; ++i) {
      const auto ex = T.rates->at_gamma(gammas[i]);
      n_above += ha(i);
      if (ex.I > 1.0e-3 * T.host.I_peak) {
        max_rel_I = std::max(max_rel_I, std::abs(hI(i) / ex.I - 1.0));
        max_rel_J = std::max(max_rel_J, std::abs(hJ(i) / ex.J - 1.0));
        ++n_checked;
      }
    }
    diag("device lookup: " + std::to_string(n_checked) + " of 1000 random Gamma with I > 1e-3 I_peak, " +
         std::to_string(n_above) + " flagged above (expected 0)");
    verdict(max_rel_I <= 1.0e-2, "device lookup I vs at_gamma, 1000 random Gamma (I > 1e-3 peak)",
            max_rel_I, 1.0e-2);
    verdict(max_rel_J <= 1.0e-2, "device lookup J vs at_gamma, 1000 random Gamma (I > 1e-3 peak)",
            max_rel_J, 1.0e-2);
    verdict(n_above == 0, "device lookup: none of the in-range Gamma flagged above", n_above, 0);

    // cell midpoints
    const int M = static_cast<int>(T.host.size()) - 1;
    array_t<double*> gm_dev { "gm", M }, Im_dev { "Im", M };
    {
      auto h = Kokkos::create_mirror_view(gm_dev);
      for (int i = 0; i < M; ++i) {
        const double ym = 0.5 * (T.host.y[i] + T.host.y[i + 1]);
        h(i)            = T.host.gamma_th + std::pow(10.0, ym);
      }
      Kokkos::deep_copy(gm_dev, h);
    }
    Kokkos::parallel_for(
      "lookup_mid", M, Lambda(const int i) { Im_dev(i) = tab.I(gm_dev(i)); });
    Kokkos::fence();
    auto hIm = Kokkos::create_mirror_view(Im_dev);
    auto hgm = Kokkos::create_mirror_view(gm_dev);
    Kokkos::deep_copy(hIm, Im_dev);
    Kokkos::deep_copy(hgm, gm_dev);
    double max_mid = 0.0, max_mid_vs_host = 0.0;
    for (int i = 0; i < M; ++i) {
      const auto ex = T.rates->at_gamma(hgm(i));
      if (ex.I > 1.0e-3 * T.host.I_peak) {
        max_mid = std::max(max_mid, std::abs(hIm(i) / ex.I - 1.0));
      }
      const double host_interp = bh::interpolate_loss_host(T.host, hgm(i));
      max_mid_vs_host = std::max(max_mid_vs_host, std::abs(hIm(i) / host_interp - 1.0));
    }
    verdict(max_mid <= 1.8e-4 + 1.0e-6,
            "device interpolation error of I at cell midpoints (I > 1e-3 peak)", max_mid,
            1.8e-4 + 1.0e-6);
    verdict(max_mid_vs_host <= 1.0e-12,
            "device vs host interpolation at cell midpoints (same arithmetic)", max_mid_vs_host,
            1.0e-12);

    // edge conventions
    array_t<double*> e_dev { "e", 4 };
    array_t<int*>    ea_dev { "ea", 4 };
    Kokkos::parallel_for(
      "edges", 1, Lambda(const int) {
        bool a;
        e_dev(0)  = tab.I(tab.gamma_min * (1.0 - 1.0e-9), a);
        ea_dev(0) = a;
        e_dev(1)  = tab.I(tab.gamma_th, a);
        ea_dev(1) = a;
        e_dev(2)  = tab.I(tab.gamma_max * 1.5, a);
        ea_dev(2) = a;
        e_dev(3)  = tab.I(tab.gamma_max, a);
        ea_dev(3) = a;
      });
    Kokkos::fence();
    auto he  = Kokkos::create_mirror_view(e_dev);
    auto hea = Kokkos::create_mirror_view(ea_dev);
    Kokkos::deep_copy(he, e_dev);
    Kokkos::deep_copy(hea, ea_dev);
    verdict(he(0) == 0.0 && he(1) == 0.0, "device lookup returns 0 below the lower edge",
            he(0) + he(1), 0.0);
    verdict(he(2) == T.host.I.back() && hea(2) == 1,
            "device lookup returns the edge value above the upper edge (flagged)",
            std::abs(he(2) / T.host.I.back() - 1.0), 0.0);
    verdict(he(3) == T.host.I.back() && hea(3) == 0,
            "device lookup at Gamma_max returns the last node (not flagged)",
            std::abs(he(3) / T.host.I.back() - 1.0), 0.0);
  }

  /* ------------------------------------------------------------------------ */
  /* T9 threshold                                                               */
  /* ------------------------------------------------------------------------ */
  void test_threshold(const Tables& T) {
    std::printf("---- T9 threshold ----\n");
    const double gth = 1.0 / EPS0 + EPS0 / 4.0;
    verdict(std::abs(T.host.gamma_th - gth) <= 1.0e-12,
            "T9: table gamma_th vs 1/eps0 + eps0/4", std::abs(T.host.gamma_th - gth), 1.0e-12);
    const double mu_d = MU_R; // Gamma_p = gamma_d
    Ions ions(4, mu_d);
    const double G_below = T.host.gamma_min * (1.0 - 1.0e-6);
    const double G_above = T.host.gamma_min * (1.0 + 1.0e-3);
    ions.set(0, mu_d * (G_below - 1.0));
    ions.set(1, mu_d * (G_below - 1.0), 0.0, 1.0, 0.0);
    ions.set(2, mu_d * (G_above - 1.0));
    ions.set(3, mu_d * (G_above - 1.0), 0.0, 0.0, 1.0);
    // tau0 such that dK/K = 1e-3 per step just above the edge
    const double I_above = bh::interpolate_loss_host(T.host, G_above);
    const double K_above = mu_d * (G_above - 1.0);
    const double dt      = 1.0;
    const double coeff   = 1.0e-3 * K_above / (I_above * dt);
    auto         prm     = default_params(coeff, dt);
    const auto   u_init  = ions.u1();
    const auto   K_init  = ions.K();
    HostLedger   H;
    for (int s = 0; s < 1000; ++s) {
      H.add(launch(ions, T, prm));
    }
    const auto u_fin = ions.u1();
    const auto K_fin = ions.K();
    const auto d     = ions.delta();
    verdict(u_fin[0] == u_init[0] && K_fin[1] == K_init[1] && d[0] == 0.0 && d[1] == 0.0,
            "T9: ions at Gamma_min (1 - 1e-6): u unchanged over 1000 steps (bitwise), delta = 0",
            std::abs(K_fin[0] - K_init[0]) + std::abs(K_fin[1] - K_init[1]), 0.0);
    verdict(K_fin[2] < K_init[2] && K_fin[3] < K_init[3],
            "T9: ions at Gamma_min (1 + 1e-3): K decreased over 1000 steps",
            (K_init[2] - K_fin[2]) / K_init[2], 0.0);
    diag("T9: Gamma_min = " + std::to_string(T.host.gamma_min) + ", below-edge K = " +
         std::to_string(K_init[0]) + ", above-edge K " + std::to_string(K_init[2]) + " -> " +
         std::to_string(K_fin[2]) + " (dK/K per step at start " +
         std::to_string(coeff * I_above * dt / K_above) + ")");
    // the two in-range ions cool to the floor K_floor = mu_r (Gamma_min - 1) in a
    // few steps (dK/K = 1e-3 per step) and then sit at the edge; n_processed
    // counts only their in-range steps, the below-edge ions never
    const double K_floor = MU_R * (T.host.gamma_min - 1.0);
    verdict(H.C.n_processed >= 2 && H.C.n_processed <= 2000,
            "T9: n_processed counts in-range ion-steps only (2 <= n <= 2000)",
            static_cast<double>(H.C.n_processed), 2000);
    verdict(std::abs(K_fin[2] - K_floor) <= 2.0 * dK_ulp_of(K_floor, mu_d) &&
              std::abs(K_fin[3] - K_floor) <= 2.0 * dK_ulp_of(K_floor, mu_d),
            "T9: the in-range ions stop at K_floor = mu_r (Gamma_min - 1) within 2 dK_ulp",
            std::abs(K_fin[2] - K_floor), 2.0 * dK_ulp_of(K_floor, mu_d));
    closure_check("T9", K_init[0] + K_init[1] + K_init[2] + K_init[3], ions, H);
  }

  /* ------------------------------------------------------------------------ */
  /* T5 cooling curve and dt convergence                                        */
  /* ------------------------------------------------------------------------ */
  struct CurveResult {
    std::vector<double> times, K6, K3, Keff6, Keff3, d6, d3;
    HostLedger          H;
    double              sum_wK_init { 0.0 };
    int                 n_steps { 0 };
  };

  /**
   * @brief Run two groups (Gamma_p = 6 and 3, n_each ions each) for n_steps,
   *        sampling every `every` steps
   */
  CurveResult run_curve(const Tables& T, double mu_d, double coeff, double dt, int n_steps,
                        int every, double eta_u = 1.0e-4, double floor_ulp = 100.0,
                        double G_hi = 6.0, double G_lo = 3.0, int n_each = 2) {
    CurveResult R;
    Ions        ions(2 * n_each, mu_d);
    for (int i = 0; i < n_each; ++i) {
      ions.set(i, MU_R * (G_hi - 1.0));
      ions.set(n_each + i, MU_R * (G_lo - 1.0));
    }
    R.sum_wK_init = ions.sum_wK();
    auto prm      = default_params(coeff, dt);
    prm.eta_u             = eta_u;
    prm.pending_floor_ulp = floor_ulp;
    for (int s = 1; s <= n_steps; ++s) {
      R.H.add(launch(ions, T, prm));
      if (s % every == 0 || s == n_steps) {
        const auto K = ions.K();
        const auto d = ions.delta();
        R.times.push_back(s * dt);
        R.K6.push_back(K[0]);
        R.K3.push_back(K[n_each]);
        R.Keff6.push_back(K[0] - d[0]);
        R.Keff3.push_back(K[n_each] - d[n_each]);
        R.d6.push_back(d[0]);
        R.d3.push_back(d[n_each]);
      }
    }
    R.n_steps = n_steps;
    // closure uses the ions object: recompute here before it dies
    const double arrays = R.sum_wK_init - ions.sum_wK();
    R.H.C.E_round       = kh::dd { arrays }; // stash the array-based total in an unused (zero) field
    return R;
  }

  double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      m = std::max(m, std::abs(a[i] - b[i]) / std::abs(b[i]));
    }
    return m;
  }

  /* the common setup of the two T5 groups (base curve and dt-convergence): dt
   * from dK/K = 1e-4 per step at Gamma_p = 6, and the step count that takes the
   * Gamma_p = 6 ions to Gamma_p = 3 (n_steps = 0 if they never get there) */
  struct T5Setup {
    double dt { 0.0 };
    int    n_steps { 0 };
  };

  T5Setup t5_setup(const Tables& T, double mu_d, double coeff) {
    const double K6 = MU_R * 5.0;
    const double I6 = bh::interpolate_loss_host(T.host, 6.0);
    T5Setup      S;
    S.dt = 1.0e-4 * K6 / (coeff * I6); // dK/K = 1e-4 per step at Gamma_p = 6
    diag("T5: dt = " + std::to_string(S.dt) + ", dK/K per step at Gamma_p = 6 = " +
         std::to_string(coeff * I6 * S.dt / K6) + ", I(6) table = " + std::to_string(I6) +
         ", I(6) exact = " + std::to_string(T.rates->at_gamma(6.0).I));
    // run until the Gamma_p = 6 ions reach Gamma_p = 3: find the step count first
    Ions   probe(1, mu_d);
    probe.set(0, K6);
    auto prm = default_params(coeff, S.dt);
    const double K_target = MU_R * 2.0;
    for (int s = 1; s <= 10000000; ++s) {
      launch(probe, T, prm);
      if (s % 64 == 0 || s < 64) {
        if (probe.K()[0] <= K_target) {
          S.n_steps = s;
          break;
        }
      }
    }
    return S;
  }

  void test_cooling_T5(const Tables& T) {
    std::printf("---- T5 cooling curve at physical mass (mu_d = mu_r) ----\n");
    const double mu_d  = MU_R;
    const double K6    = MU_R * 5.0;
    const double coeff = 1.0; // tau0 = 1, skindepth0 = 1
    const auto   S     = t5_setup(T, mu_d, coeff);
    const double dt    = S.dt;
    const int    n_steps = S.n_steps;
    auto exact = [&](double K) { return -coeff * exact_rate_K(T, MU_R, K); };
    auto table = [&](double K) { return -coeff * table_rate_K(T, MU_R, K); };
    if (n_steps == 0) {
      verdict(false, "T5: the Gamma_p = 6 ions never reached Gamma_p = 3", 0.0, 0.0);
      return;
    }
    const int every = std::max(1, n_steps / 200);
    diag("T5: Gamma_p 6 -> 3 in " + std::to_string(n_steps) + " steps (t_end = " +
         std::to_string(n_steps * dt) + "), sampled every " + std::to_string(every) + " steps");
    const auto R = run_curve(T, mu_d, coeff, dt, n_steps, every);
    const double h_ref = dt / 8.0;
    const auto ref6 = rk4_curve(K6, R.times, h_ref, exact);
    const auto ref3 = rk4_curve(MU_R * 2.0, R.times, h_ref, exact);
    const auto tab6 = rk4_curve(K6, R.times, h_ref, table);
    const auto tab3 = rk4_curve(MU_R * 2.0, R.times, h_ref, table);
    const double e6 = max_rel(R.K6, ref6), e3 = max_rel(R.K3, ref3);
    verdict(e6 <= 1.0e-3, "T5: max relative error of K(t), Gamma_p = 6 group, vs exact RK4", e6, 1.0e-3);
    verdict(e3 <= 1.0e-3, "T5: max relative error of K(t), Gamma_p = 3 group, vs exact RK4", e3, 1.0e-3);
    diag("T5: same with K_eff = K - delta: " + std::to_string(max_rel(R.Keff6, ref6)) + " (6), " +
         std::to_string(max_rel(R.Keff3, ref3)) + " (3)");
    diag("T5: vs the interpolated-table RK4: K " + std::to_string(max_rel(R.K6, tab6)) + " (6), " +
         std::to_string(max_rel(R.K3, tab3)) + " (3); K_eff " + std::to_string(max_rel(R.Keff6, tab6)) +
         " (6), " + std::to_string(max_rel(R.Keff3, tab3)) + " (3)");
    diag("T5: table-interpolation contribution (exact RK4 vs table RK4 at the output times): " +
         std::to_string(max_rel(tab6, ref6)) + " (6), " + std::to_string(max_rel(tab3, ref3)) + " (3)");
    diag("T5: end values: K_sim " + std::to_string(R.K6.back()) + " ref " + std::to_string(ref6.back()) +
         " (6); K_sim " + std::to_string(R.K3.back()) + " ref " + std::to_string(ref3.back()) + " (3); " +
         "applications " + std::to_string(R.H.C.n_applied) + ", n_substeps " +
         std::to_string(R.H.C.n_substeps));
    {
      const double rel = std::abs(R.H.C.E_round.value() - R.H.C.E_drag.value()) / R.H.C.E_drag.value();
      verdict(rel <= 1.0e-9, "T11-lite closure [T5 base]: |sum w (K_init - K_final) - E_drag| / E_drag",
              rel, 1.0e-9);
    }
    envelope_check("T5 base, Gamma_p = 6 group", R.K6, R.d6, mu_d, 1.0e-4, 100.0);
    envelope_check("T5 base, Gamma_p = 3 group", R.K3, R.d3, mu_d, 1.0e-4, 100.0);
  }

  /* the dt-convergence part of T5 (its own ctest entry, `t5fine`, since the
   * three resolutions with eta_u scaled are the long part of the group) */
  void test_dt_convergence_T5fine(const Tables& T) {
    // dt convergence: same end time, dt, dt/2, dt/4; error at the end time
    std::printf("---- T5 dt-convergence (end-time error, three resolutions) ----\n");
    const double mu_d  = MU_R;
    const double K6    = MU_R * 5.0;
    const double coeff = 1.0; // tau0 = 1, skindepth0 = 1
    const auto   S     = t5_setup(T, mu_d, coeff);
    const double dt    = S.dt;
    const int    n_steps = S.n_steps;
    auto exact = [&](double K) { return -coeff * exact_rate_K(T, MU_R, K); };
    auto table = [&](double K) { return -coeff * table_rate_K(T, MU_R, K); };
    if (n_steps == 0) {
      verdict(false, "T5 conv: the Gamma_p = 6 ions never reached Gamma_p = 3", 0.0, 0.0);
      return;
    }
    const double h_ref = dt / 8.0;
    const double t_end = n_steps * dt;
    const double ref6_end = rk4(K6, t_end, h_ref / 4.0, exact);
    const double ref3_end = rk4(MU_R * 2.0, t_end, h_ref / 4.0, exact);
    const double tab6_end = rk4(K6, t_end, h_ref / 4.0, table);
    const double tab3_end = rk4(MU_R * 2.0, t_end, h_ref / 4.0, table);
    struct Row {
      double dt, eK6_exact, eK3_exact, eKeff6_exact, eKeff3_exact, eK6_tab, eKeff6_tab, eK3_tab, eKeff3_tab;
    };
    std::vector<Row> rows, rows_scaled;
    std::vector<double> floor_ratio6, floor_ratio3;
    for (int r = 0; r < 3; ++r) {
      const int    f    = 1 << r;
      const double dtr  = dt / f;
      const auto   Rr   = run_curve(T, mu_d, coeff, dtr, n_steps * f, n_steps * f);
      rows.push_back(Row { dtr, std::abs(Rr.K6.back() / ref6_end - 1.0),
                           std::abs(Rr.K3.back() / ref3_end - 1.0),
                           std::abs(Rr.Keff6.back() / ref6_end - 1.0),
                           std::abs(Rr.Keff3.back() / ref3_end - 1.0),
                           std::abs(Rr.K6.back() / tab6_end - 1.0),
                           std::abs(Rr.Keff6.back() / tab6_end - 1.0),
                           std::abs(Rr.K3.back() / tab3_end - 1.0),
                           std::abs(Rr.Keff3.back() / tab3_end - 1.0) });
      // registered (decision (a)): eta_u scaled with dt, 1e-4 at the base dt
      const double eta_r = 1.0e-4 / f;
      const auto   Rs    = run_curve(T, mu_d, coeff, dtr, n_steps * f, n_steps * f, eta_r);
      floor_ratio6.push_back(eta_r * Rs.K6.back() / pending_floor_of(Rs.K6.back(), mu_d, 100.0));
      floor_ratio3.push_back(eta_r * Rs.K3.back() / pending_floor_of(Rs.K3.back(), mu_d, 100.0));
      envelope_check("T5 conv eta_u scaled, dt/" + std::to_string(f) + ", Gamma_p = 6 group", Rs.K6, Rs.d6, mu_d, eta_r, 100.0);
      envelope_check("T5 conv eta_u scaled, dt/" + std::to_string(f) + ", Gamma_p = 3 group", Rs.K3, Rs.d3, mu_d, eta_r, 100.0);
      rows_scaled.push_back(Row { dtr, std::abs(Rs.K6.back() / ref6_end - 1.0),
                                  std::abs(Rs.K3.back() / ref3_end - 1.0),
                                  std::abs(Rs.Keff6.back() / ref6_end - 1.0),
                                  std::abs(Rs.Keff3.back() / ref3_end - 1.0),
                                  std::abs(Rs.K6.back() / tab6_end - 1.0),
                                  std::abs(Rs.Keff6.back() / tab6_end - 1.0),
                                  std::abs(Rs.K3.back() / tab3_end - 1.0),
                                  std::abs(Rs.Keff3.back() / tab3_end - 1.0) });
    }
    auto order = [](double a, double b) { return std::log2(a / b); };
    for (const auto& r : rows) {
      std::printf("(diag) T5 conv: dt = %.6e | K6 exact %.3e Keff6 exact %.3e K6 table %.3e Keff6 table %.3e "
                  "| K3 exact %.3e Keff3 exact %.3e K3 table %.3e Keff3 table %.3e\n",
                  r.dt, r.eK6_exact, r.eKeff6_exact, r.eK6_tab, r.eKeff6_tab, r.eK3_exact, r.eKeff3_exact,
                  r.eK3_tab, r.eKeff3_tab);
    }
    const double o1 = order(rows[0].eK6_exact, rows[1].eK6_exact);
    const double o2 = order(rows[1].eK6_exact, rows[2].eK6_exact);
    diag("T5 conv (round-1 registration, fixed eta_u, kept as history): order of K_sim vs exact RK4, Gamma_p = 6 "
         "group: " + std::to_string(o1) + ", " + std::to_string(o2) + " (round-1 criterion 1.0 +- 0.3: " +
         ((std::abs(o1 - 1.0) <= 0.3 && std::abs(o2 - 1.0) <= 0.3) ? "would pass" : "would fail") + ")");
    diag("T5 conv orders (6): K exact " + std::to_string(o1) + ", " + std::to_string(o2) +
         "; Keff exact " + std::to_string(order(rows[0].eKeff6_exact, rows[1].eKeff6_exact)) + ", " +
         std::to_string(order(rows[1].eKeff6_exact, rows[2].eKeff6_exact)) + "; K table " +
         std::to_string(order(rows[0].eK6_tab, rows[1].eK6_tab)) + ", " +
         std::to_string(order(rows[1].eK6_tab, rows[2].eK6_tab)) + "; Keff table " +
         std::to_string(order(rows[0].eKeff6_tab, rows[1].eKeff6_tab)) + ", " +
         std::to_string(order(rows[1].eKeff6_tab, rows[2].eKeff6_tab)));
    diag("T5 conv orders (3): K exact " + std::to_string(order(rows[0].eK3_exact, rows[1].eK3_exact)) +
         ", " + std::to_string(order(rows[1].eK3_exact, rows[2].eK3_exact)) + "; Keff table " +
         std::to_string(order(rows[0].eKeff3_tab, rows[1].eKeff3_tab)) + ", " +
         std::to_string(order(rows[1].eKeff3_tab, rows[2].eKeff3_tab)));
    for (const auto& r : rows_scaled) {
      std::printf("(diag) T5 conv (eta_u scaled with dt): dt = %.6e | K6 exact %.3e Keff6 exact %.3e "
                  "K6 table %.3e Keff6 table %.3e | K3 exact %.3e Keff3 table %.3e\n",
                  r.dt, r.eK6_exact, r.eKeff6_exact, r.eK6_tab, r.eKeff6_tab, r.eK3_exact, r.eKeff3_tab);
    }
    diag("T5 conv orders with eta_u scaled (6): K exact " +
         std::to_string(order(rows_scaled[0].eK6_exact, rows_scaled[1].eK6_exact)) + ", " +
         std::to_string(order(rows_scaled[1].eK6_exact, rows_scaled[2].eK6_exact)) + "; Keff table " +
         std::to_string(order(rows_scaled[0].eKeff6_tab, rows_scaled[1].eKeff6_tab)) + ", " +
         std::to_string(order(rows_scaled[1].eKeff6_tab, rows_scaled[2].eKeff6_tab)));
    for (int r = 0; r < 3; ++r) {
      std::printf("(diag) T5 conv eta_u scaled: dt/%d: eta_u K / pending_floor at the end = %.3f (6), %.3f (3)\n",
                  1 << r, floor_ratio6[r], floor_ratio3[r]);
    }
    {
      const double s1 = order(rows_scaled[0].eKeff6_tab, rows_scaled[1].eKeff6_tab);
      const double s2 = order(rows_scaled[1].eKeff6_tab, rows_scaled[2].eKeff6_tab);
      const double t1 = order(rows_scaled[0].eKeff3_tab, rows_scaled[1].eKeff3_tab);
      const double t2 = order(rows_scaled[1].eKeff3_tab, rows_scaled[2].eKeff3_tab);
      const bool   floor_ok = *std::min_element(floor_ratio6.begin(), floor_ratio6.end()) > 1.0 &&
                            *std::min_element(floor_ratio3.begin(), floor_ratio3.end()) > 1.0;
      verdict(floor_ok, "T5 (re-registered): eta_u K above the pending floor at every dt used (min ratio)",
              std::min(*std::min_element(floor_ratio6.begin(), floor_ratio6.end()),
                       *std::min_element(floor_ratio3.begin(), floor_ratio3.end())), 1.0);
      verdict(std::abs(s1 - 1.0) <= 0.3 && std::abs(s2 - 1.0) <= 0.3,
              "T5 (re-registered): dt-convergence order of K_eff vs the interpolated-table RK4 with eta_u "
              "scaled as dt, Gamma_p = 6 group (both ratios within 1.0 +- 0.3); value = first ratio", s1, 0.3);
      verdict(std::abs(t1 - 1.0) <= 0.3 && std::abs(t2 - 1.0) <= 0.3,
              "T5 (re-registered): same for the Gamma_p = 3 group; value = first ratio", t1, 0.3);
      diag("T5 (re-registered) orders: (6) " + std::to_string(s1) + ", " + std::to_string(s2) + "; (3) " +
           std::to_string(t1) + ", " + std::to_string(t2));
    }
  }

  /* ------------------------------------------------------------------------ */
  /* T5b single-build part: ulp-ratio scan                                      */
  /* ------------------------------------------------------------------------ */
  void test_ulp_scan_T5b(const Tables& T) {
    std::printf("---- T5b single-build part (dK/dK_ulp scan at Gamma_p = 3, design on / off) ----\n");
    std::printf("(diag) real_t = %s; ratio = dK / dK_ulp, dK_ulp = ulp(|u|) mu_d u / gamma_d\n",
                sizeof(real_t) == 4 ? "float" : "double");
    const double ratios[] = { 0.3, 1.0, 3.0, 10.0, 100.0 };
    const double dt       = 1.0;
    for (const double mu_d : { 1.0, MU_R }) {
      const double K0 = MU_R * 2.0; // Gamma_p = 3
      const double I3 = bh::interpolate_loss_host(T.host, 3.0);
      const double dK_ulp = dK_ulp_of(K0, mu_d);
      auto exact = [&](double K) { return -1.0 * exact_rate_K(T, MU_R, K); };
      for (const double ratio : ratios) {
        const double coeff = ratio * dK_ulp / (I3 * dt); // tau0/skindepth0
        // steps until K has dropped by 1% (from the reference)
        const double K_target = 0.99 * K0;
        int          n_steps  = 0;
        {
          double K = K0, t = 0.0;
          const double h = dt;
          // the loss is tiny per step: integrate with RK4 in steps of dt, count
          while (K > K_target && n_steps < 200000000) {
            K = rk4(K, h, h, [&](double KK) { return coeff * exact(KK); });
            t += h;
            ++n_steps;
          }
        }
        const int n_steps_capped = std::min(n_steps, 400000);
        for (const bool design_on : { true, false }) {
          Ions ions(1, mu_d);
          ions.set(0, K0);
          auto prm = default_params(coeff, dt);
          if (!design_on) {
            prm.eta_u              = 0.0;
            prm.pending_floor_ulp  = 0.0;
            prm.unsafe_application = true;
          }
          HostLedger H;
          const double sum0 = ions.sum_wK();
          for (int s = 0; s < n_steps_capped; ++s) {
            H.add(launch(ions, T, prm));
          }
          const double K_sim = ions.K()[0];
          const double d     = ions.delta()[0];
          const double K_ref = rk4(K0, n_steps_capped * dt, dt, [&](double KK) { return coeff * exact(KK); });
          const double achieved = coeff * I3 * dt / dK_ulp;
          const double rel      = std::abs(K_sim / K_ref - 1.0);
          const double rel_eff  = std::abs((K_sim - d) / K_ref - 1.0);
          const double realised = (K0 - K_sim) / (K0 - K_ref);
          std::printf("(diag) T5b mu_d = %.6g ratio %.3g (achieved %.4g) design %s: steps %d%s, K_sim/K_ref - 1 = %+.3e, "
                      "K_eff/K_ref - 1 = %+.3e, realised/expected loss = %.4f, applications %llu\n",
                      mu_d, ratio, achieved, design_on ? "on " : "off", n_steps_capped,
                      n_steps_capped < n_steps ? " (capped)" : "", K_sim / K_ref - 1.0,
                      (K_sim - d) / K_ref - 1.0, realised, H.C.n_applied);
          if (design_on) {
            // the pending lag is at most eta_u K = 1e-4 K on the stored value: report
            // K_eff against the reference (the cross-build pass criterion is in the bash test)
            verdict(rel_eff <= 1.0e-5,
                    "T5b (single build) design on, mu_d " + std::to_string(mu_d) + " ratio " +
                      std::to_string(ratio) + ": |K_eff/K_ref - 1| (lag removed)",
                    rel_eff, 1.0e-5);
            verdict(rel <= 1.0e-4,
                    "T5b (single build) design on, mu_d " + std::to_string(mu_d) + " ratio " +
                      std::to_string(ratio) + ": |K_sim/K_ref - 1| (stored value, lag <= eta_u K)",
                    rel, 1.0e-4);
            closure_check("T5b on mu_d " + std::to_string(mu_d) + " ratio " + std::to_string(ratio), sum0, ions, H);
          } else {
            verdict(rel <= 1.0e-4,
                    "T5b (single build) design off, mu_d " + std::to_string(mu_d) + " ratio " +
                      std::to_string(ratio) + ": |K_sim/K_ref - 1|",
                    rel, 1.0e-4, true);
            closure_check("T5b off mu_d " + std::to_string(mu_d) + " ratio " + std::to_string(ratio), sum0, ions, H);
          }
        }
      }
    }
  }

  /* ------------------------------------------------------------------------ */
  /* T6 mu_d independence                                                       */
  /* ------------------------------------------------------------------------ */
  void test_mu_independence_T6(const Tables& T) {
    std::printf("---- T6 mu_d independence ----\n");
    const double K0    = MU_R * 5.0;
    const double coeff = 1.0;
    const double I6    = bh::interpolate_loss_host(T.host, 6.0);
    const double dt    = 3.7e-5 * K0 / (coeff * I6); // dK/K = 3.7e-5 per step (incommensurate with eta_u)
    const int    n_steps = 40000, every = 400;
    diag("T6: dt = " + std::to_string(dt) + ", dK/K per step = 3.7e-5, " + std::to_string(n_steps) +
         " steps, sampled every " + std::to_string(every));
    const double mus[] = { 1.0, 25.0, MU_R };
    std::vector<std::vector<double>> K(3), Keff(3);
    for (int m = 0; m < 3; ++m) {
      Ions ions(1, mus[m]);
      ions.set(0, K0);
      auto         prm  = default_params(coeff, dt);
      HostLedger   H;
      const double sum0 = ions.sum_wK();
      for (int s = 1; s <= n_steps; ++s) {
        H.add(launch(ions, T, prm));
        if (s % every == 0) {
          const auto k = ions.K()[0];
          const auto d = ions.delta()[0];
          K[m].push_back(k);
          Keff[m].push_back(k - d);
        }
      }
      diag("T6: mu_d = " + std::to_string(mus[m]) + ": final K = " + std::to_string(K[m].back()) +
           ", Gamma_p = " + std::to_string(1.0 + K[m].back() / MU_R) + ", applications " +
           std::to_string(H.C.n_applied));
      closure_check("T6 mu_d " + std::to_string(mus[m]), sum0, ions, H);
      {
        std::vector<double> dl(K[m].size());
        for (std::size_t i = 0; i < dl.size(); ++i) {
          dl[i] = K[m][i] - Keff[m][i];
        }
        envelope_check("T6 mu_d " + std::to_string(mus[m]), K[m], dl, mus[m], 1.0e-4, 100.0);
      }
    }
    const double e01 = max_rel(K[0], K[1]), e02 = max_rel(K[0], K[2]), e12 = max_rel(K[1], K[2]);
    const double f01 = max_rel(Keff[0], Keff[1]), f02 = max_rel(Keff[0], Keff[2]), f12 = max_rel(Keff[1], Keff[2]);
    verdict(std::max({ f01, f02, f12 }) <= 1.0e-6,
            "T6 (re-registered): pairwise max relative difference of K_eff(t) over mu_d in {1, 25, 1836.15}",
            std::max({ f01, f02, f12 }), 1.0e-6);
    std::printf("(diag) T6 (round-1 registration, stored K, kept as history): pairwise 1-25 %.3e, 1-1836 %.3e, "
                "25-1836 %.3e (round-1 criterion 1e-5: %s); K_eff pairwise %.3e, %.3e, %.3e\n",
                e01, e02, e12, std::max({ e01, e02, e12 }) <= 1.0e-5 ? "would pass" : "would fail", f01, f02, f12);
  }

  /* ------------------------------------------------------------------------ */
  /* sub-cycling and above-table                                                */
  /* ------------------------------------------------------------------------ */
  void test_subcycling(const Tables& T) {
    std::printf("---- sub-cycling (dK = 0.3 K per step at Gamma_p = 6, f_max = 0.01 (default); f_max = 0.1 as diag) ----\n");
    const double mu_d  = MU_R;
    const double K0    = MU_R * 5.0;
    const double I6    = bh::interpolate_loss_host(T.host, 6.0);
    const double dt    = 1.0;
    const double coeff = 0.3 * K0 / (I6 * dt);
    auto exact = [&](double K) { return -coeff * exact_rate_K(T, MU_R, K); };
    for (const int max_sub : { 64, 2, -1 }) { // -1: round-1 diagnostic with f_max = 0.1
      Ions ions(2, mu_d);
      ions.set(0, K0);
      ions.set(1, K0, 0.0, 1.0, 0.0);
      auto prm         = default_params(coeff, dt); // f_max = 0.01 (decision (b))
      prm.max_substeps = max_sub > 0 ? max_sub : 64;
      if (max_sub < 0) {
        prm.f_max = 0.1;
      }
      HostLedger          H;
      const double        sum0 = ions.sum_wK();
      std::vector<double> times, Ks;
      const int           n_steps = 60;
      for (int s = 1; s <= n_steps; ++s) {
        H.add(launch(ions, T, prm));
        times.push_back(s * dt);
        Ks.push_back(ions.K()[0]);
      }
      const auto   ref = rk4_curve(K0, times, dt / 256.0, exact);
      const double e   = max_rel(Ks, ref);
      if (max_sub == 64) {
        verdict(H.C.n_substeps > 0, "sub-cycling: n_substeps > 0", static_cast<double>(H.C.n_substeps), 0.0);
        verdict(H.C.n_bounded == 0 && H.C.n_capped_loss == 0,
                "sub-cycling: n_bounded = 0 and n_capped_loss = 0 with max_substeps = 64",
                static_cast<double>(H.C.n_bounded + H.C.n_capped_loss), 0.0);
        verdict(e <= 1.0e-2, "sub-cycling (f_max = 0.01): max relative error of K(t) vs exact RK4 over 60 steps", e, 1.0e-2);
        diag("sub-cycling: first-step K " + std::to_string(Ks[0]) + " ref " + std::to_string(ref[0]) +
             " (Gamma_p " + std::to_string(1.0 + Ks[0] / MU_R) + "), final K " + std::to_string(Ks.back()) +
             " ref " + std::to_string(ref.back()) + ", n_substeps " + std::to_string(H.C.n_substeps));
      } else if (max_sub < 0) {
        diag("sub-cycling (round-1 registration, f_max = 0.1, kept as history): max relative error vs exact RK4 " +
             std::to_string(e) + " (round-1 criterion 1e-2: " + (e <= 1.0e-2 ? "would pass" : "would fail") +
             "), n_substeps " + std::to_string(H.C.n_substeps));
      } else {
        verdict(H.C.n_bounded > 0, "sub-cycling: n_bounded > 0 with max_substeps = 2",
                static_cast<double>(H.C.n_bounded), 0.0);
        verdict(H.C.n_capped_loss > 0, "sub-cycling: n_capped_loss > 0 with max_substeps = 2 (decision (c))",
                static_cast<double>(H.C.n_capped_loss), 0.0);
        verdict(std::isfinite(Ks.back()) && Ks.back() > 0.0 && Ks.back() < K0,
                "sub-cycling: the run continues with max_substeps = 2 (finite, decreasing K)", Ks.back(), 0.0);
        diag("sub-cycling (max_substeps = 2): max relative error vs exact RK4 " + std::to_string(e) +
             " (clamped loss, reported only), n_bounded " + std::to_string(H.C.n_bounded));
      }
      closure_check("sub-cycling max_substeps " + std::to_string(max_sub), sum0, ions, H);
    }
  }

  /**
   * @brief R2 regression: substep requests beyond int range and non-finite
   *        requests are capped without any undefined conversion
   */
  void test_overflow_requests(const Tables& T) {
    std::printf("---- R2: overflowing and non-finite loss requests ----\n");
    const double mu_d = MU_R;
    const double K0   = MU_R * 5.0;
    struct Case {
      const char* name;
      double      coeff, dt;
    };
    for (const Case c : { Case { "coefficient 1e20 (3.6e17 substeps requested)", 1.0e20, 1.0 },
                          Case { "non-finite request (coefficient 1e308, dt 1e10)", 1.0e308, 1.0e10 } }) {
      Ions ions(1, mu_d);
      ions.set(0, K0);
      auto prm = default_params(c.coeff, c.dt);
      const double K_init = ions.K()[0];
      const auto   L      = launch(ions, T, prm);
      const double K_fin  = ions.K()[0];
      const double d      = ions.delta()[0];
      const bool   finite = std::isfinite(K_fin) && std::isfinite(d) && std::isfinite(L.E_drag.value());
      verdict(finite && L.n_bounded == 1 && L.n_capped_loss == 1 && L.n_substeps == 64,
              std::string("R2 [") + c.name + "]: finite result, n_bounded = 1, n_capped_loss = 1, n_substeps = 64",
              static_cast<double>(L.n_substeps), 64.0);
      const double applied = (K_init - K_fin) / K_init;
      verdict(std::abs(applied / prm.f_max - 1.0) <= 1.0e-4,
              std::string("R2 [") + c.name + "]: the applied loss is the cap f_max K (relative to f_max)",
              applied, prm.f_max);
      diag(std::string("R2 [") + c.name + "]: K " + std::to_string(K_init) + " -> " + std::to_string(K_fin) +
           ", delta " + std::to_string(d) + ", E_drag " + std::to_string(L.E_drag.value()));
    }
  }

  /**
   * @brief R4 regression: the ulp floor is finite for |u| ~ 1e20 (a real_t
   *        sum of squares would overflow) and no application happens before
   *        the floor is reached
   */
  void test_huge_momentum(const Tables& T) {
    std::printf("---- R4: ulp floor at |u| = 1e20 ----\n");
    const double mu_d = 1.0;
    Ions ions(1, mu_d);
    ions.set(0, 1.0e20);
    auto prm              = default_params(1.0e10, 1.0);
    prm.eta_u             = 1.0e-12;
    prm.pending_floor_ulp = 100.0;
    HostLedger H;
    for (int s = 0; s < 10; ++s) {
      H.add(launch(ions, T, prm));
    }
    const double d   = ions.delta()[0];
    const double K   = ions.K()[0];
    const double flo = pending_floor_of(K, mu_d, 100.0); // the build's real_t ulp
    const double th  = std::max(flo, prm.eta_u * K);     // the kernel's application threshold
    const double req = 10.0 * prm.tau0_over_skindepth0 * T.host.I.back() * prm.dt; // ten edge-value requests
    // the referee's probe in single precision: floor 8.8e14 > eta_u K = 1e8, the ten
    // requests (2e13) stay below the floor and nothing is applied; in a double
    // build the floor is 1.6e6 < eta_u K and the first request is applied: the
    // registered statement is "applications happen only once the threshold
    // max(pending_floor, eta_u K) is reached", with a finite floor
    const bool expect_none = req < th;
    verdict(std::isfinite(flo) && std::isfinite(d) &&
              (expect_none ? (H.C.n_applied == 0 && d > 0.0) : (H.C.n_applied >= 1)),
            std::string("R4: finite ulp floor at |u| = 1e20; applications only once max(floor, eta_u K) is reached (") +
              (expect_none ? "none expected in 10 steps" : "expected from the first step") + ")",
            static_cast<double>(H.C.n_applied), expect_none ? 0.0 : 1.0);
    verdict(std::isfinite(flo) && flo > 0.0 && (sizeof(real_t) != 4 || flo > prm.eta_u * K),
            "R4: the floor is finite and positive (and above eta_u K in single precision)", flo, prm.eta_u * K);
    diag("R4: real_t = " + std::string(sizeof(real_t) == 4 ? "float" : "double") + ", K = " + std::to_string(K) +
         ", delta after 10 steps = " + std::to_string(d) + ", pending floor = " + std::to_string(flo) +
         ", eta_u K = " + std::to_string(prm.eta_u * K) + ", ten requests = " + std::to_string(req) +
         ", n_applied " + std::to_string(H.C.n_applied) + ", n_above_table " + std::to_string(H.C.n_above_table));
  }

  void test_above_table(const Tables& T) {
    std::printf("---- above the table (Gamma_p = 1.1 Gamma_max) ----\n");
    const double mu_d  = MU_R;
    const double Gp    = 1.1 * T.host.gamma_max;
    const double K0    = MU_R * (Gp - 1.0);
    const double I_edge = T.host.I.back();
    const double dt    = 1.0;
    const double coeff = 1.0e-4 * K0 / (I_edge * dt); // dK/K = 1e-4 per step, every step applied
    Ions ions(3, mu_d);
    ions.set(0, K0, 1.0, 0.0, 0.0, 1.0);
    ions.set(1, K0, 0.0, 1.0, 0.0, 2.0);
    ions.set(2, K0, 0.0, 0.0, 1.0, 0.5);
    auto         prm  = default_params(coeff, dt);
    prm.eta_u = 1.0e-5; // apply (nearly) every step: dK/K = 1e-4 > eta_u
    HostLedger   H;
    const double sum0 = ions.sum_wK();
    const int    n_steps = 20;
    for (int s = 0; s < n_steps; ++s) {
      H.add(launch(ions, T, prm));
    }
    verdict(H.C.n_above_table == 3ull * n_steps, "above-table: n_above_table counts every ion every step (3 x 20)",
            static_cast<double>(H.C.n_above_table), 3.0 * n_steps);
    const double requested = (1.0 + 2.0 + 0.5) * n_steps * coeff * I_edge * dt;
    const double d        = ions.delta()[0] + 2.0 * ions.delta()[1] + 0.5 * ions.delta()[2];
    const double rel      = std::abs((H.C.E_drag.value() + d) / requested - 1.0);
    verdict(rel <= 1.0e-12, "above-table: E_drag + pending equals 20 steps of the edge-value loss (sum w dK)",
            rel, 1.0e-12);
    diag("above-table: E_drag " + std::to_string(H.C.E_drag.value()) + " + pending " + std::to_string(d) +
         " vs requested " + std::to_string(requested) + "; final Gamma_p " +
         std::to_string(1.0 + ions.K()[0] / MU_R) + " (still above " + std::to_string(T.host.gamma_max) + ")");
    closure_check("above-table", sum0, ions, H);
  }

  /* ------------------------------------------------------------------------ */
  /* dead ions and below-edge pending discard                                   */
  /* ------------------------------------------------------------------------ */
  void test_tags(const Tables& T) {
    std::printf("---- tag classes (dead ion discards its pending request) ----\n");
    const double mu_d = MU_R;
    Ions ions(2, mu_d);
    ions.set(0, MU_R * 5.0, 1.0, 0.0, 0.0, 1.0);
    ions.set(1, MU_R * 5.0, 1.0, 0.0, 0.0, 3.0);
    const double I6    = bh::interpolate_loss_host(T.host, 6.0);
    const double dt    = 1.0;
    const double coeff = 1.0e-6 * MU_R * 5.0 / (I6 * dt); // below the threshold: accumulates only
    auto prm = default_params(coeff, dt);
    HostLedger H;
    for (int s = 0; s < 5; ++s) {
      H.add(launch(ions, T, prm));
    }
    const auto d_before = ions.delta();
    verdict(H.C.n_applied == 0 && d_before[0] > 0.0,
            "tags: five requests below delta_th accumulate without application",
            d_before[0], 0.0);
    verdict(std::abs(H.C.pending_outstanding.value() - (d_before[0] + 3.0 * d_before[1])) <= 1e-12 * H.C.pending_outstanding.value(),
            "tags: pending_outstanding snapshot = sum w delta", H.C.pending_outstanding.value(), 0.0);
    // kill ion 1
    {
      auto ht = Kokkos::create_mirror_view(ions.arr.tag);
      Kokkos::deep_copy(ht, ions.arr.tag);
      ht(1) = ParticleTag::dead;
      Kokkos::deep_copy(ions.arr.tag, ht);
    }
    const auto L = launch(ions, T, prm);
    const auto d_after = ions.delta();
    verdict(std::abs(L.pending_discarded.value() - 3.0 * d_before[1]) <= 1e-12 * L.pending_discarded.value() && d_after[1] == 0.0,
            "tags: a dead ion's pending request is discarded (w delta) and zeroed",
            L.pending_discarded.value(), 3.0 * d_before[1]);
    verdict(L.n_alive == 1 && L.n_processed == 1, "tags: the dead ion is not processed", L.n_processed, 1);
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::printf("bh_loss: real_t = %s, Kokkos default execution space = %s\n",
                sizeof(real_t) == 4 ? "float" : "double",
                Kokkos::DefaultExecutionSpace::name());
    const auto T = make_tables();
    std::printf("bh_loss: table mono eps0 = %.3g: %zu nodes, gamma_th %.12g, gamma_min %.12g, gamma_max %.12g, "
                "I_peak %.10e\n", EPS0, T.host.size(), T.host.gamma_th, T.host.gamma_min, T.host.gamma_max,
                T.host.I_peak);
    // optional group selection (ctest registers one entry per group so that
    // the pre-registered verdicts are visible individually)
    std::vector<std::string> groups;
    for (int i = 1; i < argc; ++i) {
      if (argv[i][0] != '-') {
        groups.emplace_back(argv[i]);
      }
    }
    auto want = [&groups](const char* g) {
      return groups.empty() || std::find(groups.begin(), groups.end(), std::string(g)) != groups.end();
    };
    {
      // R7: the logarithmic slope of I in K at Gamma_p = 3 (centred difference)
      const double K3 = MU_R * 2.0, h = 1.0e-4;
      const double lp = std::log(T.rates->at_gamma(1.0 + K3 * (1.0 + h) / MU_R).I);
      const double lm = std::log(T.rates->at_gamma(1.0 + K3 * (1.0 - h) / MU_R).I);
      std::printf("(diag) d ln I / d ln K at Gamma_p = 3: %.4f; I(0.99 K)/I(K) - 1 = %+.4f\n",
                  (lp - lm) / (2.0 * h), T.rates->at_gamma(1.0 + 0.99 * K3 / MU_R).I / T.rates->at_gamma(3.0).I - 1.0);
    }
    if (want("tables")) {
      test_device_tables(T);
    }
    if (want("overflow")) {
      test_overflow_requests(T);
    }
    if (want("huge")) {
      test_huge_momentum(T);
    }
    if (want("t9")) {
      test_threshold(T);
    }
    if (want("tags")) {
      test_tags(T);
    }
    if (want("above")) {
      test_above_table(T);
    }
    if (want("subcycling")) {
      test_subcycling(T);
    }
    if (want("t6")) {
      test_mu_independence_T6(T);
    }
    if (want("t5")) {
      test_cooling_T5(T);
    }
    if (want("t5fine")) {
      test_dt_convergence_T5fine(T);
    }
    if (want("t5b")) {
      test_ulp_scan_T5b(T);
    }
    std::printf("bh_loss summary: %d passed, %d failed, %d expected failures (T5b design off)\n",
                n_pass, n_fail, n_expected_fail);
    if (n_fail > 0) {
      rc = 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_loss: exception: %s\n", e.what());
    rc = 2;
  }
  Kokkos::finalize();
  return rc;
}
