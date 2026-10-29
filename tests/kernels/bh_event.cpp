/**
 * @file tests/kernels/bh_event.cpp
 * @brief Pre-registered kernel-level tests of the event (Monte Carlo) loss
 *        mode, `mode = event` (P2 step 5; contract 2026-10-29 section 4):
 *        E1 (energy exact per event: the array closure independent of the
 *        ledger, the ledger identity, the rounding residual), E5 (mu_d
 *        independence at the RNG level), E8 (failure paths at the kernel
 *        level: the per-event bound, capacity exhaustion, the event cap).
 *        The kernel is launched directly on device arrays (no pusher, no
 *        fields); the ions keep their stored u between launches.
 *
 * Conventions (declared before any run; tolerances are the contract's and
 * are not adjusted):
 *   - bath mono eps0 = 0.5 (Gamma_th = 2.125), mu_r = 1836.15, skindepth0 = 1,
 *     dt = 0.5, ions with mu_d = 1 unless stated, pair_weight a = 1, f_max =
 *     0.01, max_events_per_step = 8; the restframe tables are the device
 *     tables of step 4 (`build_sampler_tables`)
 *   - "lambda at Gamma_p": tau0/skindepth0 = lambda a / (J(Gamma_p) dt)
 *   - every pair is attributed to its parent by i1: ion p sits at i1 = p,
 *     dx1 = 0.5 and never moves
 *   - E1 array closure: sum_ions w (K_init - K_final) (K in double from the
 *     stored u) against sum_pairs w_s (gamma- + gamma+ - eps_s) from the
 *     stored pair arrays, |difference| / E_drag <= 1e-12 (double build),
 *     <= 1e-5 (single build); ledger identity residual / E_drag <= 1e-12 in
 *     both; |E_round| / E_drag <= 1e-5 (single), <= 1e-13 (double); the same
 *     closure per ion (every ion's own loss against its own pairs) is a
 *     stronger statement, not in the contract: its criterion is the
 *     mechanism's bound, |residual| / own loss <= ulp-energy(K_init) / (2 a
 *     <D>(Gamma_init)) + 1e-12 (the worst case of one float rounding of the
 *     stored u per event, every event rounding the same way; with the delta
 *     spectrum the subtracted recoil is the same at every event of an ion, so
 *     the rounding does not average: first measured 2.6e-5 in single at
 *     Gamma_p = 6 against 1e-5, then the criterion was set from the mechanism;
 *     recorded as such in the report), and 1e-12 in the double build
 *   - E5: same seed, same inputs, mu_d = 1 and mu_d = mu_r, launched with a
 *     Serial execution policy (one RNG state consumed in particle order; the
 *     OpenMP pool assigns states by scheduling): identical event counts per
 *     ion, per-ion K after every launch equal to <= 1e-5 relative (single),
 *     <= 1e-12 (double), for delta and restframe
 *   - E8 (i): f_max = 1.1e-3 set directly at the kernel (the pgen's startup
 *     refusal is bypassed here on purpose): with the delta spectrum every
 *     event of a Gamma_p = 3 ion violates a <D> <= f_max K (<D>/K = 1.40e-3)
 *     and no event of a Gamma_p = 6 ion does (9.1e-4): n_bounded = the events
 *     of the Gamma_p = 3 ions, their K unchanged bitwise, E1 holds. (ii)
 *     capacity: n_dropped > 0, the per-ion closure holds (no recoil without
 *     a pair). (iii) max_events_per_step = 1 at lambda = 1: n_capped_events
 *     > 0, at most one pair per (ion, launch), E1 holds.
 *
 * Verdict lines: "PASS" / "FAIL" with the measured number and the tolerance;
 * "(diag)" lines are reported only. One ctest entry per group (e1, e5, e8).
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "metrics/minkowski.h"

#include "archetypes/hadronic/bh_calibrated_test.h"
#include "archetypes/hadronic/bh_interpolate.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_sampler_device.h"
#include "archetypes/hadronic/bh_sampler_io.h"
#include "archetypes/hadronic/bh_tables_device.h"
#include "framework/containers/particles.h"
#include "kernels/hadronic/bethe_heitler.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;
namespace bh  = arch::bh;
namespace kh  = kernel::hadronic;
namespace cal = arch::bh::cal;

namespace {

  int n_pass = 0, n_fail = 0;

  void verdict(bool ok, const std::string& what, double value, double tol) {
    std::printf("%s: %s : %.6e (tolerance %.3e)\n", ok ? "PASS" : "FAIL", what.c_str(), value, tol);
    if (ok) {
      ++n_pass;
    } else {
      ++n_fail;
    }
  }

  void diag(const std::string& s) {
    std::printf("(diag) %s\n", s.c_str());
  }

  template <class... Ts>
  std::string fmt(const char* f, Ts... xs) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), f, xs...);
    return std::string(buf);
  }

  constexpr double MU_R   = 1836.15;
  constexpr double EPS0   = 0.5;
  constexpr double DT     = 0.5;
  constexpr bool   SINGLE = sizeof(real_t) == 4;
  // the contract's tolerances by build
  constexpr double TOL_CLOSURE  = SINGLE ? 1.0e-5 : 1.0e-12;
  constexpr double TOL_IDENTITY = 1.0e-12;
  constexpr double TOL_EROUND   = SINGLE ? 1.0e-5 : 1.0e-13;
  constexpr double TOL_E5_K     = SINGLE ? 1.0e-5 : 1.0e-12;

  /**
   * @brief The per-ion closure bound from the mechanism: one float rounding of
   *        the stored u per event, |dK_applied - a D| <= ulp(u) mu_d u / (2
   *        gamma_d), relative to the smallest intended loss a <D>; 1e-12 in
   *        the double build (the same expression there is ~1e-13 and the
   *        registered double tolerance of E1 is kept)
   */
  double per_ion_tolerance(double K, double mu_d, double a, double Dmean) {
    if (!SINGLE) {
      return 1.0e-12;
    }
    const double u   = std::sqrt(K * (K + 2.0 * mu_d)) / mu_d;
    const double gd  = std::sqrt(1.0 + u * u);
    const float  uf  = static_cast<float>(u);
    const double ulp = static_cast<double>(std::nextafter(uf, std::numeric_limits<float>::infinity())) -
                       static_cast<double>(uf);
    return 0.5 * ulp * mu_d * u / gd / (a * Dmean) + 1.0e-12;
  }

  struct Tables {
    std::shared_ptr<const bh::PhotonBath>  bath;
    std::unique_ptr<bh::BetheHeitlerRates> rates;
    bh::HadronicTablesHost                 host;
    bh::HadronicTables                     dev;
    bh::SamplerTables                      sampler; // built on demand
    bool                                   sampler_built { false };
  };

  Tables make_tables() {
    Tables       T;
    bh::BathSpec spec;
    spec.type = "mono";
    spec.eps0 = EPS0;
    T.bath    = bh::make_bath(spec);
    T.rates   = std::make_unique<bh::BetheHeitlerRates>(T.bath);
    T.host    = bh::build_tables(*T.rates, MU_R);
    bh::validate_tables(T.host);
    T.dev = bh::mirror_tables_to_device(T.host);
    return T;
  }

  void ensure_sampler(Tables& T) {
    if (!T.sampler_built) {
      T.sampler       = bh::build_sampler_tables(*T.bath, T.host);
      T.sampler_built = true;
      std::printf("bh_event: rest-frame sampler tables built (%.1f MB)\n", static_cast<double>(T.sampler.bytes) * 1e-6);
    }
  }

  double K_of_gamma(double Gp) {
    return MU_R * (Gp - 1.0);
  }

  /**
   * @brief tau0/skindepth0 such that R dt / a = lambda at Gamma_p
   */
  double coeff_for_lambda(const Tables& T, double Gp, double lambda, double a) {
    const double J = bh::interpolate_rate_host(T.host, Gp);
    return lambda * a / (J * DT);
  }

  template <typename ViewT>
  auto mirror(const ViewT& v) {
    auto h = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h, v);
    return h;
  }

  /* ------------------------------------------------------------------------ */
  /* ion set                                                                    */
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
      auto hi    = Kokkos::create_mirror_view(arr.i1);
      auto hd    = Kokkos::create_mirror_view(arr.dx1);
      for (npart_t p = 0; p < n; ++p) {
        hi(p) = static_cast<int>(p);
        hd(p) = static_cast<prtldx_t>(0.5);
      }
      Kokkos::deep_copy(arr.i1, hi);
      Kokkos::deep_copy(arr.dx1, hd);
      Kokkos::deep_copy(arr.tag, ParticleTag::alive);
      Kokkos::deep_copy(arr.pld_r, ZERO);
    }

    static double u_of_K(double K, double mu) {
      return std::sqrt(K * (K + 2.0 * mu)) / mu;
    }

    /// all ions along +x at the given energies and weight 1
    void set_all(const std::vector<double>& K) {
      auto h1 = mirror(arr.ux1), h2 = mirror(arr.ux2), h3 = mirror(arr.ux3);
      auto hw = mirror(arr.weight);
      for (npart_t p = 0; p < n; ++p) {
        h1(p) = static_cast<real_t>(u_of_K(K[p], mu_d));
        h2(p) = ZERO;
        h3(p) = ZERO;
        hw(p) = static_cast<real_t>(1.0);
      }
      Kokkos::deep_copy(arr.ux1, h1);
      Kokkos::deep_copy(arr.ux2, h2);
      Kokkos::deep_copy(arr.ux3, h3);
      Kokkos::deep_copy(arr.weight, hw);
    }

    std::vector<double> K() const {
      auto                h1 = mirror(arr.ux1), h2 = mirror(arr.ux2), h3 = mirror(arr.ux3);
      std::vector<double> out(n);
      for (npart_t p = 0; p < n; ++p) {
        out[p] = kh::kinetic_energy(mu_d, h1(p), h2(p), h3(p));
      }
      return out;
    }

    std::vector<double> w() const {
      auto                hw = mirror(arr.weight);
      std::vector<double> out(n);
      for (npart_t p = 0; p < n; ++p) {
        out[p] = static_cast<double>(hw(p));
      }
      return out;
    }

    /// the largest |payload| over the three slots and all ions (must stay 0)
    double max_abs_payload() const {
      auto   hp = mirror(arr.pld_r);
      double m  = 0.0;
      for (npart_t p = 0; p < n; ++p) {
        for (int c = 0; c < 3; ++c) {
          m = std::max(m, std::abs(static_cast<double>(hp(p, c))));
        }
      }
      return m;
    }
  };

  /* ------------------------------------------------------------------------ */
  /* pair destinations                                                          */
  /* ------------------------------------------------------------------------ */
  struct PairRow {
    int    i1m;
    double wm, u1m, u2m, u3m, u1p, u2p, u3p, birth_m, eps_m;
  };

  struct Pairs {
    ParticleArrays   minus, plus;
    npart_t          maxn { 0 };
    npart_t          npart_m { 0 }, npart_p { 0 };
    npart_t          counter_m { 0 }, counter_p { 0 };
    array_t<npart_t> spawn_counter { "spawn_counter" };

    explicit Pairs(npart_t maxn_) : maxn { maxn_ } {
      alloc(minus, "m");
      alloc(plus, "p");
    }

    void alloc(ParticleArrays& a, const std::string& s) {
      a.i1       = array_t<int*> { s + "i1", maxn };
      a.i2       = array_t<int*> { s + "i2", maxn };
      a.i3       = array_t<int*> { s + "i3", maxn };
      a.dx1      = array_t<prtldx_t*> { s + "dx1", maxn };
      a.dx2      = array_t<prtldx_t*> { s + "dx2", maxn };
      a.dx3      = array_t<prtldx_t*> { s + "dx3", maxn };
      a.i1_prev  = array_t<int*> { s + "i1p", maxn };
      a.i2_prev  = array_t<int*> { s + "i2p", maxn };
      a.i3_prev  = array_t<int*> { s + "i3p", maxn };
      a.dx1_prev = array_t<prtldx_t*> { s + "dx1p", maxn };
      a.dx2_prev = array_t<prtldx_t*> { s + "dx2p", maxn };
      a.dx3_prev = array_t<prtldx_t*> { s + "dx3p", maxn };
      a.ux1      = array_t<real_t*> { s + "ux1", maxn };
      a.ux2      = array_t<real_t*> { s + "ux2", maxn };
      a.ux3      = array_t<real_t*> { s + "ux3", maxn };
      a.weight   = array_t<real_t*> { s + "w", maxn };
      a.tag      = array_t<short*> { s + "tag", maxn };
      a.pld_r    = array_t<real_t**> { s + "pld_r", maxn, 3 };
      a.pld_i    = array_t<npart_t**> { s + "pld_i", maxn, 1 };
      a.phi      = array_t<real_t*> { s + "phi", maxn };
      Kokkos::deep_copy(a.tag, ParticleTag::dead);
    }

    auto targets() -> kh::PairTargets {
      kh::PairTargets t;
      t.minus      = minus;
      t.plus       = plus;
      t.offset_m   = npart_m;
      t.offset_p   = npart_p;
      t.capacity   = std::min(maxn - npart_m, maxn - npart_p);
      t.counter_m  = counter_m;
      t.counter_p  = counter_p;
      t.domain_idx = 0;
      t.tracking_m = false;
      t.tracking_p = false;
      Kokkos::deep_copy(spawn_counter, static_cast<npart_t>(0));
      t.spawn_counter = spawn_counter;
      t.valid         = true;
      return t;
    }

    auto finish(const kh::PairTargets& t) -> npart_t {
      npart_t tickets = 0;
      Kokkos::deep_copy(tickets, spawn_counter);
      const npart_t n_inj = std::min(tickets, t.capacity);
      npart_m   += n_inj;
      npart_p   += n_inj;
      counter_m += n_inj;
      counter_p += n_inj;
      return n_inj;
    }

    /// statistics runs (E2, E3, E6, E7): the pairs of a launch are overwritten by the next
    /// launch (the ledger carries their energy; the arrays are not inspected)
    void recycle() {
      npart_m = 0;
      npart_p = 0;
    }

    std::vector<PairRow> rows() const {
      std::vector<PairRow> out;
      const npart_t        n = std::min(npart_m, npart_p);
      auto i1m = mirror(minus.i1);
      auto wm  = mirror(minus.weight);
      auto u1m = mirror(minus.ux1), u2m = mirror(minus.ux2), u3m = mirror(minus.ux3);
      auto u1p = mirror(plus.ux1), u2p = mirror(plus.ux2), u3p = mirror(plus.ux3);
      auto pm = mirror(minus.pld_r);
      out.reserve(n);
      for (npart_t k = 0; k < n; ++k) {
        out.push_back(PairRow { i1m(k), static_cast<double>(wm(k)), static_cast<double>(u1m(k)),
                                static_cast<double>(u2m(k)), static_cast<double>(u3m(k)), static_cast<double>(u1p(k)),
                                static_cast<double>(u2p(k)), static_cast<double>(u3p(k)), static_cast<double>(pm(k, 0)),
                                static_cast<double>(pm(k, 2)) });
      }
      return out;
    }
  };

  /* ------------------------------------------------------------------------ */
  /* host ledger and the run driver                                             */
  /* ------------------------------------------------------------------------ */
  struct HostLedger {
    kh::LossLedger C {};

    void add(const kh::LossLedger& L) {
      const auto po = L.pending_outstanding, dbo = L.debt_outstanding;
      const auto sw = L.sum_wK;
      const auto na = L.n_alive;
      C += L;
      C.pending_outstanding = po;
      C.debt_outstanding    = dbo;
      C.sum_wK              = sw;
      C.n_alive             = na;
    }

    /// E_drag - [(E_pairs - E_absorbed) + E_round + E_stochastic + debt_outstanding + E_exported + E_lost_nopairs]
    double residual() const {
      kh::dd r;
      r.add(C.E_drag);
      r.sub(C.E_pairs);
      r.add(C.E_absorbed);
      r.sub(C.E_round);
      r.sub(C.E_stochastic);
      r.sub(C.debt_outstanding);
      r.sub(C.E_exported);
      r.sub(C.E_lost_nopairs);
      return r.value();
    }
  };

  constexpr int DELTA     = static_cast<int>(kh::PairSpectrum::Delta);
  constexpr int RESTFRAME = static_cast<int>(kh::PairSpectrum::RestFrame);

  kh::HadronicParams params(double coeff, int spectrum, double a = 1.0, double f_max = 0.01, int max_events = 8) {
    kh::HadronicParams prm;
    prm.tau0_over_skindepth0 = coeff;
    prm.mu_r                 = MU_R;
    prm.eta_u                = 1.0e-4;
    prm.pending_floor_ulp    = 100.0;
    prm.f_max                = f_max;
    prm.max_substeps         = 64;
    prm.dt                   = DT;
    prm.mode                 = static_cast<int>(kh::LossMode::Event);
    prm.source               = static_cast<int>(kh::PairSource::None);
    prm.spectrum             = spectrum;
    prm.pair_weight          = a;
    prm.max_events_per_step  = max_events;
    prm.max_sampler_attempts = 100000;
    return prm;
  }

  /**
   * @brief one launch (default execution space, or Serial when serial = true);
   *        the ledger of the launch is returned
   */
  kh::LossLedger launch(Ions& ions, Pairs& pairs, const Tables& T, kh::HadronicParams prm, random_number_pool_t& pool,
                        double time_after, bool serial = false) {
    prm.time  = time_after;
    auto targ = pairs.targets();
    kh::LossLedger L;
    using K = kh::BetheHeitler_kernel<Minkowski<Dim::_1D>>;
    K kern { ions.arr, ions.mu_d, T.dev, prm, targ, pool, T.sampler };
    if (serial) {
#if defined(KOKKOS_ENABLE_SERIAL)
      Kokkos::parallel_reduce("bh_event_serial", Kokkos::RangePolicy<Kokkos::Serial>(0, ions.n), kern,
                              Kokkos::Sum<kh::LossLedger>(L));
#else
      Kokkos::parallel_reduce("bh_event", CreateParticleRangePolicy<Dim::_1D>({ 0u }, { ions.n }), kern,
                              Kokkos::Sum<kh::LossLedger>(L));
#endif
    } else {
      Kokkos::parallel_reduce("bh_event", CreateParticleRangePolicy<Dim::_1D>({ 0u }, { ions.n }), kern,
                              Kokkos::Sum<kh::LossLedger>(L));
    }
    Kokkos::fence();
    const auto n_inj = pairs.finish(targ);
    if (static_cast<unsigned long long>(n_inj) != L.n_spawned) {
      throw std::runtime_error("n_injected != n_spawned");
    }
    return L;
  }

  /* ------------------------------------------------------------------------ */
  /* the E1 closure from the arrays                                              */
  /* ------------------------------------------------------------------------ */
  struct Closure {
    double lhs { 0.0 }, rhs { 0.0 }, residual { 0.0 };
    double max_rel_per_ion { 0.0 }; // over ions with at least one pair
    double tol_per_ion { 0.0 };     // the largest per-ion tolerance used (reported)
    int    n_ions_with_pairs { 0 };
    int    n_ions_bad { 0 };        // per-ion closure beyond the tolerance
  };

  Closure closure(const Ions& ions, const std::vector<double>& K_init, const Pairs& pairs, const Tables& T, double a) {
    Closure C;
    kh::dd  lhs, rhs;
    const auto          rows = pairs.rows();
    std::vector<kh::dd> per_ion(ions.n);
    std::vector<int>    cnt(ions.n, 0);
    for (const auto& r : rows) {
      if (!(r.birth_m > 0.0)) {
        continue;
      }
      const double E_s = kh::lepton_gamma(r.u1m, r.u2m, r.u3m) + kh::lepton_gamma(r.u1p, r.u2p, r.u3p);
      lhs.add_prod(r.wm, E_s);
      lhs.add_prod(-r.wm, r.eps_m);
      if (r.i1m >= 0 && static_cast<npart_t>(r.i1m) < ions.n) {
        per_ion[r.i1m].add_prod(r.wm, E_s);
        per_ion[r.i1m].add_prod(-r.wm, r.eps_m);
        cnt[r.i1m] += 1;
      }
    }
    const auto Kf = ions.K();
    const auto w  = ions.w();
    for (npart_t p = 0; p < ions.n; ++p) {
      rhs.add_prod(w[p], K_init[p] - Kf[p]);
      if (cnt[p] > 0) {
        kh::dd d;
        d.add_prod(w[p], K_init[p] - Kf[p]);
        d.sub(per_ion[p]);
        const double own = std::abs(w[p] * (K_init[p] - Kf[p]));
        const double rel = own > 0.0 ? std::abs(d.value()) / own : std::abs(d.value());
        double       Dm = 0.0, e = 0.0;
        bh::interpolate_outcome_host(T.host, 1.0 + K_init[p] / MU_R, Dm, e);
        const double tol = per_ion_tolerance(K_init[p], ions.mu_d, a, Dm);
        C.max_rel_per_ion = std::max(C.max_rel_per_ion, rel);
        C.tol_per_ion     = std::max(C.tol_per_ion, tol);
        if (rel > tol) {
          C.n_ions_bad += 1;
        }
        C.n_ions_with_pairs += 1;
      }
    }
    kh::dd res = rhs;
    res.sub(lhs);
    C.lhs      = lhs.value();
    C.rhs      = rhs.value();
    C.residual = res.value();
    return C;
  }

  void check_E1(const std::string& tag, const HostLedger& H, const Closure& A, const Ions& ions) {
    const double E = std::abs(H.C.E_drag.value());
    verdict(E > 0.0 && std::abs(A.residual) <= TOL_CLOSURE * E,
            tag + ": E1 array closure |sum w dK - sum w_s (E- + E+ - eps)| / E_drag", E > 0.0 ? std::abs(A.residual) / E : -1.0,
            TOL_CLOSURE);
    verdict(A.n_ions_bad == 0, tag + ": per-ion closure (mechanism bound, not in the contract) on " +
                                 std::to_string(A.n_ions_with_pairs) + " ions with pairs, max relative residual",
            A.max_rel_per_ion, A.tol_per_ion);
    verdict(std::abs(H.residual()) <= TOL_IDENTITY * E, tag + ": ledger identity residual / E_drag",
            E > 0.0 ? std::abs(H.residual()) / E : -1.0, TOL_IDENTITY);
    const double er = H.C.E_round.value();
    verdict(std::abs(er) <= TOL_EROUND * E, tag + ": |E_round| / E_drag", E > 0.0 ? std::abs(er) / E : -1.0, TOL_EROUND);
    // E_round is the array residual by construction (consistency of the two bookkeepings)
    const double diff = std::abs(A.residual - er);
    verdict(diff <= 1.0e-9 * std::max(std::abs(er), 1.0e-300) || (er == 0.0 && A.residual == 0.0),
            tag + ": E_round equals the array residual, |E_round - residual| / |E_round|",
            er != 0.0 ? diff / std::abs(er) : diff, 1.0e-9);
    verdict(H.C.E_stochastic.value() == 0.0 && H.C.E_lost_nopairs.value() == 0.0 && H.C.E_exported.value() == 0.0 &&
              H.C.debt_outstanding.value() == 0.0 && H.C.pending_outstanding.value() == 0.0,
            tag + ": E_stochastic = E_lost_nopairs = E_exported = debt = pending = 0", 0.0, 0.0);
    verdict(ions.max_abs_payload() == 0.0, tag + ": the three hadronic payload slots read 0 on every ion (stateless)",
            ions.max_abs_payload(), 0.0);
    verdict(H.C.n_events == H.C.n_spawned + H.C.n_dropped + H.C.n_bounded,
            tag + ": n_events = n_spawned + n_dropped + n_bounded", static_cast<double>(H.C.n_events),
            static_cast<double>(H.C.n_spawned + H.C.n_dropped + H.C.n_bounded));
    verdict(H.C.P_drag[1] == 0.0 && H.C.P_drag[2] == 0.0, tag + ": P_drag transverse to +x is exactly 0 (recoil along u-hat)",
            std::abs(H.C.P_drag[1]) + std::abs(H.C.P_drag[2]), 0.0);
    diag(tag + fmt(": E_drag %.10g, E_pairs - E_absorbed %.10g, E_round %.6e, n_events %llu, n_spawned %llu, n_dropped %llu, "
                   "n_bounded %llu, n_capped_events %llu, n_rng %llu, n_sampler_overflow %llu, P_drag_x %.10g, P_pairs_x %.10g, "
                   "P_absorbed_x %.10g",
                   H.C.E_drag.value(), H.C.E_pairs.value() - H.C.E_absorbed.value(), er, H.C.n_events, H.C.n_spawned,
                   H.C.n_dropped, H.C.n_bounded, H.C.n_capped_events, H.C.n_rng, H.C.n_sampler_overflow, H.C.P_drag[0],
                   H.C.P_pairs[0], H.C.P_absorbed[0]));
  }

  /* ------------------------------------------------------------------------ */
  /* E1                                                                          */
  /* ------------------------------------------------------------------------ */
  void test_E1(Tables& T, int spectrum) {
    const std::string name = spectrum == DELTA ? "delta" : "restframe";
    std::printf("\n=== E1 (%s): 500 ions at Gamma_p = 3 and 500 at Gamma_p = 6, a = 1, lambda = 0.1 at Gamma_p = 6, "
                "200 launches ===\n", name.c_str());
    if (spectrum == RESTFRAME) {
      ensure_sampler(T);
    }
    const npart_t       N = 1000;
    Ions                ions(N, 1.0);
    std::vector<double> K0(N);
    for (npart_t p = 0; p < N; ++p) {
      K0[p] = K_of_gamma(p < N / 2 ? 3.0 : 6.0);
    }
    ions.set_all(K0);
    const auto K_init = ions.K();
    Pairs      pairs(400000);
    HostLedger H;
    auto       prm = params(coeff_for_lambda(T, 6.0, 0.1, 1.0), spectrum);
    random_number_pool_t pool(31337u);
    // the exact Poisson mean of the drawn counts from the per-launch states
    double mean_lambda = 0.0;
    for (int n = 0; n < 200; ++n) {
      const auto Ks = ions.K();
      for (npart_t p = 0; p < N; ++p) {
        const double Gp = 1.0 + Ks[p] / MU_R;
        mean_lambda += prm.tau0_over_skindepth0 * bh::interpolate_rate_host(T.host, Gp) * DT / prm.pair_weight;
      }
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
    }
    const auto A = closure(ions, K_init, pairs, T, prm.pair_weight);
    check_E1("E1 " + name, H, A, ions);
    const double sig = std::sqrt(mean_lambda);
    verdict(std::abs(static_cast<double>(H.C.n_events) - mean_lambda) <= 4.0 * sig,
            "E1 " + name + ": drawn events within 4 sigma of the Poisson mean from the per-launch states, |n - mean| / sigma",
            std::abs(static_cast<double>(H.C.n_events) - mean_lambda) / sig, 4.0);
    verdict(H.C.n_dropped == 0 && H.C.n_bounded == 0 && H.C.n_capped_events == 0,
            "E1 " + name + ": no dropped, bounded or capped event at a = 1, f_max = 0.01, lambda <= 0.1",
            static_cast<double>(H.C.n_dropped + H.C.n_bounded + H.C.n_capped_events), 0.0);
    // <D> consistency: E_drag / n_spawned against the mean of <D>(Gamma_p) over the spawned events is
    // reported (E3 registers the straggling; here only the first moment is shown)
    double D3 = 0.0, D6 = 0.0, e3 = 0.0, e6 = 0.0;
    bh::interpolate_outcome_host(T.host, 3.0, D3, e3);
    bh::interpolate_outcome_host(T.host, 6.0, D6, e6);
    diag("E1 " + name + fmt(": mean loss per event E_drag / n_spawned = %.6g (table <D> at Gamma_p = 3: %.6g, at 6: %.6g)",
                            H.C.E_drag.value() / static_cast<double>(H.C.n_spawned), D3, D6));
  }

  /* ------------------------------------------------------------------------ */
  /* E5                                                                          */
  /* ------------------------------------------------------------------------ */
  struct Trace {
    std::vector<std::vector<double>> K;   // [launch][ion]
    std::vector<int>                 cnt; // pairs per ion
    unsigned long long               n_events { 0 }, n_rng { 0 };
  };

  Trace run_trace(Tables& T, int spectrum, double mu_d, std::uint64_t seed, int n_launch, npart_t N, double Gp0,
                  double lambda) {
    Ions                ions(N, mu_d);
    std::vector<double> K0(N, K_of_gamma(Gp0));
    ions.set_all(K0);
    Pairs      pairs(200000);
    HostLedger H;
    auto       prm = params(coeff_for_lambda(T, Gp0, lambda, 1.0), spectrum);
    random_number_pool_t pool(seed);
    Trace                tr;
    for (int n = 0; n < n_launch; ++n) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT, true));
      tr.K.push_back(ions.K());
    }
    tr.cnt.assign(N, 0);
    for (const auto& r : pairs.rows()) {
      if (r.birth_m > 0.0 && r.i1m >= 0 && static_cast<npart_t>(r.i1m) < N) {
        tr.cnt[r.i1m] += 1;
      }
    }
    tr.n_events = H.C.n_events;
    tr.n_rng    = H.C.n_rng;
    return tr;
  }

  void test_E5(Tables& T, int spectrum) {
    const std::string name = spectrum == DELTA ? "delta" : "restframe";
    std::printf("\n=== E5 (%s): mu_d = 1 against mu_d = mu_r, same seed, 64 ions at Gamma_p = 6, lambda = 0.2, "
                "100 launches, Serial execution ===\n", name.c_str());
    if (spectrum == RESTFRAME) {
      ensure_sampler(T);
    }
    const npart_t N = 64;
    const auto    a = run_trace(T, spectrum, 1.0, 97531u, 100, N, 6.0, 0.2);
    const auto    b = run_trace(T, spectrum, MU_R, 97531u, 100, N, 6.0, 0.2);
    int    n_cnt_diff = 0;
    double max_rel    = 0.0;
    for (npart_t p = 0; p < N; ++p) {
      if (a.cnt[p] != b.cnt[p]) {
        ++n_cnt_diff;
      }
      for (std::size_t n = 0; n < a.K.size(); ++n) {
        max_rel = std::max(max_rel, std::abs(a.K[n][p] - b.K[n][p]) / a.K[n][p]);
      }
    }
    verdict(n_cnt_diff == 0 && a.n_events == b.n_events,
            "E5 " + name + ": identical event counts per ion for mu_d = 1 and mu_d = mu_r (ions with a different count)",
            static_cast<double>(n_cnt_diff), 0.0);
    verdict(max_rel <= TOL_E5_K, "E5 " + name + ": per-ion K after every launch, max relative difference", max_rel, TOL_E5_K);
    diag("E5 " + name + fmt(": n_events %llu / %llu, n_rng %llu / %llu, total pairs %d / %d", a.n_events, b.n_events, a.n_rng,
                            b.n_rng, std::accumulate(a.cnt.begin(), a.cnt.end(), 0), std::accumulate(b.cnt.begin(), b.cnt.end(), 0)));
    verdict(a.n_events > 0, "E5 " + name + ": events were drawn", static_cast<double>(a.n_events), 1.0);
  }

  /* ------------------------------------------------------------------------ */
  /* stage 3: E2, E3, E6, E7 (statistics against the host ODE and the fixture)  */
  /* ------------------------------------------------------------------------ */

  /**
   * @brief E[D^2](Gamma_p) of the unweighted law from the frozen bhref 0.2.0
   *        moments fixture (column D2mean), interpolated log-linearly in
   *        y = log10(Gamma - Gamma_th) between the fixture's nodes (Codex 1's
   *        reference, never the author's event code). Loaded from
   *        BH_TEST_DATA_DIR/bhref-0.2.0/moments_mono05.txt; the file's
   *        data_sha256 is validated by the reader.
   */
  struct D2Table {
    std::vector<double> y, lnD2, gamma, D2;
    double              gamma_th { 0.0 };
    std::string         sha;

    double at(double G) const {
      if (G <= gamma[0]) {
        return D2[0];
      }
      if (G >= gamma.back()) {
        return D2.back();
      }
      const double yy = std::log10(G - gamma_th);
      std::size_t  lo = 0, hi = y.size() - 1;
      while (hi - lo > 1) {
        const std::size_t mid = (lo + hi) / 2;
        if (y[mid] <= yy) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const double t = (yy - y[lo]) / (y[hi] - y[lo]);
      return std::exp(lnD2[lo] + t * (lnD2[hi] - lnD2[lo]));
    }
  };

  D2Table load_d2(const Tables& T) {
    D2Table d;
    d.gamma_th = T.bath->gamma_th();
#ifndef BH_TEST_DATA_DIR
#error "BH_TEST_DATA_DIR must be defined (tests/kernels/CMakeLists.txt)"
#endif
    const std::string path = std::string(BH_TEST_DATA_DIR) + "/bhref-0.2.0/moments_mono05.txt";
    const auto        f    = bh::read_fixture_file(path);
    d.sha                  = f.data_sha256;
    std::size_t ig = 0, id2 = 0;
    bool        fg = false, fd = false;
    for (std::size_t c = 0; c < f.columns.size(); ++c) {
      if (f.columns[c] == "gamma_p") {
        ig = c;
        fg = true;
      }
      if (f.columns[c] == "D2mean") {
        id2 = c;
        fd  = true;
      }
    }
    if (!fg || !fd) {
      throw std::runtime_error("moments_mono05.txt: columns gamma_p / D2mean not found");
    }
    for (const auto& r : f.rows) {
      const double G = r[ig], D2 = r[id2];
      if (G > d.gamma_th && D2 > 0.0) {
        d.gamma.push_back(G);
        d.D2.push_back(D2);
        d.y.push_back(std::log10(G - d.gamma_th));
        d.lnD2.push_back(std::log(D2));
      }
    }
    if (d.gamma.size() < 2) {
      throw std::runtime_error("moments_mono05.txt: too few nodes");
    }
    return d;
  }

  /// E[D^2] of the delta spectrum: <D>(Gamma)^2 exactly (the table's Dmean)
  double d2_delta(const Tables& T, double G) {
    double Dm = 0.0, e = 0.0;
    bh::interpolate_outcome_host(T.host, G, Dm, e);
    return Dm * Dm;
  }

  struct Series {
    std::vector<double> t, meanK, varK;  // at the recorded launches (t = n dt)
    double              K0 { 0.0 };
    npart_t             N { 0 };
    double              dt { 0.0 }, coeff { 0.0 }, a { 1.0 };
    kh::LossLedger      C {};            // cumulative ledger (counters)
    int                 n_launch { 0 };
    double              seconds { 0.0 };
  };

  /**
   * @brief One cooling run: N ions at Gamma_p0 (mu_d = 1, +x), the given mode
   *        and spectrum, launched until <K> <= stop_frac K0 or n_max launches;
   *        <K> and Var K recorded every out_every launches (two-pass in
   *        double from the stored u). The pairs are recycled per launch.
   */
  Series cooling_run(Tables& T, int mode, int spectrum, double a, double f_max, double coeff, double dt, double Gp0,
                     npart_t N, double stop_frac, int n_max, int out_every, std::uint64_t seed, double eta_u = 1.0e-4) {
    if (spectrum == RESTFRAME) {
      ensure_sampler(T);
    }
    const auto t0 = std::chrono::steady_clock::now();
    Ions                ions(N, 1.0);
    std::vector<double> K0v(N, K_of_gamma(Gp0));
    ions.set_all(K0v);
    Series S;
    S.N     = N;
    S.dt    = dt;
    S.coeff = coeff;
    S.a     = a;
    {
      const auto Ks = ions.K();
      double     m  = 0.0;
      for (const double k : Ks) {
        m += k;
      }
      S.K0 = m / static_cast<double>(N);
    }
    Pairs pairs(static_cast<npart_t>(std::max<std::size_t>(200000, static_cast<std::size_t>(N) * 10)));
    auto  prm = params(coeff, spectrum, a, f_max);
    prm.dt    = dt;
    prm.mode  = mode;
    if (mode == static_cast<int>(kh::LossMode::Continuous)) {
      prm.source = static_cast<int>(kh::PairSource::None);
      prm.eta_u  = eta_u;
    }
    random_number_pool_t pool(seed);
    HostLedger           H;
    auto record = [&](int n) {
      const auto Ks = ions.K();
      double     m  = 0.0;
      for (const double k : Ks) {
        m += k;
      }
      m /= static_cast<double>(N);
      double v = 0.0;
      for (const double k : Ks) {
        v += (k - m) * (k - m);
      }
      v /= static_cast<double>(N - 1);
      S.t.push_back(static_cast<double>(n) * dt);
      S.meanK.push_back(m);
      S.varK.push_back(v);
      return m;
    };
    record(0);
    int n = 0;
    while (n < n_max) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * dt));
      pairs.recycle();
      ++n;
      if (n % out_every == 0) {
        const double m = record(n);
        if (m <= stop_frac * S.K0) {
          break;
        }
      }
    }
    if (S.t.back() != static_cast<double>(n) * dt) {
      record(n);
    }
    S.n_launch = n;
    S.C        = H.C;
    S.seconds  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return S;
  }

  /**
   * @brief The reference ODEs on the host (step 2's RK4, T5's reference): the
   *        table-interpolated d<K>/dt = -coeff I(<K>) ("plain") and the
   *        corrected one with the Jensen term, d<K>/dt = -coeff [I(<K>) + (1/2)
   *        I''(<K>) Var K(t)], I'' by centred differences on the interpolated
   *        table with h = 0.02 K (the curvature scale of I is K; a smaller h
   *        resolves the kinks of the piecewise log-linear interpolation
   *        instead of the curvature), Var K(t) linearly interpolated between
   *        the run's records (an independent measured quantity, not the
   *        event code's mean). Alongside, the straggling prediction of E3:
   *        Var_pred(t) = int_0^t a coeff J(Gamma(t')) E[D^2](Gamma(t')) dt'
   *        along the corrected mean (trapezoid on the RK4 grid). Returned at
   *        the recorded times.
   */
  struct Reference {
    std::vector<double> K_plain, K_corr, var_pred, var_fb, jensen_rate; // at the series' times
    // var_pred: the registered E3 prediction int a R E[D^2] dt' along the corrected mean (no feedback);
    // var_fb [proposal, stage 3]: the same source term with the linear drift feedback,
    //   dV/dt = -2 c(t) V + a R E[D^2],  c = coeff dI/dK at <K>(t) (centred difference, h = 0.02 K):
    // an ion that has lost more sits at a lower rate (dlnI/dlnK ~ 4 at Gamma_p = 6), which pulls the
    // spread back; the suppression is first order, 2 (dlnI/dlnK) (Delta K / K), not second order in
    // sigma_K / K as the contract's window argument assumed (fact, stage 3 development run)
  };

  Reference reference(const Tables& T, const Series& S, const std::function<double(double)>& d2_of_gamma) {
    const double mu_r = MU_R;
    auto I_of_K       = [&](double K) {
      return bh::interpolate_loss_host(T.host, 1.0 + K / mu_r);
    };
    auto J_of_K = [&](double K) {
      return bh::interpolate_rate_host(T.host, 1.0 + K / mu_r);
    };
    auto Ipp = [&](double K) {
      const double h = 0.02 * K;
      return (I_of_K(K + h) - 2.0 * I_of_K(K) + I_of_K(K - h)) / (h * h);
    };
    auto Ip = [&](double K) {
      const double h = 0.02 * K;
      return (I_of_K(K + h) - I_of_K(K - h)) / (2.0 * h);
    };
    auto var_at = [&](double t) {
      // linear interpolation of the measured Var K(t)
      if (t <= S.t.front()) {
        return S.varK.front();
      }
      if (t >= S.t.back()) {
        return S.varK.back();
      }
      const auto        it = std::upper_bound(S.t.begin(), S.t.end(), t);
      const std::size_t hi = static_cast<std::size_t>(it - S.t.begin()), lo = hi - 1;
      const double      f  = (t - S.t[lo]) / (S.t[hi] - S.t[lo]);
      return S.varK[lo] + f * (S.varK[hi] - S.varK[lo]);
    };
    Reference R;
    const double h = S.dt; // the RK4 step: the run's dt (the smooth ODE is resolved far below the tolerance)
    double Kp = S.K0, Kc = S.K0, t = 0.0, vp = 0.0, vf = 0.0;
    auto fp = [&](double K) {
      return -S.coeff * I_of_K(K);
    };
    auto fc = [&](double K, double tt) {
      return -S.coeff * (I_of_K(K) + 0.5 * Ipp(K) * var_at(tt));
    };
    auto rk4_to = [&](double t_end) {
      while (t < t_end - 1e-12 * std::max(1.0, t_end)) {
        const double hh = std::min(h, t_end - t);
        // plain
        {
          const double k1 = fp(Kp), k2 = fp(Kp + 0.5 * hh * k1), k3 = fp(Kp + 0.5 * hh * k2), k4 = fp(Kp + hh * k3);
          Kp += hh / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        // corrected, and the straggling integral along it (trapezoid)
        {
          const double Kc_prev = Kc;
          const double g0 = S.a * S.coeff * J_of_K(Kc) * d2_of_gamma(1.0 + Kc / mu_r);
          const double k1 = fc(Kc, t), k2 = fc(Kc + 0.5 * hh * k1, t + 0.5 * hh), k3 = fc(Kc + 0.5 * hh * k2, t + 0.5 * hh),
                       k4 = fc(Kc + hh * k3, t + hh);
          Kc += hh / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
          const double g1 = S.a * S.coeff * J_of_K(Kc) * d2_of_gamma(1.0 + Kc / mu_r);
          vp += 0.5 * hh * (g0 + g1);
          // the feedback variance: linear ODE dV/dt = -2 c V + g, exact over the step for constant c, g
          // (the midpoint values); V(t + hh) = V e^{-2 c hh} + g (1 - e^{-2 c hh}) / (2 c)
          const double c  = S.coeff * Ip(0.5 * (Kc_prev + Kc));
          const double g  = 0.5 * (g0 + g1);
          const double ex = std::exp(-2.0 * c * hh);
          vf              = vf * ex + (std::abs(c) > 1e-300 ? g * (1.0 - ex) / (2.0 * c) : g * hh);
        }
        t += hh;
      }
    };
    for (std::size_t i = 0; i < S.t.size(); ++i) {
      rk4_to(S.t[i]);
      R.K_plain.push_back(Kp);
      R.K_corr.push_back(Kc);
      R.var_pred.push_back(vp);
      R.var_fb.push_back(vf);
      R.jensen_rate.push_back(-S.coeff * 0.5 * Ipp(Kc) * var_at(S.t[i]));
    }
    return R;
  }

  /// index of the first record with <K> <= f K0 (or npos)
  std::size_t first_below(const Series& S, double f) {
    for (std::size_t i = 0; i < S.meanK.size(); ++i) {
      if (S.meanK[i] <= f * S.K0) {
        return i;
      }
    }
    return std::string::npos;
  }

  /**
   * @brief E2 at the targets <K>/K0 in {0.9, 0.7, 0.5}: |<K> - K_ODE,corr| <= 4 s.e.
   *        (s.e. = sigma_K / sqrt(N) from the sample variance); the plain ODE
   *        difference and the Jensen term relative to 4 s.e. are reported.
   *        Unreachable targets (the run stopped above them) are reported as such.
   */
  void verdict_E2(const std::string& tag, const Series& S, const Reference& R) {
    for (const double f : { 0.9, 0.7, 0.5 }) {
      const std::size_t i = first_below(S, f);
      if (i == std::string::npos) {
        std::printf("(diag) %s: E2 target <K>/K0 = %.1f not reached: the run stopped at <K>/K0 = %.4f after %d launches "
                    "(t = %.6g)\n",
                    tag.c_str(), f, S.meanK.back() / S.K0, S.n_launch, S.t.back());
        continue;
      }
      const double se   = std::sqrt(S.varK[i] / static_cast<double>(S.N));
      const double diff = S.meanK[i] - R.K_corr[i];
      const double dpl  = S.meanK[i] - R.K_plain[i];
      verdict(std::abs(diff) <= 4.0 * se,
              tag + fmt(": E2 at <K>/K0 = %.1f (t = %.6g, <K> = %.6g): |<K> - K_ODE,corrected| / s.e. (s.e. = %.4g)", f,
                        S.t[i], S.meanK[i], se),
              std::abs(diff) / se, 4.0);
      std::printf("(diag) %s: E2 at %.1f: plain ODE difference %.4g (%.3f s.e.), Jensen correction K_corr - K_plain = %.4g "
                  "(%.3f of 4 s.e.), sigma_K/K = %.4g\n",
                  tag.c_str(), f, dpl, dpl / se, R.K_corr[i] - R.K_plain[i], (R.K_corr[i] - R.K_plain[i]) / (4.0 * se),
                  std::sqrt(S.varK[i]) / S.meanK[i]);
    }
  }

  /**
   * @brief E3 in the window sigma_K/K <= 0.02: |Var_meas / Var_pred - 1| <= 4 sqrt(2/N)
   *        at the records where sigma_K/K first exceeds 0.005, 0.01 and at the
   *        last record inside the window; the maximum deviation over the window
   *        (records with at least 20 expected events per ion) is reported.
   */
  void verdict_E3(const std::string& tag, const Series& S, const Reference& R) {
    const double tol = 4.0 * std::sqrt(2.0 / static_cast<double>(S.N));
    std::size_t  last_in = std::string::npos, i005 = std::string::npos, i01 = std::string::npos;
    for (std::size_t i = 1; i < S.t.size(); ++i) {
      const double rel = std::sqrt(S.varK[i]) / S.meanK[i];
      if (rel <= 0.02) {
        last_in = i;
      }
      if (i005 == std::string::npos && rel >= 0.005) {
        i005 = i;
      }
      if (i01 == std::string::npos && rel >= 0.01) {
        i01 = i;
      }
    }
    std::vector<std::pair<const char*, std::size_t>> pts = { { "sigma_K/K = 0.005", i005 }, { "sigma_K/K = 0.01", i01 },
                                                             { "last record with sigma_K/K <= 0.02", last_in } };
    for (const auto& pt : pts) {
      const std::size_t i = pt.second;
      if (i == std::string::npos || !(R.var_pred[i] > 0.0)) {
        std::printf("(diag) %s: E3 point (%s) not in the run\n", tag.c_str(), pt.first);
        continue;
      }
      const double ratio = S.varK[i] / R.var_pred[i];
      const double dKK   = (S.K0 - R.K_corr[i]) / S.K0;
      verdict(std::abs(ratio - 1.0) <= tol,
              tag + fmt(": E3 [registered: no feedback] at %s (t = %.6g, Delta K/K = %.4f, Var_meas = %.6g, Var_pred = %.6g): "
                        "|Var_meas/Var_pred - 1|",
                        pt.first, S.t[i], dKK, S.varK[i], R.var_pred[i]),
              std::abs(ratio - 1.0), tol);
      const double ratio_fb = S.varK[i] / R.var_fb[i];
      verdict(std::abs(ratio_fb - 1.0) <= tol,
              tag + fmt(": E3 [proposal: linear drift feedback] at %s (Var_fb = %.6g): |Var_meas/Var_fb - 1|", pt.first,
                        R.var_fb[i]),
              std::abs(ratio_fb - 1.0), tol);
    }
    double worst = 0.0, t_worst = 0.0, worst_fb = 0.0, t_worst_fb = 0.0;
    for (std::size_t i = 1; i < S.t.size(); ++i) {
      const double rel = std::sqrt(S.varK[i]) / S.meanK[i];
      if (rel > 0.02 || !(R.var_pred[i] > 0.0) || !(R.var_fb[i] > 0.0)) {
        continue;
      }
      // records after at least 1% of the energy is lost (about 10 events per ion: below that the
      // ratio's own noise exceeds the tolerance)
      if ((S.K0 - R.K_corr[i]) < 0.01 * S.K0) {
        continue;
      }
      const double dev = std::abs(S.varK[i] / R.var_pred[i] - 1.0);
      if (dev > worst) {
        worst   = dev;
        t_worst = S.t[i];
      }
      const double dfb = std::abs(S.varK[i] / R.var_fb[i] - 1.0);
      if (dfb > worst_fb) {
        worst_fb   = dfb;
        t_worst_fb = S.t[i];
      }
    }
    std::printf("(diag) %s: E3 max |Var_meas/Var_pred - 1| over the window (Delta K/K >= 1%%) = %.4g at t = %.6g [registered], "
                "%.4g at t = %.6g [feedback] (tolerance %.4g)\n",
                tag.c_str(), worst, t_worst, worst_fb, t_worst_fb, tol);
  }

  void print_series(const std::string& tag, const Series& S) {
    std::printf("(diag) %s: N = %u, K0 = %.6g, dt = %.6g, coeff = %.6g, a = %.3g, %d launches, final <K>/K0 = %.5f, "
                "sigma_K/K = %.4g; n_events %llu, n_spawned %llu, n_bounded %llu, n_dropped %llu, n_capped_events %llu, "
                "n_sampler_overflow %llu; %.1f s\n",
                tag.c_str(), (unsigned)S.N, S.K0, S.dt, S.coeff, S.a, S.n_launch, S.meanK.back() / S.K0,
                std::sqrt(S.varK.back()) / S.meanK.back(), S.C.n_events, S.C.n_spawned, S.C.n_bounded, S.C.n_dropped,
                S.C.n_capped_events, S.C.n_sampler_overflow, S.seconds);
  }

  constexpr int EVENT      = static_cast<int>(kh::LossMode::Event);
  constexpr int CONTINUOUS = static_cast<int>(kh::LossMode::Continuous);

  /**
   * @brief E2 and E3 at Gamma_p0 (6 or 3): a = 1, lambda = 0.05 at Gamma_p0 (dt = 0.5),
   *        delta and restframe, N ions; plus the continuous mode (eta_u = 1e-5,
   *        source none) on the same ions and dt as the small-jump statement
   */
  void test_E2E3(Tables& T, const D2Table& D2, double Gp0, npart_t N, int n_max, double stop_frac) {
    const double lambda = 0.05;
    const double coeff  = coeff_for_lambda(T, Gp0, lambda, 1.0);
    std::printf("\n=== E2 / E3 at Gamma_p0 = %g: N = %u, a = 1, lambda = %.3g at Gamma_p0 (dt = %.3g), delta and restframe, "
                "stop at <K>/K0 = %.2f or %d launches ===\n",
                Gp0, (unsigned)N, lambda, DT, stop_frac, n_max);
    if (Gp0 * 0.5 + 0.5 < T.bath->gamma_th()) {
      // K/K0 = 0.5 <=> Gamma = (Gamma_p0 + 1)/2: below the threshold for Gamma_p0 = 3 (Gamma = 2 < 2.125)
      std::printf("(diag) E2 target <K>/K0 = 0.5 is below the pair threshold at Gamma_p0 = %g (Gamma = %.4g < Gamma_th = %.4g): "
                  "not reachable by any run (fact); the run stops at %.2f\n",
                  Gp0, 0.5 * (Gp0 + 1.0), T.bath->gamma_th(), stop_frac);
    }
    std::printf("(diag) E[D^2] fixture: bhref-0.2.0/moments_mono05.txt, %zu nodes, data_sha256 %s\n", D2.gamma.size(),
                D2.sha.c_str());
    const int out_every = 25;
    for (const int spectrum : { DELTA, RESTFRAME }) {
      const std::string name = spectrum == DELTA ? "delta" : "restframe";
      const std::string tag  = fmt("G%g %s", Gp0, name.c_str());
      const auto S = cooling_run(T, EVENT, spectrum, 1.0, 0.01, coeff, DT, Gp0, N, stop_frac, n_max, out_every, 1000u + (unsigned)Gp0);
      print_series(tag, S);
      auto d2 = spectrum == DELTA ? std::function<double(double)>([&](double G) { return d2_delta(T, G); })
                                  : std::function<double(double)>([&](double G) { return D2.at(G); });
      const auto R = reference(T, S, d2);
      verdict_E2(tag, S, R);
      verdict_E3(tag, S, R);
      verdict(S.C.n_bounded == 0 && S.C.n_dropped == 0 && S.C.n_capped_events == 0,
              tag + ": no bounded, dropped or capped event", static_cast<double>(S.C.n_bounded + S.C.n_dropped + S.C.n_capped_events),
              0.0);
      // the small-jump statement: the continuous mode on the same ions, dt and coefficient
      if (spectrum == DELTA) {
        const auto Sc = cooling_run(T, CONTINUOUS, DELTA, 1.0, 0.01, coeff, DT, Gp0, N, 0.0, S.n_launch, out_every, 7u, 1.0e-5);
        print_series(tag + " continuous (eta_u = 1e-5)", Sc);
        for (const double f : { 0.9, 0.7, 0.5 }) {
          const std::size_t i = first_below(S, f);
          if (i == std::string::npos || i >= Sc.meanK.size()) {
            continue;
          }
          const double se = std::sqrt(S.varK[i] / static_cast<double>(S.N));
          verdict(std::abs(S.meanK[i] - Sc.meanK[i]) <= 4.0 * se,
                  tag + fmt(": E2 vs the continuous mode at <K>/K0 = %.1f (t = %.6g): |<K>_event - <K>_continuous| / s.e.", f,
                            S.t[i]),
                  std::abs(S.meanK[i] - Sc.meanK[i]) / se, 4.0);
        }
      }
    }
  }

  /**
   * @brief E6: fixed physical time, dt x {1, 2, 4, 8} (lambda 0.05 to 0.4 at Gamma_p0 = 6),
   *        delta, a = 1: the dt x 1 run passes E2 / E3 (checked); the deviations of
   *        <K> and Var K at the final time from the dt x 1 values fitted by a line
   *        through the origin in (dt - dt_1); residuals within 4 sqrt(2) s.e.;
   *        n_capped_events = 0 at dt x 1
   */
  void test_E6(Tables& T, npart_t N) {
    const double Gp0 = 6.0, lambda1 = 0.05;
    const double coeff = coeff_for_lambda(T, Gp0, lambda1, 1.0);
    const int    n1    = 4000; // launches at dt x 1: t_end = 2000 (about <K>/K0 ~ 0.85)
    std::printf("\n=== E6: dt x {1, 2, 4, 8} at fixed t_end = %g, Gamma_p0 = 6, delta, a = 1, N = %u, lambda 0.05 to 0.4 ===\n",
                n1 * DT, (unsigned)N);
    std::vector<Series> runs;
    for (const int k : { 1, 2, 4, 8 }) {
      const auto S = cooling_run(T, EVENT, DELTA, 1.0, 0.01, coeff, DT * k, Gp0, N, 0.0, n1 / k, std::max(1, 200 / k),
                                 5000u + k);
      print_series(fmt("E6 dt x %d", k), S);
      runs.push_back(S);
    }
    {
      auto       d2 = std::function<double(double)>([&](double G) { return d2_delta(T, G); });
      const auto R  = reference(T, runs[0], d2);
      verdict_E2("E6 dt x 1", runs[0], R);
      verdict_E3("E6 dt x 1", runs[0], R);
      verdict(runs[0].C.n_capped_events == 0, "E6: n_capped_events = 0 at dt x 1", static_cast<double>(runs[0].C.n_capped_events), 0.0);
    }
    // deviations at the final time
    const double se_m = std::sqrt(runs[0].varK.back() / static_cast<double>(N)) * std::sqrt(2.0);
    const double se_v = runs[0].varK.back() * std::sqrt(2.0 / static_cast<double>(N)) * std::sqrt(2.0);
    double sxx = 0.0, sxm = 0.0, sxv = 0.0;
    std::vector<double> x, dm, dv;
    for (std::size_t j = 1; j < runs.size(); ++j) {
      x.push_back(runs[j].dt - runs[0].dt);
      dm.push_back(runs[j].meanK.back() - runs[0].meanK.back());
      dv.push_back(runs[j].varK.back() - runs[0].varK.back());
      sxx += x.back() * x.back();
      sxm += x.back() * dm.back();
      sxv += x.back() * dv.back();
    }
    const double sm = sxm / sxx, sv = sxv / sxx;
    double rm = 0.0, rv = 0.0;
    for (std::size_t j = 0; j < x.size(); ++j) {
      rm = std::max(rm, std::abs(dm[j] - sm * x[j]));
      rv = std::max(rv, std::abs(dv[j] - sv * x[j]));
      std::printf("(diag) E6: dt x %g: <K> - <K>_1 = %.5g (%.2f s.e.), Var - Var_1 = %.5g (%.2f s.e.)\n", runs[j + 1].dt / runs[0].dt,
                  dm[j], dm[j] / se_m, dv[j], dv[j] / se_v);
    }
    std::printf("(diag) E6: fitted slopes d<K>/d(dt) = %.5g per unit dt (%.3g s.e. per unit dt), dVar/d(dt) = %.5g (%.3g s.e.)\n",
                sm, se_m, sv, se_v);
    verdict(rm <= 4.0 * se_m, "E6: <K> deviations consistent with linear growth in dt, max residual of the fit / s.e.", rm / se_m, 4.0);
    verdict(rv <= 4.0 * se_v, "E6: Var K deviations consistent with linear growth in dt, max residual of the fit / s.e.", rv / se_v, 4.0);
  }

  /**
   * @brief E7: a in {1, 2, 4} at f_max = 0.01 (registered) pass E2 at Gamma_p0 = 6 (delta,
   *        the bound cannot fire with D = <D> at these a; restframe reported with its
   *        bounded fraction); a in {10, 30} at f_max = 0.1 run as a labelled
   *        diagnostic of the bias scaling (not passed or failed); the variance
   *        against a R E[D^2] is checked for every a (contract section 1 (iii))
   */
  void test_E7(Tables& T, const D2Table& D2, npart_t N) {
    const double Gp0 = 6.0, lambda = 0.05;
    std::printf("\n=== E7: a in {1, 2, 4} at f_max = 0.01 (registered), {10, 30} at f_max = 0.1 (diagnostic), Gamma_p0 = 6, "
                "N = %u, R dt = %.3g at Gamma_p0 (lambda = R dt / a) ===\n", (unsigned)N, lambda);
    const double Rdt = lambda; // R dt at Gamma_p0; coeff from it
    const double coeff = coeff_for_lambda(T, Gp0, Rdt, 1.0);
    struct Case {
      double a, f_max;
      bool   registered;
    };
    for (const Case c : { Case { 1.0, 0.01, true }, Case { 2.0, 0.01, true }, Case { 4.0, 0.01, true }, Case { 10.0, 0.1, false },
                          Case { 30.0, 0.1, false } }) {
      for (const int spectrum : { DELTA, RESTFRAME }) {
        const std::string name = spectrum == DELTA ? "delta" : "restframe";
        const std::string tag  = fmt("E7 a = %g f_max = %g %s%s", c.a, c.f_max, name.c_str(), c.registered ? "" : " [diagnostic]");
        const auto S = cooling_run(T, EVENT, spectrum, c.a, c.f_max, coeff, DT, Gp0, N, 0.5, 40000, 25, 9000u + (unsigned)(10 * c.a));
        print_series(tag, S);
        auto d2 = spectrum == DELTA ? std::function<double(double)>([&](double G) { return d2_delta(T, G); })
                                    : std::function<double(double)>([&](double G) { return D2.at(G); });
        const auto R = reference(T, S, d2);
        if (c.registered && spectrum == DELTA) {
          verdict_E2(tag, S, R);
          verdict_E3(tag, S, R);
        } else {
          // reported, not passed or failed
          for (const double f : { 0.9, 0.7, 0.5 }) {
            const std::size_t i = first_below(S, f);
            if (i == std::string::npos) {
              continue;
            }
            const double se = std::sqrt(S.varK[i] / static_cast<double>(S.N));
            std::printf("(diag) %s: at <K>/K0 = %.1f: <K> - K_ODE,corrected = %.5g (%.2f s.e.), relative %.3e; bounded fraction "
                        "%.3e\n",
                        tag.c_str(), f, S.meanK[i] - R.K_corr[i], (S.meanK[i] - R.K_corr[i]) / se,
                        (S.meanK[i] - R.K_corr[i]) / S.meanK[i],
                        S.C.n_events > 0 ? static_cast<double>(S.C.n_bounded) / static_cast<double>(S.C.n_events) : 0.0);
          }
          const std::size_t i = S.t.size() - 1;
          std::printf("(diag) %s: final Var_meas / (a int R E[D^2]) = %.5g [no feedback], / Var_fb = %.5g [feedback]\n",
                      tag.c_str(), R.var_pred[i] > 0.0 ? S.varK[i] / R.var_pred[i] : 0.0,
                      R.var_fb[i] > 0.0 ? S.varK[i] / R.var_fb[i] : 0.0);
        }
      }
    }
  }

  /* ------------------------------------------------------------------------ */
  /* stage 4: E4, the pair spectra of the event mode against the frozen chains  */
  /* ------------------------------------------------------------------------ */

  /**
   * @brief E4 (contract section 4): restframe E-, E+, cos-, cos+ and D histograms
   *        of event-mode pairs at nearly fixed Gamma_p in {3, 10} against the
   *        frozen bhref 0.2.0 *unweighted* chains (hist_mono05_G3 / G10; D at
   *        Gamma_p = 10 on the supplement's log bins), with the calibrated AD
   *        protocol of A10 (common partition at >= 100 counts in both samples,
   *        Gaussian-multiplier null with 19,999 replicates, pass iff p >= 0.01,
   *        >= 95 % of the family of 10 comparisons), N >= 1e6 accepted events.
   *        Fixed Gamma_p: N_ion ions at Gamma_p, lambda = 0.2 per launch, the
   *        ions reset to K0 after every launch (the kernel is the event branch;
   *        within a launch the later events of an ion sit at a Gamma_p lower by
   *        at most 8 <D>/K ~ 1 %; the mean shift of the sampled Gamma_p is the
   *        fraction of second-or-later events, 1 - P(1)/E[n] = 18 % at lambda =
   *        0.2, times <D>/K: 2.5e-4 at Gamma_p = 3, 1.3e-4 at 10, below the
   *        contract's 0.1 %). Unweighted statistics: a = a2 = counts.
   */
  void test_E4(Tables& T, npart_t n_ion, int n_launch) {
    ensure_sampler(T);
    std::printf("\n=== E4: event-mode restframe pair spectra at Gamma_p in {3, 10} vs the frozen unweighted chains "
                "(bhref 0.2.0), calibrated AD (COMPARISON.md), %u ions x %d launches at lambda = 0.2, ions reset to K0 "
                "after every launch ===\n", (unsigned)n_ion, n_launch);
    struct Combo {
      std::string what;
      cal::Result r;
    };
    std::vector<Combo> combos;
    const char* const  qname[5] = { "E-", "E+", "cos-", "cos+", "D" };
    int                case_index = 0;
    for (const double g : { 3.0, 10.0 }) {
      const std::string gs = g == 3.0 ? "3" : "10";
      // the reference: unweighted counts (column 4) per quantity; D at Gamma_p >= 10 from the log-bin supplement
      std::vector<std::vector<double>> edges(5), rn(5);
      auto read_ref = [&](const std::string& file, bool only_D) {
        const bh::FixtureContents F = bh::read_fixture_file(std::string(BH_TEST_DATA_DIR) + "/" + file);
        for (const auto& r : F.rows) {
          const int q = static_cast<int>(r[0]);
          if ((only_D && q != 4) || (!only_D && q == 4 && g >= 10.0)) {
            continue;
          }
          if (edges[q].empty()) {
            edges[q].push_back(r[2]);
          }
          edges[q].push_back(r[3]);
          rn[q].push_back(r[4]); // unweighted_count
        }
      };
      read_ref("bhref-0.2.0/hist_mono05_G" + gs + ".txt", false);
      if (g >= 10.0) {
        read_ref("bhref-0.2.0-suppl1/hist_logD_mono05_G" + gs + ".txt", true);
      }
      double n_ref = 0.0;
      for (const double c : rn[0]) {
        n_ref += c;
      }
      // the run
      Ions                ions(n_ion, 1.0);
      std::vector<double> K0(n_ion, K_of_gamma(g));
      Pairs               pairs(static_cast<npart_t>(n_ion));
      const double        coeff = coeff_for_lambda(T, g, 0.2, 1.0);
      auto                prm   = params(coeff, RESTFRAME, 1.0, 0.01);
      random_number_pool_t pool(31415u + static_cast<unsigned>(g));
      std::vector<std::vector<double>> kn(5);
      for (int q = 0; q < 5; ++q) {
        kn[q].assign(edges[q].size() - 1, 0.0);
      }
      unsigned long long n_pairs = 0, outside = 0, n_second = 0, n_events_total = 0;
      double             sum_gp_shift = 0.0;
      HostLedger         H;
      for (int n = 0; n < n_launch; ++n) {
        ions.set_all(K0); // fixed Gamma_p: reset before every launch
        const auto L = launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT);
        H.add(L);
        n_events_total += L.n_events;
        const auto rows = pairs.rows();
        std::map<int, int> per_ion;
        for (const auto& r : rows) {
          if (!(r.birth_m > 0.0)) {
            continue;
          }
          per_ion[r.i1m] += 1;
          const double gm = kh::lepton_gamma(r.u1m, r.u2m, r.u3m), gp = kh::lepton_gamma(r.u1p, r.u2p, r.u3p);
          const double um = std::sqrt(r.u1m * r.u1m + r.u2m * r.u2m + r.u3m * r.u3m);
          const double up = std::sqrt(r.u1p * r.u1p + r.u2p * r.u2p + r.u3p * r.u3p);
          const double v[5] = { gm, gp, r.u1m / um, r.u1p / up, gm + gp - r.eps_m }; // the ion moves along +x
          for (int q = 0; q < 5; ++q) {
            const auto& e = edges[q];
            if (v[q] < e.front() || v[q] > e.back()) {
              ++outside;
            }
            int i = static_cast<int>(std::upper_bound(e.begin(), e.end(), v[q]) - e.begin()) - 1;
            i     = std::min(static_cast<int>(e.size()) - 2, std::max(0, i));
            kn[q][i] += 1.0;
          }
          ++n_pairs;
        }
        for (const auto& kv : per_ion) {
          if (kv.second > 1) {
            n_second += static_cast<unsigned long long>(kv.second - 1);
          }
        }
        pairs.recycle();
      }
      // the mean Gamma_p shift of the sampled events (second-or-later events of an ion within a launch sit
      // lower by about <D>/K each): reported against the contract's 0.1 %
      {
        double Dm = 0.0, e = 0.0;
        bh::interpolate_outcome_host(T.host, g, Dm, e);
        sum_gp_shift = n_pairs > 0 ? static_cast<double>(n_second) / static_cast<double>(n_pairs) * Dm / K_of_gamma(g) : 0.0;
      }
      diag(fmt("E4 Gamma_p %s: %llu pairs (reference %.0f unweighted events), %llu events drawn, second-or-later events "
               "%llu (mean relative Gamma_p - 1 shift of the sampled events ~ %.2e, contract < 1e-3), values outside the "
               "reference's edges %llu, n_bounded %llu, n_sampler_overflow %llu",
               gs.c_str(), n_pairs, n_ref, n_events_total, n_second, sum_gp_shift, outside, H.C.n_bounded,
               H.C.n_sampler_overflow));
      verdict(n_pairs >= 1000000ULL, fmt("E4 Gamma_p %s: at least 1e6 accepted events", gs.c_str()), static_cast<double>(n_pairs),
              1.0e6);
      verdict(sum_gp_shift < 1.0e-3, fmt("E4 Gamma_p %s: mean relative shift of the sampled Gamma_p below the contract's 0.1 %%",
                                        gs.c_str()),
              sum_gp_shift, 1.0e-3);
      for (int q = 0; q < 5; ++q) {
        const cal::Partition P = cal::merge_partition({ kn[q], rn[q] }, 100.0);
        const auto           a = cal::merge_sums(P, kn[q]);
        const auto           b = cal::merge_sums(P, rn[q]);
        const cal::Result    r = cal::calibrated(a, a, b, b, 2026102905ULL + 10ULL * case_index + static_cast<unsigned>(q));
        const std::string what = "Gamma_p " + gs + " " + qname[q] + (q == 4 && g >= 10.0 ? " (log bins)" : "");
        combos.push_back({ what, r });
        if (r.insufficient) {
          diag("E4 " + what + fmt(": ESS_A %.0f ESS_B %.0f, 1 merged bin: INSUFFICIENT RESOLUTION (not a pass)", r.ess_a, r.ess_b));
        } else {
          diag("E4 " + what + fmt(": N_A %.0f N_B %.0f T_AD %.4e p %.5f q99 %.4e merged bins %zu%s", r.ess_a, r.ess_b, r.T_AD, r.p,
                                  r.q99, r.merged_bins, r.pass ? "" : " FAIL"));
        }
      }
      ++case_index;
    }
    int         n_pass_c = 0;
    std::string failed;
    for (const auto& c : combos) {
      if (c.r.pass) {
        ++n_pass_c;
      } else {
        failed += " " + c.what + (c.r.insufficient ? " (insufficient resolution)" : fmt(" (p %.5f)", c.r.p)) + ";";
      }
    }
    const double frac = combos.empty() ? 0.0 : static_cast<double>(n_pass_c) / static_cast<double>(combos.size());
    verdict(!combos.empty() && frac >= 0.95,
            "E4: event-mode restframe pairs vs the frozen unweighted chains (E-, E+, cos-, cos+, D at Gamma_p 3 and 10; "
            "calibrated AD, pass iff p >= 0.01), fraction of the " +
              std::to_string(combos.size()) + " comparisons passing (>= 95 % required)",
            frac, 0.95);
    diag("E4: " + std::to_string(n_pass_c) + " of " + std::to_string(combos.size()) + " pass; failures:" +
         (failed.empty() ? std::string(" none") : failed));
  }

  /* ------------------------------------------------------------------------ */
  /* E8                                                                          */
  /* ------------------------------------------------------------------------ */
  void test_E8_bound(Tables& T) {
    std::printf("\n=== E8 (i) per-event bound: f_max = 1.1e-3 at the kernel, delta, 200 ions at Gamma_p = 3 (all events "
                "bounded) and 200 at Gamma_p = 6 (none), a = 1, lambda = 0.1 at Gamma_p = 6, 100 launches ===\n");
    const npart_t       N = 400;
    Ions                ions(N, 1.0);
    std::vector<double> K0(N);
    for (npart_t p = 0; p < N; ++p) {
      K0[p] = K_of_gamma(p < N / 2 ? 3.0 : 6.0);
    }
    ions.set_all(K0);
    const auto K_init = ions.K();
    double     D3, D6, e;
    bh::interpolate_outcome_host(T.host, 3.0, D3, e);
    bh::interpolate_outcome_host(T.host, 6.0, D6, e);
    diag(fmt("E8 (i): <D>/K = %.4e at Gamma_p = 3 and %.4e at Gamma_p = 6 against f_max = 1.1e-3", D3 / K_init[0], D6 / K_init[N - 1]));
    Pairs      pairs(200000);
    HostLedger H;
    auto       prm = params(coeff_for_lambda(T, 6.0, 0.1, 1.0), DELTA, 1.0, 1.1e-3);
    random_number_pool_t pool(4242u);
    for (int n = 0; n < 100; ++n) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
    }
    const auto Kf   = ions.K();
    int        n_changed_low = 0, n_pairs_low = 0;
    for (npart_t p = 0; p < N / 2; ++p) {
      if (Kf[p] != K_init[p]) {
        ++n_changed_low;
      }
    }
    for (const auto& r : pairs.rows()) {
      if (r.birth_m > 0.0 && r.i1m < static_cast<int>(N / 2)) {
        ++n_pairs_low;
      }
    }
    verdict(H.C.n_bounded > 0 && n_changed_low == 0, "E8 (i): the Gamma_p = 3 ions keep their stored u bitwise (ions changed)",
            static_cast<double>(n_changed_low), 0.0);
    verdict(n_pairs_low == 0, "E8 (i): no pair attributed to a Gamma_p = 3 ion", static_cast<double>(n_pairs_low), 0.0);
    verdict(H.C.n_bounded == H.C.n_events - H.C.n_spawned && H.C.n_dropped == 0,
            "E8 (i): n_bounded = n_events - n_spawned (no drops)", static_cast<double>(H.C.n_bounded),
            static_cast<double>(H.C.n_events - H.C.n_spawned));
    // the bounded fraction equals the share of the Gamma_p = 3 ions in the drawn events: lambda_3 / (lambda_3 + lambda_6)
    const double l3 = prm.tau0_over_skindepth0 * bh::interpolate_rate_host(T.host, 3.0) * DT;
    const double l6 = prm.tau0_over_skindepth0 * bh::interpolate_rate_host(T.host, 6.0) * DT;
    const double expect = l3 / (l3 + l6);
    const double frac   = static_cast<double>(H.C.n_bounded) / static_cast<double>(H.C.n_events);
    const double sig    = std::sqrt(expect * (1.0 - expect) / static_cast<double>(H.C.n_events));
    verdict(std::abs(frac - expect) <= 4.0 * sig, "E8 (i): bounded fraction vs the Gamma_p = 3 share of the rate, |diff| / sigma",
            std::abs(frac - expect) / sig, 4.0);
    const auto A = closure(ions, K_init, pairs, T, prm.pair_weight);
    check_E1("E8 (i)", H, A, ions);
  }

  void test_E8_capacity(Tables& T) {
    std::printf("\n=== E8 (ii) capacity: pair capacity 300 for about 2000 events due, delta, 200 ions at Gamma_p = 6, "
                "lambda = 0.1, 100 launches ===\n");
    const npart_t       N = 200;
    Ions                ions(N, 1.0);
    std::vector<double> K0(N, K_of_gamma(6.0));
    ions.set_all(K0);
    const auto K_init = ions.K();
    Pairs      pairs(300);
    HostLedger H;
    auto       prm = params(coeff_for_lambda(T, 6.0, 0.1, 1.0), DELTA);
    random_number_pool_t pool(777u);
    for (int n = 0; n < 100; ++n) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
    }
    verdict(H.C.n_dropped > 0, "E8 (ii): n_dropped > 0", static_cast<double>(H.C.n_dropped), 1.0);
    verdict(H.C.n_spawned == 300 && pairs.npart_m == 300 && pairs.npart_p == 300,
            "E8 (ii): n_spawned = capacity = 300 and both pair species are full", static_cast<double>(H.C.n_spawned), 300.0);
    const auto A = closure(ions, K_init, pairs, T, prm.pair_weight);
    check_E1("E8 (ii)", H, A, ions);
  }

  void test_E8_cap(Tables& T) {
    std::printf("\n=== E8 (iii) event cap: max_events_per_step = 1 at lambda = 1, delta, 200 ions at Gamma_p = 6, "
                "50 launches ===\n");
    const npart_t       N = 200;
    Ions                ions(N, 1.0);
    std::vector<double> K0(N, K_of_gamma(6.0));
    ions.set_all(K0);
    const auto K_init = ions.K();
    Pairs      pairs(200000);
    HostLedger H;
    auto       prm = params(coeff_for_lambda(T, 6.0, 1.0, 1.0), DELTA, 1.0, 0.01, 1);
    random_number_pool_t pool(2024u);
    for (int n = 0; n < 50; ++n) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
    }
    // at most one pair per (ion, launch)
    std::map<std::pair<int, int>, int> per_cell;
    for (const auto& r : pairs.rows()) {
      if (r.birth_m > 0.0) {
        per_cell[{ r.i1m, static_cast<int>(std::lround(r.birth_m / DT)) }] += 1;
      }
    }
    int max_per_cell = 0;
    for (const auto& kv : per_cell) {
      max_per_cell = std::max(max_per_cell, kv.second);
    }
    const double cells  = static_cast<double>(N) * 50.0;
    const double p_cap  = 1.0 - 2.0 * std::exp(-1.0); // P(n >= 2) at lambda = 1
    const double sig    = std::sqrt(cells * p_cap * (1.0 - p_cap));
    verdict(H.C.n_capped_events > 0, "E8 (iii): n_capped_events > 0", static_cast<double>(H.C.n_capped_events), 1.0);
    verdict(std::abs(static_cast<double>(H.C.n_capped_events) - cells * p_cap) <= 4.0 * sig,
            "E8 (iii): capped ion-steps within 4 sigma of N_cells P(n >= 2), |diff| / sigma",
            std::abs(static_cast<double>(H.C.n_capped_events) - cells * p_cap) / sig, 4.0);
    verdict(max_per_cell == 1, "E8 (iii): at most one pair per (ion, launch)", static_cast<double>(max_per_cell), 1.0);
    verdict(H.C.n_events <= static_cast<unsigned long long>(cells), "E8 (iii): n_events <= ions x launches",
            static_cast<double>(H.C.n_events), cells);
    const auto A = closure(ions, K_init, pairs, T, prm.pair_weight);
    check_E1("E8 (iii)", H, A, ions);
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::printf("bh_event: real_t = %s, Kokkos default execution space = %s, tolerances: closure %.1e, identity %.1e, "
                "E_round %.1e, E5 K %.1e\n",
                SINGLE ? "float" : "double", Kokkos::DefaultExecutionSpace::name(), TOL_CLOSURE, TOL_IDENTITY, TOL_EROUND,
                TOL_E5_K);
    auto T = make_tables();
    std::vector<std::string> groups;
    for (int i = 1; i < argc; ++i) {
      if (argv[i][0] != '-') {
        groups.emplace_back(argv[i]);
      }
    }
    auto want = [&groups](const char* g) {
      return groups.empty() || std::find(groups.begin(), groups.end(), std::string(g)) != groups.end();
    };
    if (want("e1")) {
      test_E1(T, DELTA);
      test_E1(T, RESTFRAME);
    }
    if (want("e5")) {
      test_E5(T, DELTA);
      test_E5(T, RESTFRAME);
    }
    if (want("e8")) {
      test_E8_bound(T);
      test_E8_capacity(T);
      test_E8_cap(T);
    }
    // stage 3: N from the environment for the development loop (BH_EVENT_N, default the registered 1e5)
    npart_t N_stat = 100000;
    if (const char* env = std::getenv("BH_EVENT_N")) {
      const long v = std::atol(env);
      if (v > 0 && v <= 10000000L) {
        N_stat = static_cast<npart_t>(v);
      }
    }
    if (want("e2e3")) {
      const auto D2 = load_d2(T);
      test_E2E3(T, D2, 6.0, N_stat, 40000, 0.5);
    }
    if (want("e2e3_g3")) {
      const auto D2 = load_d2(T);
      test_E2E3(T, D2, 3.0, N_stat, 150000, 0.69);
    }
    if (want("e6")) {
      test_E6(T, N_stat);
    }
    if (want("e7")) {
      const auto D2 = load_d2(T);
      test_E7(T, D2, N_stat);
    }
    if (want("e4")) {
      // 5e5 ions x 11 launches at lambda = 0.2: about 1.1e6 accepted events per Gamma_p
      test_E4(T, 500000, 11);
    }
    std::printf("bh_event summary: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0 || n_pass == 0) {
      rc = 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_event: exception: %s\n", e.what());
    rc = 2;
  }
  Kokkos::finalize();
  return rc;
}
