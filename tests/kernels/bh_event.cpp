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

#include "archetypes/hadronic/bh_interpolate.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_sampler_device.h"
#include "archetypes/hadronic/bh_tables_device.h"
#include "framework/containers/particles.h"
#include "kernels/hadronic/bethe_heitler.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;
namespace bh = arch::bh;
namespace kh = kernel::hadronic;

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
