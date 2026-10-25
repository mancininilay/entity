/**
 * @file tests/kernels/bh_pairs.cpp
 * @brief Pre-registered tests of the Bethe-Heitler pair plumbing (P2 step 3
 *        brief, section 3), kernel level: T10 (spawn statistics: budget,
 *        capped budget, closing packets, poisson, none), pairs per unit ion
 *        energy, T11 (ledger closure from the arrays), T12 (capacity), T10b
 *        (stale payloads of the pair species), T-exp (exported debt of dead
 *        ions). The kernel is launched directly on device arrays (no pusher,
 *        no fields: between launches the ions keep their stored u unless a test
 *        changes it on the host to emulate the fields).
 *
 * Conventions (declared before any run):
 *   - bath mono eps0 = 0.5 (Gamma_th = 2.125), mu_r = 1836.15, skindepth0 = 1,
 *     dt = 0.5, ions with mu_d = 1 at Gamma_p = 6 (K = 9180.75) unless stated,
 *     f_max = 0.01, eta_u = 1e-4, pending_floor_ulp = 100, delta spectrum
 *   - "ions at dK/K = 1e-4 per step": tau0/skindepth0 = 1e-4 K / (I(Gamma_p) dt)
 *   - the latency cap is off (t_latency = 0: periodic box) so that
 *     r_th = min(r_th_factor <D>, f_r K); with r_th_factor = 10, f_r = 0.02 the
 *     first term binds at Gamma_p = 6 (83.6 against 183.6)
 *   - "exact" means to double round-off: 1e-12 relative to E_drag
 *   - T10 packet count: the kernel's debt mechanism is replayed on the host
 *     from the per-step stored ion energies (the same thresholds, the same
 *     real_t rounding of the stored debt and K0, the same K0 trigger); the
 *     brief's constant-threshold formula floor(sum dK_applied / r_th) is
 *     printed as a diagnostic (r_th drifts with Gamma_p over the run)
 *   - every pair is attributed to its parent by (i1, dx1): ion p sits at
 *     i1 = p, dx1 = 0.5 and never moves
 *   - the array identity of T11: sum_pairs w_s (gamma- + gamma+ - eps_s) =
 *     sum_ions w (K_init - K_final) - sum_ions w r_stored - E_exported -
 *     E_lost_nopairs, every sum in double-double on the host from the stored
 *     values; its residual is compared with the ledger's E_round and with the
 *     rounding budget B printed by the run
 *
 * Verdict lines: "PASS" / "FAIL" with the measured number and the tolerance;
 * "(diag)" lines are reported only. One ctest entry per group.
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "metrics/minkowski.h"

#include "archetypes/hadronic/bh_interpolate.h"
#include "archetypes/hadronic/bh_physics.h"
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
#include <stdexcept>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;
namespace bh = arch::bh;
namespace kh = kernel::hadronic;

namespace {

  int n_pass = 0, n_fail = 0, n_expected_fail = 0;

  void verdict(bool ok, const std::string& what, double value, double tol, bool expected_fail = false) {
    if (expected_fail) {
      std::printf("%s (expected failure, labelled before the run): %s : %.6e (tolerance %.3e)\n",
                  ok ? "PASS" : "EXPECTED-FAIL", what.c_str(), value, tol);
      if (ok) {
        ++n_pass;
      } else {
        ++n_expected_fail;
      }
      return;
    }
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

  constexpr double MU_R = 1836.15;
  constexpr double EPS0 = 0.5;
  constexpr double DT   = 0.5;

  struct Tables {
    std::shared_ptr<const bh::PhotonBath>  bath;
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
    T.rates   = std::make_unique<bh::BetheHeitlerRates>(T.bath);
    T.host    = bh::build_tables(*T.rates, mu_r);
    bh::validate_tables(T.host);
    T.dev = bh::mirror_tables_to_device(T.host);
    return T;
  }

  double K_of_gamma(double Gp) {
    return MU_R * (Gp - 1.0);
  }

  /**
   * @brief tau0/skindepth0 such that dK/K = frac per step at Gamma_p
   */
  double coeff_for(const Tables& T, double Gp, double frac) {
    const double K = K_of_gamma(Gp);
    const double I = bh::interpolate_loss_host(T.host, Gp);
    return frac * K / (I * DT);
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
      // ion p at i1 = p, dx1 = 0.5 (unique position: parent attribution)
      auto hi = Kokkos::create_mirror_view(arr.i1);
      auto hd = Kokkos::create_mirror_view(arr.dx1);
      for (npart_t p = 0; p < n; ++p) {
        hi(p) = static_cast<int>(p);
        hd(p) = static_cast<prtldx_t>(0.5);
      }
      Kokkos::deep_copy(arr.i1, hi);
      Kokkos::deep_copy(arr.dx1, hd);
      Kokkos::deep_copy(arr.tag, ParticleTag::alive);
    }

    static double u_of_K(double K, double mu) {
      return std::sqrt(K * (K + 2.0 * mu)) / mu;
    }

    void set(npart_t p, double K, double w = 1.0, short tag = ParticleTag::alive,
             double c1 = 1.0, double c2 = 0.0, double c3 = 0.0) {
      auto h1 = mirror(arr.ux1), h2 = mirror(arr.ux2), h3 = mirror(arr.ux3);
      auto hw = mirror(arr.weight);
      auto ht = mirror(arr.tag);
      const double u = u_of_K(K, mu_d);
      h1(p)          = static_cast<real_t>(u * c1);
      h2(p)          = static_cast<real_t>(u * c2);
      h3(p)          = static_cast<real_t>(u * c3);
      hw(p)          = static_cast<real_t>(w);
      ht(p)          = tag;
      Kokkos::deep_copy(arr.ux1, h1);
      Kokkos::deep_copy(arr.ux2, h2);
      Kokkos::deep_copy(arr.ux3, h3);
      Kokkos::deep_copy(arr.weight, hw);
      Kokkos::deep_copy(arr.tag, ht);
    }

    void set_tag(npart_t p, short tag) {
      auto ht = mirror(arr.tag);
      ht(p)   = tag;
      Kokkos::deep_copy(arr.tag, ht);
    }

    void zero_payloads() {
      Kokkos::deep_copy(arr.pld_r, ZERO);
    }

    std::vector<double> K() const {
      auto h1 = mirror(arr.ux1), h2 = mirror(arr.ux2), h3 = mirror(arr.ux3);
      std::vector<double> out(n);
      for (npart_t p = 0; p < n; ++p) {
        out[p] = kh::kinetic_energy(mu_d, h1(p), h2(p), h3(p));
      }
      return out;
    }

    std::vector<double> payload(int slot) const {
      auto hp = mirror(arr.pld_r);
      std::vector<double> out(n);
      for (npart_t p = 0; p < n; ++p) {
        out[p] = static_cast<double>(hp(p, slot));
      }
      return out;
    }

    std::vector<double> w() const {
      auto hw = mirror(arr.weight);
      std::vector<double> out(n);
      for (npart_t p = 0; p < n; ++p) {
        out[p] = static_cast<double>(hw(p));
      }
      return out;
    }

    std::vector<short> tags() const {
      auto ht = mirror(arr.tag);
      return std::vector<short>(ht.data(), ht.data() + n);
    }
  };

  /* ------------------------------------------------------------------------ */
  /* pair destinations                                                          */
  /* ------------------------------------------------------------------------ */
  struct PairRow {
    npart_t k;
    int     i1m, i1p, i1pm, i1pp;
    double  dxm, dxp, dxpm, dxpp;
    double  wm, wp;
    double  u1m, u2m, u3m, u1p, u2p, u3p;
    short   tagm, tagp;
    double  birth_m, gp_m, eps_m, birth_p, gp_p, eps_p;
  };

  struct Pairs {
    ParticleArrays   minus, plus;
    npart_t          maxn { 0 };
    npart_t          npart_m { 0 }, npart_p { 0 };
    npart_t          counter_m { 0 }, counter_p { 0 };
    unsigned short   npld { 3 };
    array_t<npart_t> spawn_counter { "spawn_counter" };

    explicit Pairs(npart_t maxn_, unsigned short npld_ = 3) : maxn { maxn_ }, npld { npld_ } {
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
      a.pld_r    = array_t<real_t**> { s + "pld_r", maxn, npld };
      a.pld_i    = array_t<npart_t**> { s + "pld_i", maxn, 1 };
      a.phi      = array_t<real_t*> { s + "phi", maxn };
      Kokkos::deep_copy(a.tag, ParticleTag::dead);
    }

    void fill_payloads_nan() {
      Kokkos::deep_copy(minus.pld_r, std::numeric_limits<real_t>::quiet_NaN());
      Kokkos::deep_copy(plus.pld_r, std::numeric_limits<real_t>::quiet_NaN());
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
      t.npld_r     = npld;
      Kokkos::deep_copy(spawn_counter, static_cast<npart_t>(0));
      t.spawn_counter = spawn_counter;
      t.valid         = true;
      return t;
    }

    /**
     * @brief host bookkeeping after a launch; returns n_injected
     */
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
      auto i1m = mirror(minus.i1), i1p = mirror(plus.i1), i1pm = mirror(minus.i1_prev), i1pp = mirror(plus.i1_prev);
      auto dxm = mirror(minus.dx1), dxp = mirror(plus.dx1), dxpm = mirror(minus.dx1_prev), dxpp = mirror(plus.dx1_prev);
      auto wm = mirror(minus.weight), wp = mirror(plus.weight);
      auto u1m = mirror(minus.ux1), u2m = mirror(minus.ux2), u3m = mirror(minus.ux3);
      auto u1p = mirror(plus.ux1), u2p = mirror(plus.ux2), u3p = mirror(plus.ux3);
      auto tm = mirror(minus.tag), tp = mirror(plus.tag);
      auto pm = mirror(minus.pld_r), pp = mirror(plus.pld_r);
      for (npart_t k = 0; k < n; ++k) {
        PairRow r;
        r.k    = k;
        r.i1m  = i1m(k);
        r.i1p  = i1p(k);
        r.i1pm = i1pm(k);
        r.i1pp = i1pp(k);
        r.dxm  = dxm(k);
        r.dxp  = dxp(k);
        r.dxpm = dxpm(k);
        r.dxpp = dxpp(k);
        r.wm   = wm(k);
        r.wp   = wp(k);
        r.u1m  = u1m(k);
        r.u2m  = u2m(k);
        r.u3m  = u3m(k);
        r.u1p  = u1p(k);
        r.u2p  = u2p(k);
        r.u3p  = u3p(k);
        r.tagm = tm(k);
        r.tagp = tp(k);
        r.birth_m = pm(k, 0);
        r.gp_m    = pm(k, 1);
        r.eps_m   = pm(k, 2);
        r.birth_p = pp(k, 0);
        r.gp_p    = pp(k, 1);
        r.eps_p   = pp(k, 2);
        out.push_back(r);
      }
      return out;
    }
  };

  /* ------------------------------------------------------------------------ */
  /* host ledger and the run driver                                             */
  /* ------------------------------------------------------------------------ */
  struct HostLedger {
    kh::LossLedger C {};
    kh::dd         debt_initial {};

    void add(const kh::LossLedger& L) {
      const auto po = L.pending_outstanding;
      const auto dbo = L.debt_outstanding;
      const auto sw = L.sum_wK;
      const auto na = L.n_alive;
      C += L;
      C.pending_outstanding = po;
      C.debt_outstanding    = dbo;
      C.sum_wK              = sw;
      C.n_alive             = na;
    }

    double residual() const {
      kh::dd r;
      r.add(C.E_drag);
      r.add(debt_initial);
      r.sub(C.E_pairs);
      r.add(C.E_absorbed);
      r.sub(C.E_round);
      r.sub(C.debt_outstanding);
      r.sub(C.E_exported);
      r.sub(C.E_lost_nopairs);
      return r.value();
    }
  };

  kh::HadronicParams params(double coeff, int source, double r_th_factor = 10.0, double f_r = 0.02) {
    kh::HadronicParams prm;
    prm.tau0_over_skindepth0 = coeff;
    prm.mu_r                 = MU_R;
    prm.eta_u                = 1.0e-4;
    prm.pending_floor_ulp    = 100.0;
    prm.f_max                = 0.01;
    prm.max_substeps         = 64;
    prm.dt                   = DT;
    prm.source               = source;
    prm.spectrum             = static_cast<int>(kh::PairSpectrum::Delta);
    prm.r_th_factor          = r_th_factor;
    prm.f_r                  = f_r;
    prm.t_latency            = 0.0;
    prm.pair_weight          = 1.0;
    prm.max_events_per_step  = 8;
    return prm;
  }

  constexpr int BUDGET  = static_cast<int>(kh::PairSource::Budget);
  constexpr int POISSON = static_cast<int>(kh::PairSource::Poisson);
  constexpr int NONE    = static_cast<int>(kh::PairSource::None);

  /**
   * @brief one launch; the ledger of the launch is returned, the host does the
   *        pair bookkeeping
   */
  kh::LossLedger launch(Ions& ions, Pairs& pairs, const Tables& T, kh::HadronicParams prm,
                        random_number_pool_t& pool, double time_after) {
    prm.time   = time_after;
    auto targ  = pairs.targets();
    kh::LossLedger L;
    Kokkos::parallel_reduce("bh_pairs", CreateParticleRangePolicy<Dim::_1D>({ 0u }, { ions.n }),
                            kh::BetheHeitler_kernel<Minkowski<Dim::_1D>> { ions.arr, ions.mu_d, T.dev, prm, targ,
                                                                          pool },
                            Kokkos::Sum<kh::LossLedger>(L));
    Kokkos::fence();
    const auto n_inj = pairs.finish(targ);
    if (static_cast<unsigned long long>(n_inj) != L.n_spawned) {
      throw std::runtime_error("n_injected != n_spawned");
    }
    return L;
  }

  /**
   * @brief the array identity of T11 (double-double on the host)
   */
  struct ArrayIdentity {
    double lhs, rhs, residual, sum_wpair, n_pairs;
  };

  ArrayIdentity array_identity(const Ions& ions, const std::vector<double>& K_init, const Pairs& pairs,
                               const HostLedger& H) {
    kh::dd lhs, rhs, swp;
    const auto rows = pairs.rows();
    double     np   = 0.0;
    for (const auto& r : rows) {
      if (!(r.birth_m > 0.0)) {
        continue;
      }
      const double gm  = kh::lepton_gamma(r.u1m, r.u2m, r.u3m);
      const double gp  = kh::lepton_gamma(r.u1p, r.u2p, r.u3p);
      const double E_s = gm + gp;
      lhs.add_prod(r.wm, E_s);
      lhs.add_prod(-r.wm, r.eps_m);
      swp.add(r.wm);
      np += 1.0;
    }
    const auto Kf = ions.K();
    const auto w  = ions.w();
    const auto rr = ions.payload(1);
    const auto tg = ions.tags();
    for (npart_t p = 0; p < ions.n; ++p) {
      rhs.add_prod(w[p], K_init[p] - Kf[p]);
      if (tg[p] != ParticleTag::dead) {
        rhs.add_prod(-w[p], rr[p]);
      }
    }
    rhs.sub(H.C.E_exported);
    rhs.sub(H.C.E_lost_nopairs);
    rhs.add(H.debt_initial);
    kh::dd res = rhs;
    res.sub(lhs);
    return { lhs.value(), rhs.value(), res.value(), swp.value(), np };
  }

  void check_identity(const std::string& tag, const HostLedger& H, const ArrayIdentity& A, bool check_B = true) {
    const double E = std::abs(H.C.E_drag.value());
    verdict(std::abs(H.residual()) <= 1.0e-12 * E, tag + ": ledger identity residual / E_drag",
            E > 0.0 ? std::abs(H.residual()) / E : std::abs(H.residual()), 1.0e-12);
    const double B = H.C.B_budget.value();
    if (check_B) {
      verdict(std::abs(A.residual) <= 3.0 * B, tag + ": array identity |residual| / B (B = " + std::to_string(B) + ")",
              B > 0.0 ? std::abs(A.residual) / B : std::abs(A.residual), 3.0);
    }
    const double er   = H.C.E_round.value();
    const double diff = std::abs(A.residual - er);
    const bool   ok   = (er == 0.0 && A.residual == 0.0) || diff <= 1.0e-9 * std::abs(er);
    verdict(ok, tag + ": E_round = array residual, |E_round - residual| / |E_round| (E_round = " +
                  std::to_string(er) + ")",
            er != 0.0 ? diff / std::abs(er) : diff, 1.0e-9);
  }

  /* ------------------------------------------------------------------------ */
  /* host replay of the debt mechanism (T10)                                    */
  /* ------------------------------------------------------------------------ */
  struct Packet {
    double time, r_packet, Gp, D, w_pair;
    bool   via_k0;
  };

  struct Replay {
    std::vector<Packet> packets;
    double              r_final { 0.0 };
    unsigned long long  n_capped { 0 };
  };

  /**
   * @brief replays the kernel's steps 2-4 for one ion from its per-step stored
   *        energies K_n (n = 0 .. nsteps), with the stored-float rounding of r
   *        and K0 and the kernel's trigger rules
   */
  Replay replay_ion(const Tables& T, const std::vector<double>& Ks, double w, const kh::HadronicParams& prm,
                    bool dK_positive_every_step = true) {
    Replay R;
    double r_stored = 0.0, K0 = 0.0;
    for (std::size_t n = 1; n < Ks.size(); ++n) {
      const double dK_applied = Ks[n - 1] - Ks[n];
      kh::dd       r;
      r.add(r_stored);
      if (dK_applied > 0.0) {
        if (r_stored == 0.0) {
          K0 = static_cast<double>(static_cast<real_t>(Ks[n]));
        }
        r.add(dK_applied);
      }
      const double K_cur = Ks[n];
      double       Gp    = 1.0 + K_cur / MU_R;
      if (Gp > T.host.gamma_max) {
        Gp = T.host.gamma_max;
      }
      if (dK_positive_every_step && r.value() > 0.0 && Gp >= T.host.gamma_min) {
        double Dm, eps;
        bh::interpolate_outcome_host(T.host, Gp, Dm, eps);
        double r_th = prm.r_th_factor * Dm;
        if (prm.f_r * K_cur < r_th) {
          r_th = prm.f_r * K_cur;
        }
        bool k0_trigger = (K0 > 0.0) && (std::abs(K_cur - K0) > prm.f_r * K0);
        int  n_pk       = 0;
        while (true) {
          const double rv = r.value();
          if (!(rv > 0.0)) {
            break;
          }
          if (n_pk >= prm.max_events_per_step) {
            if (rv >= r_th) {
              R.n_capped += 1;
            }
            break;
          }
          double r_packet;
          bool   via = false;
          if (rv >= r_th) {
            r_packet = r_th;
          } else if (k0_trigger) {
            r_packet = rv;
            via      = true;
          } else {
            break;
          }
          R.packets.push_back({ static_cast<double>(n) * DT, r_packet, Gp, Dm, w * r_packet / Dm, via });
          r.add(-r_packet);
          k0_trigger = false;
          ++n_pk;
        }
      }
      r_stored = static_cast<double>(static_cast<real_t>(r.value()));
      if (r_stored == 0.0) {
        K0 = 0.0;
      }
    }
    R.r_final = r_stored;
    return R;
  }

  /* ------------------------------------------------------------------------ */
  /* T10: spawn statistics                                                       */
  /* ------------------------------------------------------------------------ */
  void test_T10_budget(const Tables& T) {
    std::printf("\n=== T10 budget + delta: 8 ions at Gamma_p = 6, dK/K = 1e-4 per step, 2000 steps, "
                "r_th_factor = 10, f_r = 0.02, t_latency = 0 ===\n");
    const npart_t N = 8;
    Ions          ions(N, 1.0);
    const double  K6 = K_of_gamma(6.0);
    std::vector<double> ws;
    for (npart_t p = 0; p < N; ++p) {
      const double w = (p % 2 == 0) ? 1.0 : 2.5;
      ions.set(p, K6, w);
      ws.push_back(static_cast<double>(static_cast<real_t>(w)));
    }
    ions.zero_payloads();
    Pairs                pairs(200000);
    random_number_pool_t pool(42u);
    auto                 prm = params(coeff_for(T, 6.0, 1.0e-4), BUDGET);
    HostLedger           H;
    const int            nsteps = 2000;
    std::vector<std::vector<double>> Khist(N);
    {
      const auto K0s = ions.K();
      for (npart_t p = 0; p < N; ++p) {
        Khist[p].push_back(K0s[p]);
      }
    }
    const auto K_init = ions.K();
    for (int n = 0; n < nsteps; ++n) {
      const auto L = launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT);
      H.add(L);
      const auto Ks = ions.K();
      for (npart_t p = 0; p < N; ++p) {
        Khist[p].push_back(Ks[p]);
      }
    }
    const auto rows = pairs.rows();
    diag("T10 budget: n_spawned " + std::to_string(H.C.n_spawned) + ", n_events " + std::to_string(H.C.n_events) +
         ", n_dropped " + std::to_string(H.C.n_dropped) + ", n_k0_trigger " + std::to_string(H.C.n_k0_trigger) +
         ", n_capped " + std::to_string(H.C.n_capped) + ", n_rng " + std::to_string(H.C.n_rng) +
         ", npart e- " + std::to_string(pairs.npart_m) + ", e+ " + std::to_string(pairs.npart_p));
    // (1) per-ion packet count and weights against the replay
    const auto r_fin = ions.payload(1);
    const auto k0_fin = ions.payload(2);
    unsigned long long total_replay = 0;
    bool count_ok = true, w_ok = true, gp_ok = true, time_ok = true, rfin_ok = true;
    double max_w_err = 0.0, max_gp_err = 0.0, max_rfin_err = 0.0;
    double sum_dK_all = 0.0;
    for (npart_t p = 0; p < N; ++p) {
      const auto R = replay_ion(T, Khist[p], ws[p], prm);
      total_replay += R.packets.size();
      std::vector<PairRow> mine;
      for (const auto& r : rows) {
        if (r.i1m == static_cast<int>(p)) {
          mine.push_back(r);
        }
      }
      auto by_time = [](const PairRow& a, const PairRow& b) {
        return a.birth_m < b.birth_m || (a.birth_m == b.birth_m && a.wm < b.wm);
      };
      std::sort(mine.begin(), mine.end(), by_time);
      auto pk = R.packets;
      std::sort(pk.begin(), pk.end(), [](const Packet& a, const Packet& b) {
        return a.time < b.time || (a.time == b.time && a.w_pair < b.w_pair);
      });
      const double sum_dK = Khist[p].front() - Khist[p].back();
      sum_dK_all         += ws[p] * sum_dK;
      double Dm6, e6;
      bh::interpolate_outcome_host(T.host, 6.0, Dm6, e6);
      const double r_th6 = prm.r_th_factor * Dm6;
      std::printf("(diag) T10 ion %u (w = %.3g): kernel packets %zu, replay packets %zu (%llu on the K0 trigger), "
                  "sum dK_applied = %.6f, r_final stored %.6f (replay %.6f), K0 payload %.6f, brief formula "
                  "floor(sum dK / r_th(Gamma_p = 6)) = %d (r_th = %.4f at Gamma_p = 6; %.4f at the final "
                  "Gamma_p %.4f)\n",
                  (unsigned)p, ws[p], mine.size(), pk.size(),
                  (unsigned long long)std::count_if(pk.begin(), pk.end(), [](const Packet& q) { return q.via_k0; }),
                  sum_dK, r_fin[p], R.r_final, k0_fin[p], static_cast<int>(std::floor(sum_dK / r_th6)), r_th6,
                  prm.r_th_factor * [&] { double d, e; bh::interpolate_outcome_host(T.host, 1.0 + Khist[p].back() / MU_R, d, e); return d; }(),
                  1.0 + Khist[p].back() / MU_R);
      if (mine.size() != pk.size()) {
        count_ok = false;
        continue;
      }
      for (std::size_t k = 0; k < pk.size(); ++k) {
        const double we = std::abs(mine[k].wm - pk[k].w_pair) / pk[k].w_pair;
        max_w_err       = std::max(max_w_err, we);
        if (we > 1.0e-6) {
          w_ok = false;
        }
        const double ge = std::abs(mine[k].gp_m - pk[k].Gp) / pk[k].Gp;
        max_gp_err      = std::max(max_gp_err, ge);
        if (ge > 1.0e-6) {
          gp_ok = false;
        }
        if (mine[k].birth_m != static_cast<real_t>(pk[k].time)) {
          time_ok = false;
        }
      }
      max_rfin_err = std::max(max_rfin_err, std::abs(r_fin[p] - R.r_final));
      if (r_fin[p] != R.r_final) {
        rfin_ok = false;
      }
    }
    verdict(count_ok, "T10: packets per ion equal the host replay of the debt mechanism, every ion (total " +
                        std::to_string(total_replay) + ")",
            count_ok ? 0.0 : 1.0, 0.0);
    verdict(count_ok && w_ok, "T10: every w_pair = w r_packet / <D>(Gamma_p at spawn), max relative error",
            max_w_err, 1.0e-6);
    verdict(count_ok && gp_ok, "T10: pair payload Gamma_p = parent's Gamma_p at spawn (float), max relative error",
            max_gp_err, 1.0e-6);
    verdict(count_ok && time_ok, "T10: pair payload birth time = time + dt of the spawning step (exact)",
            time_ok ? 0.0 : 1.0, 0.0);
    verdict(rfin_ok, "T10: stored debt after 2000 steps = replay (bit-exact), max |difference|", max_rfin_err, 0.0);
    // (2) positions, prev, eps payload, tags, equal weights of the two leptons
    bool pos_ok = true, prev_ok = true, eps_ok = true, tag_ok = true, weq_ok = true, mom_ok = true;
    double max_eps_err = 0.0;
    for (const auto& r : rows) {
      pos_ok  = pos_ok && (r.i1m == r.i1p) && (r.dxm == r.dxp) && (r.dxm == 0.5) && r.i1m >= 0 &&
               r.i1m < static_cast<int>(N);
      prev_ok = prev_ok && (r.i1pm == r.i1m) && (r.dxpm == r.dxm) && (r.i1pp == r.i1p) && (r.dxpp == r.dxp);
      max_eps_err = std::max(max_eps_err, std::abs(r.eps_m - EPS0));
      eps_ok  = eps_ok && (r.eps_m == EPS0) && (r.eps_p == EPS0) && (r.birth_p == r.birth_m) && (r.gp_p == r.gp_m);
      tag_ok  = tag_ok && (r.tagm == ParticleTag::alive) && (r.tagp == ParticleTag::alive);
      weq_ok  = weq_ok && (r.wm == r.wp);
      // delta spectrum: collinear, equal momenta along +x (the ion direction)
      mom_ok = mom_ok && (r.u1m == r.u1p) && (r.u2m == 0.0) && (r.u3m == 0.0) && (r.u2p == 0.0) && (r.u3p == 0.0) &&
               r.u1m > 0.0;
    }
    verdict(pos_ok, "T10: pair positions (i, dx) equal the parent's, both leptons", pos_ok ? 0.0 : 1.0, 0.0);
    verdict(prev_ok, "T10: *_prev = * for both leptons", prev_ok ? 0.0 : 1.0, 0.0);
    verdict(eps_ok, "T10: pair payloads (birth, Gamma_p, eps = 0.5) identical on both leptons, max |eps - 0.5|",
            max_eps_err, 0.0);
    verdict(tag_ok, "T10: tags inherited from the parent (alive)", tag_ok ? 0.0 : 1.0, 0.0);
    verdict(weq_ok && mom_ok, "T10: equal stored weights and collinear equal momenta of the two leptons (delta)",
            (weq_ok && mom_ok) ? 0.0 : 1.0, 0.0);
    verdict(pairs.npart_m == pairs.npart_p && pairs.npart_m == H.C.n_spawned,
            "T10: both species' npart grow equally (= n_spawned)", static_cast<double>(pairs.npart_m),
            static_cast<double>(H.C.n_spawned));
    verdict(H.C.n_rng == 0, "T10: the hadronic RNG counter is unchanged (budget + delta draws nothing)",
            static_cast<double>(H.C.n_rng), 0.0);
    verdict(H.C.n_dropped == 0 && H.C.n_capped == 0, "T10: no drops and no capping at max_events_per_step = 8",
            static_cast<double>(H.C.n_dropped + H.C.n_capped), 0.0);
    // (3) closure of this run (T11 on the budget run)
    const auto A = array_identity(ions, K_init, pairs, H);
    check_identity("T11 (budget, 2000 steps)", H, A);
    diag("T11 budget: E_drag " + std::to_string(H.C.E_drag.value()) + ", E_pairs " + std::to_string(H.C.E_pairs.value()) +
         ", E_absorbed " + std::to_string(H.C.E_absorbed.value()) + ", E_round " + std::to_string(H.C.E_round.value()) +
         ", debt_outstanding " + std::to_string(H.C.debt_outstanding.value()) + ", array lhs " + std::to_string(A.lhs) +
         ", rhs " + std::to_string(A.rhs) + ", residual " + std::to_string(A.residual) + ", B " +
         std::to_string(H.C.B_budget.value()) + ", |residual|/B " +
         std::to_string(std::abs(A.residual) / H.C.B_budget.value()));
  }

  /**
   * @brief T11 (iv): the momentum residual needs the physical ion mass (mu_d =
   *        mu_r) so that the ion's beta is beta_p; with mu_d = 1 the ion is
   *        ultra-relativistic and P_drag = E_drag. Before the run: the III.1
   *        prediction sum w D (1/beta_p - 1) treats the leptons as massless
   *        (p_e = E_e); with the delta spectrum at Gamma_p = 6 each lepton has
   *        gamma_e = (<D> + eps)/2 = 4.43 and the rest-mass deficit of the pair
   *        momentum, 2 (gamma_e - sqrt(gamma_e^2 - 1)) = 0.23 per interaction,
   *        exceeds D (1/beta_p - 1) = 0.12; the registered 1% criterion is
   *        therefore expected to fail by a factor of about 3 (labelled before
   *        the run). The kinematic prediction including the rest mass,
   *        sum w [D/beta_p(spawn) - (2 sqrt(gamma_e^2 - 1) - eps)], is reported
   *        as a diagnostic (the drag of a packet is accrued at earlier, higher
   *        beta_p, so it is an approximation at the per-cent level of the
   *        residual).
   */
  void test_T11_momentum(const Tables& T) {
    std::printf("\n=== T11 (iv) momentum: 8 ions with the physical mass (mu_d = mu_r) at Gamma_p = 6, dK/K = 1e-4 "
                "per step, 2000 steps ===\n");
    double D6, e6;
    bh::interpolate_outcome_host(T.host, 6.0, D6, e6);
    {
      const double beta = std::sqrt(1.0 - 1.0 / 36.0);
      const double ge   = 0.5 * (D6 + e6);
      const double pe   = std::sqrt((ge - 1.0) * (ge + 1.0));
      std::printf("(diag) before the run at Gamma_p = 6: D (1/beta_p - 1) = %.6f, lepton rest-mass deficit "
                  "2 (gamma_e - p_e) = %.6f (gamma_e = %.4f), expected ratio measured/III.1 = %.3f: the 1%% "
                  "criterion against the III.1 prediction is expected to fail\n",
                  D6 * (1.0 / beta - 1.0), 2.0 * (ge - pe), ge, (D6 * (1.0 / beta - 1.0) + 2.0 * (ge - pe)) /
                                                                   (D6 * (1.0 / beta - 1.0)));
    }
    const npart_t N = 8;
    Ions          ions(N, MU_R);
    for (npart_t p = 0; p < N; ++p) {
      ions.set(p, K_of_gamma(6.0), 1.0 + 0.5 * (p % 3));
    }
    ions.zero_payloads();
    Pairs                pairs(200000);
    random_number_pool_t pool(50u);
    auto                 prm = params(coeff_for(T, 6.0, 1.0e-4), BUDGET);
    HostLedger           H;
    const auto           K_init = ions.K();
    for (int n = 0; n < 2000; ++n) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
    }
    const auto rows = pairs.rows();
    double     pred_31 = 0.0, pred_kin = 0.0;
    for (const auto& r : rows) {
      if (!(r.birth_m > 0.0)) {
        continue;
      }
      const double Gp   = r.gp_m;
      const double beta = std::sqrt(1.0 - 1.0 / (Gp * Gp));
      double       Dm, eps;
      bh::interpolate_outcome_host(T.host, Gp, Dm, eps);
      const double ge = 0.5 * (Dm + eps);
      const double pe = std::sqrt((ge - 1.0) * (ge + 1.0));
      pred_31        += r.wm * Dm * (1.0 / beta - 1.0);
      // the stored lepton momenta and the stored weight: the exact kinematic residual uses
      // the drag of this packet r_packet = w_pair D / w through w_pair D = w_s Dm
      pred_kin       += r.wm * (Dm / beta - (2.0 * pe - eps));
    }
    const double res = H.C.P_drag[0] - (H.C.P_pairs[0] - H.C.P_absorbed[0]);
    verdict(std::abs(res - pred_31) <= 0.01 * std::abs(pred_31),
            "T11 (iv): P_drag - (P_pairs - P_absorbed) vs the III.1 prediction sum w D (1/beta_p - 1), "
            "|measured - predicted| / predicted",
            std::abs(res - pred_31) / std::abs(pred_31), 0.01, true);
    // the drag momentum of the outstanding debt is in P_drag but not yet in the pairs: subtract it
    double debt_mom = 0.0;
    {
      const auto rr = ions.payload(1);
      const auto w  = ions.w();
      const auto Kf = ions.K();
      for (npart_t p = 0; p < N; ++p) {
        const double Gp   = 1.0 + Kf[p] / MU_R;
        const double beta = std::sqrt(1.0 - 1.0 / (Gp * Gp));
        debt_mom         += w[p] * rr[p] / beta;
      }
    }
    const double res_paid = res - debt_mom;
    // not a verdict: the drag momentum of a packet is accrued over earlier steps at a higher beta_p
    // than the spawn state, so "D/beta_p at spawn" is an approximation at the 1e-2 level of the residual
    std::printf("(diag) T11 (iv): P_drag(paid) - (P_pairs - P_absorbed) vs the kinematic prediction with the lepton rest "
                "mass sum w [D/beta_p(spawn) - (2 sqrt(gamma_e^2 - 1) - eps)]: relative difference %.3e (the paid "
                "drag momentum was accrued at the earlier, higher beta_p of each ion; the exact residual is the ledger's)\n",
                std::abs(res_paid - pred_kin) / std::abs(pred_kin));
    diag("T11 (iv): measured P_drag - (P_pairs - P_absorbed) = " + std::to_string(res) + ", of which the outstanding "
         "debt's drag momentum " + std::to_string(debt_mom) + "; III.1 prediction " + std::to_string(pred_31) +
         ", kinematic prediction " + std::to_string(pred_kin) + "; ratio measured(paid)/III.1 = " +
         std::to_string(res_paid / pred_31) + "; P_drag " + std::to_string(H.C.P_drag[0]) + ", P_pairs " +
         std::to_string(H.C.P_pairs[0]) + ", P_absorbed " + std::to_string(H.C.P_absorbed[0]) + ", E_drag " +
         std::to_string(H.C.E_drag.value()));
    const auto A = array_identity(ions, K_init, pairs, H);
    check_identity("T11 (physical mass run)", H, A);
    // diagnostic at Gamma_p = 1.5 is below the table edge of this bath (Gamma_th = 2.125): the brief's
    // low-energy diagnostic is taken at Gamma_p = 2.5 instead
    {
      Ions ions2(N, MU_R);
      for (npart_t p = 0; p < N; ++p) {
        ions2.set(p, K_of_gamma(2.5), 1.0);
      }
      ions2.zero_payloads();
      Pairs      pairs2(200000);
      HostLedger H2;
      auto       prm2 = params(coeff_for(T, 2.5, 1.0e-4), BUDGET, 10.0, 0.02);
      for (int n = 0; n < 2000; ++n) {
        H2.add(launch(ions2, pairs2, T, prm2, pool, static_cast<double>(n + 1) * DT));
      }
      double D25, e25;
      bh::interpolate_outcome_host(T.host, 2.5, D25, e25);
      const double beta = std::sqrt(1.0 - 1.0 / 6.25);
      const double res2 = H2.C.P_drag[0] - (H2.C.P_pairs[0] - H2.C.P_absorbed[0]);
      diag("T11 (iv) low-energy diagnostic at Gamma_p = 2.5 (Gamma_p = 1.5 is below the threshold 2.125 of this "
           "bath): D (1/beta_p - 1)/D = " + std::to_string(1.0 / beta - 1.0) + ", measured residual " +
           std::to_string(res2) + " for E_drag " + std::to_string(H2.C.E_drag.value()) + " (" +
           std::to_string(H2.C.n_spawned) + " packets), residual / E_drag = " +
           std::to_string(res2 / H2.C.E_drag.value()));
    }
  }

  void test_T10_capped(const Tables& T) {
    std::printf("\n=== T10 capped: dK/K = 2e-2 per step (about 2 packets per step due), max_events_per_step = 1, "
                "30 steps ===\n");
    const npart_t N = 8;
    Ions          ions(N, 1.0);
    for (npart_t p = 0; p < N; ++p) {
      ions.set(p, K_of_gamma(6.0), 1.0);
    }
    ions.zero_payloads();
    Pairs                pairs(200000);
    random_number_pool_t pool(43u);
    auto                 prm        = params(coeff_for(T, 6.0, 2.0e-2), BUDGET);
    prm.max_events_per_step         = 1;
    prm.f_max                       = 0.05; // no sub-cycling at dK/K = 2e-2
    HostLedger   H;
    const auto   K_init = ions.K();
    double       debt_prev = 0.0;
    bool         debt_grows = false;
    // 30 steps: K drops by 45% (Gamma_p 6 -> 3.75), still inside the table (a 100-step run would
    // fall below the threshold after 74 steps)
    for (int n = 0; n < 30; ++n) {
      const auto L = launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT);
      H.add(L);
      if (H.C.debt_outstanding.value() > debt_prev + 10.0) {
        debt_grows = true;
      }
      debt_prev = H.C.debt_outstanding.value();
    }
    diag("T10 capped: n_spawned " + std::to_string(H.C.n_spawned) + ", n_capped " + std::to_string(H.C.n_capped) +
         ", debt_outstanding " + std::to_string(H.C.debt_outstanding.value()) + ", E_drag " +
         std::to_string(H.C.E_drag.value()));
    verdict(H.C.n_capped > 0, "T10 capped: n_capped > 0 with max_events_per_step = 1", static_cast<double>(H.C.n_capped), 1.0);
    verdict(H.C.n_spawned == 30 * N, "T10 capped: exactly one packet per ion per step (30 steps)",
            static_cast<double>(H.C.n_spawned), static_cast<double>(30 * N));
    verdict(debt_grows && H.C.debt_outstanding.value() > 0.0, "T10 capped: the debt is carried over (grows)",
            H.C.debt_outstanding.value(), 0.0);
    const auto A = array_identity(ions, K_init, pairs, H);
    check_identity("T11 (capped budget)", H, A);
  }

  void test_T10_closing(const Tables& T) {
    std::printf("\n=== T10 closing packets: ions just above the edge accumulate debt, then are decelerated "
                "below Gamma_min on the host (emulating E antiparallel to u) ===\n");
    const npart_t N = 6;
    Ions          ions(N, 1.0);
    const double  Gp0 = 2.2;
    for (npart_t p = 0; p < N; ++p) {
      ions.set(p, K_of_gamma(Gp0), 1.0 + 0.5 * p);
    }
    ions.zero_payloads();
    Pairs                pairs(100000);
    random_number_pool_t pool(44u);
    // dK/K = 1e-3 per step at Gamma_p = 2.2 so that a debt below r_th builds up in 20 steps
    auto         prm = params(coeff_for(T, Gp0, 1.0e-3), BUDGET, 1.0e6, 0.5); // r_th never reached
    HostLedger   H;
    const auto   K_init = ions.K();
    for (int n = 0; n < 20; ++n) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
    }
    const auto r_before = ions.payload(1);
    const auto w        = ions.w();
    double     sum_wr   = 0.0;
    for (npart_t p = 0; p < N; ++p) {
      sum_wr += w[p] * r_before[p];
    }
    diag("closing: after 20 steps n_spawned " + std::to_string(H.C.n_spawned) + ", sum w r = " + std::to_string(sum_wr) +
         ", debt_outstanding " + std::to_string(H.C.debt_outstanding.value()));
    verdict(H.C.n_spawned == 0 && sum_wr > 0.0, "closing: debt accumulated without packets before the deceleration",
            sum_wr, 0.0);
    // decelerate every ion below the table edge (fields), keeping the payloads
    const double Gp_low = 0.5 * (1.0 + T.host.gamma_min);
    {
      auto h1 = mirror(ions.arr.ux1);
      for (npart_t p = 0; p < N; ++p) {
        h1(p) = static_cast<real_t>(Ions::u_of_K(K_of_gamma(Gp_low), 1.0));
      }
      Kokkos::deep_copy(ions.arr.ux1, h1);
    }
    const auto K_after_field = ions.K();
    const auto L = launch(ions, pairs, T, prm, pool, 21.0 * DT);
    H.add(L);
    const auto r_after = ions.payload(1);
    const auto k0_after = ions.payload(2);
    bool zero_ok = true;
    for (npart_t p = 0; p < N; ++p) {
      zero_ok = zero_ok && r_after[p] == 0.0 && k0_after[p] == 0.0;
    }
    verdict(L.n_closing == N, "closing: n_closing = number of decelerated ions with debt", static_cast<double>(L.n_closing),
            static_cast<double>(N));
    verdict(L.n_spawned == N, "closing: one closing packet per ion", static_cast<double>(L.n_spawned), static_cast<double>(N));
    verdict(zero_ok, "closing: r = 0 and K0 = 0 after the closing packet", zero_ok ? 0.0 : 1.0, 0.0);
    // w_pair = w r / <D>(Gamma_min), payload Gamma_p = the ion's actual Gamma_p (below the edge)
    double Dmin, emin;
    bh::interpolate_outcome_host(T.host, T.host.gamma_min, Dmin, emin);
    const auto rows = pairs.rows();
    double     max_w_err = 0.0, max_gp_err = 0.0;
    for (const auto& r : rows) {
      const int    p    = r.i1m;
      const double wexp = w[p] * r_before[p] / Dmin;
      max_w_err         = std::max(max_w_err, std::abs(r.wm - wexp) / wexp);
      max_gp_err        = std::max(max_gp_err, std::abs(r.gp_m - Gp_low) / Gp_low);
    }
    verdict(rows.size() == N && max_w_err <= 1.0e-6, "closing: w_pair = w r / <D>(Gamma_min), max relative error",
            max_w_err, 1.0e-6);
    verdict(max_gp_err <= 1.0e-6, "closing: pair payload Gamma_p = the parent's (below-edge) Gamma_p", max_gp_err, 1.0e-6);
    // the identity with the field work: the array rhs uses the energies after the field kick as the new K_init
    HostLedger H2;
    H2.C          = L;
    H2.debt_initial = kh::dd {};
    {
      kh::dd di;
      for (npart_t p = 0; p < N; ++p) {
        di.add_prod(w[p], r_before[p]);
      }
      H2.debt_initial = di;
    }
    const auto A = array_identity(ions, K_after_field, pairs, H2);
    verdict(std::abs(H2.residual()) <= 1.0e-12 * sum_wr, "closing: ledger identity with debt_initial = sum w r before the step",
            std::abs(H2.residual()) / sum_wr, 1.0e-12);
    diag("closing: array residual " + std::to_string(A.residual) + " vs E_round " + std::to_string(L.E_round.value()));
  }

  void test_T10_poisson(const Tables& T) {
    std::printf("\n=== T10 poisson + delta: 100 ions x 100 steps at R dt / pair_weight = 0.2 ===\n");
    const npart_t N = 100;
    const double  Gp = 6.0;
    const double  coeff = coeff_for(T, Gp, 1.0e-4);
    const double  J     = bh::interpolate_rate_host(T.host, Gp);
    const double  lambda = 0.2;
    auto          prm   = params(coeff, POISSON);
    prm.pair_weight     = coeff * J * DT / lambda;
    auto run = [&](std::uint64_t seed, Pairs& pairs, HostLedger& H, std::vector<double>& K_init, Ions& ions) {
      for (npart_t p = 0; p < N; ++p) {
        ions.set(p, K_of_gamma(Gp), 1.0);
      }
      ions.zero_payloads();
      random_number_pool_t pool(seed);
      K_init = ions.K();
      for (int n = 0; n < 100; ++n) {
        H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
      }
    };
    Ions                ions(N, 1.0);
    Pairs               pairs(400000);
    HostLedger          H;
    std::vector<double> K_init;
    run(1001u, pairs, H, K_init, ions);
    const double mean = lambda * N * 100.0;
    const double sig  = std::sqrt(mean);
    diag("poisson: n_spawned " + std::to_string(H.C.n_spawned) + ", expected " + std::to_string(mean) + " +- " +
         std::to_string(sig) + ", n_rng " + std::to_string(H.C.n_rng) + ", n_capped_events " +
         std::to_string(H.C.n_capped_events) + ", pair_weight " + std::to_string(prm.pair_weight));
    verdict(std::abs(static_cast<double>(H.C.n_spawned) - mean) <= 4.0 * sig,
            "poisson: spawn count within 4 sigma of the Poisson mean, |n - mean| / sigma",
            std::abs(static_cast<double>(H.C.n_spawned) - mean) / sig, 4.0);
    // per (ion, step) counts: variance within 10% of the mean
    {
      std::map<std::pair<int, int>, int> counts;
      const auto rows = pairs.rows();
      for (const auto& r : rows) {
        const int step = static_cast<int>(std::lround(r.birth_m / DT));
        counts[{ r.i1m, step }] += 1;
      }
      const double n_cells = static_cast<double>(N) * 100.0;
      double       s1 = 0.0, s2 = 0.0;
      for (const auto& kv : counts) {
        s1 += kv.second;
        s2 += static_cast<double>(kv.second) * kv.second;
      }
      const double m   = s1 / n_cells;
      const double var = s2 / n_cells - m * m;
      verdict(std::abs(var - m) <= 0.10 * m, "poisson: variance of the per-ion-step counts within 10% of the mean, |var - mean| / mean",
              std::abs(var - m) / m, 0.10);
      diag("poisson: per-ion-step mean " + std::to_string(m) + ", variance " + std::to_string(var));
    }
    verdict(H.C.n_rng > 0, "poisson: the hadronic RNG was used", static_cast<double>(H.C.n_rng), 1.0);
    // pairs per unit ion energy: sum w_pair / E_drag vs 1/<D> within 4 sigma
    {
      double Dm, eps;
      bh::interpolate_outcome_host(T.host, Gp, Dm, eps);
      const auto A        = array_identity(ions, K_init, pairs, H);
      const double ratio  = A.sum_wpair / H.C.E_drag.value();
      const double expect = 1.0 / Dm;
      const double sigma  = std::sqrt(static_cast<double>(H.C.n_spawned)) * prm.pair_weight / H.C.E_drag.value();
      verdict(std::abs(ratio - expect) <= 4.0 * sigma,
              "pairs per unit ion energy (poisson): sum w_pair / E_drag vs 1/<D>, |difference| / sigma",
              std::abs(ratio - expect) / sigma, 4.0);
      check_identity("T11 (poisson)", H, A);
      diag("T11 poisson: E_drag " + std::to_string(H.C.E_drag.value()) + ", E_pairs - E_absorbed " +
           std::to_string(H.C.E_pairs.value() - H.C.E_absorbed.value()) + ", E_lost_nopairs (signed imbalance) " +
           std::to_string(H.C.E_lost_nopairs.value()) + ", E_round " + std::to_string(H.C.E_round.value()));
    }
    // determinism: a different seed differs; the same seed is reported (bit identity is registered on the
    // Serial backend at the engine level; the OpenMP pool assigns generator states by scheduling)
    {
      Ions                ions2(N, 1.0);
      Pairs               pairs2(400000);
      HostLedger          H2;
      std::vector<double> K2;
      run(2002u, pairs2, H2, K2, ions2);
      verdict(H2.C.n_spawned != H.C.n_spawned || pairs2.rows().front().i1m != pairs.rows().front().i1m ||
                pairs2.rows().front().birth_m != pairs.rows().front().birth_m,
              "poisson: a different seed gives a different realisation", 1.0, 1.0);
      Ions                ions3(N, 1.0);
      Pairs               pairs3(400000);
      HostLedger          H3;
      std::vector<double> K3;
      run(1001u, pairs3, H3, K3, ions3);
      bool same = H3.C.n_spawned == H.C.n_spawned;
      if (same) {
        auto a = pairs.rows(), b = pairs3.rows();
        auto key = [](const PairRow& r) {
          return std::make_pair(r.birth_m, r.i1m);
        };
        std::sort(a.begin(), a.end(), [&](const PairRow& x, const PairRow& y) { return key(x) < key(y); });
        std::sort(b.begin(), b.end(), [&](const PairRow& x, const PairRow& y) { return key(x) < key(y); });
        for (std::size_t k = 0; k < a.size(); ++k) {
          same = same && key(a[k]) == key(b[k]);
        }
      }
      std::printf("(diag) poisson: same seed on %s: %s (n_spawned %llu vs %llu); bit identity is registered on the "
                  "Serial backend (HADRONIC::t10_poisson_determinism)\n",
                  Kokkos::DefaultExecutionSpace::name(), same ? "identical spawn set" : "different spawn set",
                  H.C.n_spawned, H3.C.n_spawned);
    }
  }

  void test_T10_none(const Tables& T) {
    std::printf("\n=== T10 none: loss only ===\n");
    const npart_t N = 8;
    Ions          ions(N, 1.0);
    for (npart_t p = 0; p < N; ++p) {
      ions.set(p, K_of_gamma(6.0), 1.0);
    }
    ions.zero_payloads();
    Pairs                pairs(1000);
    random_number_pool_t pool(45u);
    auto                 prm = params(coeff_for(T, 6.0, 1.0e-4), NONE);
    HostLedger           H;
    const auto           K_init = ions.K();
    for (int n = 0; n < 500; ++n) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
    }
    verdict(H.C.n_spawned == 0 && H.C.n_events == 0 && pairs.npart_m == 0, "none: no spawns",
            static_cast<double>(H.C.n_spawned), 0.0);
    verdict(H.C.E_lost_nopairs.value() == H.C.E_drag.value() && H.C.E_drag.value() > 0.0,
            "none: E_lost_nopairs = E_drag (exact)", H.C.E_lost_nopairs.value() - H.C.E_drag.value(), 0.0);
    verdict(H.C.n_rng == 0, "none: no random numbers drawn", static_cast<double>(H.C.n_rng), 0.0);
    const auto A = array_identity(ions, K_init, pairs, H);
    check_identity("T11 (none)", H, A, false);
  }

  /* ------------------------------------------------------------------------ */
  /* pairs per unit ion energy (budget)                                         */
  /* ------------------------------------------------------------------------ */
  void test_pairs_per_energy(const Tables& T) {
    std::printf("\n=== pairs per unit ion energy (budget): 16 ions at Gamma_p = 6, dK/K = 1e-6 per step, 1000 steps "
                "(Gamma_p changes by 0.1%%), r_th_factor = 2e-4 so that the outstanding debt is < 2e-4 of E_drag ===\n");
    auto run = [&](double frac, int nsteps, double r_th_factor, bool registered) {
      const npart_t N = 16;
      Ions          ions(N, 1.0);
      for (npart_t p = 0; p < N; ++p) {
        ions.set(p, K_of_gamma(6.0), 1.0 + 0.25 * (p % 4));
      }
      ions.zero_payloads();
      Pairs                pairs(400000);
      random_number_pool_t pool(46u);
      auto                 prm = params(coeff_for(T, 6.0, frac), BUDGET, r_th_factor);
      HostLedger           H;
      const auto           K_init = ions.K();
      for (int n = 0; n < nsteps; ++n) {
        H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
      }
      const auto K_fin = ions.K();
      double     Gp_mid = 0.0;
      for (npart_t p = 0; p < N; ++p) {
        Gp_mid += 1.0 + 0.5 * (K_init[p] + K_fin[p]) / MU_R;
      }
      Gp_mid /= N;
      double Dm, eps;
      bh::interpolate_outcome_host(T.host, Gp_mid, Dm, eps);
      const auto   A      = array_identity(ions, K_init, pairs, H);
      const double paid   = H.C.E_drag.value() - H.C.debt_outstanding.value();
      const double expect = 1.0 / Dm;
      const double r_lit  = A.sum_wpair / H.C.E_drag.value();
      const double r_paid = A.sum_wpair / paid;
      const std::string cfg = "dK/K = " + std::to_string(frac) + ", " + std::to_string(nsteps) + " steps, r_th_factor = " +
                              std::to_string(r_th_factor);
      diag("pairs per energy (" + cfg + "): K changed by " + std::to_string(100.0 * (K_init[0] - K_fin[0]) / K_init[0]) +
           "%, Gamma_p mid " + std::to_string(Gp_mid) + ", <D> " + std::to_string(Dm) + ", sum w_pair " +
           std::to_string(A.sum_wpair) + ", E_drag " + std::to_string(H.C.E_drag.value()) + ", debt_outstanding " +
           std::to_string(H.C.debt_outstanding.value()) + " (" +
           std::to_string(H.C.debt_outstanding.value() / H.C.E_drag.value()) + " of E_drag), packets " +
           std::to_string(H.C.n_spawned) + ", sum w_pair / E_drag = " + std::to_string(r_lit) +
           ", sum w_pair / (E_drag - debt) = " + std::to_string(r_paid) + ", 1/<D> = " + std::to_string(expect));
      if (registered) {
        verdict(std::abs(r_lit - expect) <= 1.0e-3 * expect,
                "pairs per unit ion energy (budget): sum w_pair / E_drag vs 1/<D>(Gamma_p mid), relative (" + cfg + ")",
                std::abs(r_lit - expect) / expect, 1.0e-3);
        verdict(std::abs(r_paid - expect) <= 1.0e-3 * expect,
                "(impl) pairs per unit ion energy (budget): sum w_pair / (E_drag - debt_outstanding) vs 1/<D>, relative",
                std::abs(r_paid - expect) / expect, 1.0e-3);
        check_identity("T11 (pairs per energy run)", H, A);
      } else {
        std::printf("(diag) pairs per energy (%s): |sum w_pair / E_drag - 1/<D>| / (1/<D>) = %.3e, "
                    "|sum w_pair / (E_drag - debt) - 1/<D>| / (1/<D>) = %.3e (the first carries the outstanding debt, "
                    "the second the drift of <D> between the spawn and the mid-run Gamma_p)\n",
                    cfg.c_str(), std::abs(r_lit - expect) / expect, std::abs(r_paid - expect) / expect);
      }
    };
    run(1.0e-6, 1000, 2.0e-4, true);
    run(1.0e-5, 1000, 10.0, false);
  }

  /* ------------------------------------------------------------------------ */
  /* T12: capacity                                                               */
  /* ------------------------------------------------------------------------ */
  void test_T12(const Tables& T) {
    std::printf("\n=== T12 capacity: maxnpart of the pair species exhausted at step ~1000 of 2000 (warn policy: continue) "
                "===\n");
    const npart_t N = 8;
    Ions          ions(N, 1.0);
    for (npart_t p = 0; p < N; ++p) {
      ions.set(p, K_of_gamma(6.0), 1.0);
    }
    ions.zero_payloads();
    // at dK/K = 1e-4 per step and r_th = 10 <D>: one packet per ion every ~91 steps -> ~88 pairs
    // per 1000 steps; capacity 88 exhausts near step 1000
    Pairs                pairs(88);
    random_number_pool_t pool(47u);
    auto                 prm = params(coeff_for(T, 6.0, 1.0e-4), BUDGET);
    HostLedger           H;
    const auto           K_init = ions.K();
    int                  step_exhausted = -1;
    npart_t              n_inj_at = 0, cap_at = 0;
    bool                 per_launch_identity = true;
    double               max_launch_res = 0.0;
    for (int n = 0; n < 2000; ++n) {
      const auto cap_before = std::min(pairs.maxn - pairs.npart_m, pairs.maxn - pairs.npart_p);
      const auto before     = pairs.npart_m;
      const auto L          = launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT);
      // per-launch identity: E_drag_step = (E_pairs - E_absorbed + E_round)_step + (debt_after - debt_before)
      {
        kh::dd r;
        r.add(L.E_drag);
        r.sub(L.E_pairs);
        r.add(L.E_absorbed);
        r.sub(L.E_round);
        r.sub(L.debt_outstanding);
        r.add(H.C.debt_outstanding);
        max_launch_res = std::max(max_launch_res, std::abs(r.value()));
        if (std::abs(r.value()) > 1.0e-12 * std::max(1.0, H.C.E_drag.value())) {
          per_launch_identity = false;
        }
      }
      H.add(L);
      if (pairs.npart_m == pairs.maxn && step_exhausted < 0) {
        step_exhausted = n;
        n_inj_at       = pairs.npart_m - before;
        cap_at         = cap_before;
        diag("T12: capacity exhausted at step " + std::to_string(n) + ": capacity before the launch " +
             std::to_string(cap_before) + ", inserted " + std::to_string(n_inj_at) + ", n_dropped " +
             std::to_string(L.n_dropped) + ", debt_outstanding " + std::to_string(L.debt_outstanding.value()));
      }
    }
    verdict(step_exhausted >= 700 && step_exhausted <= 1300, "T12: exhaustion step near 1000", step_exhausted, 1000.0);
    verdict(step_exhausted >= 0 && n_inj_at == cap_at, "T12: n_injected = capacity exactly at the exhaustion step",
            static_cast<double>(n_inj_at), static_cast<double>(cap_at));
    verdict(pairs.npart_m == pairs.maxn && pairs.npart_p == pairs.maxn, "T12: both species full and equal (contiguous slots)",
            static_cast<double>(pairs.npart_m), static_cast<double>(pairs.maxn));
    {
      const auto rows = pairs.rows();
      bool filled = rows.size() == pairs.maxn;
      for (const auto& r : rows) {
        filled = filled && r.birth_m > 0.0 && r.birth_p > 0.0 && r.tagm == ParticleTag::alive;
      }
      verdict(filled, "T12: every slot [0, maxnpart) holds a kernel-born lepton (contiguous)", filled ? 1.0 : 0.0, 1.0);
    }
    verdict(H.C.n_dropped > 0, "T12: n_dropped > 0", static_cast<double>(H.C.n_dropped), 1.0);
    verdict(per_launch_identity, "T12: debt retained: the per-launch identity holds at every step including the drops, "
                                 "max |residual|",
            max_launch_res, 1.0e-12 * H.C.E_drag.value());
    verdict(H.C.debt_outstanding.value() > 0.3 * H.C.E_drag.value(),
            "T12: the debt of the dropped packets stays outstanding (debt_outstanding / E_drag)",
            H.C.debt_outstanding.value() / H.C.E_drag.value(), 0.3);
    const auto A = array_identity(ions, K_init, pairs, H);
    check_identity("T11 (T12 run with drops, warn policy)", H, A);
    diag("T12: n_spawned " + std::to_string(H.C.n_spawned) + ", n_events " + std::to_string(H.C.n_events) + ", n_dropped " +
         std::to_string(H.C.n_dropped) + ", debt_outstanding " + std::to_string(H.C.debt_outstanding.value()) +
         ", E_drag " + std::to_string(H.C.E_drag.value()));
  }

  /* ------------------------------------------------------------------------ */
  /* T10b: stale payloads of the pair species                                    */
  /* ------------------------------------------------------------------------ */
  void test_T10b(const Tables& T) {
    std::printf("\n=== T10b: NaN-filled pair payload arrays (5 slots), 300 steps ===\n");
    const npart_t N = 8;
    Ions          ions(N, 1.0);
    for (npart_t p = 0; p < N; ++p) {
      ions.set(p, K_of_gamma(6.0), 1.0);
    }
    ions.zero_payloads();
    Pairs pairs(10000, 5);
    pairs.fill_payloads_nan();
    random_number_pool_t pool(48u);
    auto                 prm = params(coeff_for(T, 6.0, 1.0e-3), BUDGET);
    HostLedger           H;
    for (int n = 0; n < 300; ++n) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
    }
    auto pm = mirror(pairs.minus.pld_r);
    auto pp = mirror(pairs.plus.pld_r);
    unsigned long long nan_live = 0, nan_tail = 0, extra_nonzero = 0;
    for (npart_t k = 0; k < pairs.maxn; ++k) {
      for (unsigned short c = 0; c < 5; ++c) {
        const bool live = k < pairs.npart_m;
        const bool isn  = std::isnan(static_cast<double>(pm(k, c))) || std::isnan(static_cast<double>(pp(k, c)));
        if (live && isn) {
          ++nan_live;
        }
        if (!live && isn) {
          ++nan_tail;
        }
        if (live && c >= 3 && (pm(k, c) != ZERO || pp(k, c) != ZERO)) {
          ++extra_nonzero;
        }
      }
    }
    verdict(pairs.npart_m > 0 && nan_live == 0, "T10b: no live lepton carries a NaN payload (" +
                                                   std::to_string(pairs.npart_m) + " pairs, 5 slots)",
            static_cast<double>(nan_live), 0.0);
    verdict(extra_nonzero == 0, "T10b: payload slots beyond the three specified are zero on every live lepton",
            static_cast<double>(extra_nonzero), 0.0);
    verdict(nan_tail == 5ull * (pairs.maxn - pairs.npart_m), "T10b: the unused tails still hold NaN (untouched; one "
                                                              "count per slot pair)",
            static_cast<double>(nan_tail), static_cast<double>(5ull * (pairs.maxn - pairs.npart_m)));
  }

  /* ------------------------------------------------------------------------ */
  /* T-exp: exported debt of dead ions                                           */
  /* ------------------------------------------------------------------------ */
  void test_Texp(const Tables& T) {
    std::printf("\n=== T-exp: ions tagged dead on the host (absorbing wall) export their debt ===\n");
    const npart_t N = 10;
    Ions          ions(N, 1.0);
    for (npart_t p = 0; p < N; ++p) {
      ions.set(p, K_of_gamma(6.0), 1.0 + 0.3 * p);
    }
    ions.zero_payloads();
    Pairs                pairs(100000);
    random_number_pool_t pool(49u);
    auto                 prm = params(coeff_for(T, 6.0, 1.0e-4), BUDGET);
    HostLedger           H;
    const auto           K_init = ions.K();
    for (int n = 0; n < 50; ++n) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
    }
    // host-side sums over the ions that are about to die (before the launch, as the engine's pre-sort state)
    const auto r_b = ions.payload(1), d_b = ions.payload(0);
    const auto w   = ions.w();
    kh::dd     exp_expect, disc_expect;
    for (npart_t p = 0; p < N; p += 3) {
      ions.set_tag(p, ParticleTag::dead);
      exp_expect.add_prod(w[p], r_b[p]);
      disc_expect.add_prod(w[p], d_b[p]);
    }
    const auto L = launch(ions, pairs, T, prm, pool, 51.0 * DT);
    H.add(L);
    verdict(exp_expect.value() > 0.0 && std::abs(L.E_exported.value() - exp_expect.value()) <= 1.0e-12 * exp_expect.value(),
            "T-exp: E_exported = sum w r of the dead ions (host sum before the launch), relative",
            std::abs(L.E_exported.value() - exp_expect.value()) / exp_expect.value(), 1.0e-12);
    verdict(std::abs(L.pending_discarded.value() - disc_expect.value()) <= 1.0e-12 * std::max(disc_expect.value(), 1e-300),
            "T-exp: pending_discarded = sum w delta of the dead ions, relative",
            disc_expect.value() > 0.0 ? std::abs(L.pending_discarded.value() - disc_expect.value()) / disc_expect.value()
                                      : std::abs(L.pending_discarded.value()),
            1.0e-12);
    {
      const auto r_a = ions.payload(1), d_a = ions.payload(0), k_a = ions.payload(2);
      bool zero = true;
      for (npart_t p = 0; p < N; p += 3) {
        zero = zero && r_a[p] == 0.0 && d_a[p] == 0.0 && k_a[p] == 0.0;
      }
      verdict(zero, "T-exp: r, delta, K0 of the dead ions are zero after the launch", zero ? 0.0 : 1.0, 0.0);
    }
    for (int n = 51; n < 100; ++n) {
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * DT));
    }
    // the array identity with dead ions: their K_init - K_final is their total applied loss, part of
    // which left as exported debt
    const auto A = array_identity(ions, K_init, pairs, H);
    check_identity("T11 (T-exp, with the exported term)", H, A);
    diag("T-exp: E_exported " + std::to_string(H.C.E_exported.value()) + ", pending_discarded " +
         std::to_string(H.C.pending_discarded.value()) + ", debt_outstanding " +
         std::to_string(H.C.debt_outstanding.value()) + ", n_spawned " + std::to_string(H.C.n_spawned));
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::printf("bh_pairs: real_t = %s, Kokkos default execution space = %s\n",
                sizeof(real_t) == 4 ? "float" : "double", Kokkos::DefaultExecutionSpace::name());
    const auto T = make_tables();
    double     D6, e6;
    bh::interpolate_outcome_host(T.host, 6.0, D6, e6);
    std::printf("bh_pairs: table mono eps0 = %.3g: %zu nodes, gamma_min %.12g, gamma_max %.12g; at Gamma_p = 6: "
                "I %.10e, J %.10e, <D> %.10g, eps_mean %.10g, r_th(10 <D>) = %.6g, f_r K = %.6g\n",
                EPS0, T.host.size(), T.host.gamma_min, T.host.gamma_max, bh::interpolate_loss_host(T.host, 6.0),
                bh::interpolate_rate_host(T.host, 6.0), D6, e6, 10.0 * D6, 0.02 * K_of_gamma(6.0));
    std::vector<std::string> groups;
    for (int i = 1; i < argc; ++i) {
      if (argv[i][0] != '-') {
        groups.emplace_back(argv[i]);
      }
    }
    auto want = [&groups](const char* g) {
      return groups.empty() || std::find(groups.begin(), groups.end(), std::string(g)) != groups.end();
    };
    if (want("t10")) {
      test_T10_budget(T);
      test_T10_capped(T);
      test_T10_closing(T);
    }
    if (want("poisson")) {
      test_T10_poisson(T);
    }
    if (want("none")) {
      test_T10_none(T);
    }
    if (want("energy")) {
      test_pairs_per_energy(T);
    }
    if (want("momentum")) {
      test_T11_momentum(T);
    }
    if (want("t12")) {
      test_T12(T);
    }
    if (want("t10b")) {
      test_T10b(T);
    }
    if (want("texp")) {
      test_Texp(T);
    }
    std::printf("bh_pairs summary: %d passed, %d failed, %d expected failures (T11 (iv) against the III.1 prediction)\n",
                n_pass, n_fail, n_expected_fail);
    if (n_fail > 0 || n_pass == 0) {
      rc = 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_pairs: exception: %s\n", e.what());
    rc = 2;
  }
  Kokkos::finalize();
  return rc;
}
