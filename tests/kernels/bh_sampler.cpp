/**
 * @file tests/kernels/bh_sampler.cpp
 * @brief The rest-frame sampler in the Bethe-Heitler kernel (P2 step 4, stage
 *        4): the device tables and the device chains against the host chain,
 *        A11 (mu_d independence), the ledger identity with sampled energies,
 *        A12 (the sampler's mean cost at every Gamma_p node vs I/J, pair number
 *        and energy per unit paid energy vs the host chain's quadrature, the
 *        spectrum of created pairs vs the host chain's histogram fixture).
 *
 * Conventions (declared before any run):
 *   - bath mono eps0 = 0.5 (Gamma_th = 2.125), mu_r = 1836.15, ions with
 *     mu_d = 1 unless stated, ions along +x at i1 = p, dx1 = 0.5
 *   - the device sampler tables are those of the generator's startup path
 *     (`build_sampler_tables`: 16 eps' nodes per decade, 256 x 256 cells,
 *     single-precision CDFs); the host chain is `rf::HostSampler` on the same
 *     host tables in double
 *   - "exact" means to double round-off: 1e-12 relative to E_drag
 *   - A12 cost runs: ions at the 466 rate-table nodes (2 per node), a preset
 *     debt of 1001 <D>(Gamma_i), r_th_factor = 1 (r_th = <D>), f_r = 0.02,
 *     max_events_per_step = 1000, eta_u = 1e9 (no application: the energies
 *     stay at the nodes), 10 launches: 2e4 pairs per node; cost = N / sum(1/D_j)
 *     over the node's pairs, with D_j = gamma- + gamma+ - eps from the stored
 *     values; its standard error from the sample variance of 1/D
 *   - the spectrum run: ions at Gamma_p = 3 (K = 2 mu_r, to the real_t rounding
 *     of u), 1e5 pairs, weights 1/D (r_packet constant), compared with
 *     `hist_mono05_G3.txt` (weighted chain, 1e6 accepted) by the two-sample
 *     Kolmogorov-Smirnov statistic on the binned CDFs with the 99% critical
 *     value 1.628 sqrt((n1 + n2)/(n1 n2)), n the Kish effective sample sizes
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
#include "archetypes/hadronic/bh_sampler_device.h"
#include "archetypes/hadronic/bh_sampler_host.h"
#include "archetypes/hadronic/bh_sampler_io.h"
#include "archetypes/hadronic/bh_tables_device.h"
#include "framework/containers/particles.h"
#include "kernels/hadronic/bethe_heitler.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;
namespace bh = arch::bh;
namespace rf = arch::bh::rf;
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

  std::string fmt(const char* f, double v) {
    char b[64];
    std::snprintf(b, sizeof(b), f, v);
    return b;
  }

  double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  constexpr double MU_R = 1836.15;
  constexpr double EPS0 = 0.5;

  struct Tables {
    std::shared_ptr<const bh::PhotonBath>          bath;
    std::unique_ptr<bh::BetheHeitlerRates>         rates;
    bh::HadronicTablesHost                         host;
    bh::HadronicTables                             dev;
    std::shared_ptr<const rf::RestFrameTables>     rft;
    std::shared_ptr<const rf::SamplerCumulatives>  cum;
    std::shared_ptr<const rf::BathEpsTables>       eps_tables;
    bh::SamplerTables                              sampler;
    std::shared_ptr<rf::HostSampler>               host_sampler;
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
    T.dev        = bh::mirror_tables_to_device(T.host);
    const auto t0 = std::chrono::steady_clock::now();
    T.rft        = std::make_shared<const rf::RestFrameTables>(rf::RestFrameTables::build(16, 256, 256));
    T.cum        = std::make_shared<const rf::SamplerCumulatives>();
    T.eps_tables = std::make_shared<const rf::BathEpsTables>(rf::BathEpsTables::build(*T.bath, T.host, *T.cum));
    T.sampler    = bh::mirror_sampler_to_device(*T.rft, *T.cum, *T.eps_tables);
    T.host_sampler = std::make_shared<rf::HostSampler>(T.bath, T.host, T.rft, T.cum, T.eps_tables);
    diag("sampler tables: " + std::to_string(T.sampler.n_k) + " eps' nodes, " + std::to_string(T.sampler.n_s) + " x " +
         std::to_string(T.sampler.n_tau) + " cells, " + std::to_string(T.sampler.n_cum) + " cumulative nodes, " +
         fmt("%.2f", static_cast<double>(T.sampler.bytes) * 1.0e-6) + " MB on the device (budget 64 MB), built in " +
         fmt("%.1f", seconds_since(t0)) + " s");
    return T;
  }

  template <typename ViewT>
  auto mirror(const ViewT& v) {
    auto h = Kokkos::create_mirror_view(v);
    Kokkos::deep_copy(h, v);
    return h;
  }

  /* ------------------------------------------------------------------------ */
  /* ions and pairs (as in bh_pairs.cpp)                                        */
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
      auto hi = Kokkos::create_mirror_view(arr.i1);
      auto hd = Kokkos::create_mirror_view(arr.dx1);
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

    void set_all(const std::vector<double>& K, double w = 1.0) {
      auto h1 = mirror(arr.ux1), h2 = mirror(arr.ux2), h3 = mirror(arr.ux3);
      auto hw = mirror(arr.weight);
      for (npart_t p = 0; p < n; ++p) {
        h1(p) = static_cast<real_t>(u_of_K(K[p], mu_d));
        h2(p) = ZERO;
        h3(p) = ZERO;
        hw(p) = static_cast<real_t>(w);
      }
      Kokkos::deep_copy(arr.ux1, h1);
      Kokkos::deep_copy(arr.ux2, h2);
      Kokkos::deep_copy(arr.ux3, h3);
      Kokkos::deep_copy(arr.weight, hw);
    }

    void set_payloads(const std::vector<double>& debt, const std::vector<double>& k0) {
      auto hp = mirror(arr.pld_r);
      for (npart_t p = 0; p < n; ++p) {
        hp(p, 0) = ZERO;
        hp(p, 1) = static_cast<real_t>(debt[p]);
        hp(p, 2) = static_cast<real_t>(k0[p]);
      }
      Kokkos::deep_copy(arr.pld_r, hp);
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

  struct PairRow {
    int    i1m;
    double wm;
    double u1m, u2m, u3m, u1p, u2p, u3p;
    double gp_m, eps_m;
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

    void reset() {
      npart_m = npart_p = 0;
      Kokkos::deep_copy(minus.tag, ParticleTag::dead);
      Kokkos::deep_copy(plus.tag, ParticleTag::dead);
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
        PairRow r;
        r.i1m   = i1m(k);
        r.wm    = wm(k);
        r.u1m   = u1m(k);
        r.u2m   = u2m(k);
        r.u3m   = u3m(k);
        r.u1p   = u1p(k);
        r.u2p   = u2p(k);
        r.u3p   = u3p(k);
        r.gp_m  = pm(k, 1);
        r.eps_m = pm(k, 2);
        out.push_back(r);
      }
      return out;
    }
  };

  struct HostLedger {
    kh::LossLedger C {};
    kh::dd         debt_initial {};

    void add(const kh::LossLedger& L) {
      const auto po  = L.pending_outstanding;
      const auto dbo = L.debt_outstanding;
      const auto sw  = L.sum_wK;
      const auto na  = L.n_alive;
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
      r.sub(C.E_stochastic);
      r.sub(C.debt_outstanding);
      r.sub(C.E_exported);
      r.sub(C.E_lost_nopairs);
      return r.value();
    }
  };

  kh::HadronicParams params(double coeff, int source, int spectrum, double dt) {
    kh::HadronicParams prm;
    prm.tau0_over_skindepth0 = coeff;
    prm.mu_r                 = MU_R;
    prm.eta_u                = 1.0e-4;
    prm.pending_floor_ulp    = 100.0;
    prm.f_max                = 0.01;
    prm.max_substeps         = 64;
    prm.dt                   = dt;
    prm.source               = source;
    prm.spectrum             = spectrum;
    prm.r_th_factor          = 10.0;
    prm.f_r                  = 0.02;
    prm.t_latency            = 0.0;
    prm.pair_weight          = 1.0;
    prm.max_events_per_step  = 8;
    return prm;
  }

  constexpr int BUDGET    = static_cast<int>(kh::PairSource::Budget);
  constexpr int POISSON   = static_cast<int>(kh::PairSource::Poisson);
  constexpr int RESTFRAME = static_cast<int>(kh::PairSpectrum::RestFrame);
  constexpr int DELTA     = static_cast<int>(kh::PairSpectrum::Delta);

  kh::LossLedger launch(Ions& ions, Pairs& pairs, const Tables& T, kh::HadronicParams prm, random_number_pool_t& pool,
                        double time_after) {
    prm.time  = time_after;
    auto targ = pairs.targets();
    kh::LossLedger L;
    Kokkos::parallel_reduce("bh_sampler", CreateParticleRangePolicy<Dim::_1D>({ 0u }, { ions.n }),
                            kh::BetheHeitler_kernel<Minkowski<Dim::_1D>> { ions.arr, ions.mu_d, T.dev, prm, targ, pool,
                                                                          T.sampler },
                            Kokkos::Sum<kh::LossLedger>(L));
    Kokkos::fence();
    const auto n_inj = pairs.finish(targ);
    if (static_cast<unsigned long long>(n_inj) != L.n_spawned) {
      throw std::runtime_error("n_injected != n_spawned");
    }
    return L;
  }

  struct ArrayIdentity {
    double lhs, rhs, residual, n_pairs;
  };

  ArrayIdentity array_identity(const Ions& ions, const std::vector<double>& K_init, const Pairs& pairs,
                               const HostLedger& H) {
    kh::dd     lhs, rhs;
    const auto rows = pairs.rows();
    double     np   = 0.0;
    for (const auto& r : rows) {
      const double E_s = kh::lepton_gamma(r.u1m, r.u2m, r.u3m) + kh::lepton_gamma(r.u1p, r.u2p, r.u3p);
      lhs.add_prod(r.wm, E_s);
      lhs.add_prod(-r.wm, r.eps_m);
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
    rhs.sub(H.C.E_stochastic);
    rhs.add(H.debt_initial);
    kh::dd res = rhs;
    res.sub(lhs);
    return { lhs.value(), rhs.value(), res.value(), np };
  }

  void check_identity(const std::string& tag, const HostLedger& H, const ArrayIdentity& A) {
    const double E = std::abs(H.C.E_drag.value());
    verdict(std::abs(H.residual()) <= 1.0e-12 * E, tag + ": ledger identity residual / E_drag",
            E > 0.0 ? std::abs(H.residual()) / E : std::abs(H.residual()), 1.0e-12);
    const double B = H.C.B_budget.value();
    verdict(std::abs(A.residual) <= 3.0 * B, tag + ": array identity |residual| / B (B = " + fmt("%.6g", B) + ")",
            B > 0.0 ? std::abs(A.residual) / B : std::abs(A.residual), 3.0);
    const double er   = H.C.E_round.value();
    const double diff = std::abs(A.residual - er);
    const bool   ok   = (er == 0.0 && A.residual == 0.0) || diff <= 1.0e-9 * std::abs(er);
    verdict(ok, tag + ": E_round = array residual, |E_round - residual| / |E_round| (E_round = " + fmt("%.6g", er) + ")",
            er != 0.0 ? diff / std::abs(er) : diff, 1.0e-9);
  }

  /* ------------------------------------------------------------------------ */
  /* the device chains drawn directly                                           */
  /* ------------------------------------------------------------------------ */
  struct DrawStats {
    double sD { 0.0 }, sD2 { 0.0 }, sinv { 0.0 }, sinv2 { 0.0 }, minD { 1.0e300 }, maxD { 0.0 };
    unsigned long long n { 0 }, n_w { 0 }, attempts { 0 }, oob { 0 }, nrng { 0 }, fails { 0 };

    KOKKOS_INLINE_FUNCTION
    DrawStats() = default;

    KOKKOS_INLINE_FUNCTION
    auto operator+=(const DrawStats& o) -> DrawStats& {
      sD += o.sD;
      sD2 += o.sD2;
      sinv += o.sinv;
      sinv2 += o.sinv2;
      minD = minD < o.minD ? minD : o.minD;
      maxD = maxD > o.maxD ? maxD : o.maxD;
      n += o.n;
      n_w += o.n_w;
      attempts += o.attempts;
      oob += o.oob;
      nrng += o.nrng;
      fails += o.fails;
      return *this;
    }
  };

} // namespace

namespace Kokkos {
  template <>
  struct reduction_identity<DrawStats> {
    KOKKOS_FORCEINLINE_FUNCTION static DrawStats sum() {
      return DrawStats {};
    }
  };
} // namespace Kokkos

namespace {

  struct DrawKernel {
    bh::SamplerTables    S;
    random_number_pool_t pool;
    double               gamma;
    int                  per_thread;
    bool                 weighted;
    array_t<double*>     first_D; // the first `first_D.extent(0)` D values of thread 0 (bitwise checks)

    Inline void operator()(const int t, DrawStats& st) const {
      auto gen = pool.get_state();
      const double gb = math::sqrt((gamma - 1.0) * (gamma + 1.0));
      for (int i = 0; i < per_thread; ++i) {
        bh::SampledPair ev;
        const bool ok = S.draw(gamma, weighted, gen, st.nrng, 100000, ev);
        st.attempts += ev.attempts;
        if (!ok) {
          st.fails += 1;
          continue;
        }
        const double Dmin = gamma * ev.k - ev.eps - gb * math::sqrt((ev.k - 2.0) * (ev.k + 2.0));
        if (!(ev.D >= Dmin * (1.0 - 1.0e-9) - 1.0e-12) || !(ev.D <= ev.Dmax * (1.0 + 1.0e-9) + 1.0e-12)) {
          st.oob += 1;
        }
        st.sD += ev.D;
        st.sD2 += ev.D * ev.D;
        st.sinv += 1.0 / ev.D;
        st.sinv2 += 1.0 / (ev.D * ev.D);
        st.minD = st.minD < ev.D ? st.minD : ev.D;
        st.maxD = st.maxD > ev.D ? st.maxD : ev.D;
        st.n += 1;
        if (t == 0 && static_cast<std::size_t>(i) < first_D.extent(0)) {
          first_D(i) = ev.D;
        }
      }
      pool.free_state(gen);
    }
  };

  DrawStats device_draws(const Tables& T, double gamma, bool weighted, unsigned long long seed, int n_threads,
                         int per_thread, array_t<double*> first_D) {
    random_number_pool_t pool(seed);
    DrawStats            st;
    Kokkos::parallel_reduce("draws", Kokkos::RangePolicy<>(0, n_threads),
                            DrawKernel { T.sampler, pool, gamma, per_thread, weighted, first_D },
                            Kokkos::Sum<DrawStats>(st));
    Kokkos::fence();
    return st;
  }

  void test_tables(const Tables& T) {
    std::printf("---- device tables and chains vs the host chain ----\n");
    verdict(T.sampler.bytes <= 64u * 1024u * 1024u, "device sampler tables within 64 MB (bytes)",
            static_cast<double>(T.sampler.bytes), 64.0 * 1024.0 * 1024.0);
    for (const double g : { 2.3, 3.0, 10.0, 60.0 }) {
      const auto  R  = T.rates->at_gamma(g);
      array_t<double*> fd { "fd", 1 };
      const auto t0 = std::chrono::steady_clock::now();
      const auto  su = device_draws(T, g, false, 11u, 64, 1000000 / 64, fd);
      const double tu = seconds_since(t0);
      const auto  sw = device_draws(T, g, true, 12u, 64, 200000 / 64, fd);
      const double meanD = su.sD / su.n, seD = std::sqrt(std::max(0.0, su.sD2 / su.n - meanD * meanD) / su.n);
      // the host chain at the same Gamma
      rf::SplitMix64 rng(13u);
      double         hs = 0.0, hs2 = 0.0;
      const int      Nh = 1000000;
      for (int i = 0; i < Nh; ++i) {
        const double D = T.host_sampler->draw(g, false, rng).D;
        hs += D;
        hs2 += D * D;
      }
      const double hm = hs / Nh, hse = std::sqrt(std::max(0.0, hs2 / Nh - hm * hm) / Nh);
      const double nse = std::abs(meanD - hm) / std::sqrt(seD * seD + hse * hse);
      verdict(nse <= 4.0, "device chain vs host chain, <D> at Gamma_p = " + fmt("%g", g) + " (1e6 draws each), |dev|/s.e.",
              nse, 4.0);
      verdict(su.oob == 0 && sw.oob == 0, "device draws inside [D_min, D_max] at Gamma_p = " + fmt("%g", g),
              static_cast<double>(su.oob + sw.oob), 0.0);
      const double m_inv = sw.sinv / sw.n, se_inv = std::sqrt(std::max(0.0, sw.sinv2 / sw.n - m_inv * m_inv) / sw.n);
      const double prod = m_inv * meanD;
      verdict(std::abs(prod - 1.0) <= 4.0 * se_inv * meanD,
              "device weighted chain: E_p[1/D] <D>_device = 1 at Gamma_p = " + fmt("%g", g) + ", |dev|/s.e.",
              std::abs(prod - 1.0) / (se_inv * meanD), 4.0);
      diag("Gamma_p " + fmt("%g", g) + ": device <D> " + fmt("%.5f", meanD) + " +- " + fmt("%.5f", seD) + ", host " +
           fmt("%.5f", hm) + ", I/J " + fmt("%.5f", R.Dmean) + ", min/max D " + fmt("%.4f", su.minD) + " / " +
           fmt("%.2f", su.maxD) + "; weighted: efficiency " + fmt("%.4f", static_cast<double>(sw.n) / sw.attempts) +
           ", overflow " + std::to_string(sw.fails) + "; " + fmt("%.2f", tu / su.n * 1.0e6) +
           " us per unweighted draw (OpenMP), rng per draw " + fmt("%.1f", static_cast<double>(su.nrng) / su.n));
    }
  }

  void test_a11(const Tables& T) {
    std::printf("---- A11: mu_d independence ----\n");
    // (1) the device sampler proper: it takes Gamma_p and the RNG; the three
    // mu_d labels do not enter. Same seed -> bitwise-identical draws.
    std::vector<std::vector<double>> seqs;
    for (const double mu_d : { 1.0, 25.0, 1836.15 }) {
      (void)mu_d;
      array_t<double*> fd { "fd", 1000 };
      device_draws(T, 3.0, true, 777u, 1, 1000, fd);
      auto h = mirror(fd);
      seqs.emplace_back(h.data(), h.data() + 1000);
    }
    bool same = true;
    for (int i = 0; i < 1000; ++i) {
      same = same && (seqs[0][i] == seqs[1][i]) && (seqs[0][i] == seqs[2][i]);
    }
    verdict(same, "A11: device sampler at Gamma_p = 3, same seed, 1000 weighted draws bitwise identical for mu_d in "
                  "{1, 25, 1836.15} (the sampler sees only Gamma_p)",
            same ? 0.0 : 1.0, 0.0);
    // (2) the kernel: ions at equal K for the three mu_d, preset debt, one launch,
    // same pool seed: the spectra agree; bitwise identity holds only up to the
    // real_t rounding of u (K from the stored u differs between mu_d at 1e-7)
    std::vector<std::vector<double>> Es;
    std::vector<double>              Kstored;
    for (const double mu_d : { 1.0, 25.0, 1836.15 }) {
      Ions  ions(64, mu_d);
      Pairs pairs(64 * 1000 + 16);
      std::vector<double> K(64, 2.0 * MU_R), debt(64), k0(64);
      ions.set_all(K);
      const auto Ks = ions.K();
      Kstored.push_back(Ks[0]);
      for (int p = 0; p < 64; ++p) {
        debt[p] = 500.0 * T.rates->at_gamma(1.0 + Ks[p] / MU_R).Dmean;
        k0[p]   = Ks[p];
      }
      ions.set_payloads(debt, k0);
      auto prm                = params(1.0, BUDGET, RESTFRAME, 1.0e-6);
      prm.eta_u               = 1.0e9;
      prm.r_th_factor         = 1.0;
      prm.max_events_per_step = 1000;
      random_number_pool_t pool(4242u);
      launch(ions, pairs, T, prm, pool, 1.0e-6);
      const auto rows = pairs.rows();
      std::vector<double> E;
      for (const auto& r : rows) {
        E.push_back(kh::lepton_gamma(r.u1m, r.u2m, r.u3m));
      }
      Es.push_back(E);
    }
    std::size_t nmin = std::min({ Es[0].size(), Es[1].size(), Es[2].size() });
    std::size_t n_identical = 0;
    for (std::size_t i = 0; i < nmin; ++i) {
      if (Es[0][i] == Es[1][i] && Es[0][i] == Es[2][i]) {
        ++n_identical;
      }
    }
    // distributions: sorted energies compared by the KS statistic
    auto ks = [](std::vector<double> a, std::vector<double> b) {
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      double d = 0.0;
      for (std::size_t i = 0; i < a.size(); ++i) {
        const double Fa = static_cast<double>(i + 1) / a.size();
        const double Fb = static_cast<double>(std::upper_bound(b.begin(), b.end(), a[i]) - b.begin()) / b.size();
        d               = std::max(d, std::abs(Fa - Fb));
      }
      return d;
    };
    const double k01 = ks(Es[0], Es[1]), k02 = ks(Es[0], Es[2]);
    const double crit = 1.628 * std::sqrt(2.0 / Es[0].size());
    verdict(k01 <= crit && k02 <= crit,
            "A11 (kernel): e- lab spectra of the budget+restframe kernel at equal K for mu_d in {1, 25, 1836.15} "
            "agree (KS vs the 99% critical value " + fmt("%.4f", crit) + "); value = max KS",
            std::max(k01, k02), crit);
    diag("A11 (kernel): " + std::to_string(Es[0].size()) + " / " + std::to_string(Es[1].size()) + " / " +
         std::to_string(Es[2].size()) + " pairs; stored K for the three mu_d: " + fmt("%.10f", Kstored[0]) + ", " +
         fmt("%.10f", Kstored[1]) + ", " + fmt("%.10f", Kstored[2]) + " (relative spread " +
         fmt("%.2e", (std::max({ Kstored[0], Kstored[1], Kstored[2] }) - std::min({ Kstored[0], Kstored[1], Kstored[2] })) /
                       Kstored[0]) +
         "); pairs with bitwise-identical e- energies across the three runs: " + std::to_string(n_identical) +
         " of " + std::to_string(nmin) + " (not expected to be all: Gamma_p differs at the real_t rounding of u)");
  }

  void test_identity(const Tables& T) {
    std::printf("---- the ledger identity with sampled energies (budget and poisson, restframe) ----\n");
    for (const int source : { BUDGET, POISSON }) {
      Ions  ions(64, 1.0);
      Pairs pairs(200000);
      std::vector<double> K(64, 5.0 * MU_R); // Gamma_p = 6
      ions.set_all(K);
      const auto   K_init = ions.K();
      const double I6     = bh::interpolate_loss_host(T.host, 6.0);
      const double dt     = 0.5;
      const double coeff  = 1.0e-3 * K[0] / (I6 * dt); // dK/K = 1e-3 per step
      auto         prm    = params(coeff, source, RESTFRAME, dt);
      if (source == POISSON) {
        prm.pair_weight = 50.0;
      }
      random_number_pool_t pool(99u);
      HostLedger           H;
      for (int n = 0; n < 200; ++n) {
        H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * dt));
      }
      const auto A = array_identity(ions, K_init, pairs, H);
      const std::string tag = source == BUDGET ? "budget+restframe" : "poisson+restframe";
      check_identity(tag, H, A);
      // the eps payload is the sampled eps (mono: exactly eps0) and P_absorbed has
      // the magnitude sum w eps along the sampled directions (|P_absorbed| <= E_absorbed)
      const auto rows = pairs.rows();
      double     weps = 0.0;
      for (const auto& r : rows) {
        weps = std::max(weps, std::abs(r.eps_m - EPS0));
      }
      verdict(weps == 0.0, tag + ": pair payload slot 2 = the sampled eps (mono: eps0 exactly), max |eps_s - eps0|",
              weps, 0.0);
      const double Pabs = std::sqrt(H.C.P_absorbed[0] * H.C.P_absorbed[0] + H.C.P_absorbed[1] * H.C.P_absorbed[1] +
                                    H.C.P_absorbed[2] * H.C.P_absorbed[2]);
      verdict(Pabs <= H.C.E_absorbed.value() * (1.0 + 1.0e-12),
              tag + ": |P_absorbed| <= E_absorbed (the photon momenta are no longer collinear), ratio",
              Pabs / H.C.E_absorbed.value(), 1.0);
      // the pair momenta are not collinear with the ion: transverse momentum present
      double pt = 0.0;
      for (const auto& r : rows) {
        pt = std::max(pt, std::sqrt(r.u2m * r.u2m + r.u3m * r.u3m));
      }
      verdict(pt > 0.0, tag + ": leptons carry transverse momentum (max |u_perp| of e-)", pt, 0.0);
      diag(tag + ": E_drag " + fmt("%.6f", H.C.E_drag.value()) + ", pairs " + std::to_string(H.C.n_spawned) +
           ", E_round " + fmt("%.4e", H.C.E_round.value()) + ", B " + fmt("%.4e", H.C.B_budget.value()) +
           ", E_stochastic " + fmt("%.6f", H.C.E_stochastic.value()) + ", sampler attempts " +
           std::to_string(H.C.n_sampler_attempts) + ", overflow " + std::to_string(H.C.n_sampler_overflow) +
           ", n_rng " + std::to_string(H.C.n_rng) + ", |P_absorbed|/E_absorbed " +
           fmt("%.4f", Pabs / H.C.E_absorbed.value()) + ", P_pairs x " + fmt("%.3f", H.C.P_pairs[0]) +
           " (P_drag x " + fmt("%.3f", H.C.P_drag[0]) + ")");
    }
  }

  void test_a12(const Tables& T) {
    std::printf("---- A12: mean cost, pair number and energy per unit paid energy at every rate-table node ----\n");
    const std::size_t nn = T.host.size();
    const int   per_node = 2, launches = 10, pk = 1000;
    Ions        ions(nn * per_node, 1.0);
    Pairs       pairs(nn * per_node * pk + 64);
    std::vector<double> K(nn * per_node), debt(nn * per_node), k0(nn * per_node);
    for (std::size_t i = 0; i < nn; ++i) {
      for (int j = 0; j < per_node; ++j) {
        K[i * per_node + j] = MU_R * (T.host.gamma[i] - 1.0);
      }
    }
    ions.set_all(K);
    const auto Ks = ions.K();
    for (std::size_t p = 0; p < nn * per_node; ++p) {
      const double Gp = 1.0 + Ks[p] / MU_R;
      debt[p]         = (pk + 1) * T.rates->at_gamma(Gp).Dmean;
      k0[p]           = Ks[p];
    }
    auto prm                = params(1.0, BUDGET, RESTFRAME, 1.0e-6);
    prm.eta_u               = 1.0e9;
    prm.r_th_factor         = 1.0;
    prm.max_events_per_step = pk;
    std::vector<double> sinv(nn, 0.0), sinv2(nn, 0.0), sE(nn, 0.0), cnt(nn, 0.0);
    std::vector<double> gp_pay(nn, 0.0);
    unsigned long long  attempts = 0, overflow = 0, spawned = 0;
    const auto          t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < launches; ++l) {
      ions.set_payloads(debt, k0);
      pairs.reset();
      random_number_pool_t pool(1000u + l);
      const auto L = launch(ions, pairs, T, prm, pool, 1.0e-6);
      attempts += L.n_sampler_attempts;
      overflow += L.n_sampler_overflow;
      spawned += L.n_spawned;
      const auto rows = pairs.rows();
      for (const auto& r : rows) {
        const std::size_t node = static_cast<std::size_t>(r.i1m) / per_node;
        const double      E_s  = kh::lepton_gamma(r.u1m, r.u2m, r.u3m) + kh::lepton_gamma(r.u1p, r.u2p, r.u3p);
        const double      D    = E_s - r.eps_m;
        sinv[node] += 1.0 / D;
        sinv2[node] += 1.0 / (D * D);
        sE[node] += E_s / D;
        cnt[node] += 1.0;
        gp_pay[node] = std::max(gp_pay[node], std::abs(r.gp_m / (1.0 + Ks[r.i1m] / MU_R) - 1.0));
      }
    }
    diag("A12: " + std::to_string(spawned) + " pairs at " + std::to_string(nn) + " nodes in " +
         fmt("%.0f", seconds_since(t0)) + " s; sampler attempts " + std::to_string(attempts) + " (" +
         fmt("%.1f", static_cast<double>(attempts) / spawned) + " per pair), overflow " + std::to_string(overflow));
    // the chain's quadrature (tables' q_k)
    const rf::QkFineGrid    qk_tab = rf::table_qk_fine_grid(*T.rft, 8);
    const rf::QkCumulatives cum_tab([&](double k) { return qk_tab(k); });
    double w_cost = 0.0, w_num = 0.0, w_en = 0.0, g_cost = 0.0, maxse = 0.0, worst_nse = 0.0, g_num = 0.0;
    int    n_fail_cost = 0, n_fail_num = 0, n_over_1pc = 0, n_over_2pc = 0;
    for (std::size_t i = 0; i < nn; ++i) {
      const double N      = cnt[i];
      const double minv   = sinv[i] / N;
      const double se_inv = std::sqrt(std::max(0.0, sinv2[i] / N - minv * minv) / N);
      const double cost   = 1.0 / minv;                 // paid energy per unit-weight pair
      const double rel    = std::abs(cost / T.host.Dmean[i] - 1.0);
      const double nse    = std::abs(minv - 1.0 / T.host.Dmean[i]) / se_inv;
      if (!(rel <= 1.0e-2 || nse <= 4.0)) {
        ++n_fail_cost;
      }
      if (rel > w_cost) {
        w_cost = rel;
        g_cost = T.host.gamma[i];
      }
      maxse     = std::max(maxse, se_inv / minv);
      worst_nse = std::max(worst_nse, nse);
      if (rel > 1.0e-2) {
        ++n_over_1pc;
      }
      const double Dchain = rf::direct_boost_I(*T.bath, cum_tab, T.host.gamma[i]) / T.host.J[i];
      // pairs per unit paid energy = mean(1/D) vs 1/<D>_chain (the host chain's quadrature)
      const double rel_num = std::abs(minv * Dchain - 1.0);
      const double nse_num = std::abs(minv - 1.0 / Dchain) / se_inv;
      if (!(rel_num <= 2.0e-2 || nse_num <= 4.0)) {
        ++n_fail_num;
      }
      if (rel_num > 2.0e-2) {
        ++n_over_2pc;
      }
      if (rel_num > w_num) {
        w_num = rel_num;
        g_num = T.host.gamma[i];
      }
      w_en = std::max(w_en, std::abs((sE[i] / N) / (1.0 + EPS0 / Dchain) - 1.0)); // energy per paid energy
    }
    verdict(n_fail_cost == 0,
            "A12: the sampler's mean cost per pair vs I/J at every node, within max(1%, 4 s.e.): failing nodes",
            static_cast<double>(n_fail_cost), 0.0);
    diag("A12: max |cost/(I/J) - 1| over the nodes " + fmt("%.3e", w_cost) + " at Gamma_p " + fmt("%.4f", g_cost) +
         "; nodes beyond 1%: " + std::to_string(n_over_1pc) + " (all within 4 s.e.: the relative s.e. of the cost "
         "reaches 1% at the upper nodes with 2e4 pairs per node)");
    verdict(n_fail_num == 0,
            "A12: pairs per unit paid energy vs 1/<D>_chain (host quadrature) at every node, within max(2%, 4 s.e.): "
            "failing nodes",
            static_cast<double>(n_fail_num), 0.0);
    diag("A12: max |pairs per paid energy x <D>_chain - 1| = " + fmt("%.3e", w_num) + " at Gamma_p " +
         fmt("%.4f", g_num) + "; nodes beyond 2%: " + std::to_string(n_over_2pc));
    verdict(w_en <= 2.0e-2, "A12: pair energy per unit paid energy vs 1 + eps0/<D>_chain, every node", w_en, 2.0e-2);
    double wg = 0.0;
    for (const double v : gp_pay) {
      wg = std::max(wg, v);
    }
    verdict(wg <= 1.0e-6, "A12: pair payload Gamma_p = the parent's Gamma_p", wg, 1.0e-6);
    diag("A12: relative s.e. of the cost up to " + fmt("%.2e", maxse) + " (" + fmt("%.0f", cnt[0]) +
         " pairs per node); max |dev|/s.e. vs I/J " + fmt("%.1f", worst_nse) +
         " (the step-1 fit vs the Born law, see the report); overflow " + std::to_string(overflow));
    for (const std::size_t i : { std::size_t(0), nn / 4, nn / 2, (3 * nn) / 4, nn - 1 }) {
      const double N = cnt[i], minv = sinv[i] / N;
      const double Dchain = rf::direct_boost_I(*T.bath, cum_tab, T.host.gamma[i]) / T.host.J[i];
      diag("A12 node Gamma_p " + fmt("%.5f", T.host.gamma[i]) + ": cost " + fmt("%.5f", 1.0 / minv) + ", I/J " +
           fmt("%.5f", T.host.Dmean[i]) + ", <D>_chain " + fmt("%.5f", Dchain) + ", energy per paid energy " +
           fmt("%.5f", sE[i] / N) + " vs " + fmt("%.5f", 1.0 + EPS0 / Dchain) + ", pairs " + fmt("%.0f", N));
    }
  }

  void test_spectrum(const Tables& T) {
    std::printf("---- the spectrum of created pairs vs the host chain's histogram (hist_mono05_G3) ----\n");
    const std::string path = std::string(BH_TEST_DATA_DIR) + "/cpp-sampler-candidate/hist_mono05_G3.txt";
    std::ifstream     probe(path);
    if (!probe) {
      diag("fixture " + path + " not present: the spectrum comparison is skipped");
      return;
    }
    probe.close();
    const bh::FixtureContents F = bh::read_fixture_file(path);
    const int                 NB = static_cast<int>(F.rows.size());
    // kernel run at Gamma_p = 3: 100 ions, preset debt, 1000 packets each
    const int   nions = 100, pk = 1000;
    Ions        ions(nions, 1.0);
    Pairs       pairs(nions * pk + 64);
    std::vector<double> K(nions, 2.0 * MU_R), debt(nions), k0(nions);
    ions.set_all(K);
    const auto Ks = ions.K();
    for (int p = 0; p < nions; ++p) {
      debt[p] = (pk + 1) * T.rates->at_gamma(1.0 + Ks[p] / MU_R).Dmean;
      k0[p]   = Ks[p];
    }
    ions.set_payloads(debt, k0);
    auto prm                = params(1.0, BUDGET, RESTFRAME, 1.0e-6);
    prm.eta_u               = 1.0e9;
    prm.r_th_factor         = 1.0;
    prm.max_events_per_step = pk;
    random_number_pool_t pool(2024u);
    launch(ions, pairs, T, prm, pool, 1.0e-6);
    const auto rows = pairs.rows();
    // weighted histograms on the fixture's bins
    std::vector<double> wE(NB, 0.0), wC(NB, 0.0);
    double              sw = 0.0, sw2 = 0.0;
    const double        E_lo = F.rows.front()[0], E_hi = F.rows.back()[1];
    for (const auto& r : rows) {
      const double gm   = kh::lepton_gamma(r.u1m, r.u2m, r.u3m);
      const double um   = std::sqrt(r.u1m * r.u1m + r.u2m * r.u2m + r.u3m * r.u3m);
      const double cosm = r.u1m / um; // the ion moves along +x
      const double w    = r.wm;
      sw += w;
      sw2 += w * w;
      int ib = static_cast<int>(std::floor(std::log(gm / E_lo) / std::log(E_hi / E_lo) * NB));
      ib     = std::min(NB - 1, std::max(0, ib));
      wE[ib] += w;
      int ic = static_cast<int>(std::floor((cosm + 1.0) / 2.0 * NB));
      ic     = std::min(NB - 1, std::max(0, ic));
      wC[ic] += w;
    }
    const double n_eff_k = sw * sw / sw2;
    // the fixture's effective size: weights 1/D over N accepted (header n_weighted_accepted)
    double n_acc = 1.0e6;
    if (F.header.count("n_weighted_accepted")) {
      n_acc = std::stod(F.header.at("n_weighted_accepted"));
    }
    // Kish effective size of the fixture's weighted sample from its D histogram
    // (weights 1/D, D taken at the bin centre): with the normalised bin weights
    // wt_i, the accepted count in bin i is ~ wt_i D_i, so
    // n_eff = n_acc / (sum_i wt_i D_i * sum_i wt_i / D_i)
    double s1 = 0.0, s2 = 0.0;
    for (const auto& r : F.rows) {
      const double wt = r[19], Dmid = 0.5 * (r[16] + r[17]); // w_wt_D, D bin centre
      s1 += wt * Dmid;
      s2 += wt / Dmid;
    }
    const double n_eff_fix = n_acc / (s1 * s2);
    const double crit = 1.628 * std::sqrt((n_eff_k + n_eff_fix) / (n_eff_k * n_eff_fix));
    auto ks_binned = [&](const std::vector<double>& a, int col) {
      double Fa = 0.0, Fb = 0.0, d = 0.0;
      for (int i = 0; i < NB; ++i) {
        Fa += a[i] / sw;
        Fb += F.rows[i][col];
        d = std::max(d, std::abs(Fa - Fb));
      }
      return d;
    };
    const double ksE = ks_binned(wE, 3), ksC = ks_binned(wC, 11);
    verdict(ksE <= crit, "A12 spectrum: e- lab energy of the kernel's pairs (weights 1/D, " +
                           std::to_string(rows.size()) + " pairs, N_eff " + fmt("%.0f", n_eff_k) +
                           ") vs hist_mono05_G3 weighted chain (N_eff " + fmt("%.0f", n_eff_fix) +
                           "), binned two-sample KS vs the 99% critical value",
            ksE, crit);
    verdict(ksC <= crit, "A12 spectrum: e- lab cos theta (to the ion direction) vs hist_mono05_G3, binned KS vs the 99% "
                         "critical value",
            ksC, crit);
    diag("spectrum: KS energy " + fmt("%.5f", ksE) + ", KS cos " + fmt("%.5f", ksC) + ", critical " +
         fmt("%.5f", crit) + "; fixture version " + F.header.at("version"));
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("bh_sampler: real_t = %s, Kokkos default execution space = %s\n",
                sizeof(real_t) == 4 ? "float" : "double", Kokkos::DefaultExecutionSpace::name());
    std::vector<std::string> groups;
    for (int i = 1; i < argc; ++i) {
      if (argv[i][0] != '-') {
        groups.emplace_back(argv[i]);
      }
    }
    auto want = [&groups](const char* g) {
      return groups.empty() || std::find(groups.begin(), groups.end(), std::string(g)) != groups.end();
    };
    const auto T = make_tables();
    if (want("tables")) {
      test_tables(T);
    }
    if (want("a11")) {
      test_a11(T);
    }
    if (want("identity")) {
      test_identity(T);
    }
    if (want("a12")) {
      test_a12(T);
    }
    if (want("spectrum")) {
      test_spectrum(T);
    }
    std::printf("bh_sampler summary: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0) {
      rc = 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_sampler: exception: %s\n", e.what());
    rc = 2;
  }
  Kokkos::finalize();
  return rc;
}
