/**
 * @file tests/kernels/bh_sampler.cpp
 * @brief The rest-frame sampler in the Bethe-Heitler kernel (P2 step 4, stage
 *        4): the device tables and the device chains against the host chain,
 *        A11 (mu_d independence), the ledger identity with sampled energies,
 *        A12 (the sampler's mean cost, pair number and pair energy per unit
 *        paid energy at every Gamma_p node vs the reference's Born-chain <D>;
 *        round 2), the spectrum of created pairs vs the reference's
 *        histograms (round 2).
 *
 * Conventions (declared before any run):
 *   - bath mono eps0 = 0.5 (Gamma_th = 2.125), mu_r = 1836.15, ions with
 *     mu_d = 1 unless stated, ions along +x at i1 = p, dx1 = 0.5
 *   - the device sampler tables are those of the generator's startup path
 *     (`build_sampler_tables`: 16 eps' nodes per decade, 256 x 256 cells,
 *     single-precision CDFs); the host chain is `rf::HostSampler` on the same
 *     host tables in double
 *   - the tables group compares the device and host chains for mono05 and,
 *     since round 2, for the powerlaw bath (eps_min 0.1, eps_max 1.5, index 2;
 *     the device photon-energy tables of a broad bath, per-node grids)
 *   - "exact" means to double round-off: 1e-12 relative to E_drag
 *   - A12 (round 2, decision (b) of 2026-10-27): ions at the 466 rate-table
 *     nodes (2 per node), a preset debt of 1001 <D>(Gamma_i), r_th_factor = 1
 *     (r_th = <D>), f_r = 0.02, max_events_per_step = 1000, eta_u = 1e9 (no
 *     application: the energies stay at the nodes), 100 launches: N = 2e5
 *     pairs per node, fixed; cost = N / sum(1/D_j) over the node's pairs, with
 *     D_j = gamma- + gamma+ - eps from the stored values. Literal tolerances,
 *     no uncertainty escape: mean cost within 1%, pairs per unit paid energy
 *     and pair energy per unit paid energy within 2%, of the reference's
 *     Born-chain values (bhref-0.2.0/moments_mono05.txt: <D>; its
 *     nodes must be this table's nodes to 1e-12); every failing node listed.
 *     The step-1 I/J (eq. 3.7 fit, 0.1 to 0.5% off the Born law) and the host
 *     chain's sampled <D> are diagnostics.
 *   - the spectrum runs (round 2, finding 4): for mono05 and powerlaw at the
 *     reference's histogram Gamma_p that lie inside the bath's rate table
 *     (mono05: 2.23125, 3, 10; powerlaw: 1.09375, 3, 10; Gamma_p = 100 is
 *     above both tables and is not reachable by the box), 100 ions at Gamma_p
 *     (K = mu_r (Gamma_p - 1), to the real_t rounding of u), 1e5 pairs with
 *     weights 1/D (r_packet constant), compared with the reference's
 *     weighted-chain histograms (bhref-0.2.0/hist_<bath>_G<Gamma>.txt; D at
 *     Gamma_p = 10 on the log bins of bhref-0.2.0-suppl1/hist_logD_*.txt) in
 *     E-, E+, cos theta-, cos theta+ and D on the reference's bins, by the
 *     calibrated weighted two-sample test of bhref-0.2.0-suppl1/COMPARISON.md
 *     (round 3, Claude 3's F8; bh_calibrated_test.h): bins merged from left to
 *     right until the kernel's packet count and the reference's weighted
 *     packet count both reach 100 (the device draws the weighted chain only,
 *     so these are the two counts that exist), the right-edge discrete
 *     Anderson-Darling statistic on the per-bin sums of w and w^2, its
 *     conditional Gaussian-multiplier null with 19,999 replicates, pass iff
 *     p >= 0.01; null seeds 2026102901 + 10 case + quantity (case 0..5 =
 *     mono05 2.23125, 3, 10, powerlaw 1.09375, 3, 10; quantity 0..4 = E-, E+,
 *     cos-, cos+, D); at least 95% of the 30 comparisons must pass, the
 *     failures are listed
 *
 * Verdict lines: "PASS" / "FAIL" with the measured number and the tolerance;
 * "(diag)" lines are reported only. One ctest entry per group.
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "metrics/minkowski.h"

#include "archetypes/hadronic/bh_calibrated_test.h"
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

  bh::BathSpec mono05_spec() {
    bh::BathSpec spec;
    spec.type = "mono";
    spec.eps0 = EPS0;
    return spec;
  }

  Tables make_tables(const bh::BathSpec& spec = mono05_spec()) {
    Tables T;
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

  // the powerlaw bath's tables (the device photon-energy tables of a broad bath), built once on demand;
  // they hold Kokkos views, so main releases them before Kokkos::finalize
  std::unique_ptr<Tables> g_powerlaw;

  const Tables& powerlaw_tables() {
    auto& T = g_powerlaw;
    if (!T) {
      bh::BathSpec spec;
      spec.type    = "powerlaw";
      spec.eps_min = 0.1;
      spec.eps_max = 1.5;
      spec.index   = 2.0;
      T            = std::make_unique<Tables>(make_tables(spec));
    }
    return *T;
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

  void test_tables_bath(const Tables& T, const std::string& bath, const std::vector<double>& gammas) {
    verdict(T.sampler.bytes <= 64u * 1024u * 1024u, bath + ": device sampler tables within 64 MB (bytes)",
            static_cast<double>(T.sampler.bytes), 64.0 * 1024.0 * 1024.0);
    for (const double g : gammas) {
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
      verdict(nse <= 4.0, bath + ": device chain vs host chain, <D> at Gamma_p = " + fmt("%g", g) +
                            " (1e6 draws each), |dev|/s.e.",
              nse, 4.0);
      verdict(su.oob == 0 && sw.oob == 0, bath + ": device draws inside [D_min, D_max] at Gamma_p = " + fmt("%g", g),
              static_cast<double>(su.oob + sw.oob), 0.0);
      const double m_inv = sw.sinv / sw.n, se_inv = std::sqrt(std::max(0.0, sw.sinv2 / sw.n - m_inv * m_inv) / sw.n);
      const double prod = m_inv * meanD;
      verdict(std::abs(prod - 1.0) <= 4.0 * se_inv * meanD,
              bath + ": device weighted chain: E_p[1/D] <D>_device = 1 at Gamma_p = " + fmt("%g", g) + ", |dev|/s.e.",
              std::abs(prod - 1.0) / (se_inv * meanD), 4.0);
      diag(bath + " Gamma_p " + fmt("%g", g) + ": device <D> " + fmt("%.5f", meanD) + " +- " + fmt("%.5f", seD) + ", host " +
           fmt("%.5f", hm) + ", I/J " + fmt("%.5f", R.Dmean) + ", min/max D " + fmt("%.4f", su.minD) + " / " +
           fmt("%.2f", su.maxD) + "; weighted: efficiency " + fmt("%.4f", static_cast<double>(sw.n) / sw.attempts) +
           ", overflow " + std::to_string(sw.fails) + "; " + fmt("%.2f", tu / su.n * 1.0e6) +
           " us per unweighted draw (OpenMP), rng per draw " + fmt("%.1f", static_cast<double>(su.nrng) / su.n));
    }
  }

  void test_tables(const Tables& T) {
    std::printf("---- device tables and chains vs the host chain ----\n");
    test_tables_bath(T, "mono05", { 2.3, 3.0, 10.0, 60.0 });
    // round 2: the device photon-energy tables of a broad bath (per-node grids, stage 2)
    test_tables_bath(powerlaw_tables(), "powerlaw", { 1.05, 1.3, 3.0, 10.0, 25.0 });
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

  // the reference's Born-chain moments of a bath (bhref 0.2.0 frozen fixtures): gamma_p -> (Dmean, Dmean_se)
  struct RefMoments {
    std::vector<double> gamma, D, se;
    std::string         version, sha;
  };

  RefMoments read_ref_moments(const std::string& bath) {
    const std::string         path = std::string(BH_TEST_DATA_DIR) + "/bhref-0.2.0/moments_" + bath + ".txt";
    const bh::FixtureContents F    = bh::read_fixture_file(path);
    auto col = [&](const std::string& n) {
      for (std::size_t c = 0; c < F.columns.size(); ++c) {
        if (F.columns[c] == n) {
          return c;
        }
      }
      throw std::runtime_error(path + ": no column " + n);
    };
    const std::size_t cg = col("gamma_p"), cd = col("Dmean"), cs = col("Dmean_se");
    RefMoments        R;
    for (const auto& r : F.rows) {
      R.gamma.push_back(r[cg]);
      R.D.push_back(r[cd]);
      R.se.push_back(r[cs]);
    }
    R.version = F.header.count("version") ? F.header.at("version") : "?";
    R.sha     = F.data_sha256;
    return R;
  }

  void test_a12(const Tables& T) {
    std::printf("---- A12 (decision (b)): mean cost, pair number and energy per unit paid energy at every rate-table "
                "node, N = 2e5, literal tolerances, the reference's Born-chain <D> ----\n");
    const std::size_t nn = T.host.size();
    // N = 2e5 packets per node: 2 ions per node, 1000 packets per ion and launch, 100 launches
    const int   per_node = 2, launches = 100, pk = 1000;
    const auto  R        = read_ref_moments("mono05");
    // deterministic columns: the reference's nodes must be this table's nodes (1e-12 relative)
    double w_node = (R.gamma.size() == nn) ? 0.0 : 1.0;
    for (std::size_t i = 0; i < std::min(nn, R.gamma.size()); ++i) {
      w_node = std::max(w_node, std::abs(R.gamma[i] / T.host.gamma[i] - 1.0));
    }
    verdict(w_node <= 1.0e-12,
            "A12: the reference's moments_mono05 nodes = this rate table's Gamma_p nodes (" + std::to_string(nn) +
              " nodes, deterministic, 1e-12 relative)",
            w_node, 1.0e-12);
    if (!(w_node <= 1.0e-12)) {
      return;
    }
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
         fmt("%.1f", static_cast<double>(attempts) / spawned) + " per pair), overflow " + std::to_string(overflow) +
         "; reference moments_mono05 version " + R.version + ", data_sha256 " + R.sha);
    // diagnostics: the step-1 I/J and the host chain's sampled <D> (the value the device chain converges to)
    const rf::QkFineGrid    qk16 = rf::table_qk_fine_grid(*T.rft, 16);
    const rf::SampledChainD SD(*T.cum, [&](double k) { return qk16(k); }, 16);
    double      w_cost = 0.0, w_num = 0.0, w_en = 0.0, g_cost = 0.0, g_num = 0.0, g_en = 0.0, maxse = 0.0;
    double      w_ij = 0.0, w_nse_samp = 0.0, w_nse_ref = 0.0, n_min = 1.0e300;
    double      exp_fail_cost = 0.0, exp_fail_num = 0.0; // expected failing nodes from the noise alone
    std::string fail_cost, fail_num, fail_en;
    int         n_fail_cost = 0, n_fail_num = 0, n_fail_en = 0;
    for (std::size_t i = 0; i < nn; ++i) {
      const double N      = cnt[i];
      n_min               = std::min(n_min, N);
      const double minv   = sinv[i] / N;
      const double se_inv = std::sqrt(std::max(0.0, sinv2[i] / N - minv * minv) / N);
      const double cost   = 1.0 / minv; // paid energy per unit-weight pair
      const double Dref   = R.D[i];
      const double rel    = std::abs(cost / Dref - 1.0);
      const double rel_n  = std::abs(minv * Dref - 1.0);
      const double en     = sE[i] / N, en_ref = 1.0 + EPS0 / Dref;
      const double rel_e  = std::abs(en / en_ref - 1.0);
      // combined s.e. of cost/Dref (delta method: device sample and the reference's own s.e.)
      const double se_c   = std::sqrt((se_inv / minv) * (se_inv / minv) + (R.se[i] / Dref) * (R.se[i] / Dref));
      const double Dsamp  = SD.Dmean_unweighted(*T.eps_tables, T.host.gamma[i], 16);
      w_nse_ref           = std::max(w_nse_ref, std::abs(cost / Dref - 1.0) / se_c);
      exp_fail_cost += std::erfc(1.0e-2 / (se_c * std::sqrt(2.0)));
      exp_fail_num += std::erfc(2.0e-2 / (se_c * std::sqrt(2.0)));
      w_nse_samp          = std::max(w_nse_samp, std::abs(minv - 1.0 / Dsamp) / se_inv);
      w_ij                = std::max(w_ij, std::abs(cost / T.host.Dmean[i] - 1.0));
      maxse               = std::max(maxse, se_inv / minv);
      auto note = [&](std::string& s, double v) {
        s += " Gamma_p " + fmt("%.5f", T.host.gamma[i]) + " (" + fmt("%+.3e", v) + ", " +
             fmt("%.1f", std::abs(cost / Dref - 1.0) / se_c) + " s.e.);";
      };
      if (!(rel <= 1.0e-2)) {
        ++n_fail_cost;
        note(fail_cost, cost / Dref - 1.0);
      }
      if (!(rel_n <= 2.0e-2)) {
        ++n_fail_num;
        note(fail_num, minv * Dref - 1.0);
      }
      if (!(rel_e <= 2.0e-2)) {
        ++n_fail_en;
        note(fail_en, en / en_ref - 1.0);
      }
      if (rel > w_cost) {
        w_cost = rel;
        g_cost = T.host.gamma[i];
      }
      if (rel_n > w_num) {
        w_num = rel_n;
        g_num = T.host.gamma[i];
      }
      if (rel_e > w_en) {
        w_en = rel_e;
        g_en = T.host.gamma[i];
      }
    }
    verdict(n_min >= 2.0e5, "A12: packets per node (N = 2e5 fixed), the smallest node count", n_min, 2.0e5);
    verdict(n_fail_cost == 0,
            "A12: mean cost per pair vs the reference's Born-chain <D> (moments_mono05 Dmean), |cost/<D>_ref - 1| <= 1% "
            "at every node (max " + fmt("%.3e", w_cost) + " at Gamma_p " + fmt("%.4f", g_cost) + "): failing nodes",
            static_cast<double>(n_fail_cost), 0.0);
    if (n_fail_cost > 0) {
      diag("A12 cost: failing nodes:" + fail_cost);
    }
    verdict(n_fail_num == 0,
            "A12: pairs per unit paid energy vs 1/<D>_ref, |<1/D> <D>_ref - 1| <= 2% at every node (max " +
              fmt("%.3e", w_num) + " at Gamma_p " + fmt("%.4f", g_num) + "): failing nodes",
            static_cast<double>(n_fail_num), 0.0);
    if (n_fail_num > 0) {
      diag("A12 pair number: failing nodes:" + fail_num);
    }
    verdict(n_fail_en == 0,
            "A12: pair energy per unit paid energy vs 1 + eps0/<D>_ref, <= 2% at every node (max " + fmt("%.3e", w_en) +
              " at Gamma_p " + fmt("%.4f", g_en) + "): failing nodes",
            static_cast<double>(n_fail_en), 0.0);
    if (n_fail_en > 0) {
      diag("A12 pair energy: failing nodes:" + fail_en);
    }
    double wg = 0.0;
    for (const double v : gp_pay) {
      wg = std::max(wg, v);
    }
    verdict(wg <= 1.0e-6, "A12: pair payload Gamma_p = the parent's Gamma_p", wg, 1.0e-6);
    diag("A12: relative s.e. of the cost up to " + fmt("%.2e", maxse) + "; max |cost/<D>_ref - 1| / combined s.e. " +
         fmt("%.1f", w_nse_ref) + "; max |<1/D> - 1/<D>_sampled| / s.e. " + fmt("%.1f", w_nse_samp) +
         " (the host chain's sampled density, SampledChainD); diagnostic vs the step-1 I/J (eq. 3.7 fit, expected "
         "0.1 to 0.5% off the Born law, decision (a)): max |cost/(I/J) - 1| " + fmt("%.3e", w_ij));
    diag("A12: nodes expected beyond the literal tolerance from the sampling noise alone (sum over the nodes of "
         "erfc(tol / (sqrt(2) combined relative s.e.))): " + fmt("%.2f", exp_fail_cost) + " for the 1% cost, " +
         fmt("%.2g", exp_fail_num) + " for the 2% pair number");
    for (const std::size_t i : { std::size_t(0), nn / 4, nn / 2, (3 * nn) / 4, nn - 1 }) {
      const double N = cnt[i], minv = sinv[i] / N;
      diag("A12 node Gamma_p " + fmt("%.5f", T.host.gamma[i]) + ": cost " + fmt("%.5f", 1.0 / minv) + ", <D>_ref " +
           fmt("%.5f", R.D[i]) + " +- " + fmt("%.5f", R.se[i]) + ", I/J " + fmt("%.5f", T.host.Dmean[i]) +
           ", energy per paid energy " + fmt("%.5f", sE[i] / N) + " vs " + fmt("%.5f", 1.0 + EPS0 / R.D[i]) +
           ", pairs " + fmt("%.0f", N));
    }
  }

  void test_spectrum(const Tables& Tmono) {
    std::printf("---- the spectrum of created pairs vs the reference's weighted-chain histograms (bhref 0.2.0, frozen), "
                "mono05 and powerlaw, every reference Gamma_p inside the rate table; the calibrated weighted "
                "two-sample test (bhref-0.2.0-suppl1/COMPARISON.md) ----\n");
    struct Combo {
      std::string what;
      cal::Result r;
    };
    std::vector<Combo>       combos;
    std::vector<std::string> not_reached;
    const char* const        qname[5] = { "E-", "E+", "cos-", "cos+", "D" };
    int                      case_index = 0;
    for (const std::string bath : { "mono05", "powerlaw" }) {
      const Tables& T = bath == "mono05" ? Tmono : powerlaw_tables();
      const std::vector<std::string> gs = bath == "mono05" ? std::vector<std::string> { "2.23125", "3", "10", "100" }
                                                           : std::vector<std::string> { "1.09375", "3", "10", "100" };
      for (std::size_t jg = 0; jg < gs.size(); ++jg) {
        const double g = std::stod(gs[jg]);
        if (g < T.host.gamma_min || g > T.host.gamma_max) {
          not_reached.push_back(bath + " Gamma_p " + gs[jg] + " (rate table [" + fmt("%.4f", T.host.gamma_min) + ", " +
                                fmt("%.4f", T.host.gamma_max) + "])");
          continue;
        }
        // the reference's bins and weighted-chain sums, quantity by quantity; D at Gamma_p >= 10 on the
        // supplement's log bins (as in A10)
        std::vector<std::vector<double>> edges(5), rn(5), rw(5), rw2(5);
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
            rn[q].push_back(r[5]);  // weighted_packet_count
            rw[q].push_back(r[6]);  // sum_1_over_D
            rw2[q].push_back(r[7]); // sum_1_over_D_squared
          }
        };
        read_ref("bhref-0.2.0/hist_" + bath + "_G" + gs[jg] + ".txt", false);
        if (g >= 10.0) {
          read_ref("bhref-0.2.0-suppl1/hist_logD_" + bath + "_G" + gs[jg] + ".txt", true);
        }
        // the kernel run: 100 ions at Gamma_p, 1000 packets each
        const int           nions = 100, pk = 1000;
        Ions                ions(nions, 1.0);
        Pairs               pairs(nions * pk + 64);
        std::vector<double> K(nions, MU_R * (g - 1.0)), debt(nions), k0(nions);
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
        random_number_pool_t pool(2024u + static_cast<unsigned>(jg) + (bath == "mono05" ? 0u : 100u));
        launch(ions, pairs, T, prm, pool, 1.0e-6);
        const auto rows = pairs.rows();
        std::vector<std::vector<double>> kn(5), kw(5), kw2(5);
        for (int q = 0; q < 5; ++q) {
          kn[q].assign(edges[q].size() - 1, 0.0);
          kw[q] = kw2[q] = kn[q];
        }
        std::size_t outside = 0;
        for (const auto& r : rows) {
          const double gm = kh::lepton_gamma(r.u1m, r.u2m, r.u3m), gp = kh::lepton_gamma(r.u1p, r.u2p, r.u3p);
          const double um = std::sqrt(r.u1m * r.u1m + r.u2m * r.u2m + r.u3m * r.u3m);
          const double up = std::sqrt(r.u1p * r.u1p + r.u2p * r.u2p + r.u3p * r.u3p);
          const double v[5] = { gm, gp, r.u1m / um, r.u1p / up, gm + gp - r.eps_m }; // the ion moves along +x
          const double w    = r.wm; // r_packet / D (r_packet constant)
          for (int q = 0; q < 5; ++q) {
            const auto& e = edges[q];
            if (v[q] < e.front() || v[q] > e.back()) {
              ++outside;
            }
            int i = static_cast<int>(std::upper_bound(e.begin(), e.end(), v[q]) - e.begin()) - 1;
            i     = std::min(static_cast<int>(e.size()) - 2, std::max(0, i));
            kn[q][i] += 1.0;
            kw[q][i] += w;
            kw2[q][i] += w * w;
          }
        }
        diag(bath + " Gamma_p " + gs[jg] + ": " + std::to_string(rows.size()) + " pairs; values outside the " +
             "reference's edges " + std::to_string(outside));
        for (int q = 0; q < 5; ++q) {
          // the common partition from the two packet counts that exist (the device draws the weighted chain only)
          const cal::Partition P = cal::merge_partition({ kn[q], rn[q] }, 100.0);
          const cal::Result    r = cal::calibrated(cal::merge_sums(P, kw[q]), cal::merge_sums(P, kw2[q]),
                                                   cal::merge_sums(P, rw[q]), cal::merge_sums(P, rw2[q]),
                                                   2026102901ULL + 10ULL * case_index + static_cast<unsigned>(q));
          const std::string what = bath + " Gamma_p " + gs[jg] + " " + qname[q] + (q == 4 && g >= 10.0 ? " (log bins)" : "");
          combos.push_back({ what, r });
          char b[256];
          if (r.insufficient) {
            std::snprintf(b, sizeof(b), "ESS_A %.0f ESS_B %.0f, 1 merged bin: INSUFFICIENT RESOLUTION (not a pass)",
                          r.ess_a, r.ess_b);
          } else {
            std::snprintf(b, sizeof(b), "ESS_A %.0f ESS_B %.0f T_AD %.4e p %.5f q99 %.4e merged bins %zu%s", r.ess_a,
                          r.ess_b, r.T_AD, r.p, r.q99, r.merged_bins, r.pass ? "" : " FAIL");
          }
          diag("spectrum " + what + ": " + b);
        }
        ++case_index;
      }
    }
    int         n_pass_c = 0;
    std::string failed;
    for (const auto& c : combos) {
      if (c.r.pass) {
        ++n_pass_c;
      } else {
        failed += " " + c.what + (c.r.insufficient ? " (insufficient resolution)" : " (p " + fmt("%.5f", c.r.p) + ")") +
                  ";";
      }
    }
    const double frac = combos.empty() ? 0.0 : static_cast<double>(n_pass_c) / static_cast<double>(combos.size());
    verdict(!combos.empty() && frac >= 0.95,
            "spectrum: the kernel's weighted pairs vs the reference's weighted-chain histograms (E-, E+, cos-, cos+, D; "
            "the calibrated weighted two-sample test of COMPARISON.md, pass iff p >= 0.01), fraction of the " +
              std::to_string(combos.size()) + " comparisons passing (>= 95% required)",
            frac, 0.95);
    diag("spectrum: " + std::to_string(n_pass_c) + " of " + std::to_string(combos.size()) +
         " pass; failures:" + (failed.empty() ? std::string(" none") : failed));
    std::string nr;
    for (const auto& s : not_reached) {
      nr += " " + s + ";";
    }
    diag("spectrum: not reachable by the box (outside the bath's rate table):" + nr +
         " the unweighted-chain histograms of the reference are not used here (the kernel draws the weighted chain)");
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
  g_powerlaw.reset();
  Kokkos::finalize();
  return rc;
}
