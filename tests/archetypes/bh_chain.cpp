/**
 * @file tests/archetypes/bh_chain.cpp
 * @brief The CDF tables and the two sampling chains of the rest-frame sampler
 *        on the host (P2 step 4, stage 2): the accuracy budget of contract
 *        section 3, the direct-boost loss table (A5, second part), A7 and A9
 *        of contract section 5, the acceptance efficiency, and the
 *        implementation-chosen checks of the kinematics.
 *
 * Contract items (tolerances registered in the contract, never adjusted here):
 *   budget  <E'->, <p' cos theta'>, q_k from the interpolated CDFs vs direct
 *           quadrature of the laws, <= 1e-3 at every eps' node of the tables;
 *           <D> at every Gamma_p node of every rate table <= 1e-3: round 2
 *           (finding 7), the mean of the density the unweighted chain samples
 *           (photon table, eps' interpolant, tables' q_k; rf::SampledChainD)
 *           against <D> of the laws (I with the laws' q_k over J), with its
 *           convergence under doubled quadrature orders
 *   photon  round 2 (finding 3, brief stage 2 (i)): <eps> of the photon-energy
 *           table vs the marginal by quadrature, < 1e-6 at every node of the
 *           broad baths, both chains (group `epsbias`)
 *   A5b     the direct-boost loss table I (laws' q_k) vs I_exact of step 1
 *           (eq. 3.7 q_k) <= 0.5% at every node (spec T3b)
 *   A7      <D>_MC = I/J at every rate-table node, N = 1e6 unweighted draws,
 *           |<D>_MC - I/J| <= max(1%, 4 s.e.) (4 s.e., no floor, on the Born
 *           tables: contract 1.5); the weighted chain: E_p[1/D] <D> = 1 within
 *           4 s.e. at Gamma_p in {1.05 Gamma_th, 3, 10} (group `a7w`, a smoke
 *           test of selected points) and at every rate-table node with
 *           exactly N = 1e6 accepted draws per node (group `a7wall`, round 2
 *           of step 6, Codex 1 F2: a resource limit aborts explicitly and
 *           never reduces N; the long group, run separately)
 *   A9      D >= 2 - eps_max over 1e7 draws per bath, zero violations; 1/min D
 *
 * Implementation-chosen checks (prefix "(impl)"): <D>_MC against the sampled
 * chain's quadrature within 5 s.e. at every node; E_p[1/D] <D>_sampled = 1 within 4
 * s.e.; D inside [D_min, D_max]; the lab four-momenta on shell and the inverse
 * boost returning the rest-frame energies; bitwise determinism for a fixed
 * seed. Diagnostics ([DIAG]): efficiency of the weighted chain vs Gamma_p,
 * cost per draw, table sizes.
 *
 * Seeds (convention): A7 node i of bath b uses SplitMix64(1000003 b + i);
 * the weighted smoke checks 7 000 000 + 10 b + j; the all-node weighted test
 * 7 500 000 + 1000 b + i; A9 9 000 000 + 100 b + j.
 */

#include "archetypes/hadronic/bh_born_qk.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_restframe.h"
#include "archetypes/hadronic/bh_sampler_host.h"
#include "archetypes/hadronic/bh_table_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace arch::bh;

namespace {

  int g_pass_contract = 0, g_fail_contract = 0;
  int g_pass_impl = 0, g_fail_impl = 0;

  enum Kind {
    CONTRACT,
    IMPL
  };

  void tally(Kind k, bool ok) {
    if (k == CONTRACT) {
      ok ? ++g_pass_contract : ++g_fail_contract;
    } else {
      ok ? ++g_pass_impl : ++g_fail_impl;
    }
  }

  bool check_abs_k(Kind k, const std::string& name, double measured, double tol,
                   const std::string& what = "max |rel. dev.|") {
    const bool ok = std::isfinite(measured) && std::abs(measured) <= tol;
    std::printf("[%s] %s%s: %s = %.6e (tol %.3e)\n", ok ? "PASS" : "FAIL", k == IMPL ? "(impl) " : "",
                name.c_str(), what.c_str(), measured, tol);
    tally(k, ok);
    return ok;
  }

  bool check_true_k(Kind k, const std::string& name, bool ok, const std::string& what) {
    std::printf("[%s] %s%s: %s\n", ok ? "PASS" : "FAIL", k == IMPL ? "(impl) " : "", name.c_str(), what.c_str());
    tally(k, ok);
    return ok;
  }

  bool check_abs(const std::string& name, double measured, double tol, const std::string& what = "max |rel. dev.|") {
    return check_abs_k(CONTRACT, name, measured, tol, what);
  }

  bool check_true(const std::string& name, bool ok, const std::string& what) {
    return check_true_k(CONTRACT, name, ok, what);
  }

  bool impl_abs(const std::string& name, double measured, double tol, const std::string& what = "max |rel. dev.|") {
    return check_abs_k(IMPL, name, measured, tol, what);
  }

  bool impl_true(const std::string& name, bool ok, const std::string& what) {
    return check_true_k(IMPL, name, ok, what);
  }

  template <class... Args>
  void diag(const char* fmt, Args... args) {
    std::printf("[DIAG] ");
    std::printf(fmt, args...);
    std::printf("\n");
  }

  double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  // one bath with everything the chains need
  struct BathSetup {
    rf::CanonicalBath                     cb;
    std::shared_ptr<const PhotonBath>     bath;
    std::shared_ptr<BetheHeitlerRates>    rates;
    HadronicTablesHost                    H;
    std::shared_ptr<rf::BathEpsTables>    eps_tables;
    std::shared_ptr<rf::HostSampler>      sampler;
  };

  // step 6 (contract 1.5): BH_QK_SOURCE=born runs the suite against the Born rate tables (ctest entries
  // ARCHETYPES::bh_chain_*_born); A7 then uses the re-registered criterion (4 s.e., no floor) and the
  // budget group's A5b becomes A5-numerical (1e-3 where I > 1e-3 of the peak)
  bool born_mode() {
    const char* e = std::getenv("BH_QK_SOURCE");
    return e != nullptr && std::string(e) == "born";
  }

  BathSetup make_setup(const rf::CanonicalBath& cb, const std::shared_ptr<const rf::RestFrameTables>& T,
                       const std::shared_ptr<const rf::SamplerCumulatives>& cum) {
    BathSetup S;
    S.cb         = cb;
    S.bath       = make_bath(cb.spec);
    Quadrature q;
    q.qk_source  = born_mode() ? QkSource::Born : QkSource::Fit;
    S.rates      = std::make_shared<BetheHeitlerRates>(S.bath, q);
    S.H          = build_tables(*S.rates, cb.mu_r);
    S.eps_tables = std::make_shared<rf::BathEpsTables>(rf::BathEpsTables::build(*S.bath, S.H, *cum));
    S.sampler    = std::make_shared<rf::HostSampler>(S.bath, S.H, T, cum, S.eps_tables);
    return S;
  }

  // ---------------------------------------------------------------------- //
  // accuracy budget at the eps' nodes
  // ---------------------------------------------------------------------- //
  void test_budget_nodes(const rf::RestFrameTables& T) {
    std::printf("\n== accuracy budget: table moments vs the laws at every eps' node ==\n");
    const rf::RestFrameQuadrature q;
    std::vector<double>           dE(T.n_k()), dP(T.n_k()), dQ(T.n_k());
    const auto                    t0 = std::chrono::steady_clock::now();
    rf::parallel_for_index(T.n_k(), [&](std::size_t i) {
      const double k  = 2.0 + std::exp(T.t_k[i]);
      const auto   tm = T.moments_at(k);
      const auto   m  = rf::moments(k, q);
      dE[i]           = std::abs(tm.mean_E / m.mean_E - 1.0);
      dP[i]           = std::abs(tm.mean_pcos / m.mean_pcos - 1.0);
      dQ[i]           = std::abs(tm.q_k / m.q_k - 1.0);
    });
    auto worst = [&](const std::vector<double>& v, double& kw) {
      double w = 0.0;
      kw       = 0.0;
      for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i] > w) {
          w  = v[i];
          kw = 2.0 + std::exp(T.t_k[i]);
        }
      }
      return w;
    };
    double kE, kP, kQ;
    const double wE = worst(dE, kE), wP = worst(dP, kP), wQ = worst(dQ, kQ);
    check_abs("budget: <E'-> from the interpolated CDFs vs quadrature of the laws, every eps' node", wE, 1.0e-3);
    check_abs("budget: <p' cos theta'> from the interpolated CDFs vs quadrature, every eps' node", wP, 1.0e-3);
    check_abs("budget: q_k from the interpolated CDFs vs quadrature, every eps' node", wQ, 1.0e-3);
    diag("budget: worst nodes eps' = %.6g (<E>), %.6g (<p cos>), %.6g (q_k); %zu nodes in %.1f s", kE, kP, kQ,
         T.n_k(), seconds_since(t0));
    for (const std::size_t i : { std::size_t(0), T.n_k() / 4, T.n_k() / 2, (3 * T.n_k()) / 4, T.n_k() - 1 }) {
      const double k  = 2.0 + std::exp(T.t_k[i]);
      const auto   tm = T.moments_at(k);
      const auto   m  = rf::moments(k, q);
      diag("budget: eps' = %10.5f  q_k table %.8f  laws %.8f  (%+.2e); <p cos> %+.2e; <E> %+.2e", k, tm.q_k, m.q_k,
           tm.q_k / m.q_k - 1.0, tm.mean_pcos / m.mean_pcos - 1.0, tm.mean_E / m.mean_E - 1.0);
    }
  }

  // ---------------------------------------------------------------------- //
  // <D> by quadrature over the chain at every Gamma node; A5b
  // ---------------------------------------------------------------------- //
  void test_budget_gamma(const std::vector<BathSetup>& setups, const rf::QkCumulatives& cum_laws,
                         const rf::QkCumulatives& cum_tab, const rf::SamplerCumulatives& cum,
                         const rf::QkFineGrid& qk_tab, const rf::RestFrameTables& T) {
    std::printf("\n== accuracy budget: <D> of the sampled chain at every Gamma_p node by quadrature over the tables; "
                "A5b ==\n");
    // round 2 (finding 7): the registered budget is held against the density the sampler draws from (the photon
    // table, the eps' interpolant and the tables' q_k), by quadrature with the tables' q_k on 16 sub-nodes per eps'
    // interval and Gauss-Legendre order 16; the convergence estimate doubles the sub-nodes (32) and raises the
    // orders to 24, and the verdict adds it to the deviation at every node
    (void)qk_tab;
    const auto              t_q  = std::chrono::steady_clock::now();
    const rf::QkFineGrid    qk16 = rf::table_qk_fine_grid(T, 16);
    const rf::QkFineGrid    qk32 = rf::table_qk_fine_grid(T, 32);
    const rf::SampledChainD SD(cum, [&](double k) { return qk16(k); }, 16);
    const rf::SampledChainD SDr(cum, [&](double k) { return qk32(k); }, 24);
    diag("sampled-chain quadrature: tables' q_k on 16 and 32 sub-nodes per eps' interval, %.1f s", seconds_since(t_q));
    double      w_s = 0.0, w_conv = 0.0, w_cert = 0.0, w_qonly = 0.0, w_a5b = 0.0, w_tab_vs_exact = 0.0;
    double      w_a5num = 0.0, g_a5num = 0.0;
    std::string n_s, n_a5b, n_qonly, n_a5num;
    double      g_s = 0.0;
    std::size_t n_nodes = 0;
    for (const auto& S : setups) {
      const auto          t0 = std::chrono::steady_clock::now();
      const std::size_t   ng = S.H.size();
      std::vector<double> rs(ng), rc(ng), rb(ng), ra(ng), rt(ng), Ds(ng), Dl(ng);
      rf::parallel_for_index(ng, [&](std::size_t i) {
        const double g      = S.H.gamma[i];
        const double I_laws = rf::direct_boost_I(*S.bath, cum_laws, g);
        const double I_tab  = rf::direct_boost_I(*S.bath, cum_tab, g);
        const double D_s    = SD.Dmean_unweighted(*S.eps_tables, g, 16);
        const double D_sr   = SDr.Dmean_unweighted(*S.eps_tables, g, 24);
        Ds[i]               = D_s;
        Dl[i]               = I_laws / S.H.J[i];
        rs[i]               = D_s / Dl[i] - 1.0;
        rc[i]               = D_sr / D_s - 1.0;
        rb[i]               = I_tab / I_laws - 1.0;
        ra[i]               = I_laws / S.H.I[i] - 1.0;
        rt[i]               = I_tab / S.H.I[i] - 1.0;
      });
      double ws = 0.0, wc = 0.0, wb = 0.0, wa = 0.0, wt = 0.0, gs = 0.0, ga = 0.0;
      double wa_core = 0.0, ga_core = 0.0; // A5-numerical: I_laws vs the table where I > 1e-3 of its peak
      for (std::size_t i = 0; i < ng; ++i) {
        w_cert = std::max(w_cert, std::abs(rs[i]) + std::abs(rc[i]));
        if (std::abs(rs[i]) > ws) {
          ws = std::abs(rs[i]);
          gs = S.H.gamma[i];
        }
        if (std::abs(ra[i]) > wa) {
          wa = std::abs(ra[i]);
          ga = S.H.gamma[i];
        }
        if (S.H.I[i] > 1.0e-3 * S.H.I_peak && std::abs(ra[i]) > wa_core) {
          wa_core = std::abs(ra[i]);
          ga_core = S.H.gamma[i];
        }
        wc = std::max(wc, std::abs(rc[i]));
        wb = std::max(wb, std::abs(rb[i]));
        wt = std::max(wt, std::abs(rt[i]));
      }
      n_nodes += ng;
      diag("%s: %zu nodes in %.1f s; <D>_sampled/<D>_laws - 1 max %.3e at Gamma %.5f (convergence %.1e); q_k part "
           "alone (round-1 budget) %.3e; I_laws/I_exact - 1 max %.3e at Gamma %.5f; I_tables/I_exact - 1 max %.3e; "
           "samples (Gamma, <D>_sampled, <D>_laws): %.4f %.8f %.8f | %.4f %.8f %.8f | %.4f %.6f %.6f",
           S.cb.name.c_str(), ng, seconds_since(t0), ws, gs, wc, wb, wa, ga, wt, S.H.gamma[0], Ds[0], Dl[0],
           S.H.gamma[ng / 2], Ds[ng / 2], Dl[ng / 2], S.H.gamma.back(), Ds.back(), Dl.back());
      if (ws > w_s) {
        w_s = ws;
        n_s = S.cb.name;
        g_s = gs;
      }
      if (wb > w_qonly) {
        w_qonly = wb;
        n_qonly = S.cb.name;
      }
      if (wa > w_a5b) {
        w_a5b = wa;
        n_a5b = S.cb.name;
      }
      if (wa_core > w_a5num) {
        w_a5num = wa_core;
        n_a5num = S.cb.name;
        g_a5num = ga_core;
      }
      w_conv         = std::max(w_conv, wc);
      w_tab_vs_exact = std::max(w_tab_vs_exact, wt);
    }
    check_abs("budget: <D> of the sampled chain (photon table, eps' interpolant, tables' q_k) vs <D> of the laws, "
              "every Gamma_p node of the five rate tables (" + std::to_string(n_nodes) + " nodes; worst " + n_s +
                " Gamma " + std::to_string(g_s) + ")",
              w_s, 1.0e-3);
    check_abs("budget: the same with the quadrature's convergence estimate added at every node, |dev| + |change "
              "under refinement| (q_k sub-nodes 16 -> 32, Gauss-Legendre orders 16 -> 24)",
              w_cert, 1.0e-3, "max (|dev| + |change|)");
    diag("budget: sampled-chain quadrature, max |rel. change| under refinement %.2e over all nodes", w_conv);
    if (born_mode()) {
      // contract 1.5 A5-numerical (T3b): the direct-boost chain's I (the laws' q_k) against the Born table
      check_abs("A5-numerical: direct-boost I (laws' q_k) vs the Born table I, nodes with I > 1e-3 I_peak of the five "
                "tables (worst " + n_a5num + " Gamma " + std::to_string(g_a5num) + ")",
                w_a5num, 1.0e-3);
      diag("A5-numerical: over all nodes (no mask) max %.3e (bath %s)", w_a5b, n_a5b.c_str());
    } else {
      check_abs("A5b: direct-boost loss table I (laws' q_k) vs I_exact of step 1, every node of the five tables", w_a5b,
                5.0e-3);
    }
    diag("budget: the q_k part alone (round 1's check: tables' q_k in the analytic chain vs the laws) max %.3e (bath "
         "%s); A5b worst bath %s; I_tables vs I_exact worst %.3e",
         w_qonly, n_qonly.c_str(), n_a5b.c_str(), w_tab_vs_exact);
  }

  // ---------------------------------------------------------------------- //
  // round 2 (finding 3): the photon-energy table's mean against the marginal
  // ---------------------------------------------------------------------- //
  void test_eps_marginal(const std::vector<BathSetup>& setups, const rf::SamplerCumulatives& cum) {
    std::printf("\n== photon-energy tables: <eps> of the sampled table vs the marginal by quadrature, every node ==\n");
    double      worst = 0.0, worst_conv = 0.0, worst2 = 0.0;
    std::string wname;
    double      wgamma = 0.0;
    bool        wweighted = false;
    std::size_t n_checked = 0;
    for (const auto& S : setups) {
      if (S.eps_tables->mono) {
        continue;
      }
      const auto          t0 = std::chrono::steady_clock::now();
      const std::size_t   ng = S.H.size();
      std::vector<double> bias(2 * ng), conv(2 * ng), bias2(2 * ng);
      rf::parallel_for_index(2 * ng, [&](std::size_t n) {
        const std::size_t i  = n / 2;
        const bool        wt = (n % 2) == 1;
        const double      g  = S.H.gamma[i];
        const auto        m  = rf::photon_marginal_moments(*S.bath, cum, g, wt, 16384, 16);
        const auto        mr = rf::photon_marginal_moments(*S.bath, cum, g, wt, 32768, 16);
        bias[n]              = S.eps_tables->table_moment(i, wt, 1) / m.mean - 1.0;
        bias2[n]             = S.eps_tables->table_moment(i, wt, 2) / m.mean2 - 1.0;
        conv[n]              = mr.mean / m.mean - 1.0;
      });
      double wb = 0.0, wc = 0.0, wb2 = 0.0, gb = 0.0;
      bool   wbw = false;
      for (std::size_t n = 0; n < 2 * ng; ++n) {
        if (std::abs(bias[n]) > wb) {
          wb  = std::abs(bias[n]);
          gb  = S.H.gamma[n / 2];
          wbw = (n % 2) == 1;
        }
        wc  = std::max(wc, std::abs(conv[n]));
        wb2 = std::max(wb2, std::abs(bias2[n]));
      }
      n_checked += 2 * ng;
      diag("%s: %zu nodes x 2 chains in %.1f s; max |<eps>_table/<eps>_marginal - 1| = %.3e at Gamma %.9f (%s chain); "
           "<eps^2> %.3e; quadrature refinement (16384 -> 32768 panels) %.1e; first nodes (unweighted): %+.2e %+.2e "
           "%+.2e; last %+.2e",
           S.cb.name.c_str(), ng, seconds_since(t0), wb, gb, wbw ? "weighted" : "unweighted", wb2, wc, bias[0], bias[2],
           bias[4], bias[2 * ng - 2]);
      if (wb > worst) {
        worst     = wb;
        wname     = S.cb.name;
        wgamma    = gb;
        wweighted = wbw;
      }
      worst_conv = std::max(worst_conv, wc);
      worst2     = std::max(worst2, wb2);
      if (S.cb.name == "powerlaw") {
        // the referee's node (finding 3): Gamma = 1.048099570984, the paper marginal by quadrature 1.4946719776559543
        for (std::size_t i = 0; i < ng; ++i) {
          if (std::abs(S.H.gamma[i] / 1.048099570984 - 1.0) < 1.0e-11) {
            const double tm = S.eps_tables->table_moment(i, false, 1);
            impl_abs("photon table: <eps> at the referee's powerlaw node Gamma = 1.048099570984 vs the referee's "
                     "quadrature of the paper marginal 1.4946719776559543 (round 1: 1.494613576097, -3.9e-5)",
                     tm / 1.4946719776559543 - 1.0, 1.0e-6, "rel. dev.");
            diag("photon table at Gamma = %.12f: <eps> table %.13f, own quadrature %.13f",
                 S.H.gamma[i], tm, rf::photon_marginal_moments(*S.bath, cum, S.H.gamma[i], false).mean);
          }
        }
      }
    }
    check_abs("photon table (round-2 brief stage 2 (i)): |<eps>_table / <eps>_marginal - 1| at every node of the "
              "broad baths, both chains (" + std::to_string(n_checked) + " node-chains; worst " + wname + " Gamma " +
                std::to_string(wgamma) + (wweighted ? " weighted)" : " unweighted)"),
              worst, 1.0e-6);
    impl_abs("photon table: the marginal's quadrature under refinement (16384 -> 32768 panels), every node", worst_conv,
             1.0e-9, "max |rel. change|");
    diag("photon table: max |<eps^2>_table/<eps^2>_marginal - 1| = %.3e", worst2);
  }

  // ---------------------------------------------------------------------- //
  // A7: Monte Carlo at every node
  // ---------------------------------------------------------------------- //
  struct NodeMC {
    double mean { 0.0 }, se { 0.0 }, Dmin { 1.0e300 }, Dmax { 0.0 };
  };

  NodeMC mc_unweighted(const rf::HostSampler& S, double gamma, std::size_t N, std::uint64_t seed) {
    rf::SplitMix64 rng(seed);
    double         s1 = 0.0, s2 = 0.0;
    NodeMC         r;
    for (std::size_t n = 0; n < N; ++n) {
      const auto e = S.draw(gamma, false, rng);
      s1 += e.D;
      s2 += e.D * e.D;
      r.Dmin = std::min(r.Dmin, e.D);
      r.Dmax = std::max(r.Dmax, e.D);
    }
    r.mean = s1 / N;
    r.se   = std::sqrt(std::max(0.0, s2 / N - r.mean * r.mean) / N);
    return r;
  }

  void test_A7(const std::vector<BathSetup>& setups, const rf::SamplerCumulatives& cum, const rf::QkFineGrid& qk_tab) {
    std::printf("\n== A7: <D>_MC at every rate-table node, N = 1e6 ==\n");
    const std::size_t       N = 1000000;
    const rf::SampledChainD SD(cum, [&](double k) { return qk_tab(k); }, 8);
    double worst_rel = 0.0, worst_se_exact = 0.0, worst_se_chain = 0.0;
    std::string wname;
    double      wgamma = 0.0;
    std::size_t total_nodes = 0, total_fail = 0;
    // Born mode (step 6, frozen fixtures of 2026-10-30): the same draws against the frozen bhref 0.3.0
    // tables' Dmean column (= I_Born/J of the release) at the nodes, which coincide with this table's
    double      worst_se_frozen = 0.0, worst_node_shift = 0.0;
    std::size_t total_fail_frozen = 0;
    bool        frozen_ok = true;
    for (std::size_t b = 0; b < setups.size(); ++b) {
      const auto&         S = setups[b];
      std::vector<NodeMC> R(S.H.size());
      std::vector<double> Dchain(S.H.size());
      const auto          t0 = std::chrono::steady_clock::now();
      rf::parallel_for_index(S.H.size(), [&](std::size_t i) {
        R[i]      = mc_unweighted(*S.sampler, S.H.gamma[i], N, 1000003ULL * b + i);
        Dchain[i] = SD.Dmean_unweighted(*S.eps_tables, S.H.gamma[i], 8); // the sampled density's mean
      });
      double wr = 0.0, wse = 0.0, wsc = 0.0, g_wr = 0.0;
      int    n_fail_rel = 0;
      std::vector<double> Dfrozen;
      if (born_mode()) {
        const auto FT = read_table_file(std::string(BH_TEST_DATA_DIR) + "/bhref-0.3.0/" + S.cb.name + ".txt");
        if (FT.gamma.size() != S.H.size() || FT.qk_source != "born-3D2000") {
          frozen_ok = false;
          diag("A7 %s: frozen 0.3.0 table has %zu nodes vs %zu here (qk_source %s): frozen comparison skipped", S.cb.name.c_str(),
               FT.gamma.size(), S.H.size(), FT.qk_source.c_str());
        } else {
          Dfrozen = FT.Dmean;
          for (std::size_t i = 0; i < S.H.size(); ++i) {
            worst_node_shift = std::max(worst_node_shift, std::abs(FT.gamma[i] / S.H.gamma[i] - 1.0));
          }
        }
      }
      for (std::size_t i = 0; i < S.H.size(); ++i) {
        if (!Dfrozen.empty()) {
          const double nsf = std::abs(R[i].mean - Dfrozen[i]) / R[i].se;
          worst_se_frozen  = std::max(worst_se_frozen, nsf);
          if (!(nsf <= 4.0)) {
            ++total_fail_frozen;
          }
        }
        const double rel  = std::abs(R[i].mean / S.H.Dmean[i] - 1.0);
        const double nse  = std::abs(R[i].mean - S.H.Dmean[i]) / R[i].se;
        const double nsc  = std::abs(R[i].mean - Dchain[i]) / R[i].se;
        // the registered criterion: max(1%, 4 s.e.) against the fit tables (step 4); against the Born
        // tables (contract 1.5, decision 2026-10-27 (4)): 4 s.e. with no floor
        const bool   pass = born_mode() ? (nse <= 4.0) : (rel <= 1.0e-2 || nse <= 4.0);
        if (!pass) {
          ++n_fail_rel;
        }
        if (rel > wr) {
          wr   = rel;
          g_wr = S.H.gamma[i];
        }
        wse = std::max(wse, nse);
        wsc = std::max(wsc, nsc);
      }
      total_nodes += S.H.size();
      total_fail += static_cast<std::size_t>(n_fail_rel);
      diag("A7 %s: %zu nodes x 1e6 draws in %.0f s; max |<D>_MC/(I/J) - 1| = %.3e at Gamma %.5f; max |dev|/s.e. vs I/J "
           "%.1f, vs the sampled chain's quadrature %.1f; nodes failing max(1%%, 4 s.e.): %d; samples (Gamma, <D>_MC, I/J, "
           "<D>_chain): %.4f %.5f %.5f %.5f | %.4f %.5f %.5f %.5f | %.4f %.4f %.4f %.4f",
           S.cb.name.c_str(), S.H.size(), seconds_since(t0), wr, g_wr, wse, wsc, n_fail_rel, S.H.gamma[1], R[1].mean,
           S.H.Dmean[1], Dchain[1], S.H.gamma[S.H.size() / 2], R[S.H.size() / 2].mean, S.H.Dmean[S.H.size() / 2],
           Dchain[S.H.size() / 2], S.H.gamma.back(), R.back().mean, S.H.Dmean.back(), Dchain.back());
      if (wr > worst_rel) {
        worst_rel = wr;
        wname     = S.cb.name;
        wgamma    = g_wr;
      }
      worst_se_exact = std::max(worst_se_exact, wse);
      worst_se_chain = std::max(worst_se_chain, wsc);
    }
    // the registered criterion: every node within max(1%, 4 s.e.) of I/J (round 2: the per-node count is the
    // verdict; round 1 printed it but the verdict line could not fail)
    if (born_mode()) {
      check_true("A7 [re-registered, contract 1.5]: |<D>_MC - I_Born/J| <= 4 s.e. at every rate-table node of the five "
                 "baths (N = 1e6), no floor",
                 total_fail == 0, std::to_string(total_fail) + " of " + std::to_string(total_nodes) + " nodes fail");
      check_abs("A7 [re-registered]: max |<D>_MC - I_Born/J| / s.e. over the " + std::to_string(total_nodes) + " nodes",
                worst_se_exact, 4.0, "max |dev| / s.e.");
      diag("A7 [re-registered]: max |<D>_MC/(I_Born/J) - 1| over the nodes = %.3e (bath %s, Gamma %.5f)", worst_rel,
           wname.c_str(), wgamma);
      check_true("A7 [frozen bhref 0.3.0 reference]: |<D>_MC - Dmean_frozen| <= 4 s.e. at every node of the five frozen tables "
                 "(nodes identical to this table's to 1e-9)",
                 frozen_ok && worst_node_shift <= 1.0e-9 && total_fail_frozen == 0,
                 std::to_string(total_fail_frozen) + " of " + std::to_string(total_nodes) + " nodes fail; max node shift " +
                   std::to_string(worst_node_shift) + (frozen_ok ? "" : "; a frozen table did not match"));
      check_abs("A7 [frozen bhref 0.3.0 reference]: max |<D>_MC - Dmean_frozen| / s.e. over the nodes", worst_se_frozen, 4.0,
                "max |dev| / s.e.");
    } else {
      check_true("A7: |<D>_MC - I/J| <= max(1%, 4 s.e.) at every rate-table node of the five baths (N = 1e6)",
                 total_fail == 0, std::to_string(total_fail) + " of " + std::to_string(total_nodes) + " nodes fail");
      check_abs("A7: max |<D>_MC/(I/J) - 1| over the " + std::to_string(total_nodes) + " nodes (bath " + wname +
                  ", Gamma " + std::to_string(wgamma) + ")",
                worst_rel, 1.0e-2);
    }
    impl_abs("A7: max |<D>_MC - <D>_sampled| / s.e. over all nodes (the quadrature of the density the chain samples, "
             "SampledChainD)",
             worst_se_chain, 5.0, "max |dev| / s.e.");
    if (born_mode()) {
      diag("A7: max |<D>_MC - I_Born/J| / s.e. over all nodes = %.1f (the table's own Born recoil: a sampling statistic, "
           "the registered row above; see A5-numerical)",
           worst_se_exact);
    } else {
      diag("A7: max |<D>_MC - I/J| / s.e. over all nodes = %.1f (the eq. 3.7 fit of step 1 vs the Born laws, not a "
           "sampling error; see A5b)",
           worst_se_exact);
    }
  }

  void test_A7_weighted(const std::vector<BathSetup>& setups, const rf::SamplerCumulatives& cum,
                        const rf::QkFineGrid& qk_tab) {
    const rf::SampledChainD SD(cum, [&](double k) { return qk_tab(k); }, 8);
    std::printf("\n== A7 [smoke test, selected points]: the weighted chain, E_p[1/D] <D> = 1 at Gamma_p in {1.05 Gamma_th, 3, 10} "
                "(the all-node weighted test is the a7wall group) ==\n");
    bool all_exact_N = true;
    double worst_exact = 0.0, worst_chain = 0.0;
    for (std::size_t b = 0; b < setups.size(); ++b) {
      const auto&         S = setups[b];
      const double        gth = S.H.gamma_th;
      std::vector<double> gammas;
      for (const double g : { 1.05 * gth, 3.0, 10.0 }) {
        if (g >= S.H.gamma_min && g <= S.H.gamma_max) {
          gammas.push_back(g);
        } else {
          diag("A7 weighted %s: Gamma_p = %g outside the table [%g, %g], skipped", S.cb.name.c_str(), g, S.H.gamma_min,
               S.H.gamma_max);
        }
      }
      struct Res {
        double Einv, se, eff, Dexact, Dchain;
        std::size_t n_acc;
        double      minD;
      };
      std::vector<Res> res(gammas.size());
      rf::parallel_for_index(gammas.size(), [&](std::size_t j) {
        const double   g = gammas[j];
        rf::SplitMix64 rng(7000000ULL + 10 * b + j);
        const std::size_t N_target = 1000000, attempts_cap = 60000000;
        double            s1 = 0.0, s2 = 0.0, minD = 1.0e300;
        std::size_t       n = 0, att = 0;
        while (n < N_target && att < attempts_cap) {
          const auto e = S.sampler->draw(g, true, rng);
          att += static_cast<std::size_t>(e.attempts);
          const double y = 1.0 / e.D;
          s1 += y;
          s2 += y * y;
          minD = std::min(minD, e.D);
          ++n;
        }
        const double m  = s1 / n;
        const double se = std::sqrt(std::max(0.0, s2 / n - m * m) / n);
        const auto   R  = S.rates->at_gamma(g);
        res[j]          = Res { m, se, static_cast<double>(n) / att, R.Dmean,
                       SD.Dmean_unweighted(*S.eps_tables, g, 8), n, minD };
      });
      for (std::size_t j = 0; j < gammas.size(); ++j) {
        const auto&  r  = res[j];
        const double pe = r.Einv * r.Dexact, pc = r.Einv * r.Dchain;
        const double ne = std::abs(pe - 1.0) / (r.se * r.Dexact), nc = std::abs(pc - 1.0) / (r.se * r.Dchain);
        worst_exact     = std::max(worst_exact, ne);
        worst_chain     = std::max(worst_chain, nc);
        all_exact_N     = all_exact_N && r.n_acc == 1000000;
        diag("A7 weighted [smoke] %s Gamma %.5f: N_acc %zu, E_p[1/D] (I/J) = %.5f (%.1f s.e.), E_p[1/D] <D>_sampled = %.5f (%.1f "
             "s.e.), efficiency %.4f, 1/min D = %.4f",
             S.cb.name.c_str(), gammas[j], r.n_acc, pe, ne, pc, nc, r.eff, 1.0 / r.minD);
      }
    }
    check_true("A7 [smoke test, selected points]: exactly N = 1e6 accepted weighted draws at every selected point (the "
               "attempts cap never reduced N)",
               all_exact_N, all_exact_N ? "N_acc = 1e6 at every point" : "a point stopped at the attempts cap with N_acc < 1e6");
    check_abs("A7 [smoke test, selected points; the all-node weighted test is the a7wall group]: E_p[1/D] (I/J) = 1 within 4 "
              "s.e., weighted chain, Gamma_p in {1.05 Gamma_th, 3, 10}, five baths",
              worst_exact, 4.0, "max |dev| / s.e.");
    impl_abs("A7 [smoke]: E_p[1/D] <D>_sampled = 1 within 4 s.e. (<D> of the density the unweighted chain samples)", worst_chain,
             4.0, "max |dev| / s.e.");
  }

  /**
   * @brief A7, the weighted chain at every rate-table node (round 2 of step 6, Codex 1 F2):
   *        exactly N = 1e6 accepted weighted draws at each of the 2169 nodes of the five baths,
   *        z = |mean(1/D) - 1/(I/J)| / s.e.(mean(1/D)) <= 4, no floor; in Born mode also against the
   *        frozen bhref 0.3.0 tables' Dmean. The acceptance efficiency falls to ~2.5e-3 at the top
   *        nodes (4.1e8 proposals for 1e6 accepted, Codex 1's campaign), so the whole group is
   *        ~2e10 proposals: the long group, its own ctest entry. Resource limit: a node that reaches
   *        `attempts_cap` proposals (4e9, ten times the expected maximum) ABORTS explicitly (the
   *        node named, the exact-N assertion fails, the group fails); N is never reduced silently.
   *        Seeds 7 500 000 + 1000 b + i (distinct from the smoke test's and from the unweighted A7's).
   */
  void test_A7_weighted_all(const std::vector<BathSetup>& setups) {
    std::printf("\n== A7 [all nodes, weighted chain]: E_p[1/D] (I/J) = 1 within 4 s.e. at every rate-table node, exactly N = "
                "1e6 accepted draws per node ==\n");
    const std::size_t N_target = 1000000, attempts_cap = 4000000000ULL;
    struct Res {
      double      Einv { 0.0 }, se { 0.0 }, minD { 1.0e300 };
      std::size_t n_acc { 0 }, att { 0 };
      bool        aborted { false };
    };
    double      worst_exact = 0.0, worst_frozen = 0.0, worst_node_shift = 0.0;
    std::size_t total_nodes = 0, total_fail = 0, total_fail_frozen = 0, total_att = 0, total_aborted = 0, max_att = 0;
    std::string max_att_bath, worst_bath;
    double      max_att_gamma = 0.0, worst_gamma = 0.0;
    bool        frozen_ok = born_mode();
    for (std::size_t b = 0; b < setups.size(); ++b) {
      const auto&      S  = setups[b];
      const auto       t0 = std::chrono::steady_clock::now();
      std::vector<Res> res(S.H.size());
      rf::parallel_for_index(S.H.size(), [&](std::size_t i) {
        const double   g = S.H.gamma[i];
        rf::SplitMix64 rng(7500000ULL + 1000ULL * b + i);
        double         s1 = 0.0, s2 = 0.0;
        Res&           r = res[i];
        while (r.n_acc < N_target) {
          if (r.att >= attempts_cap) {
            r.aborted = true; // explicit abort: the resource limit, never a smaller N
            break;
          }
          const auto e = S.sampler->draw(g, true, rng);
          r.att += static_cast<std::size_t>(e.attempts);
          const double y = 1.0 / e.D;
          s1 += y;
          s2 += y * y;
          r.minD = std::min(r.minD, e.D);
          ++r.n_acc;
        }
        if (r.n_acc > 0) {
          r.Einv = s1 / static_cast<double>(r.n_acc);
          r.se   = std::sqrt(std::max(0.0, s2 / static_cast<double>(r.n_acc) - r.Einv * r.Einv) /
                           static_cast<double>(r.n_acc));
        }
      });
      std::vector<double> Dfrozen;
      if (born_mode()) {
        const auto FT = read_table_file(std::string(BH_TEST_DATA_DIR) + "/bhref-0.3.0/" + S.cb.name + ".txt");
        if (FT.gamma.size() != S.H.size() || FT.qk_source != "born-3D2000") {
          frozen_ok = false;
          diag("A7 all-node weighted %s: frozen 0.3.0 table has %zu nodes vs %zu here (qk_source %s): frozen comparison "
               "skipped",
               S.cb.name.c_str(), FT.gamma.size(), S.H.size(), FT.qk_source.c_str());
        } else {
          Dfrozen = FT.Dmean;
          for (std::size_t i = 0; i < S.H.size(); ++i) {
            worst_node_shift = std::max(worst_node_shift, std::abs(FT.gamma[i] / S.H.gamma[i] - 1.0));
          }
        }
      }
      double      wz = 0.0, wzf = 0.0, g_wz = 0.0, min_eff = 1.0, g_min_eff = 0.0;
      std::size_t n_fail = 0, n_fail_frozen = 0, bath_att = 0, n_ab = 0;
      for (std::size_t i = 0; i < S.H.size(); ++i) {
        const Res& r = res[i];
        bath_att += r.att;
        if (r.att > max_att) {
          max_att       = r.att;
          max_att_bath  = S.cb.name;
          max_att_gamma = S.H.gamma[i];
        }
        if (r.aborted) {
          ++n_ab;
          std::printf("[ABORT] A7 all-node weighted %s node %zu Gamma %.10g: the attempts cap %zu reached with N_acc = %zu < "
                      "%zu; the node has no verdict and the exact-N assertion fails\n",
                      S.cb.name.c_str(), i, S.H.gamma[i], attempts_cap, r.n_acc, N_target);
          continue;
        }
        const double eff = static_cast<double>(r.n_acc) / static_cast<double>(r.att);
        if (eff < min_eff) {
          min_eff   = eff;
          g_min_eff = S.H.gamma[i];
        }
        const double z = std::abs(r.Einv - 1.0 / S.H.Dmean[i]) / r.se;
        if (!(z <= 4.0)) {
          ++n_fail;
          std::printf("[FAIL-NODE] A7 all-node weighted %s node %zu Gamma %.10g: z = %.3f (mean(1/D) %.8e, 1/(I/J) %.8e, s.e. "
                      "%.2e)\n",
                      S.cb.name.c_str(), i, S.H.gamma[i], z, r.Einv, 1.0 / S.H.Dmean[i], r.se);
        }
        if (z > wz) {
          wz   = z;
          g_wz = S.H.gamma[i];
        }
        if (!Dfrozen.empty()) {
          const double zf = std::abs(r.Einv - 1.0 / Dfrozen[i]) / r.se;
          if (!(zf <= 4.0)) {
            ++n_fail_frozen;
            std::printf("[FAIL-NODE] A7 all-node weighted %s node %zu Gamma %.10g vs the frozen Dmean: z = %.3f\n",
                        S.cb.name.c_str(), i, S.H.gamma[i], zf);
          }
          wzf = std::max(wzf, zf);
        }
      }
      if (wz > worst_exact) {
        worst_exact = wz;
        worst_bath  = S.cb.name;
        worst_gamma = g_wz;
      }
      worst_frozen = std::max(worst_frozen, wzf);
      total_nodes += S.H.size();
      total_fail += n_fail;
      total_fail_frozen += n_fail_frozen;
      total_att += bath_att;
      total_aborted += n_ab;
      diag("A7 all-node weighted %s: %zu nodes x 1e6 accepted in %.0f s, %zu proposals (max %zu at one node, min efficiency "
           "%.3e at Gamma %.5f); max z vs I/J %.3f at Gamma %.5f%s; nodes failing 4 s.e.: %zu; aborted at the cap: %zu; "
           "first/mid/last node (Gamma, E_p[1/D] <D>): %.4f %.6f | %.4f %.6f | %.4f %.6f",
           S.cb.name.c_str(), S.H.size(), seconds_since(t0), bath_att, max_att, min_eff, g_min_eff, wz, g_wz,
           Dfrozen.empty() ? "" : (", vs the frozen Dmean " + std::to_string(wzf)).c_str(), n_fail, n_ab, S.H.gamma[0],
           res[0].Einv * S.H.Dmean[0], S.H.gamma[S.H.size() / 2], res[S.H.size() / 2].Einv * S.H.Dmean[S.H.size() / 2],
           S.H.gamma.back(), res.back().Einv * S.H.Dmean.back());
    }
    diag("A7 all-node weighted: %zu nodes, %zu proposals in total, the most at one node %zu (%s, Gamma %.5f)", total_nodes,
         total_att, max_att, max_att_bath.c_str(), max_att_gamma);
    check_true("A7 [all nodes, weighted]: exactly N = 1e6 accepted weighted draws at every one of the " +
                 std::to_string(total_nodes) + " nodes (an explicit abort at the attempts cap fails this row; N is never reduced)",
               total_aborted == 0, std::to_string(total_aborted) + " nodes aborted at the cap");
    if (born_mode()) {
      check_true("A7 [all nodes, weighted; re-registered, contract 1.5]: |mean(1/D) - 1/(I_Born/J)| <= 4 s.e. at every rate-table "
                 "node of the five baths (N = 1e6 accepted), no floor",
                 total_fail == 0 && total_aborted == 0,
                 std::to_string(total_fail) + " of " + std::to_string(total_nodes) + " nodes fail");
      check_abs("A7 [all nodes, weighted; re-registered]: max |mean(1/D) - 1/(I_Born/J)| / s.e. over the nodes (bath " +
                  worst_bath + ", Gamma " + std::to_string(worst_gamma) + ")",
                worst_exact, 4.0, "max |dev| / s.e.");
      check_true("A7 [all nodes, weighted; frozen bhref 0.3.0 reference]: |mean(1/D) - 1/Dmean_frozen| <= 4 s.e. at every node "
                 "of the five frozen tables (nodes identical to this table's, max relative shift " +
                   std::to_string(worst_node_shift) + ")",
                 frozen_ok && total_fail_frozen == 0 && total_aborted == 0,
                 frozen_ok ? (std::to_string(total_fail_frozen) + " of " + std::to_string(total_nodes) + " nodes fail")
                           : "frozen tables not comparable");
      check_abs("A7 [all nodes, weighted; frozen bhref 0.3.0 reference]: max |mean(1/D) - 1/Dmean_frozen| / s.e. over the nodes",
                worst_frozen, 4.0, "max |dev| / s.e.");
    } else {
      check_true("A7 [all nodes, weighted]: |mean(1/D) - 1/(I/J)| <= 4 s.e. at every rate-table node of the five baths (N = 1e6 "
                 "accepted), the step-1 fit tables (the fit-vs-Born offset is expected to fail this at the deep nodes; history)",
                 total_fail == 0 && total_aborted == 0,
                 std::to_string(total_fail) + " of " + std::to_string(total_nodes) + " nodes fail");
      check_abs("A7 [all nodes, weighted]: max |mean(1/D) - 1/(I/J)| / s.e. over the nodes (bath " + worst_bath + ", Gamma " +
                  std::to_string(worst_gamma) + ")",
                worst_exact, 4.0, "max |dev| / s.e.");
    }
  }

  // ---------------------------------------------------------------------- //
  // A9 and the kinematic checks
  // ---------------------------------------------------------------------- //
  void test_A9(const std::vector<BathSetup>& setups) {
    std::printf("\n== A9: D >= 2 - eps_max over 1e7 draws per bath ==\n");
    std::size_t total_viol = 0, total_oob = 0;
    for (std::size_t b = 0; b < setups.size(); ++b) {
      const auto&       S     = setups[b];
      const double      bound = 2.0 - S.bath->eps_max();
      const int         ng    = 10;
      std::vector<std::size_t> viol(ng, 0), oob(ng, 0);
      std::vector<double>      minD(ng, 1.0e300);
      std::vector<double>      gam(ng);
      for (int j = 0; j < ng; ++j) {
        const double y = S.H.y_min + (S.H.y_max - S.H.y_min) * static_cast<double>(j) / (ng - 1);
        gam[j]         = S.H.gamma_th + std::pow(10.0, y);
      }
      const auto t0 = std::chrono::steady_clock::now();
      rf::parallel_for_index(ng, [&](std::size_t j) {
        rf::SplitMix64 rng(9000000ULL + 100 * b + j);
        for (std::size_t n = 0; n < 1000000; ++n) {
          const auto e = S.sampler->draw(gam[j], false, rng);
          if (!(e.D >= bound)) {
            ++viol[j];
          }
          if (!(e.D >= e.Dmin * (1.0 - 1.0e-9) - 1.0e-12) || !(e.D <= e.Dmax * (1.0 + 1.0e-9) + 1.0e-12)) {
            ++oob[j];
          }
          minD[j] = std::min(minD[j], e.D);
        }
      });
      std::size_t v = 0, o = 0;
      double      m = 1.0e300;
      for (int j = 0; j < ng; ++j) {
        v += viol[j];
        o += oob[j];
        m  = std::min(m, minD[j]);
      }
      total_viol += v;
      total_oob  += o;
      diag("A9 %s: 1e7 draws over 10 Gamma in [%.4f, %.4f] in %.0f s: %zu violations of D >= %.3f, %zu outside "
           "[D_min, D_max]; min D = %.6f, 1/min D = %.4f",
           S.cb.name.c_str(), gam.front(), gam.back(), seconds_since(t0), v, bound, o, m, 1.0 / m);
    }
    check_true("A9: every sampled D >= 2 - eps_max (1e7 draws per bath, five baths)", total_viol == 0,
               std::to_string(total_viol) + " violations");
    impl_true("A9: every sampled D inside [D_min, D_max](eps, eps') (relative slack 1e-9)", total_oob == 0,
              std::to_string(total_oob) + " outside");
  }

  void test_kinematics(const BathSetup& S) {
    std::printf("\n== kinematics (impl) ==\n");
    rf::SplitMix64 rng(424242u);
    double         w_shell = 0.0, w_inv = 0.0, w_D = 0.0, w_photon = 0.0;
    const double   g = 3.0, gb = std::sqrt(8.0), beta = gb / g;
    for (int n = 0; n < 100000; ++n) {
      const auto e = S.sampler->draw(g, n % 2 == 0, rng);
      for (const auto* q : { &e.lab_m, &e.lab_p }) {
        const double p2 = q->p.x * q->p.x + q->p.y * q->p.y + q->p.z * q->p.z;
        w_shell         = std::max(w_shell, std::abs(q->E * q->E - p2 - 1.0) / (q->E * q->E));
      }
      // inverse boost along z returns the rest-frame energies
      w_inv = std::max(w_inv, std::abs(g * (e.lab_m.E - beta * e.lab_m.p.z) / e.Em - 1.0));
      w_inv = std::max(w_inv, std::abs(g * (e.lab_p.E - beta * e.lab_p.p.z) / e.Ep - 1.0));
      // the photon: massless, energy eps, rest-frame energy eps'
      const double kp2 = e.lab_photon.p.x * e.lab_photon.p.x + e.lab_photon.p.y * e.lab_photon.p.y +
                         e.lab_photon.p.z * e.lab_photon.p.z;
      w_photon = std::max(w_photon, std::abs(std::sqrt(kp2) / e.eps - 1.0));
      w_photon = std::max(w_photon, std::abs(g * (e.eps - beta * e.lab_photon.p.z) / e.k - 1.0));
      // D from the lab values equals the rest-frame expression
      w_D = std::max(w_D, std::abs((e.lab_m.E + e.lab_p.E - e.eps) / e.D - 1.0));
    }
    impl_abs("kinematics: lab leptons on shell, |E^2 - p^2 - 1|/E^2, 1e5 draws at Gamma_p = 3 (powerlaw)", w_shell,
             1.0e-9);
    impl_abs("kinematics: inverse boost of the lab leptons returns E'-, E'+", w_inv, 1.0e-9);
    impl_abs("kinematics: the absorbed photon has |k| = eps and rest-frame energy eps' under the inverse boost",
             w_photon, 1.0e-9);
    impl_abs("kinematics: D = E- + E+ - eps as stored", w_D, 1.0e-12);
    // determinism
    rf::SplitMix64 r1(777u), r2(777u);
    bool           same = true;
    for (int n = 0; n < 1000 && same; ++n) {
      const auto a = S.sampler->draw(2.5, true, r1), c = S.sampler->draw(2.5, true, r2);
      same         = (a.D == c.D) && (a.lab_m.p.x == c.lab_m.p.x) && (a.eps == c.eps);
    }
    impl_true("determinism: the same seed gives bitwise-identical events (1000 weighted draws)", same, "identical");
  }

  void report_efficiency(const std::vector<BathSetup>& setups) {
    std::printf("\n== acceptance efficiency of the weighted chain vs Gamma_p (diag) ==\n");
    for (std::size_t b = 0; b < setups.size(); ++b) {
      const auto& S = setups[b];
      std::string line = S.cb.name + ":";
      const int   ng   = 8;
      std::vector<double> eff(ng), gam(ng), cost(ng);
      rf::parallel_for_index(ng, [&](std::size_t j) {
        const double y = S.H.y_min + (S.H.y_max - S.H.y_min) * static_cast<double>(j) / (ng - 1);
        gam[j]         = S.H.gamma_th + std::pow(10.0, y);
        rf::SplitMix64 rng(31337u + j);
        std::size_t    att = 0, n = 0;
        const auto     t0 = std::chrono::steady_clock::now();
        while (att < 1000000 && n < 1000000) {
          att += static_cast<std::size_t>(S.sampler->draw(gam[j], true, rng).attempts);
          ++n;
        }
        eff[j]  = static_cast<double>(n) / att;
        cost[j] = seconds_since(t0) / n * 1.0e6;
      });
      for (int j = 0; j < ng; ++j) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), " Gamma %.4f: %.4f (%.1f us/accepted);", gam[j], eff[j], cost[j]);
        line += buf;
      }
      diag("%s", line.c_str());
    }
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  // optional group selection (ctest registers one entry per group): budget,
  // kinematics, efficiency, a9, a7w (weighted chain, selected points: smoke test), a7wall (weighted chain at every
  // node, exact N; the long group), a7 (unweighted Monte Carlo at every node)
  // a bath name among the arguments restricts the run to that bath
  std::vector<std::string> groups, bath_filter;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a == "mono05" || a == "mono03" || a == "powerlaw" || a == "blackbody" || a == "mono05_mu100") {
      bath_filter.push_back(a);
    } else {
      groups.emplace_back(a);
    }
  }
  auto want = [&groups](const char* g) {
    return groups.empty() || std::find(groups.begin(), groups.end(), std::string(g)) != groups.end();
  };
  try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("bh_chain: rate tables with qk_source = %s%s\n", born_mode() ? "born-3D2000" : "chodorowski-3.7",
                born_mode() ? " (BH_QK_SOURCE=born; contract 1.5 criteria)" : "");
    const auto t_start = std::chrono::steady_clock::now();
    auto       t0      = std::chrono::steady_clock::now();
    auto       T       = std::make_shared<const rf::RestFrameTables>(rf::RestFrameTables::build());
    diag("tables: %zu eps' nodes (%d per decade), n_s = %d, n_tau = %d; %zu + %zu entries (%.1f MB in single "
         "precision); built in %.1f s",
         T->n_k(), T->nodes_per_decade, T->n_s, T->n_tau, T->cdf_E.size(), T->cdf_A.size(),
         (T->cdf_E.size() + T->cdf_A.size()) * 4.0e-6, seconds_since(t0));
    t0       = std::chrono::steady_clock::now();
    auto cum = std::make_shared<const rf::SamplerCumulatives>();
    diag("cumulatives: %zu nodes in t = ln(eps' - 2), h = %.4f; Phi(1000) = %.10e (step-1 Cumulatives: %.10e)",
         cum->size(), cum->h(), cum->Phi(998.0), Cumulatives().Phi(998.0));
    // the laws' q_k on the 449-node grid and the tables' q_k on the fine grid
    const rf::RestFrameQuadrature q;
    rf::QkFineGrid                qk_laws;
    qk_laws.t.clear();
    for (const double k : rf::qk_grid()) {
      qk_laws.t.push_back(std::log(k - 2.0));
    }
    qk_laws.q.assign(qk_laws.t.size(), 0.0);
    t0 = std::chrono::steady_clock::now();
    rf::parallel_for_index(qk_laws.t.size(), [&](std::size_t i) {
      qk_laws.q[i] = rf::moments(2.0 + std::exp(qk_laws.t[i]), q).q_k;
    });
    const rf::QkFineGrid qk_tab = rf::table_qk_fine_grid(*T, 8);
    diag("q_k grids: laws at %zu nodes, tables at %zu fine nodes (8 per interval), %.1f s", qk_laws.t.size(),
         qk_tab.t.size(), seconds_since(t0));
    const rf::QkCumulatives cum_laws([&](double k) { return qk_laws(k); });
    const rf::QkCumulatives cum_tab([&](double k) { return qk_tab(k); });

    std::vector<BathSetup> setups;
    for (const auto& cb : rf::canonical_baths()) {
      if (!bath_filter.empty() &&
          std::find(bath_filter.begin(), bath_filter.end(), cb.name) == bath_filter.end()) {
        continue;
      }
      setups.push_back(make_setup(cb, T, cum));
      const auto& S = setups.back();
      diag("bath %s: %zu Gamma nodes in [%.6f, %.6f], Gamma_th %.6f; eps tables %s", cb.name.c_str(), S.H.size(),
           S.H.gamma_min, S.H.gamma_max, S.H.gamma_th,
           S.eps_tables->mono ? "mono (none)" : (std::to_string(S.eps_tables->n_eps) + " cells per node in ln eps").c_str());
    }

    if (want("epsbias")) {
      test_eps_marginal(setups, *cum);
    }
    if (want("budget")) {
      test_budget_nodes(*T);
      test_budget_gamma(setups, cum_laws, cum_tab, *cum, qk_tab, *T);
    }
    if (want("kinematics")) {
      std::size_t ib = 0;
      for (std::size_t i = 0; i < setups.size(); ++i) {
        if (setups[i].cb.name == "powerlaw") {
          ib = i;
        }
      }
      test_kinematics(setups[ib]);
    }
    if (want("efficiency")) {
      report_efficiency(setups);
    }
    if (want("a9")) {
      test_A9(setups);
    }
    if (want("a7w")) {
      test_A7_weighted(setups, *cum, qk_tab);
    }
    if (want("a7wall")) {
      test_A7_weighted_all(setups);
    }
    if (want("a7")) {
      test_A7(setups, *cum, qk_tab);
    }

    std::printf("\nbh_chain summary: contract items %d passed, %d failed; implementation-chosen checks %d passed, %d "
                "failed; %.0f s\n",
                g_pass_contract, g_fail_contract, g_pass_impl, g_fail_impl, seconds_since(t_start));
    return (g_fail_contract + g_fail_impl) == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_chain: exception: %s\n", e.what());
    return 2;
  }
}
