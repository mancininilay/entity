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
 *           |<D>_MC - I/J| <= max(1%, 4 s.e.); E_p[1/D] <D> = 1 within 4 s.e.
 *           for the weighted chain at Gamma_p in {1.05 Gamma_th, 3, 10} with
 *           <D> = I/J (as registered)
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
 * the weighted checks 7 000 000 + 10 b + j; A9 9 000 000 + 100 b + j.
 */

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_restframe.h"
#include "archetypes/hadronic/bh_sampler_host.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

  BathSetup make_setup(const rf::CanonicalBath& cb, const std::shared_ptr<const rf::RestFrameTables>& T,
                       const std::shared_ptr<const rf::SamplerCumulatives>& cum) {
    BathSetup S;
    S.cb         = cb;
    S.bath       = make_bath(cb.spec);
    S.rates      = std::make_shared<BetheHeitlerRates>(S.bath);
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
    std::string n_s, n_a5b, n_qonly;
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
    check_abs("A5b: direct-boost loss table I (laws' q_k) vs I_exact of step 1, every node of the five tables", w_a5b,
              5.0e-3);
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
      for (std::size_t i = 0; i < S.H.size(); ++i) {
        const double rel  = std::abs(R[i].mean / S.H.Dmean[i] - 1.0);
        const double nse  = std::abs(R[i].mean - S.H.Dmean[i]) / R[i].se;
        const double nsc  = std::abs(R[i].mean - Dchain[i]) / R[i].se;
        const bool   pass = rel <= 1.0e-2 || nse <= 4.0;
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
    check_true("A7: |<D>_MC - I/J| <= max(1%, 4 s.e.) at every rate-table node of the five baths (N = 1e6)",
               total_fail == 0, std::to_string(total_fail) + " of " + std::to_string(total_nodes) + " nodes fail");
    check_abs("A7: max |<D>_MC/(I/J) - 1| over the " + std::to_string(total_nodes) + " nodes (bath " + wname +
                ", Gamma " + std::to_string(wgamma) + ")",
              worst_rel, 1.0e-2);
    impl_abs("A7: max |<D>_MC - <D>_sampled| / s.e. over all nodes (the quadrature of the density the chain samples, "
             "SampledChainD)",
             worst_se_chain, 5.0, "max |dev| / s.e.");
    diag("A7: max |<D>_MC - I/J| / s.e. over all nodes = %.1f (the eq. 3.7 fit of step 1 vs the Born laws, not a "
         "sampling error; see A5b)",
         worst_se_exact);
  }

  void test_A7_weighted(const std::vector<BathSetup>& setups, const rf::SamplerCumulatives& cum,
                        const rf::QkFineGrid& qk_tab) {
    const rf::SampledChainD SD(cum, [&](double k) { return qk_tab(k); }, 8);
    std::printf("\n== A7: the weighted chain, E_p[1/D] <D> = 1 at Gamma_p in {1.05 Gamma_th, 3, 10} ==\n");
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
        diag("A7 weighted %s Gamma %.5f: N_acc %zu, E_p[1/D] (I/J) = %.5f (%.1f s.e.), E_p[1/D] <D>_sampled = %.5f (%.1f "
             "s.e.), efficiency %.4f, 1/min D = %.4f",
             S.cb.name.c_str(), gammas[j], r.n_acc, pe, ne, pc, nc, r.eff, 1.0 / r.minD);
      }
    }
    check_abs("A7: E_p[1/D] (I/J) = 1 within 4 s.e., weighted chain, Gamma_p in {1.05 Gamma_th, 3, 10}, five baths",
              worst_exact, 4.0, "max |dev| / s.e.");
    impl_abs("A7: E_p[1/D] <D>_sampled = 1 within 4 s.e. (<D> of the density the unweighted chain samples)", worst_chain, 4.0,
             "max |dev| / s.e.");
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
  // kinematics, efficiency, a9, a7w (weighted chain), a7 (Monte Carlo at every node)
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
