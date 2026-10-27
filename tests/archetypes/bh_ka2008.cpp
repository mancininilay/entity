/**
 * @file tests/archetypes/bh_ka2008.cpp
 * @brief Contract A8 of the rest-frame sampler (P2 step 4; round 2, finding 2):
 *        the lab electron spectrum against the Kelner and Aharonian 2008 CDF
 *        (eqs. (58) to (64), the paper's head-on approximation) as transcribed
 *        by the reference (`bhref-0.2.0/ka2008_<bath>_G<Gamma>.txt`).
 *
 * Contract item (A8, approved reading): rest-frame events of the sampler
 * (eps', E'-, the angle theta'_gamma of the e- to the photon direction) mapped
 * head-on, E_lab = Gamma (E' - p' cos theta'_gamma), N = 1e6 each, KS <= 0.02
 * against the paper CDF at Gamma_p = 100 and 1000 for mono05 and at
 * Gamma_p = 1000 for the supplementary soft bath mono005 (eps = 0.05). The KS
 * statistic is the largest |F_MC - F_ref| over the 201 edges on which the
 * reference tabulates its CDF (a lower bound of the sup over all energies; the
 * difference is below the largest bin probability, which is printed). The
 * head-on map can give E_lab < 1; that mass sits below the first edge E = 1
 * in both CDFs.
 *
 * Diagnostics, labelled:
 *   - the original reading of round 1 (exact-chain events, exact photon
 *     direction and exact boost, against the paper CDF): it fails (KS 0.073
 *     measured by the referee) and is kept as a failed reading with its
 *     mechanism: the head-on map drops the photon's rest-frame direction, and
 *     the two maps differ on average by eps <p' cos theta'_gamma / eps'>,
 *     which tends to eps/2 when eps' >> 1 (printed: measured mean difference,
 *     the per-event prediction and eps/2);
 *   - applicability: the static-source law holds for eps' << mu_r,
 *     equivalently eps Gamma_p << mu_r (round 2 correction; round 1 wrote
 *     eps' Gamma_p << mu_r); eps'_hi/mu_r and eps Gamma_p/mu_r are printed for
 *     each case; mono05 at Gamma_p = 1000 (eps'_hi/mu_r = 0.54) is outside
 *     the small-eps' regime, a formal mathematical comparison only.
 *
 * Implementation-chosen checks:
 *   - the head-on CDF of this code's laws by deterministic quadrature
 *     (bh_exact_chain_cdf.h, LabMap::HeadOn) against the reference's CDF at its
 *     edges, max |dF| <= 1e-3 (two independent transcriptions, ours of MOK
 *     3D-2000 and the reference's of KA2008 eqs. 58-64);
 *   - the exact-chain self-check (round 1's comparison, renamed): exact-chain
 *     events against the exact-chain quadrature at mono05 Gamma_p = 100 and
 *     1000, KS <= the one-sample 99% critical value 1.628/sqrt(N) plus the
 *     quadrature's change under refinement; its convergence <= 1e-3; the mean
 *     lab energy within 4 s.e.; the exact-chain fixtures (when present)
 *     against the live quadrature, 1e-10;
 *   - each reference file's data block against its data_sha256.
 *
 * Seeds: SplitMix64(8080 + j) for case j.
 */

#include "archetypes/hadronic/bh_exact_chain_cdf.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_sampler_host.h"
#include "archetypes/hadronic/bh_sampler_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace arch::bh;

namespace {

  int g_pass_contract = 0, g_fail_contract = 0, g_pass_impl = 0, g_fail_impl = 0;

  bool check(bool contract, const std::string& name, double measured, double tol, const std::string& what) {
    const bool ok = std::isfinite(measured) && std::abs(measured) <= tol;
    std::printf("[%s] %s%s: %s = %.6e (tol %.3e)\n", ok ? "PASS" : "FAIL", contract ? "" : "(impl) ", name.c_str(),
                what.c_str(), measured, tol);
    if (contract) {
      ok ? ++g_pass_contract : ++g_fail_contract;
    } else {
      ok ? ++g_pass_impl : ++g_fail_impl;
    }
    return ok;
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

  // largest |F_emp - F| over the edges, F_emp(e) = #(x <= e)/N for sorted x
  double ks_at_edges(const std::vector<double>& sorted, const std::vector<double>& edges,
                     const std::vector<double>& F, double* where = nullptr) {
    double d = 0.0;
    for (std::size_t i = 0; i < edges.size(); ++i) {
      const auto   it   = std::upper_bound(sorted.begin(), sorted.end(), edges[i]);
      const double Femp = static_cast<double>(it - sorted.begin()) / static_cast<double>(sorted.size());
      if (std::abs(Femp - F[i]) > d) {
        d = std::abs(Femp - F[i]);
        if (where) {
          *where = edges[i];
        }
      }
    }
    return d;
  }

  struct Case {
    std::string bath;
    double      eps0, gamma;
  };

} // namespace

auto main() -> int {
  try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const auto   t_start = std::chrono::steady_clock::now();
    auto         T       = std::make_shared<const rf::RestFrameTables>(rf::RestFrameTables::build());
    auto         cum     = std::make_shared<const rf::SamplerCumulatives>();
    const double mu_r    = rf::canonical_bath("mono05").mu_r;
    diag("rest-frame tables built in %.1f s; mu_r = %g", seconds_since(t_start), mu_r);

    const std::vector<Case> cases { { "mono05", 0.5, 100.0 }, { "mono05", 0.5, 1000.0 }, { "mono005", 0.05, 1000.0 } };
    const std::size_t       N = 1000000;
    double worst_quad_vs_ref = 0.0, worst_self_ks_margin = -1.0e300, worst_conv = 0.0, worst_mean_se = 0.0;
    double worst_fix = 0.0;
    bool   any_fix = false, all_sha = true;
    for (std::size_t j = 0; j < cases.size(); ++j) {
      const Case&  c  = cases[j];
      const double g  = c.gamma;
      const double gb = std::sqrt((g - 1.0) * (g + 1.0));
      // the sampler for this mono bath (only Gamma_th of the rate table enters the chain)
      BathSpec spec;
      spec.type        = "mono";
      spec.eps0        = c.eps0;
      const auto bath  = make_bath(spec);
      BetheHeitlerRates rates(bath);
      const auto        H = build_tables(rates, mu_r);
      auto B = std::make_shared<const rf::BathEpsTables>(rf::BathEpsTables::build(*bath, H, *cum));
      rf::HostSampler S(bath, H, T, cum, B);
      // the reference's CDF
      const std::string path = std::string(BH_TEST_DATA_DIR) + "/bhref-0.2.0/ka2008_" + c.bath + "_G" +
                               (g == 100.0 ? std::string("100") : std::string("1000")) + ".txt";
      const FixtureContents ref = read_fixture_file(path);
      const bool sha_ok = ref.header.count("data_sha256") && ref.header.at("data_sha256") == "\"" + ref.data_sha256 + "\"";
      all_sha           = all_sha && sha_ok;
      std::vector<double> edges, Fref;
      for (const auto& r : ref.rows) {
        edges.push_back(r[0]);
        Fref.push_back(r[1]);
      }
      double max_bin = Fref.front();
      for (std::size_t i = 1; i < Fref.size(); ++i) {
        max_bin = std::max(max_bin, Fref[i] - Fref[i - 1]);
      }
      // the sample: head-on and exact lab energies of the same events
      rf::SplitMix64      rng(8080u + j);
      std::vector<double> Eho(N), Eex(N);
      double              s_ho = 0.0, s_ex = 0.0, s_pred = 0.0, s_ex2 = 0.0, s_ho2 = 0.0;
      auto                t0 = std::chrono::steady_clock::now();
      for (std::size_t n = 0; n < N; ++n) {
        const auto   e  = S.draw(g, false, rng);
        const double pm = rf::momentum(e.Em);
        Eho[n]          = g * (e.Em - pm * e.cosm);
        Eex[n]          = e.lab_m.E;
        s_ho += Eho[n];
        s_ho2 += Eho[n] * Eho[n];
        s_ex += Eex[n];
        s_ex2 += Eex[n] * Eex[n];
        s_pred += c.eps0 * pm * e.cosm / e.k;
      }
      const double tmc = seconds_since(t0);
      std::sort(Eho.begin(), Eho.end());
      std::sort(Eex.begin(), Eex.end());
      double       E_ho_at = 0.0, E_ex_at = 0.0;
      const double ks_ho = ks_at_edges(Eho, edges, Fref, &E_ho_at);
      const double ks_ex = ks_at_edges(Eex, edges, Fref, &E_ex_at);
      const double khi   = c.eps0 * (g + gb);
      diag("A8 %s Gamma_p = %g: eps'_hi = %.4f, eps'_hi/mu_r = %.4f, eps Gamma_p/mu_r = %.4f%s; reference %s (version "
           "%s, data_sha256 %s); N = %zu in %.1f s; largest reference bin probability %.4f",
           c.bath.c_str(), g, khi, khi / mu_r, c.eps0 * g / mu_r,
           khi / mu_r > 0.1 ? " (outside the small-eps' regime: a formal mathematical comparison only)" : "",
           path.c_str(), ref.header.count("version") ? ref.header.at("version").c_str() : "?",
           sha_ok ? "verified" : "MISMATCH", N, tmc, max_bin);
      check(true,
            "A8 (head-on reading): KS between the sampler's events mapped head-on, E = Gamma (E' - p' cos theta'_gamma), "
            "and the KA2008 CDF, " + c.bath + " Gamma_p = " + std::to_string(static_cast<int>(g)) +
              " (worst at E = " + std::to_string(E_ho_at) + ")",
            ks_ho, 0.02, "KS");
      diag("A8 original reading (labelled failed diagnostic): exact-chain events (exact photon direction and boost) vs the "
           "KA2008 CDF, %s Gamma_p = %g: KS = %.4f at E = %.4f%s; mean lab energy exact %.4f, head-on %.4f, difference %.4f "
           "(predicted eps <p' cos theta'_gamma/eps'> = %.4f; limit eps/2 = %.4f); fraction of head-on energies below 1: "
           "%.4f (reference CDF at E = 1: %.4f)",
           c.bath.c_str(), g, ks_ex, E_ex_at, ks_ex > 0.02 ? " (fails 0.02, as expected)" : " (passes 0.02 here)",
           s_ex / N, s_ho / N, (s_ex - s_ho) / N, s_pred / N, 0.5 * c.eps0,
           static_cast<double>(std::upper_bound(Eho.begin(), Eho.end(), 1.0) - Eho.begin()) / N, Fref.front());
      // the head-on CDF of this code's laws by quadrature, at the reference's edges
      t0 = std::chrono::steady_clock::now();
      double     mean_hq = 0.0;
      const auto Fhq     = rf::exact_chain_lab_cdf(g, c.eps0, edges, rf::ExactChainQuadrature {}, &mean_hq,
                                                   rf::LabMap::HeadOn);
      double     dq      = 0.0;
      for (std::size_t i = 0; i < edges.size(); ++i) {
        dq = std::max(dq, std::abs(Fhq[i] - Fref[i]));
      }
      worst_quad_vs_ref = std::max(worst_quad_vs_ref, dq);
      diag("head-on quadrature of the MOK 3D-2000 law vs the reference's KA2008 CDF, %s Gamma_p = %g: max |dF| = %.2e "
           "over %zu edges (%.1f s); <E>_head-on quadrature %.5f, sample %.5f +- %.5f",
           c.bath.c_str(), g, dq, edges.size(), seconds_since(t0), mean_hq, s_ho / N,
           std::sqrt(std::max(0.0, s_ho2 / N - (s_ho / N) * (s_ho / N)) / N));
      // the exact-chain self-check (mono05 only, round 1's comparison)
      if (c.bath == "mono05") {
        double E_lo, E_hi;
        rf::lab_energy_range(g, c.eps0, E_lo, E_hi);
        const int           NB = 200;
        std::vector<double> ed(NB + 1);
        for (int i = 0; i <= NB; ++i) {
          ed[i] = E_lo * std::pow(E_hi / E_lo, static_cast<double>(i) / NB);
        }
        const rf::ExactChainQuadrature q;
        t0                  = std::chrono::steady_clock::now();
        double     mean_q   = 0.0, mean_r = 0.0;
        const auto F        = rf::exact_chain_lab_cdf(g, c.eps0, ed, q, &mean_q);
        const auto Fr       = rf::exact_chain_lab_cdf(g, c.eps0, ed, q.refined(), &mean_r);
        double     conv     = 0.0;
        for (int i = 0; i <= NB; ++i) {
          conv = std::max(conv, std::abs(F[i] - Fr[i]));
        }
        double       E_at   = 0.0;
        const double ks     = ks_at_edges(Eex, ed, F, &E_at);
        const double crit   = 1.628 / std::sqrt(static_cast<double>(N)) + conv;
        const double mean   = s_ex / N;
        const double se     = std::sqrt(std::max(0.0, s_ex2 / N - mean * mean) / N);
        const double nse    = std::abs(mean - mean_q) / se;
        worst_self_ks_margin = std::max(worst_self_ks_margin, ks - crit);
        worst_conv           = std::max(worst_conv, conv);
        worst_mean_se        = std::max(worst_mean_se, nse);
        diag("exact-chain self-check %s Gamma_p = %g: KS = %.5f at E = %.4f vs critical %.5f (1.628/sqrt(N) + quadrature "
             "change %.2e); <E>_quad = %.5f (refined %.5f), <E>_MC = %.5f +- %.5f (%.1f s.e.); %.1f s",
             c.bath.c_str(), g, ks, E_at, crit, conv, mean_q, mean_r, mean, se, nse, seconds_since(t0));
        const std::string fpath = std::string(BH_TEST_DATA_DIR) + "/cpp-sampler-candidate/exact_chain_cdf_mono05_G" +
                                  (g == 100.0 ? std::string("100") : std::string("1000")) + ".txt";
        std::ifstream probe(fpath);
        if (probe) {
          probe.close();
          const FixtureContents fx = read_fixture_file(fpath);
          double                w  = 0.0;
          if (fx.rows.size() == static_cast<std::size_t>(NB) && fx.columns.size() == 4) {
            for (int i = 0; i < NB; ++i) {
              w = std::max(w, std::abs(fx.rows[i][0] / ed[i] - 1.0));
              w = std::max(w, std::abs(fx.rows[i][1] / ed[i + 1] - 1.0));
              w = std::max(w, std::abs(fx.rows[i][2] - F[i + 1]));
            }
          } else {
            w = 1.0e300;
          }
          worst_fix = std::max(worst_fix, w);
          any_fix   = true;
          diag("fixture %s: version %s, data_sha256 %s; max deviation from the live quadrature %.2e", fpath.c_str(),
               fx.header.at("version").c_str(), fx.data_sha256.c_str(), w);
        } else {
          diag("fixture %s not present: nothing to compare", fpath.c_str());
        }
      }
    }
    check(false, "the reference's KA2008 files: data block against the header's data_sha256", all_sha ? 0.0 : 1.0, 0.0,
          "mismatches");
    check(false, "head-on quadrature of this code's law vs the reference's KA2008 CDF, three cases", worst_quad_vs_ref,
          1.0e-3, "max |dF|");
    check(false, "exact-chain self-check: KS (exact-chain events vs the exact-chain quadrature) minus its critical value "
                 "1.628/sqrt(N) + quadrature change, mono05 Gamma_p = 100 and 1000",
          std::max(0.0, worst_self_ks_margin), 0.0, "max excess");
    check(false, "exact-chain quadrature self-convergence (order + 4, panels / 2)", worst_conv, 1.0e-3, "max |dF|");
    check(false, "exact-chain mean lab e- energy, sample vs quadrature", worst_mean_se, 4.0, "max |dev| / s.e.");
    if (any_fix) {
      check(false, "exact-chain fixtures vs the live quadrature (edges relative, CDF absolute)", worst_fix, 1.0e-10,
            "max deviation");
    }
    std::printf("\nbh_ka2008 summary: contract items %d passed, %d failed; implementation-chosen checks %d passed, %d "
                "failed; %.0f s\n",
                g_pass_contract, g_fail_contract, g_pass_impl, g_fail_impl, seconds_since(t_start));
    return (g_fail_contract + g_fail_impl) == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_ka2008: exception: %s\n", e.what());
    return 2;
  }
}
