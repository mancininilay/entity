/**
 * @file tests/archetypes/bh_ka2008.cpp
 * @brief The UR lab electron spectrum anchor of the rest-frame sampler (P2
 *        step 4, stage 3, contract A8): the unweighted sampled e- lab spectrum
 *        against the quadrature of Kelner and Aharonian 2008 eqs. (58) to (64)
 *        (bh_lab_spectrum.h) at Gamma_p = 100 and 1000 for mono05.
 *
 * Contract item: A8, KS statistic <= 0.02 with N = 1e6 at each of the two
 * points (the KS statistic is the largest |F_MC - F_quad| over the 200
 * log-spaced bin edges of the fixture, i.e. over the grid on which the
 * reference CDF is tabulated; the sup over all energies is bounded below by it
 * and, with 200 bins, differs from it by less than the bin probability, ~0.5%
 * of the mass at most, which is stated).
 *
 * Implementation-chosen checks: the quadrature's own convergence under
 * refinement (<= 1e-3 in the CDF, the 2% level being the anchor's); the mean
 * lab energy of the sample against the quadrature within 4 s.e.; the fixture
 * files, when present, against the live quadrature (1e-10).
 *
 * Applicability (stated): the condition eps' Gamma << mu_r of the brief reads
 * eps'_hi Gamma / mu_r = 5.4 at Gamma = 100 and 545 at Gamma = 1000 for
 * mono05 (mu_r = 1836.15), so both points are outside it; the sampler and the
 * quadrature use the same static-source Born law, and the comparison tests
 * the sampling chain (energy law, angular law, boost) rather than the
 * static-source model. eps'_hi = 100 at Gamma = 100 (just above the production
 * envelope 0.05 mu_r = 91.8) and 1000 at Gamma = 1000 (the kernel cap).
 *
 * Seeds: SplitMix64(8080 + j) for the two points.
 */

#include "archetypes/hadronic/bh_lab_spectrum.h"
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

} // namespace

auto main() -> int {
  try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const auto t_start = std::chrono::steady_clock::now();
    auto       T       = std::make_shared<const rf::RestFrameTables>(rf::RestFrameTables::build());
    auto       cum     = std::make_shared<const rf::SamplerCumulatives>();
    const auto cb      = rf::canonical_bath("mono05");
    const auto bath    = make_bath(cb.spec);
    BetheHeitlerRates rates(bath);
    const auto        H = build_tables(rates, cb.mu_r);
    auto B = std::make_shared<const rf::BathEpsTables>(rf::BathEpsTables::build(*bath, H, *cum));
    rf::HostSampler S(bath, H, T, cum, B);
    const double eps0 = cb.spec.eps0;
    diag("tables built in %.1f s; mono05 eps0 = %g, mu_r = %g", seconds_since(t_start), eps0, cb.mu_r);

    const std::size_t N  = 1000000;
    const int         NB = 200;
    double            worst_ks = 0.0, worst_conv = 0.0, worst_mean_se = 0.0, worst_fix = 0.0;
    bool              any_fix = false;
    for (int j = 0; j < 2; ++j) {
      const double g  = (j == 0) ? 100.0 : 1000.0;
      const double gb = std::sqrt((g - 1.0) * (g + 1.0));
      double       E_lo, E_hi;
      rf::lab_energy_range(g, eps0, E_lo, E_hi);
      std::vector<double> edges(NB + 1);
      for (int i = 0; i <= NB; ++i) {
        edges[i] = E_lo * std::pow(E_hi / E_lo, static_cast<double>(i) / NB);
      }
      const rf::LabSpectrumQuadrature q;
      auto                            t0 = std::chrono::steady_clock::now();
      double                          mean_q = 0.0, mean_r = 0.0;
      const auto                      F  = rf::lab_electron_cdf(g, eps0, edges, q, &mean_q);
      const double                    tq = seconds_since(t0);
      t0                                 = std::chrono::steady_clock::now();
      const auto                      Fr = rf::lab_electron_cdf(g, eps0, edges, q.refined(), &mean_r);
      const double                    tr = seconds_since(t0);
      double                          conv = 0.0;
      for (int i = 0; i <= NB; ++i) {
        conv = std::max(conv, std::abs(F[i] - Fr[i]));
      }
      // the sample
      rf::SplitMix64      rng(8080u + j);
      std::vector<double> E(N);
      double              s1 = 0.0, s2 = 0.0;
      t0 = std::chrono::steady_clock::now();
      for (std::size_t n = 0; n < N; ++n) {
        const auto e = S.draw(g, false, rng);
        E[n]         = e.lab_m.E;
        s1 += E[n];
        s2 += E[n] * E[n];
      }
      const double tmc = seconds_since(t0);
      std::sort(E.begin(), E.end());
      double ks = 0.0, ks_E = 0.0, ks_F = 0.0;
      for (int i = 0; i <= NB; ++i) {
        const auto   it   = std::upper_bound(E.begin(), E.end(), edges[i]);
        const double Femp = static_cast<double>(it - E.begin()) / N;
        const double d    = std::abs(Femp - F[i]);
        if (d > ks) {
          ks   = d;
          ks_E = edges[i];
          ks_F = F[i];
        }
      }
      const double mean_mc = s1 / N;
      const double se      = std::sqrt(std::max(0.0, s2 / N - mean_mc * mean_mc) / N);
      const double nse     = std::abs(mean_mc - mean_q) / se;
      diag("Gamma_p = %g: eps'_hi = %.4f, eps'_hi Gamma/mu_r = %.1f, E- lab range [%.3f, %.1f]; quadrature %.1f s "
           "(refined %.1f s, max |dF| = %.2e); <E->_quad = %.5f (refined %.5f), <E->_MC = %.5f +- %.5f (%.1f s.e.); KS "
           "= %.5f at E = %.4f (F = %.4f); sampling %.1f s; sample quantiles: median %.4f, 90%% %.3f, 99%% %.2f, max %.1f",
           g, eps0 * (g + gb), eps0 * (g + gb) * g / cb.mu_r, E_lo, E_hi, tq, tr, conv, mean_q, mean_r, mean_mc, se,
           nse, ks, ks_E, ks_F, tmc, E[N / 2], E[(9 * N) / 10], E[(99 * N) / 100], E.back());
      worst_ks      = std::max(worst_ks, ks);
      worst_conv    = std::max(worst_conv, conv);
      worst_mean_se = std::max(worst_mean_se, nse);
      // the fixture, when present
      const std::string path = std::string(BH_TEST_DATA_DIR) + "/cpp-sampler-candidate/ka2008_mono05_G" +
                               (j == 0 ? std::string("100") : std::string("1000")) + ".txt";
      std::ifstream probe(path);
      if (probe) {
        probe.close();
        try {
          const FixtureContents c = read_fixture_file(path);
          double                w = 0.0;
          if (c.rows.size() == static_cast<std::size_t>(NB) && c.columns.size() == 4) {
            for (int i = 0; i < NB; ++i) {
              w = std::max(w, std::abs(c.rows[i][0] / edges[i] - 1.0));
              w = std::max(w, std::abs(c.rows[i][1] / edges[i + 1] - 1.0));
              w = std::max(w, std::abs(c.rows[i][2] - F[i + 1]));
            }
          } else {
            w = 1.0e300;
          }
          worst_fix = std::max(worst_fix, w);
          any_fix   = true;
          diag("fixture %s: producer %s, version %s, data_sha256 %s; max deviation from the live quadrature %.2e",
               path.c_str(), c.header.at("producer").c_str(), c.header.at("version").c_str(), c.data_sha256.c_str(),
               w);
        } catch (const std::exception& e) {
          worst_fix = 1.0e300;
          any_fix   = true;
          diag("fixture %s: unreadable: %s", path.c_str(), e.what());
        }
      } else {
        diag("fixture %s not present: nothing to compare", path.c_str());
      }
    }
    check(true, "A8: KS statistic between the sampled e- lab spectrum (N = 1e6) and the KA2008 CDF, Gamma_p = 100 and "
                "1000, mono05",
          worst_ks, 0.02, "max KS");
    check(false, "A8 quadrature self-convergence (order + 4, panels / 2): max |dF| over the edges", worst_conv, 1.0e-3,
          "max |dF|");
    check(false, "A8: mean lab e- energy, sample vs quadrature", worst_mean_se, 4.0, "max |dev| / s.e.");
    if (any_fix) {
      check(false, "A8 fixtures vs the live quadrature (edges relative, CDF absolute)", worst_fix, 1.0e-10,
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
