/**
 * @file tests/archetypes/bh_sampler_fixtures.cpp
 * @brief Contract A10 inside ctest (P2 step 4, round 2, stage 6): the C++
 *        sampler fixtures (`tests/data/hadronic/cpp-sampler-candidate/`)
 *        against the Python reference's (`tests/data/hadronic/bhref-0.2.0/`,
 *        bhref 0.2.0 (frozen), unfrozen), file against file; no Python.
 *
 * Contract items (tolerances of contract A10 and of the round-2 brief, never
 * adjusted here):
 *   qk     `qk_table.txt`: sigma, q_k, <E'->, <p' cos theta'> at every eps'
 *          node within 1e-4 relative; the eps' nodes themselves equal to 1e-12
 *   mom    `moments_<bath>.txt`, five baths: at every Gamma_p node, every
 *          Monte Carlo moment present in both files within 4 combined standard
 *          errors: <D>, <D^2>, <E->, <E+>, <cos theta-> (lab), <eps> (broad
 *          baths), and the weighted chain's E_p[1/D] <D>_MC; deterministic
 *          columns compared at 1e-12 relative, not by a standard-error
 *          criterion: the Gamma_p nodes, and <eps> of a mono bath (eps = eps0
 *          exactly, zero variance)
 *   hist   `hist_<bath>_G<Gamma>.txt`, mono05 and powerlaw at four Gamma_p:
 *          the binned two-sample Kolmogorov-Smirnov statistic of every quantity
 *          (E-, E+, cos theta-, cos theta+, D) and both chains, on the
 *          reference's bins, against the 99% critical value
 *          1.628 sqrt((n1 + n2)/(n1 n2)) with the effective sample sizes (the
 *          counts N for the unweighted chain; the Kish sizes (sum w)^2/sum w^2,
 *          the files' weighted_ESS, for the 1/D-weighted chain); at least 95%
 *          of the combinations must pass, the failures are listed
 *   A8     the head-on A8 KS: rest-frame events of the C++ sampler mapped
 *          head-on, E = Gamma (E' - p' cos theta'_gamma), N = 1e6, against the
 *          reference's KA2008 CDF (`ka2008_<bath>_G<Gamma>.txt`), KS <= 0.02 at
 *          mono05 Gamma_p = 100, 1000 and mono005 Gamma_p = 1000
 *
 * Standard errors (convention, stated): the reference publishes s.e. for <D>
 * and for the weighted product only; for <D^2>, <E->, <E+>, <cos theta->,
 * <eps> its s.e. is estimated from the C++ s.e. of the same column scaled by
 * sqrt(N_cpp/N_ref) (the same population variance; both files use N = 1e6
 * unweighted draws per node), so the combined s.e. is sqrt(se_cpp^2 + se_ref^2).
 *
 * Diagnostics: the acceptance efficiencies of the weighted chains (C++
 * n_accepted/n_attempts against the reference's accepted/proposal counts),
 * the extremes D_min, D_max, the C++ deterministic <D>_sampled against both
 * Monte Carlo means.
 *
 * Seeds (A8 part): SplitMix64(8080 + j), as in bh_ka2008.cpp.
 */

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_sampler_host.h"
#include "archetypes/hadronic/bh_sampler_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace arch::bh;

namespace {

  int g_pass = 0, g_fail = 0;

  bool check(const std::string& name, bool ok, double measured, double tol, const std::string& what) {
    std::printf("[%s] %s: %s = %.6e (tol %.3e)\n", ok ? "PASS" : "FAIL", name.c_str(), what.c_str(), measured, tol);
    ok ? ++g_pass : ++g_fail;
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

  const std::string DATA = BH_TEST_DATA_DIR;

  struct Table {
    FixtureContents f;
    std::string     path;

    std::size_t col(const std::string& n) const {
      for (std::size_t c = 0; c < f.columns.size(); ++c) {
        if (f.columns[c] == n) {
          return c;
        }
      }
      throw std::runtime_error(path + ": no column " + n);
    }

    double at(std::size_t r, const std::string& n) const {
      return f.rows[r][col(n)];
    }

    std::string head(const std::string& k) const {
      return f.header.count(k) ? f.header.at(k) : std::string("?");
    }
  };

  Table load(const std::string& rel) {
    Table t;
    t.path = DATA + "/" + rel;
    t.f    = read_fixture_file(t.path); // validates data_sha256
    return t;
  }

  double rel(double a, double b) {
    return std::abs(a - b) / std::max(std::abs(b), 1.0e-300);
  }

  // ---------------------------------------------------------------------- //
  void test_qk() {
    std::printf("\n== A10 item 1: qk_table.txt, C++ vs reference ==\n");
    const Table C = load("cpp-sampler-candidate/qk_table.txt");
    const Table R = load("bhref-0.2.0/qk_table.txt");
    const bool  same_n = C.f.rows.size() == R.f.rows.size();
    check("qk: node count", same_n, static_cast<double>(C.f.rows.size()), static_cast<double>(R.f.rows.size()),
          "C++ nodes (reference count as tol)");
    if (!same_n) {
      return;
    }
    double      w_node = 0.0;
    const char* pairs[4][3] = { { "sigma_from_g", "sigma_energy_law", "sigma" },
                                { "q_k", "qk_machinery", "q_k" },
                                { "mean_E_minus", "mean_Eminus_prime", "<E'->" },
                                { "mean_p_cos_theta", "mean_pcos", "<p' cos theta'>" } };
    double      w[4] = { 0, 0, 0, 0 }, kw[4] = { 0, 0, 0, 0 };
    for (std::size_t i = 0; i < C.f.rows.size(); ++i) {
      w_node = std::max(w_node, rel(C.at(i, "eps_prime"), R.at(i, "eps_prime")));
      for (int q = 0; q < 4; ++q) {
        const double d = rel(C.at(i, pairs[q][0]), R.at(i, pairs[q][1]));
        if (d > w[q]) {
          w[q]  = d;
          kw[q] = C.at(i, "eps_prime");
        }
      }
    }
    check("qk: eps' nodes equal (deterministic)", w_node <= 1.0e-12, w_node, 1.0e-12, "max rel. dev.");
    for (int q = 0; q < 4; ++q) {
      check(std::string("qk: ") + pairs[q][2] + " at every eps' node (worst eps' = " + std::to_string(kw[q]) + ")",
            w[q] <= 1.0e-4, w[q], 1.0e-4, "max rel. dev.");
    }
    diag("qk: C++ %s (data_sha256 %s), reference %s (data_sha256 %s)", C.head("version").c_str(), C.f.data_sha256.c_str(),
         R.head("version").c_str(), R.f.data_sha256.c_str());
  }

  // ---------------------------------------------------------------------- //
  void test_moments() {
    std::printf("\n== A10 item 2: moments_<bath>.txt, C++ vs reference, every node ==\n");
    struct Pair {
      const char* cpp;
      const char* cpp_se;
      const char* ref;
      const char* ref_se; // nullptr: estimated from the C++ s.e.
      const char* name;
    };
    const std::vector<Pair> P = {
      { "Dmean_mc", "Dmean_se", "Dmean", "Dmean_se", "<D>" },
      { "D2mean", "D2mean_se", "D2mean", nullptr, "<D^2>" },
      { "Emean_minus", "Emean_minus_se", "Eminus_mean", nullptr, "<E->" },
      { "Emean_plus", "Emean_plus_se", "Eplus_mean", nullptr, "<E+>" },
      { "cos_minus_lab_mean", "cos_minus_lab_mean_se", "cosminus_lab_mean", nullptr, "<cos theta->" },
      { "eps_mean_mc", "eps_mean_mc_se", "eps_mean", nullptr, "<eps>" },
      { "Einv_Dmean_mc", "Einv_Dmean_mc_se", "weighted_invD_times_Dmean", "weighted_product_se", "E_p[1/D] <D>" },
    };
    std::size_t n_cmp = 0, n_fail = 0, n_det = 0, n_det_fail = 0;
    double      worst_z = 0.0, worst_det = 0.0;
    std::string fails, worst_where;
    for (const std::string bath : { "mono05", "mono03", "powerlaw", "blackbody", "mono05_mu100" }) {
      const Table C = load("cpp-sampler-candidate/moments_" + bath + ".txt");
      const Table R = load("bhref-0.2.0/moments_" + bath + ".txt");
      const bool  mono = C.head("photon_field").find("\"mono\"") != std::string::npos;
      if (C.f.rows.size() != R.f.rows.size()) {
        ++n_det_fail;
        fails += " " + bath + ": node counts differ;";
        continue;
      }
      const double N_cpp = C.at(0, "n_unweighted");
      const double N_ref = std::stod(R.head("N_unweighted"));
      const double scale = std::sqrt(N_cpp / N_ref);
      double       bz = 0.0, beff = 0.0, bsamp = 0.0;
      std::string  bwhere;
      for (std::size_t i = 0; i < C.f.rows.size(); ++i) {
        const double g = C.at(i, "gamma_p");
        // deterministic columns
        {
          const double d = rel(g, R.at(i, "gamma_p"));
          ++n_det;
          worst_det = std::max(worst_det, d);
          if (!(d <= 1.0e-12)) {
            ++n_det_fail;
            fails += " " + bath + " node " + std::to_string(i) + " Gamma_p differs;";
          }
        }
        if (mono) {
          const double d = rel(C.at(i, "eps_mean_mc"), R.at(i, "eps_mean"));
          ++n_det;
          worst_det = std::max(worst_det, d);
          if (!(d <= 1.0e-12) || C.at(i, "eps_mean_mc_se") != 0.0) {
            ++n_det_fail;
            fails += " " + bath + " Gamma_p " + std::to_string(g) + " <eps> (deterministic);";
          }
        }
        // Monte Carlo moments
        for (const auto& p : P) {
          if (mono && std::string(p.name) == "<eps>") {
            continue;
          }
          const double a  = C.at(i, p.cpp), sa = C.at(i, p.cpp_se);
          const double b  = R.at(i, p.ref);
          const double sb = p.ref_se ? R.at(i, p.ref_se) : sa * scale;
          const double se = std::sqrt(sa * sa + sb * sb);
          const double z  = se > 0.0 ? std::abs(a - b) / se : (a == b ? 0.0 : 1.0e300);
          ++n_cmp;
          if (z > bz) {
            bz     = z;
            bwhere = std::string(p.name) + " at Gamma_p " + std::to_string(g);
          }
          if (!(z <= 4.0)) {
            ++n_fail;
            fails += " " + bath + " Gamma_p " + std::to_string(g) + " " + p.name + " (" + std::to_string(z) +
                     " s.e.);";
          }
        }
        // diagnostics
        {
          const double ea = C.at(i, "efficiency"), sea = C.at(i, "efficiency_se");
          const double eb = R.at(i, "acceptance_efficiency");
          const double nb = R.at(i, "accepted_count");
          const double seb = eb * std::sqrt(std::max(0.0, 1.0 - eb) / nb);
          beff = std::max(beff, std::abs(ea - eb) / std::sqrt(sea * sea + seb * seb));
          const double Ds = C.at(i, "Dmean_sampled");
          bsamp = std::max(bsamp, std::abs(R.at(i, "Dmean") - Ds) / R.at(i, "Dmean_se"));
        }
      }
      if (bz > worst_z) {
        worst_z     = bz;
        worst_where = bath + " " + bwhere;
      }
      diag("moments %s: %zu nodes; max |C++ - ref| / combined s.e. = %.2f (%s); acceptance efficiency max %.2f "
           "combined s.e.; reference <D> vs the C++ sampled-density <D> max %.2f ref s.e.; C++ %s, reference %s",
           bath.c_str(), C.f.rows.size(), bz, bwhere.c_str(), beff, bsamp, C.f.data_sha256.substr(0, 12).c_str(),
           R.f.data_sha256.substr(0, 12).c_str());
    }
    check("moments: deterministic columns (Gamma_p nodes; <eps> of the mono baths, s.e. 0) at 1e-12 relative, " +
            std::to_string(n_det) + " values",
          n_det_fail == 0, worst_det, 1.0e-12, "max rel. dev.");
    check("moments: every Monte Carlo moment within 4 combined s.e. at every node of the five baths (" +
            std::to_string(n_cmp) + " comparisons; worst " + worst_where + ")",
          n_fail == 0, worst_z, 4.0, "max |dev| / s.e.");
    if (!fails.empty()) {
      diag("moments: failures:%s", fails.c_str());
    }
  }

  // ---------------------------------------------------------------------- //
  struct KsOut {
    double ks, crit;
  };

  // binned two-sample KS of two histograms given as per-bin weights, with effective sizes n1, n2
  KsOut ks2(const std::vector<double>& a, double n1, const std::vector<double>& b, double n2) {
    double A = 0.0, B = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      A += a[i];
      B += b[i];
    }
    double Fa = 0.0, Fb = 0.0, d = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      Fa += a[i] / A;
      Fb += b[i] / B;
      d = std::max(d, std::abs(Fa - Fb));
    }
    return { d, 1.628 * std::sqrt((n1 + n2) / (n1 * n2)) };
  }

  void test_hist() {
    std::printf("\n== A10 item 3: hist_<bath>_G<Gamma>.txt, C++ vs reference, weighted two-sample KS at 99%% ==\n");
    const char* const qname[5] = { "E-", "E+", "cos-", "cos+", "D" };
    std::size_t       n_comb = 0, n_pass = 0;
    std::string       fails;
    double            w_edge = 0.0;
    for (const std::string bath : { "mono05", "powerlaw" }) {
      const std::vector<std::string> gs = bath == "mono05" ? std::vector<std::string> { "2.23125", "3", "10", "100" }
                                                           : std::vector<std::string> { "1.09375", "3", "10", "100" };
      for (const auto& gstr : gs) {
        const Table C = load("cpp-sampler-candidate/hist_" + bath + "_G" + gstr + ".txt");
        const Table R = load("bhref-0.2.0/hist_" + bath + "_G" + gstr + ".txt");
        if (C.f.rows.size() != R.f.rows.size()) {
          ++n_comb;
          fails += " " + bath + " G" + gstr + ": row counts differ;";
          continue;
        }
        const double Nc = std::stod(C.head("n_unweighted")), Nr = std::stod(R.head("N_unweighted"));
        const double Ec = std::stod(C.head("weighted_ESS")), Er = std::stod(R.head("weighted_ESS"));
        std::string  line;
        for (int q = 0; q < 5; ++q) {
          std::vector<double> cu, ru, cw, rw;
          for (std::size_t r = 0; r < C.f.rows.size(); ++r) {
            if (static_cast<int>(C.at(r, "quantity_id")) != q) {
              continue;
            }
            w_edge = std::max({ w_edge, rel(C.at(r, "lo"), R.at(r, "lo")), rel(C.at(r, "hi"), R.at(r, "hi")) });
            cu.push_back(C.at(r, "unweighted_count"));
            ru.push_back(R.at(r, "unweighted_count"));
            cw.push_back(C.at(r, "sum_1_over_D"));
            rw.push_back(R.at(r, "sum_1_over_D"));
          }
          const KsOut u  = ks2(cu, Nc, ru, Nr);
          const KsOut wt = ks2(cw, Ec, rw, Er);
          for (int chain = 0; chain < 2; ++chain) {
            const KsOut& k = chain == 0 ? u : wt;
            ++n_comb;
            if (k.ks <= k.crit) {
              ++n_pass;
            } else {
              fails += " " + bath + " G" + gstr + " " + qname[q] + (chain == 0 ? " unweighted" : " weighted") +
                       " (KS " + std::to_string(k.ks) + " > " + std::to_string(k.crit) + ");";
            }
          }
          char b[160];
          std::snprintf(b, sizeof(b), " %s %.5f/%.5f%s %.5f/%.5f%s", qname[q], u.ks, u.crit, u.ks <= u.crit ? "" : "!",
                        wt.ks, wt.crit, wt.ks <= wt.crit ? "" : "!");
          line += b;
        }
        diag("hist %s Gamma_p %s (KS/critical, unweighted then weighted; ESS C++ %.0f, ref %.0f):%s", bath.c_str(),
             gstr.c_str(), Ec, Er, line.c_str());
      }
    }
    check("hist: bin edges of the C++ fixtures = the reference's (taken from them by --bins-from)", w_edge <= 1.0e-12,
          w_edge, 1.0e-12, "max rel. dev.");
    const double frac = n_comb ? static_cast<double>(n_pass) / static_cast<double>(n_comb) : 0.0;
    check("hist: fraction of the " + std::to_string(n_comb) +
            " combinations (2 baths x 4 Gamma_p x 5 quantities x 2 chains) passing the weighted two-sample KS at 99%",
          n_comb > 0 && frac >= 0.95, frac, 0.95, "fraction (>= 0.95 required)");
    if (!fails.empty()) {
      diag("hist: failing combinations:%s", fails.c_str());
    }
  }

  // ---------------------------------------------------------------------- //
  void test_a8_headon() {
    std::printf("\n== A10 item 4: the head-on A8 KS against the reference's KA2008 CDF ==\n");
    const auto t0  = std::chrono::steady_clock::now();
    auto       T   = std::make_shared<const rf::RestFrameTables>(rf::RestFrameTables::build());
    auto       cum = std::make_shared<const rf::SamplerCumulatives>();
    diag("rest-frame tables built in %.1f s", seconds_since(t0));
    struct Case {
      std::string bath;
      double      eps0, gamma;
      std::string gs;
    };
    const std::vector<Case> cases { { "mono05", 0.5, 100.0, "100" }, { "mono05", 0.5, 1000.0, "1000" },
                                    { "mono005", 0.05, 1000.0, "1000" } };
    const std::size_t       N = 1000000;
    for (std::size_t j = 0; j < cases.size(); ++j) {
      const auto& c = cases[j];
      BathSpec    spec;
      spec.type   = "mono";
      spec.eps0   = c.eps0;
      const auto        bath = make_bath(spec);
      BetheHeitlerRates rates(bath);
      const auto        H = build_tables(rates, 1836.15);
      auto B = std::make_shared<const rf::BathEpsTables>(rf::BathEpsTables::build(*bath, H, *cum));
      rf::HostSampler S(bath, H, T, cum, B);
      const Table     R = load("bhref-0.2.0/ka2008_" + c.bath + "_G" + c.gs + ".txt");
      std::vector<double> E(N);
      rf::SplitMix64      rng(8080u + j);
      for (std::size_t n = 0; n < N; ++n) {
        const auto e = S.draw(c.gamma, false, rng);
        E[n]         = c.gamma * (e.Em - rf::momentum(e.Em) * e.cosm);
      }
      std::sort(E.begin(), E.end());
      double ks = 0.0;
      for (std::size_t r = 0; r < R.f.rows.size(); ++r) {
        const double edge = R.at(r, "Eminus_edge");
        const double Femp = static_cast<double>(std::upper_bound(E.begin(), E.end(), edge) - E.begin()) / N;
        ks                = std::max(ks, std::abs(Femp - R.at(r, "cdf")));
      }
      check("A8 head-on KS, " + c.bath + " Gamma_p = " + c.gs + " (eps'_hi/mu_r = " +
              std::to_string(c.eps0 * (c.gamma + std::sqrt(c.gamma * c.gamma - 1.0)) / 1836.15) + ")",
            ks <= 0.02, ks, 0.02, "KS");
    }
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  std::vector<std::string> groups(argv + 1, argv + argc);
  auto want = [&groups](const char* g) {
    return groups.empty() || std::find(groups.begin(), groups.end(), std::string(g)) != groups.end();
  };
  try {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const auto t0 = std::chrono::steady_clock::now();
    if (want("qk")) {
      test_qk();
    }
    if (want("moments")) {
      test_moments();
    }
    if (want("hist")) {
      test_hist();
    }
    if (want("a8")) {
      test_a8_headon();
    }
    std::printf("\nbh_sampler_fixtures summary: %d passed, %d failed; %.0f s\n", g_pass, g_fail, seconds_since(t0));
    return g_fail == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_sampler_fixtures: exception: %s\n", e.what());
    return 2;
  }
}
