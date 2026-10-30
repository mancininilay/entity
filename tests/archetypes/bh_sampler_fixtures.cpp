/**
 * @file tests/archetypes/bh_sampler_fixtures.cpp
 * @brief Contract A10 inside ctest (P2 step 4, round 2, stage 6): the C++
 *        sampler fixtures (`tests/data/hadronic/cpp-sampler-candidate/`)
 *        against the Python reference's (`tests/data/hadronic/bhref-0.2.0/`,
 *        bhref 0.2.0, frozen), file against file; no Python.
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
 *   hist   `hist_<bath>_G<Gamma>.txt`, mono05 and powerlaw at four Gamma_p,
 *          every quantity (E-, E+, cos theta-, cos theta+, D) and both chains:
 *          the calibrated weighted two-sample test of
 *          `bhref-0.2.0-suppl1/COMPARISON.md` (round 3, Claude 3's F8;
 *          `bh_calibrated_test.h`): the common partition (adjacent bins merged
 *          until the C++ and reference unweighted counts and weighted packet
 *          counts all reach 100), the right-edge discrete Anderson-Darling
 *          statistic, its conditional Gaussian-multiplier null with the
 *          per-bin sums of w and w^2 (w = 1/D), 19,999 replicates, pass iff
 *          p >= 0.01, one merged bin = insufficient resolution (not a pass).
 *          D at Gamma_p = 10 and 100 on the supplement's log bins
 *          (`hist_logD_*.txt`, both directories) instead of the linear bins
 *          (8 substitutions; the linear-D results are printed as diagnostics,
 *          not counted); at least 95% of the 80 comparisons must pass, the
 *          failures are listed. Null seeds 2026102701 + 10 case + 5 chain +
 *          quantity (COMPARISON.md; case = mono05 2.23125, 3, 10, 100,
 *          powerlaw 1.09375, 3, 10, 100; chain 0 unweighted, 1 weighted).
 *          Diagnostic, not counted: within each file (C++ and reference) the
 *          weighted chain re-weighted by 1/D against the file's own unweighted
 *          chain, by the same test (it locates a C++-vs-reference difference
 *          on one side)
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
 * Implementation-chosen (group `calnull`): the size of the calibrated test
 * under its null, 100 comparisons of synthetic weighted samples drawn from one
 * distribution: at most 5 with p < 0.01, mean p within 0.5 +- 0.1.
 *
 * Diagnostics: the acceptance efficiencies of the weighted chains (C++
 * n_accepted/n_attempts against the reference's accepted/proposal counts),
 * the extremes D_min, D_max, the C++ deterministic <D>_sampled against both
 * Monte Carlo means.
 *
 * Seeds (A8 part): SplitMix64(8080 + j), as in bh_ka2008.cpp.
 */

#include "archetypes/hadronic/bh_born_qk.h" // step 6 round 2 (F9): the Born q_k provider
#include "archetypes/hadronic/bh_calibrated_test.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_sampler_host.h"
#include "archetypes/hadronic/bh_sampler_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace arch::bh;

namespace {

  // step 6 round 2 (Codex 1 F9): the loss tables of the host chain follow the BH_QK_SOURCE selector
  // (born | fit; absent = the library default, fit), as in bh_chain.cpp, so that the _born ctest
  // entry exercises the Born table construction in this consumer
  Quadrature loss_quadrature() {
    Quadrature q;
    if (const char* e = std::getenv("BH_QK_SOURCE")) {
      const std::string v(e);
      if (v == "born") {
        q.qk_source = QkSource::Born;
      } else if (v != "fit") {
        throw std::runtime_error("BH_QK_SOURCE = '" + v + "' is neither born nor fit");
      }
    }
    return q;
  }

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
  // one quantity's histogram in a fixture: the series the calibrated test needs
  struct QHist {
    std::vector<double> lo, hi, nu, nw, sw, sw2;
  };

  QHist qhist(const Table& T, int q) {
    QHist H;
    for (std::size_t r = 0; r < T.f.rows.size(); ++r) {
      if (static_cast<int>(T.at(r, "quantity_id")) != q) {
        continue;
      }
      H.lo.push_back(T.at(r, "lo"));
      H.hi.push_back(T.at(r, "hi"));
      H.nu.push_back(T.at(r, "unweighted_count"));
      H.nw.push_back(T.at(r, "weighted_packet_count"));
      H.sw.push_back(T.at(r, "sum_1_over_D"));
      H.sw2.push_back(T.at(r, "sum_1_over_D_squared"));
    }
    return H;
  }

  struct CalRow {
    std::string what;
    cal::Result r;
  };

  std::string describe(const cal::Result& r) {
    char b[256];
    if (r.insufficient) {
      std::snprintf(b, sizeof(b), "ESS_A %.0f ESS_B %.0f, 1 merged bin: INSUFFICIENT RESOLUTION (not a pass)", r.ess_a,
                    r.ess_b);
    } else {
      std::snprintf(b, sizeof(b), "ESS_A %.0f ESS_B %.0f T_AD %.4e p %.5f q99 %.4e merged bins %zu%s", r.ess_a,
                    r.ess_b, r.T_AD, r.p, r.q99, r.merged_bins, r.pass ? "" : " FAIL");
    }
    return b;
  }

  // both chains of one quantity on the common partition (all four counts >= 100); C = C++, R = reference
  void compare_quantity(const QHist& C, const QHist& R, int case_index, int q, const std::string& label,
                        std::vector<CalRow>& out, double& w_edge) {
    if (C.lo.size() != R.lo.size() || C.lo.empty()) {
      throw std::runtime_error(label + ": the two histograms have different bins");
    }
    for (std::size_t i = 0; i < C.lo.size(); ++i) {
      w_edge = std::max({ w_edge, rel(C.lo[i], R.lo[i]), rel(C.hi[i], R.hi[i]) });
    }
    const cal::Partition P = cal::merge_partition({ C.nu, R.nu, C.nw, R.nw }, 100.0);
    const auto           a_u = cal::merge_sums(P, C.nu), b_u = cal::merge_sums(P, R.nu);
    const auto           a_w = cal::merge_sums(P, C.sw), a2_w = cal::merge_sums(P, C.sw2);
    const auto           b_w = cal::merge_sums(P, R.sw), b2_w = cal::merge_sums(P, R.sw2);
    // unweighted events: the sums of weights and of squared weights both equal the count
    out.push_back({ label + " unweighted", cal::calibrated(a_u, a_u, b_u, b_u, cal::comparison_seed(case_index, 0, q)) });
    out.push_back({ label + " weighted", cal::calibrated(a_w, a2_w, b_w, b2_w, cal::comparison_seed(case_index, 1, q)) });
  }

  void test_hist() {
    std::printf("\n== A10 item 3: histograms, C++ vs reference, the calibrated weighted two-sample test "
                "(bhref-0.2.0-suppl1/COMPARISON.md) ==\n");
    const char* const   qname[5] = { "E-", "E+", "cos-", "cos+", "D" };
    std::vector<CalRow> fam, linD;
    double              w_edge = 0.0;
    int                 case_index = 0;
    for (const std::string bath : { "mono05", "powerlaw" }) {
      const std::vector<std::string> gs = bath == "mono05" ? std::vector<std::string> { "2.23125", "3", "10", "100" }
                                                           : std::vector<std::string> { "1.09375", "3", "10", "100" };
      for (const auto& gstr : gs) {
        const Table C = load("cpp-sampler-candidate/hist_" + bath + "_G" + gstr + ".txt");
        const Table R = load("bhref-0.2.0/hist_" + bath + "_G" + gstr + ".txt");
        const bool  logD = gstr == "10" || gstr == "100";
        for (int q = 0; q < 5; ++q) {
          const std::string label = bath + " Gamma_p " + gstr + " " + qname[q];
          if (q == 4 && logD) {
            // the eight log-D substitutions (supplement); the linear-D comparison is a diagnostic only
            const Table CL = load("cpp-sampler-candidate/hist_logD_" + bath + "_G" + gstr + ".txt");
            const Table RL = load("bhref-0.2.0-suppl1/hist_logD_" + bath + "_G" + gstr + ".txt");
            compare_quantity(qhist(CL, 4), qhist(RL, 4), case_index, q, label + " (log bins)", fam, w_edge);
            compare_quantity(qhist(C, 4), qhist(R, 4), case_index, q, label + " (linear bins, diagnostic)", linD,
                             w_edge);
          } else {
            compare_quantity(qhist(C, q), qhist(R, q), case_index, q, label, fam, w_edge);
          }
        }
        ++case_index;
      }
    }
    std::size_t n_pass = 0;
    std::string fails;
    for (const auto& c : fam) {
      diag("A10 hist %s: %s", c.what.c_str(), describe(c.r).c_str());
      if (c.r.pass) {
        ++n_pass;
      } else {
        fails += " " + c.what + (c.r.insufficient ? " (insufficient resolution)" : " (p " + std::to_string(c.r.p) + ")") +
                 ";";
      }
    }
    for (const auto& c : linD) {
      diag("A10 hist %s: %s (not counted)", c.what.c_str(), describe(c.r).c_str());
    }
    // diagnostic, not counted: within one file the weighted chain re-weighted by 1/D must reproduce that file's
    // unweighted chain (the weighted density is D times the unweighted one); the same calibrated test, the partition
    // from the file's own two counts; null seeds 2026102801 + 10 case + quantity (C++), + 5 (reference)
    {
      int ci = 0;
      for (const std::string bath : { "mono05", "powerlaw" }) {
        const std::vector<std::string> gs = bath == "mono05" ? std::vector<std::string> { "2.23125", "3", "10", "100" }
                                                             : std::vector<std::string> { "1.09375", "3", "10", "100" };
        for (const auto& gstr : gs) {
          for (int side = 0; side < 2; ++side) {
            const Table T = load(side == 0 ? "cpp-sampler-candidate/hist_" + bath + "_G" + gstr + ".txt"
                                           : "bhref-0.2.0/hist_" + bath + "_G" + gstr + ".txt");
            std::string line;
            for (int q = 0; q < 5; ++q) {
              const QHist          H = qhist(T, q);
              const cal::Partition P = cal::merge_partition({ H.nu, H.nw }, 100.0);
              const auto           u = cal::merge_sums(P, H.nu);
              const cal::Result    r = cal::calibrated(cal::merge_sums(P, H.sw), cal::merge_sums(P, H.sw2), u, u,
                                                       2026102801ULL + 10ULL * ci + 5ULL * side + q);
              char b[64];
              std::snprintf(b, sizeof(b), " %s p %.4f%s", qname[q], r.p, r.p < 0.01 ? " (< 0.01)" : "");
              line += b;
            }
            diag("within-file consistency (diagnostic), %s %s Gamma_p %s, weighted (1/D) vs unweighted:%s",
                 side == 0 ? "C++" : "reference", bath.c_str(), gstr.c_str(), line.c_str());
          }
          ++ci;
        }
      }
    }
    check("hist: bin edges of the C++ fixtures = the reference's (taken from them by --bins-from; the log-D edges "
          "from the supplement's serialized edges)",
          w_edge <= 1.0e-12, w_edge, 1.0e-12, "max rel. dev.");
    // the referee's replay (COMPARISON.md): mono05 Gamma_p = 3 weighted linear D, 150 merged bins, p = 0.01805
    for (const auto& c : fam) {
      if (c.what == "mono05 Gamma_p 3 D weighted") {
        check("hist: the referee's replay case (mono05 Gamma_p = 3, weighted D): merged-bin count 150 (deterministic "
              "partition); p = " + std::to_string(c.r.p) + " against the replay's 0.01805 (a different Gaussian stream: "
              "diagnostic)",
              c.r.merged_bins == 150, static_cast<double>(c.r.merged_bins), 150.0, "merged bins");
      }
    }
    const double frac = fam.empty() ? 0.0 : static_cast<double>(n_pass) / static_cast<double>(fam.size());
    check("hist: fraction of the " + std::to_string(fam.size()) +
            " comparisons (2 baths x 4 Gamma_p x 5 quantities x 2 chains; D on log bins at Gamma_p = 10 and 100) "
            "passing the calibrated test at p >= 0.01",
          !fam.empty() && frac >= 0.95, frac, 0.95, "fraction (>= 0.95 required)");
    diag("hist: %zu of %zu pass; failures:%s", n_pass, fam.size(), fails.empty() ? " none" : fails.c_str());
  }

  // ---------------------------------------------------------------------- //
  // implementation-chosen: the size of the calibrated test under its null
  void test_calibration_null() {
    std::printf("\n== the calibrated test under the null: synthetic weighted samples from one distribution ==\n");
    // 100 comparisons of two weighted samples drawn from one distribution (x exponential with unit mean,
    // binned on 60 bins of [0, 6] with the overflow in the last bin, weight w = 1/(0.5 + x) as a stand-in for
    // 1/D), sizes 20,000 and 50,000; the production function with 19,999 replicates
    const auto     t0 = std::chrono::steady_clock::now();
    std::mt19937_64 gen(20261027ULL);
    auto            unif = [&]() { return (static_cast<double>(gen() >> 11) + 0.5) * (1.0 / 9007199254740992.0); };
    const int       NB = 60, reps = 100;
    int             n_reject = 0;
    double          psum = 0.0;
    for (int r = 0; r < reps; ++r) {
      std::vector<double> an(NB, 0), aw(NB, 0), aw2(NB, 0), bn(NB, 0), bw(NB, 0), bw2(NB, 0);
      auto fill = [&](int n, std::vector<double>& cn, std::vector<double>& cw, std::vector<double>& cw2) {
        for (int i = 0; i < n; ++i) {
          const double x = -std::log(unif());
          const int    b = std::min(NB - 1, static_cast<int>(x / 0.1));
          const double w = 1.0 / (0.5 + x);
          cn[b] += 1.0;
          cw[b] += w;
          cw2[b] += w * w;
        }
      };
      fill(20000, an, aw, aw2);
      fill(50000, bn, bw, bw2);
      const cal::Partition P = cal::merge_partition({ an, bn }, 100.0);
      const cal::Result    R = cal::calibrated(cal::merge_sums(P, aw), cal::merge_sums(P, aw2), cal::merge_sums(P, bw),
                                               cal::merge_sums(P, bw2), 4000000ULL + static_cast<unsigned>(r));
      psum += R.p;
      n_reject += R.p < 0.01 ? 1 : 0;
    }
    diag("calibrated test under the null: %d comparisons, %d with p < 0.01 (expected 1; binomial 99.95%% bound 5), "
         "mean p %.3f (expected 0.5, s.e. 0.029); %.1f s",
         reps, n_reject, psum / reps, seconds_since(t0));
    check("calibration null: rejections at p < 0.01 among 100 null comparisons (binomial(100, 0.01), P(>= 6) = 5e-4)",
          n_reject <= 5, n_reject, 5.0, "rejections");
    check("calibration null: mean p of the 100 null comparisons within 0.5 +- 0.1 (+-3.4 s.e.)",
          std::abs(psum / reps - 0.5) <= 0.1, psum / reps - 0.5, 0.1, "mean p - 0.5");
  }

  // ---------------------------------------------------------------------- //
  void test_a8_headon() {
    std::printf("\n== A10 item 4: the head-on A8 KS against the reference's KA2008 CDF ==\n");
    const auto t0  = std::chrono::steady_clock::now();
    auto       T   = std::make_shared<const rf::RestFrameTables>(rf::RestFrameTables::build());
    auto       cum = std::make_shared<const rf::SamplerCumulatives>();
    diag("rest-frame tables built in %.1f s", seconds_since(t0));
    diag("loss tables of the host chain: qk_source = %s (BH_QK_SOURCE; step 6 round 2, F9)",
         loss_quadrature().qk_source == QkSource::Born ? "born-3D2000" : "chodorowski-3.7");
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
      BetheHeitlerRates rates(bath, loss_quadrature()); // step 6 round 2 (F9): BH_QK_SOURCE selects the tables
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
    if (want("calnull")) {
      test_calibration_null();
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
