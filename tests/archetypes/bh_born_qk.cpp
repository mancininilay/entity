/**
 * @file tests/archetypes/bh_born_qk.cpp
 * @brief Step 6 (table contract revision 1.5): the q_k^Born provider.
 *   A10b  the C++ direct quadrature at the 449 nodes of the contract grid
 *         against the frozen bhref 0.3.0 `qk_machinery` column
 *         (tests/data/hadronic/bhref-0.3.0/qk_table.txt): <= 1e-8
 *         relative at every node; the table's own node values (the dense grid
 *         contains the contract grid) likewise.
 *   A10c  the interpolant against the direct quadrature at the 448 midpoints
 *         of the contract grid (the release's `qk_midpoints.txt` eps_prime
 *         column): <= 1e-8 relative; the difference to the release's
 *         `q_direct` there is a diagnostic.
 *   A3    exact limits, re-registered on the slope form (Nilay, 2026-10-30,
 *         "yes to all"; Codex 1 review F1): |q_k^Born(2 + delta) - (2 -
 *         delta/4)| <= 1e-10 at delta = 1e-9, direct quadrature and provider
 *         (the Born law has q_k(2 + delta) = 2 - delta/4 + O(delta^2), so the
 *         brief's constant-limit form |q_k(2 + 1e-9) - 2| <= 1e-10 was
 *         unreachable by any correct provider: that row is printed as history,
 *         no verdict); the provider's continuation slope against 1/4
 *         (implementation check); the asymptotic ratio q_k eps'/(4 ln^2 eps')
 *         at eps' = 1e3 reported against 3/14 (diagnostic).
 *   Fit-path regression: the cumulatives with qk_source = fit reproduce the
 *         step-1 values (the Born switch did not touch that path): spot values
 *         of Phi, phi, psi at the default quadrature against the same
 *         computation with a standalone copy of the step-1 integrand, 1e-15.
 *   Header keys by name (contract 1.5.1 (ii), freeze 2026-10-30): the five
 *         frozen moments fixtures carry `qk_source` at header position 9, the
 *         rate tables after `status`; both readers take keys by name, so the
 *         position is immaterial: checked on the frozen files and on a rate
 *         table whose `qk_source` line is moved to the end of the header.
 *   Startup cost of the Born table printed (seconds).
 * Host-only, no Kokkos.
 */

#include "archetypes/hadronic/bh_born_qk.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_sampler_io.h"
#include "archetypes/hadronic/bh_table_io.h"

#include <fstream>
#include <sstream>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace arch::bh;

namespace {

  int n_pass = 0, n_fail = 0;

  void verdict(bool ok, const std::string& what, double value, double tol) {
    std::printf("%s: %s : %.6e (tolerance %.3e)\n", ok ? "PASS" : "FAIL", what.c_str(), value, tol);
    (ok ? n_pass : n_fail) += 1;
  }

  void diag(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    std::printf("(diag) ");
    std::vprintf(f, ap);
    std::printf("\n");
    va_end(ap);
  }

  FixtureContents load(const std::string& name) {
    return read_fixture_file(std::string(BH_TEST_DATA_DIR) + "/bhref-0.3.0/" + name);
  }

  std::size_t col(const FixtureContents& F, const std::string& name) {
    for (std::size_t c = 0; c < F.columns.size(); ++c) {
      if (F.columns[c] == name) {
        return c;
      }
    }
    throw std::runtime_error("column " + name + " not found");
  }

} // namespace

int main() {
  try {
    // the table (built on first use; its cost is the startup cost of the Born path)
    const auto t0 = std::chrono::steady_clock::now();
    const auto& BT = born::table();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("bh_born_qk: Born q_k table: %zu nodes (%d per decade of eps' - 2 from %.1e, last node k = %.10g), "
                "built in %.3f s (measured here %.3f s wall, host threads), threshold slope (2 - q)/delta at the first "
                "node %.10g\n",
                BT.size(), born::NODES_PER_DECADE, born::DELTA_MIN, BT.nodes_k().back(), BT.build_seconds(), wall,
                BT.threshold_slope());

    // ---- A10b: direct quadrature at the frozen release's 449 nodes vs its qk_machinery
    {
      const auto   F  = load("qk_table.txt");
      const auto   ik = col(F, "eps_prime"), iq = col(F, "qk_machinery"), i37 = col(F, "qk_eq3_7");
      double       worst = 0.0, k_worst = 0.0, worst_tab = 0.0, k_worst_tab = 0.0, worst_fit = 0.0;
      std::vector<double> ks, q_direct;
      for (const auto& r : F.rows) {
        ks.push_back(r[ik]);
      }
      q_direct.assign(ks.size(), 0.0);
      const auto t1 = std::chrono::steady_clock::now();
      rf::parallel_for_index(ks.size(), [&](std::size_t i) { q_direct[i] = born::direct(ks[i]); });
      const double t_direct = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
      for (std::size_t i = 0; i < F.rows.size(); ++i) {
        const double ref = F.rows[i][iq];
        const double e   = std::abs(q_direct[i] / ref - 1.0);
        if (e > worst) {
          worst   = e;
          k_worst = ks[i];
        }
        const double et = std::abs(BT.q_k(ks[i]) / ref - 1.0);
        if (et > worst_tab) {
          worst_tab   = et;
          k_worst_tab = ks[i];
        }
        // the fit column of the release against the C++ fit (consistency of the anchor)
        worst_fit = std::max(worst_fit, std::abs(q_k(ks[i]) / F.rows[i][i37] - 1.0));
      }
      verdict(worst <= 1.0e-8, "A10b: direct q_k^Born at the 449 contract nodes vs the frozen 0.3.0 qk_machinery, max relative",
              worst, 1.0e-8);
      diag("A10b: worst node eps' = %.10g; %zu direct evaluations in %.3f s (%.2f ms each, threads)", k_worst, ks.size(),
           t_direct, 1e3 * t_direct / ks.size());
      verdict(worst_tab <= 1.0e-8, "A10b: the table's interpolant at the 449 contract nodes vs the frozen 0.3.0 table, max relative",
              worst_tab, 1.0e-8);
      diag("A10b: worst interpolant node eps' = %.10g; the release's eq. 3.7 column vs the C++ fit: max %.3e", k_worst_tab,
           worst_fit);
    }

    // ---- A10c: interpolant vs direct at the 448 midpoints
    {
      const auto F  = load("qk_midpoints.txt");
      const auto ik = col(F, "eps_prime"), id = col(F, "q_direct");
      std::vector<double> ks, q_direct;
      for (const auto& r : F.rows) {
        ks.push_back(r[ik]);
      }
      q_direct.assign(ks.size(), 0.0);
      rf::parallel_for_index(ks.size(), [&](std::size_t i) { q_direct[i] = born::direct(ks[i]); });
      double worst = 0.0, k_worst = 0.0, worst_cand = 0.0;
      for (std::size_t i = 0; i < ks.size(); ++i) {
        const double e = std::abs(BT.q_k(ks[i]) / q_direct[i] - 1.0);
        if (e > worst) {
          worst   = e;
          k_worst = ks[i];
        }
        worst_cand = std::max(worst_cand, std::abs(q_direct[i] / F.rows[i][id] - 1.0));
      }
      verdict(worst <= 1.0e-8, "A10c: interpolant vs direct quadrature at the 448 contract midpoints, max relative", worst,
              1.0e-8);
      diag("A10c: worst midpoint eps' = %.10g; C++ direct vs the release's q_direct at the midpoints: max %.3e (diagnostic)",
           k_worst, worst_cand);
    }

    // ---- header keys by name, any position (contract 1.5.1 (ii); frozen fixtures of 2026-10-30)
    {
      const std::string dir = std::string(BH_TEST_DATA_DIR) + "/bhref-0.3.0/";
      // the moments fixture: qk_source is the 9th header line of the file (fact of the release) and the
      // reader returns it by name
      std::ifstream in(dir + "moments_mono05.txt");
      std::string   line, text;
      int           pos_qk = 0, n = 0;
      while (std::getline(in, line)) {
        text += line + "\n";
        if (!line.empty() && line[0] == '#') {
          ++n;
          if (line.rfind("# qk_source:", 0) == 0) {
            pos_qk = n;
          }
        }
      }
      const auto M = load("moments_mono05.txt");
      verdict(pos_qk == 9 && M.header.count("qk_source") == 1 && M.header.at("qk_source") == "\"born-3D2000\"",
              "frozen moments_mono05: qk_source at header position 9 (found " + std::to_string(pos_qk) +
                "), read by name as born-3D2000; status " + (M.header.count("status") ? M.header.at("status") : "?"),
              static_cast<double>(pos_qk), 9.0);
      // a rate table with its qk_source line moved to the end of the header parses to the same result
      std::ifstream tin(dir + "mono05.txt");
      std::string   ttext, qk_line, moved;
      while (std::getline(tin, line)) {
        ttext += line + "\n";
      }
      {
        std::istringstream ls(ttext);
        std::string        header_rest, data;
        bool               in_header = true;
        while (std::getline(ls, line)) {
          if (in_header && !line.empty() && line[0] == '#') {
            if (line.rfind("# qk_source:", 0) == 0) {
              qk_line = line;
            } else {
              header_rest += line + "\n";
            }
          } else {
            in_header = false;
            data += line + "\n";
          }
        }
        moved = header_rest + qk_line + "\n" + data;
      }
      const auto A = parse_table_file(ttext, "mono05 as frozen");
      const auto B = parse_table_file(moved, "mono05 with qk_source last");
      verdict(!qk_line.empty() && A.qk_source == "born-3D2000" && B.qk_source == "born-3D2000" && A.status == "frozen:0.3.0" &&
                B.status == A.status && B.data_sha256 == A.data_sha256 && B.gamma == A.gamma && B.I == A.I,
              "frozen mono05: qk_source after status and moved to the last header line read identically (born-3D2000, "
              "frozen:0.3.0, same data)",
              1.0, 1.0);
    }

    // ---- A3: the exact limits (re-registered on the slope form, 2026-10-30)
    {
      const double d     = 1.0e-9;
      const double qd    = born::direct(2.0 + d);
      const double qt    = BT.q_k(2.0 + d);
      const double q_lim = 2.0 - d / 4.0; // the Born threshold law q_k = 2 - delta/4 + O(delta^2)
      verdict(std::abs(qd - q_lim) <= 1.0e-10,
              "A3 [re-registered, decision 2026-10-30]: |q_k^Born(2 + 1e-9) - (2 - 1e-9/4)| <= 1e-10 by direct quadrature",
              std::abs(qd - q_lim), 1.0e-10);
      verdict(std::abs(qt - q_lim) <= 1.0e-10,
              "A3 [re-registered, decision 2026-10-30]: |q_k^Born(2 + 1e-9) - (2 - 1e-9/4)| <= 1e-10 by the provider",
              std::abs(qt - q_lim), 1.0e-10);
      // history (brief, stage 1; superseded 2026-10-30): the constant-limit form |q_k(2 + 1e-9) - 2| <= 1e-10,
      // which no correct provider can meet (the exact value is 2 - 2.5e-10); reported, no verdict
      diag("A3 [history, superseded 2026-10-30]: |q_k^Born(2 + 1e-9) - 2| = %.4e (direct), %.4e (provider) vs the brief's "
           "1e-10: the constant-limit row, failing by the physical slope 1/4, kept as history",
           std::abs(qd - 2.0), std::abs(qt - 2.0));
      // the Born threshold slope from the dense nodes: (2 - q)/delta at the first nodes
      const auto& kn = BT.nodes_k();
      const auto& qn = BT.nodes_q();
      diag("A3: (2 - q_k)/delta at delta = %.2e, %.2e, %.2e: %.10g, %.10g, %.10g (the exact threshold slope is 1/4); direct "
           "at 2 + 1e-9: 2 - q = %.4e, provider: %.4e",
           kn[0] - 2.0, kn[1] - 2.0, kn[2] - 2.0, (2.0 - qn[0]) / (kn[0] - 2.0), (2.0 - qn[1]) / (kn[1] - 2.0),
           (2.0 - qn[2]) / (kn[2] - 2.0), 2.0 - qd, 2.0 - qt);
      // implementation check: the provider's threshold continuation slope (the secant from the exact threshold
      // q_k(2) = 2 to the first node) against the exact 1/4; the O(delta) correction at the first node
      // (delta = 1e-5) is the expected size of the difference
      verdict(std::abs(BT.threshold_slope() - 0.25) <= 1.0e-6,
              "A3 [implementation check]: the provider's continuation slope s (secant threshold -> first node) vs the exact "
              "1/4, |s - 1/4|",
              std::abs(BT.threshold_slope() - 0.25), 1.0e-6);
      const double L = std::log(1000.0);
      diag("A3: asymptotic ratio q_k eps'/(4 ln^2 eps') at eps' = 1e3: %.6f (leading-order limit 3/14 = %.6f; eq. 3.7 gives %.6f)",
           BT.q_k(1000.0) * 1000.0 / (4.0 * L * L), 3.0 / 14.0, q_k(1000.0) * 1000.0 / (4.0 * L * L));
    }

    // ---- the fit path is untouched: the cumulatives with QkSource::Fit against a standalone
    //      re-integration with the step-1 integrand (same grid and Gauss rule), spot deltas
    {
      Quadrature q; // default: qk_source = Fit
      const Cumulatives C(q);
      const GaussLegendre gl(q.gl_order);
      auto standalone = [&](double delta) {
        // the step-1 integrand k sigma q_fit delta in t over [ln delta0, ln delta] on the same cells
        const double t0 = std::log(q.delta0), t1 = std::log(constants::K_JUNCTION - 2.0), t2 = std::log(delta);
        const auto   n1 = static_cast<std::size_t>(std::ceil((t1 - t0) / q.h_log_delta));
        std::vector<double> edges;
        for (std::size_t i = 0; i < n1; ++i) {
          edges.push_back(t0 + (t1 - t0) * static_cast<double>(i) / static_cast<double>(n1));
        }
        edges.push_back(t1);
        if (t2 > t1) {
          const double tk = std::log(q.k_max - 2.0);
          const auto   n2 = static_cast<std::size_t>(std::ceil((tk - t1) / q.h_log_delta));
          for (std::size_t i = 1; i < n2; ++i) {
            const double e = t1 + (tk - t1) * static_cast<double>(i) / static_cast<double>(n2);
            if (e < t2) {
              edges.push_back(e);
            }
          }
        } else {
          while (!edges.empty() && edges.back() > t2) {
            edges.pop_back();
          }
        }
        edges.push_back(t2);
        Cumulatives::Values v = Cumulatives::analytic(q.delta0);
        for (std::size_t j = 0; j + 1 < edges.size(); ++j) {
          const double mid = 0.5 * (edges[j] + edges[j + 1]), half = 0.5 * (edges[j + 1] - edges[j]);
          for (int i = 0; i < gl.order(); ++i) {
            const double t  = mid + half * gl.x[i];
            const double dl = std::exp(t), k = 2.0 + dl, s = sigma_tilde_delta(dl), qq = q_k_delta(dl);
            const double w  = half * gl.w[i];
            v.Phi += w * k * s * dl;
            v.phi += w * k * s * qq * dl;
            v.psi += w * s * qq * dl;
          }
        }
        return v;
      };
      double worst = 0.0;
      for (const double delta : { 1.0e-3, 0.5, 1.9, 7.0, 50.0, 400.0 }) {
        const auto a = C(delta), b = standalone(delta);
        worst = std::max({ worst, std::abs(a.Phi / b.Phi - 1.0), std::abs(a.phi / b.phi - 1.0), std::abs(a.psi / b.psi - 1.0) });
      }
      verdict(worst <= 1.0e-12, "fit path: Cumulatives(QkSource::Fit) vs a standalone step-1 re-integration (Phi, phi, psi at six deltas), max relative",
              worst, 1.0e-12);
      // and the Born cumulatives differ from the fit ones by the fit budget, not more
      Quadrature qb;
      qb.qk_source = QkSource::Born;
      const Cumulatives B(qb);
      double worst_diff = 0.0, k_w = 0.0;
      for (const double delta : { 0.05, 0.5, 1.9, 7.0, 50.0, 400.0, 998.0 }) {
        const double r = std::abs(B.phi(delta) / C.phi(delta) - 1.0);
        if (r > worst_diff) {
          worst_diff = r;
          k_w        = 2.0 + delta;
        }
      }
      diag("Born vs fit cumulatives phi: max |phi_Born/phi_fit - 1| = %.4e at k = %.4g over seven k (the fit budget of A4 is 2%%; "
           "Phi identical by construction)",
           worst_diff, k_w);
      verdict(B.Phi(50.0) == C.Phi(50.0), "Born and fit cumulatives share Phi exactly (q_k enters phi and psi only)",
              std::abs(B.Phi(50.0) - C.Phi(50.0)), 0.0);
    }
  } catch (const std::exception& e) {
    std::printf("FAIL: exception: %s\n", e.what());
    ++n_fail;
  }
  std::printf("bh_born_qk summary: %d passed, %d failed\n", n_pass, n_fail);
  return n_fail > 0 ? 1 : 0;
}
