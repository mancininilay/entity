/**
 * @file tests/archetypes/bh_restframe.cpp
 * @brief Pre-registered anchors of the rest-frame laws of the Bethe-Heitler
 *        pair sampler (P2 step 4, contract revision 1.0 section 5, A1 to A5),
 *        plus the implementation-chosen checks of the transcription and of
 *        the quadrature convergence.
 *
 * Contract items (tolerances registered in the contract, never adjusted here):
 *   A1  int g dE'  = sigma~ Maximon, <= 1e-3 at every eps' node in [2.001, 1000]
 *   A2  int h dcos = g, <= 1e-3 at every (eps', E') node; e-/e+ symmetry of g
 *   A3  eps' -> 2: E'- -> 1 and q_k -> 2 to 1e-3 at eps' - 2 = 1e-3; the
 *       asymptotic q_k eps'/(4 ln^2 eps') reported (leading-order limit 3/14)
 *   A4  q_k (machinery) vs Chodorowski eq. 3.7 on [2.05, 1000] <= 2%;
 *       vs phi_fit'/(kappa sigma~) <= 2%
 *   A5  phi(kappa) by int eps' sigma~ q_k (machinery) vs the step-1 fit <= 2%
 *       (the direct-boost loss table part of A5 belongs to the sampling
 *       chains, stage 2)
 *
 * Implementation-chosen checks (prefix "(impl)"): the closed form of the
 * angular law against the numerical integral of the fully differential law;
 * the normalisation of the fully differential law through its own angular
 * integral; the self-convergence of every quadrature under refinement; the
 * fixture `qk_table.txt` against the live computation when the file exists.
 * `[DIAG]` lines are reported only.
 *
 * Nodes (convention): eps' on the 449-node grid of `rf::qk_grid` (64 per
 * decade in log10(eps' - 2) from 1e-4, plus eps' = 1000); E' on the 31
 * interior nodes x_j = (1 - cos(pi j/32))/2, j = 1..31, of x = (E' - 1)/(eps' - 2)
 * (the two end points carry g = h = 0 identically).
 */

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_restframe.h"
#include "archetypes/hadronic/bh_sampler_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
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

  bool check_abs_k(Kind               k,
                   const std::string& name,
                   double             measured,
                   double             tol,
                   const std::string& what = "max |rel. dev.|") {
    const bool ok = std::isfinite(measured) && std::abs(measured) <= tol;
    std::printf("[%s] %s%s: %s = %.6e (tol %.3e)\n",
                ok ? "PASS" : "FAIL",
                k == IMPL ? "(impl) " : "",
                name.c_str(),
                what.c_str(),
                measured,
                tol);
    tally(k, ok);
    return ok;
  }

  bool check_true_k(Kind k, const std::string& name, bool ok, const std::string& what) {
    std::printf("[%s] %s%s: %s\n",
                ok ? "PASS" : "FAIL",
                k == IMPL ? "(impl) " : "",
                name.c_str(),
                what.c_str());
    tally(k, ok);
    return ok;
  }

  bool check_abs(const std::string& name,
                 double             measured,
                 double             tol,
                 const std::string& what = "max |rel. dev.|") {
    return check_abs_k(CONTRACT, name, measured, tol, what);
  }

  bool impl_abs(const std::string& name,
                double             measured,
                double             tol,
                const std::string& what = "max |rel. dev.|") {
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

  // ---------------------------------------------------------------------- //
  // Reference formulas (as in tests/archetypes/bh_physics.cpp): Chodorowski,
  // Zdziarski & Sikora 1992 phi(kappa) fits, eq. 3.14 (kappa < 25) and
  // eqs. 3.16/3.18 (kappa >= 25), coefficients of spec v3.1.3 section VII
  // ---------------------------------------------------------------------- //
  namespace chodorowski {
    constexpr double c1 = 0.8048, c2 = 0.1459, c3 = 1.137e-3, c4 = -3.879e-6;
    constexpr double d0 = -86.07, d1 = 50.96, d2 = -14.45, d3 = 8.0 / 3.0;
    constexpr double f1 = 2.910, f2 = 78.35, f3 = 1837.0;
    constexpr double junction = 25.0;

    double phi_fit(double k) {
      if (k < junction) {
        const double d = k - 2.0;
        const double D = 1.0 + d * (c1 + d * (c2 + d * (c3 + d * c4)));
        return (constants::PI / 12.0) * d * d * d * d / D;
      }
      const double L = std::log(k);
      const double N = d0 + L * (d1 + L * (d2 + L * d3));
      const double M = 1.0 - f1 / k - f2 / (k * k) - f3 / (k * k * k);
      return k * N / M;
    }

    double phi_fit_deriv(double k) {
      if (k < junction) {
        const double d  = k - 2.0;
        const double D  = 1.0 + d * (c1 + d * (c2 + d * (c3 + d * c4)));
        const double Dp = c1 + d * (2.0 * c2 + d * (3.0 * c3 + d * 4.0 * c4));
        return (constants::PI / 12.0) * (4.0 * d * d * d * D - d * d * d * d * Dp) / (D * D);
      }
      const double L  = std::log(k);
      const double N  = d0 + L * (d1 + L * (d2 + L * d3));
      const double Np = d1 + L * (2.0 * d2 + L * 3.0 * d3);
      const double M  = 1.0 - f1 / k - f2 / (k * k) - f3 / (k * k * k);
      const double Mp = f1 / (k * k) + 2.0 * f2 / (k * k * k) + 3.0 * f3 / (k * k * k * k);
      return (N + Np) / M - k * N * Mp / (M * M);
    }
  } // namespace chodorowski

  // the E' nodes in x = (E' - 1)/(eps' - 2)
  std::vector<double> x_nodes() {
    std::vector<double> x;
    for (int j = 1; j <= 31; ++j) {
      x.push_back(0.5 * (1.0 - std::cos(constants::PI * j / 32.0)));
    }
    return x;
  }

  // ---------------------------------------------------------------------- //
  // A1 and the symmetry of g
  // ---------------------------------------------------------------------- //
  void test_A1(const std::vector<double>& grid, const rf::RestFrameQuadrature& q) {
    std::printf("\n== A1: int g dE' = sigma~ Maximon ==\n");
    double worst = 0.0, worst_k = 0.0;
    double worst_all = 0.0, worst_all_k = 0.0;
    for (const double k : grid) {
      const double s   = rf::sigma_from_energy_law(k, q);
      const double dev = std::abs(s / sigma_tilde(k) - 1.0);
      if (k >= 2.001 && dev > worst) {
        worst   = dev;
        worst_k = k;
      }
      if (dev > worst_all) {
        worst_all   = dev;
        worst_all_k = k;
      }
    }
    check_abs("A1: int g dE' vs sigma~ Maximon at every eps' node in [2.001, 1000]", worst, 1.0e-3);
    diag("A1: worst node eps' = %.6f; over the whole grid (from eps' - 2 = 1e-4): %.3e at eps' = %.6f",
         worst_k, worst_all, worst_all_k);
    for (const double k : { 2.0001, 2.001, 2.5, 4.0, 10.0, 100.0, 1000.0 }) {
      diag("A1: eps' = %9.4f  int g = %.10e  sigma~ = %.10e  ratio - 1 = %+.3e",
           k, rf::sigma_from_energy_law(k, q), sigma_tilde(k),
           rf::sigma_from_energy_law(k, q) / sigma_tilde(k) - 1.0);
    }
    // self-convergence of the energy quadrature
    double conv = 0.0;
    for (const double k : { 2.0001, 2.001, 2.01, 2.5, 4.0, 10.0, 100.0, 1000.0 }) {
      const double s  = rf::sigma_from_energy_law(k, q);
      const double sr = rf::sigma_from_energy_law(k, q.refined());
      conv            = std::max(conv, std::abs(sr / s - 1.0));
    }
    impl_abs("A1 self-convergence: int g under refinement (order + 8, panels x 2), 8 nodes", conv, 1.0e-7);
    // symmetry: exactly complementary energies are bitwise equal; generic
    // energies to the rounding of eps' - E'
    bool   bitwise = true;
    double gen     = 0.0;
    for (const double k : { 3.0, 10.0, 100.0, 1000.0 }) {
      for (const double E : { 1.25, 1.5, 2.5 }) {
        if (E < k - 1.0) {
          bitwise = bitwise && (rf::dsigma_dE(k, E) == rf::dsigma_dE(k, k - E));
        }
      }
      for (const double x : x_nodes()) {
        const double E  = 1.0 + (k - 2.0) * x;
        const double a  = rf::dsigma_dE(k, E), b = rf::dsigma_dE(k, k - E);
        gen             = std::max(gen, std::abs(a / b - 1.0));
      }
    }
    check_true_k(CONTRACT, "A2: e-/e+ symmetry of g, exactly complementary energies", bitwise,
                 "g(E') == g(eps' - E') bitwise at 12 (eps', E') pairs");
    impl_abs("A2: e-/e+ symmetry of g at generic nodes (rounding of eps' - E' only)", gen, 1.0e-12);
  }

  // ---------------------------------------------------------------------- //
  // A2: the angular law integrates to g; the transcription checks
  // ---------------------------------------------------------------------- //
  void test_A2(const std::vector<double>& grid, const rf::RestFrameQuadrature& q) {
    std::printf("\n== A2: int h dcos theta' = g at every (eps', E') node ==\n");
    const auto          xs = x_nodes();
    std::vector<double> worst(grid.size(), 0.0);
    const auto          t0 = std::chrono::steady_clock::now();
    rf::parallel_for_index(grid.size(), [&](std::size_t i) {
      const double k = grid[i];
      double       w = 0.0;
      for (const double x : xs) {
        const double E = 1.0 + (k - 2.0) * x;
        const double a = rf::dsigma_dE_from_angular_law(k, E, q);
        const double g = rf::dsigma_dE(k, E);
        w              = std::max(w, std::abs(a / g - 1.0));
      }
      worst[i] = w;
    });
    double      wmax = 0.0;
    std::size_t imax = 0;
    for (std::size_t i = 0; i < grid.size(); ++i) {
      if (worst[i] > wmax) {
        wmax = worst[i];
        imax = i;
      }
    }
    check_abs("A2: int h dcos vs g at every (eps', E') node (449 x 31)", wmax, 1.0e-3);
    diag("A2: worst at eps' = %.6f; %zu angular integrals in %.1f s", grid[imax],
         grid.size() * xs.size(), seconds_since(t0));
    // self-convergence of the angular quadrature
    double conv = 0.0;
    for (const double k : { 2.001, 2.1, 3.0, 10.0, 100.0, 1000.0 }) {
      for (const double x : { 0.01, 0.5, 0.99 }) {
        const double E  = 1.0 + (k - 2.0) * x;
        const double a  = rf::dsigma_dE_from_angular_law(k, E, q);
        const double ar = rf::dsigma_dE_from_angular_law(k, E, q.refined());
        conv            = std::max(conv, std::abs(ar / a - 1.0));
      }
    }
    impl_abs("A2 self-convergence: int h dcos under refinement, 18 nodes", conv, 1.0e-8);

    // the closed form against the numerical integral of the fully
    // differential law (the transcription check of MOK 3D-2000)
    std::printf("\n== transcription: closed form (MOK 3D-2000) vs numerical integral of MOK 3D-3000 ==\n");
    double      wtr = 0.0;
    int         n_pts = 0;
    double      wtr_k = 0, wtr_x = 0, wtr_omc = 0;
    for (const double k : { 2.001, 2.01, 2.2, 3.0, 7.0, 25.0, 200.0, 1000.0 }) {
      for (const double x : { 0.01, 0.1, 0.3, 0.5, 0.77, 0.95 }) {
        const double E = 1.0 + (k - 2.0) * x;
        for (const double omc : { 1.0e-8, 1.0e-5, 1.0e-3, 0.03, 0.3, 1.0, 1.6, 1.99 }) {
          const double n = rf::dsigma_dE_dcos_numeric(k, E, omc, q.angular);
          const double c = rf::dsigma_dE_dcos_closed(k, E, omc);
          const double r = std::abs(c / n - 1.0);
          ++n_pts;
          if (!(r <= wtr)) {
            wtr     = std::isfinite(r) ? r : 1.0e300;
            wtr_k   = k;
            wtr_x   = x;
            wtr_omc = omc;
          }
        }
      }
    }
    impl_abs("transcription: closed form (MOK 3D-2000, signs fixed) vs numerical integral of law 2, " +
               std::to_string(n_pts) + " points with x in [0.01, 0.95]",
             wtr, 1.0e-6);
    diag("transcription: worst at eps' = %g, x = %g, 1 - cos = %g", wtr_k, wtr_x, wtr_omc);
    // the signs: every other combination disagrees (the same 12 points as the scratch run)
    {
      struct P {
        double k, E, omc;
      };
      const std::vector<P> pts = { { 3.0, 1.5, 0.3 },     { 3.0, 1.7, 1.2 },      { 5.0, 2.0, 0.05 },
                                   { 5.0, 3.5, 0.9 },     { 10.0, 4.0, 0.01 },    { 10.0, 8.0, 1.7 },
                                   { 30.0, 15.0, 0.002 }, { 100.0, 60.0, 1.0e-4 }, { 100.0, 20.0, 0.5 },
                                   { 1000.0, 500.0, 1.0e-6 }, { 1000.0, 900.0, 0.3 }, { 2.5, 1.2, 1.0 } };
      int  n_other_matching = 0;
      for (int c = 0; c < 16; ++c) {
        rf::ClosedFormSigns s;
        s.s0 = (c & 1) ? -1.0 : 1.0;
        s.sL = (c & 2) ? -1.0 : 1.0;
        s.se = (c & 4) ? -1.0 : 1.0;
        s.sQ = (c & 8) ? -1.0 : 1.0;
        const auto& ref = rf::closed_form_signs();
        if (s.s0 == ref.s0 && s.sL == ref.sL && s.se == ref.se && s.sQ == ref.sQ) {
          continue;
        }
        double w = 0.0;
        for (const auto& p : pts) {
          const double n = rf::dsigma_dE_dcos_numeric(p.k, p.E, p.omc, q.angular);
          const double v = rf::dsigma_dE_dcos_closed_signed(p.k, p.E, p.omc, s);
          w              = std::max(w, std::abs(v / n - 1.0));
        }
        if (w < 1.0e-3) {
          ++n_other_matching;
        }
      }
      impl_true("transcription: the sign choice is unique", n_other_matching == 0,
                "none of the 15 other sign combinations matches the numerical integral to 1e-3 at 12 points");
    }
    // the normalisation of law 2 on its own: its angular integral equals g
    {
      double w = 0.0;
      for (const double k : { 2.5, 10.0, 300.0 }) {
        const double E  = 1.0 + (k - 2.0) * 0.3;
        const double an = rf::dsigma_dE_from_angular_law_numeric(k, E, q);
        w               = std::max(w, std::abs(an / rf::dsigma_dE(k, E) - 1.0));
      }
      impl_abs("normalisation of law 2: int (numerical law) dcos vs g at 3 nodes", w, 1.0e-6);
    }
    {
      double conv = 0.0;
      for (const double k : { 2.5, 10.0, 1000.0 }) {
        for (const double omc : { 1.0e-6, 0.1, 1.5 }) {
          const double E = 1.0 + (k - 2.0) * 0.4;
          const double a = rf::dsigma_dE_dcos_numeric(k, E, omc, q.angular);
          const double b = rf::dsigma_dE_dcos_numeric(k, E, omc, q.angular.refined());
          conv           = std::max(conv, std::abs(b / a - 1.0));
        }
      }
      impl_abs("law 2 quadrature self-convergence (order + 4, panels / 2), 9 points", conv, 1.0e-8);
    }
  }

  // ---------------------------------------------------------------------- //
  // A3, A4, A5 from the moments on the grid
  // ---------------------------------------------------------------------- //
  struct GridMoments {
    std::vector<double>      grid;
    std::vector<rf::Moments> m;
  };

  GridMoments compute_moments(const std::vector<double>& grid, const rf::RestFrameQuadrature& q) {
    GridMoments G;
    G.grid = grid;
    G.m.resize(grid.size());
    const auto t0 = std::chrono::steady_clock::now();
    rf::parallel_for_index(grid.size(), [&](std::size_t i) { G.m[i] = rf::moments(grid[i], q); });
    diag("moments at %zu nodes in %.1f s (%u threads)", grid.size(), seconds_since(t0),
         std::max(1u, std::thread::hardware_concurrency()));
    return G;
  }

  // q_k(k) from the grid: linear in t = ln(k - 2) between nodes, 2 below the grid
  double qk_interp(const GridMoments& G, double k) {
    const double d = k - 2.0;
    if (!(d > 0.0) || k <= G.grid.front()) {
      return 2.0;
    }
    if (k >= G.grid.back()) {
      return G.m.back().q_k;
    }
    auto        it = std::upper_bound(G.grid.begin(), G.grid.end(), k);
    std::size_t j  = static_cast<std::size_t>(it - G.grid.begin()) - 1;
    const double t0 = std::log(G.grid[j] - 2.0), t1 = std::log(G.grid[j + 1] - 2.0);
    const double w  = (std::log(d) - t0) / (t1 - t0);
    return (1.0 - w) * G.m[j].q_k + w * G.m[j + 1].q_k;
  }

  // phi(kappa) = int_2^kappa k sigma~(k) q_k(k) dk with the machinery's q_k
  double phi_from_grid(const GridMoments& G, double kappa, const GaussLegendre& gl) {
    // integrate in t = ln(k - 2) from ln(1e-8) (threshold law below) with the
    // grid nodes as panel edges
    double       sum = 0.0;
    const double d0  = 1.0e-8;
    sum              = (constants::PI / 12.0) * d0 * d0 * d0 * d0; // phi threshold law
    std::vector<double> edges { std::log(d0) };
    for (const double k : G.grid) {
      if (k - 2.0 > d0 && k < kappa) {
        edges.push_back(std::log(k - 2.0));
      }
    }
    edges.push_back(std::log(kappa - 2.0));
    for (std::size_t s = 0; s + 1 < edges.size(); ++s) {
      const double L = edges[s + 1] - edges[s];
      if (!(L > 0.0)) {
        continue;
      }
      // subdivide long panels (below the grid)
      const int    np = std::max(1, static_cast<int>(std::ceil(L / 0.05)));
      const double hp = L / np;
      for (int p = 0; p < np; ++p) {
        const double a = edges[s] + hp * p;
        const double mid = a + 0.5 * hp, half = 0.5 * hp;
        for (int i = 0; i < gl.order(); ++i) {
          const double t     = mid + half * gl.x[i];
          const double delta = std::exp(t);
          const double k     = 2.0 + delta;
          sum += half * gl.w[i] * k * sigma_tilde_delta(delta) * qk_interp(G, k) * delta;
        }
      }
    }
    return sum;
  }

  void test_A3_A4_A5(const GridMoments& G, const rf::RestFrameQuadrature& q) {
    std::printf("\n== A3: threshold limits ==\n");
    {
      const rf::Moments m = rf::moments(2.001, q);
      check_abs("A3: q_k -> 2 at eps' - 2 = 1e-3", m.q_k - 2.0, 1.0e-3, "|q_k - 2|");
      check_abs("A3: <E'-> -> 1 at eps' - 2 = 1e-3", m.mean_E - 1.0, 1.0e-3, "|<E'-> - 1|");
      diag("A3: eps' = 2.001: q_k = %.8f (eq. 3.7: %.8f), <E'-> = %.8f, <p cos> = %.6e", m.q_k,
           q_k(2.001), m.mean_E, m.mean_pcos);
      for (const double k : { 2.0001, 2.00001 }) {
        const rf::Moments mm = rf::moments(k, q);
        diag("A3: eps' = %.6f: q_k = %.10f, 2 - q_k = %.3e, (eps' - 2) = %.1e", k, mm.q_k, 2.0 - mm.q_k, k - 2.0);
      }
    }
    {
      // asymptotic form (round 2, finding 8): with sigma~ -> (28/9) ln k (Maximon
      // eq. 12) and phi -> (8/3) k ln^3 k (the leading term of the Chodorowski
      // fit, eqs. 3.16/3.18), q_k = phi'/(k sigma~) -> (6/7) ln^2 k / k, i.e.
      // q_k k/(4 ln^2 k) -> 3/14 (leading coefficient; the subleading terms are
      // of relative order 1/ln k, so the limit is approached slowly)
      std::string s = "A3: q_k eps'/(4 ln^2 eps') at eps' = ";
      for (const double k : { 10.0, 30.0, 100.0, 300.0, 1000.0 }) {
        const rf::Moments m = rf::moments(k, q);
        char              buf[80];
        const double      L = std::log(k);
        std::snprintf(buf, sizeof(buf), "%g: %.5f; ", k, m.q_k * k / (4.0 * L * L));
        s += buf;
      }
      auto r37 = [](double k) {
        const double L = std::log(k);
        return q_k(k) * k / (4.0 * L * L);
      };
      diag("%s(leading-order limit 3/14 = %.5f; the eq. 3.7 values are %.5f, %.5f, %.5f, %.5f, %.5f)", s.c_str(),
           3.0 / 14.0, r37(10.0), r37(30.0), r37(100.0), r37(300.0), r37(1000.0));
    }

    std::printf("\n== A4: q_k machinery vs eq. 3.7 and vs phi_fit'/(kappa sigma~) on [2.05, 1000] ==\n");
    {
      double w37 = 0.0, wphi = 0.0, k37 = 0.0, kphi = 0.0;
      for (std::size_t i = 0; i < G.grid.size(); ++i) {
        const double k = G.grid[i];
        if (k < 2.05) {
          continue;
        }
        const double r37 = std::abs(G.m[i].q_k / q_k(k) - 1.0);
        const double rp  = std::abs(G.m[i].q_k / (chodorowski::phi_fit_deriv(k) / (k * sigma_tilde(k))) - 1.0);
        if (r37 > w37) {
          w37 = r37;
          k37 = k;
        }
        if (rp > wphi) {
          wphi = rp;
          kphi = k;
        }
      }
      check_abs("A4: q_k machinery vs Chodorowski eq. 3.7, eps' nodes in [2.05, 1000]", w37, 2.0e-2);
      check_abs("A4: q_k machinery vs phi_fit'/(kappa sigma~), eps' nodes in [2.05, 1000]", wphi, 2.0e-2);
      diag("A4: worst vs eq. 3.7 at eps' = %.4f, worst vs phi_fit' at eps' = %.4f", k37, kphi);
      for (const double k : { 2.05, 2.5, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 1000.0 }) {
        const rf::Moments m = rf::moments(k, q);
        diag("A4: eps' = %7.2f  q_k = %.6f  eq. 3.7 = %.6f  (ratio - 1 = %+.3e)  <p cos> = %.6f", k, m.q_k,
             q_k(k), m.q_k / q_k(k) - 1.0, m.mean_pcos);
      }
      // convergence of q_k under refinement
      double conv = 0.0;
      for (const double k : { 2.0001, 2.001, 2.1, 3.0, 10.0, 100.0, 1000.0 }) {
        const rf::Moments a = rf::moments(k, q), b = rf::moments(k, q.refined());
        conv                = std::max(conv, std::abs(b.q_k / a.q_k - 1.0));
        conv                = std::max(conv, std::abs(b.mean_pcos / a.mean_pcos - 1.0));
      }
      impl_abs("q_k and <p cos> self-convergence under refinement, 7 nodes", conv, 1.0e-7);
      // the angular-law total agrees with Maximon at every node (consistency of
      // the normalisation used in the moments)
      double wh = 0.0;
      for (const auto& m : G.m) {
        wh = std::max(wh, std::abs(m.sigma_h / m.sigma_maximon - 1.0));
      }
      impl_abs("int int h dcos dE' vs sigma~ Maximon at every node (from eps' - 2 = 1e-4)", wh, 1.0e-3);
    }

    std::printf("\n== A5: phi(kappa) by int eps' sigma~ q_k (machinery) vs the step-1 fit ==\n");
    {
      const GaussLegendre gl(16);
      double              w = 0.0, wk = 0.0;
      const int           n = 301;
      for (int i = 0; i < n; ++i) {
        const double kappa = 2.0 + 1.0e-3 * std::pow(998.0 / 1.0e-3, static_cast<double>(i) / (n - 1));
        const double p     = phi_from_grid(G, kappa, gl);
        const double r     = std::abs(p / chodorowski::phi_fit(kappa) - 1.0);
        if (r > w) {
          w  = r;
          wk = kappa;
        }
      }
      check_abs("A5: phi by int eps' sigma~ q_k (machinery) vs phi_fit, 301 kappa in [2.001, 1000]", w, 2.0e-2);
      diag("A5: worst at kappa = %.4f", wk);
      for (const double kappa : { 2.5, 3.0, 5.0, 10.0, 25.0, 100.0, 1000.0 }) {
        diag("A5: kappa = %7.2f  phi_machinery = %.8e  phi_fit = %.8e  ratio - 1 = %+.3e", kappa,
             phi_from_grid(G, kappa, gl), chodorowski::phi_fit(kappa),
             phi_from_grid(G, kappa, gl) / chodorowski::phi_fit(kappa) - 1.0);
      }
      // and against the step-1 cumulative (eq. 3.7 q_k) for the record
      const Cumulatives cum;
      double            wc = 0.0;
      for (const double kappa : { 2.5, 5.0, 25.0, 100.0, 1000.0 }) {
        wc = std::max(wc, std::abs(phi_from_grid(G, kappa, gl) / cum.phi(kappa - 2.0) - 1.0));
      }
      diag("A5: phi_machinery vs the step-1 cumulative phi (eq. 3.7 q_k), 5 kappa: max |ratio - 1| = %.3e", wc);
    }
  }

  // ---------------------------------------------------------------------- //
  // the fixture, when present
  // ---------------------------------------------------------------------- //
  void test_fixture(const GridMoments& G) {
    std::printf("\n== fixture qk_table.txt ==\n");
    const std::string path = std::string(BH_TEST_DATA_DIR) + "/cpp-sampler-candidate/qk_table.txt";
    std::ifstream     probe(path);
    if (!probe) {
      diag("fixture %s not present: nothing to compare", path.c_str());
      return;
    }
    probe.close();
    try {
      const FixtureContents c = read_fixture_file(path);
      const std::vector<std::string> cols = { "eps_prime", "q_k", "q_k_eq37", "mean_E_minus",
                                              "mean_p_cos_theta", "sigma_from_g", "sigma_maximon" };
      impl_true("fixture: columns canonical", c.columns == cols, "7 named columns");
      impl_true("fixture: node count", c.rows.size() == G.grid.size(),
                std::to_string(c.rows.size()) + " rows vs " + std::to_string(G.grid.size()) + " grid nodes");
      if (c.rows.size() == G.grid.size() && c.columns == cols) {
        double w = 0.0;
        for (std::size_t i = 0; i < G.grid.size(); ++i) {
          const auto& r = c.rows[i];
          const auto& m = G.m[i];
          w             = std::max(w, std::abs(r[0] / G.grid[i] - 1.0));
          w             = std::max(w, std::abs(r[1] / m.q_k - 1.0));
          w             = std::max(w, std::abs(r[2] / q_k(G.grid[i]) - 1.0));
          w             = std::max(w, std::abs(r[3] / m.mean_E - 1.0));
          w             = std::max(w, std::abs(r[4] / m.mean_pcos - 1.0));
          w             = std::max(w, std::abs(r[5] / m.sigma_g - 1.0));
          w             = std::max(w, std::abs(r[6] / m.sigma_maximon - 1.0));
        }
        impl_abs("fixture: every column vs the live computation (printed at %.12e)", w, 1.0e-10);
      }
      diag("fixture: producer %s, version %s, data_sha256 %s", c.header.at("producer").c_str(),
           c.header.at("version").c_str(), c.data_sha256.c_str());
    } catch (const std::exception& e) {
      impl_true("fixture: readable", false, e.what());
    }
  }

} // namespace

auto main() -> int {
  try {
    const auto                t_start = std::chrono::steady_clock::now();
    const rf::RestFrameQuadrature q;
    const std::vector<double> grid = rf::qk_grid();
    std::printf("bh_restframe: %zu eps' nodes (first %.6f, last regular %.6f, cap %.1f); quadrature gl %d, "
                "energy panels %d, h_t %.2f; angular gl %d, max panels %.2f / %.2f\n",
                grid.size(), grid.front(), grid[grid.size() - 2], grid.back(), q.gl_order, q.energy_panels,
                q.h_t_angle, q.angular.gl_order, q.angular.max_panel_u, q.angular.max_panel_psi);
    test_A1(grid, q);
    test_A2(grid, q);
    const GridMoments G = compute_moments(grid, q);
    test_A3_A4_A5(G, q);
    test_fixture(G);
    std::printf("\nbh_restframe summary: contract items %d passed, %d failed; implementation-chosen checks "
                "%d passed, %d failed; %.1f s\n",
                g_pass_contract, g_fail_contract, g_pass_impl, g_fail_impl, seconds_since(t_start));
    return (g_fail_contract + g_fail_impl) == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_restframe: exception: %s\n", e.what());
    return 2;
  }
}
