/**
 * @file tests/archetypes/bh_physics.cpp
 * @brief Acceptance tests of the Bethe-Heitler startup tables
 *        (contract revision 1.3, section 5, items 1 to 6, plus the
 *        self-convergence requirement of section 4 and the interpolation
 *        target T3 of the technical specification).
 *
 * Every registered check prints `[PASS]` or `[FAIL]` with the measured value
 * and the pre-registered tolerance; `[DIAG]` lines are diagnostics that are
 * reported but never decide the outcome. The executable returns non-zero if
 * any registered check fails. No Python, no external data: the external
 * anchors are transcribed below with their source stated.
 */

#include "archetypes/hadronic/bh_physics.h"

#include "archetypes/hadronic/bh_options.h"
#include "archetypes/hadronic/bh_table_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace arch::bh;

namespace {

  // registered contract items (section 5 and section 4 of the contract, spec
  // T3) are counted separately from implementation-chosen checks
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

  // contract items
  bool check_abs(const std::string& name,
                 double             measured,
                 double             tol,
                 const std::string& what = "max |rel. dev.|") {
    return check_abs_k(CONTRACT, name, measured, tol, what);
  }

  bool check_true(const std::string& name, bool ok, const std::string& what) {
    return check_true_k(CONTRACT, name, ok, what);
  }

  // implementation-chosen checks
  bool impl_abs(const std::string& name,
                double             measured,
                double             tol,
                const std::string& what = "max |rel. dev.|") {
    return check_abs_k(IMPL, name, measured, tol, what);
  }

  bool impl_true(const std::string& name, bool ok, const std::string& what) {
    return check_true_k(IMPL, name, ok, what);
  }

  // does f() throw? (for the negative tests)
  template <class F>
  bool throws(F&& f, std::string* msg = nullptr) {
    try {
      f();
    } catch (const std::exception& e) {
      if (msg) {
        *msg = e.what();
      }
      return true;
    }
    return false;
  }

  void diag(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

  void diag(const char* fmt, ...) {
    std::printf("[DIAG] ");
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    va_end(ap);
    std::printf("\n");
  }

  // ---------------------------------------------------------------------- //
  // Reference formulas transcribed from the literature (sources stated)
  // ---------------------------------------------------------------------- //

  // Chodorowski, Zdziarski & Sikora 1992, ApJ 400, 181: phi(kappa) fits
  // (coefficients as listed in technical spec v3.1.3 section VII; functional
  // forms eq. 3.14 for kappa < 25 and eqs. 3.16/3.18 for kappa >= 25)
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
        return (constants::PI / 12.0) *
               (4.0 * d * d * d * D - d * d * d * d * Dp) / (D * D);
      }
      const double L  = std::log(k);
      const double N  = d0 + L * (d1 + L * (d2 + L * d3));
      const double Np = d1 + L * (2.0 * d2 + L * 3.0 * d3);
      const double M  = 1.0 - f1 / k - f2 / (k * k) - f3 / (k * k * k);
      const double Mp = f1 / (k * k) + 2.0 * f2 / (k * k * k) +
                        3.0 * f3 / (k * k * k * k);
      return (N + Np) / M - k * N * Mp / (M * M);
    }
  } // namespace chodorowski

  // Maximon 1968, J. Res. NBS 72B, 79: Table 1 at the original MeV
  // coordinates, converted with m_e c^2 = 0.511 MeV (contract 5.1)
  struct Anchor {
    double mev, sigma;
  };

  const std::vector<Anchor> maximon_table1 = {
    {   2.5, 0.5836 },
    {   3.0, 0.8716 },
    {   4.0,  1.416 },
    {   5.0,  1.900 },
    {  10.0,  3.647 },
    {  20.0,  5.618 },
    {  50.0,  8.377 },
    { 100.0, 10.511 }
  };

  // study-guide K(eps') values (two-digit roundings, diagnostics only)
  const std::vector<std::pair<double, double>> study_guide_K = {
    {   2.05,  1.99 },
    {    2.5,  1.88 },
    {    3.0,  1.77 },
    {    5.0,  1.41 },
    {   10.0,  0.97 },
    {   20.0,  0.64 },
    {   50.0,  0.36 },
    {  100.0,  0.23 },
    { 1000.0, 0.044 }
  };

  // CRPropa ElectronPairProduction loss rate on the CMB, (1/gamma) dgamma/dx
  // in 1/Mpc versus log10(gamma): six rows of the distributed file
  // lossrate_CMB.txt, read from the fixture
  // tests/data/hadronic/anchors/crpropa-lossrate-CMB-6rows.txt, which carries
  // the provenance (distributed-file SHA-256, CRPropa3-data commit, verified
  // by Codex 1 on 2026-10-24). No expected value is regenerated here.
  struct CRPropaRow {
    double log10_gamma, rate_per_mpc;
  };

  std::vector<CRPropaRow> read_crpropa_fixture() {
    const std::string path = std::string(BH_TEST_DATA_DIR) +
                             "/anchors/crpropa-lossrate-CMB-6rows.txt";
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot open the CRPropa fixture " + path);
    }
    std::vector<CRPropaRow> rows;
    std::string             line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::istringstream ss(line);
      std::string        a, b;
      double             lg = 0.0, rate = 0.0;
      if (!(ss >> a >> b) || !parse_number_strict(a, lg) ||
          !parse_number_strict(b, rate)) {
        throw std::runtime_error("malformed row in the CRPropa fixture: " + line);
      }
      rows.push_back({ lg, rate });
    }
    return rows;
  }

  // conversion constants of CRPropa3-data units.py (CODATA 2006):
  constexpr double crpropa_theta_cmb = 4.5961675847e-10; // k T / m_e c^2
  constexpr double crpropa_n_ph      = 4.1071773549e+08; // m^-3, unclipped CMB
  constexpr double crpropa_a = 9.7380184875e-13; // alpha r_e^2 (m_e/m_p) Mpc, m^3

  // ---------------------------------------------------------------------- //
  // Helpers
  // ---------------------------------------------------------------------- //
  struct Case {
    std::string                       name;
    std::shared_ptr<const PhotonBath> bath;
    double                            mu_r;
  };

  // composite Gauss-Legendre in ln(eps) with breakpoints
  double integrate_log(const std::function<double(double)>& f,
                       double                               a,
                       double                               b,
                       std::vector<double>                  breaks,
                       double                               h = 0.005) {
    static const GaussLegendre gl { 16 };
    std::vector<double>        edges { std::log(a) };
    for (const double x : breaks) {
      if (x > a && x < b) {
        edges.push_back(std::log(x));
      }
    }
    edges.push_back(std::log(b));
    std::sort(edges.begin(), edges.end());
    double sum = 0.0;
    for (std::size_t s = 0; s + 1 < edges.size(); ++s) {
      const double L    = edges[s + 1] - edges[s];
      const int    npan = std::max(1, static_cast<int>(std::ceil(L / h)));
      const double hp   = L / npan;
      for (int p = 0; p < npan; ++p) {
        const double mid  = edges[s] + hp * (p + 0.5);
        const double half = 0.5 * hp;
        for (int i = 0; i < gl.order(); ++i) {
          const double u  = mid + half * gl.x[i];
          const double e  = std::exp(u);
          sum            += half * gl.w[i] * f(e) * e;
        }
      }
    }
    return sum;
  }

  // Chodorowski eq. 3.11 with the phi fits: I_UR = (1/2Gamma) int n/eps^2 phi(2 Gamma eps)
  double I_ur_fit(const PhotonBath& bath, double gamma) {
    if (bath.is_mono()) {
      const double e0 = bath.eps0();
      return chodorowski::phi_fit(2.0 * gamma * e0) / (2.0 * gamma * e0 * e0);
    }
    const double a = std::max(bath.eps_min(), 1.0 / gamma);
    const double b = bath.eps_max();
    if (!(b > a)) {
      return 0.0;
    }
    std::vector<double> br = bath.breakpoints();
    br.push_back(2.0 / gamma);                           // eps' = 4
    br.push_back(chodorowski::junction / (2.0 * gamma)); // fit junction
    const double S = integrate_log(
      [&](double e) {
        return bath.density(e) / (e * e) * chodorowski::phi_fit(2.0 * gamma * e);
      },
      a,
      b,
      br);
    return S / (2.0 * gamma);
  }

  // least-squares slope of ln(v) against ln(x)
  double lsq_slope(const std::vector<double>& x, const std::vector<double>& v) {
    double       sx = 0, sy = 0, sxx = 0, sxy = 0;
    const double n = static_cast<double>(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double lx = std::log(x[i]), ly = std::log(v[i]);
      sx  += lx;
      sy  += ly;
      sxx += lx * lx;
      sxy += lx * ly;
    }
    return (n * sxy - sx * sy) / (n * sxx - sx * sx);
  }

  // the 33 asymptotic nodes of contract 5.4: (Gamma - Gamma_th)/(Gamma_th - 1) in [1e-5, 1e-4]
  std::vector<double> asymptotic_offsets(const PhotonBath& bath) {
    std::vector<double> x;
    const double        scale = bath.gamma_th() - 1.0;
    for (int i = 0; i < 33; ++i) {
      x.push_back(scale * std::pow(10.0, -5.0 + i / 32.0));
    }
    return x;
  }

  double seconds_since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  // ---------------------------------------------------------------------- //
  // Item 1: sigma against Maximon Table 1
  // ---------------------------------------------------------------------- //
  void test_maximon() {
    std::printf("\n== item 1: sigma~ against Maximon 1968 Table 1 (m_e c^2 = "
                "0.511 MeV) ==\n");
    double worst = 0.0;
    for (const auto& a : maximon_table1) {
      const double k   = a.mev / 0.511;
      const double s   = sigma_tilde(k);
      const double rel = s / a.sigma - 1.0;
      worst            = std::max(worst, std::abs(rel));
      char name[64];
      std::snprintf(name, sizeof(name), "maximon k=%.4f (%.1f MeV)", k, a.mev);
      check_abs(name, rel, 1.0e-3, "sigma/table - 1");
      diag("  sigma~(%.6f) = %.6f, table %.4f", k, s, a.sigma);
    }
    diag("worst Maximon anchor deviation %.3e", worst);
    const double s10 = sigma_tilde_series10_delta(2.0);
    const double s12 = sigma_tilde_series12(4.0);
    diag("junction k = 4: eq.10 -> %.10f, eq.12 -> %.10f, relative step %.4e "
         "(paper: 1.5e-4)",
         s10,
         s12,
         s12 / s10 - 1.0);
    for (const double d : { 1.0e-6, 1.0e-4, 1.0e-2 }) {
      diag("threshold law: sigma~(2+%g)/((pi/12) d^3) = %.8f",
           d,
           sigma_tilde_delta(d) / ((constants::PI / 12.0) * d * d * d));
    }
  }

  // ---------------------------------------------------------------------- //
  // Item 2: phi(kappa) by quadrature against the Chodorowski fits
  // ---------------------------------------------------------------------- //
  void test_phi(const Cumulatives& cum) {
    std::printf("\n== item 2: phi(kappa) quadrature vs Chodorowski fits (1%% "
                "fit budget) ==\n");
    double    worst = 0.0, worst_k = 0.0;
    const int n = 300;
    for (int i = 0; i <= n; ++i) {
      const double d = std::pow(10.0, -3.0 + (std::log10(998.0) + 3.0) * i / n);
      const double k = 2.0 + d;
      const double q = cum.phi(d);
      const double f = chodorowski::phi_fit(k);
      const double rel = q / f - 1.0;
      if (std::abs(rel) > worst) {
        worst   = std::abs(rel);
        worst_k = k;
      }
    }
    check_abs("phi quadrature vs fits, kappa in [2.001, 1000]", worst, 1.0e-2);
    diag("worst phi deviation at kappa = %.4f", worst_k);
    for (const double k :
         { 2.5, 3.0, 5.0, 10.0, 24.999999, 25.000001, 50.0, 100.0, 1000.0 }) {
      diag("  phi(%.6f): quadrature %.8e, fit %.8e, ratio-1 = %+.3e",
           k,
           cum.phi(k - 2.0),
           chodorowski::phi_fit(k),
           cum.phi(k - 2.0) / chodorowski::phi_fit(k) - 1.0);
    }
    diag("fit junction continuity at 25: phi_fit(25-)/phi_fit(25+) - 1 = %+.3e",
         chodorowski::phi_fit(24.999999999) / chodorowski::phi_fit(25.0) - 1.0);
    for (const double d : { 1.0e-6, 1.0e-4, 1.0e-2 }) {
      const double d4 = d * d * d * d;
      diag("threshold laws at kappa-2 = %g: phi/((pi/12) d^4) = %.8f, "
           "Phi/((pi/24) d^4) = %.8f, psi/((pi/24) d^4) = %.8f",
           d,
           cum.phi(d) / ((constants::PI / 12.0) * d4),
           cum.Phi(d) / ((constants::PI / 24.0) * d4),
           cum.psi(d) / ((constants::PI / 24.0) * d4));
    }
    impl_abs("phi threshold law at kappa-2 = 1e-4",
             cum.phi(1.0e-4) / ((constants::PI / 12.0) * 1.0e-16) - 1.0,
             1.0e-3,
             "phi/((pi/12)(kappa-2)^4) - 1");
  }

  // ---------------------------------------------------------------------- //
  // Item 3: q_k eq. 3.7 against phi_fit'/(kappa sigma)
  // ---------------------------------------------------------------------- //
  void test_qk() {
    std::printf("\n== item 3: q_k (eq. 3.7) vs phi_fit'/(kappa sigma~) ==\n");
    double              worst = 0.0, worst_k = 0.0;
    std::vector<double> ks;
    const int           n = 400;
    for (int i = 0; i <= n; ++i) {
      ks.push_back(2.0 + std::pow(10.0, -3.0 + (std::log10(998.0) + 3.0) * i / n));
    }
    ks.push_back(25.0 * (1.0 - 1.0e-9));
    ks.push_back(25.0 * (1.0 + 1.0e-9));
    for (const double k : ks) {
      const double deriv = chodorowski::phi_fit_deriv(k) / (k * sigma_tilde(k));
      const double rel   = q_k(k) / deriv - 1.0;
      if (std::abs(rel) > worst) {
        worst   = std::abs(rel);
        worst_k = k;
      }
    }
    check_abs("q_k eq.3.7 vs phi_fit'/(kappa sigma), kappa in (2, 1000]",
              worst,
              1.0e-2);
    diag("worst q_k deviation at kappa = %.6f (contract: 0.77%% at the "
         "one-sided derivative at 25)",
         worst_k);
    diag("  at 25-: %+.3e, at 25+: %+.3e",
         q_k(25.0 * (1.0 - 1.0e-9)) /
             (chodorowski::phi_fit_deriv(25.0 * (1.0 - 1.0e-9)) /
              (25.0 * sigma_tilde(25.0))) -
           1.0,
         q_k(25.0 * (1.0 + 1.0e-9)) /
             (chodorowski::phi_fit_deriv(25.0 * (1.0 + 1.0e-9)) /
              (25.0 * sigma_tilde(25.0))) -
           1.0);
    for (const auto& [k, K] : study_guide_K) {
      const double q    = q_k(k);
      const double half = (K < 0.1) ? 0.0005 : 0.005;
      diag("study guide K(%.2f) = %.3f (rounding interval +-%.4f): q_k = %.5f "
           "-> %s",
           k,
           K,
           half,
           q,
           std::abs(q - K) <= half ? "inside" : "OUTSIDE");
    }
  }

  // ---------------------------------------------------------------------- //
  // Items 4 and 6a: threshold exponents and <D> -> 2 Gamma beta
  // ---------------------------------------------------------------------- //
  void test_threshold(const Case&               c,
                      const BetheHeitlerRates&  R,
                      const HadronicTablesHost& T,
                      double                    expected_slope,
                      bool                      registered) {
    std::printf("\n== items 4 and 6: threshold behaviour, %s ==\n", c.name.c_str());
    const auto          xs = asymptotic_offsets(*c.bath);
    std::vector<double> J, I;
    double              worstD = 0.0;
    for (const double x : xs) {
      const RateValues r = R.at_offset(x);
      J.push_back(r.J);
      I.push_back(r.I);
      worstD = std::max(worstD, std::abs(r.Dmean / (2.0 * r.gamma * r.beta) - 1.0));
    }
    const double sJ = lsq_slope(xs, J), sI = lsq_slope(xs, I);
    diag("Gamma_th = %.10f, asymptotic nodes x/(Gamma_th-1) in [1e-5, 1e-4], x "
         "in [%.3e, %.3e]",
         c.bath->gamma_th(),
         xs.front(),
         xs.back());
    if (registered) {
      check_abs(c.name + ": threshold exponent of J",
                sJ - expected_slope,
                0.02,
                "slope - " + std::to_string(expected_slope));
      check_abs(c.name + ": threshold exponent of I",
                sI - expected_slope,
                0.02,
                "slope - " + std::to_string(expected_slope));
    } else {
      diag("%s: threshold slopes (diagnostic only): J %.4f, I %.4f",
           c.name.c_str(),
           sJ,
           sI);
    }
    check_abs(c.name + ": <D>/(2 Gamma beta) - 1 on the asymptotic nodes",
              worstD,
              1.0e-2);
    // first table decade (diagnostic)
    {
      std::vector<double> x1, J1, I1;
      double              worstD1 = 0.0;
      for (std::size_t i = 0; i < T.n_fine; ++i) {
        x1.push_back(T.gamma[i] - T.gamma_th);
        J1.push_back(T.J[i]);
        I1.push_back(T.I[i]);
        worstD1 = std::max(
          worstD1,
          std::abs(T.Dmean[i] / (2.0 * T.gamma[i] * T.beta[i]) - 1.0));
      }
      diag("%s: first table decade (x/(Gamma_th-1) from %.3e to %.3e): slopes "
           "J %.4f, I %.4f; max |<D>/(2Gb)-1| = %.3e",
           c.name.c_str(),
           x1.front() / (T.gamma_th - 1.0),
           x1.back() / (T.gamma_th - 1.0),
           lsq_slope(x1, J1),
           lsq_slope(x1, I1),
           worstD1);
    }
  }

  // ---------------------------------------------------------------------- //
  // Item 5: UR limit against Chodorowski eq. 3.11 (phi fits)
  // ---------------------------------------------------------------------- //
  void test_ur(const Case&               c,
               const BetheHeitlerRates&  R,
               const HadronicTablesHost& T) {
    std::printf("\n== item 5: UR limit, %s ==\n", c.name.c_str());
    const double gamma_ur = 450.0 /
                            c.bath->eps_cut(); // 2 Gamma eps_c = 900 (in-domain)
    const RateValues r    = R.at_gamma(gamma_ur);
    const RateValues u    = R.ur_limit(gamma_ur);
    const double     Ifit = I_ur_fit(*c.bath, gamma_ur);
    check_abs(c.name + ": I(exact, Gamma=" + std::to_string(gamma_ur) +
                ") vs eq.3.11 with phi fits",
              r.I / Ifit - 1.0,
              1.0e-2,
              "I/I_UR,fit - 1");
    diag("  Gamma = %.1f: I = %.8e, I_UR(fits) = %.8e, I_UR(own phi) = %.8e, "
         "J/J_UR(own Phi) - 1 = %+.2e, I/I_UR(own) - 1 = %+.2e",
         gamma_ur,
         r.I,
         Ifit,
         u.I,
         r.J / u.J - 1.0,
         r.I / u.I - 1.0);
    const RateValues rm = R.at_gamma(T.gamma_max);
    const RateValues um = R.ur_limit(T.gamma_max);
    diag("  at the table edge Gamma_max = %.4f (beta = %.6f): I/I_UR(fits) - 1 "
         "= %+.3e, J/J_UR(own) - 1 = %+.3e, I/I_UR(own) - 1 = %+.3e",
         T.gamma_max,
         rm.beta,
         rm.I / I_ur_fit(*c.bath, T.gamma_max) - 1.0,
         rm.J / um.J - 1.0,
         rm.I / um.I - 1.0);
  }

  void test_crpropa(const Quadrature& q) {
    std::printf("\n== item 5b: external UR anchor, CRPropa "
                "ElectronPairProduction (CMB) ==\n");
    const double theta   = crpropa_theta_cmb;
    const double eps_max = 40.0 * theta;
    auto         bath    = std::make_shared<BlackbodyBath>(theta, 0.0, eps_max);
    BetheHeitlerRates R(bath, q);
    const double n_ph = crpropa_n_ph * (1.0 - bath->clipped_number_fraction());
    diag("CMB bath: theta = %.6e, eps_max = 40 theta (clipped number fraction "
         "%.2e), n_ph = %.6e m^-3",
         theta,
         bath->clipped_number_fraction(),
         n_ph);
    const auto crpropa_cmb = read_crpropa_fixture();
    impl_true("CRPropa fixture read",
              crpropa_cmb.size() == 6,
              std::to_string(crpropa_cmb.size()) + " rows");
    double worst = 0.0;
    for (const auto& row : crpropa_cmb) {
      const double     gamma = std::pow(10.0, row.log10_gamma);
      const RateValues r     = R.at_gamma(gamma);
      const double     mine  = crpropa_a * n_ph * r.I / gamma;
      const double     rel   = mine / row.rate_per_mpc - 1.0;
      worst                  = std::max(worst, std::abs(rel));
      diag("  log10 gamma = %.2f: C++ %.6e /Mpc, CRPropa %.6e /Mpc, ratio-1 = "
           "%+.3e (eps'_hi = %.1f)",
           row.log10_gamma,
           mine,
           row.rate_per_mpc,
           rel,
           2.0 * gamma * eps_max);
    }
    check_abs("CRPropa CMB loss rate, log10 gamma in [9, 10.25]", worst, 3.0e-2);
  }

  // ---------------------------------------------------------------------- //
  // Item 6b: UR-error ratios of contract section 2
  // ---------------------------------------------------------------------- //
  void test_ur_ratios(const Quadrature& q) {
    std::printf("\n== item 6b: UR-error ratios (contract section 2) ==\n");

    struct Row {
      double gamma, eps0, Jratio, Dratio, Iratio;
    };

    const std::vector<Row> rows = {
      {  1.5, 1.0,  4.2, 1.317,  5.6 },
      {  2.5, 0.5,  2.2, 1.084,  2.4 },
      {  3.0, 0.5,  1.3, 1.057, 1.37 },
      {  5.0, 0.5, 1.03, 1.021, 1.06 },
      { 10.0, 0.5, 1.00, 1.006, 1.01 }
    };
    for (const auto& row : rows) {
      auto              bath = std::make_shared<MonoBath>(row.eps0);
      BetheHeitlerRates R(bath, q);
      const RateValues  r = R.at_gamma(row.gamma);
      const RateValues  u = R.ur_limit(row.gamma);
      char              name[96];
      std::snprintf(name,
                    sizeof(name),
                    "J_UR/J at Gamma=%.1f, eps0=%.1f (ref %.3g)",
                    row.gamma,
                    row.eps0,
                    row.Jratio);
      check_abs(name, (u.J / r.J) / row.Jratio - 1.0, 1.0e-2, "ratio/ref - 1");
      std::snprintf(name,
                    sizeof(name),
                    "<D>_UR/<D> at Gamma=%.1f, eps0=%.1f (ref %.4g)",
                    row.gamma,
                    row.eps0,
                    row.Dratio);
      check_abs(name, (u.Dmean / r.Dmean) / row.Dratio - 1.0, 1.0e-2, "ratio/ref - 1");
      diag("  Gamma=%.1f eps0=%.1f: J_UR/J = %.4f, <D>_UR/<D> = %.4f, I_UR/I = "
           "%.4f (spec III.2: %.3g)",
           row.gamma,
           row.eps0,
           u.J / r.J,
           u.Dmean / r.Dmean,
           u.I / r.I,
           row.Iratio);
    }
  }

  // ---------------------------------------------------------------------- //
  // Section 4: self-convergence, validity, interpolation target T3
  // ---------------------------------------------------------------------- //
  void test_table(const Case&               c,
                  const BetheHeitlerRates&  R,
                  const HadronicTablesHost& T,
                  const Quadrature&         q) {
    std::printf("\n== table %s: validity, self-convergence, interpolation ==\n",
                c.name.c_str());
    bool valid = true;
    try {
      validate_tables(T);
    } catch (const std::exception& e) {
      valid = false;
      diag("validate_tables: %s", e.what());
    }
    check_true(c.name + ": table monotone, finite, J and I > 0",
               valid,
               std::to_string(T.size()) + " nodes");
    diag("%s: Gamma_th = %.10g, Gamma_min = %.10g, Gamma_max = %.10g (envelope "
         "%.10g, eps'_max = %g), y in [%.6f, %.6f], nodes %zu = %zu fine + %zu "
         "coarse, I_peak = %.6e",
         c.name.c_str(),
         T.gamma_th,
         T.gamma_min,
         T.gamma_max,
         T.gamma_envelope,
         T.eps_prime_max,
         T.y_min,
         T.y_max,
         T.size(),
         T.n_fine,
         T.n_coarse,
         T.I_peak);
    diag("%s: eps'_hi at the last node = %.6f, I(Gamma_min)/I_peak = %.3e",
         c.name.c_str(),
         T.gamma_max * (1.0 + T.beta.back()) * c.bath->eps_cut(),
         T.I.front() / T.I_peak);

    // self-convergence: halve both quadrature steps, re-evaluate at the nodes
    {
      const auto        t0 = std::chrono::steady_clock::now();
      BetheHeitlerRates Rh(c.bath, q.halved());
      double worstJ = 0.0, worstI = 0.0, worstJ_all = 0.0, worstI_all = 0.0;
      for (std::size_t i = 0; i < T.size(); ++i) {
        const RateValues r  = Rh.at_offset(std::pow(10.0, T.y[i]));
        const double     dJ = std::abs(r.J / T.J[i] - 1.0);
        const double     dI = std::abs(r.I / T.I[i] - 1.0);
        worstJ_all          = std::max(worstJ_all, dJ);
        worstI_all          = std::max(worstI_all, dI);
        if (T.I[i] > 1.0e-6 * T.I_peak) {
          worstJ = std::max(worstJ, dJ);
          worstI = std::max(worstI, dI);
        }
      }
      check_abs(c.name + ": self-convergence of J (step halved), I > 1e-6 peak",
                worstJ,
                1.0e-8);
      check_abs(c.name + ": self-convergence of I (step halved), I > 1e-6 peak",
                worstI,
                1.0e-8);
      diag("%s: over all nodes: max |dJ/J| = %.3e, max |dI/I| = %.3e (%.2f s "
           "for the halved pass)",
           c.name.c_str(),
           worstJ_all,
           worstI_all,
           seconds_since(t0));
    }

    // interpolation target T3: linear in y of log J, log I at the midpoints
    {
      double worstI = 0.0, worstJ = 0.0, worstD = 0.0;
      for (std::size_t i = 0; i + 1 < T.size(); ++i) {
        const double     ym = 0.5 * (T.y[i] + T.y[i + 1]);
        const RateValues r  = R.at_offset(std::pow(10.0, ym));
        if (!(r.I > 1.0e-3 * T.I_peak)) {
          continue;
        }
        const double Ii = std::exp(0.5 * (std::log(T.I[i]) + std::log(T.I[i + 1])));
        const double Ji = std::exp(0.5 * (std::log(T.J[i]) + std::log(T.J[i + 1])));
        const double Di = Ii / Ji;
        worstI          = std::max(worstI, std::abs(Ii / r.I - 1.0));
        worstJ          = std::max(worstJ, std::abs(Ji / r.J - 1.0));
        worstD          = std::max(worstD, std::abs(Di / r.Dmean - 1.0));
      }
      check_abs(
        c.name + ": interpolation error of I at midpoints, I > 1e-3 peak (T3)",
        worstI,
        1.0e-2);
      diag("%s: midpoint interpolation errors: J %.3e, I %.3e, <D> %.3e",
           c.name.c_str(),
           worstJ,
           worstI,
           worstD);
    }
  }

  // ---------------------------------------------------------------------- //
  // Table bath: power-law segments reproduce the analytic power law
  // ---------------------------------------------------------------------- //
  void test_table_bath(const Quadrature& q) {
    std::printf("\n== table bath: log-log segments vs analytic power law; file "
                "reader ==\n");
    auto pl = std::make_shared<PowerlawBath>(0.1, 1.5, 2.0);
    auto tb = std::make_shared<TableBath>(std::vector<double> { 0.1, 1.5 },
                                          std::vector<double> { 100.0, 1.0 / 2.25 });
    BetheHeitlerRates Rp(pl, q), Rt(tb, q);
    double            worst = 0.0;
    for (const double g : { 1.5, 2.0, 3.0, 6.0, 30.0 }) {
      const RateValues a = Rp.at_gamma(g), b = Rt.at_gamma(g);
      worst = std::max({ worst,
                         std::abs(b.J / a.J - 1.0),
                         std::abs(b.I / a.I - 1.0),
                         std::abs(b.eps_mean / a.eps_mean - 1.0) });
    }
    impl_abs("two-node table bath vs powerlaw(0.1, 1.5, 2)", worst, 1.0e-10);

    // three-node table with a spectral break, written to and read from a file
    const std::vector<double> e3 { 0.1, 0.5, 1.5 }, n3 { 100.0, 4.0, 0.44 };
    const auto                tmp = std::filesystem::temp_directory_path() /
                     "bh_physics_test_spectrum.txt";
    {
      std::ofstream f(tmp);
      f << "# eps  dn/deps (arbitrary normalisation)\n";
      for (std::size_t i = 0; i < e3.size(); ++i) {
        f << e3[i] << " " << n3[i] << "\n";
      }
    }
    auto   t_file = std::make_shared<TableBath>(tmp.string());
    auto   t_vec  = std::make_shared<TableBath>(e3, n3);
    double dens   = 0.0;
    for (const double e : { 0.1, 0.2, 0.5, 0.7, 1.49 }) {
      dens = std::max(dens, std::abs(t_file->density(e) / t_vec->density(e) - 1.0));
    }
    impl_abs("table bath from file vs from vectors (density)", dens, 1.0e-14);
    diag("three-node table: breakpoints %zu, clipped fractions (%.3e, %.3e)",
         t_vec->breakpoints().size(),
         t_vec->clipped_number_fraction(),
         t_vec->clipped_energy_fraction());
    BetheHeitlerRates R3(t_vec, q), R3h(t_vec, q.halved());
    double            conv = 0.0;
    for (const double g : { 1.2, 2.0, 3.0, 6.0, 30.0 }) {
      const RateValues a = R3.at_gamma(g), b = R3h.at_gamma(g);
      conv = std::max(
        { conv, std::abs(b.J / a.J - 1.0), std::abs(b.I / a.I - 1.0) });
    }
    impl_abs("three-node table bath: self-convergence (step halved)", conv, 1.0e-8);
    // clipped table: eps_max above 2 must be refused without a clip
    bool refused = false;
    try {
      TableBath bad(std::vector<double> { 0.1, 3.0 },
                    std::vector<double> { 1.0, 1.0 });
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    impl_true("table bath with eps_max >= 2 refused",
              refused,
              "invalid_argument thrown");
    TableBath clipped(std::vector<double> { 0.1, 3.0 },
                      std::vector<double> { 1.0, 1.0 },
                      -1.0,
                      1.9);
    diag("table clipped at 1.9: clipped number fraction %.6f (exact %.6f), "
         "energy %.6f (exact %.6f)",
         clipped.clipped_number_fraction(),
         1.0 - 1.8 / 2.9,
         clipped.clipped_energy_fraction(),
         1.0 - (1.9 * 1.9 - 0.01) / (9.0 - 0.01));
    std::filesystem::remove(tmp);
  }

  // ---------------------------------------------------------------------- //
  // Blackbody clipping
  // ---------------------------------------------------------------------- //
  void test_blackbody() {
    std::printf("\n== blackbody normalisation and clipping ==\n");
    BlackbodyBath bb(0.1, 0.0, 1.9);
    diag("blackbody theta=0.1 on [0, 1.9]: clipped number fraction %.6e (of 2 "
         "zeta(3) theta^3), energy fraction %.6e (of pi^4/15 theta^4)",
         bb.clipped_number_fraction(),
         bb.clipped_energy_fraction());
    impl_abs("blackbody moment closure (number)",
             bb.closure_number(),
             1.0e-12,
             "(head+retained+tail)/total - 1");
    impl_abs("blackbody moment closure (energy)",
             bb.closure_energy(),
             1.0e-12,
             "(head+retained+tail)/total - 1");
    // unit integral of the normalised density
    const double one = integrate_log(
      [&](double e) {
        return bb.density(e);
      },
      1.0e-6,
      1.9,
      {});
    impl_abs("blackbody density normalised to unity on [1e-6, 1.9]",
             one - 1.0,
             1.0e-10,
             "integral - 1");
    BlackbodyBath bb2(0.1, 0.05, 1.0);
    diag("blackbody theta=0.1 on [0.05, 1.0]: clipped number %.6e, energy %.6e",
         bb2.clipped_number_fraction(),
         bb2.clipped_energy_fraction());
    bool refused = false;
    try {
      BlackbodyBath bad(0.1, 0.0, 2.0);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    impl_true("eps_max = 2 refused", refused, "invalid_argument thrown");
  }

  // ---------------------------------------------------------------------- //
  // Exchange file round trip
  // ---------------------------------------------------------------------- //
  // header lines of `original` with selected keys overridden, followed by
  // `data`; data_sha256 is recomputed from `data` unless overridden
  std::string rebuild_file(const std::string&                 original,
                           const std::string&                 data,
                           std::map<std::string, std::string> overrides) {
    std::size_t pos = 0;
    while (pos < original.size() && original[pos] == '#') {
      pos = original.find('\n', pos) + 1;
    }
    const std::string header = original.substr(0, pos);
    overrides.emplace("data_sha256", json_string(Sha256::digest(data)));
    std::string out;
    std::size_t s0 = 0;
    while (s0 < header.size()) {
      const auto  nl          = header.find('\n', s0);
      std::string line        = header.substr(s0, nl - s0);
      s0                      = nl + 1;
      const auto        colon = line.find(':');
      const std::string key   = line.substr(2, colon - 2);
      const auto        it    = overrides.find(key);
      if (it != overrides.end()) {
        line = "# " + key + ": " + it->second;
      }
      out += line + "\n";
    }
    return out + data;
  }

  void test_file(const Case& c, const HadronicTablesHost& T, const Quadrature& q) {
    std::printf("\n== exchange file round trip and reader validation (%s) ==\n",
                c.name.c_str());
    const auto tmp = std::filesystem::temp_directory_path() /
                     ("bh_physics_test_" + c.name + ".txt");
    TableFileMeta meta;
    meta.status = "test";
    write_table_file(tmp.string(), T, *c.bath, q, meta);
    TableFileContents back;
    std::string       err;
    const bool        ok = !throws(
      [&] {
        back = read_table_file(tmp.string());
      },
      &err);
    impl_true(
      c.name + ": exchange file reads back (schema 1.4, digest verified)",
      ok,
      ok ? back.data_sha256 : err);
    if (ok) {
      double worst = 0.0;
      for (std::size_t i = 0; i < T.size(); ++i) {
        worst = std::max({ worst,
                           std::abs(back.gamma[i] / T.gamma[i] - 1.0),
                           std::abs(back.J[i] / T.J[i] - 1.0),
                           std::abs(back.I[i] / T.I[i] - 1.0),
                           std::abs(back.Dmean[i] / T.Dmean[i] - 1.0),
                           std::abs(back.eps_mean[i] / T.eps_mean[i] - 1.0) });
      }
      impl_abs(c.name + ": columns read back at %.12e",
               worst,
               1.0e-11,
               "max |rel. rounding|");
      impl_true(
        c.name + ": header metadata decoded",
        back.node_count == T.size() && back.first_decade_node_count == T.n_fine &&
          back.gamma_th == T.gamma_th && back.gamma_max == T.gamma_max &&
          back.producer == "entity-cpp" && back.status == "test" &&
          back.format_revision == 1 && back.columns == canonical_columns() &&
          back.header.count("photon_field") == 1 &&
          back.gamma_at_I_peak == T.gamma_at_I_peak,
        "photon_field = " + back.header.at("photon_field"));
    }

    // negative tests on the literal bytes (R2)
    std::string bytes;
    {
      std::ifstream      in(tmp, std::ios::binary);
      std::ostringstream ss;
      ss << in.rdbuf();
      bytes = ss.str();
    }
    std::size_t pos = 0;
    while (pos < bytes.size() && bytes[pos] == '#') {
      pos = bytes.find('\n', pos) + 1;
    }
    const std::string header = bytes.substr(0, pos), data = bytes.substr(pos);
    impl_true(c.name + ": unmodified bytes parse",
              !throws(
                [&] {
                  parse_table_file(bytes);
                },
                &err),
              err);
    {
      std::string d = data;
      const auto  k = d.find_first_of("123456789");
      d[k]          = (d[k] == '9') ? '8' : static_cast<char>(d[k] + 1);
      impl_true("R2: changed data token rejected",
                throws(
                  [&] {
                    parse_table_file(header + d);
                  },
                  &err),
                err);
    }
    const auto nl1 = data.find('\n');
    const std::string first = data.substr(0, nl1 + 1), rest = data.substr(nl1 + 1);
    {
      const std::string d = first + "\n" + rest;
      impl_true("R2: inserted blank line rejected (stale digest)",
                throws(
                  [&] {
                    parse_table_file(header + d);
                  },
                  &err),
                err);
      impl_true("R2: inserted blank line rejected (recomputed digest)",
                throws(
                  [&] {
                    parse_table_file(rebuild_file(bytes, d, {}));
                  },
                  &err),
                err);
    }
    {
      const std::string d = first + "# a comment inside the data\n" + rest;
      impl_true("R2: comment line inside the data rejected (recomputed digest)",
                throws(
                  [&] {
                    parse_table_file(rebuild_file(bytes, d, {}));
                  },
                  &err),
                err);
    }
    {
      const std::string d = data.substr(0, data.size() - 1);
      impl_true("R2: missing terminal newline rejected (stale digest)",
                throws(
                  [&] {
                    parse_table_file(header + d);
                  },
                  &err),
                err);
      impl_true("R2: missing terminal newline rejected (recomputed digest)",
                throws(
                  [&] {
                    parse_table_file(rebuild_file(bytes, d, {}));
                  },
                  &err),
                err);
    }
    impl_true("R2: empty data block with a valid digest rejected",
              throws(
                [&] {
                  parse_table_file(rebuild_file(bytes,
                                                "",
                                                {
                                                  { "node_count", "0" }
                  }));
                },
                &err),
              err);
    {
      const std::string row = "2.500000000000e+00 9.165151389912e-01 "
                              "-4.259687322723e-01 -8.697646740000e-04 "
                              "3.830788020000e-03 4.404396000000e+00 "
                              "5.000000000000e-01\n";
      impl_true(
        "R2: one-row table with negative J and a correct digest rejected",
        throws(
          [&] {
            parse_table_file(rebuild_file(bytes,
                                          row,
                                          {
                                            { "node_count", "1" }
            }));
          },
          &err),
        err);
    }
    {
      const std::string rows = "2.500000000000e+00 9.165151389912e-01 "
                               "-4.259687322723e-01 -8.697646740000e-04 "
                               "3.830788020000e-03 4.404396000000e+00 "
                               "5.000000000000e-01\n"
                               "3.000000000000e+00 9.428090415821e-01 "
                               "-5.799194697769e-02 1.272033260000e-02 "
                               "6.561922080000e-02 5.158609000000e+00 "
                               "5.000000000000e-01\n";
      impl_true(
        "R2: two-row table with negative J and a correct digest rejected",
        throws(
          [&] {
            parse_table_file(rebuild_file(bytes,
                                          rows,
                                          {
                                            { "node_count", "2" }
            }));
          },
          &err),
        err);
      const std::string swapped = rows.substr(rows.find('\n') + 1) +
                                  rows.substr(0, rows.find('\n') + 1);
      std::string pos_rows = swapped;
      const auto  neg      = pos_rows.find("-8.697646740000e-04");
      pos_rows.replace(neg, 1, "+");
      impl_true("R2: non-increasing gamma_p with a correct digest rejected",
                throws(
                  [&] {
                    parse_table_file(rebuild_file(bytes,
                                                  pos_rows,
                                                  {
                                                    { "node_count", "2" }
                    }));
                  },
                  &err),
                err);
    }
    impl_true("R2: header claiming 999 nodes rejected",
              throws(
                [&] {
                  parse_table_file(rebuild_file(bytes,
                                                data,
                                                {
                                                  { "node_count", "999" }
                  }));
                },
                &err),
              err);
    {
      std::string h2;
      std::size_t s0 = 0;
      while (s0 < header.size()) {
        const auto        nl   = header.find('\n', s0);
        const std::string line = header.substr(s0, nl - s0 + 1);
        s0                     = nl + 1;
        if (line.rfind("# data_sha256:", 0) != 0) {
          h2 += line;
        }
      }
      impl_true("R2: missing data_sha256 rejected",
                throws(
                  [&] {
                    parse_table_file(h2 + data);
                  },
                  &err),
                err);
    }
    {
      const std::string d = first + "1 2 3 4 5 6 7 8\n" + rest;
      impl_true("R2: eight-token row rejected (recomputed digest)",
                throws(
                  [&] {
                    parse_table_file(rebuild_file(
                      bytes,
                      d,
                      {
                        { "node_count", std::to_string(T.size() + 1) }
                    }));
                  },
                  &err),
                err);
    }
    // SHA-256 known-answer test (FIPS 180-4 example)
    impl_true(
      "SHA-256 known answer ('abc')",
      Sha256::digest("abc") ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      Sha256::digest("abc"));
    std::filesystem::remove(tmp);
  }

  // ---------------------------------------------------------------------- //
  // R5: the reader against a hand-written contract-1.4 file
  // ---------------------------------------------------------------------- //
  void test_foreign_reader() {
    std::printf("\n== R5: reader against a hand-written contract-1.4 file (not "
                "produced by the C++ writer) ==\n");
    const std::string data =
      "2.200000000000e+00 8.907234634e-01 -1.124938736608e+00 "
      "2.330853550000e-06 9.064062080000e-06 3.888731000000e+00 "
      "5.000000000000e-01\n"
      "2.500000000000e+00 9.165151389912e-01 -4.259687322723e-01 "
      "8.697646740000e-04 3.830788020000e-03 4.404396000000e+00 "
      "5.000000000000e-01\n"
      "3.000000000000e+00 9.428090415821e-01 -5.799194697769e-02 "
      "1.272033260000e-02 6.561922080000e-02 5.158609000000e+00 "
      "5.000000000000e-01\n";
    std::string f;
    f += "# format: \"bh-rate-table\"\n";
    f += "# format_revision: 1\n";
    f += "# producer: \"bhref\"\n";
    f += "# version: \"v0.3.1\"\n";
    f += "# status: \"frozen:\\\"v0.3.1\\\"\"\n";
    f += "# physical_mass_ratio: 1836.15\n";
    f += "# photon_field: {\"type\": \"table\", \"file\": "
         "\"spec\\\\dir\\\\n\\\"q\\\".txt\", \"file_sha256\": \"00\"}\n";
    f += "# gamma_p_max_input: null\n";
    f += "# eps_prime_max: 91.8075\n";
    f += "# gamma_envelope: 91.81022308907225\n";
    f += "# gamma_th: 2.125\n";
    f += "# gamma_min: 2.2\n";
    f += "# gamma_max: 3.0\n";
    f += "# y_min: -1.124938736608\n";
    f += "# y_max: -0.05799194697769\n";
    f += "# grid_variable: \"log10(gamma_p - gamma_th)\"\n";
    f += "# lower_edge_fraction: 1e-12\n";
    f += "# first_decade_intervals_per_decade: 256\n";
    f += "# other_intervals_per_decade: 64\n";
    f += "# endpoint_coalescing: 1e-9\n";
    f += "# first_decade_node_count: 3\n";
    f += "# node_count: 3\n";
    f += "# I_peak: 6.56192208e-02\n";
    f += "# gamma_at_I_peak: 3.0\n";
    f += "# number_normalization: 1.0\n";
    f += "# mean_photon_energy: 0.5\n";
    f += "# discarded_number_fraction: 0.0\n";
    f += "# discarded_energy_fraction: 0.0\n";
    f += "# quadrature: {\"method\": \"scipy.integrate.quad\", \"epsrel\": "
         "1e-12}\n";
    f += "# columns: "
         "[\"gamma_p\",\"beta_p\",\"y\",\"J\",\"I\",\"Dmean\",\"eps_mean\"]\n";
    f += "# extra_unknown_key: {\"ignored\": true}\n";
    f += "# data_sha256: \"" + Sha256::digest(data) + "\"\n";
    f += data;
    TableFileContents c;
    std::string       err;
    const bool        ok = !throws(
      [&] {
        c = parse_table_file(f, "hand-written");
      },
      &err);
    impl_true("R5: hand-written schema-1.4 file accepted", ok, ok ? "" : err);
    if (ok) {
      impl_true(
        "R5: decoded metadata",
        c.producer == "bhref" && c.status == "frozen:\"v0.3.1\"" &&
          c.node_count == 3 && c.format_revision == 1 &&
          c.physical_mass_ratio == 1836.15 && c.columns == canonical_columns() &&
          c.header.count("extra_unknown_key") == 1 && c.version == "v0.3.1",
        "status = " + c.status);
      impl_abs("R5: data values",
               std::abs(c.J[1] / 8.69764674e-04 - 1.0) + std::abs(c.gamma[2] - 3.0),
               1.0e-12,
               "|dJ/J| + |dgamma|");
    }
    impl_true("R5: JSON string escapes decoded",
              json_unescape("\"spec\\\\dir\\\\n\\\"q\\\".txt\"") ==
                "spec\\dir\\n\"q\".txt",
              json_unescape("\"spec\\\\dir\\\\n\\\"q\\\".txt\""));
    {
      const std::string nasty = "a\"b\\c\n\t\x01z";
      impl_true("R5: json_string / json_unescape round trip",
                json_unescape(json_string(nasty)) == nasty,
                json_string(nasty));
    }
    {
      std::string g = f;
      const auto  i = g.find("# gamma_th:");
      g.erase(i, g.find('\n', i) - i + 1);
      impl_true("R5: missing required key rejected",
                throws(
                  [&] {
                    parse_table_file(g);
                  },
                  &err),
                err);
    }
    {
      std::string g = f;
      const auto  i = g.find("\"Dmean\"");
      g.replace(i, 7, "\"D\"");
      impl_true("R5: non-canonical columns rejected",
                throws(
                  [&] {
                    parse_table_file(g);
                  },
                  &err),
                err);
    }
    {
      std::string g = f;
      const auto  i = g.find("# format_revision: 1");
      g.replace(i, 20, "# format_revision: 2");
      impl_true("R5: unsupported format_revision rejected",
                throws(
                  [&] {
                    parse_table_file(g);
                  },
                  &err),
                err);
    }
    {
      std::string g = f;
      const auto  i = g.find("# I_peak: 6.56192208e-02");
      g.replace(i, 24, "# I_peak: NaN");
      impl_true("R5: non-finite numeric metadata rejected",
                throws(
                  [&] {
                    parse_table_file(g);
                  },
                  &err),
                err);
    }
  }

  // ---------------------------------------------------------------------- //
  // R1: validity envelope guards
  // ---------------------------------------------------------------------- //
  void test_envelope(const Quadrature& q) {
    std::printf("\n== R1: validity envelope guards ==\n");
    auto              bath = std::make_shared<MonoBath>(0.5);
    BetheHeitlerRates R(bath, q);
    std::string       err;
    impl_true("R1: mu_r = 1 rejected (k_cap = 0.05)",
              throws(
                [&] {
                  build_tables(R, 1.0);
                },
                &err),
              err);
    impl_true("R1: mu_r = 40 rejected (k_cap = 2)",
              throws(
                [&] {
                  build_tables(R, 40.0);
                },
                &err),
              err);
    impl_true("R1: mu_r = NaN rejected",
              throws(
                [&] {
                  build_tables(R, std::nan(""));
                },
                &err),
              err);
    impl_true("R1: mu_r = -1836.15 rejected",
              throws(
                [&] {
                  build_tables(R, -1836.15);
                },
                &err),
              err);
    impl_true("R1: gamma_p_max = inf rejected",
              throws(
                [&] {
                  build_tables(R, 1836.15, INFINITY);
                },
                &err),
              err);
    impl_true("R1: gamma_p_max = 1 rejected",
              throws(
                [&] {
                  build_tables(R, 1836.15, 1.0);
                },
                &err),
              err);
    impl_true("R1: gamma_p_max = 2 (below Gamma_th = 2.125) rejected",
              throws(
                [&] {
                  build_tables(R, 1836.15, 2.0);
                },
                &err),
              err);

    struct Row {
      double mu_r, eps_prime_max, gamma_max; // gamma_max 0 = not checked
    };

    for (const Row& r : {
           Row {   100.0,    5.0, 5.05 },
           Row { 19999.0, 999.95,  0.0 },
           Row { 20001.0, 1000.0,  0.0 }
    }) {
      const auto  T    = build_tables(R, r.mu_r);
      const char* name = r.mu_r == 100.0
                           ? "mu_r = 100"
                           : (r.mu_r < 20000.0 ? "mu_r = 19999" : "mu_r = 20001");
      validate_tables(T);
      impl_abs(std::string("R1: ") + name + ": eps'_max",
               T.eps_prime_max - r.eps_prime_max,
               1.0e-12,
               "eps'_max - " + std::to_string(r.eps_prime_max));
      if (r.gamma_max > 0.0) {
        impl_abs(std::string("R1: ") + name + ": Gamma_max",
                 T.gamma_max - r.gamma_max,
                 1.0e-12,
                 "Gamma_max - 5.05");
      }
      const double eps_hi_last = T.gamma.back() * (1.0 + T.beta.back()) * 0.5;
      impl_abs(std::string("R1: ") + name + ": eps'_hi at the last node = eps'_max",
               eps_hi_last / T.eps_prime_max - 1.0,
               1.0e-12,
               "relative");
      diag("  %s: Gamma_max = %.12f, eps'_hi(last) = %.12f, %zu nodes, I_peak "
           "= %.6e at Gamma = %.6f",
           name,
           T.gamma_max,
           eps_hi_last,
           T.size(),
           T.I_peak,
           T.gamma_at_I_peak);
    }
    {
      const auto T = build_tables(R, 1836.15, 10.0);
      impl_abs("R1: explicit gamma_p_max = 10 honoured",
               T.gamma_max - 10.0,
               1.0e-12,
               "Gamma_max - 10");
    }
  }

  // ---------------------------------------------------------------------- //
  // R3: strict spectrum rows and option strings
  // ---------------------------------------------------------------------- //
  void test_options() {
    std::printf("\n== R3: strict spectrum rows and option parsing ==\n");
    const auto tmp = std::filesystem::temp_directory_path() /
                     "bh_physics_test_badspec.txt";
    auto write = [&tmp](const std::string& content) {
      std::ofstream f(tmp);
      f << content;
    };
    std::string err;
    write("0.1 100\n0.5 abc\n1.5 0.44\n");
    impl_true("R3: non-numeric spectrum token rejected",
              throws(
                [&] {
                  TableBath t(tmp.string());
                },
                &err),
              err);
    write("0.1 100\n0.5 4 7\n1.5 0.44\n");
    impl_true("R3: three-column spectrum row rejected",
              throws(
                [&] {
                  TableBath t(tmp.string());
                },
                &err),
              err);
    write("0.1 100\n0.5\n1.5 0.44\n");
    impl_true("R3: one-column spectrum row rejected",
              throws(
                [&] {
                  TableBath t(tmp.string());
                },
                &err),
              err);
    write("0.1 100\n0.5 inf\n1.5 0.44\n");
    impl_true("R3: non-finite spectrum value rejected",
              throws(
                [&] {
                  TableBath t(tmp.string());
                },
                &err),
              err);
    write("# comment\n0.1 100   # trailing comment\n\n0.5 4\n1.5 0.44\n");
    err = "";
    impl_true("R3: comments and blank lines in a spectrum file accepted",
              !throws(
                [&] {
                  TableBath t(tmp.string());
                },
                &err),
              err);
    std::filesystem::remove(tmp);

    using V      = std::vector<std::string>;
    auto rejects = [&err](const std::string& name, const V& v) {
      impl_true(name,
                throws(
                  [&] {
                    parse_dump_options(v);
                  },
                  &err),
                err);
    };
    rejects("R3: --eps0 '.5 malformed' rejected",
            V { "--type", "mono", "--eps0", ".5 malformed", "--out", "x" });
    rejects(
      "R3: --gamma_p_max not-a-number rejected",
      V { "--type", "mono", "--eps0", "0.5", "--gamma_p_max", "not-a-number", "--out", "x" });
    rejects(
      "R3: --gamma_p_max 0 rejected (no sentinel)",
      V { "--type", "mono", "--eps0", "0.5", "--gamma_p_max", "0", "--out", "x" });
    rejects("R3: --eps0 2.5 rejected",
            V { "--type", "mono", "--eps0", "2.5", "--out", "x" });
    rejects("R3: --eps0 nan rejected",
            V { "--type", "mono", "--eps0", "nan", "--out", "x" });
    rejects("R3: --eps0 1e400 rejected",
            V { "--type", "mono", "--eps0", "1e400", "--out", "x" });
    rejects("R3: --mu_r -5 rejected",
            V { "--type", "mono", "--eps0", "0.5", "--mu_r", "-5", "--out", "x" });
    rejects(
      "R3: --theta 0 rejected",
      V { "--type", "blackbody", "--theta", "0", "--eps_max", "1.9", "--out", "x" });
    rejects("R3: unknown option rejected",
            V { "--type", "mono", "--eps0", "0.5", "--bogus", "1", "--out", "x" });
    rejects("R3: missing --out rejected", V { "--type", "mono", "--eps0", "0.5" });
    {
      const DumpOptions o = parse_dump_options(
        V { "--type", "mono", "--eps0", "0.5", "--gamma_p_max", "auto", "--out", "x" });
      impl_true("R3: --gamma_p_max auto selects the envelope",
                o.gamma_p_max == 0.0 && o.spec.eps0 == 0.5 && o.mu_r == 1836.15,
                "gamma_p_max = " + std::to_string(o.gamma_p_max));
    }
    {
      const DumpOptions o = parse_dump_options(V { "--type",
                                                   "blackbody",
                                                   "--theta",
                                                   "0.1",
                                                   "--eps_min",
                                                   "0",
                                                   "--eps_max",
                                                   "1.9",
                                                   "--mu_r",
                                                   "100",
                                                   "--gamma_p_max",
                                                   "4.5",
                                                   "--out",
                                                   "x" });
      impl_true("R3: valid blackbody options parsed",
                o.spec.theta == 0.1 && o.spec.eps_min == 0.0 &&
                  o.spec.eps_max == 1.9 && o.mu_r == 100.0 &&
                  o.gamma_p_max == 4.5 && o.out == "x",
                "ok");
    }
  }

  // ---------------------------------------------------------------------- //
  // R4: power-segment integral
  // ---------------------------------------------------------------------- //
  void test_power_integral() {
    std::printf(
      "\n== R4: power-segment integral vs Gauss-Legendre reference ==\n");
    static const GaussLegendre gl { 16 };
    auto                       reference = [](double a, double b, double p) {
      const int    npan = 256;
      const double h    = (b - a) / npan;
      double       sum  = 0.0;
      for (int k = 0; k < npan; ++k) {
        const double mid = a + h * (k + 0.5), half = 0.5 * h;
        for (int i = 0; i < gl.order(); ++i) {
          sum += half * gl.w[i] * std::pow(mid + half * gl.x[i], p);
        }
      }
      return sum;
    };

    struct C {
      double a, b, p;
    };

    double worst = 0.0;
    for (const C& c : {
           C { 0.1,    1.5, -1.0 + 2.0e-12 },
           C { 0.1,    1.5, -1.0 + 1.0e-11 },
           C { 0.1,    1.5,           -0.9 },
           C { 0.1,    1.5,           -1.1 },
           C { 0.1,    1.5,           -1.0 },
           C { 0.1,    1.5,           -2.0 },
           C { 0.5, 0.5001, -1.0 + 2.0e-12 },
           C { 0.5, 0.5001, -1.0 + 1.0e-11 },
           C { 0.5, 0.5001,           -0.9 },
           C { 0.5, 0.5001,           -1.1 },
           C { 0.5, 0.5001,           -1.0 }
    }) {
      const double v   = PowerlawBath::power_integral(c.a, c.b, c.p);
      const double ref = reference(c.a, c.b, c.p);
      const double rel = v / ref - 1.0;
      worst            = std::max(worst, std::abs(rel));
      diag("  int_%g^%g eps^(%.13g): %.16e, reference %.16e, rel %+.2e",
           c.a,
           c.b,
           c.p,
           v,
           ref,
           rel);
    }
    impl_abs("R4: power_integral vs Gauss-Legendre reference", worst, 1.0e-13);
    PowerlawBath pl(0.1, 1.5, 2.0);
    const double exact = std::log(15.0) / (10.0 - 1.0 / 1.5);
    impl_abs("R4: powerlaw(0.1, 1.5, 2) mean photon energy (t = 0 branch)",
             pl.mean_eps() / exact - 1.0,
             1.0e-14,
             "mean_eps/exact - 1");
    diag("powerlaw mean photon energy %.15f (exact ln 15 / (10 - 2/3) = %.15f)",
         pl.mean_eps(),
         exact);
  }

  // ---------------------------------------------------------------------- //
  // Design numbers of contract section 6 (diagnostics)
  // ---------------------------------------------------------------------- //
  void design_numbers(const Case&               c,
                      const BetheHeitlerRates&  R,
                      const HadronicTablesHost& T) {
    std::printf(
      "\n== design numbers (contract section 6, diagnostics), %s ==\n",
      c.name.c_str());
    for (const double g : { 1.5, 2.0, 3.0, 4.0, 6.0 }) {
      if (g <= T.gamma_th || g > T.gamma_max) {
        diag("  Gamma_p = %.1f: outside (Gamma_th, Gamma_max]", g);
        continue;
      }
      const RateValues r = R.at_gamma(g);
      const double     K = c.mu_r * (g - 1.0);
      diag("  Gamma_p = %.1f: J = %.6e, I = %.6e, <D> = %.6f, eps_mean = %.6f, "
           "K/<D> = %.1f",
           g,
           r.J,
           r.I,
           r.Dmean,
           r.eps_mean,
           K / r.Dmean);
    }
    // integrated interaction count N_int(Gamma0) = mu_r int_{Gamma_th}^{Gamma0} dGamma/<D>
    for (const double g0 : { 3.0, 6.0 }) {
      if (g0 <= T.gamma_th || g0 > T.gamma_max) {
        continue;
      }
      static const GaussLegendre gl { 8 };
      const double y_hi = std::log10(g0 - T.gamma_th), y_lo = y_hi - 12.0;
      const int    npan = 600;
      const double hp   = (y_hi - y_lo) / npan;
      double       sum  = 0.0;
      for (int p = 0; p < npan; ++p) {
        const double mid = y_lo + hp * (p + 0.5), half = 0.5 * hp;
        for (int i = 0; i < gl.order(); ++i) {
          const double y = mid + half * gl.x[i];
          const double x = std::pow(10.0, y);
          sum += half * gl.w[i] * x * std::log(10.0) / R.at_offset(x).Dmean;
        }
      }
      diag("  N_int(Gamma0 = %.0f) = mu_r int dGamma/<D> = %.1f", g0, c.mu_r * sum);
    }
  }

} // namespace

int main() {
  const auto t_start = std::chrono::steady_clock::now();
  Quadrature q; // registered defaults: h_log_delta = h_log_eps = 0.02, delta0 = 1e-8, GL-16
  std::printf("Bethe-Heitler startup tables: acceptance tests (contract 1.3 "
              "section 5, items 1-6; exchange format of contract 1.4)\n");
  std::printf("quadrature: h_log_delta = %g, h_log_eps = %g, delta0 = %g, GL "
              "order %d, k_max = %g\n",
              q.h_log_delta,
              q.h_log_eps,
              q.delta0,
              q.gl_order,
              q.k_max);

  try {
    const Cumulatives cum(q);
    diag("cumulative grid: %zu cells in ln(k-2) on [ln %g, ln %g], breakpoint "
         "at k = 4",
         cum.n_cells(),
         q.delta0,
         q.k_max - 2.0);

    test_maximon();
    test_phi(cum);
    test_qk();

    const std::vector<Case> cases = {
      { "mono05", std::make_shared<MonoBath>(0.5), 1836.15 },
      { "mono03", std::make_shared<MonoBath>(0.3), 1836.15 },
      { "powerlaw", std::make_shared<PowerlawBath>(0.1, 1.5, 2.0), 1836.15 },
      { "blackbody", std::make_shared<BlackbodyBath>(0.1, 0.0, 1.9), 1836.15 },
      { "mono05_mu100", std::make_shared<MonoBath>(0.5), 100.0 },
    };

    for (const auto& c : cases) {
      const auto               t0 = std::chrono::steady_clock::now();
      const BetheHeitlerRates  R(c.bath, q);
      const HadronicTablesHost T = build_tables(R, c.mu_r);
      diag("%s: table built in %.2f s", c.name.c_str(), seconds_since(t0));
      test_table(c, R, T, q);
      if (c.name != "mono05_mu100") {
        // the rates do not depend on mu_r: the mu_r = 100 repeat changes only the grid
        const bool   mono       = c.bath->is_mono();
        const double slope      = mono ? 4.0 : 5.0;
        const bool   registered = (c.name != "blackbody");
        test_threshold(c, R, T, slope, registered);
        test_ur(c, R, T);
      } else {
        diag("mono05_mu100: rates identical to mono05 (mu_r enters only the "
             "envelope); Gamma_max = %.6f",
             T.gamma_max);
      }
      design_numbers(c, R, T);
      if (c.name == "mono05" || c.name == "blackbody") {
        test_file(c, T, q);
      }
    }

    test_ur_ratios(q);
    test_crpropa(q);
    test_table_bath(q);
    test_blackbody();
    test_envelope(q);
    test_options();
    test_power_integral();
    test_foreign_reader();
  } catch (const std::exception& e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    ++g_fail_impl;
  }

  std::printf("\n== summary: contract items %d passed, %d failed; "
              "implementation-chosen checks %d passed, %d failed (%.1f s) ==\n",
              g_pass_contract,
              g_fail_contract,
              g_pass_impl,
              g_fail_impl,
              seconds_since(t_start));
  return (g_fail_contract + g_fail_impl) == 0 ? 0 : 1;
}
