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

#include "archetypes/hadronic/bh_table_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace arch::bh;

namespace {

  int g_pass = 0, g_fail = 0;

  bool check_abs(const std::string& name,
                 double             measured,
                 double             tol,
                 const std::string& what = "max |rel. dev.|") {
    const bool ok = std::isfinite(measured) && std::abs(measured) <= tol;
    std::printf("[%s] %s: %s = %.6e (tol %.3e)\n",
                ok ? "PASS" : "FAIL",
                name.c_str(),
                what.c_str(),
                measured,
                tol);
    ok ? ++g_pass : ++g_fail;
    return ok;
  }

  bool check_true(const std::string& name, bool ok, const std::string& what) {
    std::printf("[%s] %s: %s\n", ok ? "PASS" : "FAIL", name.c_str(), what.c_str());
    ok ? ++g_pass : ++g_fail;
    return ok;
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
  // in 1/Mpc versus log10(gamma). Source: the public generator of the CRPropa
  // table, CRPropa3-data `calc_pairproduction.py` (Chodorowski eq. 3.11 with
  // the phi fits, CMB T = 2.72548 K from `photonField.py`, constants from
  // `units.py`, CODATA 2006), executed on 2026-10-24 at these Lorentz factors
  // because the distributed data file was not reachable from the build
  // container. Values are therefore a regeneration of the public table, not a
  // transcription of the distributed file (see the step-1 report).
  struct CRPropaRow {
    double log10_gamma, rate_per_mpc;
  };

  const std::vector<CRPropaRow> crpropa_cmb = {
    {  9.00, 3.223586e-05 },
    {  9.25, 1.322157e-04 },
    {  9.50, 3.162189e-04 },
    {  9.75, 5.350970e-04 },
    { 10.00, 7.191011e-04 },
    { 10.25, 8.238957e-04 }
  };
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
    check_abs("phi threshold law at kappa-2 = 1e-4",
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
    check_abs("two-node table bath vs powerlaw(0.1, 1.5, 2)", worst, 1.0e-10);

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
    check_abs("table bath from file vs from vectors (density)", dens, 1.0e-14);
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
    check_abs("three-node table bath: self-convergence (step halved)", conv, 1.0e-8);
    // clipped table: eps_max above 2 must be refused without a clip
    bool refused = false;
    try {
      TableBath bad(std::vector<double> { 0.1, 3.0 },
                    std::vector<double> { 1.0, 1.0 });
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    check_true("table bath with eps_max >= 2 refused",
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
    check_abs("blackbody moment closure (number)",
              bb.closure_number(),
              1.0e-12,
              "(head+retained+tail)/total - 1");
    check_abs("blackbody moment closure (energy)",
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
    check_abs("blackbody density normalised to unity on [1e-6, 1.9]",
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
    check_true("eps_max = 2 refused", refused, "invalid_argument thrown");
  }

  // ---------------------------------------------------------------------- //
  // Exchange file round trip
  // ---------------------------------------------------------------------- //
  void test_file(const Case& c, const HadronicTablesHost& T, const Quadrature& q) {
    std::printf("\n== exchange file round trip (%s) ==\n", c.name.c_str());
    const auto tmp = std::filesystem::temp_directory_path() /
                     ("bh_physics_test_" + c.name + ".txt");
    TableFileMeta meta;
    meta.status = "test";
    write_table_file(tmp.string(), T, *c.bath, q, meta);
    const auto back = read_table_file(tmp.string());
    check_true(c.name + ": SHA-256 of the data block verifies",
               back.sha_ok,
               back.sha256_data);
    double worst = 0.0;
    for (std::size_t i = 0; i < T.size(); ++i) {
      worst = std::max({ worst,
                         std::abs(back.gamma[i] / T.gamma[i] - 1.0),
                         std::abs(back.J[i] / T.J[i] - 1.0),
                         std::abs(back.I[i] / T.I[i] - 1.0),
                         std::abs(back.Dmean[i] / T.Dmean[i] - 1.0),
                         std::abs(back.eps_mean[i] / T.eps_mean[i] - 1.0) });
    }
    check_abs(c.name + ": columns read back at %.12e",
              worst,
              1.0e-11,
              "max |rel. rounding|");
    check_true(
      c.name + ": header carries the bath, mu_r, grid and hash keys",
      back.header.count("bath") && back.header.count("physical_mass_ratio") &&
        back.header.count("gamma_th") && back.header.count("gamma_min") &&
        back.header.count("gamma_max") && back.header.count("n_nodes") &&
        back.header.count("clipped_number_fraction") &&
        back.header.count("clipped_energy_fraction") &&
        back.header.count("sha256_data") && back.header.count("status"),
      "bath = " + back.header.at("bath"));
    // SHA-256 known-answer test (FIPS 180-4 example)
    check_true(
      "SHA-256 known answer ('abc')",
      Sha256::digest("abc") ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
      Sha256::digest("abc"));
    std::filesystem::remove(tmp);
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
              "section 5, items 1-6)\n");
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
  } catch (const std::exception& e) {
    std::printf("[FAIL] exception: %s\n", e.what());
    ++g_fail;
  }

  std::printf("\n== summary: %d passed, %d failed (%.1f s) ==\n",
              g_pass,
              g_fail,
              seconds_since(t_start));
  return g_fail == 0 ? 0 : 1;
}
