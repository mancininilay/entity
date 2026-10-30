/**
 * @file archetypes/hadronic/bh_physics.h
 * @brief Bethe-Heitler pair-production physics for the startup rate tables
 *
 * Host-only code, double precision throughout regardless of the build's
 * `real_t`. Nothing here depends on Kokkos, the problem generator or the
 * input parser; the file is a self-contained implementation of the rates
 * step of the Bethe-Heitler module (contract revision 1.3, sections 1 to 4):
 *
 *   - sigma_tilde(k): total Bethe-Heitler cross section on a static source
 *     in units of alpha r_e^2 (Maximon 1968, eqs. 10 and 12, junction k = 4)
 *   - q_k(k): mean momentum absorbed by the source along the photon
 *     direction (Chodorowski, Zdziarski & Sikora 1992, eq. 3.7 fit)
 *   - Cumulatives: Phi(x) = int_2^x k sigma dk, phi(x) = int_2^x k sigma q_k
 * dk, psi(x) = int_2^x sigma q_k dk, built once on a grid in log(k - 2)
 *   - PhotonBath (mono, powerlaw, blackbody, table): normalised number
 *     spectrum of the isotropic simulation-frame photon field
 *   - BetheHeitlerRates: exact-in-beta J(Gamma), I(Gamma), <D> = I/J and the
 *     interaction-weighted mean photon energy
 *   - HadronicTablesHost / build_tables: the registered node grid in
 *     y = log10(Gamma - Gamma_th)
 *
 * Units: energies in m_e c^2, cross sections in alpha r_e^2, J and I per unit
 * photon density in alpha r_e^2 c (dimensionless).
 *
 * @namespaces:
 *   - arch::bh::
 */

#ifndef ARCHETYPES_HADRONIC_BH_PHYSICS_H
#define ARCHETYPES_HADRONIC_BH_PHYSICS_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace arch::bh {

  // ------------------------------------------------------------------------ //
  // Constants
  // ------------------------------------------------------------------------ //
  namespace constants {
    inline constexpr double PI         = 3.14159265358979323846264338327950288;
    inline constexpr double ZETA3      = 1.20205690315959428539973816151144999;
    // kernel domain of the q_k fit (contract section 4)
    inline constexpr double K_MIN      = 2.0;
    inline constexpr double K_MAX      = 1000.0;
    // junction of Maximon's two series
    inline constexpr double K_JUNCTION = 4.0;
    // validity envelope: eps'_hi <= ENVELOPE_FRACTION * mu_r
    inline constexpr double ENVELOPE_FRACTION       = 0.05;
    // fraction of the peak of I that defines the lower grid edge
    inline constexpr double LOWER_EDGE_FRACTION     = 1.0e-12;
    // registered node densities per decade of y
    inline constexpr int    NODES_PER_DECADE_FINE   = 256;
    inline constexpr int    NODES_PER_DECADE_COARSE = 64;
    // a regular node closer than this (in y) to the upper edge is replaced by it
    inline constexpr double ENDPOINT_COALESCING     = 1.0e-9;
  } // namespace constants

  // ------------------------------------------------------------------------ //
  // Microscopic functions of the rest-frame photon energy k (= eps')
  // ------------------------------------------------------------------------ //

  /**
   * @brief Maximon 1968 eq. (10): threshold series, 2 < k < 4
   * @param delta k - 2 (passed directly to avoid cancellation near threshold)
   */
  inline double sigma_tilde_series10_delta(double delta) {
    if (delta <= 0.0) {
      return 0.0;
    }
    const double k    = 2.0 + delta;
    const double rho  = 2.0 * delta / (k + 2.0 + 2.0 * std::sqrt(2.0 * k));
    const double r    = delta / k;
    const double poly = 1.0 + rho * (0.5 + rho * (23.0 / 40.0 +
                                                  rho * (11.0 / 60.0 +
                                                         rho * (29.0 / 960.0))));
    return (2.0 * constants::PI / 3.0) * r * r * r * poly;
  }

  /**
   * @brief Maximon 1968 eq. (12): asymptotic series in (2/k)^2, k >= 4
   */
  inline double sigma_tilde_series12(double k) {
    const double L   = std::log(2.0 * k);
    const double L2  = L * L;
    const double L3  = L2 * L;
    const double u   = 2.0 / k;
    const double u2  = u * u;
    const double u4  = u2 * u2;
    const double u6  = u4 * u2;
    const double pi2 = constants::PI * constants::PI;
    const double t0  = (28.0 / 9.0) * L - 218.0 / 27.0;
    const double t2  = 6.0 * L - 3.5 + (2.0 / 3.0) * L3 - L2 - (pi2 / 3.0) * L +
                      2.0 * constants::ZETA3 + pi2 / 6.0;
    const double t4 = (3.0 / 16.0) * L + 0.125;
    const double t6 = (29.0 / (9.0 * 256.0)) * L - 77.0 / (27.0 * 512.0);
    return t0 + u2 * t2 - u4 * t4 - u6 * t6;
  }

  /**
   * @brief sigma_tilde(2 + delta) in units of alpha r_e^2 (0 for delta <= 0)
   */
  inline double sigma_tilde_delta(double delta) {
    if (delta <= 0.0) {
      return 0.0;
    }
    if (delta < constants::K_JUNCTION - 2.0) {
      return sigma_tilde_series10_delta(delta);
    }
    return sigma_tilde_series12(2.0 + delta);
  }

  /**
   * @brief sigma_tilde(k), total cross section, units of alpha r_e^2
   */
  inline double sigma_tilde(double k) {
    return sigma_tilde_delta(k - 2.0);
  }

  /**
   * @brief Chodorowski et al. 1992 eq. (3.7): mean momentum absorbed by the
   *        static source along the photon direction, q_k(2 + delta)
   */
  inline double q_k_delta(double delta) {
    if (delta <= 0.0) {
      return 2.0;
    }
    const double omega = 2.0 + delta;
    const double l     = std::log1p(delta); // ln(omega - 1)
    return (4.0 / omega) * (1.0 + l * (0.3958 + l * (0.1000 + l * 0.00781)));
  }

  inline double q_k(double k) {
    return q_k_delta(k - 2.0);
  }

  /**
   * @brief Source of q_k in the loss cumulatives (table contract revision 1.5,
   *        2026-10-29): the Chodorowski eq. 3.7 fit of step 1 (`Fit`,
   *        exchange-file value "chodorowski-3.7") or the Born machinery of the
   *        pair sampler (`Born`, "born-3D2000", `bh_born_qk.h`). The library
   *        default is `Fit` so that every existing table and test is
   *        byte-identical in behaviour (the regression and reference path); the
   *        problem generator's input default is `born`, the decided production
   *        default (Nilay, 2026-10-30, contract 1.5 section 3).
   */
  enum class QkSource : int {
    Fit  = 0,
    Born = 1
  };

  inline const char* qk_source_name(QkSource s) {
    return s == QkSource::Born ? "born-3D2000" : "chodorowski-3.7";
  }

  /**
   * @brief The Born provider hook: bh_born_qk.h (which depends on the Born
   *        laws of bh_restframe.h, itself including this header) registers
   *        its q_k^Born(2 + delta) here at static initialisation; a program
   *        that builds Born cumulatives must include bh_born_qk.h in at least
   *        one translation unit, otherwise the call below throws (never a
   *        silent fallback to the fit)
   */
  using QkFunction = double (*)(double);

  inline QkFunction& born_qk_hook() {
    static QkFunction p = nullptr;
    return p;
  }

  inline double born_q_k_delta(double delta) {
    const QkFunction p = born_qk_hook();
    if (p == nullptr) {
      throw std::logic_error("born_q_k_delta: the Born q_k provider is not registered (include "
                             "archetypes/hadronic/bh_born_qk.h in the program)");
    }
    return p(delta);
  }

  /// q_k from the chosen source
  inline double q_k_delta_from(QkSource s, double delta) {
    return s == QkSource::Born ? born_q_k_delta(delta) : q_k_delta(delta);
  }

  /**
   * @brief Strict decimal parsing: the whole token must be consumed, no
   *        leading whitespace, and the value must be finite
   */
  inline bool parse_number_strict(const std::string& s, double& out) {
    if (s.empty() || std::isspace(static_cast<unsigned char>(s[0]))) {
      return false;
    }
    char*        end = nullptr;
    const double v   = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0' || !std::isfinite(v)) {
      return false;
    }
    out = v;
    return true;
  }

  // ------------------------------------------------------------------------ //
  // Gauss-Legendre rule on [-1, 1]
  // ------------------------------------------------------------------------ //
  struct GaussLegendre {
    std::vector<double> x, w;

    explicit GaussLegendre(int n) {
      if (n < 2 || n > 64) {
        throw std::invalid_argument("GaussLegendre: order must be in [2, 64]");
      }
      x.resize(n);
      w.resize(n);
      for (int i = 0; i < n; ++i) {
        double z  = std::cos(constants::PI * (i + 0.75) / (n + 0.5));
        double pp = 1.0;
        for (int it = 0; it < 100; ++it) {
          double p1 = 1.0, p2 = 0.0;
          for (int j = 0; j < n; ++j) {
            const double p3 = p2;
            p2              = p1;
            p1              = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
          }
          pp              = n * (z * p1 - p2) / (z * z - 1.0);
          const double z1 = z;
          z               = z1 - p1 / pp;
          if (std::abs(z - z1) < 1.0e-16) {
            break;
          }
        }
        x[i] = -z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
      }
    }

    int order() const {
      return static_cast<int>(x.size());
    }
  };

  // ------------------------------------------------------------------------ //
  // Quadrature parameters (one "step" for the self-convergence test)
  // ------------------------------------------------------------------------ //
  struct Quadrature {
    // cell width in t = ln(k - 2) of the cumulative grid
    double h_log_delta = 0.02;
    // panel width in v = ln(eps / eps_th) of the outer photon-energy integral
    double h_log_eps   = 0.02;
    // start of the cumulative grid, k - 2 = delta0 (analytic below)
    double delta0      = 1.0e-8;
    // Gauss-Legendre order per cell / panel
    int    gl_order    = 16;
    // upper end of the kernel domain in k
    double k_max       = constants::K_MAX;
    // the source of q_k in the cumulatives (contract 1.5); Fit, the library
    // default, keeps every existing result byte-identical (regression path);
    // Born is the decided production default (2026-10-30), selected by the
    // problem generator's input default
    QkSource qk_source = QkSource::Fit;

    Quadrature halved() const {
      Quadrature q   = *this;
      q.h_log_delta *= 0.5;
      q.h_log_eps   *= 0.5;
      return q;
    }
  };

  // ------------------------------------------------------------------------ //
  // Cumulatives Phi, phi, psi
  // ------------------------------------------------------------------------ //
  class Cumulatives {
  public:
    struct Values {
      double Phi { 0.0 }; // int_2^x k sigma dk
      double phi { 0.0 }; // int_2^x k sigma q_k dk
      double psi { 0.0 }; // int_2^x sigma q_k dk
    };

    explicit Cumulatives(const Quadrature& q = Quadrature {})
      : m_delta0 { q.delta0 }
      , m_kmax { q.k_max }
      , m_h { q.h_log_delta }
      , m_gl { q.gl_order }
      , m_qk_source { q.qk_source } {
      if (!(m_delta0 > 0.0) || m_delta0 >= 1.0) {
        throw std::invalid_argument("Cumulatives: delta0 must be in (0, 1)");
      }
      if (m_kmax <= constants::K_JUNCTION) {
        throw std::invalid_argument("Cumulatives: k_max must exceed 4");
      }
      if (!(m_h > 0.0)) {
        throw std::invalid_argument("Cumulatives: h_log_delta must be > 0");
      }
      const double t0 = std::log(m_delta0);
      const double t1 = std::log(constants::K_JUNCTION - 2.0);
      const double t2 = std::log(m_kmax - 2.0);
      // segment 1: [t0, t1], segment 2: [t1, t2]; node exactly at t1
      const auto   n1 = static_cast<std::size_t>(std::ceil((t1 - t0) / m_h));
      const auto   n2 = static_cast<std::size_t>(std::ceil((t2 - t1) / m_h));
      m_t.reserve(n1 + n2 + 1);
      for (std::size_t i = 0; i < n1; ++i) {
        m_t.push_back(
          t0 + (t1 - t0) * static_cast<double>(i) / static_cast<double>(n1));
      }
      m_t.push_back(t1);
      for (std::size_t i = 1; i < n2; ++i) {
        m_t.push_back(
          t1 + (t2 - t1) * static_cast<double>(i) / static_cast<double>(n2));
      }
      m_t.push_back(t2);
      m_junction = n1;
      // cumulative sums; the start is the analytic threshold law
      m_cum.resize(m_t.size());
      m_cum[0] = analytic(m_delta0);
      for (std::size_t i = 1; i < m_t.size(); ++i) {
        const Values c = cell(m_t[i - 1], m_t[i]);
        m_cum[i].Phi   = m_cum[i - 1].Phi + c.Phi;
        m_cum[i].phi   = m_cum[i - 1].phi + c.phi;
        m_cum[i].psi   = m_cum[i - 1].psi + c.psi;
      }
    }

    /**
     * @brief Cumulatives at k = 2 + delta (zero for delta <= 0)
     */
    Values operator()(double delta) const {
      if (!(delta > 0.0)) {
        return Values {};
      }
      if (delta <= m_delta0) {
        return analytic(delta);
      }
      const double dmax = m_kmax - 2.0;
      if (delta > dmax) {
        if (delta <= dmax * (1.0 + 1.0e-9)) {
          delta = dmax;
        } else {
          throw std::domain_error(
            "Cumulatives: k = " + std::to_string(2.0 + delta) +
            " outside the kernel domain [2, " + std::to_string(m_kmax) + "]");
        }
      }
      const double t  = std::log(delta);
      // locate the cell: t in [m_t[j], m_t[j+1])
      auto         it = std::upper_bound(m_t.begin(), m_t.end(), t);
      std::size_t  j  = static_cast<std::size_t>(it - m_t.begin());
      if (j == 0) {
        j = 1;
      }
      if (j >= m_t.size()) {
        j = m_t.size() - 1;
      }
      --j;
      const Values p = cell(m_t[j], t);
      return Values { m_cum[j].Phi + p.Phi,
                      m_cum[j].phi + p.phi,
                      m_cum[j].psi + p.psi };
    }

    double Phi(double delta) const {
      return (*this)(delta).Phi;
    }

    double phi(double delta) const {
      return (*this)(delta).phi;
    }

    double psi(double delta) const {
      return (*this)(delta).psi;
    }

    double delta0() const {
      return m_delta0;
    }

    double k_max() const {
      return m_kmax;
    }

    double h() const {
      return m_h;
    }

    std::size_t n_cells() const {
      return m_t.size() - 1;
    }

    QkSource qk_source() const {
      return m_qk_source;
    }

    /**
     * @brief Threshold laws: Phi -> (pi/24) d^4, phi -> (pi/12) d^4, psi -> (pi/24) d^4
     *        (q_k(2) = 2 for both sources, so the leading law is the same; the
     *        start of the grid delta0 = 1e-8 makes the next order, O(delta0) relative
     *        on a part that is (delta0/delta)^4 of any cumulative, irrelevant)
     */
    static Values analytic(double delta) {
      const double d4 = delta * delta * delta * delta;
      return Values { (constants::PI / 24.0) * d4,
                      (constants::PI / 12.0) * d4,
                      (constants::PI / 24.0) * d4 };
    }

  private:
    // integrands in t = ln(delta), including the Jacobian dk = delta dt; q_k from
    // the chosen source (the Fit branch is the step-1 code, unchanged)
    Values integrand(double t) const {
      const double delta = std::exp(t);
      const double k     = 2.0 + delta;
      const double s     = sigma_tilde_delta(delta);
      const double q     = m_qk_source == QkSource::Born ? born_q_k_delta(delta) : q_k_delta(delta);
      return Values { k * s * delta, k * s * q * delta, s * q * delta };
    }

    Values cell(double a, double b) const {
      Values       out;
      const double mid  = 0.5 * (a + b);
      const double half = 0.5 * (b - a);
      for (int i = 0; i < m_gl.order(); ++i) {
        const Values f  = integrand(mid + half * m_gl.x[i]);
        const double w  = half * m_gl.w[i];
        out.Phi        += w * f.Phi;
        out.phi        += w * f.phi;
        out.psi        += w * f.psi;
      }
      return out;
    }

    double              m_delta0;
    double              m_kmax;
    double              m_h;
    GaussLegendre       m_gl;
    std::vector<double> m_t;
    std::vector<Values> m_cum;
    std::size_t         m_junction { 0 };
    QkSource            m_qk_source { QkSource::Fit };
  };

  // ------------------------------------------------------------------------ //
  // Photon baths: number spectrum n~(eps) normalised to unit integral over the
  // retained range [eps_min, eps_max]
  // ------------------------------------------------------------------------ //
  class PhotonBath {
  public:
    virtual ~PhotonBath() = default;

    virtual std::string type() const = 0;

    // (key, JSON value) pairs of the bath specification, for the file header
    virtual std::vector<std::pair<std::string, std::string>> parameters() const = 0;

    virtual bool is_mono() const {
      return false;
    }

    virtual double eps0() const {
      throw std::logic_error("PhotonBath: eps0 is defined only for mono baths");
    }

    // retained range (for a mono bath both equal eps0)
    virtual double eps_min() const = 0;
    virtual double eps_max() const = 0;

    // normalised number spectrum (not meaningful for mono)
    virtual double density(double eps) const = 0;

    // interior breakpoints of the spectrum (table nodes)
    virtual std::vector<double> breakpoints() const {
      return {};
    }

    // discarded number and energy fractions of the unclipped spectrum
    virtual double clipped_number_fraction() const {
      return 0.0;
    }

    virtual double clipped_energy_fraction() const {
      return 0.0;
    }

    // normalisation constant C of the number spectrum n~ = C f(eps) on the
    // retained range (1 for a mono bath)
    virtual double number_normalization() const {
      return 1.0;
    }

    // mean photon energy of the normalised spectrum on the retained range
    virtual double mean_eps() const = 0;

    // source file of a tabulated spectrum, "" otherwise
    virtual std::string source_file() const {
      return "";
    }

    // the cutoff energy that sets Gamma_th: eps0 or eps_max
    double eps_cut() const {
      return is_mono() ? eps0() : eps_max();
    }

    double gamma_th() const {
      const double e = eps_cut();
      return 1.0 / e + 0.25 * e;
    }

    // Gamma_th beta_th = 1/e - e/4 exactly (so that Gamma_th (1 + beta_th) = 2/e)
    double gamma_th_beta_th() const {
      const double e = eps_cut();
      return 1.0 / e - 0.25 * e;
    }

  protected:
    static void require_eps_max_below_two(double eps_max) {
      if (!(eps_max < 2.0)) {
        throw std::invalid_argument(
          "PhotonBath: eps_max = " + std::to_string(eps_max) +
          " must be < 2 (ion loss positive for every outcome)");
      }
    }
  };

  class MonoBath : public PhotonBath {
  public:
    explicit MonoBath(double eps0) : m_eps0 { eps0 } {
      if (!(eps0 > 0.0)) {
        throw std::invalid_argument("MonoBath: eps0 must be > 0");
      }
      require_eps_max_below_two(eps0);
    }

    std::string type() const override {
      return "mono";
    }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
      return {
        { "eps0", json_number(m_eps0) }
      };
    }

    bool is_mono() const override {
      return true;
    }

    double eps0() const override {
      return m_eps0;
    }

    double eps_min() const override {
      return m_eps0;
    }

    double eps_max() const override {
      return m_eps0;
    }

    double density(double) const override {
      throw std::logic_error("MonoBath: density is a delta function");
    }

    double mean_eps() const override {
      return m_eps0;
    }

    // shortest decimal representation that round-trips in double
    static std::string json_number(double v) {
      char buf[64];
      for (int prec = 15; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) {
          break;
        }
      }
      return std::string(buf);
    }

  private:
    double m_eps0;
  };

  inline std::string json_number(double v) {
    return MonoBath::json_number(v);
  }

  /**
   * @brief JSON string literal: quotes, backslashes and control characters escaped
   */
  inline std::string json_string(const std::string& str) {
    std::string out = "\"";
    for (const char c : str) {
      const auto ch = static_cast<unsigned char>(c);
      switch (ch) {
        case '"':
          out += "\\\"";
          break;
        case '\\':
          out += "\\\\";
          break;
        case '\b':
          out += "\\b";
          break;
        case '\f':
          out += "\\f";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        case '\t':
          out += "\\t";
          break;
        default:
          if (ch < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
            out += buf;
          } else {
            out += c;
          }
      }
    }
    return out + "\"";
  }

  class PowerlawBath : public PhotonBath {
  public:
    PowerlawBath(double eps_min, double eps_max, double index)
      : m_eps_min { eps_min }
      , m_eps_max { eps_max }
      , m_index { index } {
      if (!(eps_min > 0.0) || !(eps_max > eps_min)) {
        throw std::invalid_argument(
          "PowerlawBath: require 0 < eps_min < eps_max");
      }
      require_eps_max_below_two(eps_max);
      m_norm = 1.0 / power_integral(eps_min, eps_max, -index);
    }

    std::string type() const override {
      return "powerlaw";
    }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
      return {
        { "eps_min", json_number(m_eps_min) },
        { "eps_max", json_number(m_eps_max) },
        {   "index",   json_number(m_index) }
      };
    }

    double eps_min() const override {
      return m_eps_min;
    }

    double eps_max() const override {
      return m_eps_max;
    }

    double density(double eps) const override {
      if (eps < m_eps_min || eps > m_eps_max) {
        return 0.0;
      }
      return m_norm * std::pow(eps, -m_index);
    }

    double index() const {
      return m_index;
    }

    double number_normalization() const override {
      return m_norm;
    }

    double mean_eps() const override {
      return m_norm * power_integral(m_eps_min, m_eps_max, 1.0 - m_index);
    }

    /**
     * @brief int_a^b eps^p deps (exact)
     */
    static double power_integral(double a, double b, double p) {
      // (b^t - a^t)/t = a^t L expm1(x)/x with t = p + 1, L = ln(b/a), x = t L;
      // the limit expm1(x)/x -> 1 is taken on |x|, not on |t| alone
      const double t = p + 1.0;
      const double L = std::log(b / a);
      const double x = t * L;
      if (std::abs(x) < 1.0e-18) {
        return std::pow(a, t) * L;
      }
      return std::pow(a, t) * L * (std::expm1(x) / x);
    }

  private:
    double m_eps_min, m_eps_max, m_index, m_norm;
  };

  class BlackbodyBath : public PhotonBath {
  public:
    BlackbodyBath(double theta, double eps_min, double eps_max)
      : m_theta { theta }
      , m_eps_min { eps_min }
      , m_eps_max { eps_max } {
      if (!(theta > 0.0)) {
        throw std::invalid_argument("BlackbodyBath: theta must be > 0");
      }
      if (eps_min < 0.0 || !(eps_max > eps_min)) {
        throw std::invalid_argument(
          "BlackbodyBath: require 0 <= eps_min < eps_max");
      }
      require_eps_max_below_two(eps_max);
      // retained and discarded moments of the unclipped Planck spectrum
      const double n_total = 2.0 * constants::ZETA3 * theta * theta * theta;
      const double e_total = (std::pow(constants::PI, 4) / 15.0) * theta *
                             theta * theta * theta;
      const double n_ret  = moment(eps_min, eps_max, 0);
      const double e_ret  = moment(eps_min, eps_max, 1);
      const double n_head = eps_min > 0.0 ? moment(0.0, eps_min, 0) : 0.0;
      const double e_head = eps_min > 0.0 ? moment(0.0, eps_min, 1) : 0.0;
      const double n_tail = tail_number(eps_max / theta) * theta * theta * theta;
      const double e_tail = tail_energy(eps_max / theta) * theta * theta *
                            theta * theta;
      m_norm           = 1.0 / n_ret;
      m_mean           = e_ret / n_ret;
      m_clipped_number = (n_head + n_tail) / n_total;
      m_clipped_energy = (e_head + e_tail) / e_total;
      m_closure_number = (n_head + n_ret + n_tail) / n_total - 1.0;
      m_closure_energy = (e_head + e_ret + e_tail) / e_total - 1.0;
    }

    std::string type() const override {
      return "blackbody";
    }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
      return {
        {   "theta",   json_number(m_theta) },
        { "eps_min", json_number(m_eps_min) },
        { "eps_max", json_number(m_eps_max) }
      };
    }

    double eps_min() const override {
      return m_eps_min;
    }

    double eps_max() const override {
      return m_eps_max;
    }

    double density(double eps) const override {
      if (eps < m_eps_min || eps > m_eps_max) {
        return 0.0;
      }
      return m_norm * planck(eps);
    }

    double clipped_number_fraction() const override {
      return m_clipped_number;
    }

    double clipped_energy_fraction() const override {
      return m_clipped_energy;
    }

    // residuals of (head + retained + tail) / total - 1 (quadrature check)
    double closure_number() const {
      return m_closure_number;
    }

    double closure_energy() const {
      return m_closure_energy;
    }

    double theta() const {
      return m_theta;
    }

    double number_normalization() const override {
      return m_norm;
    }

    double mean_eps() const override {
      return m_mean;
    }

    // unnormalised Planck number spectrum eps^2 / (exp(eps/theta) - 1)
    double planck(double eps) const {
      if (eps <= 0.0) {
        return 0.0;
      }
      return eps * eps / std::expm1(eps / m_theta);
    }

    /**
     * @brief int_a^inf x^2 / (e^x - 1) dx = sum_n e^{-na} (a^2/n + 2a/n^2 + 2/n^3)
     */
    static double tail_number(double a) {
      double sum = 0.0;
      for (int n = 1; n < 2000000; ++n) {
        const double term = std::exp(-n * a) *
                            (a * a / n + 2.0 * a / (static_cast<double>(n) * n) +
                             2.0 / (static_cast<double>(n) * n * n));
        sum += term;
        if (term < 1.0e-18 * sum) {
          break;
        }
      }
      return sum;
    }

    /**
     * @brief int_a^inf x^3 / (e^x - 1) dx
     *        = sum_n e^{-na} (a^3/n + 3a^2/n^2 + 6a/n^3 + 6/n^4)
     */
    static double tail_energy(double a) {
      double sum = 0.0;
      for (int n = 1; n < 2000000; ++n) {
        const double dn   = static_cast<double>(n);
        const double term = std::exp(-n * a) *
                            (a * a * a / dn + 3.0 * a * a / (dn * dn) +
                             6.0 * a / (dn * dn * dn) + 6.0 / (dn * dn * dn * dn));
        sum += term;
        if (term < 1.0e-18 * sum) {
          break;
        }
      }
      return sum;
    }

  private:
    // int_a^b eps^m planck(eps) deps by composite Gauss-Legendre
    double moment(double a, double b, int m) const {
      const GaussLegendre gl { 16 };
      const int           npan = std::max(
        64,
        static_cast<int>(std::ceil((b - a) / (0.05 * m_theta))));
      const double h   = (b - a) / npan;
      double       sum = 0.0;
      for (int p = 0; p < npan; ++p) {
        const double lo   = a + h * p;
        const double mid  = lo + 0.5 * h;
        const double half = 0.5 * h;
        for (int i = 0; i < gl.order(); ++i) {
          const double e = mid + half * gl.x[i];
          sum += half * gl.w[i] * planck(e) * (m == 0 ? 1.0 : std::pow(e, m));
        }
      }
      return sum;
    }

    double m_theta, m_eps_min, m_eps_max;
    double m_norm { 1.0 }, m_mean { 0.0 };
    double m_clipped_number { 0.0 }, m_clipped_energy { 0.0 };
    double m_closure_number { 0.0 }, m_closure_energy { 0.0 };
  };

  /**
   * @brief Tabulated bath: two columns (eps, dn/deps), log-log linear
   *        interpolation between nodes (power-law segments), zero outside.
   *        Optional clip range [eps_min, eps_max]; eps_max < 2 required.
   */
  class TableBath : public PhotonBath {
  public:
    TableBath(const std::string& file,
              double             eps_min_clip = -1.0,
              double             eps_max_clip = -1.0)
      : m_file { file } {
      std::ifstream in(file);
      if (!in) {
        throw std::invalid_argument("TableBath: cannot open " + file);
      }
      std::string line;
      std::size_t lineno = 0;
      while (std::getline(in, line)) {
        ++lineno;
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
          line = line.substr(0, hash);
        }
        if (line.find_first_not_of(" \t\r") == std::string::npos) {
          continue; // blank or comment-only line
        }
        std::istringstream  ss(line);
        std::string         tok;
        std::vector<double> vals;
        while (ss >> tok) {
          double v = 0.0;
          if (!parse_number_strict(tok, v)) {
            throw std::invalid_argument("TableBath: " + file + ":" +
                                        std::to_string(lineno) + ": token '" +
                                        tok + "' is not a finite number");
          }
          vals.push_back(v);
        }
        if (vals.size() != 2) {
          throw std::invalid_argument(
            "TableBath: " + file + ":" + std::to_string(lineno) +
            ": expected exactly two columns (eps, dn/deps), found " +
            std::to_string(vals.size()));
        }
        m_eps.push_back(vals[0]);
        m_n.push_back(vals[1]);
      }
      init(eps_min_clip, eps_max_clip);
    }

    TableBath(std::vector<double> eps,
              std::vector<double> dn_deps,
              double              eps_min_clip = -1.0,
              double              eps_max_clip = -1.0)
      : m_file { "" }
      , m_eps { std::move(eps) }
      , m_n { std::move(dn_deps) } {
      init(eps_min_clip, eps_max_clip);
    }

    std::string type() const override {
      return "table";
    }

    std::vector<std::pair<std::string, std::string>> parameters() const override {
      return {
        {    "file",          json_string(m_file) },
        { "eps_min",       json_number(m_eps_min) },
        { "eps_max",       json_number(m_eps_max) },
        { "n_nodes", std::to_string(m_eps.size()) }
      };
    }

    double eps_min() const override {
      return m_eps_min;
    }

    double eps_max() const override {
      return m_eps_max;
    }

    double density(double eps) const override {
      if (eps < m_eps_min || eps > m_eps_max) {
        return 0.0;
      }
      const std::size_t s = segment(eps);
      return m_norm * m_n[s] * std::pow(eps / m_eps[s], m_p[s]);
    }

    std::vector<double> breakpoints() const override {
      std::vector<double> b;
      for (const double e : m_eps) {
        if (e > m_eps_min && e < m_eps_max) {
          b.push_back(e);
        }
      }
      return b;
    }

    double clipped_number_fraction() const override {
      return m_clipped_number;
    }

    double clipped_energy_fraction() const override {
      return m_clipped_energy;
    }

    double number_normalization() const override {
      return m_norm;
    }

    double mean_eps() const override {
      return m_mean;
    }

    std::string source_file() const override {
      return m_file;
    }

  private:
    void init(double eps_min_clip, double eps_max_clip) {
      if (m_eps.size() < 2 || m_eps.size() != m_n.size()) {
        throw std::invalid_argument(
          "TableBath: need at least two (eps, dn/deps) rows");
      }
      for (std::size_t i = 0; i < m_eps.size(); ++i) {
        if (!(m_eps[i] > 0.0) || !(m_n[i] > 0.0)) {
          throw std::invalid_argument(
            "TableBath: eps and dn/deps must be > 0 (log-log interpolation)");
        }
        if (i > 0 && !(m_eps[i] > m_eps[i - 1])) {
          throw std::invalid_argument(
            "TableBath: eps must be strictly increasing");
        }
      }
      m_p.resize(m_eps.size() - 1);
      for (std::size_t i = 0; i + 1 < m_eps.size(); ++i) {
        m_p[i] = std::log(m_n[i + 1] / m_n[i]) / std::log(m_eps[i + 1] / m_eps[i]);
      }
      m_eps_min = eps_min_clip > 0.0 ? std::max(eps_min_clip, m_eps.front())
                                     : m_eps.front();
      m_eps_max = eps_max_clip > 0.0 ? std::min(eps_max_clip, m_eps.back())
                                     : m_eps.back();
      if (!(m_eps_max > m_eps_min)) {
        throw std::invalid_argument("TableBath: empty retained range");
      }
      require_eps_max_below_two(m_eps_max);
      const double n_total = integral(m_eps.front(), m_eps.back(), 0);
      const double e_total = integral(m_eps.front(), m_eps.back(), 1);
      const double n_ret   = integral(m_eps_min, m_eps_max, 0);
      const double e_ret   = integral(m_eps_min, m_eps_max, 1);
      m_norm               = 1.0 / n_ret;
      m_mean               = e_ret / n_ret;
      m_clipped_number     = 1.0 - n_ret / n_total;
      m_clipped_energy     = 1.0 - e_ret / e_total;
    }

    std::size_t segment(double eps) const {
      auto        it = std::upper_bound(m_eps.begin(), m_eps.end(), eps);
      std::size_t s  = static_cast<std::size_t>(it - m_eps.begin());
      if (s == 0) {
        return 0;
      }
      if (s >= m_eps.size()) {
        return m_eps.size() - 2;
      }
      return s - 1;
    }

    // exact int_a^b eps^m n(eps) deps over the power segments (unnormalised)
    double integral(double a, double b, int m) const {
      double sum = 0.0;
      for (std::size_t s = 0; s + 1 < m_eps.size(); ++s) {
        const double lo = std::max(a, m_eps[s]);
        const double hi = std::min(b, m_eps[s + 1]);
        if (hi <= lo) {
          continue;
        }
        const double A  = m_n[s] * std::pow(m_eps[s], -m_p[s]);
        sum            += A * PowerlawBath::power_integral(lo, hi, m_p[s] + m);
      }
      return sum;
    }

    std::string         m_file;
    std::vector<double> m_eps, m_n, m_p;
    double m_eps_min { 0.0 }, m_eps_max { 0.0 }, m_norm { 1.0 }, m_mean { 0.0 };
    double m_clipped_number { 0.0 }, m_clipped_energy { 0.0 };
  };

  // ------------------------------------------------------------------------ //
  // Exact-in-beta rates
  // ------------------------------------------------------------------------ //
  struct RateValues {
    double gamma { 0.0 };
    double beta { 0.0 };
    double J { 0.0 };     // interactions per unit photon density, alpha r_e^2 c
    double I { 0.0 };     // ion energy loss (m_e c^2) per unit photon density
    double Dmean { 0.0 }; // I / J
    double eps_mean { 0.0 }; // interaction-weighted mean photon energy
  };

  class BetheHeitlerRates {
  public:
    BetheHeitlerRates(std::shared_ptr<const PhotonBath> bath,
                      const Quadrature&                 q = Quadrature {})
      : m_bath { std::move(bath) }
      , m_q { q }
      , m_cum { q }
      , m_gl { q.gl_order } {
      if (!m_bath) {
        throw std::invalid_argument("BetheHeitlerRates: null bath");
      }
      if (!(m_q.h_log_eps > 0.0)) {
        throw std::invalid_argument("BetheHeitlerRates: h_log_eps must be > 0");
      }
    }

    const PhotonBath& bath() const {
      return *m_bath;
    }

    const Cumulatives& cumulatives() const {
      return m_cum;
    }

    const Quadrature& quadrature() const {
      return m_q;
    }

    double gamma_th() const {
      return m_bath->gamma_th();
    }

    /**
     * @brief eps_c Gamma (1 + beta) - 2 for Gamma = Gamma_th + x, cancellation-free
     */
    double delta_hi_cut(double x) const {
      const double gth   = m_bath->gamma_th();
      const double gbth  = m_bath->gamma_th_beta_th();
      const double gamma = gth + x;
      const double gb    = std::sqrt((gamma - 1.0) * (gamma + 1.0));
      return m_bath->eps_cut() * x * (1.0 + (gamma + gth) / (gb + gbth));
    }

    /**
     * @brief Rates at Gamma = Gamma_th + x (x > 0)
     */
    RateValues at_offset(double x) const {
      if (!(x > 0.0)) {
        throw std::invalid_argument(
          "BetheHeitlerRates: x = Gamma - Gamma_th must be > 0");
      }
      RateValues   out;
      const double gamma = m_bath->gamma_th() + x;
      const double gb = std::sqrt((gamma - 1.0) * (gamma + 1.0)); // Gamma beta
      const double beta  = gb / gamma;
      const double A     = gamma + gb; // Gamma (1 + beta)
      out.gamma          = gamma;
      out.beta           = beta;
      const double d_max = delta_hi_cut(x); // eps_c A - 2

      if (m_bath->is_mono()) {
        const double              e0 = m_bath->eps0();
        const Cumulatives::Values c  = m_cum(d_max);
        // eps'_lo = Gamma eps (1 - beta) = eps / A < 2 for eps < 2: Phi(lo) = 0
        out.J                        = c.Phi / (2.0 * gamma * gb * e0 * e0);
        out.I        = (c.phi - (e0 / gamma) * c.psi) / (2.0 * gb * e0 * e0);
        out.eps_mean = e0;
        out.Dmean    = out.J > 0.0 ? out.I / out.J : 0.0;
        return out;
      }

      // broad bath: integrate in v = ln(eps / eps_th), eps_th = 2 / A
      const double eps_th = 2.0 / A;
      const double v_max  = std::log1p(0.5 * d_max); // ln(eps_max / eps_th)
      const double v_min  = m_bath->eps_min() > eps_th
                              ? std::log(m_bath->eps_min() / eps_th)
                              : 0.0;
      if (!(v_max > v_min)) {
        return out;
      }
      std::vector<double> edges { v_min };
      const double        v4 = std::log(2.0); // eps' = 4 (series junction)
      if (v4 > v_min && v4 < v_max) {
        edges.push_back(v4);
      }
      for (const double eb : m_bath->breakpoints()) {
        const double vb = std::log(eb / eps_th);
        if (vb > v_min && vb < v_max) {
          edges.push_back(vb);
        }
      }
      edges.push_back(v_max);
      std::sort(edges.begin(), edges.end());

      double SJ = 0.0, SI = 0.0, SE = 0.0;
      for (std::size_t s = 0; s + 1 < edges.size(); ++s) {
        const double L = edges[s + 1] - edges[s];
        if (!(L > 0.0)) {
          continue;
        }
        const int    npan = std::max(1,
                                  static_cast<int>(std::ceil(L / m_q.h_log_eps)));
        const double hp   = L / npan;
        for (int p = 0; p < npan; ++p) {
          const double lo   = edges[s] + hp * p;
          const double mid  = lo + 0.5 * hp;
          const double half = 0.5 * hp;
          for (int i = 0; i < m_gl.order(); ++i) {
            const double v   = mid + half * m_gl.x[i];
            const double w   = half * m_gl.w[i];
            const double eps = eps_th * std::exp(v);
            const double n   = m_bath->density(eps);
            if (!(n > 0.0)) {
              continue;
            }
            const double              d_hi  = 2.0 * std::expm1(v); // eps A - 2
            const double              d_lo  = eps / A - 2.0; // < 0 for eps < 2
            const Cumulatives::Values hi    = m_cum(d_hi);
            const Cumulatives::Values lo_   = m_cum(d_lo);
            const double              dPhi  = hi.Phi - lo_.Phi;
            const double              dphi  = hi.phi - lo_.phi;
            const double              dpsi  = hi.psi - lo_.psi;
            // d eps = eps dv; integrands n/eps^2 (...) and n/eps (...)
            SJ                             += w * n * dPhi / eps;
            SI += w * n * (dphi - (eps / gamma) * dpsi) / eps;
            SE += w * n * dPhi;
          }
        }
      }
      out.J        = SJ / (2.0 * gamma * gb);
      out.I        = SI / (2.0 * gb);
      out.eps_mean = SJ > 0.0 ? SE / SJ : 0.0;
      out.Dmean    = out.J > 0.0 ? out.I / out.J : 0.0;
      return out;
    }

    RateValues at_gamma(double gamma) const {
      return at_offset(gamma - m_bath->gamma_th());
    }

    /**
     * @brief Ultra-relativistic limits (contract section 2):
     *        J_UR = (1/2Gamma^2) int n~/eps^2 Phi(2 Gamma eps),
     *        I_UR = (1/2Gamma)   int n~/eps^2 phi(2 Gamma eps)
     *        computed with this object's own cumulatives.
     */
    RateValues ur_limit(double gamma) const {
      RateValues out;
      out.gamma = gamma;
      out.beta  = 1.0;
      if (m_bath->is_mono()) {
        const double e0 = m_bath->eps0();
        const auto   c  = m_cum(2.0 * gamma * e0 - 2.0);
        out.J           = c.Phi / (2.0 * gamma * gamma * e0 * e0);
        out.I           = c.phi / (2.0 * gamma * e0 * e0);
        out.eps_mean    = e0;
      } else {
        const double        eps_th = 1.0 / gamma;
        const double        v_max  = std::log(m_bath->eps_max() / eps_th);
        const double        v_min  = m_bath->eps_min() > eps_th
                                       ? std::log(m_bath->eps_min() / eps_th)
                                       : 0.0;
        std::vector<double> edges { v_min };
        const double        v4 = std::log(2.0);
        if (v4 > v_min && v4 < v_max) {
          edges.push_back(v4);
        }
        for (const double eb : m_bath->breakpoints()) {
          const double vb = std::log(eb / eps_th);
          if (vb > v_min && vb < v_max) {
            edges.push_back(vb);
          }
        }
        edges.push_back(v_max);
        std::sort(edges.begin(), edges.end());
        double SJ = 0.0, SI = 0.0, SE = 0.0;
        for (std::size_t s = 0; s + 1 < edges.size(); ++s) {
          const double L    = edges[s + 1] - edges[s];
          const int    npan = std::max(1,
                                    static_cast<int>(std::ceil(L / m_q.h_log_eps)));
          const double hp   = L / npan;
          for (int p = 0; p < npan; ++p) {
            const double lo   = edges[s] + hp * p;
            const double mid  = lo + 0.5 * hp;
            const double half = 0.5 * hp;
            for (int i = 0; i < m_gl.order(); ++i) {
              const double v   = mid + half * m_gl.x[i];
              const double w   = half * m_gl.w[i];
              const double eps = eps_th * std::exp(v);
              const double n   = m_bath->density(eps);
              if (!(n > 0.0)) {
                continue;
              }
              const auto c  = m_cum(2.0 * std::expm1(v));
              SJ           += w * n * c.Phi / eps;
              SI           += w * n * c.phi / eps;
              SE           += w * n * c.Phi;
            }
          }
        }
        out.J        = SJ / (2.0 * gamma * gamma);
        out.I        = SI / (2.0 * gamma);
        out.eps_mean = SJ > 0.0 ? SE / SJ : 0.0;
      }
      out.Dmean = out.J > 0.0 ? out.I / out.J : 0.0;
      return out;
    }

  private:
    std::shared_ptr<const PhotonBath> m_bath;
    Quadrature                        m_q;
    Cumulatives                       m_cum;
    GaussLegendre                     m_gl;
  };

  // ------------------------------------------------------------------------ //
  // The startup tables (host)
  // ------------------------------------------------------------------------ //
  struct HadronicTablesHost {
    // grid
    double gamma_th { 0.0 }; // threshold Lorentz factor (0 if the bath has no cutoff)
    double      gamma_min { 0.0 }; // lower edge (I = 1e-12 of its peak)
    double      gamma_max { 0.0 }; // upper edge (envelope or gamma_p_max)
    double      y_min { 0.0 };
    double      y_max { 0.0 };
    std::size_t n_fine { 0 };   // nodes in the first decade (incl. both ends)
    std::size_t n_coarse { 0 }; // nodes after the first decade (excl. y_min + 1)
    int    nodes_per_decade_fine { constants::NODES_PER_DECADE_FINE };
    int    nodes_per_decade_coarse { constants::NODES_PER_DECADE_COARSE };
    // envelope
    double mu_r { 0.0 };
    double gamma_p_max_input { 0.0 }; // 0 = auto
    double eps_prime_max { 0.0 };     // min(0.05 mu_r, 1000)
    double gamma_envelope { 0.0 };    // Gamma at which eps'_hi = eps_prime_max
    double I_peak { 0.0 };
    double gamma_at_I_peak { 0.0 }; // Gamma of the peak of I (Gamma_max if there)
    // bath
    double              clipped_number_fraction { 0.0 };
    double              clipped_energy_fraction { 0.0 };
    // nodes
    std::vector<double> gamma, beta, y, J, I, Dmean, eps_mean;

    std::size_t size() const {
      return gamma.size();
    }
  };

  /**
   * @brief Gamma (1 + beta) = A  <=>  Gamma = (A + 1/A) / 2
   */
  inline double gamma_from_A(double A) {
    return 0.5 * (A + 1.0 / A);
  }

  /**
   * @brief Upper edge of the table: Gamma at which eps'_hi = Gamma eps_c (1 +
   * beta) reaches min(0.05 mu_r, 1000), or gamma_p_max if smaller (> 0)
   */
  inline double table_upper_edge(const PhotonBath& bath,
                                 double            mu_r,
                                 double            gamma_p_max,
                                 double*           eps_prime_max_out = nullptr,
                                 double* gamma_envelope_out = nullptr) {
    if (!std::isfinite(mu_r) || !(mu_r > 0.0)) {
      throw std::invalid_argument(
        "table_upper_edge: physical_mass_ratio must be finite and positive, "
        "got " +
        std::to_string(mu_r));
    }
    const double kmax = std::min(constants::ENVELOPE_FRACTION * mu_r,
                                 constants::K_MAX);
    if (!(kmax > constants::K_MIN)) {
      throw std::invalid_argument(
        "table_upper_edge: no Bethe-Heitler-active interval: "
        "physical_mass_ratio = " +
        std::to_string(mu_r) + " gives a kernel cap min(0.05 mu_r, 1000) = " +
        std::to_string(kmax) + " <= 2");
    }
    if (gamma_p_max != 0.0 &&
        (!std::isfinite(gamma_p_max) || !(gamma_p_max > 1.0))) {
      throw std::invalid_argument(
        "table_upper_edge: gamma_p_max must be a finite number > 1 (0 = "
        "auto), got " +
        std::to_string(gamma_p_max));
    }
    const double A    = kmax / bath.eps_cut();
    const double genv = gamma_from_A(A);
    if (eps_prime_max_out) {
      *eps_prime_max_out = kmax;
    }
    if (gamma_envelope_out) {
      *gamma_envelope_out = genv;
    }
    return gamma_p_max > 0.0 ? std::min(gamma_p_max, genv) : genv;
  }

  /**
   * @brief Build the registered table: y = log10(Gamma - Gamma_th), 256 nodes
   *        per decade over the first decade above the lower edge, 64 per decade
   *        after, last node exactly at the upper edge.
   */
  inline HadronicTablesHost build_tables(const BetheHeitlerRates& rates,
                                         double                   mu_r,
                                         double gamma_p_max = 0.0) {
    using namespace constants;
    const PhotonBath&  bath = rates.bath();
    HadronicTablesHost T;
    T.mu_r              = mu_r;
    T.gamma_p_max_input = gamma_p_max;
    T.gamma_th          = bath.gamma_th();
    T.gamma_max         = table_upper_edge(bath,
                                   mu_r,
                                   gamma_p_max,
                                   &T.eps_prime_max,
                                   &T.gamma_envelope);
    if (!(T.gamma_max > T.gamma_th)) {
      throw std::invalid_argument(
        "build_tables: upper edge Gamma_max = " + std::to_string(T.gamma_max) +
        " does not exceed Gamma_th = " + std::to_string(T.gamma_th));
    }
    T.clipped_number_fraction = bath.clipped_number_fraction();
    T.clipped_energy_fraction = bath.clipped_energy_fraction();

    const double x_max = T.gamma_max - T.gamma_th;
    T.y_max            = std::log10(x_max);

    // peak of I over the active interval: coarse scan in y, then local refinement
    double y_lo_scan = T.y_max - 14.0;
    double y_peak    = T.y_max;
    T.I_peak         = 0.0;
    {
      const int    nscan = 141;
      const double dy    = (T.y_max - y_lo_scan) / (nscan - 1);
      for (int i = 0; i < nscan; ++i) {
        const double yy = y_lo_scan + dy * i;
        const double Ii = rates.at_offset(std::pow(10.0, yy)).I;
        if (Ii > T.I_peak) {
          T.I_peak = Ii;
          y_peak   = yy;
        }
      }
      // golden-section refinement around the scan maximum
      double a        = std::max(y_lo_scan, y_peak - dy),
             b        = std::min(T.y_max, y_peak + dy);
      const double gr = 0.5 * (std::sqrt(5.0) - 1.0);
      double       c = b - gr * (b - a), d = a + gr * (b - a);
      double       fc = rates.at_offset(std::pow(10.0, c)).I;
      double       fd = rates.at_offset(std::pow(10.0, d)).I;
      for (int it = 0; it < 60; ++it) {
        if (fc > fd) {
          b  = d;
          d  = c;
          fd = fc;
          c  = b - gr * (b - a);
          fc = rates.at_offset(std::pow(10.0, c)).I;
        } else {
          a  = c;
          c  = d;
          fc = fd;
          d  = a + gr * (b - a);
          fd = rates.at_offset(std::pow(10.0, d)).I;
        }
      }
      const double Im = std::max(fc, fd);
      if (Im > T.I_peak) {
        T.I_peak = Im;
        y_peak   = fc > fd ? c : d;
      }
      T.gamma_at_I_peak = (std::abs(y_peak - T.y_max) < ENDPOINT_COALESCING)
                            ? T.gamma_max
                            : T.gamma_th + std::pow(10.0, y_peak);
    }

    // lower edge: I(y) = LOWER_EDGE_FRACTION * I_peak by bisection in y
    const double target = LOWER_EDGE_FRACTION * T.I_peak;
    {
      double a = y_lo_scan, b = y_peak;
      double fa = rates.at_offset(std::pow(10.0, a)).I;
      if (fa >= target) {
        // extend downwards (should not happen for any admissible bath)
        for (int it = 0; it < 20 && fa >= target; ++it) {
          a  -= 2.0;
          fa  = rates.at_offset(std::pow(10.0, a)).I;
        }
        if (fa >= target) {
          throw std::runtime_error(
            "build_tables: cannot bracket the lower edge");
        }
      }
      for (int it = 0; it < 200; ++it) {
        const double m  = 0.5 * (a + b);
        const double fm = rates.at_offset(std::pow(10.0, m)).I;
        if (fm < target) {
          a = m;
        } else {
          b = m;
        }
        if (b - a < 1.0e-13) {
          break;
        }
      }
      T.y_min = 0.5 * (a + b);
    }
    T.gamma_min = T.gamma_th + std::pow(10.0, T.y_min);

    // nodes
    std::vector<double> ys;
    const double        y_tiny = ENDPOINT_COALESCING;
    const double        dyf    = 1.0 / NODES_PER_DECADE_FINE;
    const double        dyc    = 1.0 / NODES_PER_DECADE_COARSE;
    for (int i = 0; i <= NODES_PER_DECADE_FINE; ++i) {
      const double yy = T.y_min + dyf * i;
      if (yy >= T.y_max - y_tiny) {
        break;
      }
      ys.push_back(yy);
    }
    T.n_fine = ys.size();
    if (T.y_min + 1.0 < T.y_max - y_tiny) {
      for (int j = 1;; ++j) {
        const double yy = T.y_min + 1.0 + dyc * j;
        if (yy >= T.y_max - y_tiny) {
          break;
        }
        ys.push_back(yy);
      }
    }
    ys.push_back(T.y_max);
    T.n_coarse = ys.size() - T.n_fine;

    T.y.reserve(ys.size());
    for (const double yy : ys) {
      const double     x = std::pow(10.0, yy);
      const RateValues r = rates.at_offset(x);
      T.y.push_back(yy);
      T.gamma.push_back(r.gamma);
      T.beta.push_back(r.beta);
      T.J.push_back(r.J);
      T.I.push_back(r.I);
      T.Dmean.push_back(r.Dmean);
      T.eps_mean.push_back(r.eps_mean);
    }
    // the last node is exactly the upper edge
    T.gamma.back() = T.gamma_max;
    return T;
  }

  /**
   * @brief Sanity checks of a built table: monotone Gamma, finite, J, I > 0
   */
  inline void validate_tables(const HadronicTablesHost& T) {
    if (T.size() < 2) {
      throw std::runtime_error("validate_tables: fewer than two nodes");
    }
    for (std::size_t i = 0; i < T.size(); ++i) {
      if (!std::isfinite(T.gamma[i]) || !std::isfinite(T.J[i]) ||
          !std::isfinite(T.I[i]) || !std::isfinite(T.Dmean[i]) ||
          !std::isfinite(T.eps_mean[i]) || !std::isfinite(T.beta[i])) {
        throw std::runtime_error(
          "validate_tables: non-finite entry at node " + std::to_string(i));
      }
      if (!(T.J[i] > 0.0) || !(T.I[i] > 0.0)) {
        throw std::runtime_error(
          "validate_tables: J or I <= 0 at node " + std::to_string(i));
      }
      if (i > 0 && !(T.gamma[i] > T.gamma[i - 1])) {
        throw std::runtime_error(
          "validate_tables: Gamma not increasing at node " + std::to_string(i));
      }
    }
  }

  // ------------------------------------------------------------------------ //
  // Bath specification (string-based, for the dump tool and the tests)
  // ------------------------------------------------------------------------ //
  struct BathSpec {
    std::string type; // mono | powerlaw | blackbody | table
    double      eps0 { 0.0 };
    double      eps_min { -1.0 };
    double      eps_max { -1.0 };
    double      index { 0.0 };
    double      theta { 0.0 };
    std::string file;
  };

  inline std::shared_ptr<const PhotonBath> make_bath(const BathSpec& s) {
    if (s.type == "mono") {
      return std::make_shared<MonoBath>(s.eps0);
    }
    if (s.type == "powerlaw") {
      return std::make_shared<PowerlawBath>(s.eps_min, s.eps_max, s.index);
    }
    if (s.type == "blackbody") {
      return std::make_shared<BlackbodyBath>(s.theta,
                                             s.eps_min < 0.0 ? 0.0 : s.eps_min,
                                             s.eps_max);
    }
    if (s.type == "table") {
      return std::make_shared<TableBath>(s.file, s.eps_min, s.eps_max);
    }
    throw std::invalid_argument(
      "make_bath: unknown photon_field.type '" + s.type + "'");
  }

} // namespace arch::bh

#endif // ARCHETYPES_HADRONIC_BH_PHYSICS_H
