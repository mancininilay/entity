/**
 * @file archetypes/hadronic/bh_born_qk.h
 * @brief q_k^Born(eps'): the mean momentum absorbed by the static source along
 *        the rest-frame photon direction, computed from the Born laws of the
 *        pair sampler (table contract revision 1.5, 2026-10-29, section 1):
 *          q_k^Born(k) = k - 2 <p' cos theta'>_h = 2 <(E' - p') + p' (1 - cos theta')>_h,
 *        h the single-lepton energy-angle law (Motz, Olsen and Koch 1969
 *        formula 3D-2000 = Blumenthal 1970 eq. 10, `rf::dsigma_dE_dcos_closed`)
 *        normalised by its own integral; the quadrature is the one of step 4
 *        (`rf::moments`: energy panels in s with E' = 1 + (k - 2) sin^2(pi s/2),
 *        angular panels in t = ln(1 - cos theta' + w), Gauss-Legendre per
 *        panel), at an order chosen here so that the node values agree with the
 *        reference to 1e-8 (A10b). Host-only, Kokkos-free.
 *
 * Provider (contract 1.5 section 2, "a node table plus an interpolant"):
 *   - a dense node grid in t = ln(k - 2): NODES_PER_DECADE nodes per decade of
 *     delta = k - 2 from DELTA_MIN to k_max - 2 (the last node exactly at
 *     k_max), every node by direct quadrature (parallel over the host threads);
 *   - a not-a-knot cubic spline of ln q_k in t between the nodes (A10c: the
 *     interpolant against direct quadrature at the midpoints of the 449-node
 *     contract grid must be <= 1e-8 relative; the spline error bound is
 *     (5/384) h^4 max|f''''| with h = ln(10)/NODES_PER_DECADE, which at 128 per
 *     decade is 1.4e-9 |f''''| for f = ln q(t), a smooth function with
 *     derivatives of order one; at 64 per decade, the choice here, the measured
 *     A10c is 5.2e-10, at 128 per decade 2.9e-13);
 *   - below DELTA_MIN the threshold continuation q_k = 2 - s delta with the
 *     slope s from the first node pair (q_k^Born(2) = 2 exactly, A3; the
 *     continuation's error is O(delta^2), i.e. below 1e-10 relative at
 *     DELTA_MIN and smaller below; the cumulatives are O(delta^4) there);
 *   - above k_max the value at k_max (the kernel domain ends there, as for the
 *     fit).
 *   The table is built once per process (function-static), and its build time
 *   is recorded for the startup report.
 *
 * Independence (contract 1.5): the reference (bhref 0.3.0, Codex 1) is not
 * read here; the C++ machinery is the step-4 one, which the step-4 referee
 * compared with the reference's laws (A1, A2, A10 of step 4).
 *
 * @namespaces:
 *   - arch::bh::born::
 */

#ifndef ARCHETYPES_HADRONIC_BH_BORN_QK_H
#define ARCHETYPES_HADRONIC_BH_BORN_QK_H

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_restframe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace arch::bh::born {

  /// grid: nodes per decade of delta = k - 2, in log10; the first node
  inline constexpr int    NODES_PER_DECADE = 64;
  inline constexpr double DELTA_MIN        = 1.0e-5;

  /**
   * @brief The quadrature of the direct evaluation: step 4's default refined
   *        once (A10b of this step sets the order: the default of step 4
   *        agreed with the reference to 1e-4 on the q_k column; see the test
   *        `bh_born_qk` for the measured agreement of this choice)
   */
  inline rf::RestFrameQuadrature direct_quadrature() {
    rf::RestFrameQuadrature q;
    q.gl_order      = 24;
    q.energy_panels = 32;
    q.h_t_angle     = 0.25;
    return q;
  }

  /**
   * @brief q_k^Born(k) by direct quadrature of the Born laws (no table)
   */
  inline double direct(double k, const rf::RestFrameQuadrature& q = direct_quadrature()) {
    if (!(k > 2.0)) {
      return 2.0;
    }
    return rf::moments<double>(k, q).q_k;
  }

  /**
   * @brief Not-a-knot cubic spline on strictly increasing abscissae
   */
  class CubicSpline {
  public:
    CubicSpline() = default;

    CubicSpline(std::vector<double> x, std::vector<double> y) : m_x { std::move(x) }, m_y { std::move(y) } {
      const std::size_t n = m_x.size();
      if (n < 4 || m_y.size() != n) {
        throw std::invalid_argument("CubicSpline: need at least four nodes");
      }
      for (std::size_t i = 1; i < n; ++i) {
        if (!(m_x[i] > m_x[i - 1])) {
          throw std::invalid_argument("CubicSpline: abscissae must increase");
        }
      }
      // second derivatives M_i from the not-a-knot conditions (third derivative
      // continuous at x_1 and x_{n-2}); tridiagonal system solved by elimination
      std::vector<double> h(n - 1);
      for (std::size_t i = 0; i + 1 < n; ++i) {
        h[i] = m_x[i + 1] - m_x[i];
      }
      std::vector<double> a(n, 0.0), b(n, 0.0), c(n, 0.0), d(n, 0.0);
      // row 0: not-a-knot at x_1: h1 M0 - (h0 + h1) M1 + h0 M2 = 0
      // expressed through rows 0 and 1 to keep the system tridiagonal
      for (std::size_t i = 1; i + 1 < n; ++i) {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 6.0 * ((m_y[i + 1] - m_y[i]) / h[i] - (m_y[i] - m_y[i - 1]) / h[i - 1]);
      }
      // not-a-knot: M0 = ((h0 + h1) M1 - h0 M2) / h1  and  M_{n-1} = ((h_{n-2} + h_{n-3}) M_{n-2} - h_{n-2} M_{n-3}) / h_{n-3}
      // substitute into rows 1 and n-2
      {
        const double h0 = h[0], h1 = h[1];
        // row 1: a1 M0 + b1 M1 + c1 M2 = d1 with M0 = ((h0+h1) M1 - h0 M2)/h1
        b[1] += a[1] * (h0 + h1) / h1;
        c[1] -= a[1] * h0 / h1;
        a[1]  = 0.0;
        const double hm = h[n - 2], hp = h[n - 3];
        // row n-2: a M_{n-3} + b M_{n-2} + c M_{n-1} = d with M_{n-1} = ((hm+hp) M_{n-2} - hm M_{n-3})/hp
        b[n - 2] += c[n - 2] * (hm + hp) / hp;
        a[n - 2] -= c[n - 2] * hm / hp;
        c[n - 2]  = 0.0;
      }
      // Thomas algorithm on rows 1..n-2
      std::vector<double> M(n, 0.0);
      {
        std::vector<double> cp(n, 0.0), dp(n, 0.0);
        cp[1] = c[1] / b[1];
        dp[1] = d[1] / b[1];
        for (std::size_t i = 2; i + 1 < n; ++i) {
          const double den = b[i] - a[i] * cp[i - 1];
          cp[i]            = c[i] / den;
          dp[i]            = (d[i] - a[i] * dp[i - 1]) / den;
        }
        M[n - 2] = dp[n - 2];
        for (std::size_t i = n - 3; i >= 1; --i) {
          M[i] = dp[i] - cp[i] * M[i + 1];
        }
        M[0]     = ((h[0] + h[1]) * M[1] - h[0] * M[2]) / h[1];
        M[n - 1] = ((h[n - 2] + h[n - 3]) * M[n - 2] - h[n - 2] * M[n - 3]) / h[n - 3];
      }
      m_M = std::move(M);
    }

    double operator()(double x) const {
      const std::size_t n = m_x.size();
      if (x <= m_x.front()) {
        return m_y.front();
      }
      if (x >= m_x.back()) {
        return m_y.back();
      }
      std::size_t lo = 0, hi = n - 1;
      while (hi - lo > 1) {
        const std::size_t mid = (lo + hi) / 2;
        if (m_x[mid] <= x) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const double h = m_x[hi] - m_x[lo];
      const double A = (m_x[hi] - x) / h, B = (x - m_x[lo]) / h;
      return A * m_y[lo] + B * m_y[hi] +
             ((A * A * A - A) * m_M[lo] + (B * B * B - B) * m_M[hi]) * h * h / 6.0;
    }

    const std::vector<double>& x() const {
      return m_x;
    }

  private:
    std::vector<double> m_x, m_y, m_M;
  };

  /**
   * @brief The node table and its interpolant
   */
  class Table {
  public:
    Table() = default;

    /**
     * @brief Build the table: nodes delta_j = 10^(log10(DELTA_MIN) + j / NODES_PER_DECADE)
     *        while 2 + delta_j < k_max, then k_max exactly; direct quadrature at
     *        every node in parallel
     */
    explicit Table(double k_max, const rf::RestFrameQuadrature& q = direct_quadrature(), unsigned n_threads = 0) {
      if (!(k_max > 2.0 + DELTA_MIN)) {
        throw std::invalid_argument("born::Table: k_max must exceed 2 + DELTA_MIN");
      }
      const auto t0 = std::chrono::steady_clock::now();
      const double l0 = std::log10(DELTA_MIN);
      for (int j = 0;; ++j) {
        const double delta = std::pow(10.0, l0 + static_cast<double>(j) / NODES_PER_DECADE);
        if (2.0 + delta >= k_max * (1.0 - 1.0e-12)) {
          break;
        }
        m_k.push_back(2.0 + delta);
      }
      m_k.push_back(k_max);
      m_q.assign(m_k.size(), 0.0);
      std::vector<double>& qv = m_q;
      const std::vector<double>& kv = m_k;
      rf::parallel_for_index(
        m_k.size(), [&qv, &kv, &q](std::size_t i) { qv[i] = direct(kv[i], q); }, n_threads);
      std::vector<double> t(m_k.size()), lnq(m_k.size());
      for (std::size_t i = 0; i < m_k.size(); ++i) {
        if (!(m_q[i] > 0.0) || !std::isfinite(m_q[i])) {
          throw std::runtime_error("born::Table: q_k not positive and finite at k = " + std::to_string(m_k[i]));
        }
        t[i]   = std::log(m_k[i] - 2.0);
        lnq[i] = std::log(m_q[i]);
      }
      m_spline = CubicSpline(t, lnq);
      // threshold continuation below the first node: q = 2 - s delta, s the secant from the exact
      // threshold value q_k(2) = 2 (the pair at rest in the proton frame) to the first positive node
      // (delta = 1e-5): s = 1/4 + O(delta), the Born threshold slope being exactly 1/4 (A3)
      m_slope = (2.0 - m_q[0]) / (m_k[0] - 2.0);
      m_k_max = k_max;
      m_build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    /// q_k^Born at k = 2 + delta
    double q_k_delta(double delta) const {
      if (!(delta > 0.0)) {
        return 2.0;
      }
      if (delta < m_k[0] - 2.0) {
        return 2.0 - m_slope * delta;
      }
      if (delta >= m_k_max - 2.0) {
        return m_q.back();
      }
      return std::exp(m_spline(std::log(delta)));
    }

    double q_k(double k) const {
      return q_k_delta(k - 2.0);
    }

    const std::vector<double>& nodes_k() const {
      return m_k;
    }

    const std::vector<double>& nodes_q() const {
      return m_q;
    }

    double threshold_slope() const {
      return m_slope;
    }

    double build_seconds() const {
      return m_build_seconds;
    }

    std::size_t size() const {
      return m_k.size();
    }

  private:
    std::vector<double> m_k, m_q;
    CubicSpline         m_spline;
    double              m_slope { 0.25 };
    double              m_k_max { constants::K_MAX };
    double              m_build_seconds { 0.0 };
  };

  /**
   * @brief The process-wide table on the kernel domain [2, K_MAX], built on
   *        first use (thread-safe); the cumulatives of the Born path read it
   */
  inline const Table& table() {
    static const Table T(constants::K_MAX);
    return T;
  }

} // namespace arch::bh::born

namespace arch::bh::born {

  /// the provider registered in bh_physics.h's hook (Born path of the cumulatives)
  inline double hook_q_k_delta(double delta) {
    return table().q_k_delta(delta);
  }

  /// registration at static initialisation of every translation unit including this header
  inline const bool registered = (born_qk_hook() = &hook_q_k_delta, true);

} // namespace arch::bh::born

#endif // ARCHETYPES_HADRONIC_BH_BORN_QK_H
