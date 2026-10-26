/**
 * @file archetypes/hadronic/bh_sampler_host.h
 * @brief The rest-frame pair sampler on the host (P2 step 4, stage 2):
 *        the CDF tables of the rest-frame laws, the cumulatives of the eps'
 *        marginals, the photon-energy tables of a broad bath, the two
 *        sampling chains of the contract (section 2) with the rotation and
 *        the boost, and the direct-boost loss rate by quadrature over the
 *        chain (spec T3b).
 *
 * Host-only, double precision, Kokkos-free. The device views of the same
 * tables are built from these structures in stage 4.
 *
 * Grids (conventions, each implementation's choice within the accuracy
 * budget of contract section 3):
 *   - eps' nodes: log10(eps' - 2) from 1e-4 to the kernel cap 1000,
 *     `nodes_per_decade` (default 16) per decade, the cap appended;
 *   - lepton energy on s in [0, 1] with E' = 1 + (eps' - 2) sin^2(pi s/2)
 *     (x = (E' - 1)/(eps' - 2) = sin^2(pi s/2)), `n_s` cells (default 64);
 *   - lepton angle on tau in [0, 1] with t = ln(1 - cos theta' + w) =
 *     ln w + tau (ln(2 + w) - ln w), w = (E' - p')/p' the forward-peak scale,
 *     `n_tau` cells (default 128);
 *   - the CDFs are stored at the cell edges; sampling inverts the mixture of
 *     the bracketing nodes' CDFs (linear weights in ln(eps' - 2) and in s), by
 *     bisection over the edges and linear interpolation inside a cell (the
 *     sampled density is piecewise constant in s and in t per cell);
 *   - the eps' marginals use ln Phi, ln Psi2 = ln int k^2 sigma~ dk and
 *     ln Psir = ln int k sigma~ sqrt(k^2 - 4) dk on a fine grid in
 *     t = ln(k - 2) (h = 0.005 from k - 2 = 1e-8 to 998), linear in t;
 *   - a broad bath's eps marginal is tabulated at every Gamma node of the
 *     step-1 rate table on a common grid in ln eps (`n_eps` cells, default
 *     512), for both chains, and mixed linearly in y = log10(Gamma - Gamma_th)
 *     between nodes.
 *
 * @namespaces:
 *   - arch::bh::rf::
 */

#ifndef ARCHETYPES_HADRONIC_BH_SAMPLER_HOST_H
#define ARCHETYPES_HADRONIC_BH_SAMPLER_HOST_H

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_restframe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace arch::bh::rf {

  // ------------------------------------------------------------------------ //
  // Random numbers: SplitMix64 (Steele, Lea and Flood 2014), uniform in (0, 1)
  // ------------------------------------------------------------------------ //
  struct SplitMix64 {
    std::uint64_t state;

    explicit SplitMix64(std::uint64_t seed) : state { seed } {}

    std::uint64_t next() {
      std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
      z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      return z ^ (z >> 31);
    }

    /// uniform in the open interval (0, 1): 53 random bits plus half an ulp
    double uniform() {
      return (static_cast<double>(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }
  };

  // ------------------------------------------------------------------------ //
  // Generic inversion of a mixture of tabulated CDFs
  // ------------------------------------------------------------------------ //

  /**
   * @brief Position x in [0, n_cells] (cell units) with sum_a w_a C_a[x] = u,
   *        for CDF arrays C_a on n_cells + 1 edges (0 at the first, 1 at the
   *        last), by bisection over the edges and linear interpolation inside
   *        the cell
   */
  template <int NA, class T>
  inline double invert_mixture(const T* const (&C)[NA], const double (&w)[NA], int n_cells, double u) {
    auto value = [&](int j) {
      double v = 0.0;
      for (int a = 0; a < NA; ++a) {
        v += w[a] * static_cast<double>(C[a][j]);
      }
      return v;
    };
    int lo = 0, hi = n_cells;
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (value(mid) <= u) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const double c0 = value(lo), c1 = value(lo + 1);
    double       f  = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
    f               = std::min(1.0, std::max(0.0, f));
    return static_cast<double>(lo) + f;
  }

  // ------------------------------------------------------------------------ //
  // The cumulatives of the eps' marginals on a fine grid in t = ln(k - 2)
  // ------------------------------------------------------------------------ //
  class SamplerCumulatives {
  public:
    explicit SamplerCumulatives(double h      = 0.005,
                                double delta0 = 1.0e-8,
                                double kmax   = constants::K_MAX,
                                int    gl     = 16)
      : m_h { h }
      , m_delta0 { delta0 }
      , m_kmax { kmax } {
      if (!(h > 0.0) || !(delta0 > 0.0) || !(kmax > 4.0)) {
        throw std::invalid_argument("SamplerCumulatives: bad grid parameters");
      }
      const GaussLegendre G(gl);
      const double        t0 = std::log(delta0), t1 = std::log(kmax - 2.0);
      const auto          n  = static_cast<std::size_t>(std::ceil((t1 - t0) / h));
      m_t.resize(n + 1);
      for (std::size_t i = 0; i <= n; ++i) {
        m_t[i] = t0 + (t1 - t0) * static_cast<double>(i) / static_cast<double>(n);
      }
      // threshold laws at delta0: k sigma -> (pi/6) d^3, k^2 sigma -> (pi/3) d^3,
      // k sigma sqrt(k^2 - 4) -> (pi/3) d^3.5
      double Phi  = (constants::PI / 24.0) * std::pow(delta0, 4.0);
      double Psi2 = (constants::PI / 12.0) * std::pow(delta0, 4.0);
      double Psir = (2.0 * constants::PI / 27.0) * std::pow(delta0, 4.5);
      m_lnPhi.resize(n + 1);
      m_lnPsi2.resize(n + 1);
      m_lnPsir.resize(n + 1);
      m_lnPhi[0]  = std::log(Phi);
      m_lnPsi2[0] = std::log(Psi2);
      m_lnPsir[0] = std::log(Psir);
      for (std::size_t i = 1; i <= n; ++i) {
        const double a = m_t[i - 1], b = m_t[i];
        const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
        double       c1 = 0.0, c2 = 0.0, c3 = 0.0;
        for (int g = 0; g < G.order(); ++g) {
          const double t     = mid + half * G.x[g];
          const double delta = std::exp(t);
          const double k     = 2.0 + delta;
          const double s     = sigma_tilde_delta(delta);
          const double wgt   = half * G.w[g] * delta; // dk = delta dt
          c1 += wgt * k * s;
          c2 += wgt * k * k * s;
          c3 += wgt * k * s * std::sqrt(delta * (k + 2.0));
        }
        Phi  += c1;
        Psi2 += c2;
        Psir += c3;
        m_lnPhi[i]  = std::log(Phi);
        m_lnPsi2[i] = std::log(Psi2);
        m_lnPsir[i] = std::log(Psir);
      }
    }

    double h() const {
      return m_h;
    }

    double delta0() const {
      return m_delta0;
    }

    double kmax() const {
      return m_kmax;
    }

    std::size_t size() const {
      return m_t.size();
    }

    const std::vector<double>& t() const {
      return m_t;
    }

    const std::vector<double>& lnPhi() const {
      return m_lnPhi;
    }

    const std::vector<double>& lnPsi2() const {
      return m_lnPsi2;
    }

    const std::vector<double>& lnPsir() const {
      return m_lnPsir;
    }

    double Phi(double delta) const {
      return eval(m_lnPhi, delta, 4.0);
    }

    double Psi2(double delta) const {
      return eval(m_lnPsi2, delta, 4.0);
    }

    double Psir(double delta) const {
      return eval(m_lnPsir, delta, 4.5);
    }

    /// the D_max-weighted cumulative: gamma Psi2 - eps Phi + gamma beta Psir
    double F(double delta, double gamma, double gamma_beta, double eps) const {
      return gamma * Psi2(delta) - eps * Phi(delta) + gamma_beta * Psir(delta);
    }

    /// delta with Phi(delta) = target (0 < target <= Phi(kmax - 2))
    double invert_Phi(double target) const {
      if (!(target > 0.0)) {
        return 0.0;
      }
      const double lt = std::log(target);
      if (lt <= m_lnPhi.front()) {
        return std::pow(24.0 * target / constants::PI, 0.25);
      }
      if (lt >= m_lnPhi.back()) {
        return m_kmax - 2.0;
      }
      const auto        it = std::upper_bound(m_lnPhi.begin(), m_lnPhi.end(), lt);
      const std::size_t j  = static_cast<std::size_t>(it - m_lnPhi.begin()) - 1;
      const double      f  = (lt - m_lnPhi[j]) / (m_lnPhi[j + 1] - m_lnPhi[j]);
      return std::exp(m_t[j] + f * (m_t[j + 1] - m_t[j]));
    }

    /// delta with F(delta; gamma, gamma beta, eps) = target, F increasing
    double invert_F(double target, double gamma, double gamma_beta, double eps) const {
      if (!(target > 0.0)) {
        return 0.0;
      }
      auto lnF = [&](std::size_t j) {
        return std::log(gamma * std::exp(m_lnPsi2[j]) - eps * std::exp(m_lnPhi[j]) +
                        gamma_beta * std::exp(m_lnPsir[j]));
      };
      const double lt = std::log(target);
      if (lt <= lnF(0)) {
        // threshold law: F ~ (gamma (pi/12) - eps (pi/24)) d^4 (Psir is higher order)
        const double c = gamma * constants::PI / 12.0 - eps * constants::PI / 24.0;
        return c > 0.0 ? std::pow(target / c, 0.25) : m_delta0;
      }
      std::size_t lo = 0, hi = m_t.size() - 1;
      if (lt >= lnF(hi)) {
        return m_kmax - 2.0;
      }
      while (hi - lo > 1) {
        const std::size_t mid = (lo + hi) / 2;
        if (lnF(mid) <= lt) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const double a = lnF(lo), b = lnF(lo + 1);
      const double f = (lt - a) / (b - a);
      return std::exp(m_t[lo] + f * (m_t[lo + 1] - m_t[lo]));
    }

  private:
    double eval(const std::vector<double>& ln, double delta, double power) const {
      if (!(delta > 0.0)) {
        return 0.0;
      }
      if (delta <= m_delta0) {
        return std::exp(ln.front()) * std::pow(delta / m_delta0, power);
      }
      const double dmax = m_kmax - 2.0;
      if (delta >= dmax) {
        return std::exp(ln.back());
      }
      const double t  = std::log(delta);
      const auto   it = std::upper_bound(m_t.begin(), m_t.end(), t);
      std::size_t  j  = static_cast<std::size_t>(it - m_t.begin());
      j               = std::min(std::max<std::size_t>(j, 1), m_t.size() - 1) - 1;
      const double f  = (t - m_t[j]) / (m_t[j + 1] - m_t[j]);
      return std::exp(ln[j] + f * (ln[j + 1] - ln[j]));
    }

    double              m_h, m_delta0, m_kmax;
    std::vector<double> m_t, m_lnPhi, m_lnPsi2, m_lnPsir;
  };

  // ------------------------------------------------------------------------ //
  // The CDF tables of the rest-frame laws
  // ------------------------------------------------------------------------ //
  struct RestFrameTables {
    // grid
    std::vector<double> t_k; // ln(eps' - 2) at the nodes
    int                 n_s { 256 };
    int                 n_tau { 256 };
    int                 nodes_per_decade { 16 };
    double              p_closed_min { 1.0e-3 }; // below this p', the law 2 integral is used
    // CDFs at the edges: cdf_E[ik][js], cdf_A[ik][js][jt]
    std::vector<double> cdf_E;
    std::vector<double> cdf_A;

    std::size_t n_k() const {
      return t_k.size();
    }

    const double* cdfE(std::size_t ik) const {
      return cdf_E.data() + ik * static_cast<std::size_t>(n_s + 1);
    }

    const double* cdfA(std::size_t ik, std::size_t js) const {
      return cdf_A.data() + (ik * static_cast<std::size_t>(n_s + 1) + js) * static_cast<std::size_t>(n_tau + 1);
    }

    // conversions
    static double E_of_s(double k, double s) {
      const double sn = std::sin(0.5 * constants::PI * s);
      return 1.0 + (k - 2.0) * sn * sn;
    }

    static double omc_of_tau(double E, double tau) {
      const double w  = angular_scale(E);
      const double t  = std::log(w) + tau * (std::log(2.0 + w) - std::log(w));
      const double om = std::exp(t) - w;
      return std::min(2.0, std::max(0.0, om));
    }

    /// the single-lepton law used for the tables (closed form, or the law-2
    /// integral where the closed form loses digits)
    double law(double k, double E, double omc) const {
      if (momentum(E) < p_closed_min) {
        return dsigma_dE_dcos_numeric(k, E, omc, AngularQuadrature {});
      }
      return dsigma_dE_dcos_closed(k, E, omc);
    }

    /// bracketing eps' nodes and the weight of the upper one (clamped)
    void locate_k(double k, std::size_t& i, double& w) const {
      const double t = std::log(std::max(k - 2.0, 1.0e-300));
      if (t <= t_k.front()) {
        i = 0;
        w = 0.0;
        return;
      }
      if (t >= t_k.back()) {
        i = t_k.size() - 2;
        w = 1.0;
        return;
      }
      const auto it = std::upper_bound(t_k.begin(), t_k.end(), t);
      i             = static_cast<std::size_t>(it - t_k.begin()) - 1;
      w             = (t - t_k[i]) / (t_k[i + 1] - t_k[i]);
    }

    /// s in [0, 1] from the energy CDF at continuous eps'
    double sample_s(double k, double u) const {
      std::size_t i;
      double      w;
      locate_k(k, i, w);
      const double* const C[2] = { cdfE(i), cdfE(i + 1) };
      const double        W[2] = { 1.0 - w, w };
      return invert_mixture<2>(C, W, n_s, u) / n_s;
    }

    /// tau in [0, 1] from the angular CDF at continuous (eps', s)
    double sample_tau(double k, double s, double u) const {
      std::size_t i;
      double      w;
      locate_k(k, i, w);
      const double sc = std::min(std::max(s, 0.0), 1.0) * n_s;
      std::size_t  j  = static_cast<std::size_t>(std::min(static_cast<int>(sc), n_s - 1));
      const double v  = sc - static_cast<double>(j);
      const double* const C[4] = { cdfA(i, j), cdfA(i, j + 1), cdfA(i + 1, j), cdfA(i + 1, j + 1) };
      const double        W[4] = { (1.0 - w) * (1.0 - v), (1.0 - w) * v, w * (1.0 - v), w * v };
      return invert_mixture<4>(C, W, n_tau, u) / n_tau;
    }

    /**
     * @brief The moments of the table-defined law at continuous eps' by
     *        quadrature over the piecewise-constant densities (the accuracy
     *        budget check of contract section 3)
     */
    struct TableMoments {
      double mean_E { 0.0 }, mean_pcos { 0.0 }, q_k { 0.0 };
    };

    TableMoments moments_at(double k, int gl_sub = 4) const {
      std::size_t i;
      double      w;
      locate_k(k, i, w);
      const GaussLegendre G(gl_sub);
      const double*       CE[2] = { cdfE(i), cdfE(i + 1) };
      TableMoments        m;
      const double        hs = 1.0 / n_s, ht = 1.0 / n_tau;
      for (int j = 0; j < n_s; ++j) {
        const double dC = (1.0 - w) * (CE[0][j + 1] - CE[0][j]) + w * (CE[1][j + 1] - CE[1][j]);
        if (!(dC > 0.0)) {
          continue;
        }
        const double* CA[4] = { cdfA(i, j), cdfA(i, j + 1), cdfA(i + 1, j), cdfA(i + 1, j + 1) };
        double        sE = 0.0, sP = 0.0;
        for (int g = 0; g < G.order(); ++g) {
          const double s  = (j + 0.5 * (1.0 + G.x[g])) * hs;
          const double v  = 0.5 * (1.0 + G.x[g]); // position inside the s cell
          const double E  = E_of_s(k, s);
          const double p  = momentum(E);
          const double W4[4] = { (1.0 - w) * (1.0 - v), (1.0 - w) * v, w * (1.0 - v), w * v };
          // <cos theta> of the mixed piecewise-constant density in tau at this E
          const double wsc = angular_scale(E);
          const double ta = std::log(wsc), tb = std::log(2.0 + wsc);
          double       cbar = 0.0;
          for (int l = 0; l < n_tau; ++l) {
            double dA = 0.0;
            for (int a = 0; a < 4; ++a) {
              dA += W4[a] * (CA[a][l + 1] - CA[a][l]);
            }
            if (!(dA > 0.0)) {
              continue;
            }
            double cell = 0.0;
            for (int gg = 0; gg < G.order(); ++gg) {
              const double tau = (l + 0.5 * (1.0 + G.x[gg])) * ht;
              const double om  = std::min(2.0, std::max(0.0, std::exp(ta + tau * (tb - ta)) - wsc));
              cell += 0.5 * G.w[gg] * (1.0 - om);
            }
            cbar += dA * cell;
          }
          sE += 0.5 * G.w[g] * E;
          sP += 0.5 * G.w[g] * p * cbar;
        }
        m.mean_E    += dC * sE;
        m.mean_pcos += dC * sP;
      }
      m.q_k = k - 2.0 * m.mean_pcos;
      return m;
    }

    /**
     * @brief Build the tables (threads over the eps' nodes)
     */
    static RestFrameTables build(int nodes_per_decade = 16, int n_s = 256, int n_tau = 256, int gl_cell = 8,
                                 unsigned n_threads = 0) {
      RestFrameTables T;
      T.n_s              = n_s;
      T.n_tau            = n_tau;
      T.nodes_per_decade = nodes_per_decade;
      for (int i = 0;; ++i) {
        const double delta = std::pow(10.0, -4.0 + static_cast<double>(i) / nodes_per_decade);
        if (2.0 + delta >= constants::K_MAX) {
          break;
        }
        T.t_k.push_back(std::log(delta));
      }
      T.t_k.push_back(std::log(constants::K_MAX - 2.0));
      const std::size_t nk = T.t_k.size();
      T.cdf_E.assign(nk * (n_s + 1), 0.0);
      T.cdf_A.assign(nk * (n_s + 1) * (n_tau + 1), 0.0);
      const GaussLegendre G(gl_cell);
      parallel_for_index(
        nk,
        [&](std::size_t ik) {
          const double k  = 2.0 + std::exp(T.t_k[ik]);
          double*      CE = T.cdf_E.data() + ik * (n_s + 1);
          // energy CDF on the s cells
          const double hs = 1.0 / n_s;
          CE[0]           = 0.0;
          for (int j = 0; j < n_s; ++j) {
            double cell = 0.0;
            for (int g = 0; g < G.order(); ++g) {
              const double s   = (j + 0.5 * (1.0 + G.x[g])) * hs;
              const double E   = E_of_s(k, s);
              const double jac = (k - 2.0) * 0.5 * constants::PI * std::sin(constants::PI * s);
              cell += 0.5 * hs * G.w[g] * jac * dsigma_dE(k, E);
            }
            CE[j + 1] = CE[j] + cell;
          }
          const double tot = CE[n_s];
          for (int j = 0; j <= n_s; ++j) {
            CE[j] = tot > 0.0 ? std::min(1.0, CE[j] / tot) : static_cast<double>(j) / n_s;
          }
          CE[n_s] = 1.0;
          // angular CDFs at the s nodes (end nodes evaluated half a cell inside)
          const double ht = 1.0 / n_tau;
          for (int j = 0; j <= n_s; ++j) {
            const double s_eval = std::min(1.0 - 0.5 * hs, std::max(0.5 * hs, static_cast<double>(j) * hs));
            const double E      = E_of_s(k, s_eval);
            const double wsc    = angular_scale(E);
            const double ta = std::log(wsc), tb = std::log(2.0 + wsc);
            double*      CA = T.cdf_A.data() + (ik * (n_s + 1) + j) * (n_tau + 1);
            CA[0]           = 0.0;
            for (int l = 0; l < n_tau; ++l) {
              double cell = 0.0;
              for (int g = 0; g < G.order(); ++g) {
                const double tau = (l + 0.5 * (1.0 + G.x[g])) * ht;
                const double t   = ta + tau * (tb - ta);
                const double om  = std::min(2.0, std::max(0.0, std::exp(t) - wsc));
                cell += 0.5 * ht * G.w[g] * (tb - ta) * std::exp(t) * T.law(k, E, om);
              }
              CA[l + 1] = CA[l] + std::max(0.0, cell);
            }
            const double totA = CA[n_tau];
            for (int l = 0; l <= n_tau; ++l) {
              CA[l] = totA > 0.0 ? std::min(1.0, CA[l] / totA) : static_cast<double>(l) / n_tau;
            }
            CA[n_tau] = 1.0;
          }
        },
        n_threads);
      return T;
    }
  };

  /**
   * @brief q_k of the table-defined law on a fine grid in ln(eps' - 2) (`sub`
   *        points per table interval), for the direct-boost rates
   */
  struct QkFineGrid {
    std::vector<double> t, q;

    double operator()(double k) const {
      const double tt = std::log(std::max(k - 2.0, 1.0e-300));
      if (tt <= t.front()) {
        return q.front();
      }
      if (tt >= t.back()) {
        return q.back();
      }
      const auto        it = std::upper_bound(t.begin(), t.end(), tt);
      const std::size_t j  = static_cast<std::size_t>(it - t.begin()) - 1;
      const double      f  = (tt - t[j]) / (t[j + 1] - t[j]);
      return (1.0 - f) * q[j] + f * q[j + 1];
    }
  };

  inline QkFineGrid table_qk_fine_grid(const RestFrameTables& T, int sub = 8, unsigned n_threads = 0) {
    QkFineGrid F;
    for (std::size_t i = 0; i + 1 < T.n_k(); ++i) {
      for (int s = 0; s < sub; ++s) {
        F.t.push_back(T.t_k[i] + (T.t_k[i + 1] - T.t_k[i]) * static_cast<double>(s) / sub);
      }
    }
    F.t.push_back(T.t_k.back());
    F.q.assign(F.t.size(), 0.0);
    parallel_for_index(
      F.t.size(),
      [&](std::size_t n) { F.q[n] = T.moments_at(2.0 + std::exp(F.t[n])).q_k; },
      n_threads);
    return F;
  }

  // ------------------------------------------------------------------------ //
  // Cumulatives with a given q_k(k): phi = int k sigma q_k, psi = int sigma q_k
  // (the step-1 machinery with the fit replaced by any q_k)
  // ------------------------------------------------------------------------ //
  class QkCumulatives {
  public:
    struct Values {
      double phi { 0.0 }, psi { 0.0 };
    };

    QkCumulatives(std::function<double(double)> qk,
                  double                        h      = 0.02,
                  double                        delta0 = 1.0e-8,
                  double                        kmax   = constants::K_MAX,
                  int                           gl     = 16)
      : m_qk { std::move(qk) }
      , m_delta0 { delta0 }
      , m_kmax { kmax }
      , m_gl { gl } {
      const double t0 = std::log(delta0), t1 = std::log(constants::K_JUNCTION - 2.0), t2 = std::log(kmax - 2.0);
      const auto   n1 = static_cast<std::size_t>(std::ceil((t1 - t0) / h));
      const auto   n2 = static_cast<std::size_t>(std::ceil((t2 - t1) / h));
      for (std::size_t i = 0; i < n1; ++i) {
        m_t.push_back(t0 + (t1 - t0) * static_cast<double>(i) / static_cast<double>(n1));
      }
      m_t.push_back(t1);
      for (std::size_t i = 1; i < n2; ++i) {
        m_t.push_back(t1 + (t2 - t1) * static_cast<double>(i) / static_cast<double>(n2));
      }
      m_t.push_back(t2);
      m_cum.resize(m_t.size());
      m_cum[0] = analytic(delta0);
      for (std::size_t i = 1; i < m_t.size(); ++i) {
        const Values c = cell(m_t[i - 1], m_t[i]);
        m_cum[i].phi   = m_cum[i - 1].phi + c.phi;
        m_cum[i].psi   = m_cum[i - 1].psi + c.psi;
      }
    }

    Values operator()(double delta) const {
      if (!(delta > 0.0)) {
        return Values {};
      }
      if (delta <= m_delta0) {
        return analytic(delta);
      }
      const double dmax = m_kmax - 2.0;
      if (delta > dmax) {
        delta = dmax;
      }
      const double t  = std::log(delta);
      auto         it = std::upper_bound(m_t.begin(), m_t.end(), t);
      std::size_t  j  = static_cast<std::size_t>(it - m_t.begin());
      j               = std::min(std::max<std::size_t>(j, 1), m_t.size() - 1) - 1;
      const Values p  = cell(m_t[j], t);
      return Values { m_cum[j].phi + p.phi, m_cum[j].psi + p.psi };
    }

    static Values analytic(double delta) {
      const double d4 = delta * delta * delta * delta;
      return Values { (constants::PI / 12.0) * d4, (constants::PI / 24.0) * d4 };
    }

  private:
    Values cell(double a, double b) const {
      Values       out;
      const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
      for (int i = 0; i < m_gl.order(); ++i) {
        const double t     = mid + half * m_gl.x[i];
        const double delta = std::exp(t);
        const double k     = 2.0 + delta;
        const double s     = sigma_tilde_delta(delta);
        const double q     = m_qk(k);
        const double w     = half * m_gl.w[i] * delta;
        out.phi += w * k * s * q;
        out.psi += w * s * q;
      }
      return out;
    }

    std::function<double(double)> m_qk;
    double                        m_delta0, m_kmax;
    GaussLegendre                 m_gl;
    std::vector<double>           m_t;
    std::vector<Values>           m_cum;
  };

  /**
   * @brief The direct-boost loss rate I(Gamma) = (1/(2 Gamma beta)) int deps
   *        n~/eps^2 [phi(hi) - (eps/Gamma) psi(hi)] with the given cumulatives
   *        (spec T3b), same outer quadrature as step 1
   */
  inline double direct_boost_I(const PhotonBath& bath, const QkCumulatives& cum, double gamma,
                               double h_log_eps = 0.02, int gl = 16) {
    const double gb = std::sqrt((gamma - 1.0) * (gamma + 1.0));
    const double A  = gamma + gb;
    if (bath.is_mono()) {
      const double e0 = bath.eps0();
      const auto   c  = cum(e0 * A - 2.0);
      return (c.phi - (e0 / gamma) * c.psi) / (2.0 * gb * e0 * e0);
    }
    const double        eps_th = 2.0 / A;
    const double        v_max  = std::log(bath.eps_max() / eps_th);
    const double        v_min  = bath.eps_min() > eps_th ? std::log(bath.eps_min() / eps_th) : 0.0;
    if (!(v_max > v_min)) {
      return 0.0;
    }
    std::vector<double> edges { v_min };
    const double        v4 = std::log(2.0);
    if (v4 > v_min && v4 < v_max) {
      edges.push_back(v4);
    }
    for (const double eb : bath.breakpoints()) {
      const double vb = std::log(eb / eps_th);
      if (vb > v_min && vb < v_max) {
        edges.push_back(vb);
      }
    }
    edges.push_back(v_max);
    std::sort(edges.begin(), edges.end());
    const GaussLegendre G(gl);
    double              SI = 0.0;
    for (std::size_t s = 0; s + 1 < edges.size(); ++s) {
      const double L = edges[s + 1] - edges[s];
      if (!(L > 0.0)) {
        continue;
      }
      const int    npan = std::max(1, static_cast<int>(std::ceil(L / h_log_eps)));
      const double hp   = L / npan;
      for (int p = 0; p < npan; ++p) {
        const double lo = edges[s] + hp * p, mid = lo + 0.5 * hp, half = 0.5 * hp;
        for (int i = 0; i < G.order(); ++i) {
          const double v   = mid + half * G.x[i];
          const double w   = half * G.w[i];
          const double eps = eps_th * std::exp(v);
          const double n   = bath.density(eps);
          if (!(n > 0.0)) {
            continue;
          }
          const auto hi  = cum(2.0 * std::expm1(v));
          const auto lo_ = cum(eps / A - 2.0);
          SI += w * n * ((hi.phi - lo_.phi) - (eps / gamma) * (hi.psi - lo_.psi)) / eps;
        }
      }
    }
    return SI / (2.0 * gb);
  }

  // ------------------------------------------------------------------------ //
  // The photon-energy tables of a broad bath (both chains) at the Gamma nodes
  // ------------------------------------------------------------------------ //
  struct BathEpsTables {
    bool                mono { true };
    double              eps0 { 0.0 };
    double              gamma_th { 0.0 };
    std::vector<double> y;      // log10(Gamma - Gamma_th) at the rate-table nodes
    std::vector<double> gamma;  // the nodes
    std::vector<double> ln_eps; // n_eps + 1 edges
    int                 n_eps { 2048 };
    std::vector<double> cdf_u; // [n_gamma][n_eps + 1] unweighted chain
    std::vector<double> cdf_w; // [n_gamma][n_eps + 1] D_max-weighted chain

    const double* cdf(bool weighted, std::size_t ig) const {
      return (weighted ? cdf_w : cdf_u).data() + ig * static_cast<std::size_t>(n_eps + 1);
    }

    void locate_gamma(double g, std::size_t& i, double& w) const {
      const double yy = std::log10(std::max(g - gamma_th, 1.0e-300));
      if (yy <= y.front()) {
        i = 0;
        w = 0.0;
        return;
      }
      if (yy >= y.back()) {
        i = y.size() - 2;
        w = 1.0;
        return;
      }
      const auto it = std::upper_bound(y.begin(), y.end(), yy);
      i             = static_cast<std::size_t>(it - y.begin()) - 1;
      w             = (yy - y[i]) / (y[i + 1] - y[i]);
    }

    /// eps from the chain's marginal at continuous Gamma
    double sample_eps(double g, bool weighted, double u) const {
      if (mono) {
        return eps0;
      }
      std::size_t i;
      double      w;
      locate_gamma(g, i, w);
      const double* const C[2] = { cdf(weighted, i), cdf(weighted, i + 1) };
      const double        W[2] = { 1.0 - w, w };
      const double        pos  = invert_mixture<2>(C, W, n_eps, u);
      const std::size_t   j    = static_cast<std::size_t>(std::min(static_cast<int>(pos), n_eps - 1));
      const double        f    = pos - static_cast<double>(j);
      return std::exp(ln_eps[j] + f * (ln_eps[j + 1] - ln_eps[j]));
    }

    static BathEpsTables build(const PhotonBath& bath, const HadronicTablesHost& H, const SamplerCumulatives& cum,
                               int n_eps = 2048, int gl_cell = 8, unsigned n_threads = 0) {
      return build_at(bath, H.gamma, cum, n_eps, gl_cell, n_threads);
    }

    /// the same at an explicit, increasing list of Gamma nodes (> Gamma_th)
    static BathEpsTables build_at(const PhotonBath& bath, const std::vector<double>& gammas,
                                  const SamplerCumulatives& cum, int n_eps = 2048, int gl_cell = 8,
                                  unsigned n_threads = 0) {
      BathEpsTables B;
      B.gamma_th = bath.gamma_th();
      B.gamma    = gammas;
      B.mono     = bath.is_mono();
      if (B.mono) {
        B.eps0 = bath.eps0();
        return B;
      }
      if (gammas.size() < 2) {
        throw std::invalid_argument("BathEpsTables: at least two Gamma nodes are needed");
      }
      B.y.resize(gammas.size());
      for (std::size_t i = 0; i < gammas.size(); ++i) {
        if (!(gammas[i] > B.gamma_th) || (i > 0 && !(gammas[i] > gammas[i - 1]))) {
          throw std::invalid_argument("BathEpsTables: Gamma nodes must be increasing and above Gamma_th");
        }
        B.y[i] = std::log10(gammas[i] - B.gamma_th);
      }
      B.n_eps = n_eps;
      // common grid in ln eps from the smallest threshold energy (at Gamma_max)
      // or the bath's lower edge, to eps_max
      const double gmax   = gammas.back();
      const double Amax   = gmax + std::sqrt((gmax - 1.0) * (gmax + 1.0));
      const double eps_lo = std::max(bath.eps_min(), 2.0 / Amax);
      const double eps_hi = bath.eps_max();
      if (!(eps_hi > eps_lo)) {
        throw std::invalid_argument("BathEpsTables: empty photon-energy range");
      }
      B.ln_eps.resize(n_eps + 1);
      for (int j = 0; j <= n_eps; ++j) {
        B.ln_eps[j] = std::log(eps_lo) + (std::log(eps_hi) - std::log(eps_lo)) * static_cast<double>(j) / n_eps;
      }
      const std::size_t ng = gammas.size();
      B.cdf_u.assign(ng * (n_eps + 1), 0.0);
      B.cdf_w.assign(ng * (n_eps + 1), 0.0);
      const GaussLegendre G(gl_cell);
      parallel_for_index(
        ng,
        [&](std::size_t ig) {
          const double g  = gammas[ig];
          const double gb = std::sqrt((g - 1.0) * (g + 1.0));
          const double A  = g + gb;
          double*      Cu = B.cdf_u.data() + ig * (n_eps + 1);
          double*      Cw = B.cdf_w.data() + ig * (n_eps + 1);
          Cu[0] = Cw[0] = 0.0;
          for (int j = 0; j < n_eps; ++j) {
            double cu = 0.0, cw = 0.0;
            const double a = B.ln_eps[j], b = B.ln_eps[j + 1];
            const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
            for (int q = 0; q < G.order(); ++q) {
              const double le  = mid + half * G.x[q];
              const double eps = std::exp(le);
              const double n   = bath.density(eps);
              const double dhi = eps * A - 2.0;
              if (!(n > 0.0) || !(dhi > 0.0)) {
                continue;
              }
              const double w = half * G.w[q] * n / eps; // d eps / eps^2 = d(ln eps) / eps
              cu += w * cum.Phi(dhi);
              cw += w * cum.F(dhi, g, gb, eps);
            }
            Cu[j + 1] = Cu[j] + cu;
            Cw[j + 1] = Cw[j] + cw;
          }
          for (int j = 0; j <= n_eps; ++j) {
            Cu[j] = Cu[n_eps] > 0.0 ? std::min(1.0, Cu[j] / Cu[n_eps]) : static_cast<double>(j) / n_eps;
            Cw[j] = Cw[n_eps] > 0.0 ? std::min(1.0, Cw[j] / Cw[n_eps]) : static_cast<double>(j) / n_eps;
          }
          Cu[n_eps] = Cw[n_eps] = 1.0;
        },
        n_threads);
      return B;
    }
  };

  // ------------------------------------------------------------------------ //
  // The five acceptance baths of step 1 (contract 1.4 section 6)
  // ------------------------------------------------------------------------ //
  struct CanonicalBath {
    std::string name;
    BathSpec    spec;
    double      mu_r { 1836.15 };
  };

  inline std::vector<CanonicalBath> canonical_baths() {
    std::vector<CanonicalBath> v;
    {
      CanonicalBath b;
      b.name      = "mono05";
      b.spec.type = "mono";
      b.spec.eps0 = 0.5;
      v.push_back(b);
    }
    {
      CanonicalBath b;
      b.name      = "mono03";
      b.spec.type = "mono";
      b.spec.eps0 = 0.3;
      v.push_back(b);
    }
    {
      CanonicalBath b;
      b.name         = "powerlaw";
      b.spec.type    = "powerlaw";
      b.spec.eps_min = 0.1;
      b.spec.eps_max = 1.5;
      b.spec.index   = 2.0;
      v.push_back(b);
    }
    {
      CanonicalBath b;
      b.name         = "blackbody";
      b.spec.type    = "blackbody";
      b.spec.theta   = 0.1;
      b.spec.eps_min = 0.0;
      b.spec.eps_max = 1.9;
      v.push_back(b);
    }
    {
      CanonicalBath b;
      b.name      = "mono05_mu100";
      b.spec.type = "mono";
      b.spec.eps0 = 0.5;
      b.mu_r      = 100.0;
      v.push_back(b);
    }
    return v;
  }

  inline const CanonicalBath& canonical_bath(const std::string& name) {
    static const std::vector<CanonicalBath> all = canonical_baths();
    for (const auto& b : all) {
      if (b.name == name) {
        return b;
      }
    }
    throw std::invalid_argument("canonical_bath: unknown bath '" + name +
                                "' (mono05, mono03, powerlaw, blackbody, mono05_mu100)");
  }

  // ------------------------------------------------------------------------ //
  // Lab kinematics
  // ------------------------------------------------------------------------ //
  struct Vec3 {
    double x { 0.0 }, y { 0.0 }, z { 0.0 };
  };

  struct FourMomentum {
    double E { 0.0 };
    Vec3   p;
  };

  /// orthonormal basis (e1, e2) perpendicular to the unit vector n
  inline void perpendicular_basis(const Vec3& n, Vec3& e1, Vec3& e2) {
    // pick the axis least aligned with n
    if (std::abs(n.x) < 0.9) {
      // e1 = x^ - (x^.n) n, normalised
      const double c = n.x;
      e1             = Vec3 { 1.0 - c * n.x, -c * n.y, -c * n.z };
    } else {
      const double c = n.y;
      e1             = Vec3 { -c * n.x, 1.0 - c * n.y, -c * n.z };
    }
    const double l = std::sqrt(e1.x * e1.x + e1.y * e1.y + e1.z * e1.z);
    e1             = Vec3 { e1.x / l, e1.y / l, e1.z / l };
    e2             = Vec3 { n.y * e1.z - n.z * e1.y, n.z * e1.x - n.x * e1.z, n.x * e1.y - n.y * e1.x };
  }

  /// a direction with polar angle (cos, sin) and azimuth phi about the axis n
  inline Vec3 direction_about(const Vec3& n, const Vec3& e1, const Vec3& e2, double cos_t, double sin_t, double phi) {
    const double cx = sin_t * std::cos(phi), cy = sin_t * std::sin(phi);
    return Vec3 { cx * e1.x + cy * e2.x + cos_t * n.x, cx * e1.y + cy * e2.y + cos_t * n.y,
                  cx * e1.z + cy * e2.z + cos_t * n.z };
  }

  /// boost along the unit vector u with (gamma, gamma beta): E = gamma (E' + beta p'_par), ...
  inline FourMomentum boost_along(const FourMomentum& q, const Vec3& u, double gamma, double gamma_beta) {
    const double ppar = q.p.x * u.x + q.p.y * u.y + q.p.z * u.z;
    const double E    = gamma * q.E + gamma_beta * ppar;
    const double par  = gamma * ppar + gamma_beta * q.E;
    const double d    = par - ppar;
    return FourMomentum { E, Vec3 { q.p.x + d * u.x, q.p.y + d * u.y, q.p.z + d * u.z } };
  }

  // ------------------------------------------------------------------------ //
  // The two sampling chains
  // ------------------------------------------------------------------------ //
  struct PairEvent {
    double eps { 0.0 };     // lab photon energy
    double k { 0.0 };       // rest-frame photon energy eps'
    double Em { 0.0 }, Ep { 0.0 }; // rest-frame lepton energies (e-, e+)
    double cosm { 0.0 }, cosp { 0.0 }; // rest-frame cosines to k^'
    double cos_k_prime { 0.0 };        // rest-frame photon cosine to u^
    FourMomentum lab_m, lab_p, lab_photon; // lab four-momenta, u^ = +z
    double D { 0.0 }, Dmin { 0.0 }, Dmax { 0.0 };
    int    attempts { 1 }; // weighted chain: trials until acceptance
  };

  class HostSampler {
  public:
    HostSampler(std::shared_ptr<const PhotonBath>       bath,
                const HadronicTablesHost&               H,
                std::shared_ptr<const RestFrameTables>  tables,
                std::shared_ptr<const SamplerCumulatives> cum,
                std::shared_ptr<const BathEpsTables>    eps_tables,
                int                                     max_attempts = 100000)
      : m_bath { std::move(bath) }
      , m_T { std::move(tables) }
      , m_cum { std::move(cum) }
      , m_B { std::move(eps_tables) }
      , m_gamma_th { H.gamma_th }
      , m_max_attempts { max_attempts } {
      if (!m_bath || !m_T || !m_cum || !m_B) {
        throw std::invalid_argument("HostSampler: null table");
      }
    }

    const PhotonBath& bath() const {
      return *m_bath;
    }

    const RestFrameTables& tables() const {
      return *m_T;
    }

    const SamplerCumulatives& cumulatives() const {
      return *m_cum;
    }

    static void bounds(double gamma, double gb, double eps, double k, double& Dmin, double& Dmax) {
      const double r = gb * std::sqrt(std::max(0.0, (k - 2.0) * (k + 2.0)));
      Dmin           = gamma * k - eps - r;
      Dmax           = gamma * k - eps + r;
    }

    /// one draw of the unweighted (weighted = false) or D-weighted chain
    PairEvent draw(double gamma, bool weighted, SplitMix64& rng) const {
      const double gb   = std::sqrt((gamma - 1.0) * (gamma + 1.0));
      const double beta = gb / gamma;
      const double A    = gamma + gb; // Gamma (1 + beta)
      const Vec3   u { 0.0, 0.0, 1.0 };
      const Vec3   ue1 { 1.0, 0.0, 0.0 }, ue2 { 0.0, 1.0, 0.0 };
      PairEvent    ev;
      for (int attempt = 1; attempt <= m_max_attempts; ++attempt) {
        ev.attempts = attempt;
        // eps and eps'
        double eps = m_B->sample_eps(gamma, weighted, rng.uniform());
        double dhi = eps * A - 2.0;
        if (!(dhi > 0.0)) {
          // below the threshold of this Gamma (mixture tail): the smallest active eps
          eps = 2.0 / A * (1.0 + 1.0e-12);
          dhi = eps * A - 2.0;
        }
        const double u2 = rng.uniform();
        double       delta;
        if (weighted) {
          delta = m_cum->invert_F(u2 * m_cum->F(dhi, gamma, gb, eps), gamma, gb, eps);
        } else {
          delta = m_cum->invert_Phi(u2 * m_cum->Phi(dhi));
        }
        delta          = std::min(std::max(delta, 0.0), dhi);
        const double k = 2.0 + delta;
        // the photon's rest-frame direction relative to u^
        double cos_k = (eps / (gamma * k) - 1.0) / beta;
        cos_k        = std::min(1.0, std::max(-1.0, cos_k));
        const double sin_k = std::sqrt(std::max(0.0, (1.0 - cos_k) * (1.0 + cos_k)));
        const double phi_k = 2.0 * constants::PI * rng.uniform();
        const Vec3   kh    = direction_about(u, ue1, ue2, cos_k, sin_k, phi_k);
        // lepton energies
        const double s  = m_T->sample_s(k, rng.uniform());
        const double Em = RestFrameTables::E_of_s(k, s);
        const double Ep = k - Em;
        const double pm = momentum(Em), pp = momentum(Ep);
        // angles relative to k^' (independent draws), azimuths uniform
        const double omcm = RestFrameTables::omc_of_tau(Em, m_T->sample_tau(k, s, rng.uniform()));
        const double omcp = RestFrameTables::omc_of_tau(Ep, m_T->sample_tau(k, 1.0 - s, rng.uniform()));
        const double cosm = 1.0 - omcm, cosp = 1.0 - omcp;
        const double sinm = std::sqrt(std::max(0.0, omcm * (2.0 - omcm)));
        const double sinp = std::sqrt(std::max(0.0, omcp * (2.0 - omcp)));
        Vec3         ke1, ke2;
        perpendicular_basis(kh, ke1, ke2);
        const Vec3 dm = direction_about(kh, ke1, ke2, cosm, sinm, 2.0 * constants::PI * rng.uniform());
        const Vec3 dp = direction_about(kh, ke1, ke2, cosp, sinp, 2.0 * constants::PI * rng.uniform());
        // boost
        const FourMomentum qm { Em, Vec3 { pm * dm.x, pm * dm.y, pm * dm.z } };
        const FourMomentum qp { Ep, Vec3 { pp * dp.x, pp * dp.y, pp * dp.z } };
        ev.lab_m = boost_along(qm, u, gamma, gb);
        ev.lab_p = boost_along(qp, u, gamma, gb);
        // the absorbed photon in the lab: eps along k^ with cos theta_k = (1 - eps'/(Gamma eps))/beta
        double cos_lab = (1.0 - k / (gamma * eps)) / beta;
        cos_lab        = std::min(1.0, std::max(-1.0, cos_lab));
        const double sin_lab = std::sqrt(std::max(0.0, (1.0 - cos_lab) * (1.0 + cos_lab)));
        const Vec3   klab    = direction_about(u, ue1, ue2, cos_lab, sin_lab, phi_k);
        ev.lab_photon        = FourMomentum { eps, Vec3 { eps * klab.x, eps * klab.y, eps * klab.z } };
        // the ion's loss
        ev.D = ev.lab_m.E + ev.lab_p.E - eps;
        bounds(gamma, gb, eps, k, ev.Dmin, ev.Dmax);
        ev.eps         = eps;
        ev.k           = k;
        ev.Em          = Em;
        ev.Ep          = Ep;
        ev.cosm        = cosm;
        ev.cosp        = cosp;
        ev.cos_k_prime = cos_k;
        if (!weighted) {
          return ev;
        }
        // accept with probability D / D_max
        if (rng.uniform() * ev.Dmax <= ev.D) {
          return ev;
        }
      }
      throw std::runtime_error("HostSampler: the weighted chain did not accept within " +
                               std::to_string(m_max_attempts) + " attempts");
    }

  private:
    std::shared_ptr<const PhotonBath>         m_bath;
    std::shared_ptr<const RestFrameTables>    m_T;
    std::shared_ptr<const SamplerCumulatives> m_cum;
    std::shared_ptr<const BathEpsTables>      m_B;
    double                                    m_gamma_th;
    int                                       m_max_attempts;
  };

} // namespace arch::bh::rf

#endif // ARCHETYPES_HADRONIC_BH_SAMPLER_HOST_H
