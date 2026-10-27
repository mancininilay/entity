/**
 * @file archetypes/hadronic/bh_sampler_device.h
 * @brief Device mirror of the rest-frame sampler tables (P2 step 4, stage 4)
 *        and the device-callable sampling chains used by the Bethe-Heitler
 *        kernel for `spectrum = restframe`.
 *
 * The host tables (`rf::RestFrameTables`, `rf::SamplerCumulatives`,
 * `rf::BathEpsTables` of bh_sampler_host.h) are built in double on the host;
 * `mirror_sampler_to_device` copies the CDF tables into single-precision views
 * (contract section 3: the CDFs may be stored in single precision, 1e-7
 * resolution suffices) and the small arrays (eps' node positions, the three
 * logarithmic cumulatives, the Gamma and ln eps grids) into double views.
 * Every kinematic computation after a draw is in double; the inversions
 * return a position in double from the float CDF values.
 *
 * The chains are the same as the host sampler's (`rf::HostSampler::draw`):
 * the unweighted chain (poisson source) and the D_max-weighted chain with the
 * rejection on D/D_max (budget source); the rejection loop is bounded by
 * `max_attempts` and the caller counts the overflow. The lab four-momenta are
 * returned in the frame whose z axis is the ion velocity u^; the kernel rotates
 * them to the simulation frame with the ion's direction.
 *
 * @namespaces:
 *   - arch::bh::
 */

#ifndef ARCHETYPES_HADRONIC_BH_SAMPLER_DEVICE_H
#define ARCHETYPES_HADRONIC_BH_SAMPLER_DEVICE_H

#include "arch/kokkos_aliases.h"

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_sampler_host.h"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace arch::bh {

  /**
   * @brief One sampled pair event (device output); momenta in the frame with
   *        z along the ion velocity, energies in m_e c^2
   */
  struct SampledPair {
    double eps { 0.0 };       // lab photon energy
    double k { 0.0 };         // rest-frame photon energy eps'
    double Em { 0.0 };        // rest-frame e- energy
    double Ep { 0.0 };        // rest-frame e+ energy
    double E_m { 0.0 };       // lab e- energy
    double E_p { 0.0 };       // lab e+ energy
    double pm[3] { 0.0, 0.0, 0.0 }; // lab e- momentum (x, y, z = u^)
    double pp[3] { 0.0, 0.0, 0.0 }; // lab e+ momentum
    double kph[3] { 0.0, 0.0, 0.0 }; // lab photon momentum (|kph| = eps)
    double D { 0.0 };         // E_m + E_p - eps
    double Dmax { 0.0 };
    int    attempts { 0 };
  };

  /**
   * @brief Device views of the sampler tables and the device chains
   */
  struct SamplerTables {
    bool valid { false };
    // rest-frame CDF tables
    int              n_k { 0 }, n_s { 0 }, n_tau { 0 };
    array_t<double*> t_k;   // ln(eps' - 2) at the nodes
    array_t<float*>  cdf_E; // [n_k][n_s + 1]
    array_t<float*>  cdf_A; // [n_k][n_s + 1][n_tau + 1]
    // the cumulatives in t = ln(k - 2): uniform nodes t0 + i dt_cum, i = 0..n_cum-1
    int              n_cum { 0 };
    double           cum_t0 { 0.0 }, cum_dt { 0.0 }, cum_delta0 { 0.0 }, cum_kmax { 0.0 };
    array_t<double*> ln_phi, ln_psi2, ln_psir;
    // the photon-energy tables of a broad bath
    bool             mono { true };
    double           eps0 { 0.0 }, gamma_th { 0.0 };
    int              n_gamma { 0 }, n_eps { 0 };
    double           eps_min_bath { 0.0 }, ln_eps_hi { 0.0 }; // the per-node grids (round 2)
    array_t<double*> y_nodes; // log10(Gamma - Gamma_th)
    array_t<float*>  cdf_u, cdf_w; // [n_gamma][n_eps + 1] at x_j = j/n_eps
    // bookkeeping
    std::size_t bytes { 0 };

    // ---------------------------------------------------------------------- //
    // inversion helpers
    // ---------------------------------------------------------------------- //

    /// position in [0, n] (cell units) where the 2-mixture of float CDFs reaches u
    Inline auto invert2(const std::size_t o0, const std::size_t o1, double w0, double w1, int n, double u,
                        const array_t<float*>& C) const -> double {
      int lo = 0, hi = n;
      while (hi - lo > 1) {
        const int    mid = (lo + hi) / 2;
        const double v   = w0 * static_cast<double>(C(o0 + mid)) + w1 * static_cast<double>(C(o1 + mid));
        if (v <= u) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const double c0 = w0 * static_cast<double>(C(o0 + lo)) + w1 * static_cast<double>(C(o1 + lo));
      const double c1 = w0 * static_cast<double>(C(o0 + lo + 1)) + w1 * static_cast<double>(C(o1 + lo + 1));
      double       f  = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5;
      f               = f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f);
      return static_cast<double>(lo) + f;
    }

    /// position where the 4-mixture of float CDFs reaches u
    Inline auto invert4(const std::size_t o[4], const double w[4], int n, double u, const array_t<float*>& C) const
      -> double {
      auto value = [&](int j) {
        double v = 0.0;
        for (int a = 0; a < 4; ++a) {
          v += w[a] * static_cast<double>(C(o[a] + j));
        }
        return v;
      };
      int lo = 0, hi = n;
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
      f               = f < 0.0 ? 0.0 : (f > 1.0 ? 1.0 : f);
      return static_cast<double>(lo) + f;
    }

    /// bracketing eps' nodes and the weight of the upper one
    Inline void locate_k(double k, int& i, double& w) const {
      const double t = math::log(k - 2.0 > 1.0e-300 ? k - 2.0 : 1.0e-300);
      if (t <= t_k(0)) {
        i = 0;
        w = 0.0;
        return;
      }
      if (t >= t_k(n_k - 1)) {
        i = n_k - 2;
        w = 1.0;
        return;
      }
      int lo = 0, hi = n_k - 1;
      while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (t_k(mid) <= t) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      i = lo;
      w = (t - t_k(lo)) / (t_k(lo + 1) - t_k(lo));
    }

    Inline auto sample_s(double k, double u) const -> double {
      int    i;
      double w;
      locate_k(k, i, w);
      const std::size_t o0 = static_cast<std::size_t>(i) * (n_s + 1);
      const std::size_t o1 = static_cast<std::size_t>(i + 1) * (n_s + 1);
      return invert2(o0, o1, 1.0 - w, w, n_s, u, cdf_E) / n_s;
    }

    Inline auto sample_tau(double k, double s, double u) const -> double {
      int    i;
      double w;
      locate_k(k, i, w);
      const double sc = (s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s)) * n_s;
      int          j  = static_cast<int>(sc);
      if (j > n_s - 1) {
        j = n_s - 1;
      }
      const double      v   = sc - static_cast<double>(j);
      const std::size_t row = static_cast<std::size_t>(n_tau + 1);
      const std::size_t o[4] = { (static_cast<std::size_t>(i) * (n_s + 1) + j) * row,
                                 (static_cast<std::size_t>(i) * (n_s + 1) + j + 1) * row,
                                 (static_cast<std::size_t>(i + 1) * (n_s + 1) + j) * row,
                                 (static_cast<std::size_t>(i + 1) * (n_s + 1) + j + 1) * row };
      const double      W[4] = { (1.0 - w) * (1.0 - v), (1.0 - w) * v, w * (1.0 - v), w * v };
      return invert4(o, W, n_tau, u, cdf_A) / n_tau;
    }

    Inline static auto E_of_s(double k, double s) -> double {
      const double sn = math::sin(0.5 * constants::PI * s);
      return 1.0 + (k - 2.0) * sn * sn;
    }

    Inline static auto momentum(double E) -> double {
      return E > 1.0 ? math::sqrt((E - 1.0) * (E + 1.0)) : 0.0;
    }

    Inline static auto omc_of_tau(double E, double tau) -> double {
      const double p  = momentum(E);
      const double w  = p > 0.0 ? 1.0 / ((E + p) * p) : 2.0;
      const double t  = math::log(w) + tau * (math::log(2.0 + w) - math::log(w));
      const double om = math::exp(t) - w;
      return om < 0.0 ? 0.0 : (om > 2.0 ? 2.0 : om);
    }

    // ---------------------------------------------------------------------- //
    // the cumulatives
    // ---------------------------------------------------------------------- //
    Inline auto cum_eval(const array_t<double*>& ln, double delta, double power) const -> double {
      if (!(delta > 0.0)) {
        return 0.0;
      }
      if (delta <= cum_delta0) {
        return math::exp(ln(0)) * math::pow(delta / cum_delta0, power);
      }
      if (delta >= cum_kmax - 2.0) {
        return math::exp(ln(n_cum - 1));
      }
      const double t  = math::log(delta);
      int          j  = static_cast<int>((t - cum_t0) / cum_dt);
      j               = j < 0 ? 0 : (j > n_cum - 2 ? n_cum - 2 : j);
      const double tj = cum_t0 + j * cum_dt;
      const double f  = (t - tj) / cum_dt;
      return math::exp(ln(j) + f * (ln(j + 1) - ln(j)));
    }

    Inline auto Phi(double delta) const -> double {
      return cum_eval(ln_phi, delta, 4.0);
    }

    Inline auto F(double delta, double gamma, double gamma_beta, double eps) const -> double {
      return gamma * cum_eval(ln_psi2, delta, 4.0) - eps * Phi(delta) + gamma_beta * cum_eval(ln_psir, delta, 4.5);
    }

    Inline auto invert_Phi(double target) const -> double {
      if (!(target > 0.0)) {
        return 0.0;
      }
      const double lt = math::log(target);
      if (lt <= ln_phi(0)) {
        return math::pow(24.0 * target / constants::PI, 0.25);
      }
      if (lt >= ln_phi(n_cum - 1)) {
        return cum_kmax - 2.0;
      }
      int lo = 0, hi = n_cum - 1;
      while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (ln_phi(mid) <= lt) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const double f = (lt - ln_phi(lo)) / (ln_phi(lo + 1) - ln_phi(lo));
      return math::exp(cum_t0 + (lo + f) * cum_dt);
    }

    Inline auto invert_F(double target, double gamma, double gamma_beta, double eps) const -> double {
      if (!(target > 0.0)) {
        return 0.0;
      }
      auto lnF = [&](int j) {
        return math::log(gamma * math::exp(ln_psi2(j)) - eps * math::exp(ln_phi(j)) +
                         gamma_beta * math::exp(ln_psir(j)));
      };
      const double lt = math::log(target);
      if (lt <= lnF(0)) {
        const double c = gamma * constants::PI / 12.0 - eps * constants::PI / 24.0;
        return c > 0.0 ? math::pow(target / c, 0.25) : cum_delta0;
      }
      int lo = 0, hi = n_cum - 1;
      if (lt >= lnF(hi)) {
        return cum_kmax - 2.0;
      }
      while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (lnF(mid) <= lt) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const double a = lnF(lo), b = lnF(lo + 1);
      const double f = (lt - a) / (b - a);
      return math::exp(cum_t0 + (lo + f) * cum_dt);
    }

    // ---------------------------------------------------------------------- //
    // the photon energy of a broad bath
    // ---------------------------------------------------------------------- //
    Inline auto sample_eps(double gamma, bool weighted, double u) const -> double {
      if (mono) {
        return eps0;
      }
      const double yy = math::log10(gamma - gamma_th > 1.0e-300 ? gamma - gamma_th : 1.0e-300);
      int          i;
      double       w;
      if (yy <= y_nodes(0)) {
        i = 0;
        w = 0.0;
      } else if (yy >= y_nodes(n_gamma - 1)) {
        i = n_gamma - 2;
        w = 1.0;
      } else {
        int lo = 0, hi = n_gamma - 1;
        while (hi - lo > 1) {
          const int mid = (lo + hi) / 2;
          if (y_nodes(mid) <= yy) {
            lo = mid;
          } else {
            hi = mid;
          }
        }
        i = lo;
        w = (yy - y_nodes(lo)) / (y_nodes(lo + 1) - y_nodes(lo));
      }
      const std::size_t o0  = static_cast<std::size_t>(i) * (n_eps + 1);
      const std::size_t o1  = static_cast<std::size_t>(i + 1) * (n_eps + 1);
      const double      pos = invert2(o0, o1, 1.0 - w, w, n_eps, u, weighted ? cdf_w : cdf_u);
      // x = pos/n_eps on the grid of this Gamma: [eps_lo(Gamma), eps_max] in ln eps
      const double A      = gamma + math::sqrt((gamma - 1.0) * (gamma + 1.0));
      const double ln_lo  = math::log(eps_min_bath > 2.0 / A ? eps_min_bath : 2.0 / A);
      return math::exp(ln_lo + (pos / n_eps) * (ln_eps_hi - ln_lo));
    }

    // ---------------------------------------------------------------------- //
    // the chains
    // ---------------------------------------------------------------------- //

    /**
     * @brief One draw of the unweighted (weighted = false) or D_max-weighted
     *        chain for an ion of Lorentz factor gamma; lab momenta in the frame
     *        with z along u^. Every random number drawn is counted in n_rng.
     * @return true on success; false when the weighted chain did not accept
     *         within max_attempts (out holds the last trial; the caller counts)
     */
    template <class Gen>
    Inline auto draw(double gamma, bool weighted, Gen& gen, unsigned long long& n_rng, int max_attempts,
                     SampledPair& out) const -> bool {
      const double gb   = math::sqrt((gamma - 1.0) * (gamma + 1.0));
      const double beta = gb / gamma;
      const double A    = gamma + gb;
      for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        out.attempts = attempt;
        double eps   = sample_eps(gamma, weighted, gen.drand());
        double dhi   = eps * A - 2.0;
        n_rng += 1;
        if (!(dhi > 0.0)) {
          eps = 2.0 / A * (1.0 + 1.0e-12);
          dhi = eps * A - 2.0;
        }
        const double u2 = gen.drand();
        n_rng += 1;
        double delta = weighted ? invert_F(u2 * F(dhi, gamma, gb, eps), gamma, gb, eps)
                                : invert_Phi(u2 * Phi(dhi));
        delta        = delta < 0.0 ? 0.0 : (delta > dhi ? dhi : delta);
        const double k = 2.0 + delta;
        // photon rest-frame direction relative to u^ = z
        double cos_k = (eps / (gamma * k) - 1.0) / beta;
        cos_k        = cos_k < -1.0 ? -1.0 : (cos_k > 1.0 ? 1.0 : cos_k);
        const double sin_k = math::sqrt((1.0 - cos_k) * (1.0 + cos_k) > 0.0 ? (1.0 - cos_k) * (1.0 + cos_k) : 0.0);
        const double phi_k = 2.0 * constants::PI * gen.drand();
        n_rng += 1;
        const double kx = sin_k * math::cos(phi_k), ky = sin_k * math::sin(phi_k), kz = cos_k;
        // lepton energies
        const double s  = sample_s(k, gen.drand());
        n_rng += 1;
        const double Em = E_of_s(k, s), Ep = k - Em;
        const double pm_ = momentum(Em), pp_ = momentum(Ep);
        // angles to k^' and azimuths
        const double omcm = omc_of_tau(Em, sample_tau(k, s, gen.drand()));
        const double omcp = omc_of_tau(Ep, sample_tau(k, 1.0 - s, gen.drand()));
        const double phm = 2.0 * constants::PI * gen.drand(), php = 2.0 * constants::PI * gen.drand();
        n_rng += 4;
        const double cm = 1.0 - omcm, sm = math::sqrt(omcm * (2.0 - omcm) > 0.0 ? omcm * (2.0 - omcm) : 0.0);
        const double cp = 1.0 - omcp, sp = math::sqrt(omcp * (2.0 - omcp) > 0.0 ? omcp * (2.0 - omcp) : 0.0);
        // orthonormal basis (e1, e2) perpendicular to k^' (same construction as the host)
        double e1x, e1y, e1z;
        if (math::abs(kx) < 0.9) {
          e1x = 1.0 - kx * kx;
          e1y = -kx * ky;
          e1z = -kx * kz;
        } else {
          e1x = -ky * kx;
          e1y = 1.0 - ky * ky;
          e1z = -ky * kz;
        }
        const double l1 = math::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
        e1x /= l1;
        e1y /= l1;
        e1z /= l1;
        const double e2x = ky * e1z - kz * e1y, e2y = kz * e1x - kx * e1z, e2z = kx * e1y - ky * e1x;
        // rest-frame momenta
        double qm[3], qp[3];
        {
          const double ax = sm * math::cos(phm), ay = sm * math::sin(phm);
          qm[0] = pm_ * (ax * e1x + ay * e2x + cm * kx);
          qm[1] = pm_ * (ax * e1y + ay * e2y + cm * ky);
          qm[2] = pm_ * (ax * e1z + ay * e2z + cm * kz);
          const double bx = sp * math::cos(php), by = sp * math::sin(php);
          qp[0] = pp_ * (bx * e1x + by * e2x + cp * kx);
          qp[1] = pp_ * (bx * e1y + by * e2y + cp * ky);
          qp[2] = pp_ * (bx * e1z + by * e2z + cp * kz);
        }
        // boost along z: E = gamma E' + gamma beta p'_z, p_z = gamma p'_z + gamma beta E'
        out.E_m   = gamma * Em + gb * qm[2];
        out.pm[0] = qm[0];
        out.pm[1] = qm[1];
        out.pm[2] = gamma * qm[2] + gb * Em;
        out.E_p   = gamma * Ep + gb * qp[2];
        out.pp[0] = qp[0];
        out.pp[1] = qp[1];
        out.pp[2] = gamma * qp[2] + gb * Ep;
        // the absorbed photon in the lab: cos theta_k = (1 - eps'/(Gamma eps))/beta, same azimuth
        double cos_l = (1.0 - k / (gamma * eps)) / beta;
        cos_l        = cos_l < -1.0 ? -1.0 : (cos_l > 1.0 ? 1.0 : cos_l);
        const double sin_l = math::sqrt((1.0 - cos_l) * (1.0 + cos_l) > 0.0 ? (1.0 - cos_l) * (1.0 + cos_l) : 0.0);
        out.kph[0] = eps * sin_l * math::cos(phi_k);
        out.kph[1] = eps * sin_l * math::sin(phi_k);
        out.kph[2] = eps * cos_l;
        out.eps    = eps;
        out.k      = k;
        out.Em     = Em;
        out.Ep     = Ep;
        out.D      = out.E_m + out.E_p - eps;
        out.Dmax   = gamma * k - eps + gb * math::sqrt((k - 2.0) * (k + 2.0) > 0.0 ? (k - 2.0) * (k + 2.0) : 0.0);
        if (!weighted) {
          return true;
        }
        const double ua = gen.drand();
        n_rng += 1;
        if (ua * out.Dmax <= out.D) {
          return true;
        }
      }
      return false;
    }
  };

  /**
   * @brief Copy the host sampler tables to device views (single precision for
   *        the CDFs); throws on inconsistent tables; the total size in bytes is
   *        recorded in `bytes`
   */
  inline auto mirror_sampler_to_device(const rf::RestFrameTables&    T,
                                       const rf::SamplerCumulatives& C,
                                       const rf::BathEpsTables&      B) -> SamplerTables {
    SamplerTables D;
    if (T.n_k() < 2 || T.n_s < 1 || T.n_tau < 1) {
      throw std::invalid_argument("mirror_sampler_to_device: empty rest-frame tables");
    }
    if (T.cdf_E.size() != T.n_k() * static_cast<std::size_t>(T.n_s + 1) ||
        T.cdf_A.size() != T.n_k() * static_cast<std::size_t>(T.n_s + 1) * static_cast<std::size_t>(T.n_tau + 1)) {
      throw std::invalid_argument("mirror_sampler_to_device: inconsistent rest-frame table sizes");
    }
    D.n_k   = static_cast<int>(T.n_k());
    D.n_s   = T.n_s;
    D.n_tau = T.n_tau;
    D.t_k   = array_t<double*> { "bh_s_tk", T.n_k() };
    D.cdf_E = array_t<float*> { "bh_s_cdfE", T.cdf_E.size() };
    D.cdf_A = array_t<float*> { "bh_s_cdfA", T.cdf_A.size() };
    {
      auto h = Kokkos::create_mirror_view(D.t_k);
      for (std::size_t i = 0; i < T.n_k(); ++i) {
        h(i) = T.t_k[i];
      }
      Kokkos::deep_copy(D.t_k, h);
    }
    {
      auto h = Kokkos::create_mirror_view(D.cdf_E);
      for (std::size_t i = 0; i < T.cdf_E.size(); ++i) {
        h(i) = static_cast<float>(T.cdf_E[i]);
      }
      Kokkos::deep_copy(D.cdf_E, h);
    }
    {
      auto h = Kokkos::create_mirror_view(D.cdf_A);
      for (std::size_t i = 0; i < T.cdf_A.size(); ++i) {
        h(i) = static_cast<float>(T.cdf_A[i]);
      }
      Kokkos::deep_copy(D.cdf_A, h);
    }
    // cumulatives
    if (C.size() < 2) {
      throw std::invalid_argument("mirror_sampler_to_device: empty cumulatives");
    }
    D.n_cum      = static_cast<int>(C.size());
    D.cum_t0     = C.t().front();
    D.cum_dt     = (C.t().back() - C.t().front()) / static_cast<double>(C.size() - 1);
    D.cum_delta0 = C.delta0();
    D.cum_kmax   = C.kmax();
    D.ln_phi     = array_t<double*> { "bh_s_lnphi", C.size() };
    D.ln_psi2    = array_t<double*> { "bh_s_lnpsi2", C.size() };
    D.ln_psir    = array_t<double*> { "bh_s_lnpsir", C.size() };
    {
      auto h1 = Kokkos::create_mirror_view(D.ln_phi);
      auto h2 = Kokkos::create_mirror_view(D.ln_psi2);
      auto h3 = Kokkos::create_mirror_view(D.ln_psir);
      for (std::size_t i = 0; i < C.size(); ++i) {
        h1(i) = C.lnPhi()[i];
        h2(i) = C.lnPsi2()[i];
        h3(i) = C.lnPsir()[i];
      }
      Kokkos::deep_copy(D.ln_phi, h1);
      Kokkos::deep_copy(D.ln_psi2, h2);
      Kokkos::deep_copy(D.ln_psir, h3);
    }
    // bath
    D.mono     = B.mono;
    D.eps0     = B.eps0;
    D.gamma_th = B.gamma_th;
    std::size_t bath_bytes = 0;
    if (!B.mono) {
      if (B.gamma.size() < 2 || B.n_eps < 1 || B.cdf_u.size() != B.gamma.size() * static_cast<std::size_t>(B.n_eps + 1)) {
        throw std::invalid_argument("mirror_sampler_to_device: inconsistent photon-energy tables");
      }
      D.n_gamma      = static_cast<int>(B.gamma.size());
      D.n_eps        = B.n_eps;
      D.eps_min_bath = B.eps_min_bath;
      D.ln_eps_hi    = B.ln_eps_hi;
      D.y_nodes      = array_t<double*> { "bh_s_y", B.y.size() };
      D.cdf_u        = array_t<float*> { "bh_s_cdfu", B.cdf_u.size() };
      D.cdf_w        = array_t<float*> { "bh_s_cdfw", B.cdf_w.size() };
      auto hy = Kokkos::create_mirror_view(D.y_nodes);
      auto hu = Kokkos::create_mirror_view(D.cdf_u);
      auto hw = Kokkos::create_mirror_view(D.cdf_w);
      for (std::size_t i = 0; i < B.y.size(); ++i) {
        hy(i) = B.y[i];
      }
      for (std::size_t i = 0; i < B.cdf_u.size(); ++i) {
        hu(i) = static_cast<float>(B.cdf_u[i]);
        hw(i) = static_cast<float>(B.cdf_w[i]);
      }
      Kokkos::deep_copy(D.y_nodes, hy);
      Kokkos::deep_copy(D.cdf_u, hu);
      Kokkos::deep_copy(D.cdf_w, hw);
      bath_bytes = B.y.size() * sizeof(double) + 2 * B.cdf_u.size() * sizeof(float);
    }
    D.bytes = T.n_k() * sizeof(double) + (T.cdf_E.size() + T.cdf_A.size()) * sizeof(float) +
              3 * C.size() * sizeof(double) + bath_bytes;
    D.valid = true;
    return D;
  }

  /**
   * @brief Build the host sampler tables for a bath and its rate table and
   *        mirror them to the device (the generator's startup path)
   */
  inline auto build_sampler_tables(const PhotonBath& bath, const HadronicTablesHost& H, unsigned n_threads = 0)
    -> SamplerTables {
    const rf::RestFrameTables    T = rf::RestFrameTables::build(16, 256, 256, 8, n_threads);
    const rf::SamplerCumulatives C;
    const rf::BathEpsTables      B = rf::BathEpsTables::build(bath, H, C, 2048, 8, n_threads);
    return mirror_sampler_to_device(T, C, B);
  }

} // namespace arch::bh

#endif // ARCHETYPES_HADRONIC_BH_SAMPLER_DEVICE_H
