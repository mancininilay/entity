/**
 * @file tests/kernels/reduced_stats_weights.cpp
 * @brief T-P5: the reduced particle statistics (N, Rho, Charge, Npart, T00,
 *        T0i, Tij) with `use_weights = true`, mass != 1, weights != 1 and both
 *        charge signs, against a direct double sum over the arrays of
 *          N      = sum w dV,  Rho = sum w m dV,  Charge = sum w q dV,
 *          Npart  = count,
 *          T^{ab} = sum w dV p^a p^b / E,  p^0 = E = m gamma, p^i = m u^i,
 *        2D Minkowski, SRPIC. Tolerance 1e-6 relative to the sum of the
 *        absolute contributions (so that cancelling sums such as the net charge
 *        of a quasi-neutral set are judged against their natural scale). The
 *        statistics accumulate in real_t, so the particle count is kept small
 *        (8) for the registered checks; a 64-particle set is reported as (diag).
 *        Also checked: `use_weights = false` reproduces the unweighted sums
 *        (weights ignored, mass and charge factors kept).
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include "framework/containers/particles.h"
#include "kernels/reduced_stats.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;

namespace {

  int n_pass = 0, n_fail = 0;

  void verdict(bool ok, const std::string& what, double value, double tol) {
    std::printf("%s: %s : %.6e (tolerance %.3e)\n", ok ? "PASS" : "FAIL", what.c_str(), value, tol);
    if (ok) {
      ++n_pass;
    } else {
      ++n_fail;
    }
  }

  using M2 = Minkowski<Dim::_2D>;

  struct Set {
    std::vector<int>    i1, i2;
    std::vector<double> dx1, dx2, u1, u2, u3, w;
  };

  auto splitmix(std::uint64_t x) -> std::uint64_t {
    x += 0x9E3779B97F4A7C15ULL;
    x  = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x  = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
  }

  auto unit(std::uint64_t x) -> double {
    return static_cast<double>(splitmix(x) >> 11) * 0x1.0p-53;
  }

  auto make_set(int n, int n1, int n2, std::uint64_t seed) -> Set {
    Set s;
    for (int p = 0; p < n; ++p) {
      const std::uint64_t h = seed + 17ULL * p;
      s.i1.push_back(static_cast<int>(unit(h + 1) * n1));
      s.i2.push_back(static_cast<int>(unit(h + 2) * n2));
      // positions and momenta rounded to the storage types so that the reference
      // sum sees exactly the stored values
      s.dx1.push_back(static_cast<double>(static_cast<prtldx_t>(unit(h + 3))));
      s.dx2.push_back(static_cast<double>(static_cast<prtldx_t>(unit(h + 4))));
      s.u1.push_back(static_cast<double>(static_cast<real_t>(4.0 * unit(h + 5) - 2.0)));
      s.u2.push_back(static_cast<double>(static_cast<real_t>(4.0 * unit(h + 6) - 2.0)));
      s.u3.push_back(static_cast<double>(static_cast<real_t>(4.0 * unit(h + 7) - 2.0)));
      s.w.push_back(static_cast<double>(static_cast<real_t>(0.5 + 3.0 * unit(h + 8))));
    }
    return s;
  }

  template <typename T, typename ViewT>
  void upload(ViewT& view, const std::vector<T>& v) {
    auto h = Kokkos::create_mirror_view(view);
    for (std::size_t p = 0; p < v.size(); ++p) {
      h(p) = v[p];
    }
    Kokkos::deep_copy(view, h);
  }

  template <StatsID::type P>
  auto compute(const Particles<Dim::_2D, Coord::Cartesian>& prtls,
               const std::vector<uint8_t>&                  comps,
               bool                                         use_weights,
               const M2&                                    metric) -> double {
    real_t buff = ZERO;
    Kokkos::parallel_reduce("stat", prtls.rangeActiveParticles(),
                            kernel::ReducedParticleMoments_kernel<SimEngine::SRPIC, M2, P>(
                              comps, prtls, use_weights, metric),
                            buff);
    return static_cast<double>(buff);
  }

  /**
   * @brief reference sums in double; `scale` receives the sum of absolute contributions
   */
  struct Ref {
    double N { 0 }, Rho { 0 }, Charge { 0 }, Npart { 0 };
    double T[4][4] {};
    double sN { 0 }, sRho { 0 }, sCharge { 0 }, sT[4][4] {};
  };

  auto reference(const Set& s, const M2& metric, double mass, double charge, bool use_weights) -> Ref {
    Ref r;
    for (std::size_t p = 0; p < s.w.size(); ++p) {
      coord_t<Dim::_2D> x { static_cast<real_t>(s.i1[p] + s.dx1[p]), static_cast<real_t>(s.i2[p] + s.dx2[p]) };
      const double dV = static_cast<double>(metric.sqrt_det_h(x));
      const double w  = use_weights ? s.w[p] : 1.0;
      const double E  = mass * std::sqrt(1.0 + s.u1[p] * s.u1[p] + s.u2[p] * s.u2[p] + s.u3[p] * s.u3[p]);
      const double pv[4] = { E, mass * s.u1[p], mass * s.u2[p], mass * s.u3[p] };
      r.N      += w * dV;
      r.sN     += std::abs(w * dV);
      r.Rho    += w * mass * dV;
      r.sRho   += std::abs(w * mass * dV);
      r.Charge += w * charge * dV;
      r.sCharge += std::abs(w * charge * dV);
      r.Npart  += 1.0;
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          const double t = w * dV * pv[a] * pv[b] / E;
          r.T[a][b]  += t;
          r.sT[a][b] += std::abs(t);
        }
      }
    }
    return r;
  }

  void run_case(int n, double mass, double charge, bool use_weights, bool registered, std::uint64_t seed) {
    const std::vector<ncells_t> res { 8, 6 };
    const boundaries_t<real_t>  ext { { ZERO, static_cast<real_t>(4.0) }, { ZERO, static_cast<real_t>(3.0) } };
    M2                          metric { res, ext, {} };
    const auto                  s = make_set(n, 8, 6, seed);
    Particles<Dim::_2D, Coord::Cartesian> prtls { 1u, "s", static_cast<float>(mass), static_cast<float>(charge),
                                                  static_cast<npart_t>(n), 0, 0, ParticlePusher::BORIS, false,
                                                  RadiativeDrag::NONE, EmissionType::NONE, 0, 0 };
    upload(prtls.i1, s.i1);
    upload(prtls.i2, s.i2);
    {
      std::vector<prtldx_t> d1, d2;
      std::vector<real_t>   a, b, c, w;
      for (std::size_t p = 0; p < s.w.size(); ++p) {
        d1.push_back(static_cast<prtldx_t>(s.dx1[p]));
        d2.push_back(static_cast<prtldx_t>(s.dx2[p]));
        a.push_back(static_cast<real_t>(s.u1[p]));
        b.push_back(static_cast<real_t>(s.u2[p]));
        c.push_back(static_cast<real_t>(s.u3[p]));
        w.push_back(static_cast<real_t>(s.w[p]));
      }
      upload(prtls.dx1, d1);
      upload(prtls.dx2, d2);
      upload(prtls.ux1, a);
      upload(prtls.ux2, b);
      upload(prtls.ux3, c);
      upload(prtls.weight, w);
    }
    Kokkos::deep_copy(prtls.tag, ParticleTag::alive);
    prtls.set_npart(static_cast<npart_t>(n));
    Kokkos::fence();

    // the species mass/charge are floats: the reference uses the stored float values
    const double m_f = static_cast<double>(prtls.mass());
    const double q_f = static_cast<double>(prtls.charge());
    const auto   ref = reference(s, metric, m_f, q_f, use_weights);
    const double tol = 1.0e-6;
    const std::string tag = std::string(registered ? "T-P5" : "(diag) T-P5") + " n = " + std::to_string(n) +
                            ", m = " + std::to_string(m_f) + ", q = " + std::to_string(q_f) +
                            (use_weights ? ", weighted" : ", unweighted") + ": ";
    auto check = [&](const std::string& what, double got, double expect, double scale) {
      const double err = std::abs(got - expect) / (scale > 0.0 ? scale : 1.0);
      if (registered) {
        verdict(err <= tol, tag + what, err, tol);
      } else {
        std::printf("(diag) %s%s : relative error %.3e (got %.10g, expected %.10g)\n", tag.c_str(),
                    what.c_str(), err, got, expect);
      }
    };
    check("N", compute<StatsID::N>(prtls, {}, use_weights, metric), ref.N, ref.sN);
    check("Rho (mass factor)", compute<StatsID::Rho>(prtls, {}, use_weights, metric), ref.Rho, ref.sRho);
    check("Charge (signed charge factor)", compute<StatsID::Charge>(prtls, {}, use_weights, metric), ref.Charge,
          ref.sCharge);
    check("Npart", compute<StatsID::Npart>(prtls, {}, use_weights, metric), ref.Npart, ref.Npart);
    check("T00", compute<StatsID::T>(prtls, { 0, 0 }, use_weights, metric), ref.T[0][0], ref.sT[0][0]);
    for (uint8_t i = 1; i <= 3; ++i) {
      check("T0" + std::to_string(i), compute<StatsID::T>(prtls, { 0, i }, use_weights, metric), ref.T[0][i],
            ref.sT[0][i]);
    }
    for (uint8_t i = 1; i <= 3; ++i) {
      for (uint8_t j = i; j <= 3; ++j) {
        check("T" + std::to_string(i) + std::to_string(j),
              compute<StatsID::T>(prtls, { i, j }, use_weights, metric), ref.T[i][j], ref.sT[i][j]);
      }
    }
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::printf("reduced_stats_weights: real_t = %s, execution space = %s\n",
                sizeof(real_t) == 4 ? "float" : "double", Kokkos::DefaultExecutionSpace::name());
    // registered: 8 particles, m = 3.5, charges -2 and +2, weighted and unweighted
    run_case(8, 3.5, -2.0, true, true, 11u);
    run_case(8, 3.5, 2.0, true, true, 23u);
    run_case(8, 3.5, -2.0, false, true, 37u);
    run_case(8, 1836.15, 1.0, true, true, 41u);
    // diagnostics: larger sets (float accumulation round-off grows with the count)
    run_case(64, 3.5, -2.0, true, false, 53u);
    run_case(64, 0.5, 1.0, true, false, 59u);
    std::printf("reduced_stats_weights summary: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0 || n_pass == 0) {
      rc = 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "reduced_stats_weights: exception: %s\n", e.what());
    rc = 2;
  }
  Kokkos::finalize();
  return rc;
}
