// Reproduction on Entity 1.6.0rc (861b19a7), no Bethe-Heitler code:
// continuity of the current deposit for one particle in 3D Cartesian.
//
//   repro_deposit_3d
//
// One particle of charge 1 and weight 1 moves in a straight line by
// (a, b, c) cells in one step. rho is the charge on the nodes with the linear
// (cloud-in-cell) shape at the old and at the new position; J is what
// kernel::DepositCurrents_kernel deposits. The program prints the largest
// |rho_new - rho_old + dt div J| over the nodes, and a b c / 16.
// Compiled with SHAPE_ORDER 0 (zig-zag) or 1 (Esirkepov, linear shape).
#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "metrics/minkowski.h"

#include "framework/containers/particles.h"
#include "kernels/deposition/currents/global.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace ntt;

namespace {
  using metric_t  = metric::Minkowski<Dim::_3D>;
  using species_t = Particles<Dim::_3D, Coord::Cartesian>;
  constexpr int N = 8;

  auto node(int i, int j, int k) -> std::size_t {
    const auto w = [](int v) {
      return static_cast<std::size_t>(((v % N) + N) % N);
    };
    return w(i) + N * (w(j) + N * w(k));
  }

  // charge of a unit particle at (cell + offset) on the nodes, linear shape
  void charge(const int cell[3], const double off[3], double sign, std::vector<double>& rho) {
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        for (int c = 0; c < 2; ++c) {
          rho[node(cell[0] + a, cell[1] + b, cell[2] + c)] +=
            sign * (a ? off[0] : 1.0 - off[0]) * (b ? off[1] : 1.0 - off[1]) *
            (c ? off[2] : 1.0 - off[2]);
        }
      }
    }
  }

  // largest |rho_new - rho_old + dt div J| for a move from x0 by d (cells)
  auto residual(const double x0[3], const double d[3], real_t dt) -> double {
    const metric_t metric {
      { N, N, N },
      { { 0.0, static_cast<real_t>(N) },
       { 0.0, static_cast<real_t>(N) },
       { 0.0, static_cast<real_t>(N) } },
      {}
    };
    species_t prtls(1, "p", 1.0f, 1.0f, 1u, 0u, 0u, ParticlePusher::BORIS, false,
                    RadiativeDrag::NONE, EmissionType::NONE, 0u, 0u);
    int    c0[3], c1[3];
    double o0[3], o1[3];
    for (int a = 0; a < 3; ++a) {
      c0[a] = static_cast<int>(std::floor(x0[a]));
      o0[a] = x0[a] - c0[a];
      const double x1 = x0[a] + d[a];
      c1[a] = static_cast<int>(std::floor(x1));
      o1[a] = x1 - c1[a];
    }
    Kokkos::deep_copy(prtls.i1_prev, c0[0]);
    Kokkos::deep_copy(prtls.i2_prev, c0[1]);
    Kokkos::deep_copy(prtls.i3_prev, c0[2]);
    Kokkos::deep_copy(prtls.dx1_prev, static_cast<prtldx_t>(o0[0]));
    Kokkos::deep_copy(prtls.dx2_prev, static_cast<prtldx_t>(o0[1]));
    Kokkos::deep_copy(prtls.dx3_prev, static_cast<prtldx_t>(o0[2]));
    Kokkos::deep_copy(prtls.i1, c1[0]);
    Kokkos::deep_copy(prtls.i2, c1[1]);
    Kokkos::deep_copy(prtls.i3, c1[2]);
    Kokkos::deep_copy(prtls.dx1, static_cast<prtldx_t>(o1[0]));
    Kokkos::deep_copy(prtls.dx2, static_cast<prtldx_t>(o1[1]));
    Kokkos::deep_copy(prtls.dx3, static_cast<prtldx_t>(o1[2]));
    // momentum consistent with the displacement (not used by the 3D deposit)
    const double v2 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / (dt * dt);
    const double g  = 1.0 / std::sqrt(1.0 - v2);
    Kokkos::deep_copy(prtls.ux1, static_cast<real_t>(g * d[0] / dt));
    Kokkos::deep_copy(prtls.ux2, static_cast<real_t>(g * d[1] / dt));
    Kokkos::deep_copy(prtls.ux3, static_cast<real_t>(g * d[2] / dt));
    Kokkos::deep_copy(prtls.weight, ONE);
    Kokkos::deep_copy(prtls.tag, static_cast<short>(ParticleTag::alive));
    prtls.set_npart(1u);
    // the offsets as stored
    auto read = [](const array_t<prtldx_t*>& v) {
      auto h = Kokkos::create_mirror_view(v);
      Kokkos::deep_copy(h, v);
      return static_cast<double>(h(0));
    };
    const double s0[3] = { read(prtls.dx1_prev), read(prtls.dx2_prev), read(prtls.dx3_prev) };
    const double s1[3] = { read(prtls.dx1), read(prtls.dx2), read(prtls.dx3) };

    const auto NG = static_cast<int>(N_GHOSTS);
    ndfield_t<Dim::_3D, 3> cur { "cur", N + 2 * NG, N + 2 * NG, N + 2 * NG };
    auto scatter_cur = Kokkos::Experimental::create_scatter_view(cur);
    Kokkos::parallel_for(
      "CurrentsDeposit",
      prtls.rangeActiveParticles(),
      kernel::DepositCurrents_kernel<SimEngine::SRPIC, metric_t, SHAPE_ORDER>(
        scatter_cur,
        prtls,
        metric,
        ONE,
        dt));
    Kokkos::Experimental::contribute(cur, scatter_cur);
    auto cur_h = Kokkos::create_mirror_view(cur);
    Kokkos::deep_copy(cur_h, cur);
    std::vector<std::vector<double>> J(3, std::vector<double>(N * N * N, 0.0));
    for (int i = 0; i < N + 2 * NG; ++i) {
      for (int j = 0; j < N + 2 * NG; ++j) {
        for (int k = 0; k < N + 2 * NG; ++k) {
          for (int c = 0; c < 3; ++c) {
            J[c][node(i - NG, j - NG, k - NG)] += static_cast<double>(cur_h(i, j, k, c));
          }
        }
      }
    }
    std::vector<double> drho(N * N * N, 0.0);
    charge(c1, s1, +1.0, drho);
    charge(c0, s0, -1.0, drho);
    double r = 0.0;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
          const double div = (J[0][node(i, j, k)] - J[0][node(i - 1, j, k)]) +
                             (J[1][node(i, j, k)] - J[1][node(i, j - 1, k)]) +
                             (J[2][node(i, j, k)] - J[2][node(i, j, k - 1)]);
          r = std::max(r, std::abs(drho[node(i, j, k)] + static_cast<double>(dt) * div));
        }
      }
    }
    return r;
  }
} // namespace

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);
  {
    std::printf("SHAPE_ORDER %d, real_t %zu bytes, prtldx_t %zu bytes\n",
                static_cast<int>(SHAPE_ORDER), sizeof(real_t), sizeof(prtldx_t));
    const double v[3] = { 0.6, 0.5, 0.4 };
    std::printf("-- within one cell: start (3.2, 3.3, 3.4), velocity (0.6, 0.5, 0.4)\n");
    for (const double dt : { 0.2, 0.1, 0.05, 0.025 }) {
      const double x0[3] = { 3.2, 3.3, 3.4 };
      const double d[3]  = { v[0] * dt, v[1] * dt, v[2] * dt };
      std::printf("dt %.3f: displacement (%.4f, %.4f, %.4f), residual %.11e, a b c / 16 = %.11e\n",
                  dt, d[0], d[1], d[2], residual(x0, d, static_cast<real_t>(dt)),
                  d[0] * d[1] * d[2] / 16.0);
    }
    std::printf("-- within one cell, in a plane: velocity (0.6, 0.5, 0)\n");
    for (const double dt : { 0.2, 0.1 }) {
      const double x0[3] = { 3.2, 3.3, 3.4 };
      const double d[3]  = { 0.6 * dt, 0.5 * dt, 0.0 };
      std::printf("dt %.3f: residual %.11e\n", dt, residual(x0, d, static_cast<real_t>(dt)));
    }
    std::printf("-- across a face in x1: start (3.95, 3.3, 3.4), velocity (0.6, 0.5, 0.4)\n");
    for (const double dt : { 0.2, 0.1 }) {
      const double x0[3] = { 3.95, 3.3, 3.4 };
      const double d[3]  = { v[0] * dt, v[1] * dt, v[2] * dt };
      // two straight segments, split at the face in x1 and at the midpoint in x2, x3
      const double a1 = 4.0 - x0[0], a2 = d[0] - a1;
      std::printf("dt %.3f: residual %.11e, (a1 + a2) b c / 16 = %.11e\n", dt,
                  residual(x0, d, static_cast<real_t>(dt)),
                  (a1 + a2) * d[1] * d[2] / 16.0);
    }
  }
  GlobalFinalize();
  return 0;
}
