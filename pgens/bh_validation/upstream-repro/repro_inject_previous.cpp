// Reproduction on Entity 1.6.0rc (861b19a7), no Bethe-Heitler code: charged
// particles inserted with kernel::InjectParticle and deposited before their
// first push.
//
//   repro_inject_previous
//
// Two species of charge -1 and +1 in 2D Cartesian. Slot 0 of each still holds
// the previous position of a particle that was there earlier: two different
// points of one cell, written directly here. One particle of each species is
// then inserted at the same point of that cell with kernel::InjectParticle,
// as an emission policy that creates a pair would do, and the currents are
// deposited with no push in between, which is what happens to a species
// listed before the one that emits. The pair is neutral and has not moved, so
// the deposit should give no in-plane current. The program prints the largest
// in-plane current and the largest |div J| over the nodes, first with the
// slots as InjectParticle leaves them, then with the previous position set
// equal to the position by hand.
#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "metrics/minkowski.h"

#include "framework/containers/particles.h"
#include "kernels/deposition/currents/global.hpp"
#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace ntt;

namespace {
  using metric_t  = metric::Minkowski<Dim::_2D>;
  using species_t = Particles<Dim::_2D, Coord::Cartesian>;
  constexpr ncells_t NX1 = 8u, NX2 = 8u;
  constexpr int      CELL = 4;

  auto make_species(spidx_t index, float charge) -> species_t {
    return species_t(index,
                     "s" + std::to_string(index),
                     1.0f,
                     charge,
                     4u,
                     0u,
                     0u,
                     ParticlePusher::BORIS,
                     false,
                     RadiativeDrag::NONE,
                     EmissionType::NONE,
                     0u,
                     0u);
  }

  // what a particle that occupied slot 0 earlier has left in it
  void leave_previous(species_t& species, prtldx_t dx1, prtldx_t dx2) {
    Kokkos::deep_copy(species.i1_prev, CELL);
    Kokkos::deep_copy(species.i2_prev, CELL);
    Kokkos::deep_copy(species.dx1_prev, dx1);
    Kokkos::deep_copy(species.dx2_prev, dx2);
  }

  // one particle in slot 0, at the middle of the cell, with kernel::InjectParticle
  void insert(species_t& species) {
    const auto i1 = species.i1, i2 = species.i2, i3 = species.i3;
    const auto dx1 = species.dx1, dx2 = species.dx2, dx3 = species.dx3;
    const auto ux1 = species.ux1, ux2 = species.ux2, ux3 = species.ux3;
    const auto phi = species.phi, weight = species.weight;
    const auto tag   = species.tag;
    const auto pld_i = species.pld_i;
    Kokkos::parallel_for(
      "Insert",
      Kokkos::RangePolicy<>(0, 1),
      Lambda(const std::size_t) {
        const tuple_t<int, Dim::_2D>      xi { CELL, CELL };
        const tuple_t<prtldx_t, Dim::_2D> dxi { static_cast<prtldx_t>(0.5),
                                                static_cast<prtldx_t>(0.5) };
        kernel::InjectParticle<Dim::_2D, Coord::Cartesian, false>(
          0u, i1, i2, i3, dx1, dx2, dx3, ux1, ux2, ux3, phi, weight, tag, pld_i,
          xi, dxi, { static_cast<real_t>(0.3), static_cast<real_t>(-0.2), ZERO },
          ONE, ZERO);
      });
    species.set_npart(1u);
  }

  // previous position = position, by hand
  void set_previous_to_position(species_t& species) {
    Kokkos::deep_copy(species.i1_prev, species.i1);
    Kokkos::deep_copy(species.i2_prev, species.i2);
    Kokkos::deep_copy(species.dx1_prev, species.dx1);
    Kokkos::deep_copy(species.dx2_prev, species.dx2);
  }

  // currents of one species folded periodically onto the active cells
  void deposit(const species_t&                  species,
               const metric_t&                   metric,
               real_t                            dt,
               std::vector<std::vector<double>>& J) {
    ndfield_t<Dim::_2D, 3> cur { "cur", NX1 + 2u * N_GHOSTS, NX2 + 2u * N_GHOSTS };
    auto scatter_cur = Kokkos::Experimental::create_scatter_view(cur);
    Kokkos::parallel_for(
      "CurrentsDeposit",
      species.rangeActiveParticles(),
      kernel::DepositCurrents_kernel<SimEngine::SRPIC, metric_t, SHAPE_ORDER>(
        scatter_cur,
        species,
        metric,
        static_cast<real_t>(species.charge()),
        dt));
    Kokkos::Experimental::contribute(cur, scatter_cur);
    auto cur_h = Kokkos::create_mirror_view(cur);
    Kokkos::deep_copy(cur_h, cur);
    const int n1 = static_cast<int>(NX1), n2 = static_cast<int>(NX2);
    const int ng = static_cast<int>(N_GHOSTS);
    for (int g1 = 0; g1 < n1 + 2 * ng; ++g1) {
      for (int g2 = 0; g2 < n2 + 2 * ng; ++g2) {
        const int i = ((g1 - ng) % n1 + n1) % n1;
        const int j = ((g2 - ng) % n2 + n2) % n2;
        for (int c = 0; c < 3; ++c) {
          J[c][i + n1 * j] += static_cast<double>(cur_h(g1, g2, c));
        }
      }
    }
  }

  void report(const char* label, const species_t& electrons, const species_t& positrons,
              const metric_t& metric, real_t dt) {
    const int n1 = static_cast<int>(NX1), n2 = static_cast<int>(NX2);
    std::vector<std::vector<double>> J(3, std::vector<double>(NX1 * NX2, 0.0));
    deposit(electrons, metric, dt, J);
    deposit(positrons, metric, dt, J);
    double j_max = 0.0, div_max = 0.0;
    for (int i = 0; i < n1; ++i) {
      for (int j = 0; j < n2; ++j) {
        const int im = (i - 1 + n1) % n1, jm = (j - 1 + n2) % n2;
        j_max   = std::max({ j_max, std::abs(J[0][i + n1 * j]), std::abs(J[1][i + n1 * j]) });
        div_max = std::max(div_max,
                           std::abs((J[0][i + n1 * j] - J[0][im + n1 * j]) +
                                    (J[1][i + n1 * j] - J[1][i + n1 * jm])));
      }
    }
    std::printf("%-44s largest in-plane current %.6e, largest |div J| %.6e\n", label, j_max, div_max);
  }
} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  {
    const metric_t metric {
      { NX1, NX2 },
      { { 0.0, static_cast<real_t>(NX1) }, { 0.0, static_cast<real_t>(NX2) } },
      {}
    };
    const auto dt        = static_cast<real_t>(0.5);
    auto       electrons = make_species(1, -1.0f);
    auto       positrons = make_species(2, 1.0f);
    leave_previous(electrons, static_cast<prtldx_t>(0.2), static_cast<prtldx_t>(0.7));
    leave_previous(positrons, static_cast<prtldx_t>(0.8), static_cast<prtldx_t>(0.3));
    insert(electrons);
    insert(positrons);
    std::printf("precision %s, SHAPE_ORDER %d, dt %.2f\n",
                sizeof(real_t) == 8 ? "double" : "single", static_cast<int>(SHAPE_ORDER),
                static_cast<double>(dt));
    report("after kernel::InjectParticle:", electrons, positrons, metric, dt);
    set_previous_to_position(electrons);
    set_previous_to_position(positrons);
    report("previous position set to the position:", electrons, positrons, metric, dt);
  }
  Kokkos::finalize();
  return 0;
}
