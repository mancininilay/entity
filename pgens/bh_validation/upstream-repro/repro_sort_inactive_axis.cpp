// Reproduction on Entity 1.6.0rc (861b19a7), no Bethe-Heitler code:
// Particles::SortSpatially on one live particle in D dimensions.
//
//   repro_sort_inactive_axis <1|2|3>
//
// sort::PositionToTileIndex (src/global/utils/sorting.h) reads i2(p) and
// i3(p) (and their previous values) for every D, before the `if constexpr`
// on the dimension; in 1D and 2D these views have no elements. Built with
// -DKOKKOS_ENABLE_DEBUG_BOUNDS_CHECK the read aborts.
#include "enums.h"
#include "global.h"

#include "framework/containers/particles.h"
#include "framework/domain/grid.h"

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>

template <Dimension D>
void sort_one(const ntt::Grid<D>& grid) {
  auto prtls = ntt::Particles<D, ntt::Coord::Cartesian>(1,
                                                        "test",
                                                        1.0f,
                                                        1.0f,
                                                        4u,
                                                        0u,
                                                        0u,
                                                        ntt::ParticlePusher::BORIS,
                                                        false,
                                                        ntt::RadiativeDrag::NONE,
                                                        ntt::EmissionType::NONE,
                                                        0u,
                                                        0u);
  // one live particle in cell 1 of every axis, not moved since the last step
  Kokkos::deep_copy(prtls.tag, static_cast<short>(ntt::ParticleTag::alive));
  Kokkos::deep_copy(prtls.i1, 1);
  Kokkos::deep_copy(prtls.i1_prev, 1);
  if constexpr (D == Dim::_2D or D == Dim::_3D) {
    Kokkos::deep_copy(prtls.i2, 1);
    Kokkos::deep_copy(prtls.i2_prev, 1);
  }
  if constexpr (D == Dim::_3D) {
    Kokkos::deep_copy(prtls.i3, 1);
    Kokkos::deep_copy(prtls.i3_prev, 1);
  }
  prtls.set_npart(1u);
  std::cout << "extents: i1 " << prtls.i1.extent(0) << ", i2 "
            << prtls.i2.extent(0) << ", i3 " << prtls.i3.extent(0) << std::endl;
  prtls.SortSpatially(grid);
  std::cout << "SortSpatially completed" << std::endl;
}

auto main(int argc, char* argv[]) -> int {
  const int d = argc > 1 ? std::atoi(argv[1]) : 1;
  ntt::GlobalInitialize(argc, argv);
  if (d == 1) {
    sort_one(ntt::Grid<Dim::_1D> { { 16u }, { { 0.0, 16.0 } } });
  } else if (d == 2) {
    sort_one(ntt::Grid<Dim::_2D> {
      {          16u,           16u },
      { { 0.0, 16.0 }, { 0.0, 16.0 } }
    });
  } else {
    sort_one(ntt::Grid<Dim::_3D> {
      {          16u,           16u,           16u },
      { { 0.0, 16.0 }, { 0.0, 16.0 }, { 0.0, 16.0 } }
    });
  }
  ntt::GlobalFinalize();
  return 0;
}
