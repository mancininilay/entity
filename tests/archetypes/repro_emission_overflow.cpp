// Reproduction of an upstream issue (not a test of the Bethe-Heitler policy):
// the built-in synchrotron emission policy inserts the photon at
// `offset + index` without comparing with the capacity of the photon species.
// 200 emitters in a uniform magnetic field all emit in one step into a photon
// species of capacity 8.
//   - built with -DKOKKOS_ENABLE_DEBUG_BOUNDS_CHECK: Kokkos aborts on the
//     out-of-bounds access inside the pusher kernel;
//   - built without: the kernel writes past the end of the photon arrays;
//     the number of photons reported exceeds the capacity (printed), and
//     Particles::set_npart would raise only afterwards.
#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/param_container.h"

#include "metrics/minkowski.h"

#include "archetypes/emission/synchrotron.h"
#include "framework/containers/particles.h"
#include "kernels/pushers/context.h"
#include "kernels/pushers/sr.hpp"
#include "kernels/pushers/sr_policies.h"

#include <Kokkos_Core.hpp>

#include <iostream>

using namespace ntt;

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);
  int rc = 0;
  try {
    using metric_t  = metric::Minkowski<Dim::_2D>;
    using species_t = Particles<Dim::_2D, Coord::Cartesian>;
    const metric_t metric { { 16u, 16u }, { { 0.0, 16.0 }, { 0.0, 16.0 } }, {} };
    const npart_t  n_emitters = 200u, capacity = 8u;
    species_t      emitters(1, "e-", 1.0f, -1.0f, n_emitters, 0u, 0u, ParticlePusher::BORIS,
                            false, RadiativeDrag::NONE, EmissionType::SYNCHROTRON, 0u, 0u);
    species_t      photons(2, "ph", 0.0f, 0.0f, capacity, 0u, 0u, ParticlePusher::PHOTON,
                           false, RadiativeDrag::NONE, EmissionType::NONE, 0u, 0u);
    {
      // views by value: the container itself cannot be captured
      auto i1 = emitters.i1, i2 = emitters.i2;
      auto dx1 = emitters.dx1, dx2 = emitters.dx2;
      auto ux1 = emitters.ux1, ux2 = emitters.ux2, ux3 = emitters.ux3;
      auto weight = emitters.weight;
      auto tag    = emitters.tag;
      Kokkos::parallel_for(
        "Fill",
        n_emitters,
        KOKKOS_LAMBDA(const npart_t p) {
          i1(p)     = static_cast<int>(p % 16u);
          i2(p)     = static_cast<int>((p / 16u) % 16u);
          dx1(p)    = static_cast<prtldx_t>(0.5);
          dx2(p)    = static_cast<prtldx_t>(0.5);
          ux1(p)    = static_cast<real_t>(10.0);
          ux2(p)    = ZERO;
          ux3(p)    = ZERO;
          weight(p) = ONE;
          tag(p)    = ParticleTag::alive;
        });
    }
    emitters.set_npart(n_emitters);
    // uniform B along x3
    ndfield_t<Dim::_2D, 6> EB { "EB", 16u + 2u * N_GHOSTS, 16u + 2u * N_GHOSTS };
    Kokkos::parallel_for(
      "B",
      CreateRangePolicy<Dim::_2D>({ 0, 0 }, { 16u + 2u * N_GHOSTS, 16u + 2u * N_GHOSTS }),
      KOKKOS_LAMBDA(const int i, const int j) { EB(i, j, em::bx3) = ONE; });

    prm::Parameters params;
    params.set("radiation.emission.synchrotron.photon_weight", ONE);
    params.set("radiation.emission.synchrotron.photon_energy_min", ZERO);
    params.set("radiation.emission.synchrotron.nominal_probability",
               static_cast<real_t>(1.0e6));
    params.set("radiation.emission.synchrotron.nominal_photon_energy",
               static_cast<real_t>(1.0e-6));
    random_number_pool_t random_pool { 1 };
    using emission_t = arch::emission::Synchrotron<metric_t>;
    const emission_t emission { photons, 2, emitters.mass(), emitters.charge(),
                                RadiativeDrag::NONE, 0u, params, random_pool };
    const auto policy = kernel::sr::PusherPolicy<metric_t, emission_t> { emission };
    const auto bc     = kernel::sr::PusherBoundaries<Dim::_2D> {
      { { PrtlBC::PERIODIC, PrtlBC::PERIODIC }, { PrtlBC::PERIODIC, PrtlBC::PERIODIC } }
    };
    const auto dt = static_cast<real_t>(0.01);
    Kokkos::parallel_for(
      "ParticlePusher",
      emitters.rangeActiveParticles(),
      kernel::sr::Pusher_kernel<metric_t, decltype(policy)>(
        { emitters.index(), emitters.pusher(), emitters.radiative_drag_flags(),
          emitters.mass(), emitters.charge(), static_cast<simtime_t>(0), dt, ONE, 16, 16, 1 },
        bc,
        emitters,
        EB,
        metric,
        policy));
    const auto n = emission.numbers_injected()[0];
    std::cout << "photons inserted by the kernel: " << n << ", capacity of the photon species: "
              << capacity << (n > capacity ? "  -> wrote past the end of the arrays" : "")
              << std::endl;
    rc = n > capacity ? 3 : 0;
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    rc = 1;
  }
  GlobalFinalize();
  return rc;
}
