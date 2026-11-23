/**
 * The replenishing injector of `pgens/reconnection` on a uniform plasma without
 * a current layer: a diagnostic generator for upstream issue U17 (the stock
 * injector overfills its slab). Not used by any run of the shakedown.
 *
 * A uniform magnetic field along x1 and a uniform pair of species at the
 * density n0 and the temperature `setup.bg_temperature`. At every step the two
 * slabs of `pgens/reconnection` (ten cells thick, at `setup.inj_ypad` from the
 * two boundaries in x2) are replenished.
 *
 *   setup.flow_beta = 0   no electric field, no flow (periodic boundaries);
 *   setup.flow_beta > 0   a uniform electric field along x3 that makes the
 *                         plasma drift towards -x2 at that speed: the upper
 *                         slab then feeds a flow, as in the reconnection runs.
 *                         Boundaries in x2: MATCH and ABSORB; the fields are
 *                         matched to zero electric field at the upper boundary
 *                         (as `pgens/reconnection` does) and to the drift
 *                         field at the lower one, which drives the flow.
 *
 *   setup.replenish = 0   the block of `CustomPostStep` of `pgens/reconnection`
 *                         (lines 232 to 277 at 97d78619), unchanged: the
 *                         density of each cell (moment of order zero) is
 *                         compared with 0.9 of the target, in the slab only;
 *   setup.replenish = 1   the proposed correction, two changes to that block:
 *                         the replenished region runs from the inner edge of
 *                         the slab to the boundary (so that no cell is fed
 *                         from an empty region), and the rule sees the density
 *                         averaged over blocks of `setup.smooth_cells` cells
 *                         along x1 and three rows (so that the shot noise of a
 *                         single cell does not trigger it). 2D. The code is
 *                         `bhr::ReplenishCorrected` of
 *                         `pgens/reconnection_bh/bhr_common.hpp`, the one the
 *                         reconnection generator uses; it is valid on several
 *                         domains.
 *
 *   setup.replenish = 2   the corrected rule with a target that follows the
 *                         field strength (`follow_field` of `bhr_common.hpp`).
 *
 *   setup.profile_interval   `bh_profile.txt` every so many steps (0 = never,
 *                            the default; and after the first and the last
 *                            step): see `bhr_common.hpp`.
 */

#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#include "../reconnection_bh/bhr_common.hpp"

#include <Kokkos_Core.hpp>

#include <vector>

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(real_t bg_B, real_t flow_beta)
      : bg_B { bg_B }
      , flow_beta { flow_beta } {}

    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return bg_B;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return -flow_beta * bg_B;
    }

  private:
    const real_t bg_B, flow_beta;
  };

  // targets of the MATCH boundaries in x2 (used with setup.flow_beta > 0)
  template <Dimension D>
  struct BoundaryFieldsInX2 {
    BoundaryFieldsInX2(real_t bg_B, real_t flow_beta, real_t y_mid)
      : bg_B { bg_B }
      , flow_beta { flow_beta }
      , y_mid { y_mid } {}

    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return bg_B;
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
      return (x_Ph[1] > y_mid) ? ZERO : -flow_beta * bg_B;
    }

  private:
    const real_t bg_B, flow_beta, y_mid;
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    // compatibility traits for the problem generator
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions { ::traits::pgen::compatible_with<Dim::_2D> {} };

    const SimulationParams& params;
    Metadomain<S, M>&       metadomain;

    const real_t bg_B, bg_temperature, inj_ypad, flow_beta;
    const real_t ymin, ymax;
    const int    replenish, smooth_cells, profile_interval;
    bool         profile_started { false };

    InitFields<D> init_flds;

    PGen(const SimulationParams& p, Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , bg_B { params.template get<real_t>("setup.bg_B", 1.0) }
      , bg_temperature { params.template get<real_t>("setup.bg_temperature", 0.001) }
      , inj_ypad { params.template get<real_t>("setup.inj_ypad", (real_t)0.05) }
      , flow_beta { params.template get<real_t>("setup.flow_beta", (real_t)0.0) }
      , ymin { m.mesh().extent(in::x2).first }
      , ymax { m.mesh().extent(in::x2).second }
      , replenish { params.template get<int>("setup.replenish", 0) }
      , smooth_cells { params.template get<int>("setup.smooth_cells", 64) }
      , profile_interval { bhr::opt<int>(params, "setup.profile_interval", 0) }
      , init_flds { bg_B, flow_beta } {}

    auto MatchFieldsInX2(simtime_t) const -> BoundaryFieldsInX2<D> {
      return BoundaryFieldsInX2<D> { bg_B, flow_beta, HALF * (ymin + ymax) };
    }

    void InitPrtls(Domain<S, M>& local_domain) {
      // background, drifting with the flow
      const auto u_d = flow_beta / math::sqrt(ONE - SQR(flow_beta));
      const std::vector<real_t> drift { ZERO, -u_d, ZERO };
      arch::InjectUniformMaxwellian<S, M>(params,
                                          local_domain,
                                          ONE,
                                          bg_temperature,
                                          { 1, 2 },
                                          { drift, drift });
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      if (replenish == 1 or replenish == 2) {
        bhr::ReplenishCorrected(params,
                                metadomain,
                                domain,
                                1,
                                2,
                                bg_temperature,
                                ymin,
                                ymax,
                                inj_ypad,
                                smooth_cells,
                                replenish == 2,
                                bg_B,
                                bhr::opt<int>(params, "setup.smooth_rows", 1));
      } else {
        Stock(domain);
      }
      if (profile_interval > 0) {
        const auto dt = params.template get<real_t>("algorithms.timestep.dt");
        const auto max_steps = static_cast<timestep_t>(
          params.template get<simtime_t>("simulation.runtime") / dt);
        const auto steps_done = static_cast<long long>(step) + 1;
        if (steps_done == 1 or steps_done >= static_cast<long long>(max_steps) or
            steps_done % profile_interval == 0) {
          std::vector<std::pair<long, long>> regions;
          if (replenish == 1 or replenish == 2) {
            regions = bhr::Regions(metadomain, domain, ymin, ymax, inj_ypad);
          }
          bhr::WriteProfile(params,
                            metadomain,
                            domain,
                            1,
                            2,
                            ZERO,
                            ZERO,
                            regions,
                            steps_done,
                            static_cast<double>(time) + static_cast<double>(dt),
                            profile_started);
        }
      }
    }

    void Stock(Domain<S, M>& domain) {
      // ---- from here: `pgens/reconnection`, CustomPostStep, unchanged ----
      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        bg_temperature);

      const auto dx = domain.mesh.metric.template sqrt_h_<1, 1>({});

      boundaries_t<real_t> inj_box_up, inj_box_down;
      inj_box_up.push_back(Range::All);
      inj_box_down.push_back(Range::All);
      inj_box_up.push_back({ ymax - inj_ypad - 10 * dx, ymax - inj_ypad });
      inj_box_down.push_back({ ymin + inj_ypad, ymin + inj_ypad + 10 * dx });

      if constexpr (M::Dim == Dim::_3D) {
        inj_box_up.push_back(Range::All);
        inj_box_down.push_back(Range::All);
      }

      // compute density of species #1 and #2
      arch::ComputeMomentWithSpecies<S, M, FldsID::Rho, 3>(params,
                                                           domain,
                                                           { 1, 2 },
                                                           domain.fields.buff);

      const auto replenish_sdist = arch::spatial_dist::ReplenishUniform<M, 3>(
        domain.mesh.metric,
        domain.fields.buff,
        0u,
        ONE);
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
        params,
        domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        replenish_sdist,
        ONE,
        params.template get<bool>("particles.use_weights"),
        inj_box_up);
      arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
        params,
        domain,
        { 1, 2 },
        { energy_dist, energy_dist },
        replenish_sdist,
        ONE,
        params.template get<bool>("particles.use_weights"),
        inj_box_down);
    }
  };

} // namespace user

#endif
