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
 *                         single cell does not trigger it). One domain, 2D.
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
    const int    replenish, smooth_cells;

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

    // The correction: component 1 of the buffer, on the cells of the box, gets
    // the density of component 0 averaged over blocks of `smooth_cells` cells
    // along x1 and over the row and its two neighbours inside the box.
    void SmoothBox(Domain<S, M>& domain, const boundaries_t<real_t>& box) const {
      boundaries_t<bool> incl_ghosts;
      for (auto d = 0; d < M::Dim; ++d) {
        incl_ghosts.emplace_back(false, false);
      }
      const auto extent = domain.mesh.ExtentToRange(box, incl_ghosts);
      const auto i_min = extent[0].first, i_max = extent[0].second;
      const auto j_min = extent[1].first, j_max = extent[1].second;
      if (i_max <= i_min or j_max <= j_min) {
        return;
      }
      const ncells_t    ni = i_max - i_min, nj = j_max - j_min;
      const ncells_t    w    = static_cast<ncells_t>(smooth_cells);
      const ncells_t    nb   = (ni + w - 1) / w;
      auto              buff = domain.fields.buff;
      array_t<real_t**> block("replenish_block", nb, nj);
      Kokkos::parallel_for(
        "ReplenishBlock",
        CreateRangePolicy<Dim::_2D>({ 0, 0 }, { nb, nj }),
        Lambda(cellidx_t b, cellidx_t r) {
          const ncells_t lo = b * w;
          const ncells_t hi = (lo + w < ni) ? lo + w : ni;
          real_t         s { ZERO };
          for (auto k = lo; k < hi; ++k) {
            s += buff(i_min + k, j_min + r, 0);
          }
          block(b, r) = s / static_cast<real_t>(hi - lo);
        });
      Kokkos::parallel_for(
        "ReplenishSmooth",
        CreateRangePolicy<Dim::_2D>({ i_min, j_min }, { i_max, j_max }),
        Lambda(cellidx_t i, cellidx_t j) {
          const ncells_t b  = (i - i_min) / w;
          const ncells_t r  = j - j_min;
          const ncells_t r0 = (r > 0) ? r - 1 : 0;
          const ncells_t r1 = (r + 1 < nj) ? r + 1 : nj - 1;
          real_t         s { ZERO };
          for (auto q = r0; q <= r1; ++q) {
            s += block(b, q);
          }
          buff(i, j, 1) = s / static_cast<real_t>(r1 - r0 + 1);
        });
    }

    void CustomPostStep(timestep_t, simtime_t, Domain<S, M>& domain) {
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
      // ---- the correction: the region reaches the boundary, and the rule
      // ---- reads the smoothed density
      idx_t density_component = 0u;
      if (replenish == 1) {
        inj_box_up[1]   = { ymax - inj_ypad - 10 * dx, ymax };
        inj_box_down[1] = { ymin, ymin + inj_ypad + 10 * dx };
        SmoothBox(domain, inj_box_up);
        SmoothBox(domain, inj_box_down);
        density_component = 1u;
      }

      const auto replenish_sdist = arch::spatial_dist::ReplenishUniform<M, 3>(
        domain.mesh.metric,
        domain.fields.buff,
        density_component,
        ONE);
      // ---- from here unchanged again ----
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
