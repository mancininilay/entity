/**
 * Reconnection with the Bethe-Heitler emission policy: the stock generator
 * `pgens/reconnection` with diagnostics added.
 *
 * The setup (fields, particles, boundaries, replenishment) is the code of
 * `pgens/reconnection`, unchanged. What is added reads the state after a
 * step and writes two text files in the run directory; it changes neither
 * the fields, nor the particles, nor the random streams:
 *
 *   - `bh_counters.txt`, every `setup.diag_interval` steps (and after the
 *     first and the last step): the counters of the policy
 *     (`arch::emission::bh::GetCounters()`: pairs, particle-steps above the
 *     tables, particle-steps above the probability bound, interactions
 *     refused by the loss bound, pairs dropped for capacity, absorbed photon
 *     energy), and for every species the number of particles, the sum of
 *     their weights, the sum of w K and the largest K, with
 *     K = m (gamma - 1) in units of m_e c^2;
 *
 *   - `bh_gauss.txt`, every `setup.gauss_interval` steps (0 = never): the
 *     residual of Gauss's law, r = div E - kappa F[rho], on the nodes of a
 *     window that leaves out `setup.gauss_xpad` and `setup.gauss_ypad`
 *     (physical units) at the boundaries in x1 and x2. rho is the charge of
 *     all species on the nodes with the first-order shape of the zig-zag
 *     deposit, F the binomial filter of the currents applied
 *     `algorithms.current_filters` times, div the node divergence of
 *     `ntt::kernel::ComputeDivergence_kernel`, kappa = q0 / (B0 V0). The
 *     deposit and the filter conserve this residual exactly away from the
 *     boundaries, where particles are absorbed or injected and the fields
 *     are matched. 2D, zig-zag deposit only. On several domains the charge
 *     is deposited by every rank on the nodes of the global grid and summed
 *     over the ranks, the filter is applied to that sum, and every rank
 *     evaluates the nodes whose cell it owns (the first ghost cells of E
 *     hold the neighbour's values of the same step); the window must then
 *     leave out the MATCH layers. On one domain the arithmetic and its
 *     order are those of the one-domain diagnostic.
 *
 *   - `bh_ranks.txt`, every `setup.rank_interval` steps (0 = never; and
 *     after the first and the last step): one line per rank with the fill of
 *     the particle arrays of every species on that rank (`npart()`, which
 *     the capacity `maxnpart` limits rank by rank) and the counters of the
 *     policy of that rank. With MPI the counters of the policy are those of
 *     the calling rank; `bh_counters.txt` holds their sums over the ranks.
 *
 * Columns are given in the first line of each file. The time is that of the
 * state written: the time at the end of the step.
 *
 *   - `bh_profile.txt`, every `setup.profile_interval` steps (0 = never):
 *     see `bhr_common.hpp`.
 *
 * Options of `[setup]` added for the resolved series. With none of them
 * given the generator does what `pgens/reconnection` does:
 *
 *   injector          0 (default): the stock replenishing block, unchanged;
 *                     1: the corrected rule of `bhr_common.hpp` (2D);
 *                     -1: no replenishment (tests only).
 *   smooth_cells      block length of the corrected rule along x1 (64).
 *   primary_species   the two species of the initial plasma and of the
 *                     replenishment ([1, 2]); with the lepton species of the
 *                     policy listed first in the input, [3, 4].
 *   seed              0 (default): the random pool as the engine built it;
 *                     otherwise the pool of every domain is rebuilt from this
 *                     number before its first use (another realisation).
 *   profile_interval, profile_xmax   `bh_profile.txt`.
 *   init_list         tests only: the initial plasma from one list that is
 *                     the same on every rank, so that runs on different
 *                     decompositions start from the same particles (2D).
 *   negative_control  tests only: with `emission = "custom"` on the emitter,
 *                     the policy with the write of the previous position of
 *                     the new leptons left out (the charge check must fail
 *                     when the lepton species are listed before the emitter).
 *   stale_prev        tests only: the previous positions of the unused slots
 *                     of the two lepton species are set to random points of
 *                     the domain, different for the two species.
 *   gauss_local       tests only: the charge of the Gauss-law diagnostic is
 *                     not summed over the ranks (a deliberate fault).
 *
 * The file compiles against a tree without the policy (the counters are then
 * zero).
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
#include "kernels/injectors.hpp"

#include "bhr_common.hpp"

#if __has_include("archetypes/emission/bethe_heitler.h")
  #include "archetypes/emission/bethe_heitler.h"
  #define BHR_HAS_POLICY 1
#endif

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace user {
  using namespace ntt;

  // helpers of the tests (the helpers of the diagnostics are in
  // `bhr_common.hpp`)
  namespace bhr {
    // counter-based generator: uniform in [0, 1) from (seed, particle, draw)
    inline auto uniform(std::uint64_t seed,
                        std::uint64_t particle,
                        std::uint64_t draw) -> double {
      std::uint64_t z = seed * 0x9E3779B97F4A7C15ULL +
                        particle * 0xBF58476D1CE4E5B9ULL +
                        draw * 0x94D049BB133111EBULL + 0x2545F4914F6CDD1DULL;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z = z ^ (z >> 31);
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z = z ^ (z >> 31);
      return static_cast<double>(z >> 11) * 0x1.0p-53;
    }

#if defined(BHR_HAS_POLICY)
    /**
     * The policy with the write of the previous position left out (negative
     * control of the charge test): the insertion of the built-in policies.
     */
    template <MetricClass M>
    struct NoPrevWrite {
      using base_t  = ::arch::emission::BetheHeitler<M>;
      using Payload = typename base_t::Payload;
      base_t base;

      NoPrevWrite(const base_t& base) : base { base } {}

      auto emitted_species_indices() const -> std::vector<spidx_t> {
        return base.emitted_species_indices();
      }

      auto numbers_injected() const -> std::vector<npart_t> {
        return base.numbers_injected();
      }

      Inline auto shouldEmit(const coord_t<M::PrtlDim>& x_Cd,
                             const coord_t<M::PrtlDim>& x_Ph,
                             const vec_t<Dim::_3D>&     u_Ph,
                             const vec_t<Dim::_3D>&     ep_Ph,
                             const vec_t<Dim::_3D>&     bp_Ph,
                             vec_t<Dim::_3D>&           delta_u_Ph,
                             Payload& payload) const -> Kokkos::pair<bool, bool> {
        return base.shouldEmit(x_Cd, x_Ph, u_Ph, ep_Ph, bp_Ph, delta_u_Ph, payload);
      }

      Inline void emit(const tuple_t<int, M::Dim>&      xi_Cd,
                       const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                       const vec_t<Dim::_3D>&,
                       real_t         weight,
                       real_t         phi,
                       const Payload& payload) const {
        const auto w = static_cast<real_t>(base.pair_weight) * weight;
        kernel::InjectParticle<M::Dim, M::CoordType, false>(
          base.electron_offset + payload.ticket,
          base.electron_i1,
          base.electron_i2,
          base.electron_i3,
          base.electron_dx1,
          base.electron_dx2,
          base.electron_dx3,
          base.electron_ux1,
          base.electron_ux2,
          base.electron_ux3,
          base.electron_phi,
          base.electron_weight,
          base.electron_tag,
          base.electron_pld_i,
          xi_Cd,
          dxi_Cd,
          { payload.u_electron[0], payload.u_electron[1], payload.u_electron[2] },
          w,
          phi);
        kernel::InjectParticle<M::Dim, M::CoordType, false>(
          base.positron_offset + payload.ticket,
          base.positron_i1,
          base.positron_i2,
          base.positron_i3,
          base.positron_dx1,
          base.positron_dx2,
          base.positron_dx3,
          base.positron_ux1,
          base.positron_ux2,
          base.positron_ux3,
          base.positron_phi,
          base.positron_weight,
          base.positron_tag,
          base.positron_pld_i,
          xi_Cd,
          dxi_Cd,
          { payload.u_positron[0], payload.u_positron[1], payload.u_positron[2] },
          w,
          phi);
      }
    };
#endif // BHR_HAS_POLICY
  } // namespace bhr

  template <Dimension D>
  struct CurrentLayer {
    CurrentLayer(real_t cs_width, real_t center_x, real_t cs_y)
      : cs_width { cs_width }
      , center_x { center_x }
      , cs_y { cs_y } {}

    Inline auto operator()(const coord_t<D>& x_Ph) const -> real_t {
      return ONE / SQR(math::cosh((x_Ph[1] - cs_y) / cs_width)) *
             (ONE - math::exp(-SQR((x_Ph[0] - center_x) / cs_width)));
    }

  private:
    const real_t cs_width, center_x, cs_y;
  };

  // field initializer
  template <Dimension D>
  struct InitFields {
    InitFields(real_t bg_B, real_t bg_Bguide, real_t cs_width, real_t cs_y)
      : bg_B { bg_B }
      , bg_Bguide { bg_Bguide }
      , cs_width { cs_width }
      , cs_y { cs_y } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return bg_B * (math::tanh((x_Ph[1] - cs_y) / cs_width));
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return bg_Bguide;
    }

  private:
    const real_t bg_B, bg_Bguide, cs_width, cs_y;
  };

  template <Dimension D>
  struct BoundaryFieldsInX1 {
    BoundaryFieldsInX1(real_t bg_B,
                       real_t bg_Bguide,
                       real_t beta_rec,
                       real_t cs_width,
                       real_t cs_x,
                       real_t cs_y)
      : bg_B { bg_B }
      , bg_Bguide { bg_Bguide }
      , beta_rec { beta_rec }
      , cs_width { cs_width }
      , cs_x { cs_x }
      , cs_y { cs_y } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return bg_B * (math::tanh((x_Ph[1] - cs_y) / cs_width));
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return beta_rec * bg_B * (math::tanh((x_Ph[0] - cs_x) / cs_width));
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return bg_Bguide;
    }

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
      return beta_rec * bg_Bguide * math::tanh((x_Ph[1] - cs_y) / cs_width);
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return -beta_rec * bg_B;
    }

  private:
    const real_t bg_B, bg_Bguide, beta_rec, cs_width, cs_x, cs_y;
  };

  template <Dimension D>
  struct BoundaryFieldsInX2 {
    BoundaryFieldsInX2(real_t bg_B, real_t bg_Bguide, real_t cs_width, real_t cs_y)
      : bg_B { bg_B }
      , bg_Bguide { bg_Bguide }
      , cs_width { cs_width }
      , cs_y { cs_y } {}

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return bg_B * (math::tanh((x_Ph[1] - cs_y) / cs_width));
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return bg_Bguide;
    }

    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return ZERO;
    }

  private:
    const real_t bg_B, bg_Bguide, cs_width, cs_y;
  };

  // constant particle density for particle boundaries
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
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_2D, Dim::_3D> {}
    };

    const SimulationParams& params;
    Metadomain<S, M>&       metadomain;

    const real_t    bg_B, bg_Bguide, bg_temperature, inj_ypad;
    const real_t    cs_width, cs_overdensity, cs_x, cs_y;
    const real_t    ymin, ymax;
    const simtime_t t_open;
    bool            bc_opened { false };

    InitFields<D> init_flds;

    // diagnostics (see the note at the top of the file)
    const int         diag_interval, gauss_interval, rank_interval;
    const real_t      gauss_xpad, gauss_ypad;
    bool              counters_started { false }, gauss_started { false };
    bool              ranks_started { false };
    array_t<double**> gauss_rho, gauss_tmp;

    // options of the resolved series (see the note at the top of the file)
    const spidx_t          sp_a, sp_b;
    const int              injector, smooth_cells, seed, profile_interval;
    const real_t           profile_xmax;
    const bool             init_list, negative_control, stale_prev, gauss_local;
    bool                   profile_started { false };
    std::set<unsigned int> seeded;

    PGen(const SimulationParams& p, Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , bg_B { params.template get<real_t>("setup.bg_B", 1.0) }
      , bg_Bguide { params.template get<real_t>("setup.bg_Bguide", 0.0) }
      , bg_temperature { params.template get<real_t>("setup.bg_temperature", 0.001) }
      , inj_ypad { params.template get<real_t>("setup.inj_ypad", (real_t)0.05) }
      , cs_width { params.template get<real_t>("setup.cs_width") }
      , cs_overdensity { params.template get<real_t>("setup.cs_overdensity") }
      , cs_x { INV_2 *
               (m.mesh().extent(in::x1).second + m.mesh().extent(in::x1).first) }
      , cs_y { INV_2 *
               (m.mesh().extent(in::x2).second + m.mesh().extent(in::x2).first) }
      , ymin { m.mesh().extent(in::x2).first }
      , ymax { m.mesh().extent(in::x2).second }
      , t_open { params.template get<simtime_t>(
          "setup.t_open",
          1.5 * HALF *
            (m.mesh().extent(in::x1).second - m.mesh().extent(in::x1).first)) }
      , init_flds { bg_B, bg_Bguide, cs_width, cs_y }
      , diag_interval { params.template get<int>("setup.diag_interval", 0) }
      , gauss_interval { params.template get<int>("setup.gauss_interval", 0) }
      , rank_interval { params.template get<int>("setup.rank_interval", 0) }
      , gauss_xpad { params.template get<real_t>("setup.gauss_xpad", (real_t)20.0) }
      , gauss_ypad { params.template get<real_t>(
          "setup.gauss_ypad",
          params.template get<real_t>("setup.inj_ypad", (real_t)0.05) + (real_t)10.0) }
      , sp_a { primary(params, 0) }
      , sp_b { primary(params, 1) }
      , injector { bhr::opt<int>(params, "setup.injector", 0) }
      , smooth_cells { bhr::opt<int>(params, "setup.smooth_cells", 64) }
      , seed { bhr::opt<int>(params, "setup.seed", 0) }
      , profile_interval { bhr::opt<int>(params, "setup.profile_interval", 0) }
      , profile_xmax { bhr::opt<real_t>(params, "setup.profile_xmax", ZERO) }
      , init_list { bhr::opt<bool>(params, "setup.init_list", false) }
      , negative_control { bhr::opt<bool>(params, "setup.negative_control", false) }
      , stale_prev { bhr::opt<bool>(params, "setup.stale_prev", false) }
      , gauss_local { bhr::opt<bool>(params, "setup.gauss_local", false) } {
      raise::ErrorIf(injector < -1 or injector > 1,
                     "reconnection_bh: setup.injector is -1, 0 or 1",
                     HERE);
      raise::ErrorIf(smooth_cells < 1,
                     "reconnection_bh: setup.smooth_cells must be positive",
                     HERE);
    }

    static auto primary(const SimulationParams& p, std::size_t k) -> spidx_t {
      const auto v = bhr::opt<std::vector<int>>(p,
                                                "setup.primary_species",
                                                std::vector<int> { 1, 2 });
      raise::ErrorIf(v.size() != 2 or v[0] < 1 or v[1] < 1 or v[0] == v[1],
                     "reconnection_bh: setup.primary_species needs two "
                     "different species indices",
                     HERE);
      return static_cast<spidx_t>(v[k]);
    }

    // another realisation: the pool of the domain rebuilt from `setup.seed`
    void ApplySeed(Domain<S, M>& domain) {
      if (seed == 0 or seeded.count(domain.index()) > 0) {
        return;
      }
      domain.random_pool() = random_number_pool_t(
        constant::RandomSeed + static_cast<std::uint64_t>(domain.index()) +
        0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(seed));
      seeded.insert(domain.index());
    }

    auto MatchFieldsInX1(simtime_t) const -> BoundaryFieldsInX1<D> {
      return BoundaryFieldsInX1<D> { bg_B,     bg_Bguide, (real_t)0.1,
                                     cs_width, cs_x,      cs_y };
    }

    auto MatchFieldsInX2(simtime_t) const -> BoundaryFieldsInX2<D> {
      return BoundaryFieldsInX2<D> { bg_B, bg_Bguide, cs_width, cs_y };
    }

    void InitPrtls(Domain<S, M>& local_domain) {
      ApplySeed(local_domain);
      if (init_list) {
        InitFromList(local_domain);
        if (stale_prev) {
          FillStalePrev(local_domain);
        }
        return;
      }
      // background
      arch::InjectUniformMaxwellian<S, M>(params,
                                          local_domain,
                                          ONE,
                                          bg_temperature,
                                          { sp_a, sp_b });

      const auto sigma = params.template get<real_t>("scales.sigma0");
      const auto c_omp = params.template get<real_t>("scales.skindepth0");
      const auto cs_drift_beta = math::sqrt(sigma) * c_omp /
                                 (cs_width * cs_overdensity);
      const auto cs_drift_gamma = ONE / math::sqrt(ONE - SQR(cs_drift_beta));
      const auto cs_drift_u     = cs_drift_beta * cs_drift_gamma;
      const auto cs_temperature = HALF * sigma / cs_overdensity;

      // current layer
      auto edist_cs = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        local_domain.random_pool(),
        cs_temperature);
      const auto sdist_cs = CurrentLayer<M::Dim>(cs_width, cs_x, cs_y);
      arch::InjectNonUniform<S, M, decltype(edist_cs), decltype(edist_cs), decltype(sdist_cs)>(
        params,
        local_domain,
        { sp_a, sp_b },
        { edist_cs, edist_cs },
        sdist_cs,
        cs_overdensity);
      if (stale_prev) {
        FillStalePrev(local_domain);
      }
    }

    // Tests only. The initial plasma from one list, the same on every rank
    // (a fixed sequence, cell by cell of the global mesh); each rank keeps
    // the particles of its own cells. Background at `bg_temperature` and a
    // layer with the profile and the overdensity of the stock setup, electron
    // and ion at the same place; the momenta are normal variates (a stand-in
    // for the stock distributions, not a copy of them).
    void InitFromList(Domain<S, M>& local_domain) {
      if constexpr (M::Dim == Dim::_2D) {
        const long   NX = static_cast<long>(metadomain.mesh().n_active(in::x1));
        const long   NY = static_cast<long>(metadomain.mesh().n_active(in::x2));
        const auto   gx1  = metadomain.mesh().extent(in::x1);
        const auto   gx2  = metadomain.mesh().extent(in::x2);
        const double x_lo = static_cast<double>(gx1.first);
        const double y_lo = static_cast<double>(gx2.first);
        const double dxg  = (static_cast<double>(gx1.second) - x_lo) / NX;
        const double dyg  = (static_cast<double>(gx2.second) - y_lo) / NY;
        const int    per_species = static_cast<int>(
          params.template get<real_t>("particles.ppc0") / 2);
        const double t_bg = static_cast<double>(bg_temperature);
        const double t_cs = 0.5 *
                            static_cast<double>(
                              params.template get<real_t>("scales.sigma0")) /
                            static_cast<double>(cs_overdensity);
        std::mt19937_64                            gen(20261122ULL);
        std::uniform_real_distribution<double>     uni(0.0, 1.0);
        std::normal_distribution<double>           nrm(0.0, 1.0);
        std::map<std::string, std::vector<real_t>> a, b;
        const auto add = [&](long I, long J, double s) {
          const double x = x_lo +
                           (static_cast<double>(I) + 0.001 + 0.998 * uni(gen)) * dxg;
          const double y = y_lo +
                           (static_cast<double>(J) + 0.001 + 0.998 * uni(gen)) * dyg;
          for (auto* m : { &a, &b }) {
            (*m)["x1"].push_back(static_cast<real_t>(x));
            (*m)["x2"].push_back(static_cast<real_t>(y));
            (*m)["ux1"].push_back(static_cast<real_t>(s * nrm(gen)));
            (*m)["ux2"].push_back(static_cast<real_t>(s * nrm(gen)));
            (*m)["ux3"].push_back(static_cast<real_t>(s * nrm(gen)));
          }
        };
        for (long J = 0; J < NY; ++J) {
          for (long I = 0; I < NX; ++I) {
            for (int k = 0; k < per_species; ++k) {
              add(I, J, std::sqrt(t_bg));
            }
            const double xc = x_lo + (static_cast<double>(I) + 0.5) * dxg;
            const double yc = y_lo + (static_cast<double>(J) + 0.5) * dyg;
            const double ch = std::cosh(
              (yc - static_cast<double>(cs_y)) / static_cast<double>(cs_width));
            const double ex = (xc - static_cast<double>(cs_x)) /
                              static_cast<double>(cs_width);
            const double want = per_species * static_cast<double>(cs_overdensity) *
                                (1.0 - std::exp(-ex * ex)) / (ch * ch);
            int n = static_cast<int>(std::floor(want));
            if (uni(gen) < want - n) {
              ++n;
            }
            for (int k = 0; k < n; ++k) {
              add(I, J, std::sqrt(t_cs));
            }
          }
        }
        arch::InjectGlobally<S, M>(metadomain, local_domain, sp_a, a);
        arch::InjectGlobally<S, M>(metadomain, local_domain, sp_b, b);
      } else {
        raise::Error("reconnection_bh: setup.init_list is 2D only", HERE);
      }
    }

    // Tests only. Previous positions of the unused slots of the two lepton
    // species of the policy set to random points of the domain, as dead
    // particles would leave them, different for the two species.
    void FillStalePrev(Domain<S, M>& domain) const {
      if constexpr (M::Dim == Dim::_2D) {
        for (const char* key :
             { "radiation.emission.bethe_heitler.electron_species",
               "radiation.emission.bethe_heitler.positron_species" }) {
          if (not params.contains(key)) {
            continue;
          }
          const auto species = params.template get<spidx_t>(key);
          auto&      sp      = domain.species[species - 1];
          const int  n1      = static_cast<int>(domain.mesh.n_active(in::x1));
          const int  n2      = static_cast<int>(domain.mesh.n_active(in::x2));
          auto       i1_h    = Kokkos::create_mirror_view(sp.i1_prev);
          auto       i2_h    = Kokkos::create_mirror_view(sp.i2_prev);
          auto       dx1_h   = Kokkos::create_mirror_view(sp.dx1_prev);
          auto       dx2_h   = Kokkos::create_mirror_view(sp.dx2_prev);
          Kokkos::deep_copy(i1_h, sp.i1_prev);
          Kokkos::deep_copy(i2_h, sp.i2_prev);
          Kokkos::deep_copy(dx1_h, sp.dx1_prev);
          Kokkos::deep_copy(dx2_h, sp.dx2_prev);
          const auto stream = static_cast<std::uint64_t>(20261122u) +
                              7919u * species + 104729u * domain.index();
          for (npart_t p = sp.npart(); p < sp.maxnpart(); ++p) {
            i1_h(p) = static_cast<int>(n1 * bhr::uniform(stream, p, 10));
            i2_h(p) = static_cast<int>(n2 * bhr::uniform(stream, p, 11));
            dx1_h(p) = static_cast<prtldx_t>(0.999 * bhr::uniform(stream, p, 12));
            dx2_h(p) = static_cast<prtldx_t>(0.999 * bhr::uniform(stream, p, 13));
          }
          Kokkos::deep_copy(sp.i1_prev, i1_h);
          Kokkos::deep_copy(sp.i2_prev, i2_h);
          Kokkos::deep_copy(sp.dx1_prev, dx1_h);
          Kokkos::deep_copy(sp.dx2_prev, dx2_h);
        }
      } else {
        raise::Error("reconnection_bh: setup.stale_prev is 2D only", HERE);
      }
    }

#if defined(BHR_HAS_POLICY)
    // Tests only (`emission = "custom"` on the emitter): the policy with the
    // write of the previous position of the new leptons left out.
    auto EmissionPolicy(simtime_t, spidx_t emitter, Domain<S, M>& domain) const {
      raise::ErrorIf(not negative_control,
                     "reconnection_bh: emission = custom needs "
                     "setup.negative_control = true",
                     HERE);
      const auto dt    = params.template get<real_t>("algorithms.timestep.dt");
      const auto e_idx = params.template get<spidx_t>(
        "radiation.emission.bethe_heitler.electron_species");
      const auto p_idx = params.template get<spidx_t>(
        "radiation.emission.bethe_heitler.positron_species");
      auto& tables = ::arch::emission::bh::GetTables(params, dt);
      return bhr::NoPrevWrite<M> {
        ::arch::emission::BetheHeitler<M> { domain.species[e_idx - 1],
                                           domain.species[p_idx - 1],
                                           e_idx, p_idx,
                                           domain.species[emitter - 1].mass(),
                                           dt, domain.index(),
                                           params, tables,
                                           domain.random_pool() }
      };
    }
#endif

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      ApplySeed(domain);
      // open boundaries if not yet opened at time = t_open
      if ((t_open > 0.0) and (not bc_opened) and (time > t_open)) {
        bc_opened = true;
        metadomain.setFldsBC(bc_in::Mx1, FldsBC::MATCH);
        metadomain.setPrtlBC(bc_in::Mx1, PrtlBC::ABSORB);
        metadomain.setFldsBC(bc_in::Px1, FldsBC::MATCH);
        metadomain.setPrtlBC(bc_in::Px1, PrtlBC::ABSORB);
      }

      if (injector == 0) {
        // the block of `pgens/reconnection`, unchanged but for the species
        // indices, which are 1 and 2 by default
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
                                                             { sp_a, sp_b },
                                                             domain.fields.buff);

        const auto replenish_sdist = arch::spatial_dist::ReplenishUniform<M, 3>(
          domain.mesh.metric,
          domain.fields.buff,
          0u,
          ONE);
        arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
          params,
          domain,
          { sp_a, sp_b },
          { energy_dist, energy_dist },
          replenish_sdist,
          ONE,
          params.template get<bool>("particles.use_weights"),
          inj_box_up);
        arch::InjectNonUniform<S, M, decltype(energy_dist), decltype(energy_dist), decltype(replenish_sdist)>(
          params,
          domain,
          { sp_a, sp_b },
          { energy_dist, energy_dist },
          replenish_sdist,
          ONE,
          params.template get<bool>("particles.use_weights"),
          inj_box_down);
      } else if (injector == 1) {
        if constexpr (M::Dim == Dim::_2D) {
          bhr::ReplenishCorrected(params,
                                  metadomain,
                                  domain,
                                  sp_a,
                                  sp_b,
                                  bg_temperature,
                                  ymin,
                                  ymax,
                                  inj_ypad,
                                  smooth_cells);
        } else {
          raise::Error("reconnection_bh: setup.injector = 1 is 2D only", HERE);
        }
      }

      // diagnostics: read-only
      Diagnostics(step, time, domain);
    }

    // ---------------------------------------------------------------------- //
    // Diagnostics. `step` and `time` are those of the step just taken: the
    // state is that after `step + 1` steps, at the time `time + dt`.
    // ---------------------------------------------------------------------- //
    void Diagnostics(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      if (diag_interval <= 0 and gauss_interval <= 0 and rank_interval <= 0 and
          profile_interval <= 0) {
        return;
      }
      const auto dt = params.template get<real_t>("algorithms.timestep.dt");
      // the number of steps of the run, as the engine computes it
      const auto max_steps = static_cast<timestep_t>(
        params.template get<simtime_t>("simulation.runtime") / dt);
      const auto steps_done = static_cast<long long>(step) + 1;
      const auto time_now = static_cast<double>(time) + static_cast<double>(dt);
      const bool edge     = (steps_done == 1) or
                        (steps_done >= static_cast<long long>(max_steps));
      if (diag_interval > 0 and (edge or steps_done % diag_interval == 0)) {
        WriteCounters(steps_done, time_now, domain);
      }
      if (gauss_interval > 0 and (edge or steps_done % gauss_interval == 0)) {
        WriteGauss(steps_done, time_now, domain);
      }
      if (rank_interval > 0 and (edge or steps_done % rank_interval == 0)) {
        WriteRanks(steps_done, time_now, domain);
      }
      if (profile_interval > 0 and (edge or steps_done % profile_interval == 0)) {
        if constexpr (M::Dim == Dim::_2D) {
          std::vector<std::pair<long, long>> regions;
          if (injector == 1) {
            regions = bhr::Regions(metadomain, domain, ymin, ymax, inj_ypad);
          }
          bhr::WriteProfile(params,
                            metadomain,
                            domain,
                            sp_a,
                            sp_b,
                            cs_x,
                            profile_xmax,
                            regions,
                            steps_done,
                            time_now,
                            profile_started);
        } else {
          raise::Error("reconnection_bh: bh_profile.txt is 2D only", HERE);
        }
      }
    }

    void WriteRanks(long long steps_done, double time_now, Domain<S, M>& domain) {
      const auto          nspec = domain.species.size();
      // fill and capacity of every species, then pairs and capacity_dropped
      std::vector<double> mine(2 * nspec + 2, 0.0);
      for (std::size_t s = 0; s < nspec; ++s) {
        mine[2 * s + 0] = static_cast<double>(domain.species[s].npart());
        mine[2 * s + 1] = static_cast<double>(domain.species[s].maxnpart());
      }
#if defined(BHR_HAS_POLICY)
      {
        const auto c        = ::arch::emission::bh::GetCounters();
        mine[2 * nspec + 0] = static_cast<double>(c.pairs);
        mine[2 * nspec + 1] = static_cast<double>(c.capacity_dropped);
      }
#endif
      const auto every = bhr::gather(mine);
      if (not bhr::is_root()) {
        return;
      }
      std::FILE* f = std::fopen("bh_ranks.txt", ranks_started ? "a" : "w");
      raise::ErrorIf(f == nullptr, "reconnection_bh: cannot open bh_ranks.txt", HERE);
      if (not ranks_started) {
        std::fprintf(f, "# steps time rank");
        for (std::size_t s = 0; s < nspec; ++s) {
          std::fprintf(f, " npart_%zu maxnpart_%zu", s + 1, s + 1);
        }
        std::fprintf(f, " pairs capacity_dropped\n");
        ranks_started = true;
      }
      const std::size_t nr = every.size() / mine.size();
      for (std::size_t r = 0; r < nr; ++r) {
        std::fprintf(f, "%lld %.17g %zu", steps_done, time_now, r);
        for (std::size_t k = 0; k < mine.size(); ++k) {
          std::fprintf(f, " %.0f", every[r * mine.size() + k]);
        }
        std::fprintf(f, "\n");
      }
      std::fclose(f);
    }

    void WriteCounters(long long steps_done, double time_now, Domain<S, M>& domain) {
      const auto          nspec = domain.species.size();
      // per species: particles, sum w, sum w K; and the largest K
      std::vector<double> sums(3 * nspec, 0.0), kmax(nspec, 0.0);
      for (std::size_t s = 0; s < nspec; ++s) {
        const auto& sp = domain.species[s];
        if (sp.npart() == 0) {
          continue;
        }
        const auto m   = static_cast<double>(sp.mass());
        const auto ux1 = sp.ux1, ux2 = sp.ux2, ux3 = sp.ux3;
        const auto weight = sp.weight;
        const auto tag    = sp.tag;
        double     s_n = 0.0, s_w = 0.0, s_k = 0.0, s_max = 0.0;
        Kokkos::parallel_reduce(
          "BHRSums",
          sp.rangeActiveParticles(),
          Lambda(prtlidx_t p, double& l_n, double& l_w, double& l_k, double& l_max) {
            if (tag(p) != ParticleTag::alive) {
              return;
            }
            const double a   = static_cast<double>(ux1(p));
            const double b   = static_cast<double>(ux2(p));
            const double c   = static_cast<double>(ux3(p));
            const double u2  = a * a + b * b + c * c;
            const double K   = m * u2 / (math::sqrt(1.0 + u2) + 1.0);
            const double w   = static_cast<double>(weight(p));
            l_n             += 1.0;
            l_w             += w;
            l_k             += w * K;
            if (K > l_max) {
              l_max = K;
            }
          },
          s_n,
          s_w,
          s_k,
          Kokkos::Max<double>(s_max));
        sums[3 * s + 0] = s_n;
        sums[3 * s + 1] = s_w;
        sums[3 * s + 2] = s_k;
        kmax[s]         = s_max;
      }
      // pairs, above_tables, probability_exceeded, loss_bound_skipped,
      // capacity_dropped, absorbed_energy
      std::vector<double> cnt(6, 0.0);
#if defined(BHR_HAS_POLICY)
      {
        const auto c = ::arch::emission::bh::GetCounters();
        cnt[0]       = static_cast<double>(c.pairs);
        cnt[1]       = static_cast<double>(c.above_tables);
        cnt[2]       = static_cast<double>(c.probability_exceeded);
        cnt[3]       = static_cast<double>(c.loss_bound_skipped);
        cnt[4]       = static_cast<double>(c.capacity_dropped);
        cnt[5]       = c.absorbed_energy;
      }
#endif
      bhr::reduce(sums, false);
      bhr::reduce(kmax, true);
      bhr::reduce(cnt, false);
      if (not bhr::is_root()) {
        return;
      }
      std::FILE* f = std::fopen("bh_counters.txt", counters_started ? "a" : "w");
      raise::ErrorIf(f == nullptr,
                     "reconnection_bh: cannot open bh_counters.txt",
                     HERE);
      if (not counters_started) {
        std::fprintf(f,
                     "# steps time pairs above_tables probability_exceeded "
                     "loss_bound_skipped capacity_dropped absorbed_energy");
        for (std::size_t s = 0; s < nspec; ++s) {
          std::fprintf(f, " n_%zu w_%zu K_%zu Kmax_%zu", s + 1, s + 1, s + 1, s + 1);
        }
        std::fprintf(f, "\n");
        counters_started = true;
      }
      std::fprintf(f,
                   "%lld %.17g %.0f %.0f %.0f %.0f %.0f %.17g",
                   steps_done,
                   time_now,
                   cnt[0],
                   cnt[1],
                   cnt[2],
                   cnt[3],
                   cnt[4],
                   cnt[5]);
      for (std::size_t s = 0; s < nspec; ++s) {
        std::fprintf(f,
                     " %.0f %.17g %.17g %.9g",
                     sums[3 * s + 0],
                     sums[3 * s + 1],
                     sums[3 * s + 2],
                     kmax[s]);
      }
      std::fprintf(f, "\n");
      std::fclose(f);
    }

    void WriteGauss(long long steps_done, double time_now, Domain<S, M>& domain) {
      if constexpr (M::Dim == Dim::_2D) {
        // SHAPE_ORDER is 0 for the zig-zag deposit (global.h)
#if SHAPE_ORDER != 0
        raise::Error("reconnection_bh: the Gauss-law diagnostic assumes the "
                     "zig-zag deposit",
                     HERE);
#endif
        const auto res = params.template get<std::vector<ncells_t>>(
          "grid.resolution");
        // the global grid (nx, ny), and the cells of this domain in it
        const int  nx  = static_cast<int>(res[0]);
        const int  ny  = static_cast<int>(res[1]);
        const int  lx  = static_cast<int>(domain.mesh.n_active(in::x1));
        const int  ly  = static_cast<int>(domain.mesh.n_active(in::x2));
        const auto off = domain.offset_ncells();
        const int  ox = static_cast<int>(off[0]), oy = static_cast<int>(off[1]);
        const bool one_domain = (lx == nx and ly == ny);
        if (not one_domain and params.contains("grid.boundaries.match.ds")) {
          // the ghost cells of E are the neighbour's values of this step only
          // outside the MATCH layers (applied after the exchange)
          const auto ds = params.template get<boundaries_t<real_t>>(
            "grid.boundaries.match.ds");
          raise::ErrorIf(
            ds.size() < 2 or gauss_xpad < std::max(ds[0].first, ds[0].second) or
              gauss_ypad < std::max(ds[1].first, ds[1].second),
            "reconnection_bh: on several domains the window of the Gauss-law "
            "diagnostic must leave out the MATCH layers (gauss_xpad, "
            "gauss_ypad >= grid.boundaries.match.ds)",
            HERE);
        }
        // nodes (i, j) of the global grid, i = 0 .. nx - 1 (wrapped in x1),
        // j = 0 .. ny
        if (gauss_rho.extent(0) == 0) {
          gauss_rho = array_t<double**> { "bhr_rho",
                                          static_cast<std::size_t>(nx),
                                          static_cast<std::size_t>(ny + 1) };
          gauss_tmp = array_t<double**> { "bhr_tmp",
                                          static_cast<std::size_t>(nx),
                                          static_cast<std::size_t>(ny + 1) };
        }
        auto rho = gauss_rho;
        auto tmp = gauss_tmp;
        Kokkos::deep_copy(rho, 0.0);
        for (const auto& sp : domain.species) {
          const double q = static_cast<double>(sp.charge());
          if (q == 0.0 or sp.npart() == 0) {
            continue;
          }
          const auto i1 = sp.i1, i2 = sp.i2;
          const auto dx1 = sp.dx1, dx2 = sp.dx2;
          const auto weight = sp.weight;
          const auto tag    = sp.tag;
          Kokkos::parallel_for(
            "BHRCharge",
            sp.rangeActiveParticles(),
            Lambda(prtlidx_t p) {
              if (tag(p) != ParticleTag::alive) {
                return;
              }
              const int li = i1(p), lj = i2(p);
              if (li < 0 or li >= lx or lj < 0 or lj >= ly) {
                return;
              }
              const int    i = ox + li, j = oy + lj;
              const double fx = static_cast<double>(dx1(p));
              const double fy = static_cast<double>(dx2(p));
              const double w  = q * static_cast<double>(weight(p));
              const int    ip = (i + 1 == nx) ? 0 : i + 1;
              Kokkos::atomic_add(&rho(i, j), w * (1.0 - fx) * (1.0 - fy));
              Kokkos::atomic_add(&rho(ip, j), w * fx * (1.0 - fy));
              Kokkos::atomic_add(&rho(i, j + 1), w * (1.0 - fx) * fy);
              Kokkos::atomic_add(&rho(ip, j + 1), w * fx * fy);
            });
        }
        // a node on a rank boundary gets the charge of the particles on both
        // sides through the sum over the ranks
        if (not one_domain and not gauss_local) {
          bhr::reduce_view(rho);
        }
        // the filter of the currents: (1/4, 1/2, 1/4) in each direction per
        // pass; rows beyond the ends in x2 count as empty and x1 is wrapped,
        // which is the filter of the engine away from the boundaries
        const auto nfilter = params.template get<unsigned short>(
          "algorithms.current_filters");
        for (unsigned short pass = 0; pass < nfilter; ++pass) {
          Kokkos::parallel_for(
            "BHRFilter",
            CreateRangePolicy<Dim::_2D>(
              { static_cast<ncells_t>(0), static_cast<ncells_t>(0) },
              { static_cast<ncells_t>(nx), static_cast<ncells_t>(ny + 1) }),
            Lambda(cellidx_t ii, cellidx_t jj) {
              const int    i = static_cast<int>(ii), j = static_cast<int>(jj);
              const int    im  = (i == 0) ? nx - 1 : i - 1;
              const int    ip  = (i == nx - 1) ? 0 : i + 1;
              const double mid = 0.25 * rho(im, j) + 0.5 * rho(i, j) +
                                 0.25 * rho(ip, j);
              double lo = 0.0, hi = 0.0;
              if (j > 0) {
                lo = 0.25 * rho(im, j - 1) + 0.5 * rho(i, j - 1) +
                     0.25 * rho(ip, j - 1);
              }
              if (j < ny) {
                hi = 0.25 * rho(im, j + 1) + 0.5 * rho(i, j + 1) +
                     0.25 * rho(ip, j + 1);
              }
              tmp(i, j) = 0.25 * lo + 0.5 * mid + 0.25 * hi;
            });
          std::swap(rho, tmp);
        }
        // the window
        const auto   x1 = metadomain.mesh().extent(in::x1);
        const auto   x2 = metadomain.mesh().extent(in::x2);
        const double dx = static_cast<double>(x1.second - x1.first) / nx;
        const double dy = static_cast<double>(x2.second - x2.first) / ny;
        const int px = std::max(1, static_cast<int>(std::ceil(gauss_xpad / dx)));
        const int py = std::max(1, static_cast<int>(std::ceil(gauss_ypad / dy)));
        raise::ErrorIf(2 * px >= nx or 2 * py >= ny,
                       "reconnection_bh: the window of the Gauss-law "
                       "diagnostic is empty",
                       HERE);
        const double kappa =
          static_cast<double>(params.template get<real_t>("scales.q0")) /
          (static_cast<double>(params.template get<real_t>("scales.B0")) *
           static_cast<double>(params.template get<real_t>("scales.V0")));
        const auto EM   = domain.fields.em;
        const int  ng   = static_cast<int>(N_GHOSTS);
        double     s_r2 = 0.0, s_q2 = 0.0, s_d2 = 0.0;
        double     m_r = 0.0, m_q = 0.0, m_d = 0.0;
        // the nodes of the window whose cell this domain owns (local indices);
        // on one domain: px <= i < nx - px, py <= j < ny + 1 - py
        const int i_lo = std::max(px - ox, 0), i_hi = std::min(nx - px - ox, lx);
        const int j_lo = std::max(py - oy, 0);
        const int j_hi = std::min(ny + 1 - py - oy, ly);
        if (i_hi > i_lo and j_hi > j_lo) {
          Kokkos::parallel_reduce(
            "BHRGauss",
            CreateRangePolicy<Dim::_2D>(
              { static_cast<ncells_t>(i_lo), static_cast<ncells_t>(j_lo) },
              { static_cast<ncells_t>(i_hi), static_cast<ncells_t>(j_hi) }),
            Lambda(cellidx_t ii,
                   cellidx_t jj,
                   double&   l_r2,
                   double&   l_q2,
                   double&   l_d2,
                   double&   l_r,
                   double&   l_q,
                   double&   l_d) {
              const int    i = static_cast<int>(ii), j = static_cast<int>(jj);
              const double div =
                (static_cast<double>(EM(i + ng, j + ng, em::ex1)) -
                 static_cast<double>(EM(i - 1 + ng, j + ng, em::ex1))) +
                (static_cast<double>(EM(i + ng, j + ng, em::ex2)) -
                 static_cast<double>(EM(i + ng, j - 1 + ng, em::ex2)));
              const double q  = kappa * rho(ox + i, oy + j);
              const double r  = div - q;
              l_r2           += r * r;
              l_q2           += q * q;
              l_d2           += div * div;
              if (math::abs(r) > l_r) {
                l_r = math::abs(r);
              }
              if (math::abs(q) > l_q) {
                l_q = math::abs(q);
              }
              if (math::abs(div) > l_d) {
                l_d = math::abs(div);
              }
            },
            s_r2,
            s_q2,
            s_d2,
            Kokkos::Max<double>(m_r),
            Kokkos::Max<double>(m_q),
            Kokkos::Max<double>(m_d));
        }
        if (not one_domain) {
          std::vector<double> sums { s_r2, s_q2, s_d2 }, maxs { m_r, m_q, m_d };
          bhr::reduce(sums, false);
          bhr::reduce(maxs, true);
          s_r2 = sums[0];
          s_q2 = sums[1];
          s_d2 = sums[2];
          m_r  = maxs[0];
          m_q  = maxs[1];
          m_d  = maxs[2];
          if (not bhr::is_root()) {
            return;
          }
        }
        const double nodes = static_cast<double>(nx - 2 * px) *
                             static_cast<double>(ny + 1 - 2 * py);
        std::FILE* f = std::fopen("bh_gauss.txt", gauss_started ? "a" : "w");
        raise::ErrorIf(f == nullptr,
                       "reconnection_bh: cannot open bh_gauss.txt",
                       HERE);
        if (not gauss_started) {
          std::fprintf(f,
                       "# window: nodes %d <= i < %d, %d <= j < %d of %d x %d "
                       "cells; kappa %.17g; filter passes %d\n",
                       px,
                       nx - px,
                       py,
                       ny + 1 - py,
                       nx,
                       ny,
                       kappa,
                       static_cast<int>(nfilter));
          std::fprintf(f,
                       "# steps time nodes max_residual rms_residual "
                       "max_charge rms_charge max_divE rms_divE\n");
          gauss_started = true;
        }
        std::fprintf(f,
                     "%lld %.17g %.0f %.9e %.9e %.9e %.9e %.9e %.9e\n",
                     steps_done,
                     time_now,
                     nodes,
                     m_r,
                     std::sqrt(s_r2 / nodes),
                     m_q,
                     std::sqrt(s_q2 / nodes),
                     m_d,
                     std::sqrt(s_d2 / nodes));
        std::fclose(f);
      } else {
        raise::Error("reconnection_bh: the Gauss-law diagnostic is 2D only", HERE);
      }
    }
  };

} // namespace user

#endif
