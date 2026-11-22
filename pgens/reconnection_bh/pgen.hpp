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
 *     are matched. 2D, one domain, zig-zag deposit only.
 *
 * Columns are given in the first line of each file. The time is that of the
 * state written: the time at the end of the step.
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

#if __has_include("archetypes/emission/bethe_heitler.h")
  #include "archetypes/emission/bethe_heitler.h"
  #define BHR_HAS_POLICY 1
#endif

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace user {
  using namespace ntt;

  // helpers of the diagnostics (see the note at the top of the file)
  namespace bhr {
    inline auto is_root() -> bool {
#if defined(MPI_ENABLED)
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      return rank == 0;
#else
      return true;
#endif
    }

    inline void reduce(std::vector<double>& v, bool maximum) {
#if defined(MPI_ENABLED)
      std::vector<double> out(v.size(), 0.0);
      MPI_Allreduce(v.data(),
                    out.data(),
                    static_cast<int>(v.size()),
                    MPI_DOUBLE,
                    maximum ? MPI_MAX : MPI_SUM,
                    MPI_COMM_WORLD);
      v = out;
#else
      (void)v;
      (void)maximum;
#endif
    }
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
    const int         diag_interval, gauss_interval;
    const real_t      gauss_xpad, gauss_ypad;
    bool              counters_started { false }, gauss_started { false };
    array_t<double**> gauss_rho, gauss_tmp;

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
      , gauss_xpad { params.template get<real_t>("setup.gauss_xpad", (real_t)20.0) }
      , gauss_ypad { params.template get<real_t>(
          "setup.gauss_ypad",
          params.template get<real_t>("setup.inj_ypad", (real_t)0.05) +
            (real_t)10.0) } {}

    auto MatchFieldsInX1(simtime_t) const -> BoundaryFieldsInX1<D> {
      return BoundaryFieldsInX1<D> { bg_B,     bg_Bguide, (real_t)0.1,
                                     cs_width, cs_x,      cs_y };
    }

    auto MatchFieldsInX2(simtime_t) const -> BoundaryFieldsInX2<D> {
      return BoundaryFieldsInX2<D> { bg_B, bg_Bguide, cs_width, cs_y };
    }

    void InitPrtls(Domain<S, M>& local_domain) {
      // background
      arch::InjectUniformMaxwellian<S, M>(params,
                                          local_domain,
                                          ONE,
                                          bg_temperature,
                                          { 1, 2 });

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
        { 1, 2 },
        { edist_cs, edist_cs },
        sdist_cs,
        cs_overdensity);
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      // open boundaries if not yet opened at time = t_open
      if ((t_open > 0.0) and (not bc_opened) and (time > t_open)) {
        bc_opened = true;
        metadomain.setFldsBC(bc_in::Mx1, FldsBC::MATCH);
        metadomain.setPrtlBC(bc_in::Mx1, PrtlBC::ABSORB);
        metadomain.setFldsBC(bc_in::Px1, FldsBC::MATCH);
        metadomain.setPrtlBC(bc_in::Px1, PrtlBC::ABSORB);
      }

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

      // diagnostics: read-only
      Diagnostics(step, time, domain);
    }

    // ---------------------------------------------------------------------- //
    // Diagnostics. `step` and `time` are those of the step just taken: the
    // state is that after `step + 1` steps, at the time `time + dt`.
    // ---------------------------------------------------------------------- //
    void Diagnostics(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      if (diag_interval <= 0 and gauss_interval <= 0) {
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
        const int nx = static_cast<int>(domain.mesh.n_active(in::x1));
        const int ny = static_cast<int>(domain.mesh.n_active(in::x2));
        raise::ErrorIf(nx != static_cast<int>(res[0]) or
                         ny != static_cast<int>(res[1]),
                       "reconnection_bh: the Gauss-law diagnostic needs one "
                       "domain",
                       HERE);
        // nodes (i, j), i = 0 .. nx - 1 (wrapped in x1), j = 0 .. ny
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
              const int i = i1(p), j = i2(p);
              if (i < 0 or i >= nx or j < 0 or j >= ny) {
                return;
              }
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
        const auto   x1 = domain.mesh.extent(in::x1);
        const auto   x2 = domain.mesh.extent(in::x2);
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
        Kokkos::parallel_reduce(
          "BHRGauss",
          CreateRangePolicy<Dim::_2D>(
            { static_cast<ncells_t>(px), static_cast<ncells_t>(py) },
            { static_cast<ncells_t>(nx - px), static_cast<ncells_t>(ny + 1 - py) }),
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
            const double q  = kappa * rho(i, j);
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
