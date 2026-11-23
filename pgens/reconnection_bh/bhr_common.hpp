/**
 * Parts shared by `pgens/reconnection_bh` and `pgens/injector_slab`:
 *
 *   - `bhr::is_root`, `bhr::reduce`, `bhr::gather`, `bhr::opt`: small helpers;
 *
 *   - `bhr::ReplenishCorrected`: the corrected replenishing rule (upstream
 * issue U17 of the fork's notes). Two changes to the block of `CustomPostStep`
 *     of `pgens/reconnection`: the replenished region runs from the inner
 *     edge of the stock slab to the boundary in x2, and the rule reads the
 *     density averaged over blocks of `smooth_cells` cells along x1 and over
 *     the row and its `smooth_rows` neighbours on each side inside the region
 *     (one by default). Blocks and rows are
 *     defined on global indices and the block sums are added over the ranks,
 *     so the smoothed density does not depend on the decomposition. 2D.
 *     With `follow_field` the target of a cell is not n0 but n0 times the
 *     field strength over `bg_B`, smoothed in the same way: the density a
 *     flux tube of the inflow carries, so that a compression or rarefaction
 *     of plasma and field together is not taken for a deficit;
 *
 *   - `bhr::WriteProfile`: `bh_profile.txt`, one line per global row with the
 *     mean over the row of the density of two species (order zero, units of
 *     n0), of B^2 (physical components, units of B0^2) and, in the replenished
 *     regions, of the smoothed density the
 *     corrected rule read and of its square. Read-only.
 */

#ifndef PGENS_RECONNECTION_BH_COMMON_HPP
#define PGENS_RECONNECTION_BH_COMMON_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"
#include "archetypes/utils.h"
#include "framework/domain/metadomain.h"

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace user {
  using namespace ntt;

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

    // values of every rank, rank after rank, on the root rank
    inline auto gather(const std::vector<double>& v) -> std::vector<double> {
#if defined(MPI_ENABLED)
      int size, rank;
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      std::vector<double> out(rank == 0 ? v.size() * size : 0, 0.0);
      MPI_Gather(v.data(),
                 static_cast<int>(v.size()),
                 MPI_DOUBLE,
                 out.data(),
                 static_cast<int>(v.size()),
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);
      return out;
#else
      return v;
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

    // a view of doubles summed over the ranks, element by element
    template <class V>
    inline void reduce_view(V& view) {
#if defined(MPI_ENABLED)
      auto view_h = Kokkos::create_mirror_view(view);
      Kokkos::deep_copy(view_h, view);
      std::vector<double> out(view_h.size(), 0.0);
      MPI_Allreduce(view_h.data(),
                    out.data(),
                    static_cast<int>(view_h.size()),
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD);
      for (std::size_t k = 0; k < out.size(); ++k) {
        view_h.data()[k] = out[k];
      }
      Kokkos::deep_copy(view, view_h);
#else
      (void)view;
#endif
    }

    // a key of [setup] with a default, without the warning of a missing key
    template <typename T>
    inline auto opt(const SimulationParams& params,
                    const std::string&      key,
                    const T&                def) -> T {
      if (params.contains(key)) {
        return params.template get<T>(key);
      }
      return def;
    }

    // the rows [first, last) of the global mesh that a range of x2 covers
    template <SimEngine::type S, class M>
    inline auto GlobalRows(const Metadomain<S, M>& metadomain,
                           real_t                  y_lo,
                           real_t y_hi) -> std::pair<long, long> {
      boundaries_t<real_t> box;
      box.push_back(Range::All);
      box.push_back({ y_lo, y_hi });
      boundaries_t<bool> incl_ghosts;
      incl_ghosts.emplace_back(false, false);
      incl_ghosts.emplace_back(false, false);
      const auto range = metadomain.mesh().ExtentToRange(box, incl_ghosts);
      return { static_cast<long>(range[1].first) - static_cast<long>(N_GHOSTS),
               static_cast<long>(range[1].second) - static_cast<long>(N_GHOSTS) };
    }

    /**
     * Component 1 of the buffer gets, on the cells of the two regions, the
     * density of component 0 averaged over global blocks of `w` cells along x1
     * and over the row and its `smooth_rows` neighbours on each side inside
     * the region; every other cell
     * of the domain gets `target` there (such a cell is never replenished).
     * Component 2 gets the target of the cell: `target` everywhere, or, with
     * `follow_field`, on the cells of the regions, `target` times the field
     * strength (physical components) over `bg_B`, smoothed like the density.
     */
    template <SimEngine::type S, class M>
    inline void SmoothRegions(const Metadomain<S, M>& metadomain,
                              Domain<S, M>&           domain,
                              const std::vector<std::pair<long, long>>& regions,
                              int    smooth_cells,
                              real_t target,
                              bool   follow_field = false,
                              real_t bg_B         = ONE,
                              int    smooth_rows  = 1) {
      static_assert(M::Dim == Dim::_2D, "the corrected rule is 2D only");
      raise::ErrorIf(smooth_cells < 1 or smooth_rows < 0,
                     "bhr: smooth_cells must be positive and smooth_rows not "
                     "negative",
                     HERE);
      const long rh  = static_cast<long>(smooth_rows);
      const long w   = static_cast<long>(smooth_cells);
      const long NX  = static_cast<long>(metadomain.mesh().n_active(in::x1));
      const long nb  = (NX + w - 1) / w;
      const long nx  = static_cast<long>(domain.mesh.n_active(in::x1));
      const long ny  = static_cast<long>(domain.mesh.n_active(in::x2));
      const auto off = domain.offset_ncells();
      const long ox = static_cast<long>(off[0]), oy = static_cast<long>(off[1]);
      const long ng    = static_cast<long>(N_GHOSTS);
      long       nrows = 0;
      for (const auto& r : regions) {
        nrows += r.second - r.first;
      }
      auto buff = domain.fields.buff;
      Kokkos::parallel_for(
        "BHRTarget",
        CreateRangePolicy<Dim::_2D>(
          { static_cast<ncells_t>(ng), static_cast<ncells_t>(ng) },
          { static_cast<ncells_t>(nx + ng), static_cast<ncells_t>(ny + ng) }),
        Lambda(cellidx_t i, cellidx_t j) {
          buff(i, j, 1) = target;
          buff(i, j, 2) = target;
        });
      if (nrows <= 0) {
        return;
      }
      array_t<double**> sums("bhr_block_sums",
                             static_cast<std::size_t>(nb),
                             static_cast<std::size_t>(nrows));
      // with `follow_field`: the field strength of the same blocks
      array_t<double**> fsums("bhr_block_field",
                              static_cast<std::size_t>(follow_field ? nb : 1),
                              static_cast<std::size_t>(follow_field ? nrows : 1));
      const auto        EM = domain.fields.em;
      const double      h1 = static_cast<double>(
        domain.mesh.metric.template sqrt_h_<1, 1>({}));
      const double h2 = static_cast<double>(
        domain.mesh.metric.template sqrt_h_<2, 2>({}));
      long row_offset = 0;
      for (const auto& r : regions) {
        const long j0 = r.first, j1 = r.second;
        const long jl0 = (j0 - oy > 0) ? j0 - oy : 0;
        const long jl1 = (j1 - oy < ny) ? j1 - oy : ny;
        const long ro  = row_offset;
        if (jl1 > jl0) {
          Kokkos::parallel_for(
            "BHRBlockSums",
            CreateRangePolicy<Dim::_2D>(
              { static_cast<ncells_t>(0), static_cast<ncells_t>(jl0) },
              { static_cast<ncells_t>(nx), static_cast<ncells_t>(jl1) }),
            Lambda(cellidx_t ii, cellidx_t jj) {
              const long i = static_cast<long>(ii), j = static_cast<long>(jj);
              Kokkos::atomic_add(&sums((ox + i) / w, ro + (oy + j - j0)),
                                 static_cast<double>(buff(i + ng, j + ng, 0)));
              if (follow_field) {
                const double b1 = h1 * static_cast<double>(
                                         EM(i + ng, j + ng, em::bx1));
                const double b2 = h2 * static_cast<double>(
                                         EM(i + ng, j + ng, em::bx2));
                const double b3 = static_cast<double>(EM(i + ng, j + ng, em::bx3));
                Kokkos::atomic_add(&fsums((ox + i) / w, ro + (oy + j - j0)),
                                   math::sqrt(b1 * b1 + b2 * b2 + b3 * b3));
              }
            });
        }
        row_offset += j1 - j0;
      }
      reduce_view(sums);
      if (follow_field) {
        reduce_view(fsums);
      }
      const double inv_bg = 1.0 / static_cast<double>(bg_B);
      row_offset          = 0;
      for (const auto& r : regions) {
        const long j0 = r.first, j1 = r.second;
        const long jl0 = (j0 - oy > 0) ? j0 - oy : 0;
        const long jl1 = (j1 - oy < ny) ? j1 - oy : ny;
        const long ro  = row_offset;
        const long nr  = j1 - j0;
        if (jl1 > jl0) {
          Kokkos::parallel_for(
            "BHRSmooth",
            CreateRangePolicy<Dim::_2D>(
              { static_cast<ncells_t>(0), static_cast<ncells_t>(jl0) },
              { static_cast<ncells_t>(nx), static_cast<ncells_t>(jl1) }),
            Lambda(cellidx_t ii, cellidx_t jj) {
              const long i = static_cast<long>(ii), j = static_cast<long>(jj);
              const long b     = (ox + i) / w;
              const long cells = (b * w + w <= NX) ? w : NX - b * w;
              const long r     = oy + j - j0;
              const long r0    = (r > rh) ? r - rh : 0;
              const long r1    = (r + rh < nr) ? r + rh : nr - 1;
              double     s = 0.0, f = 0.0;
              for (long q = r0; q <= r1; ++q) {
                s += sums(b, ro + q);
                if (follow_field) {
                  f += fsums(b, ro + q);
                }
              }
              const double norm = static_cast<double>(cells) *
                                  static_cast<double>(r1 - r0 + 1);
              buff(i + ng, j + ng, 1) = static_cast<real_t>(s / norm);
              if (follow_field) {
                buff(i + ng, j + ng, 2) = static_cast<real_t>(
                  static_cast<double>(target) * (f / norm) * inv_bg);
              }
            });
        }
        row_offset += nr;
      }
    }

    /**
     * As `arch::spatial_dist::ReplenishUniform`, with the target read cell by
     * cell from a component of the buffer: below 0.9 of its target a cell
     * gets the difference to the target (in units of n0).
     */
    template <MetricClass M, int N>
    struct ReplenishToTarget {
      const M                    metric;
      const ndfield_t<M::Dim, N> buffer;
      const idx_t                idx_density, idx_target;

      ReplenishToTarget(const M&                    metric,
                        const ndfield_t<M::Dim, N>& buffer,
                        idx_t                       idx_density,
                        idx_t                       idx_target)
        : metric { metric }
        , buffer { buffer }
        , idx_density { idx_density }
        , idx_target { idx_target } {}

      Inline auto operator()(const coord_t<M::Dim>& x_Ph) const -> real_t {
        coord_t<M::Dim> x_Cd { ZERO };
        metric.template convert<Crd::Ph, Crd::Cd>(x_Ph, x_Cd);
        const auto i    = static_cast<ncells_t>(x_Cd[0]) + N_GHOSTS;
        const auto j    = static_cast<ncells_t>(x_Cd[1]) + N_GHOSTS;
        const auto dens = buffer(i, j, idx_density);
        const auto tgt  = buffer(i, j, idx_target);
        if ((real_t)(0.9) * tgt > dens) {
          return tgt - dens;
        } else {
          return ZERO;
        }
      }
    };

    // the two replenished regions of the corrected rule, as global rows
    template <SimEngine::type S, class M>
    inline auto Regions(const Metadomain<S, M>& metadomain,
                        const Domain<S, M>&     domain,
                        real_t                  ymin,
                        real_t                  ymax,
                        real_t inj_ypad) -> std::vector<std::pair<long, long>> {
      const auto dx = domain.mesh.metric.template sqrt_h_<1, 1>({});
      return { GlobalRows(metadomain, ymax - inj_ypad - 10 * dx, ymax),
               GlobalRows(metadomain, ymin, ymin + inj_ypad + 10 * dx) };
    }

    /**
     * The corrected replenishing rule. The calls and their order are those of
     * the stock block; what differs is the box (it reaches the boundary) and
     * the component of the buffer the rule reads (the smoothed density). With
     * `follow_field` the target is n0 times the smoothed field strength over
     * `bg_B` instead of n0.
     */
    template <SimEngine::type S, class M>
    inline void ReplenishCorrected(const SimulationParams& params,
                                   const Metadomain<S, M>& metadomain,
                                   Domain<S, M>&           domain,
                                   spidx_t                 sp_a,
                                   spidx_t                 sp_b,
                                   real_t                  temperature,
                                   real_t                  ymin,
                                   real_t                  ymax,
                                   real_t                  inj_ypad,
                                   int                     smooth_cells,
                                   bool                    follow_field = false,
                                   real_t                  bg_B         = ONE,
                                   int                     smooth_rows  = 1) {
      static_assert(M::Dim == Dim::_2D, "the corrected rule is 2D only");
      const auto energy_dist = arch::energy_dist::Maxwellian<M::Dim, M::CoordType>(
        domain.random_pool(),
        temperature);

      const auto dx = domain.mesh.metric.template sqrt_h_<1, 1>({});

      boundaries_t<real_t> inj_box_up, inj_box_down;
      inj_box_up.push_back(Range::All);
      inj_box_down.push_back(Range::All);
      inj_box_up.push_back({ ymax - inj_ypad - 10 * dx, ymax });
      inj_box_down.push_back({ ymin, ymin + inj_ypad + 10 * dx });

      arch::ComputeMomentWithSpecies<S, M, FldsID::Rho, 3>(params,
                                                           domain,
                                                           { sp_a, sp_b },
                                                           domain.fields.buff);
      SmoothRegions(metadomain,
                    domain,
                    Regions(metadomain, domain, ymin, ymax, inj_ypad),
                    smooth_cells,
                    ONE,
                    follow_field,
                    bg_B,
                    smooth_rows);

      if (not follow_field) {
        const auto replenish_sdist = arch::spatial_dist::ReplenishUniform<M, 3>(
          domain.mesh.metric,
          domain.fields.buff,
          1u,
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
      } else {
        const auto replenish_sdist = ReplenishToTarget<M, 3>(domain.mesh.metric,
                                                             domain.fields.buff,
                                                             1u,
                                                             2u);
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
      }
    }

    /**
     * `bh_profile.txt`. Columns: steps, time, global row, x2 of the centre of
     * the row, cells averaged, density of the species `sp_a` and `sp_b`
     * together, B^2, then (zero outside `regions`, or when `regions` is empty)
     * the smoothed density component 1 of the buffer holds and its square.
     * The cells of a row are those with abs(x1 - xc) <= xmax (all if
     * xmax <= 0). Component 0 of the buffer is overwritten.
     */
    template <SimEngine::type S, class M>
    inline void WriteProfile(const SimulationParams& params,
                             const Metadomain<S, M>& metadomain,
                             Domain<S, M>&           domain,
                             spidx_t                 sp_a,
                             spidx_t                 sp_b,
                             real_t                  xc,
                             real_t                  xmax,
                             const std::vector<std::pair<long, long>>& regions,
                             long long steps_done,
                             double    time_now,
                             bool&     started) {
      static_assert(M::Dim == Dim::_2D, "the profile is 2D only");
      const long NX  = static_cast<long>(metadomain.mesh().n_active(in::x1));
      const long NY  = static_cast<long>(metadomain.mesh().n_active(in::x2));
      const long nx  = static_cast<long>(domain.mesh.n_active(in::x1));
      const long ny  = static_cast<long>(domain.mesh.n_active(in::x2));
      const auto off = domain.offset_ncells();
      const long ox = static_cast<long>(off[0]), oy = static_cast<long>(off[1]);
      const long ng     = static_cast<long>(N_GHOSTS);
      const auto gx1    = metadomain.mesh().extent(in::x1);
      const auto gx2    = metadomain.mesh().extent(in::x2);
      const double x_lo = static_cast<double>(gx1.first);
      const double dxg  = (static_cast<double>(gx1.second) - x_lo) /
                         static_cast<double>(NX);
      const double y_lo = static_cast<double>(gx2.first);
      const double dyg  = (static_cast<double>(gx2.second) - y_lo) /
                         static_cast<double>(NY);
      const double x_c = static_cast<double>(xc), x_m = static_cast<double>(xmax);
      const double h1 = static_cast<double>(
        domain.mesh.metric.template sqrt_h_<1, 1>({}));
      const double h2 = static_cast<double>(
        domain.mesh.metric.template sqrt_h_<2, 2>({}));
      // the third component is not scaled in 2D (`Minkowski::transform`)
      const double h3 = 1.0;

      // the smoothed density is read before component 0 is recomputed (the
      // two are different components, so the order does not matter)
      arch::ComputeMomentWithSpecies<S, M, FldsID::N, 3>(params,
                                                         domain,
                                                         { sp_a, sp_b },
                                                         domain.fields.buff);
      const long j_up0 = regions.empty() ? NY : regions[0].first;
      const long j_up1 = regions.empty() ? NY : regions[0].second;
      const long j_dn0 = (regions.size() < 2) ? 0 : regions[1].first;
      const long j_dn1 = (regions.size() < 2) ? 0 : regions[1].second;

      // per global row: cells, density, B^2, smoothed density, its square
      array_t<double**> rows("bhr_profile", static_cast<std::size_t>(NY), 5);
      auto              buff = domain.fields.buff;
      const auto        EM   = domain.fields.em;
      Kokkos::parallel_for(
        "BHRProfile",
        CreateRangePolicy<Dim::_2D>(
          { static_cast<ncells_t>(0), static_cast<ncells_t>(0) },
          { static_cast<ncells_t>(nx), static_cast<ncells_t>(ny) }),
        Lambda(cellidx_t ii, cellidx_t jj) {
          const long   i = static_cast<long>(ii), j = static_cast<long>(jj);
          const double x = x_lo + (static_cast<double>(ox + i) + 0.5) * dxg;
          if (x_m > 0.0 and math::abs(x - x_c) > x_m) {
            return;
          }
          const long J = oy + j;
          // the stored components are contravariant: times sqrt(h_ii) for the
          // physical ones
          const double b1 = h1 * static_cast<double>(EM(i + ng, j + ng, em::bx1));
          const double b2 = h2 * static_cast<double>(EM(i + ng, j + ng, em::bx2));
          const double b3 = h3 * static_cast<double>(EM(i + ng, j + ng, em::bx3));
          Kokkos::atomic_add(&rows(J, 0), 1.0);
          Kokkos::atomic_add(&rows(J, 1),
                             static_cast<double>(buff(i + ng, j + ng, 0)));
          Kokkos::atomic_add(&rows(J, 2), b1 * b1 + b2 * b2 + b3 * b3);
          if ((J >= j_up0 and J < j_up1) or (J >= j_dn0 and J < j_dn1)) {
            const double s = static_cast<double>(buff(i + ng, j + ng, 1));
            Kokkos::atomic_add(&rows(J, 3), s);
            Kokkos::atomic_add(&rows(J, 4), s * s);
          }
        });
      reduce_view(rows);
      if (not is_root()) {
        return;
      }
      auto rows_h = Kokkos::create_mirror_view(rows);
      Kokkos::deep_copy(rows_h, rows);
      std::FILE* f = std::fopen("bh_profile.txt", started ? "a" : "w");
      raise::ErrorIf(f == nullptr, "bhr: cannot open bh_profile.txt", HERE);
      if (not started) {
        std::fprintf(
          f,
          "# rows of %ld x %ld cells; species %d and %d; abs(x1 - %.9g) "
          "<= %.9g (0: all); regions of the corrected rule: rows %ld to "
          "%ld and %ld to %ld (empty: none)\n",
          NX,
          NY,
          static_cast<int>(sp_a),
          static_cast<int>(sp_b),
          x_c,
          x_m,
          j_dn0,
          j_dn1,
          j_up0,
          j_up1);
        std::fprintf(f, "# steps time row x2 cells density B2 smoothed smoothed2\n");
        started = true;
      }
      for (long J = 0; J < NY; ++J) {
        const double c = rows_h(J, 0);
        const double d = (c > 0.0) ? 1.0 / c : 0.0;
        std::fprintf(f,
                     "%lld %.17g %ld %.9g %.0f %.17g %.17g %.17g %.17g\n",
                     steps_done,
                     time_now,
                     J,
                     y_lo + (static_cast<double>(J) + 0.5) * dyg,
                     c,
                     rows_h(J, 1) * d,
                     rows_h(J, 2) * d,
                     rows_h(J, 3) * d,
                     rows_h(J, 4) * d);
      }
      std::fclose(f);
    }
  } // namespace bhr

} // namespace user

#endif // PGENS_RECONNECTION_BH_COMMON_HPP
