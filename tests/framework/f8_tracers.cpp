/**
 * @file tests/framework/f8_tracers.cpp
 * @brief Codex 2's F8 GPU tracer qualification (weights audit 2026-10-29 §F8; scaling-harness
 *        contract 2026-10-30 §2; Codex 1's harness review F3): injected ID/weight/payload
 *        tracers with dead holes through the production spatial sort (the vendor sort on
 *        CUDA/HIP/SYCL builds, BinSort elsewhere), the next-step current deposition after the
 *        sort, the split invariance of the deposited current, and a checkpoint round trip of
 *        the sorted population, in 1D, 2D and 3D, in the precision of the build. Every
 *        assertion is enumerated and counted; the program fails when any assertion fails or
 *        when a selection is empty (no silent pass).
 *
 * Tracers (per dimension): N = 1024 particles, id = slot; w(id) = (id + 1)/8 (injective,
 * exact in binary); real payloads pld_r(id, 0) = w + 1/2, pld_r(id, 1) = 10.5 + id; integer
 * payload pld_i(id, 0) = id + 10 (the id carrier); momenta exact dyadic functions of id;
 * cells shuffled (id * 7919 mod n_cells per axis); every fourth particle dead (tag dead,
 * weight -1). The assertions after SortSpatially:
 *   S1  npart(): unchanged on the BinSort path, equal to the alive count on the compacting
 *       (tiled_deposit vendor) path;
 *   S2  the alive particles form a prefix in non-decreasing tile index (tile = cell for
 *       tile size 1), no alive particle after a dead one;
 *   S3  the alive id set (from pld_i) equals the injected alive id set exactly;
 *   S4  for every alive slot: weight, both real payloads, the three momenta and the cell/
 *       displacement equal the id's injected values exactly (one permutation for every SoA
 *       member);
 *   S5  sum(w) and sum(w ux1) over the alive particles equal the injected sums exactly
 *       (dyadic rationals: exact in double, and in float at these counts).
 * Next-step deposition (the pusher rewrites the previous positions before any deposit, so the
 * sort's non-permutation of *_prev is immaterial; the test rewrites them from the id as the
 * pusher would, then deposits with the production DepositCurrents_kernel):
 *   D1  J(sorted population) = J(unsorted copy) to 1e-10 (double) / 1e-5 (single) in the
 *       relative L2 norm over the active cells, with a non-zero J (gate: |J| > 0);
 *   D2  split invariance: the same population with every particle split in two of half weight
 *       at the same phase-space point gives the same J to the same tolerance.
 * Checkpoint round trip (OUTPUT_ENABLED builds; the Metadomain writer/reader of Entity):
 *   C1  npart after the read equals npart before;
 *   C2  every array (i, dx, ux, weight, tag, pld_r, pld_i) equals the written one exactly over
 *       [0, npart).
 * Host-side comparisons only; the test writes its checkpoint under test_f8_tracers_<D>d/ and
 * removes it.
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include "metrics/minkowski.h"

#include "framework/containers/particles.h"
#include "framework/containers/species.h"
#include "framework/domain/grid.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

#include "kernels/deposition/currents/global.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#if defined(OUTPUT_ENABLED)
  #include <adios2.h>
  #include <toml11/toml.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;

namespace {

  int n_pass = 0, n_fail = 0;

  void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    ok ? ++n_pass : ++n_fail;
  }

  constexpr npart_t  NTR    = 1024;
  constexpr ncells_t NCELLS = 16;

  // the injected values as functions of the id
  double w_of(npart_t id) {
    return static_cast<double>(id + 1) / 8.0;
  }

  bool dead_of(npart_t id) {
    return id % 4 == 3;
  }

  int cell_of(npart_t id, int axis) {
    return static_cast<int>((static_cast<unsigned long long>(id) * 7919ull + 131ull * static_cast<unsigned long long>(axis)) %
                            NCELLS);
  }

  double dx_of(npart_t id, int axis) {
    return ((id + axis) % 2 == 0) ? 0.25 : 0.75;
  }

  double u_of(npart_t id, int c) {
    switch (c) {
      case 0:
        return 0.5 * (static_cast<double>(id % 3) - 1.0);
      case 1:
        return 0.25 * (static_cast<double>(id % 5) - 2.0);
      default:
        return 0.125 * (static_cast<double>(id % 7) - 3.0);
    }
  }

  // the previous position the pusher would leave: a displacement of 1/8 cell towards +x within the cell
  double dx_prev_of(npart_t id, int axis) {
    return dx_of(id, axis) - (axis == 0 ? 0.125 : 0.0);
  }

  template <Dimension D>
  struct Pop {
    using P = Particles<D, Coord::Cartesian>;
    std::unique_ptr<P> prtls;

    Pop(npart_t maxn, const std::string& label) {
      prtls = std::make_unique<P>(static_cast<spidx_t>(1), label, 1.0f, 1.0f, maxn, timestep_t { 0 }, timestep_t { 0 },
                                  ParticlePusher::BORIS, false, RadiativeDrag::NONE, EmissionType::NONE,
                                  static_cast<unsigned short>(2), static_cast<unsigned short>(1));
    }

    // fill slots [0, n) with the tracers; split > 1 puts `split` copies of each id with weight / split
    void fill(npart_t n_ids, int split = 1) {
      auto& p  = *prtls;
      auto  i1 = Kokkos::create_mirror_view(p.i1), i2 = Kokkos::create_mirror_view(p.i2), i3 = Kokkos::create_mirror_view(p.i3);
      auto  d1 = Kokkos::create_mirror_view(p.dx1), d2 = Kokkos::create_mirror_view(p.dx2), d3 = Kokkos::create_mirror_view(p.dx3);
      auto  p1 = Kokkos::create_mirror_view(p.i1_prev), p2 = Kokkos::create_mirror_view(p.i2_prev), p3 = Kokkos::create_mirror_view(p.i3_prev);
      auto  q1 = Kokkos::create_mirror_view(p.dx1_prev), q2 = Kokkos::create_mirror_view(p.dx2_prev), q3 = Kokkos::create_mirror_view(p.dx3_prev);
      auto  u1 = Kokkos::create_mirror_view(p.ux1), u2 = Kokkos::create_mirror_view(p.ux2), u3 = Kokkos::create_mirror_view(p.ux3);
      auto  w  = Kokkos::create_mirror_view(p.weight);
      auto  tg = Kokkos::create_mirror_view(p.tag);
      auto  pr = Kokkos::create_mirror_view(p.pld_r);
      auto  pi = Kokkos::create_mirror_view(p.pld_i);
      const npart_t n = n_ids * static_cast<npart_t>(split);
      for (npart_t s = 0; s < p.maxnpart(); ++s) {
        tg(s) = ParticleTag::dead;
        w(s)  = static_cast<real_t>(-1.0);
      }
      for (npart_t s = 0; s < n; ++s) {
        const npart_t id = s / static_cast<npart_t>(split);
        i1(s) = cell_of(id, 0);
        d1(s) = static_cast<prtldx_t>(dx_of(id, 0));
        p1(s) = i1(s);
        q1(s) = static_cast<prtldx_t>(dx_prev_of(id, 0));
        if constexpr (D >= Dim::_2D) {
          i2(s) = cell_of(id, 1);
          d2(s) = static_cast<prtldx_t>(dx_of(id, 1));
          p2(s) = i2(s);
          q2(s) = static_cast<prtldx_t>(dx_prev_of(id, 1));
        }
        if constexpr (D == Dim::_3D) {
          i3(s) = cell_of(id, 2);
          d3(s) = static_cast<prtldx_t>(dx_of(id, 2));
          p3(s) = i3(s);
          q3(s) = static_cast<prtldx_t>(dx_prev_of(id, 2));
        }
        u1(s)    = static_cast<real_t>(u_of(id, 0));
        u2(s)    = static_cast<real_t>(u_of(id, 1));
        u3(s)    = static_cast<real_t>(u_of(id, 2));
        tg(s)    = dead_of(id) ? ParticleTag::dead : ParticleTag::alive;
        w(s)     = dead_of(id) ? static_cast<real_t>(-1.0) : static_cast<real_t>(w_of(id) / static_cast<double>(split));
        pr(s, 0) = static_cast<real_t>(w_of(id) + 0.5);
        pr(s, 1) = static_cast<real_t>(10.5 + static_cast<double>(id));
        pi(s, 0) = id + 10;
      }
      Kokkos::deep_copy(p.i1, i1);
      Kokkos::deep_copy(p.dx1, d1);
      Kokkos::deep_copy(p.i1_prev, p1);
      Kokkos::deep_copy(p.dx1_prev, q1);
      if constexpr (D >= Dim::_2D) {
        Kokkos::deep_copy(p.i2, i2);
        Kokkos::deep_copy(p.dx2, d2);
        Kokkos::deep_copy(p.i2_prev, p2);
        Kokkos::deep_copy(p.dx2_prev, q2);
      }
      if constexpr (D == Dim::_3D) {
        Kokkos::deep_copy(p.i3, i3);
        Kokkos::deep_copy(p.dx3, d3);
        Kokkos::deep_copy(p.i3_prev, p3);
        Kokkos::deep_copy(p.dx3_prev, q3);
      }
      Kokkos::deep_copy(p.ux1, u1);
      Kokkos::deep_copy(p.ux2, u2);
      Kokkos::deep_copy(p.ux3, u3);
      Kokkos::deep_copy(p.weight, w);
      Kokkos::deep_copy(p.tag, tg);
      Kokkos::deep_copy(p.pld_r, pr);
      Kokkos::deep_copy(p.pld_i, pi);
      p.set_npart(n);
    }

    // what the pusher does before any deposit: rewrite the previous positions of every alive slot (from the id
    // carried in pld_i, as a deterministic stand-in for the pushed trajectory)
    void rewrite_prev() {
      auto&      p  = *prtls;
      const auto n  = p.npart();
      auto       i1 = p.i1, i2 = p.i2, i3 = p.i3;
      auto       p1 = p.i1_prev, p2 = p.i2_prev, p3 = p.i3_prev;
      auto       q1 = p.dx1_prev, q2 = p.dx2_prev, q3 = p.dx3_prev;
      auto       d1 = p.dx1, d2 = p.dx2, d3 = p.dx3;
      auto       tg = p.tag;
      Kokkos::parallel_for(
        "rewrite_prev", n, Lambda(npart_t s) {
          if (tg(s) == ParticleTag::dead) {
            return;
          }
          p1(s) = i1(s);
          q1(s) = d1(s) - static_cast<prtldx_t>(0.125);
          if constexpr (D >= Dim::_2D) {
            p2(s) = i2(s);
            q2(s) = d2(s);
          }
          if constexpr (D == Dim::_3D) {
            p3(s) = i3(s);
            q3(s) = d3(s);
          }
        });
      Kokkos::fence();
    }
  };

  template <Dimension D>
  auto make_grid() {
    std::vector<ncells_t> res(static_cast<std::size_t>(D), NCELLS);
    boundaries_t<real_t>  ext;
    for (int d = 0; d < static_cast<int>(D); ++d) {
      ext.push_back({ static_cast<real_t>(0.0), static_cast<real_t>(NCELLS) });
    }
    return std::make_pair(res, ext);
  }

  // the current of a population through the production deposit kernel; returned as a host vector over all cells
  // and the three components
  template <Dimension D>
  std::vector<double> deposit(Pop<D>& pop) {
    using M          = Minkowski<D>;
    auto [res, ext]  = make_grid<D>();
    M    metric { res, ext, {} };
    auto& p = *pop.prtls;
    std::vector<double> out;
    if constexpr (D == Dim::_1D) {
      ndfield_t<D, 3> J { "J", NCELLS + 2 * N_GHOSTS };
      auto            Js = Kokkos::Experimental::create_scatter_view(J);
      Kokkos::parallel_for("deposit", p.npart(),
                           kernel::DepositCurrents_kernel<SimEngine::SRPIC, M, 1>(Js, static_cast<ParticleArrays&>(p), metric,
                                                                                  static_cast<real_t>(1.0), static_cast<real_t>(1.0)));
      Kokkos::Experimental::contribute(J, Js);
      auto h = Kokkos::create_mirror_view(J);
      Kokkos::deep_copy(h, J);
      for (std::size_t i = 0; i < h.extent(0); ++i) {
        for (int c = 0; c < 3; ++c) {
          out.push_back(static_cast<double>(h(i, c)));
        }
      }
    } else if constexpr (D == Dim::_2D) {
      ndfield_t<D, 3> J { "J", NCELLS + 2 * N_GHOSTS, NCELLS + 2 * N_GHOSTS };
      auto            Js = Kokkos::Experimental::create_scatter_view(J);
      Kokkos::parallel_for("deposit", p.npart(),
                           kernel::DepositCurrents_kernel<SimEngine::SRPIC, M, 1>(Js, static_cast<ParticleArrays&>(p), metric,
                                                                                  static_cast<real_t>(1.0), static_cast<real_t>(1.0)));
      Kokkos::Experimental::contribute(J, Js);
      auto h = Kokkos::create_mirror_view(J);
      Kokkos::deep_copy(h, J);
      for (std::size_t i = 0; i < h.extent(0); ++i) {
        for (std::size_t j = 0; j < h.extent(1); ++j) {
          for (int c = 0; c < 3; ++c) {
            out.push_back(static_cast<double>(h(i, j, c)));
          }
        }
      }
    } else {
      ndfield_t<D, 3> J { "J", NCELLS + 2 * N_GHOSTS, NCELLS + 2 * N_GHOSTS, NCELLS + 2 * N_GHOSTS };
      auto            Js = Kokkos::Experimental::create_scatter_view(J);
      Kokkos::parallel_for("deposit", p.npart(),
                           kernel::DepositCurrents_kernel<SimEngine::SRPIC, M, 1>(Js, static_cast<ParticleArrays&>(p), metric,
                                                                                  static_cast<real_t>(1.0), static_cast<real_t>(1.0)));
      Kokkos::Experimental::contribute(J, Js);
      auto h = Kokkos::create_mirror_view(J);
      Kokkos::deep_copy(h, J);
      for (std::size_t i = 0; i < h.extent(0); ++i) {
        for (std::size_t j = 0; j < h.extent(1); ++j) {
          for (std::size_t k = 0; k < h.extent(2); ++k) {
            for (int c = 0; c < 3; ++c) {
              out.push_back(static_cast<double>(h(i, j, k, c)));
            }
          }
        }
      }
    }
    Kokkos::fence();
    return out;
  }

  double rel_l2(const std::vector<double>& a, const std::vector<double>& b, double& norm_b) {
    double d = 0.0, n = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      d += (a[i] - b[i]) * (a[i] - b[i]);
      n += b[i] * b[i];
    }
    norm_b = std::sqrt(n);
    return n > 0.0 ? std::sqrt(d / n) : std::sqrt(d);
  }

  template <Dimension D>
  void run_dimension() {
    const int    Di      = static_cast<int>(D);
    const double tol_dep = sizeof(real_t) == 8 ? 1.0e-10 : 1.0e-5;
    std::printf("\n=== F8 tracers, %dD, real_t = %s, %u tracers (%u alive), %u cells per axis ===\n", Di,
                sizeof(real_t) == 8 ? "double" : "float", (unsigned)NTR, (unsigned)(NTR - NTR / 4), (unsigned)NCELLS);
    auto [res, ext] = make_grid<D>();
    Grid<D> grid { res, ext };

    // injected sums over the alive tracers
    std::set<npart_t> alive_ids;
    double            sum_w = 0.0, sum_wu = 0.0;
    for (npart_t id = 0; id < NTR; ++id) {
      if (!dead_of(id)) {
        alive_ids.insert(id);
        sum_w += w_of(id);
        sum_wu += w_of(id) * u_of(id, 0);
      }
    }
    const npart_t n_alive = static_cast<npart_t>(alive_ids.size());

    // ---- the sorted population
    Pop<D> sorted(2 * NTR + 64, "tr");
    sorted.fill(NTR);
    sorted.prtls->SortSpatially(grid);
    Kokkos::fence();
    auto& p  = *sorted.prtls;
    auto  i1 = Kokkos::create_mirror_view(p.i1), i2 = Kokkos::create_mirror_view(p.i2), i3 = Kokkos::create_mirror_view(p.i3);
    auto  d1 = Kokkos::create_mirror_view(p.dx1), d2 = Kokkos::create_mirror_view(p.dx2), d3 = Kokkos::create_mirror_view(p.dx3);
    auto  u1 = Kokkos::create_mirror_view(p.ux1), u2 = Kokkos::create_mirror_view(p.ux2), u3 = Kokkos::create_mirror_view(p.ux3);
    auto  w  = Kokkos::create_mirror_view(p.weight);
    auto  tg = Kokkos::create_mirror_view(p.tag);
    auto  pr = Kokkos::create_mirror_view(p.pld_r);
    auto  pi = Kokkos::create_mirror_view(p.pld_i);
    Kokkos::deep_copy(i1, p.i1);
    Kokkos::deep_copy(i2, p.i2);
    Kokkos::deep_copy(i3, p.i3);
    Kokkos::deep_copy(d1, p.dx1);
    Kokkos::deep_copy(d2, p.dx2);
    Kokkos::deep_copy(d3, p.dx3);
    Kokkos::deep_copy(u1, p.ux1);
    Kokkos::deep_copy(u2, p.ux2);
    Kokkos::deep_copy(u3, p.ux3);
    Kokkos::deep_copy(w, p.weight);
    Kokkos::deep_copy(tg, p.tag);
    Kokkos::deep_copy(pr, p.pld_r);
    Kokkos::deep_copy(pi, p.pld_i);

    // S1
#if defined(TILED_DEPOSIT)
    check(p.npart() == n_alive, fmt::format("%dD S1: compacting sort: npart() = %u = alive count %u", Di, (unsigned)p.npart(), (unsigned)n_alive));
    const ncells_t T = static_cast<ncells_t>(TILED_DEPOSIT_TILE_SIZE);
#else
    check(p.npart() == NTR, fmt::format("%dD S1: BinSort path: npart() = %u unchanged", Di, (unsigned)p.npart()));
    const ncells_t T = 1u;
#endif
    // S2: alive prefix in non-decreasing tile index, no alive after a dead one
    auto tile_of = [&](npart_t s) -> unsigned long long {
      unsigned long long t = static_cast<unsigned long long>(i1(s)) / T;
      if constexpr (D >= Dim::_2D) {
        t = t * ((NCELLS + T - 1) / T) + static_cast<unsigned long long>(i2(s)) / T;
      }
      if constexpr (D == Dim::_3D) {
        t = t * ((NCELLS + T - 1) / T) + static_cast<unsigned long long>(i3(s)) / T;
      }
      return t;
    };
    bool               seen_dead = false, prefix_ok = true, order_ok = true, have_prev = false;
    unsigned long long prev_tile = 0;
    npart_t            n_alive_seen = 0;
    std::set<npart_t>  ids_seen;
    bool               members_ok = true;
    double             s_w = 0.0, s_wu = 0.0;
    for (npart_t s = 0; s < p.npart(); ++s) {
      if (tg(s) == ParticleTag::dead) {
        seen_dead = true;
        continue;
      }
      if (seen_dead) {
        prefix_ok = false;
      }
      const auto t = tile_of(s);
      if (have_prev && t < prev_tile) {
        order_ok = false;
      }
      prev_tile = t;
      have_prev = true;
      ++n_alive_seen;
      const npart_t id = pi(s, 0) - 10;
      ids_seen.insert(id);
      bool ok = static_cast<double>(w(s)) == w_of(id) && static_cast<double>(pr(s, 0)) == static_cast<double>(static_cast<real_t>(w_of(id) + 0.5)) &&
                static_cast<double>(pr(s, 1)) == static_cast<double>(static_cast<real_t>(10.5 + static_cast<double>(id))) &&
                static_cast<double>(u1(s)) == u_of(id, 0) && static_cast<double>(u2(s)) == u_of(id, 1) &&
                static_cast<double>(u3(s)) == u_of(id, 2) && i1(s) == cell_of(id, 0) && static_cast<double>(d1(s)) == dx_of(id, 0);
      if constexpr (D >= Dim::_2D) {
        ok = ok && i2(s) == cell_of(id, 1) && static_cast<double>(d2(s)) == dx_of(id, 1);
      }
      if constexpr (D == Dim::_3D) {
        ok = ok && i3(s) == cell_of(id, 2) && static_cast<double>(d3(s)) == dx_of(id, 2);
      }
      if (!ok) {
        members_ok = false;
      }
      s_w += static_cast<double>(w(s));
      s_wu += static_cast<double>(w(s)) * static_cast<double>(u1(s));
    }
    check(prefix_ok && order_ok, fmt::format("%dD S2: alive prefix in non-decreasing tile index (tile size %u), no alive after a dead", Di, (unsigned)T));
    check(n_alive_seen == n_alive && ids_seen == alive_ids,
          fmt::format("%dD S3: the alive id set after the sort equals the injected one (%u ids)", Di, (unsigned)n_alive_seen));
    check(members_ok, fmt::format("%dD S4: weight, payloads, momenta, cell and displacement of every alive slot equal the id's injected values (one permutation)", Di));
    check(s_w == sum_w && s_wu == sum_wu,
          fmt::format("%dD S5: sum(w) = %.6f and sum(w ux1) = %.6f over the alive particles equal the injected sums exactly", Di, s_w, s_wu));

    // ---- next-step deposition: sorted vs unsorted copy, after the pusher's rewrite of the previous positions
    Pop<D> unsorted(2 * NTR + 64, "tr_unsorted");
    unsorted.fill(NTR);
    sorted.rewrite_prev();
    unsorted.rewrite_prev();
    const auto J_sorted   = deposit(sorted);
    const auto J_unsorted = deposit(unsorted);
    double     normJ      = 0.0;
    const double d_sort   = rel_l2(J_sorted, J_unsorted, normJ);
    check(normJ > 0.0, fmt::format("%dD D1 gate: the deposited current is non-zero (|J| = %.6e)", Di, normJ));
    check(d_sort <= tol_dep, fmt::format("%dD D1: J(sorted) vs J(unsorted) relative L2 = %.3e (tolerance %.0e)", Di, d_sort, tol_dep));
    // ---- split invariance: every tracer split in two of half weight, same phase-space point
    Pop<D> split(2 * NTR + 64, "tr_split");
    split.fill(NTR, 2);
    split.prtls->SortSpatially(grid);
    Kokkos::fence();
    split.rewrite_prev();
    const auto   J_split = deposit(split);
    double       n2      = 0.0;
    const double d_split = rel_l2(J_split, J_unsorted, n2);
    check(d_split <= tol_dep, fmt::format("%dD D2: J(split in two, sorted) vs J(unsplit) relative L2 = %.3e (tolerance %.0e)", Di, d_split, tol_dep));

#if defined(OUTPUT_ENABLED)
    // ---- checkpoint round trip of the sorted population through Entity's writer and reader
    {
      using M = Minkowski<D>;
      const std::string dir = fmt::format("test_f8_tracers_%dd", Di);
      std::filesystem::remove_all(dir);
      boundaries_t<FldsBC> fbc;
      boundaries_t<PrtlBC> pbc;
      for (int d = 0; d < Di; ++d) {
        fbc.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
        pbc.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
      }
      const std::vector<int>             decomp(static_cast<std::size_t>(Di), -1);
      const std::vector<ParticleSpecies> sp_params { ParticleSpecies { static_cast<spidx_t>(1), "tr", 1.0f, 1.0f, 2 * NTR + 64, timestep_t { 0 },
                                                                       timestep_t { 0 }, ParticlePusher::BORIS, false, RadiativeDrag::NONE,
                                                                       EmissionType::NONE, static_cast<unsigned short>(2),
                                                                       static_cast<unsigned short>(1) } };
      adios2::ADIOS adios;
      const auto    ckpt_step = timestep_t { 3 };
      {
        Metadomain<SimEngine::SRPIC, M> md { 1u, decomp, res, ext, fbc, pbc, {}, sp_params };
        auto*                           local = md.subdomain_ptr(md.l_subdomain_indices()[0]);
        auto&                           sp    = local->species[0];
        // copy the sorted population into the species (device to device)
        const auto n = p.npart();
        sp.set_npart(n);
        auto cp = [&](auto& dst, auto& src) { Kokkos::deep_copy(Kokkos::subview(dst, std::make_pair(npart_t { 0 }, n)), Kokkos::subview(src, std::make_pair(npart_t { 0 }, n))); };
        cp(sp.i1, p.i1);
        cp(sp.dx1, p.dx1);
        if constexpr (D >= Dim::_2D) {
          cp(sp.i2, p.i2);
          cp(sp.dx2, p.dx2);
        }
        if constexpr (D == Dim::_3D) {
          cp(sp.i3, p.i3);
          cp(sp.dx3, p.dx3);
        }
        cp(sp.ux1, p.ux1);
        cp(sp.ux2, p.ux2);
        cp(sp.ux3, p.ux3);
        cp(sp.weight, p.weight);
        cp(sp.tag, p.tag);
        Kokkos::deep_copy(Kokkos::subview(sp.pld_r, std::make_pair(npart_t { 0 }, n), Kokkos::ALL), Kokkos::subview(p.pld_r, std::make_pair(npart_t { 0 }, n), Kokkos::ALL));
        Kokkos::deep_copy(Kokkos::subview(sp.pld_i, std::make_pair(npart_t { 0 }, n), Kokkos::ALL), Kokkos::subview(p.pld_i, std::make_pair(npart_t { 0 }, n), Kokkos::ALL));
        Kokkos::fence();
        SimulationParams params;
        params.set("checkpoint.write_path", dir);
        params.set("checkpoint.interval", timestep_t { 1 });
        params.set("checkpoint.interval_time", simtime_t { -1.0 });
        params.set("checkpoint.keep", 2);
        params.set("checkpoint.walltime", std::string { "" });
        params.setRawData(toml::value { toml::table {} });
        md.InitCheckpointWriter(&adios, params);
        const bool wrote = md.WriteCheckpoint(params, ckpt_step, ckpt_step, simtime_t { 0.0 }, simtime_t { 0.0 });
        check(wrote, fmt::format("%dD C0: the checkpoint of the sorted population was written (%u particles)", Di, (unsigned)n));
      }
      {
        Metadomain<SimEngine::SRPIC, M> md2 { 1u, decomp, res, ext, fbc, pbc, {}, sp_params };
        SimulationParams                params2;
        params2.set("checkpoint.read_path", dir);
        params2.set("checkpoint.start_step", ckpt_step);
        md2.ContinueFromCheckpoint(&adios, params2);
        auto* local = md2.subdomain_ptr(md2.l_subdomain_indices()[0]);
        auto& sp    = local->species[0];
        check(sp.npart() == p.npart(), fmt::format("%dD C1: npart after the read-back %u = %u", Di, (unsigned)sp.npart(), (unsigned)p.npart()));
        auto equal_arrays = [&](auto& a, auto& b, const char* name, std::string& bad) {
          auto ha = Kokkos::create_mirror_view(a);
          auto hb = Kokkos::create_mirror_view(b);
          Kokkos::deep_copy(ha, a);
          Kokkos::deep_copy(hb, b);
          for (npart_t s = 0; s < p.npart(); ++s) {
            if (ha(s) != hb(s)) {
              bad += std::string(" ") + name;
              return;
            }
          }
        };
        std::string bad;
        equal_arrays(sp.i1, p.i1, "i1", bad);
        equal_arrays(sp.dx1, p.dx1, "dx1", bad);
        if constexpr (D >= Dim::_2D) {
          equal_arrays(sp.i2, p.i2, "i2", bad);
          equal_arrays(sp.dx2, p.dx2, "dx2", bad);
        }
        if constexpr (D == Dim::_3D) {
          equal_arrays(sp.i3, p.i3, "i3", bad);
          equal_arrays(sp.dx3, p.dx3, "dx3", bad);
        }
        equal_arrays(sp.ux1, p.ux1, "ux1", bad);
        equal_arrays(sp.ux2, p.ux2, "ux2", bad);
        equal_arrays(sp.ux3, p.ux3, "ux3", bad);
        equal_arrays(sp.weight, p.weight, "weight", bad);
        equal_arrays(sp.tag, p.tag, "tag", bad);
        {
          auto ha = Kokkos::create_mirror_view(sp.pld_r);
          auto hb = Kokkos::create_mirror_view(p.pld_r);
          auto ia = Kokkos::create_mirror_view(sp.pld_i);
          auto ib = Kokkos::create_mirror_view(p.pld_i);
          Kokkos::deep_copy(ha, sp.pld_r);
          Kokkos::deep_copy(hb, p.pld_r);
          Kokkos::deep_copy(ia, sp.pld_i);
          Kokkos::deep_copy(ib, p.pld_i);
          bool okr = true, oki = true;
          for (npart_t s = 0; s < p.npart(); ++s) {
            okr = okr && ha(s, 0) == hb(s, 0) && ha(s, 1) == hb(s, 1);
            oki = oki && ia(s, 0) == ib(s, 0);
          }
          if (!okr) {
            bad += " pld_r";
          }
          if (!oki) {
            bad += " pld_i";
          }
        }
        check(bad.empty(), fmt::format("%dD C2: every particle array equals the written one exactly over [0, npart)%s", Di,
                                       bad.empty() ? "" : (": differs in" + bad).c_str()));
      }
      std::filesystem::remove_all(dir);
    }
#else
    std::printf("(diag) %dD: checkpoint round trip not compiled (output=OFF build)\n", Di);
#endif
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);
  int rc = 0;
  try {
    std::printf("f8_tracers: execution space %s, real_t = %s, sort path %s, tiled_deposit %s\n", Kokkos::DefaultExecutionSpace::name(),
                sizeof(real_t) == 8 ? "double" : "float",
#if (defined(CUDA_ENABLED) && defined(THRUST_ENABLED)) || (defined(HIP_ENABLED) && defined(ROCTHRUST_ENABLED)) || (defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED))
                "vendor",
#else
                "BinSort",
#endif
#if defined(TILED_DEPOSIT)
                "ON");
#else
                "OFF");
#endif
    run_dimension<Dim::_1D>();
    run_dimension<Dim::_2D>();
    run_dimension<Dim::_3D>();
    std::printf("f8_tracers summary: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0 || n_pass == 0) {
      rc = 1;
    }
  } catch (const std::exception& e) {
    std::printf("FAIL: f8_tracers: exception: %s\n", e.what());
    rc = 1;
  }
  GlobalFinalize();
  return rc;
}
