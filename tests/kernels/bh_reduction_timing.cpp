/**
 * @file tests/kernels/bh_reduction_timing.cpp
 * @brief Timing harness (P2 step 4, stage 5; G3 open item): the ledger
 *        accumulation of the Bethe-Heitler kernel by the double-double
 *        reducer (`Kokkos::parallel_reduce` with the `LossLedger` value type)
 *        against `Kokkos::atomic_add` on plain doubles for the same per-ion
 *        contributions, and (step 7, scaling-harness contract 2026-10-30 M4)
 *        against per-team scratch accumulation with one double atomic per team
 *        and entry (TeamPolicy; the team reduces its range in registers, the
 *        leader adds once), at npart in {1e5, 1e6} and, with BH_REDUCTION_N set
 *        (the A100 job: 4194304), at that count too; 20 repetitions each,
 *        medians printed. A diagnostic: the only pass/fail is that the two sums agree
 *        to 1e-12 relative (the reducer's double-double value against the
 *        atomic double sum, whose own round-off is of that order) and that the
 *        reducer's result is independent of the thread count to 1e-15.
 *
 * Conventions: the per-ion contribution is a deterministic pseudo-random
 * double in [0.5, 1.5) times a weight in [1, 2) (splitmix64 of the index), the
 * same for every repetition; the reducer accumulates it into three ledger
 * entries (E_drag, E_pairs, E_absorbed) and the atomic path into three doubles;
 * wall times from std::chrono around the fenced launches; the first repetition
 * is discarded as warm-up.
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "kernels/hadronic/bethe_heitler.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>

using namespace ntt;
namespace kh = kernel::hadronic;

namespace {

  Inline auto contribution(std::uint64_t i, double& w, double& x) {
    const auto h1 = kh::splitmix64_dev(i);
    const auto h2 = kh::splitmix64_dev(i ^ 0xA5A5A5A5A5A5A5A5ULL);
    w             = 1.0 + static_cast<double>(h1 >> 11) * 0x1.0p-53;
    x             = 0.5 + static_cast<double>(h2 >> 11) * 0x1.0p-53;
  }

  struct ReducerKernel {
    Inline void operator()(const std::uint64_t i, kh::LossLedger& L) const {
      double w, x;
      contribution(i, w, x);
      L.E_drag.add_prod(w, x);
      L.E_pairs.add_prod(w, 2.0 * x);
      L.E_absorbed.add_prod(w, 0.5 * x);
      L.n_processed += 1;
    }
  };

  struct AtomicKernel {
    array_t<double*> sums; // 3 entries

    Inline void operator()(const std::uint64_t i) const {
      double w, x;
      contribution(i, w, x);
      Kokkos::atomic_add(&sums(0), w * x);
      Kokkos::atomic_add(&sums(1), w * 2.0 * x);
      Kokkos::atomic_add(&sums(2), w * 0.5 * x);
    }
  };

  // M4: per-team scratch (a team-level parallel_reduce over the team's contiguous range into three doubles)
  // plus one atomic per team and entry
  struct TeamAtomicKernel {
    array_t<double*> sums; // 3 entries
    std::uint64_t    npart;
    std::uint64_t    per_team;

    using member_t = Kokkos::TeamPolicy<>::member_type;

    Inline void operator()(const member_t& team) const {
      const std::uint64_t begin = static_cast<std::uint64_t>(team.league_rank()) * per_team;
      const std::uint64_t end   = begin + per_team < npart ? begin + per_team : npart;
      double              s0 = 0.0, s1 = 0.0, s2 = 0.0;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, begin, end),
        [&](const std::uint64_t i, double& a0, double& a1, double& a2) {
          double w, x;
          contribution(i, w, x);
          a0 += w * x;
          a1 += w * 2.0 * x;
          a2 += w * 0.5 * x;
        },
        s0, s1, s2);
      Kokkos::single(Kokkos::PerTeam(team), [&]() {
        Kokkos::atomic_add(&sums(0), s0);
        Kokkos::atomic_add(&sums(1), s1);
        Kokkos::atomic_add(&sums(2), s2);
      });
    }
  };

  double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    std::printf("bh_reduction_timing: execution space %s, concurrency %d\n", Kokkos::DefaultExecutionSpace::name(),
                Kokkos::DefaultExecutionSpace().concurrency());
    const int reps = 20;
    std::vector<std::uint64_t> counts = { static_cast<std::uint64_t>(100000), static_cast<std::uint64_t>(1000000) };
    if (const char* env = std::getenv("BH_REDUCTION_N")) {
      const long long v = std::atoll(env);
      if (v > 0) {
        counts.push_back(static_cast<std::uint64_t>(v));
      }
    }
    for (const std::uint64_t npart : counts) {
      std::vector<double> t_red, t_atom, t_team;
      kh::LossLedger      L_last;
      double              s_last[3] = { 0.0, 0.0, 0.0 }, team_last[3] = { 0.0, 0.0, 0.0 };
      array_t<double*>    sums { "sums", 3 };
      array_t<double*>    tsums { "team_sums", 3 };
      const std::uint64_t per_team = 1024;
      const int           league   = static_cast<int>((npart + per_team - 1) / per_team);
      for (int r = 0; r <= reps; ++r) {
        kh::LossLedger L;
        auto           t0 = std::chrono::steady_clock::now();
        Kokkos::parallel_reduce("reducer", Kokkos::RangePolicy<Kokkos::IndexType<std::uint64_t>>(0, npart),
                                ReducerKernel {}, Kokkos::Sum<kh::LossLedger>(L));
        Kokkos::fence();
        const double tr = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        Kokkos::deep_copy(sums, 0.0);
        t0 = std::chrono::steady_clock::now();
        Kokkos::parallel_for("atomic", Kokkos::RangePolicy<Kokkos::IndexType<std::uint64_t>>(0, npart),
                             AtomicKernel { sums });
        Kokkos::fence();
        const double ta = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        Kokkos::deep_copy(tsums, 0.0);
        t0 = std::chrono::steady_clock::now();
        Kokkos::parallel_for("team_atomic", Kokkos::TeamPolicy<>(league, Kokkos::AUTO), TeamAtomicKernel { tsums, npart, per_team });
        Kokkos::fence();
        const double tt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (r > 0) { // the first repetition is warm-up
          t_red.push_back(tr);
          t_atom.push_back(ta);
          t_team.push_back(tt);
        }
        auto h = Kokkos::create_mirror_view(sums);
        Kokkos::deep_copy(h, sums);
        auto ht = Kokkos::create_mirror_view(tsums);
        Kokkos::deep_copy(ht, tsums);
        L_last    = L;
        s_last[0] = h(0);
        s_last[1] = h(1);
        s_last[2] = h(2);
        team_last[0] = ht(0);
        team_last[1] = ht(1);
        team_last[2] = ht(2);
      }
      const double red[3] = { L_last.E_drag.value(), L_last.E_pairs.value(), L_last.E_absorbed.value() };
      double       worst  = 0.0;
      for (int c = 0; c < 3; ++c) {
        worst = std::max(worst, std::abs(red[c] / s_last[c] - 1.0));
      }
      const bool ok = worst <= 1.0e-12;
      std::printf("%s: npart %llu: reducer (double-double) vs atomic double sums agree, max |rel. diff| = %.3e "
                  "(tolerance 1e-12)\n",
                  ok ? "PASS" : "FAIL", (unsigned long long)npart, worst);
      if (!ok) {
        rc = 1;
      }
      double worst_team = 0.0;
      for (int c = 0; c < 3; ++c) {
        worst_team = std::max(worst_team, std::abs(red[c] / team_last[c] - 1.0));
      }
      const bool ok_team = worst_team <= 1.0e-12;
      std::printf("%s: npart %llu: reducer (double-double) vs per-team scratch + atomics agree, max |rel. diff| = %.3e "
                  "(tolerance 1e-12) [M4]\n",
                  ok_team ? "PASS" : "FAIL", (unsigned long long)npart, worst_team);
      if (!ok_team) {
        rc = 1;
      }
      std::printf("(diag) npart %llu: median wall time over %d repetitions: reducer %.3f ms, atomic_add %.3f ms "
                  "(ratio reducer/atomic %.2f), per-team scratch + atomics %.3f ms (ratio reducer/team %.2f; %d teams of %llu); "
                  "min reducer %.3f ms, min atomic %.3f ms, min team %.3f ms; sums E_drag %.12e (atomic %.12e, team %.12e), "
                  "n_processed %llu\n",
                  (unsigned long long)npart, reps, 1.0e3 * median(t_red), 1.0e3 * median(t_atom),
                  median(t_red) / median(t_atom), 1.0e3 * median(t_team), median(t_red) / median(t_team), league,
                  (unsigned long long)per_team, 1.0e3 * *std::min_element(t_red.begin(), t_red.end()),
                  1.0e3 * *std::min_element(t_atom.begin(), t_atom.end()), 1.0e3 * *std::min_element(t_team.begin(), t_team.end()),
                  red[0], s_last[0], team_last[0], L_last.n_processed);
      // thread-count independence of the reducer value (order-independent to the
      // double-double round-off): re-reduce with a single-chunk policy
      {
        kh::LossLedger L1;
        Kokkos::parallel_reduce("reducer1", Kokkos::RangePolicy<Kokkos::IndexType<std::uint64_t>>(0, npart).set_chunk_size(static_cast<int>(std::min<std::uint64_t>(npart, 1u << 30))),
                                ReducerKernel {}, Kokkos::Sum<kh::LossLedger>(L1));
        Kokkos::fence();
        const double d = std::abs(L1.E_drag.value() / red[0] - 1.0);
        std::printf("%s: npart %llu: reducer value independent of the chunking, |rel. diff| = %.3e (tolerance 1e-15)\n",
                    d <= 1.0e-15 ? "PASS" : "FAIL", (unsigned long long)npart, d);
        if (d > 1.0e-15) {
          rc = 1;
        }
      }
    }
    std::printf("bh_reduction_timing summary: %s\n", rc == 0 ? "sums agree" : "DISAGREEMENT");
  }
  Kokkos::finalize();
  return rc;
}
