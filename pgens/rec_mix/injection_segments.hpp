// P13-D43, Codex 1: geometry and counting for an opt-in segmented injector.
#ifndef RECMIX_INJECTION_SEGMENTS_HPP
#define RECMIX_INJECTION_SEGMENTS_HPP

#include "enums.h"
#include "global.h"
#include "arch/kokkos_aliases.h"
#include <cmath>
#include <random>

namespace user::recmix {
  // A segment must be wholly owned by one domain, with global origin zero.
  inline auto segments_aligned(int offset, int length, int width) -> bool {
    return width > 0 and offset >= 0 and length > 0 and
           offset % width == 0 and length % width == 0;
  }

  struct CountSegments_kernel {
    array_t<int*> i1, i2;
    array_t<short*> tag;
    array_t<unsigned long long*> counts;
    int ja, jb, width, n1;

    Inline void operator()(npart_t p) const {
      if (tag(p) == ntt::ParticleTag::alive and i2(p) >= ja and i2(p) < jb and
          i1(p) >= 0 and i1(p) < n1) {
        Kokkos::atomic_add(&counts(i1(p) / width), 1ULL);
      }
    }
  };

  // Exclusive cumulative ends; repeated ends are empty segments.
  Inline auto segment_for_trial(npart_t p, const array_t<npart_t*>& ends) -> int {
    int lo = 0, hi = static_cast<int>(ends.extent(0));
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2;
      if (p < ends(mid)) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  inline auto segment_pair_deficit(unsigned long long count, int rows, int width,
                                   double ppc0) -> double {
    const double deficit = ppc0 - static_cast<double>(count) /
                                    (static_cast<double>(rows) * width);
    return deficit > 0.0 ? 0.5 * deficit : 0.0;
  }

  // Unchanged Tristan rule, shared by the default and segmented paths.
  inline auto tristan_round(double n, std::mt19937_64& rng) -> std::size_t {
    if (n <= 0.0) {
      return 0;
    }
    if (n < 10.0) {
      std::poisson_distribution<long long> pd(n);
      return static_cast<std::size_t>(pd(rng));
    }
    return static_cast<std::size_t>(std::ceil(n));
  }
}
#endif
