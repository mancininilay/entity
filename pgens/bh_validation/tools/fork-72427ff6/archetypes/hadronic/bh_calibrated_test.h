/**
 * @file archetypes/hadronic/bh_calibrated_test.h
 * @brief The calibrated weighted two-sample histogram test of the step-4
 *        comparisons (Claude 3's F8; specification: Codex 1,
 *        `tests/data/hadronic/bhref-0.2.0-suppl1/COMPARISON.md`, implemented
 *        as written there): the common partition of the bins, the right-edge
 *        discrete Anderson-Darling statistic and its conditional
 *        Gaussian-multiplier null.
 *
 * Partition (`merge_partition`): adjacent bins are merged from left to right
 * until every one of the given accumulated counts reaches `min_count` (100 in
 * COMPARISON.md: the C++ unweighted events, the reference's unweighted events,
 * the C++ weighted packets, the reference's weighted packets); the next merged
 * bin starts there; an incomplete final tail is joined to its previous merged
 * bin; when only one merged bin remains the comparison has insufficient
 * resolution (not a pass). The same partition serves both chain comparisons.
 *
 * Statistic (`calibrated`), per chain, with a_j, b_j the merged sums of weights
 * of the two samples and a2_j, b2_j their merged sums of squared weights (for
 * unweighted events both equal the count; for weighted packets w = 1/D and
 * a2_j = sum w_i^2, not (sum w_i)^2):
 *   A = sum a, B = sum b, p = a/A, q = b/B, v = a2/A^2, u = b2/B^2,
 *   h = (p + q)/2, H = cumsum(h), delta = cumsum(p - q),
 *   J = {j : 0 < H_j < 1 - 1e-12},
 *   T_AD = sum_{j in J} delta_j^2 h_j / [H_j (1 - H_j)];
 * null: per replicate z_j ~ N(0, v_j), y_j ~ N(0, u_j) independently, centred
 * zc = z - p sum z, yc = y - q sum y, delta* = cumsum(zc - yc), T* as T_AD with
 * delta*; 19,999 replicates; p = (1 + #{T* >= T_AD})/20,000; pass iff
 * p >= 0.01. The empirical 99th percentile of T* (NumPy's default linear
 * interpolation) is a diagnostic. ESS_A = A^2/sum a2, ESS_B = B^2/sum b2.
 *
 * Gaussian generator (stated; the method, not the reference's NumPy PCG64
 * stream, is reproduced): std::mt19937_64 seeded with the integer seed,
 * uniforms u = ((x >> 11) + 0.5) 2^-53 in (0, 1), standard normals by the
 * Box-Muller transform, both members of each pair used (cos first, then sin);
 * per replicate the full z vector is drawn, then the full y vector.
 * std::mt19937_64 is fully specified by the C++ standard, so the stream (and
 * the p-values) are the same with every standard library.
 *
 * Host-only, double precision, Kokkos-free.
 *
 * @namespaces:
 *   - arch::bh::cal::
 */

#ifndef ARCHETYPES_HADRONIC_BH_CALIBRATED_TEST_H
#define ARCHETYPES_HADRONIC_BH_CALIBRATED_TEST_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace arch::bh::cal {

  /// [first, last] bin index ranges of the merged bins
  struct Partition {
    std::vector<std::size_t> first, last;

    std::size_t size() const {
      return first.size();
    }
  };

  /**
   * @brief The common partition: merge adjacent bins from left to right until
   *        every count series has accumulated at least min_count; join an
   *        incomplete final tail to its previous merged bin
   * @param counts the count series (all of the same length, the number of bins)
   */
  inline Partition merge_partition(const std::vector<std::vector<double>>& counts, double min_count = 100.0) {
    if (counts.empty()) {
      throw std::invalid_argument("merge_partition: no count series");
    }
    const std::size_t n = counts.front().size();
    for (const auto& c : counts) {
      if (c.size() != n) {
        throw std::invalid_argument("merge_partition: count series of different lengths");
      }
    }
    Partition           P;
    std::vector<double> acc(counts.size(), 0.0);
    std::size_t         start = 0;
    for (std::size_t i = 0; i < n; ++i) {
      bool all = true;
      for (std::size_t s = 0; s < counts.size(); ++s) {
        acc[s] += counts[s][i];
        all = all && acc[s] >= min_count;
      }
      if (all) {
        P.first.push_back(start);
        P.last.push_back(i);
        std::fill(acc.begin(), acc.end(), 0.0);
        start = i + 1;
      }
    }
    if (start < n) {
      if (P.size() > 0) {
        P.last.back() = n - 1; // the incomplete tail joins its previous merged bin
      } else {
        P.first.push_back(0); // a single merged bin: insufficient resolution
        P.last.push_back(n - 1);
      }
    }
    return P;
  }

  /// sums of a per-bin series over the merged bins
  inline std::vector<double> merge_sums(const Partition& P, const std::vector<double>& x) {
    std::vector<double> out(P.size(), 0.0);
    for (std::size_t m = 0; m < P.size(); ++m) {
      for (std::size_t i = P.first[m]; i <= P.last[m]; ++i) {
        out[m] += x[i];
      }
    }
    return out;
  }

  struct Result {
    bool        insufficient { false }; // one merged bin: insufficient resolution, not a pass
    std::size_t merged_bins { 0 };
    double      ess_a { 0.0 }, ess_b { 0.0 };
    double      T_AD { 0.0 }, p { 0.0 }, q99 { 0.0 };
    int         replicates { 0 };
    bool        pass { false };
  };

  /// standard normals: std::mt19937_64, 53-bit uniforms in (0, 1), Box-Muller (cos, then sin)
  class Normal {
  public:
    explicit Normal(std::uint64_t seed)
      : m_gen { seed } {}

    double operator()() {
      if (m_have) {
        m_have = false;
        return m_next;
      }
      const double u1 = uniform(), u2 = uniform();
      const double r  = std::sqrt(-2.0 * std::log(u1));
      const double t  = 6.283185307179586476925286766559 * u2;
      m_next          = r * std::sin(t);
      m_have          = true;
      return r * std::cos(t);
    }

  private:
    double uniform() {
      return (static_cast<double>(m_gen() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    }

    std::mt19937_64 m_gen;
    bool            m_have { false };
    double          m_next { 0.0 };
  };

  /**
   * @brief The calibrated statistic of one chain on merged bins
   * @param a, a2 the first sample's merged sums of weights and of squared weights
   * @param b, b2 the second sample's
   * @param seed the null-calibration seed
   */
  inline Result calibrated(const std::vector<double>& a, const std::vector<double>& a2, const std::vector<double>& b,
                           const std::vector<double>& b2, std::uint64_t seed, int replicates = 19999,
                           double alpha = 0.01) {
    const std::size_t n = a.size();
    if (a2.size() != n || b.size() != n || b2.size() != n) {
      throw std::invalid_argument("calibrated: series of different lengths");
    }
    Result R;
    R.merged_bins = n;
    R.replicates  = replicates;
    double A = 0.0, B = 0.0, SA2 = 0.0, SB2 = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      A += a[j];
      B += b[j];
      SA2 += a2[j];
      SB2 += b2[j];
    }
    R.ess_a = SA2 > 0.0 ? A * A / SA2 : 0.0;
    R.ess_b = SB2 > 0.0 ? B * B / SB2 : 0.0;
    if (n <= 1) {
      R.insufficient = true;
      return R;
    }
    if (!(A > 0.0) || !(B > 0.0)) {
      throw std::invalid_argument("calibrated: empty sample");
    }
    std::vector<double> p(n), q(n), sv(n), su(n), h(n), H(n), wgt(n, 0.0);
    double              cum = 0.0, dcum = 0.0, T = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      p[j]  = a[j] / A;
      q[j]  = b[j] / B;
      sv[j] = std::sqrt(a2[j]) / A; // sqrt(v_j), v_j = a2_j / A^2
      su[j] = std::sqrt(b2[j]) / B; // sqrt(u_j)
      h[j]  = 0.5 * (p[j] + q[j]);
      cum += h[j];
      H[j] = cum;
      dcum += p[j] - q[j];
      if (H[j] > 0.0 && H[j] < 1.0 - 1.0e-12) {
        wgt[j] = h[j] / (H[j] * (1.0 - H[j]));
        T += dcum * dcum * wgt[j];
      }
    }
    R.T_AD = T;
    Normal              N(seed);
    std::vector<double> z(n), y(n), Ts(static_cast<std::size_t>(replicates));
    std::size_t         n_ge = 0;
    for (int r = 0; r < replicates; ++r) {
      double sz = 0.0, sy = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        z[j] = sv[j] * N();
        sz += z[j];
      }
      for (std::size_t j = 0; j < n; ++j) {
        y[j] = su[j] * N();
        sy += y[j];
      }
      double d = 0.0, Tr = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        d += (z[j] - p[j] * sz) - (y[j] - q[j] * sy);
        Tr += d * d * wgt[j];
      }
      Ts[static_cast<std::size_t>(r)] = Tr;
      if (Tr >= T) {
        ++n_ge;
      }
    }
    R.p = (1.0 + static_cast<double>(n_ge)) / (static_cast<double>(replicates) + 1.0);
    std::sort(Ts.begin(), Ts.end());
    // NumPy's default (linear) percentile: position 0.99 (n - 1)
    const double      pos = 0.99 * static_cast<double>(replicates - 1);
    const std::size_t lo  = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi  = std::min(lo + 1, Ts.size() - 1);
    R.q99                 = Ts[lo] + (pos - static_cast<double>(lo)) * (Ts[hi] - Ts[lo]);
    R.pass                = R.p >= alpha;
    return R;
  }

  /// the null-calibration seed rule of COMPARISON.md: 2026102701 + 10 case + 5 chain + quantity
  inline std::uint64_t comparison_seed(int case_index, int chain, int quantity) {
    return 2026102701ULL + 10ULL * static_cast<std::uint64_t>(case_index) + 5ULL * static_cast<std::uint64_t>(chain) +
           static_cast<std::uint64_t>(quantity);
  }

} // namespace arch::bh::cal

#endif // ARCHETYPES_HADRONIC_BH_CALIBRATED_TEST_H
