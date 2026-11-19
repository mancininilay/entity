// V4: pair spectra of a bh_validation run against the frozen reference
// histograms, with the calibrated two-sample test of the fork's event-mode
// test E4 (tests/kernels/bh_event.cpp at dev/hadronic 72427ff6: same
// quantities, same reference files, same partition, same statistic, same
// seeds) and the Holm step-down rule over the family.
//
//   bhv_spectra <tests/data/hadronic> <eps0> {<Gamma_p: 3|10> <electrons.bin> <positrons.bin>}...
//
// The binaries are the dumps of pgens/bh_validation (species files of one
// rank; pair k is electron k and positron k; the emitters move along +x1).
// Compile against the headers of the fork copied beside this file:
//   clang++ -std=c++20 -O2 -I fork-72427ff6 bhv_spectra.cpp -o bhv_spectra
#include "archetypes/hadronic/bh_calibrated_test.h"
#include "archetypes/hadronic/bh_sampler_io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace bh  = arch::bh;
namespace cal = arch::bh::cal;

struct Species {
  std::uint64_t       n { 0 };
  std::vector<double> ux1, ux2, ux3, weight;
  std::vector<short>  tag;
};

static Species read_species(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    throw std::runtime_error("cannot open " + path);
  }
  std::uint64_t hdr[2];
  if (std::fread(hdr, sizeof(std::uint64_t), 2, f) != 2) {
    throw std::runtime_error("short header in " + path);
  }
  Species S;
  S.n = hdr[0];
  auto read_real = [&](std::vector<double>& v) {
    v.resize(S.n);
    if (hdr[1] == 4) {
      std::vector<float> b(S.n);
      if (std::fread(b.data(), 4, S.n, f) != S.n) {
        throw std::runtime_error("short data in " + path);
      }
      std::copy(b.begin(), b.end(), v.begin());
    } else {
      if (std::fread(v.data(), 8, S.n, f) != S.n) {
        throw std::runtime_error("short data in " + path);
      }
    }
  };
  read_real(S.ux1);
  read_real(S.ux2);
  read_real(S.ux3);
  read_real(S.weight);
  S.tag.resize(S.n);
  if (std::fread(S.tag.data(), sizeof(short), S.n, f) != S.n) {
    throw std::runtime_error("short tags in " + path);
  }
  std::fclose(f);
  return S;
}

int main(int argc, char* argv[]) {
  if (argc < 6 || (argc - 3) % 3 != 0) {
    std::fprintf(stderr, "usage: bhv_spectra <data dir> <eps0> {<3|10> <electrons.bin> <positrons.bin>}...\n");
    return 2;
  }
  const std::string data = argv[1];
  const double      eps0 = std::atof(argv[2]);
  struct Combo {
    std::string what;
    cal::Result r;
  };
  std::vector<Combo> combos;
  const char* const  qname[5] = { "E-", "E+", "cos-", "cos+", "D" };
  for (int c = 3; c < argc; c += 3) {
    const std::string gs = argv[c];
    const double      g  = std::atof(argv[c]);
    // the E4 case index: 0 for Gamma_p = 3, 1 for Gamma_p = 10 (it enters the seeds)
    const int case_index = g >= 10.0 ? 1 : 0;
    std::vector<std::vector<double>> edges(5), rn(5);
    auto read_ref = [&](const std::string& file, bool only_D) {
      const bh::FixtureContents F = bh::read_fixture_file(data + "/" + file);
      for (const auto& r : F.rows) {
        const int q = static_cast<int>(r[0]);
        if ((only_D && q != 4) || (!only_D && q == 4 && g >= 10.0)) {
          continue;
        }
        if (edges[q].empty()) {
          edges[q].push_back(r[2]);
        }
        edges[q].push_back(r[3]);
        rn[q].push_back(r[4]); // unweighted count
      }
    };
    read_ref("bhref-0.2.0/hist_mono05_G" + gs + ".txt", false);
    if (g >= 10.0) {
      read_ref("bhref-0.2.0-suppl1/hist_logD_mono05_G" + gs + ".txt", true);
    }
    double n_ref = 0.0;
    for (const double x : rn[0]) {
      n_ref += x;
    }
    const Species el = read_species(argv[c + 1]);
    const Species po = read_species(argv[c + 2]);
    if (el.n != po.n) {
      throw std::runtime_error("electron and positron counts differ");
    }
    std::vector<std::vector<double>> kn(5);
    for (int q = 0; q < 5; ++q) {
      kn[q].assign(edges[q].size() - 1, 0.0);
    }
    unsigned long long n_pairs = 0, outside = 0;
    double             sum_D = 0.0, sum_D2 = 0.0;
    for (std::uint64_t k = 0; k < el.n; ++k) {
      const double um2 = el.ux1[k] * el.ux1[k] + el.ux2[k] * el.ux2[k] + el.ux3[k] * el.ux3[k];
      const double up2 = po.ux1[k] * po.ux1[k] + po.ux2[k] * po.ux2[k] + po.ux3[k] * po.ux3[k];
      const double gm = std::sqrt(1.0 + um2), gp = std::sqrt(1.0 + up2);
      const double v[5] = { gm, gp, el.ux1[k] / std::sqrt(um2), po.ux1[k] / std::sqrt(up2), gm + gp - eps0 };
      for (int q = 0; q < 5; ++q) {
        const auto& e = edges[q];
        if (v[q] < e.front() || v[q] > e.back()) {
          ++outside;
        }
        int i = static_cast<int>(std::upper_bound(e.begin(), e.end(), v[q]) - e.begin()) - 1;
        i     = std::min(static_cast<int>(e.size()) - 2, std::max(0, i));
        kn[q][i] += 1.0;
      }
      sum_D  += v[4];
      sum_D2 += v[4] * v[4];
      ++n_pairs;
    }
    const double mD = sum_D / n_pairs;
    std::printf("Gamma_p %s: %llu pairs (reference %.0f events), values outside the reference's edges %llu, "
                "mean E- + E+ - eps0 = %.6f +- %.6f\n",
                gs.c_str(), n_pairs, n_ref, outside, mD, std::sqrt((sum_D2 / n_pairs - mD * mD) / n_pairs));
    std::printf("%s Gamma_p %s: at least 1e6 events (%llu)\n", n_pairs >= 1000000ULL ? "PASS" : "FAIL", gs.c_str(), n_pairs);
    for (int q = 0; q < 5; ++q) {
      const cal::Partition P = cal::merge_partition({ kn[q], rn[q] }, 100.0);
      const auto           a = cal::merge_sums(P, kn[q]);
      const auto           b = cal::merge_sums(P, rn[q]);
      const cal::Result    r = cal::calibrated(a, a, b, b, 2026102905ULL + 10ULL * case_index + static_cast<unsigned>(q));
      const std::string what = "Gamma_p " + gs + " " + qname[q] + (q == 4 && g >= 10.0 ? " (log bins)" : "");
      combos.push_back({ what, r });
      if (r.insufficient) {
        std::printf("  %s: INSUFFICIENT RESOLUTION (not a pass)\n", what.c_str());
      } else {
        std::printf("  %s: N_A %.0f N_B %.0f T_AD %.4e p %.5f q99 %.4e merged bins %zu\n", what.c_str(), r.ess_a, r.ess_b,
                    r.T_AD, r.p, r.q99, r.merged_bins);
      }
    }
  }
  // Holm step-down over the family at level 0.01: reject while p_(i) < 0.01 / (m - i + 1)
  const std::size_t        m = combos.size();
  std::vector<std::size_t> order(m);
  for (std::size_t i = 0; i < m; ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return combos[a].r.p < combos[b].r.p; });
  bool   family_pass = true, any_insufficient = false;
  double adj_prev    = 0.0, min_adj = 1.0;
  for (std::size_t i = 0; i < m; ++i) {
    const auto& cb = combos[order[i]];
    any_insufficient = any_insufficient || cb.r.insufficient;
    const double adj = std::max(adj_prev, std::min(1.0, static_cast<double>(m - i) * cb.r.p));
    adj_prev         = adj;
    min_adj          = std::min(min_adj, adj);
    std::printf("  Holm %zu: %-28s p %.5f adjusted %.5f\n", i + 1, cb.what.c_str(), cb.r.p, adj);
    if (adj < 0.01) {
      family_pass = false;
    }
  }
  int n_single = 0;
  for (const auto& cb : combos) {
    n_single += cb.r.pass ? 1 : 0;
  }
  std::printf("%s V4 family of %zu comparisons, Holm at 0.01: smallest adjusted p = %.5f%s\n",
              (family_pass && !any_insufficient) ? "PASS" : "FAIL", m, min_adj,
              any_insufficient ? " (a comparison has insufficient resolution)" : "");
  std::printf("(history) rule of test E4: %d of %zu comparisons with p >= 0.01 (>= 95 %% required)\n", n_single, m);
  return (family_pass && !any_insufficient) ? 0 : 1;
}
