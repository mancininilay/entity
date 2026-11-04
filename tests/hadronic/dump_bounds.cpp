/**
 * @file tests/hadronic/dump_bounds.cpp
 * @brief [fork:hadronic/P2 closing round, Codex 1's F1] the pair dump of bh_proto under bounds-checked views.
 *
 *        `write_pairs` iterates to max(npart-, npart+), so a read of the shorter side would run past that species'
 *        allocation when the capacities differ (Entity sizes each species on its own). This harness compiles the method
 *        verbatim: `dump_bounds_extract.py` copies its text out of pgens/bh_proto/pgen.hpp at build time (Codex 1's
 *        control of 2026-11-03 as a test), and with BH_DUMP_OLD_METHOD the frozen method of 22449c32 (the negative
 *        control, dump_bounds_method_22449c32.inc). The arrays are Kokkos host views with
 *        KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK defined here, so that an access outside an allocation aborts.
 *
 * Usage: test-hadronic-dump_bounds.xc <capacity e-> <npart e-> <capacity e+> <npart e+> <file>
 * The species are filled with a value per array and slot; the dump is read back and every present side must hold its own
 * values, every absent side tag -1 and zeros, the row count max(npart-, npart+). Verdicts "PASS"/"FAIL", exit 1 on a FAIL.
 */

#include <Kokkos_Macros.hpp>
#define KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK
#include <Kokkos_Core.hpp>

#include "global.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

  template <class T>
  using view1 = Kokkos::View<T*, Kokkos::HostSpace>;

  // the arrays `write_pairs` reads, with the element types of ntt::Particles
  struct Species {
    npart_t           n;
    view1<int>        i1, i1_prev;
    view1<prtldx_t>   dx1, dx1_prev;
    view1<real_t>     weight, ux1, ux2, ux3;
    view1<short>      tag;
    Kokkos::View<real_t**, Kokkos::HostSpace> pld_r;

    Species(npart_t capacity, npart_t npart)
      : n { npart }
      , i1 { "i1", capacity }
      , i1_prev { "i1_prev", capacity }
      , dx1 { "dx1", capacity }
      , dx1_prev { "dx1_prev", capacity }
      , weight { "weight", capacity }
      , ux1 { "ux1", capacity }
      , ux2 { "ux2", capacity }
      , ux3 { "ux3", capacity }
      , tag { "tag", capacity }
      , pld_r { "pld_r", capacity, 3 } {}

    auto npart() const -> npart_t {
      return n;
    }
  };

  template <class S, class M>
  struct Domain {
    std::vector<Species> species;
  };

  struct S {};

  struct M {};

  // the members of the generator that the method uses
  struct Dump {
    unsigned short pair_minus { 1 }, pair_plus { 2 };
    std::string    path;

    auto per_rank_name(const std::string&) const -> std::string {
      return path;
    }

#if defined(BH_DUMP_OLD_METHOD)
  #include "dump_bounds_method_22449c32.inc"
#else
  #include "dump_bounds_method.inc"
#endif
  };

  // the value of an array at a slot: side 0 (e-) or 1 (e+), array a, slot k; exact in float
  auto value(int side, int a, npart_t k) -> double {
    return 1.0 + 0.25 * a + 64.0 * side + 128.0 * static_cast<double>(k);
  }

  void fill(Species& sp, int side) {
    for (npart_t k = 0; k < sp.n; ++k) {
      sp.i1(k)       = static_cast<int>(value(side, 0, k));
      sp.dx1(k)      = static_cast<prtldx_t>(value(side, 1, k));
      sp.weight(k)   = static_cast<real_t>(value(side, 2, k));
      sp.ux1(k)      = static_cast<real_t>(value(side, 3, k));
      sp.ux2(k)      = static_cast<real_t>(value(side, 4, k));
      sp.ux3(k)      = static_cast<real_t>(value(side, 5, k));
      sp.tag(k)      = 1;
      sp.pld_r(k, 0) = static_cast<real_t>(value(side, 6, k));
      sp.pld_r(k, 1) = static_cast<real_t>(value(side, 7, k));
      sp.pld_r(k, 2) = static_cast<real_t>(value(side, 8, k));
      sp.i1_prev(k)  = static_cast<int>(value(side, 9, k));
      sp.dx1_prev(k) = static_cast<prtldx_t>(value(side, 10, k));
    }
  }

  // the columns of one side in a row of 25, with the array of `value` each holds
  struct Column {
    int col, array;
  };

  const std::vector<Column> SIDE_M { { 1, 0 },  { 2, 1 },  { 5, 2 },  { 7, 3 },  { 8, 4 },  { 9, 5 },
                                     { 15, 6 }, { 16, 7 }, { 17, 8 }, { 21, 9 }, { 22, 10 } };
  const std::vector<Column> SIDE_P { { 3, 0 },  { 4, 1 },  { 6, 2 },  { 10, 3 }, { 11, 4 }, { 12, 5 },
                                     { 18, 6 }, { 19, 7 }, { 20, 8 }, { 23, 9 }, { 24, 10 } };

  int n_pass = 0, n_fail = 0;

  void verdict(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    (ok ? n_pass : n_fail) += 1;
  }

  void check_side(const std::vector<std::vector<double>>& rows, const std::vector<Column>& cols, int tag_col, int side,
                  npart_t n, const std::string& name) {
    bool present = true, absent = true;
    for (std::size_t k = 0; k < rows.size(); ++k) {
      const auto& r = rows[k];
      if (k < n) {
        present = present && r[tag_col] == 1.0;
        for (const auto& c : cols) {
          // the integer arrays hold the truncated value
          const double v = value(side, c.array, static_cast<npart_t>(k));
          present        = present && r[c.col] == ((c.array == 0 || c.array == 9) ? std::floor(v) : v);
        }
      } else {
        absent = absent && r[tag_col] == -1.0;
        for (const auto& c : cols) {
          absent = absent && r[c.col] == 0.0;
        }
      }
    }
    verdict(present, name + ": every slot up to its own count " + std::to_string(n) + " holds its own values");
    verdict(absent, name + ": every absent slot holds tag -1 and zeros");
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  if (argc != 6) {
    std::fprintf(stderr, "usage: %s <capacity e-> <npart e-> <capacity e+> <npart e+> <file>\n", argv[0]);
    return 2;
  }
  Kokkos::initialize(argc, argv);
  {
    const auto cap_m = static_cast<npart_t>(std::atoi(argv[1])), n_m = static_cast<npart_t>(std::atoi(argv[2]));
    const auto cap_p = static_cast<npart_t>(std::atoi(argv[3])), n_p = static_cast<npart_t>(std::atoi(argv[4]));
    Domain<S, M> domain;
    domain.species.emplace_back(cap_m, n_m);
    domain.species.emplace_back(cap_p, n_p);
    fill(domain.species[0], 0);
    fill(domain.species[1], 1);
    Dump dump;
    dump.path = argv[5];
    dump.write_pairs(domain);

    std::vector<std::vector<double>> rows;
    std::ifstream                    in(argv[5]);
    std::string                      line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::istringstream  ss(line);
      std::vector<double> row;
      double              v;
      while (ss >> v) {
        row.push_back(v);
      }
      rows.push_back(row);
    }
    bool widths = true;
    for (const auto& r : rows) {
      widths = widths && r.size() == 25;
    }
    verdict(widths, "every row has 25 columns");
    verdict(rows.size() == std::max(n_m, n_p), "rows " + std::to_string(rows.size()) + " = max(npart-, npart+)");
    if (widths) {
      check_side(rows, SIDE_M, 13, 0, n_m, "e-");
      check_side(rows, SIDE_P, 14, 1, n_p, "e+");
    }
  }
  Kokkos::finalize();
  std::printf("dump_bounds summary: %d passed, %d failed\n", n_pass, n_fail);
  return n_fail > 0 ? 1 : 0;
}
