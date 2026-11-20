// Copy of dump_device_rates.cpp with the thread count of every threaded builder set to BH_THREADS
// (generated; the only differences from the original are the lines that carry BH_THREADS).
#ifndef BH_THREADS
#define BH_THREADS 0
#endif
// Oracle dump of the device lookup of the Bethe-Heitler interaction rate
// (arch::bh::mirror_tables_to_device and arch::bh::HadronicTables::J), the
// lookup that the emission policy uses for the probability of an interaction.
// Compiled once against the original headers with -DBH_HAS_QK_SOURCE and once
// against the reduced headers without it: the two outputs must be
// byte-identical. Usage: dump_device_rates > out.txt
#include "archetypes/emission/bethe_heitler/bh_born_qk.h"
#include "archetypes/emission/bethe_heitler/bh_physics.h"
#include "archetypes/emission/bethe_heitler/bh_tables_device.h"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

using namespace arch::bh;

namespace {

  std::uint64_t fnv(const void* data, std::size_t n, std::uint64_t h) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
      h ^= p[i];
      h *= 1099511628211ULL;
    }
    return h;
  }

  void dump_bath(const std::string& name, const BathSpec& spec, double mu_r, double gamma_p_max) {
    const auto bath = make_bath(spec);
    Quadrature q;
#if defined(BH_HAS_QK_SOURCE)
    q.qk_source = QkSource::Born;
#endif
    const BetheHeitlerRates  rates(bath, q);
    const HadronicTablesHost H = build_tables(rates, mu_r, gamma_p_max);
    validate_tables(H);
    const HadronicTables D = mirror_tables_to_device(H);
    std::printf("== bath %s mu_r %a gamma_p_max %a\n", name.c_str(), mu_r, gamma_p_max);
    std::printf("device gamma_th %a gamma_min %a gamma_max %a y_min %a y_max %a dy_fine %a dy_coarse %a n_fine %d n %d\n",
                D.gamma_th,
                D.gamma_min,
                D.gamma_max,
                D.y_min,
                D.y_max,
                D.dy_fine,
                D.dy_coarse,
                D.n_fine,
                D.n);
    // J on a dense set of Lorentz factors, from below the lower edge to above
    // the upper edge, uniform in log10(Gamma - Gamma_th)
    const int     n_points = 20001;
    const double  y_lo = D.y_min - 0.5, y_hi = D.y_max + 0.25;
    std::uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n_points; ++i) {
      const double y = y_lo + (y_hi - y_lo) * static_cast<double>(i) / (n_points - 1);
      const double g = D.gamma_th + std::pow(10.0, y);
      const double J = D.J(g);
      h              = fnv(&g, sizeof(g), h);
      h              = fnv(&J, sizeof(J), h);
      if (i % 1000 == 0) {
        std::printf("rate %d %a %a\n", i, g, J);
      }
    }
    std::printf("hash rate %016llx\n", (unsigned long long)h);
    // at the nodes and at the edges
    auto y_h = Kokkos::create_mirror_view(D.y);
    auto g_h = Kokkos::create_mirror_view(D.gamma);
    Kokkos::deep_copy(y_h, D.y);
    Kokkos::deep_copy(g_h, D.gamma);
    std::uint64_t hn = 1469598103934665603ULL;
    for (int i = 0; i < D.n; ++i) {
      const double J = D.J(g_h(i));
      hn             = fnv(&y_h(i), sizeof(double), hn);
      hn             = fnv(&g_h(i), sizeof(double), hn);
      hn             = fnv(&J, sizeof(J), hn);
    }
    std::printf("hash nodes %016llx edges %a %a %a %a\n",
                (unsigned long long)hn,
                D.J(D.gamma_min),
                D.J(std::nextafter(D.gamma_min, 0.0)),
                D.J(D.gamma_max),
                D.J(std::nextafter(D.gamma_max, 1.0e300)));
  }

} // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  born::table(BH_THREADS);
  {
    BathSpec mono05;
    mono05.type = "mono";
    mono05.eps0 = 0.5;
    dump_bath("mono05", mono05, 1836.15, 0.0);
    // the table of the unit test: short, ends inside the coarse part
    dump_bath("mono05_g12", mono05, 1836.15267343, 12.0);
    // a short table that ends close to the threshold
    dump_bath("mono05_g2p14", mono05, 1836.15, 2.14);
    BathSpec pl;
    pl.type    = "powerlaw";
    pl.eps_min = 0.1;
    pl.eps_max = 1.5;
    pl.index   = 2.0;
    dump_bath("powerlaw", pl, 1836.15, 0.0);
    BathSpec bb;
    bb.type    = "blackbody";
    bb.theta   = 0.1;
    bb.eps_min = 0.0;
    bb.eps_max = 1.5;
    dump_bath("blackbody", bb, 1836.15, 0.0);
  }
  Kokkos::finalize();
  return 0;
}
