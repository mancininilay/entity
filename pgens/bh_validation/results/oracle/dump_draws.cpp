// Oracle dump of the draws of the device sampler of the Bethe-Heitler pair
// (arch::bh::SamplerTables::draw, unweighted law), with a deterministic
// generator. Compiled once against the original headers with
//   -DBH_HAS_QK_SOURCE -DBH_DRAW_OLD_SIGNATURE
// and once against the reduced headers without the two switches: the two
// outputs must be byte-identical. Usage: dump_draws > out.txt
//
//   BH_HAS_QK_SOURCE        the headers still have Quadrature::qk_source; the
//                           Born source is selected explicitly (as the consumer
//                           does); without it Born is the only source
//   BH_DRAW_OLD_SIGNATURE   draw(gamma, false, gen, n_rng, 1, out) instead of
//                           draw(gamma, gen, out)
#include "archetypes/emission/bethe_heitler/bh_born_qk.h"
#include "archetypes/emission/bethe_heitler/bh_physics.h"
#include "archetypes/emission/bethe_heitler/bh_sampler_device.h"

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace arch::bh;

namespace {

  // splitmix64 -> uniform in [0, 1)
  struct Generator {
    std::uint64_t state;

    explicit Generator(std::uint64_t seed) : state { seed } {}

    double drand() {
      state           += 0x9E3779B97F4A7C15ULL;
      std::uint64_t z  = state;
      z                = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z                = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z                = z ^ (z >> 31);
      return static_cast<double>(z >> 11) * 0x1.0p-53;
    }
  };

  std::uint64_t fnv(const void* data, std::size_t n, std::uint64_t h) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) {
      h ^= p[i];
      h *= 1099511628211ULL;
    }
    return h;
  }

  constexpr int N_DRAWS = 20000;

  void dump_gamma(const char*          tag,
                  const std::string&   name,
                  const SamplerTables& sampler,
                  double               gamma,
                  std::uint64_t        seed) {
    Generator     gen { seed };
    std::uint64_t h = 1469598103934665603ULL;
    double        first[3][16];
#if defined(BH_DRAW_OLD_SIGNATURE)
    unsigned long long n_rng = 0;
#endif
    for (int n = 0; n < N_DRAWS; ++n) {
      SampledPair out;
#if defined(BH_DRAW_OLD_SIGNATURE)
      const bool ok = sampler.draw(gamma, false, gen, n_rng, 1, out);
      if (!ok) {
        std::fprintf(stderr, "dump_draws: the unweighted draw returned false\n");
        std::exit(2);
      }
#else
      sampler.draw(gamma, gen, out);
#endif
      const double v[16] = { out.eps,   out.k,     out.Em,     out.Ep,
                             out.E_m,   out.E_p,   out.pm[0],  out.pm[1],
                             out.pm[2], out.pp[0], out.pp[1],  out.pp[2],
                             out.kph[0], out.kph[1], out.kph[2], out.D };
      h = fnv(v, sizeof(v), h);
      if (n < 3) {
        for (int c = 0; c < 16; ++c) {
          first[n][c] = v[c];
        }
      }
    }
#if defined(BH_DRAW_OLD_SIGNATURE)
    // the unweighted law consumes eight uniform numbers per draw
    if (n_rng != 8ULL * N_DRAWS) {
      std::fprintf(stderr, "dump_draws: %llu random numbers for %d draws\n", n_rng, N_DRAWS);
      std::exit(2);
    }
#endif
    std::printf("%s bath %s gamma %a draws %d hash %016llx\n",
                tag,
                name.c_str(),
                gamma,
                N_DRAWS,
                (unsigned long long)h);
    for (int n = 0; n < 3; ++n) {
      std::printf("draw %d", n);
      for (int c = 0; c < 16; ++c) {
        std::printf(" %a", first[n][c]);
      }
      std::printf("\n");
    }
  }

  void dump_bath(const std::string&         name,
                 const BathSpec&            spec,
                 std::uint64_t              bath_index,
                 const std::vector<double>& gammas,
                 const std::vector<double>& extra_gammas) {
    const double mu_r        = 1836.15;
    const double gamma_p_max = 0.0;
    const auto   bath        = make_bath(spec);
    Quadrature   q;
#if defined(BH_HAS_QK_SOURCE)
    q.qk_source = QkSource::Born;
#endif
    const BetheHeitlerRates  rates(bath, q);
    const HadronicTablesHost H = build_tables(rates, mu_r, gamma_p_max);
    validate_tables(H);
    const SamplerTables sampler = build_sampler_tables(*bath, H);
    std::printf("== bath %s mu_r %a gamma_p_max %a\n", name.c_str(), mu_r, gamma_p_max);
    std::printf("tables n_k %d n_s %d n_tau %d n_cum %d mono %d n_gamma %d n_eps %d\n",
                sampler.n_k,
                sampler.n_s,
                sampler.n_tau,
                sampler.n_cum,
                (int)sampler.mono,
                sampler.n_gamma,
                sampler.n_eps);
    for (std::size_t i = 0; i < gammas.size(); ++i) {
      dump_gamma("==", name, sampler, gammas[i], 1000ULL * (bath_index + 1ULL) + i);
    }
    // additional coverage (near and below the lower edge, above the upper edge)
    for (std::size_t i = 0; i < extra_gammas.size(); ++i) {
      dump_gamma("== extra", name, sampler, extra_gammas[i], 1000ULL * (bath_index + 1ULL) + 500ULL + i);
    }
  }

} // namespace

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const std::vector<double> gammas { 2.2, 3.0, 6.0, 10.0, 30.0, 80.0 };

    BathSpec mono05;
    mono05.type = "mono";
    mono05.eps0 = 0.5;
    dump_bath("mono05", mono05, 0, gammas, { 2.1251, 2.13, 91.0, 200.0 });

    BathSpec pl;
    pl.type    = "powerlaw";
    pl.eps_min = 0.1;
    pl.eps_max = 1.5;
    pl.index   = 2.0;
    dump_bath("powerlaw", pl, 1, gammas, { 1.042, 1.05, 1.2, 10.025, 30.5 });

    // not required by the comparison: a third spectrum for coverage
    BathSpec bb;
    bb.type    = "blackbody";
    bb.theta   = 0.1;
    bb.eps_min = 0.0;
    bb.eps_max = 1.5;
    dump_bath("blackbody", bb, 2, {}, { 1.05, 1.2, 3.0, 10.0, 30.0 });
  }
  Kokkos::finalize();
  return 0;
}
