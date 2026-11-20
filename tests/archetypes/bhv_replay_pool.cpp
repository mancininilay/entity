/**
 * Replay on the host of the random streams of a Kokkos XorShift64 pool
 * through the Bethe-Heitler policy's decision and sampler.
 *
 * A pool of `n_states` generators is seeded as Kokkos does
 * (`Random_XorShift64_Pool::init_impl`). Every use of a generator by the
 * policy is one trial: one uniform number against the probability per step,
 * and for an interaction eight more for the outcome. The program walks the
 * first `n_trials` trials of every generator, for emitters of one Lorentz
 * factor moving along +x1 (the V4 runs), and
 *   - prints the statistics of the outcomes in windows of the trial index;
 *   - writes the outcomes (state, trial, lepton momenta along the emitter)
 *     to a binary file, to be matched against the pairs of a run.
 *
 * On a GPU the pool has maxThreadsPerMultiProcessor * multiProcessorCount
 * generators (221184 on an A100) and each is used in the order in which the
 * threads take it, so the pairs of the first step of a run on the GPU are the
 * outcomes of the first trials of these generators.
 *
 *   bhv-replay-pool.xc <gamma_p> <gamma_max> <optical_depth> <dt> <n_states>
 *                      <n_trials> <window> <seed (hex) | default> [out.bin]
 *
 * With n_states = 0 it prints instead the tabulated mean loss at the Lorentz
 * factors read from the standard input.
 *
 * Not a test: built, not registered.
 */

#include "global.h"

#include "utils/numeric.h"
#include "utils/param_container.h"

#include "archetypes/emission/bethe_heitler.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ntt;

auto main(int argc, char* argv[]) -> int {
  if (argc < 9) {
    std::fprintf(stderr,
                 "usage: bhv-replay-pool.xc <gamma_p> <gamma_max> "
                 "<optical_depth> <dt> <n_states> <n_trials> <window> <seed "
                 "(hex) | default> [out.bin]\n");
    return 2;
  }
  GlobalInitialize(argc, argv);
  int rc = 0;
  {
    const double gamma_p_in    = std::atof(argv[1]);
    const double gamma_max     = std::atof(argv[2]);
    const double optical_depth = std::atof(argv[3]);
    const double dt            = std::atof(argv[4]);
    const auto   n_states    = static_cast<std::uint64_t>(std::atoll(argv[5]));
    const int    n_trials    = std::atoi(argv[6]);
    const int    window      = std::atoi(argv[7]);
    const std::uint64_t seed = std::string(argv[8]) == "default"
                                 ? constant::RandomSeed
                                 : std::strtoull(argv[8], nullptr, 16);
    const double        mass_ratio = 1836.15;

    prm::Parameters   params;
    const std::string prefix = "radiation.emission.bethe_heitler.";
    params.set(prefix + "electron_species", static_cast<spidx_t>(1));
    params.set(prefix + "positron_species", static_cast<spidx_t>(2));
    params.set(prefix + "optical_depth", static_cast<real_t>(optical_depth));
    params.set(prefix + "mass_ratio", static_cast<real_t>(mass_ratio));
    params.set(prefix + "pair_weight", ONE);
    params.set(prefix + "max_probability", static_cast<real_t>(0.1));
    params.set(prefix + "max_loss_fraction", static_cast<real_t>(0.04));
    params.set(prefix + "gamma_max", static_cast<real_t>(gamma_max));
    params.set(prefix + "photon_spectrum", std::string("mono"));
    params.set(prefix + "photon_energy", static_cast<real_t>(0.5));
    params.set(prefix + "photon_energy_min", ZERO);
    params.set(prefix + "photon_energy_max", ZERO);
    params.set(prefix + "photon_index", ZERO);
    params.set(prefix + "photon_temperature", ZERO);
    params.set("particles.use_weights", false);
    auto& tables = arch::emission::bh::GetTables(params, static_cast<real_t>(dt));

    if (n_states == 0) {
      // table mode: the tabulated mean loss at the Lorentz factors read from
      // the standard input
      double g;
      while (std::scanf("%lf", &g) == 1) {
        std::printf("%.17g %.17g\n", g, tables.mean_loss(g));
      }
      GlobalFinalize();
      return 0;
    }

    // the Lorentz factor as the policy computes it from the momentum that the
    // problem generator sets (emitters of mass 1)
    const double mu_d        = 1.0;
    const double K0          = mass_ratio * (gamma_p_in - 1.0);
    const double gamma0      = 1.0 + K0 / mu_d;
    const double u           = std::sqrt((gamma0 - 1.0) * (gamma0 + 1.0));
    const double u_sqr       = u * u;
    const double gamma       = std::sqrt(1.0 + u_sqr);
    const double K           = mu_d * u_sqr / (gamma + 1.0);
    const double Gamma_p     = 1.0 + K / mass_ratio;
    const double probability = optical_depth * dt * tables.rates.J(Gamma_p);
    const double D_table     = tables.mean_loss(Gamma_p);
    std::printf(
      "Gamma_p %.17g probability %.17g tabulated <D> %.9f seed %016llx "
      "states %llu trials %d\n",
      Gamma_p,
      probability,
      D_table,
      static_cast<unsigned long long>(seed),
      static_cast<unsigned long long>(n_states),
      n_trials);

    using gen_t = Kokkos::Random_XorShift64<Kokkos::DefaultHostExecutionSpace>;
    // the seeds of the states, as in Random_XorShift64_Pool::init_impl
    std::vector<std::uint64_t> states(n_states);
    {
      gen_t master(seed == 0 ? std::uint64_t(1318319) : seed, 0);
      for (int i = 0; i < 17; ++i) {
        master.rand();
      }
      for (std::uint64_t i = 0; i < n_states; ++i) {
        const int n1 = master.rand();
        const int n2 = master.rand();
        const int n3 = master.rand();
        const int n4 = master.rand();
        states[i]    = (((static_cast<std::uint64_t>(n1)) & 0xffff) << 00) |
                    (((static_cast<std::uint64_t>(n2)) & 0xffff) << 16) |
                    (((static_cast<std::uint64_t>(n3)) & 0xffff) << 32) |
                    (((static_cast<std::uint64_t>(n4)) & 0xffff) << 48);
      }
    }

    const int n_windows = (n_trials + window - 1) / window;
    // per window: n, sum D, sum D^2, sum E-, sum E+, sum (E- - E+)^2, sum E-^2, sum E+^2
    std::vector<std::vector<double>> acc(n_windows, std::vector<double>(8, 0.0));
    std::FILE*    out      = argc > 9 ? std::fopen(argv[9], "wb") : nullptr;
    std::uint64_t n_events = 0;
    if (out != nullptr) {
      std::fwrite(&n_events, sizeof(n_events), 1, out);
    }
    const auto& sampler = tables.sampler;
    for (std::uint64_t i = 0; i < n_states; ++i) {
      gen_t gen(states[i], static_cast<int>(i));
      for (int t = 0; t < n_trials; ++t) {
        if (not(gen.drand() < probability)) {
          continue;
        }
        ::arch::bh::SampledPair o;
        sampler.draw(Gamma_p, gen, o);
        const double Em = std::sqrt(
          1.0 + o.pm[0] * o.pm[0] + o.pm[1] * o.pm[1] + o.pm[2] * o.pm[2]);
        const double Ep = std::sqrt(
          1.0 + o.pp[0] * o.pp[0] + o.pp[1] * o.pp[1] + o.pp[2] * o.pp[2]);
        auto& a  = acc[t / window];
        a[0]    += 1.0;
        a[1]    += o.D;
        a[2]    += o.D * o.D;
        a[3]    += Em;
        a[4]    += Ep;
        a[5]    += (Em - Ep) * (Em - Ep);
        a[6]    += Em * Em;
        a[7]    += Ep * Ep;
        ++n_events;
        if (out != nullptr) {
          const std::uint32_t id[2]  = { static_cast<std::uint32_t>(i),
                                         static_cast<std::uint32_t>(t) };
          // momentum along the emitter (x1 in the run) of e- and e+, and D
          const double        rec[3] = { o.pm[2], o.pp[2], o.D };
          std::fwrite(id, sizeof(std::uint32_t), 2, out);
          std::fwrite(rec, sizeof(double), 3, out);
        }
      }
    }
    if (out != nullptr) {
      std::fseek(out, 0, SEEK_SET);
      std::fwrite(&n_events, sizeof(n_events), 1, out);
      std::fclose(out);
    }
    const double        E_table = 0.5 * (D_table + 0.5);
    std::vector<double> tot(8, 0.0);
    auto report = [&](const std::string& label, const std::vector<double>& a) {
      const double n = a[0];
      if (n < 2.0) {
        return;
      }
      const double mD = a[1] / n, sD = std::sqrt((a[2] / n - mD * mD) / n);
      const double mm = a[3] / n, sm = std::sqrt((a[6] / n - mm * mm) / n);
      const double mp = a[4] / n, sp = std::sqrt((a[7] / n - mp * mp) / n);
      const double ma = mm - mp, sa = std::sqrt((a[5] / n - ma * ma) / n);
      std::printf(
        "%s: %9.0f pairs; D - <D> %+.6f (%+.2f s.e.); E- %+.6f (%+.2f); "
        "E+ %+.6f (%+.2f); E- - E+ %+.6f (%+.2f)\n",
        label.c_str(),
        n,
        mD - D_table,
        (mD - D_table) / sD,
        mm - E_table,
        (mm - E_table) / sm,
        mp - E_table,
        (mp - E_table) / sp,
        ma,
        ma / sa);
    };
    for (int w = 0; w < n_windows; ++w) {
      report("trials [" + std::to_string(w * window) + ", " +
               std::to_string((w + 1) * window) + ")",
             acc[w]);
      for (int k = 0; k < 8; ++k) {
        tot[k] += acc[w][k];
      }
    }
    report("all", tot);
  }
  GlobalFinalize();
  return rc;
}
