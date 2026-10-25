/**
 * @file pgens/bh_proto/pgen.hpp
 * @brief Bethe-Heitler prototype (P2 steps 2 and 3): a test problem generator
 *        that injects ions with prescribed kinetic energies in a periodic (or
 *        absorbing) Cartesian box, optionally with uniform E and B fields, and
 *        launches the `BetheHeitler_kernel` on every ion species flagged
 *        hadronic at the end of each step (prototype placement of spec II.1:
 *        `CustomPostStep`, after the push, the deposit and
 *        `CommunicateParticles`, before the sort). Pairs created by the kernel
 *        are inserted into two dedicated species (`pair_species`) with the
 *        atomic-ticket mechanism of II.3; the host computes the capacity before
 *        the launch and raises `set_npart` after it.
 *
 * Outputs (in the working directory of the run):
 *   hadronic_ledger_<species>.txt  every `ledger.every` steps and at the last
 *                                  step: cumulative double ledger (II.2 step 6)
 *                                  with a header naming every column and unit,
 *                                  the identity residual as a column
 *   hadronic_ions_<species>.txt    per-ion K, Gamma_p, |u|, pitch cosine,
 *                                  delta, debt r, K0
 *   hadronic_pairs_<species>.txt   at the last step: every pair (both leptons)
 *   hadronic_gauss.txt             (test.gauss) the discrete Gauss residual
 *   hadronic_table.txt             the startup table in the exchange format of
 *                                  step 1 (provenance; same code path as
 *                                  bh_tables_dump)
 *   stdout                         grid summary, per-K expectations, the
 *                                  startup capacity estimate, and at the last
 *                                  step the deterministic checksums of every
 *                                  particle array and of the six field arrays,
 *                                  the array-based closure and the ledger
 *
 * Payload slots: this tree declares real payloads per species with the input
 * key `n_payloads_real`; the generator requires every hadronic species to
 * declare at least max(payload_pending, payload_debt, payload_debt_k0) + 1
 * slots (3 with the defaults 0, 1, 2) and each pair species at least three
 * (birth time, parent Gamma_p, absorbed photon energy), and refuses otherwise.
 * The slots of newly injected particles are zero: the injector kernels write
 * them through the P9 overload of `kernel::InjectParticle`, and the generator
 * additionally zeroes [npart_before, npart_after) after its own injection.
 *
 * Time-level convention: the loss acts on u^{n+1/2} after the push of step n;
 * ledger rows and pair birth times are stamped with the time after the step,
 * (step + 1) dt.
 *
 * @namespaces:
 *   - user::
 */

#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include "archetypes/energy_dist.h"
#include "archetypes/hadronic/bh_interpolate.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_table_io.h"
#include "archetypes/hadronic/bh_tables_device.h"
#include "archetypes/particle_injector.h"
#include "archetypes/spatial_dist.h"
#include "framework/containers/particles.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "kernels/hadronic/bethe_heitler.hpp"
#include "kernels/injectors.hpp"

#include "bh_proto_params.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace user {
  using namespace ntt;

  /* ---------------------------------------------------------------------- */
  /* uniform fields                                                          */
  /* ---------------------------------------------------------------------- */
  template <Dimension D>
  struct UniformEMFields {
    real_t e1 { ZERO }, e2 { ZERO }, e3 { ZERO };
    real_t b1 { ZERO }, b2 { ZERO }, b3 { ZERO };

    Inline auto ex1(const coord_t<D>&) const -> real_t {
      return e1;
    }

    Inline auto ex2(const coord_t<D>&) const -> real_t {
      return e2;
    }

    Inline auto ex3(const coord_t<D>&) const -> real_t {
      return e3;
    }

    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return b1;
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return b2;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return b3;
    }
  };

  /* ---------------------------------------------------------------------- */
  /* deterministic hashing (positions and isotropic directions)              */
  /* ---------------------------------------------------------------------- */
  Inline auto splitmix64(std::uint64_t x) -> std::uint64_t {
    x += 0x9E3779B97F4A7C15ULL;
    x  = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x  = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
  }

  /**
   * @brief Uniform double in [0, 1) from a 64-bit hash
   */
  Inline auto hash_unit(std::uint64_t x) -> double {
    return static_cast<double>(splitmix64(x) >> 11) * 0x1.0p-53;
  }

  /* ---------------------------------------------------------------------- */
  /* ion injection: group g = p / per_K has kinetic energy Ks(g)              */
  /* ---------------------------------------------------------------------- */
  template <Dimension D, bool T>
  struct InjectIons_kernel {
    array_t<int*>      i1, i2, i3;
    array_t<prtldx_t*> dx1, dx2, dx3;
    array_t<real_t*>   ux1, ux2, ux3, phi, weight;
    array_t<short*>    tag;
    array_t<npart_t**> pld_i;
    array_t<real_t**>  pld_r;
    array_t<double*>   Ks;
    npart_t            offset, counter;
    int                per_K;
    int                n1, n2, n3; // active cells of the local domain
    double             mu_d, weight_value;
    bool               isotropic;
    std::uint64_t      seed;

    Inline void operator()(npart_t k) const {
      const npart_t  p = offset + k;
      const int      g = static_cast<int>(k) / per_K;
      const double   K = Ks(g);
      const double   u = math::sqrt(K * (K + 2.0 * mu_d)) / mu_d;
      std::uint64_t  h = seed + 7919ULL * static_cast<std::uint64_t>(counter + k);

      tuple_t<int, D>      xi;
      tuple_t<prtldx_t, D> dxi;
      {
        const double r = hash_unit(h + 1);
        xi[0]          = static_cast<int>(r * n1);
        if (xi[0] >= n1) {
          xi[0] = n1 - 1;
        }
        dxi[0] = static_cast<prtldx_t>(hash_unit(h + 2));
        if (dxi[0] >= static_cast<prtldx_t>(1.0)) {
          dxi[0] = static_cast<prtldx_t>(0.999999);
        }
      }
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        const double r = hash_unit(h + 3);
        xi[1]          = static_cast<int>(r * n2);
        if (xi[1] >= n2) {
          xi[1] = n2 - 1;
        }
        dxi[1] = static_cast<prtldx_t>(hash_unit(h + 4));
        if (dxi[1] >= static_cast<prtldx_t>(1.0)) {
          dxi[1] = static_cast<prtldx_t>(0.999999);
        }
      }
      if constexpr (D == Dim::_3D) {
        const double r = hash_unit(h + 5);
        xi[2]          = static_cast<int>(r * n3);
        if (xi[2] >= n3) {
          xi[2] = n3 - 1;
        }
        dxi[2] = static_cast<prtldx_t>(hash_unit(h + 6));
        if (dxi[2] >= static_cast<prtldx_t>(1.0)) {
          dxi[2] = static_cast<prtldx_t>(0.999999);
        }
      }
      vec_t<Dim::_3D> v { ZERO, ZERO, ZERO };
      if (isotropic) {
        const double ct = 2.0 * hash_unit(h + 7) - 1.0;
        const double st = math::sqrt((ct > -1.0 && ct < 1.0) ? 1.0 - ct * ct : 0.0);
        const double ph = 2.0 * 3.14159265358979323846 * hash_unit(h + 8);
        v[0]            = static_cast<real_t>(u * st * math::cos(ph));
        v[1]            = static_cast<real_t>(u * st * math::sin(ph));
        v[2]            = static_cast<real_t>(u * ct);
      } else {
        v[0] = static_cast<real_t>(u);
      }
      // P9 overload: every real payload slot of the new ion is zero
      kernel::InjectParticle<D, Coord::Cartesian, T>(p,
                                                      i1, i2, i3,
                                                      dx1, dx2, dx3,
                                                      ux1, ux2, ux3,
                                                      phi, weight, tag, pld_i, pld_r,
                                                      xi, dxi, v,
                                                      static_cast<real_t>(weight_value),
                                                      ZERO, 0u, counter + k);
    }
  };

  /* ---------------------------------------------------------------------- */
  /* checksums: order-sensitive 64-bit combination of the bit patterns       */
  /* ---------------------------------------------------------------------- */
  namespace bhproto {

    inline void mix_bits(std::uint64_t& h, std::uint64_t bits) {
      h ^= bits;
      h *= 0x100000001B3ULL;
      h ^= (h >> 29);
    }

    template <typename T>
    inline auto bits_of(T v) -> std::uint64_t {
      std::uint64_t b = 0;
      static_assert(sizeof(T) <= 8, "bits_of: type too wide");
      std::memcpy(&b, &v, sizeof(T));
      return b;
    }

    template <typename ViewT>
    inline void checksum_1d(std::uint64_t& h, const ViewT& view, std::size_t n) {
      if (view.extent(0) == 0 || n == 0) {
        mix_bits(h, 0xA5A5ULL);
        return;
      }
      auto hv = Kokkos::create_mirror_view(view);
      Kokkos::deep_copy(hv, view);
      mix_bits(h, static_cast<std::uint64_t>(n));
      for (std::size_t p = 0; p < n; ++p) {
        mix_bits(h, bits_of(hv(p)));
      }
    }

    template <typename ViewT>
    inline void checksum_2d(std::uint64_t& h, const ViewT& view, std::size_t n) {
      if (view.extent(0) == 0 || view.extent(1) == 0 || n == 0) {
        mix_bits(h, 0x5A5AULL);
        return;
      }
      auto hv = Kokkos::create_mirror_view(view);
      Kokkos::deep_copy(hv, view);
      mix_bits(h, static_cast<std::uint64_t>(n));
      mix_bits(h, static_cast<std::uint64_t>(view.extent(1)));
      for (std::size_t p = 0; p < n; ++p) {
        for (std::size_t c = 0; c < view.extent(1); ++c) {
          mix_bits(h, bits_of(hv(p, c)));
        }
      }
    }

    template <Dimension D>
    inline auto checksum_fields(const ndfield_t<D, 6>& em) -> std::uint64_t {
      std::uint64_t h  = 0xCBF29CE484222325ULL;
      auto          hv = Kokkos::create_mirror_view(em);
      Kokkos::deep_copy(hv, em);
      if constexpr (D == Dim::_1D) {
        mix_bits(h, em.extent(0));
        for (std::size_t i = 0; i < em.extent(0); ++i) {
          for (std::size_t c = 0; c < 6; ++c) {
            mix_bits(h, bits_of(hv(i, c)));
          }
        }
      } else if constexpr (D == Dim::_2D) {
        mix_bits(h, em.extent(0));
        mix_bits(h, em.extent(1));
        for (std::size_t i = 0; i < em.extent(0); ++i) {
          for (std::size_t j = 0; j < em.extent(1); ++j) {
            for (std::size_t c = 0; c < 6; ++c) {
              mix_bits(h, bits_of(hv(i, j, c)));
            }
          }
        }
      } else {
        mix_bits(h, em.extent(0));
        mix_bits(h, em.extent(1));
        mix_bits(h, em.extent(2));
        for (std::size_t i = 0; i < em.extent(0); ++i) {
          for (std::size_t j = 0; j < em.extent(1); ++j) {
            for (std::size_t k = 0; k < em.extent(2); ++k) {
              for (std::size_t c = 0; c < 6; ++c) {
                mix_bits(h, bits_of(hv(i, j, k, c)));
              }
            }
          }
        }
      }
      return h;
    }

    inline auto hex(std::uint64_t h) -> std::string {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
      return std::string(buf);
    }

    /**
     * @brief Host accumulation of the per-launch ledgers of one species
     */
    struct HostLedger {
      kernel::hadronic::LossLedger cumulative {}; // snapshots overwritten per launch
      kernel::hadronic::dd         debt_initial {}; // sum w r at the start of the segment
      bool                         debt_initial_set { false };
      bool                         file_started { false };
      bool                         ions_started { false };
      unsigned long long           n_capacity_warnings { 0 };
      unsigned long long           n_max_events_warnings { 0 };
      kernel::hadronic::dd         exported_host {};  // sum w r over dead ions, host, before each launch
      kernel::hadronic::dd         discarded_host {}; // sum w delta over dead ions, host, before each launch
      unsigned long long           n_dead_host { 0 }; // dead ions seen with a non-zero debt or pending request

      void accumulate(const kernel::hadronic::LossLedger& L) {
        const auto snap_pending = L.pending_outstanding;
        const auto snap_debt    = L.debt_outstanding;
        const auto snap_sum     = L.sum_wK;
        const auto snap_alive   = L.n_alive;
        cumulative += L;
        cumulative.pending_outstanding = snap_pending;
        cumulative.debt_outstanding    = snap_debt;
        cumulative.sum_wK              = snap_sum;
        cumulative.n_alive             = snap_alive;
      }

      /**
       * @brief The identity residual (II.2 step 6), in dd:
       *        E_drag + debt_initial - [(E_pairs - E_absorbed) + E_round +
       *        debt_outstanding + E_exported + E_lost_nopairs]
       */
      auto identity_residual() const -> double {
        kernel::hadronic::dd r;
        const auto&          C = cumulative;
        r.add(C.E_drag);
        r.add(debt_initial);
        r.sub(C.E_pairs);
        r.add(C.E_absorbed);
        r.sub(C.E_round);
        r.sub(C.debt_outstanding);
        r.sub(C.E_exported);
        r.sub(C.E_lost_nopairs);
        return r.value();
      }
    };

    /**
     * @brief The hadronic RNG seed (II.4): splitmix64 chained over the tuple
     *        (seed, species, domain, start step); never a sum
     */
    inline auto hadronic_seed(std::uint64_t seed,
                              unsigned int  species,
                              unsigned int  domain,
                              std::uint64_t start_step) -> std::uint64_t {
      std::uint64_t h = splitmix64(seed);
      h               = splitmix64(h ^ (0x1000000000000000ULL | static_cast<std::uint64_t>(species)));
      h               = splitmix64(h ^ (0x2000000000000000ULL | static_cast<std::uint64_t>(domain)));
      h               = splitmix64(h ^ (0x3000000000000000ULL | start_step));
      return h;
    }

  } // namespace bhproto

  /* ---------------------------------------------------------------------- */
  /* the problem generator                                                   */
  /* ---------------------------------------------------------------------- */
  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_1D, Dim::_2D, Dim::_3D> {}
    };

    const SimulationParams&  params;
    const Metadomain<S, M>*  metadomain_ptr { nullptr };
    bhproto::Setup           setup;
    UniformEMFields<D>       init_flds;

    // run constants
    double     dt { 0.0 };
    double     skindepth0 { 1.0 };
    double     omegaB0 { 1.0 };
    double     runtime { 0.0 };
    timestep_t max_steps { 0 };
    timestep_t start_step { 0 };
    bool       stage_active { false }; // enabled and tau0 > 0
    bool       use_weights { false };
    double     t_latency { 0.0 }; // resolved (0 = off)
    double     L_min { 0.0 };
    bool       any_absorbing { false };
    std::uint64_t base_seed { 0 };
    bool          seed_set { false };
    double        gauss_coeff { 0.0 }; // q0 / (B0 V0)

    // tables
    std::shared_ptr<const arch::bh::PhotonBath> bath;
    arch::bh::Quadrature                        quadrature;
    arch::bh::HadronicTablesHost                tables_host;
    arch::bh::HadronicTables                    tables;

    // species bookkeeping
    std::vector<spidx_t>                       ion_species; // 1-based indices
    std::map<spidx_t, double>                  ion_mass;
    std::map<spidx_t, bhproto::HostLedger>     ledgers;
    std::map<spidx_t, std::vector<double>>     initial_wK; // sum w K at injection
    std::map<spidx_t, std::unique_ptr<random_number_pool_t>> pools;
    std::map<spidx_t, array_t<npart_t>>        spawn_counters;
    std::uint64_t                              inject_seed { 0x5BD1E995ULL };
    spidx_t                                    pair_minus { 0 }, pair_plus { 0 };
    npart_t                                    pair_maxnpart_m { 0 }, pair_maxnpart_p { 0 };
    bool                                       gauss_started { false };
    std::vector<double>                        gauss_R0;                 // Gauss residual per node at the first write
    double                                     estimate_bare { 0.0 };     // expected pairs per species (no margins)
    double                                     estimate_sized { 0.0 };    // with c_layer and comm_headroom

    PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
      : params { p }
      , metadomain_ptr { &metadomain } {
      // ---- [setup] -------------------------------------------------------
      try {
        const auto& raw = params.data();
        if (raw.is_table() && raw.contains("setup")) {
          setup = bhproto::parse_setup(raw.at("setup"));
        } else {
          setup = bhproto::parse_setup(toml::value(toml::table {}));
        }
      } catch (const std::invalid_argument& e) {
        raise::Error(e.what(), HERE);
      }
      dt         = static_cast<double>(params.template get<real_t>("algorithms.timestep.dt"));
      skindepth0 = static_cast<double>(params.template get<real_t>("scales.skindepth0"));
      omegaB0    = static_cast<double>(params.template get<real_t>("scales.omegaB0"));
      {
        runtime   = static_cast<double>(params.template get<simtime_t>("simulation.runtime"));
        max_steps = static_cast<timestep_t>(runtime / dt); // as engine.hpp
      }
      start_step = params.template get<timestep_t>("checkpoint.start_step");
      use_weights = params.template get<bool>("particles.use_weights");
      raise::ErrorIf(not use_weights,
                     "bh_proto: particles.use_weights must be true (the ions "
                     "are injected with explicit weights; required with source = budget)",
                     HERE);
      raise::ErrorIf(M::CoordType != Coord::Cartesian,
                     "bh_proto: Cartesian metric required",
                     HERE);
      seed_set = params.template get<bool>("simulation.seed_set");
      base_seed = seed_set ? params.template get<std::uint64_t>("simulation.seed")
                           : constant::RandomSeed;
      {
        const auto q0 = static_cast<double>(params.template get<real_t>("scales.q0"));
        const auto B0 = static_cast<double>(params.template get<real_t>("scales.B0"));
        const auto V0 = static_cast<double>(params.template get<real_t>("scales.V0"));
        gauss_coeff   = q0 / (B0 * V0);
      }

      init_flds.e1 = static_cast<real_t>(setup.E[0]);
      init_flds.e2 = static_cast<real_t>(setup.E[1]);
      init_flds.e3 = static_cast<real_t>(setup.E[2]);
      init_flds.b1 = static_cast<real_t>(setup.B[0]);
      init_flds.b2 = static_cast<real_t>(setup.B[1]);
      init_flds.b3 = static_cast<real_t>(setup.B[2]);

      // ---- box and boundaries ----------------------------------------------
      {
        const auto& mesh = metadomain.mesh();
        L_min            = std::numeric_limits<double>::infinity();
        for (unsigned short d = 0; d < static_cast<unsigned short>(D); ++d) {
          const auto ext = mesh.extent(static_cast<in>(d));
          L_min          = std::min(L_min, static_cast<double>(ext.second - ext.first));
        }
        for (const auto& bc : mesh.prtl_bc()) {
          if (bc.first == PrtlBC::ABSORB || bc.second == PrtlBC::ABSORB) {
            any_absorbing = true;
          }
        }
      }

      // ---- species checks --------------------------------------------------
      const auto& species = metadomain.species_params();
      auto find_species = [&species](int idx) -> const ParticleSpecies* {
        for (const auto& s : species) {
          if (static_cast<int>(s.index()) == idx) {
            return &s;
          }
        }
        return nullptr;
      };
      const int need_slots = std::max(setup.payload_pending,
                                      std::max(setup.payload_debt,
                                               setup.payload_debt_k0)) + 1;
      for (const int idx : setup.species) {
        const ParticleSpecies* sp = find_species(idx);
        raise::ErrorIf(sp == nullptr,
                       fmt::format("bh_proto: setup.hadronic.species lists species %d "
                                   "which is not defined in [[particles.species]]",
                                   idx),
                       HERE);
        raise::ErrorIf(not(sp->mass() > 0.0f),
                       fmt::format("bh_proto: hadronic species %d must have mass > 0 "
                                   "(got %g)", idx, (double)sp->mass()),
                       HERE);
        raise::ErrorIf(sp->charge() != 1.0f,
                       fmt::format("bh_proto: hadronic species %d must have charge +1 "
                                   "(Z = 1 only; got %g)", idx, (double)sp->charge()),
                       HERE);
        raise::ErrorIf(static_cast<int>(sp->npld_r()) < need_slots,
                       fmt::format("bh_proto: hadronic species %d declares "
                                   "n_payloads_real = %d but the loss operator needs "
                                   "%d real payload slots (payload_pending = %d, "
                                   "payload_debt = %d, payload_debt_k0 = %d)",
                                   idx, (int)sp->npld_r(), need_slots,
                                   setup.payload_pending, setup.payload_debt,
                                   setup.payload_debt_k0),
                       HERE);
        raise::ErrorIf(sp->clearing_interval() != 0u,
                       fmt::format("bh_proto: hadronic species %d must set "
                                   "clear_interval = 0 (the per-ion output relies on "
                                   "the array order; prototype limitation)", idx),
                       HERE);
        raise::ErrorIf(sp->spatial_sorting_interval() != 0u,
                       fmt::format("bh_proto: hadronic species %d must set "
                                   "spatial_sorting_interval = 0 (array order; "
                                   "prototype limitation)", idx),
                       HERE);
        raise::ErrorIf((sp->pusher() & ParticlePusher::GCA) != 0,
                       fmt::format("bh_proto: hadronic species %d uses pusher '%s': a GCA "
                                   "pusher is not supported on a hadronic species (spec "
                                   "III.7, P11)", idx,
                                   ParticlePusher::to_string(sp->pusher()).c_str()),
                       HERE);
        if (sp->deposit()) {
          std::printf("bh_proto: warning: species %d deposits current; the uniform "
                      "fields of the validation boxes are exact only with deposit = "
                      "false\n", idx);
        }
        ion_species.push_back(static_cast<spidx_t>(idx));
        ion_mass[static_cast<spidx_t>(idx)] = static_cast<double>(sp->mass());
        ledgers[static_cast<spidx_t>(idx)]  = bhproto::HostLedger {};
      }

      // ---- pair species (II.3) ----------------------------------------------
      if (setup.pair_minus != 0) {
        const int idxs[2]   = { setup.pair_minus, setup.pair_plus };
        const float charges[2] = { -1.0f, 1.0f };
        const char* names[2]   = { "pair_species[0] (e-)", "pair_species[1] (e+)" };
        for (int k = 0; k < 2; ++k) {
          const ParticleSpecies* sp = find_species(idxs[k]);
          raise::ErrorIf(sp == nullptr,
                         fmt::format("bh_proto: setup.%s = %d is not defined in "
                                     "[[particles.species]]", names[k], idxs[k]),
                         HERE);
          raise::ErrorIf(sp->mass() != 1.0f,
                         fmt::format("bh_proto: setup.%s = %d must have mass 1 (got %g)",
                                     names[k], idxs[k], (double)sp->mass()),
                         HERE);
          raise::ErrorIf(sp->charge() != charges[k],
                         fmt::format("bh_proto: setup.%s = %d must have charge %+g (got %g)",
                                     names[k], idxs[k], (double)charges[k], (double)sp->charge()),
                         HERE);
          raise::ErrorIf(sp->npld_r() < 3,
                         fmt::format("bh_proto: setup.%s = %d declares n_payloads_real = %d "
                                     "but a pair species needs 3 real payload slots (birth "
                                     "time, parent Gamma_p, absorbed photon energy)",
                                     names[k], idxs[k], (int)sp->npld_r()),
                         HERE);
          raise::ErrorIf(sp->clearing_interval() != 0u || sp->spatial_sorting_interval() != 0u,
                         fmt::format("bh_proto: setup.%s = %d must set clear_interval = 0 and "
                                     "spatial_sorting_interval = 0 (the pair output relies on "
                                     "contiguous slots; prototype limitation)", names[k], idxs[k]),
                         HERE);
          raise::ErrorIf(sp->use_tracking() && sp->npld_i() < 1,
                         fmt::format("bh_proto: setup.%s = %d tracks but has no integer payload",
                                     names[k], idxs[k]),
                         HERE);
        }
        pair_minus      = static_cast<spidx_t>(setup.pair_minus);
        pair_plus       = static_cast<spidx_t>(setup.pair_plus);
        pair_maxnpart_m = find_species(setup.pair_minus)->maxnpart();
        pair_maxnpart_p = find_species(setup.pair_plus)->maxnpart();
      }
      raise::ErrorIf(setup.test_gauss && D != Dim::_1D,
                     "bh_proto: setup.test.gauss is implemented for 1D runs only",
                     HERE);
      raise::ErrorIf((setup.background_ppc > 0.0 || setup.background_n_global > 0) &&
                       setup.pair_minus == 0,
                     "bh_proto: setup.background.* needs setup.pair_species (the background "
                     "is injected into the two pair species)",
                     HERE);

      // ---- latency cap ------------------------------------------------------
      if (setup.pairs_enabled() && setup.source == "budget") {
        if (setup.t_latency_auto) {
          t_latency = 0.1 * L_min; // c = 1
        } else {
          t_latency = setup.t_latency;
        }
        raise::ErrorIf(t_latency == 0.0 && any_absorbing,
                       "bh_proto: setup.t_latency = 0 (latency cap off) is refused when a "
                       "particle boundary is absorbing (open box: the debt of an ion that "
                       "leaves before its first packet becomes exported energy, III.4)",
                       HERE);
      }

      // ---- tables ------------------------------------------------------------
      stage_active = setup.enabled && setup.tau0 > 0.0;
      if (setup.enabled) {
        try {
          bath = arch::bh::make_bath(setup.bath);
          raise::ErrorIf(setup.source == "budget" && !(bath->gamma_th() > 0.0),
                         "bh_proto: setup.photon_field.eps_max < 2 is required with source = "
                         "budget (the bath has no cutoff below the pair threshold; D > 0 on "
                         "the support of the outcome law is needed, III.4)",
                         HERE);
          const arch::bh::BetheHeitlerRates rates(bath, quadrature);
          tables_host = arch::bh::build_tables(rates, setup.mu_r, setup.gamma_p_max);
          arch::bh::validate_tables(tables_host);
          tables = arch::bh::mirror_tables_to_device(tables_host);
          if (setup.table_dump) {
            arch::bh::TableFileMeta meta;
            meta.status = "cpp_candidate";
#ifdef ENTITY_GIT_HASH
            meta.version = ENTITY_GIT_HASH;
#endif
            arch::bh::write_table_file("hadronic_table.txt", tables_host, *bath,
                                       quadrature, meta);
          }
        } catch (const std::exception& e) {
          raise::Error(std::string("bh_proto: table build failed: ") + e.what(), HERE);
        }
        print_summary();
        if (setup.pairs_enabled()) {
          capacity_estimate();
        }
      } else {
        std::printf("bh_proto: hadronic.enabled = false: no tables, no stage\n");
      }
      std::printf("bh_proto: real_t = %s, dt = %.17g, skindepth0 = %.17g, omegaB0 = %.17g, "
                  "max_steps = %u, stage %s, source %s, spectrum %s, t_latency = %.17g (%s), "
                  "L_min = %.17g, seed %s%llu, start_step %u\n",
                  sizeof(real_t) == 4 ? "float" : "double", dt, skindepth0, omegaB0, (unsigned)max_steps,
                  stage_active ? "active" : (setup.enabled ? "not launched (tau0 = 0)"
                                                           : "disabled"),
                  setup.source.c_str(), setup.spectrum.c_str(), t_latency,
                  setup.t_latency_auto ? "auto" : "given", L_min,
                  seed_set ? "simulation.seed = " : "default ", (unsigned long long)base_seed,
                  (unsigned)start_step);
      std::fflush(stdout);
    }

    /**
     * @brief Grid summary and the per-step expectations at each initial K
     */
    void print_summary() const {
      const auto& T = tables_host;
      std::printf("bh_proto: table: %zu nodes (fine %zu, coarse %zu), gamma_th = %.12g, "
                  "gamma_min = %.12g, gamma_max = %.12g, y_min = %.10g, y_max = %.10g, "
                  "eps'_max = %.6g, I_peak = %.10e at gamma = %.10g, mu_r = %.10g\n",
                  T.size(), T.n_fine, T.n_coarse, T.gamma_th, T.gamma_min, T.gamma_max,
                  T.y_min, T.y_max, T.eps_prime_max, T.I_peak, T.gamma_at_I_peak, T.mu_r);
      const arch::bh::BetheHeitlerRates rates(bath, quadrature);
      const double coeff = setup.tau0 / skindepth0;
      for (const auto sp : ion_species) {
        const double mu_d = ion_mass.at(sp);
        for (std::size_t g = 0; g < setup.ions_K.size(); ++g) {
          const double K  = setup.ions_K[g];
          const double Gp = 1.0 + K / setup.mu_r;
          const double u  = std::sqrt(K * (K + 2.0 * mu_d)) / mu_d;
          const double gd = std::sqrt(1.0 + u * u);
          const real_t ur = static_cast<real_t>(u);
          const double ulp = kernel::hadronic::ulp_real(ur);
          const double dK_ulp = ulp * mu_d * u / gd; // energy resolution of the stored u
          double I_exact = 0.0, I_table = 0.0;
          std::string where = "in table";
          if (Gp < T.gamma_min) {
            where = "below the lower edge (no loss)";
          } else if (Gp > T.gamma_max) {
            where   = "above the upper edge (edge value, counted)";
            I_exact = rates.at_gamma(T.gamma_max).I;
            I_table = arch::bh::interpolate_loss_host(T, T.gamma_max);
          } else {
            I_exact = rates.at_gamma(Gp).I;
            I_table = arch::bh::interpolate_loss_host(T, Gp);
          }
          const double dK = coeff * I_table * dt;
          const double floor = setup.pending_floor_ulp * dK_ulp;
          const double dth   = std::max(floor, setup.eta_u * K);
          double Dm = 0.0, em = 0.0;
          arch::bh::interpolate_outcome_host(T, Gp, Dm, em);
          std::printf("bh_proto: species %d K = %.10g (Gamma_p = %.10g, u = %.10g, mu_d = %.6g) %s: "
                      "I_exact = %.10e, I_table = %.10e, dK/step = %.10e, dK/K = %.4e, "
                      "ulp(u) = %.4e, dK_ulp = ulp mu_d u/gamma = %.4e, dK/dK_ulp = %.4e, "
                      "delta_th = %.4e (floor %.4e, eta_u K %.4e), steps per application ~ %.3g, "
                      "sub-cycling %s, <D> = %.10g, eps_mean = %.10g, r_th terms: %.6g (r_th_factor <D>) "
                      "%.6g (f_r K) %.6g (dK t_latency/dt)\n",
                      (int)sp, K, Gp, u, mu_d, where.c_str(), I_exact, I_table, dK, dK / K,
                      ulp, dK_ulp, dK_ulp > 0.0 ? dK / dK_ulp : 0.0, dth, floor,
                      setup.eta_u * K, dK > 0.0 ? std::ceil(dth / dK) : 0.0,
                      dK > setup.f_max * K ? "yes" : "no", Dm, em,
                      setup.r_th_factor * Dm, setup.f_r * K,
                      t_latency > 0.0 && dt > 0.0 ? dK * t_latency / dt : 0.0);
        }
      }
      std::fflush(stdout);
    }

    /**
     * @brief Expected spawn rate per active ion per unit time at Gamma (III.8):
     *        budget: coeff I / r_th with r_th the actual minimum of its terms;
     *        poisson: coeff J / pair_weight
     */
    auto spawn_rate_at(double Gp) const -> double {
      const auto&  T     = tables_host;
      const double coeff = setup.tau0 / skindepth0;
      const double I     = arch::bh::interpolate_loss_host(T, Gp);
      if (!(I > 0.0)) {
        return 0.0;
      }
      if (setup.source == "poisson") {
        return coeff * arch::bh::interpolate_rate_host(T, Gp) / setup.pair_weight;
      }
      const double K = setup.mu_r * (Gp - 1.0);
      double Dm = 0.0, em = 0.0;
      arch::bh::interpolate_outcome_host(T, Gp, Dm, em);
      double r_th = std::min(setup.r_th_factor * Dm, setup.f_r * K);
      if (t_latency > 0.0) {
        r_th = std::min(r_th, coeff * I * t_latency);
      }
      return r_th > 0.0 ? coeff * I / r_th : 0.0;
    }

    /**
     * @brief Startup capacity estimate (III.8), printed and enforced by
     *        on_capacity_estimate
     */
    void capacity_estimate() {
      double rate = 0.0;
      std::string at;
      if (setup.gamma_p_typical > 0.0) {
        rate = spawn_rate_at(setup.gamma_p_typical);
        at   = fmt::format("gamma_p_typical = %.10g", setup.gamma_p_typical);
      } else {
        // average over dN/dGamma ~ Gamma^-index on [gamma_min, gamma_max] (trapezoid in ln Gamma)
        const int    nq  = 400;
        const double l0  = std::log(setup.envelope_gamma_min), l1 = std::log(setup.envelope_gamma_max);
        double       num = 0.0, den = 0.0;
        for (int i = 0; i <= nq; ++i) {
          const double l  = l0 + (l1 - l0) * static_cast<double>(i) / nq;
          const double G  = std::exp(l);
          const double wq = (i == 0 || i == nq) ? 0.5 : 1.0;
          const double f  = std::pow(G, 1.0 - setup.envelope_index); // Gamma^-index dGamma = Gamma^(1-index) dlnGamma
          num            += wq * f * spawn_rate_at(G);
          den            += wq * f;
        }
        rate = den > 0.0 ? num / den : 0.0;
        at   = fmt::format("envelope index %.6g on [%.10g, %.10g]", setup.envelope_index,
                           setup.envelope_gamma_min, setup.envelope_gamma_max);
      }
      const double N_ions = static_cast<double>(setup.ions_K.size()) *
                            static_cast<double>(setup.ions_per_K) *
                            static_cast<double>(ion_species.size());
      estimate_bare  = N_ions * setup.active_fraction * rate * runtime;
      estimate_sized = estimate_bare * setup.c_layer * (1.0 + setup.comm_headroom);
      // background particles injected at InitPrtls occupy slots too (the InjectNonUniform
      // part depends on the cell count and is not known here; InjectGlobally's is)
      const double background = static_cast<double>(setup.background_n_global);
      const double need       = estimate_sized + background;
      const bool   exceeded = need > static_cast<double>(pair_maxnpart_m) ||
                            need > static_cast<double>(pair_maxnpart_p);
      std::printf("bh_proto: capacity estimate (III.8): spawn rate per active ion per unit time = %.10g "
                  "at %s; N_ions = %.0f, active_fraction = %.6g, t_run = %.10g, c_layer = %.6g, "
                  "comm_headroom = %.6g: expected macro-particles per pair species = %.10g (bare: "
                  "%.10g, no margins); maxnpart e- = %u, e+ = %u : %s\n",
                  rate, at.c_str(), N_ions, setup.active_fraction, runtime, setup.c_layer,
                  setup.comm_headroom, estimate_sized, estimate_bare, (unsigned)pair_maxnpart_m,
                  (unsigned)pair_maxnpart_p, exceeded ? "EXCEEDED" : "within capacity");
      std::fflush(stdout);
      if (exceeded) {
        const auto msg = fmt::format(
          "bh_proto: on_capacity_estimate: the startup estimate of %.10g macro-particles per "
          "pair species (bare %.10g) exceeds maxnpart (e- %u, e+ %u) [setup.on_capacity_estimate = %s]",
          estimate_sized, estimate_bare, (unsigned)pair_maxnpart_m, (unsigned)pair_maxnpart_p,
          setup.on_capacity_estimate.c_str());
        if (setup.on_capacity_estimate == "error") {
          raise::Error(msg, HERE);
        } else {
          std::printf("%s (warning, continuing)\n", msg.c_str());
        }
      }
    }

    /* -------------------------------------------------------------------- */
    /* initial conditions                                                    */
    /* -------------------------------------------------------------------- */
    inline void InitPrtls(Domain<S, M>& domain) {
      raise::ErrorIf(domain.index() != 0u,
                     "bh_proto: multi-domain runs are out of scope (prototype)",
                     HERE);
      const int n1 = static_cast<int>(domain.mesh.n_active(in::x1));
      const int n2 = (D == Dim::_2D || D == Dim::_3D)
                       ? static_cast<int>(domain.mesh.n_active(in::x2)) : 1;
      const int n3 = (D == Dim::_3D) ? static_cast<int>(domain.mesh.n_active(in::x3)) : 1;

      // T10b: every real payload slot of every species is NaN before any injection
      if (setup.test_nan_payloads) {
        for (auto& sp : domain.species) {
          if (sp.npld_r() > 0) {
            Kokkos::deep_copy(sp.pld_r, std::numeric_limits<real_t>::quiet_NaN());
          }
        }
        Kokkos::fence();
        std::printf("bh_proto: test.nan_payloads: every real payload slot of every species set to NaN\n");
      }

      array_t<double*> Ks { "bh_Ks", setup.ions_K.size() };
      {
        auto h = Kokkos::create_mirror_view(Ks);
        for (std::size_t g = 0; g < setup.ions_K.size(); ++g) {
          h(g) = setup.ions_K[g];
        }
        Kokkos::deep_copy(Ks, h);
      }
      const npart_t n_new = static_cast<npart_t>(setup.ions_K.size()) *
                            static_cast<npart_t>(setup.ions_per_K);

      for (auto& sp : domain.species) {
        if (ion_mass.count(sp.index()) == 0) {
          continue;
        }
        const npart_t npart_before = sp.npart();
        raise::ErrorIf(npart_before + n_new > sp.maxnpart(),
                       fmt::format("bh_proto: species %d: %u ions requested but "
                                   "maxnpart = %u", (int)sp.index(),
                                   (unsigned)(npart_before + n_new),
                                   (unsigned)sp.maxnpart()),
                       HERE);
        const double mu_d = ion_mass.at(sp.index());
        auto launch = [&](auto kern) {
          kern.i1 = sp.i1; kern.i2 = sp.i2; kern.i3 = sp.i3;
          kern.dx1 = sp.dx1; kern.dx2 = sp.dx2; kern.dx3 = sp.dx3;
          kern.ux1 = sp.ux1; kern.ux2 = sp.ux2; kern.ux3 = sp.ux3;
          kern.phi = sp.phi; kern.weight = sp.weight; kern.tag = sp.tag;
          kern.pld_i = sp.pld_i; kern.pld_r = sp.pld_r; kern.Ks = Ks;
          kern.offset = npart_before; kern.counter = sp.counter();
          kern.per_K = setup.ions_per_K;
          kern.n1 = n1; kern.n2 = n2; kern.n3 = n3;
          kern.mu_d = mu_d; kern.weight_value = setup.ions_weight;
          kern.isotropic = (setup.ions_direction == "isotropic");
          kern.seed = inject_seed + 1000003ULL * static_cast<std::uint64_t>(sp.index());
          Kokkos::parallel_for("bh_proto::InjectIons",
                               CreateParticleRangePolicy<Dim::_1D>({ 0u }, { n_new }),
                               kern);
        };
        if (sp.use_tracking()) {
          launch(InjectIons_kernel<D, true> {});
        } else {
          launch(InjectIons_kernel<D, false> {});
        }
        sp.set_npart(npart_before + n_new);
        sp.set_counter(sp.counter() + n_new);
        sp.set_unsorted();
        // P9: every real payload slot of a newly injected ion is zero (the
        // injector overload already wrote them; the explicit pass is kept)
        zero_payloads(sp, npart_before, npart_before + n_new);
        Kokkos::fence();

        // record the injected energy (double, from the stored arrays)
        const double wK = sum_wK_from_arrays(sp, mu_d);
        initial_wK[sp.index()].push_back(wK);
        std::printf("bh_proto: species %d: injected %u ions (%zu energies x %d), "
                    "sum w K from the stored arrays = %.17g\n",
                    (int)sp.index(), (unsigned)n_new, setup.ions_K.size(),
                    setup.ions_per_K, wK);
      }

      // ---- background pair plasma through the stock injector paths (T10b, T13)
      if (pair_minus != 0 && setup.background_ppc > 0.0) {
        auto& spm = domain.species[pair_minus - 1];
        auto& spp = domain.species[pair_plus - 1];
        const npart_t before_m = spm.npart(), before_p = spp.npart();
        const auto   temperature = static_cast<real_t>(setup.background_temperature);
        const arch::energy_dist::Maxwellian<D, M::CoordType> maxw_m { domain.random_pool(), temperature };
        const arch::energy_dist::Maxwellian<D, M::CoordType> maxw_p { domain.random_pool(), temperature };
        const arch::spatial_dist::Uniform<D>                 uniform {};
        // InjectNonUniform: ppc = number_density * ppc0 / 2 per species
        const auto ppc0 = static_cast<double>(params.template get<real_t>("particles.ppc0"));
        const auto number_density = static_cast<real_t>(2.0 * setup.background_ppc / ppc0);
        arch::InjectNonUniform<S, M>(params, domain, { pair_minus, pair_plus },
                                     std::make_pair(maxw_m, maxw_p), uniform, number_density,
                                     true);
        spm.set_unsorted();
        spp.set_unsorted();
        Kokkos::fence();
        std::printf("bh_proto: background: arch::InjectNonUniform injected %u e- and %u e+ "
                    "(ppc %.6g per species, temperature %.6g) through the stock NonUniformInjector_kernel\n",
                    (unsigned)(spm.npart() - before_m), (unsigned)(spp.npart() - before_p),
                    setup.background_ppc, setup.background_temperature);
      }
      if (pair_minus != 0 && setup.background_n_global > 0) {
        for (const spidx_t sidx : { pair_minus, pair_plus }) {
          auto&       sp     = domain.species[sidx - 1];
          const auto  before = sp.npart();
          const int   n      = setup.background_n_global;
          std::map<std::string, std::vector<real_t>> data;
          const auto ext1 = domain.mesh.extent(in::x1);
          for (int k = 0; k < n; ++k) {
            const std::uint64_t h = 0xABCDEF1234ULL + 977ULL * static_cast<std::uint64_t>(k) +
                                    31ULL * static_cast<std::uint64_t>(sidx);
            data["x1"].push_back(static_cast<real_t>(ext1.first + (ext1.second - ext1.first) * hash_unit(h + 1)));
            if constexpr (D == Dim::_2D || D == Dim::_3D) {
              const auto ext2 = domain.mesh.extent(in::x2);
              data["x2"].push_back(static_cast<real_t>(ext2.first + (ext2.second - ext2.first) * hash_unit(h + 2)));
            }
            if constexpr (D == Dim::_3D) {
              const auto ext3 = domain.mesh.extent(in::x3);
              data["x3"].push_back(static_cast<real_t>(ext3.first + (ext3.second - ext3.first) * hash_unit(h + 3)));
            }
            data["ux1"].push_back(static_cast<real_t>(0.1 * (2.0 * hash_unit(h + 4) - 1.0)));
            data["ux2"].push_back(ZERO);
            data["ux3"].push_back(ZERO);
          }
          // weight 1 (use_weights = false in the injector: the weighted path of
          // GlobalInjector_kernel reads the destination array, an upstream bug
          // listed under P10 and out of scope here)
          arch::InjectGlobally<S, M>(*metadomain_ptr, domain, sidx, data, false);
          sp.set_unsorted();
          Kokkos::fence();
          std::printf("bh_proto: background: arch::InjectGlobally injected %u particles into species %d "
                      "through the stock GlobalInjector_kernel\n",
                      (unsigned)(sp.npart() - before), (int)sidx);
        }
      }

      // the debt at the start of the segment (0 for a fresh run; from the arrays)
      for (auto& sp : domain.species) {
        if (ion_mass.count(sp.index()) == 0) {
          continue;
        }
        auto& H = ledgers[sp.index()];
        H.debt_initial     = sum_w_payload_from_arrays(sp, static_cast<unsigned short>(setup.payload_debt));
        H.debt_initial_set = true;
      }
      std::fflush(stdout);
    }

    static void zero_payloads(Particles<D, M::CoordType>& sp, npart_t p0, npart_t p1) {
      if (sp.npld_r() == 0 || p1 <= p0) {
        return;
      }
      auto           pld_r = sp.pld_r;
      const auto     npld  = sp.npld_r();
      Kokkos::parallel_for("bh_proto::ZeroPayloads",
                           CreateParticleRangePolicy<Dim::_1D>({ p0 }, { p1 }),
                           Lambda(prtlidx_t p) {
                             for (unsigned short c = 0; c < npld; ++c) {
                               pld_r(p, c) = ZERO;
                             }
                           });
    }

    /**
     * @brief sum over alive ions of w K computed in double from the stored u,
     *        accumulated in dd on the host (order-independent to ~1e-32)
     */
    static auto sum_wK_from_arrays(const Particles<D, M::CoordType>& sp, double mu_d)
      -> double {
      auto h1 = Kokkos::create_mirror_view(sp.ux1);
      auto h2 = Kokkos::create_mirror_view(sp.ux2);
      auto h3 = Kokkos::create_mirror_view(sp.ux3);
      auto hw = Kokkos::create_mirror_view(sp.weight);
      auto ht = Kokkos::create_mirror_view(sp.tag);
      Kokkos::deep_copy(h1, sp.ux1);
      Kokkos::deep_copy(h2, sp.ux2);
      Kokkos::deep_copy(h3, sp.ux3);
      Kokkos::deep_copy(hw, sp.weight);
      Kokkos::deep_copy(ht, sp.tag);
      kernel::hadronic::dd acc;
      for (npart_t p = 0; p < sp.npart(); ++p) {
        if (ht(p) == ParticleTag::dead) {
          continue;
        }
        acc.add_prod(static_cast<double>(hw(p)),
                     kernel::hadronic::kinetic_energy(mu_d, static_cast<double>(h1(p)),
                                                      static_cast<double>(h2(p)),
                                                      static_cast<double>(h3(p))));
      }
      return acc.value();
    }

    /**
     * @brief sum over alive ions of w times a real payload slot, in dd (host loop)
     */
    static auto sum_w_payload_from_arrays(const Particles<D, M::CoordType>& sp, unsigned short slot)
      -> kernel::hadronic::dd {
      kernel::hadronic::dd acc;
      if (sp.npld_r() == 0) {
        return acc;
      }
      auto hw = Kokkos::create_mirror_view(sp.weight);
      auto ht = Kokkos::create_mirror_view(sp.tag);
      auto hp = Kokkos::create_mirror_view(sp.pld_r);
      Kokkos::deep_copy(hw, sp.weight);
      Kokkos::deep_copy(ht, sp.tag);
      Kokkos::deep_copy(hp, sp.pld_r);
      for (npart_t p = 0; p < sp.npart(); ++p) {
        if (ht(p) == ParticleTag::dead) {
          continue;
        }
        acc.add_prod(static_cast<double>(hw(p)), static_cast<double>(hp(p, slot)));
      }
      return acc;
    }

    /* -------------------------------------------------------------------- */
    /* the stage                                                             */
    /* -------------------------------------------------------------------- */
    inline void CustomPostStep(timestep_t step, simtime_t /*time*/, Domain<S, M>& domain) {
      const double t_after = static_cast<double>(step + 1) * dt; // t^{n+1}
      const bool   last    = (step + 1 >= max_steps);
      const bool   row     = ((step + 1) % static_cast<timestep_t>(setup.ledger_every) == 0u) || last;
      if (setup.test_gauss && step == start_step) {
        write_gauss(domain, step, t_after, "before_births");
      }
      if (stage_active) {
        kernel::hadronic::HadronicParams prm;
        prm.tau0_over_skindepth0 = setup.tau0 / skindepth0;
        prm.mu_r                 = setup.mu_r;
        prm.eta_u                = setup.eta_u;
        prm.pending_floor_ulp    = setup.pending_floor_ulp;
        prm.f_max                = setup.f_max;
        prm.max_substeps         = setup.max_substeps;
        prm.pld_pending          = static_cast<unsigned short>(setup.payload_pending);
        prm.pld_debt             = static_cast<unsigned short>(setup.payload_debt);
        prm.pld_debt_k0          = static_cast<unsigned short>(setup.payload_debt_k0);
        prm.dt                   = dt;
        prm.time                 = t_after;
        prm.profile              = static_cast<int>(kernel::hadronic::BathProfile::Uniform);
        prm.unsafe_application   = setup.design_off();
        prm.source = setup.source == "budget"    ? static_cast<int>(kernel::hadronic::PairSource::Budget)
                     : setup.source == "poisson" ? static_cast<int>(kernel::hadronic::PairSource::Poisson)
                                                 : static_cast<int>(kernel::hadronic::PairSource::None);
        prm.spectrum = setup.spectrum == "threshold"
                         ? static_cast<int>(kernel::hadronic::PairSpectrum::Threshold)
                         : static_cast<int>(kernel::hadronic::PairSpectrum::Delta);
        prm.r_th_factor         = setup.r_th_factor;
        prm.f_r                 = setup.f_r;
        prm.t_latency           = t_latency;
        prm.pair_weight         = setup.pair_weight;
        prm.max_events_per_step = setup.max_events_per_step;
        prm.kick                = setup.test_kick;

        for (auto& sp : domain.species) {
          if (ion_mass.count(sp.index()) == 0 || sp.npart() == 0) {
            continue;
          }
          auto& H = ledgers[sp.index()];
          // T-exp: host-side sums over the ions tagged dead this step (absorbing
          // walls), taken before the launch and before the sort; the kernel's
          // E_exported / pending_discarded must reproduce them exactly
          if (any_absorbing) {
            auto hw = Kokkos::create_mirror_view(sp.weight);
            auto ht = Kokkos::create_mirror_view(sp.tag);
            auto hp = Kokkos::create_mirror_view(sp.pld_r);
            Kokkos::deep_copy(hw, sp.weight);
            Kokkos::deep_copy(ht, sp.tag);
            Kokkos::deep_copy(hp, sp.pld_r);
            unsigned long long n_dead_now = 0;
            for (npart_t p = 0; p < sp.npart(); ++p) {
              if (ht(p) == ParticleTag::dead) {
                const double w = static_cast<double>(hw(p));
                H.exported_host.add_prod(w, static_cast<double>(hp(p, setup.payload_debt)));
                H.discarded_host.add_prod(w, static_cast<double>(hp(p, setup.payload_pending)));
                if (hp(p, setup.payload_debt) != ZERO || hp(p, setup.payload_pending) != ZERO) {
                  ++n_dead_now;
                }
              }
            }
            H.n_dead_host += n_dead_now;
          }
          if (!H.debt_initial_set) {
            H.debt_initial     = sum_w_payload_from_arrays(sp, static_cast<unsigned short>(setup.payload_debt));
            H.debt_initial_set = true;
          }
          // the hadronic RNG pool of this species (II.4), built on first use
          if (pools.count(sp.index()) == 0) {
            const auto seed = bhproto::hadronic_seed(base_seed, sp.index(), domain.index(),
                                                     static_cast<std::uint64_t>(start_step));
            pools[sp.index()] = std::make_unique<random_number_pool_t>(seed);
            spawn_counters[sp.index()] = array_t<npart_t> { "bh_spawn_counter" };
            std::printf("bh_proto: species %d: hadronic RNG pool seeded with splitmix64(seed %llu, "
                        "species %d, domain %u, start step %u) = %llu\n",
                        (int)sp.index(), (unsigned long long)base_seed, (int)sp.index(),
                        domain.index(), (unsigned)start_step, (unsigned long long)seed);
          }
          kernel::hadronic::PairTargets targets;
          npart_t npart_m = 0, npart_p = 0;
          if (setup.pairs_enabled()) {
            auto& spm          = domain.species[pair_minus - 1];
            auto& spp          = domain.species[pair_plus - 1];
            npart_m            = spm.npart();
            npart_p            = spp.npart();
            targets.minus      = static_cast<const ParticleArrays&>(spm);
            targets.plus       = static_cast<const ParticleArrays&>(spp);
            targets.offset_m   = npart_m;
            targets.offset_p   = npart_p;
            targets.capacity   = std::min(spm.maxnpart() - npart_m, spp.maxnpart() - npart_p);
            targets.counter_m  = spm.counter();
            targets.counter_p  = spp.counter();
            targets.domain_idx = domain.index();
            targets.tracking_m = spm.use_tracking();
            targets.tracking_p = spp.use_tracking();
            targets.npld_r     = std::min(spm.npld_r(), spp.npld_r());
            targets.spawn_counter = spawn_counters[sp.index()];
            Kokkos::deep_copy(targets.spawn_counter, static_cast<npart_t>(0));
            targets.valid = true;
          }
          kernel::hadronic::LossLedger L;
          Kokkos::parallel_reduce(
            "bh_proto::BetheHeitler",
            CreateParticleRangePolicy<Dim::_1D>({ 0u }, { sp.npart() }),
            kernel::hadronic::BetheHeitler_kernel<M> {
              static_cast<const ParticleArrays&>(sp), ion_mass.at(sp.index()), tables, prm,
              targets, *pools[sp.index()] },
            Kokkos::Sum<kernel::hadronic::LossLedger>(L));
          Kokkos::fence();
          if (setup.pairs_enabled()) {
            auto& spm = domain.species[pair_minus - 1];
            auto& spp = domain.species[pair_plus - 1];
            npart_t tickets = 0;
            Kokkos::deep_copy(tickets, targets.spawn_counter);
            const npart_t n_injected = std::min(tickets, targets.capacity);
            spm.set_npart(npart_m + n_injected);
            spp.set_npart(npart_p + n_injected);
            spm.set_counter(spm.counter() + n_injected);
            spp.set_counter(spp.counter() + n_injected);
            spm.set_unsorted();
            spp.set_unsorted();
            raise::ErrorIf(static_cast<unsigned long long>(n_injected) != L.n_spawned,
                           fmt::format("bh_proto: internal: n_injected %u != n_spawned %llu",
                                       (unsigned)n_injected, L.n_spawned),
                           HERE);
            H.accumulate(L);
            if (L.n_dropped > 0) {
              const auto msg = fmt::format(
                "bh_proto: on_capacity: pair capacity exhausted at step %u (species %d): "
                "capacity = %u, tickets = %u, inserted = %u, n_dropped = %llu this step; e- npart %u / "
                "maxnpart %u, e+ npart %u / maxnpart %u; the debt is kept [setup.on_capacity = %s]",
                (unsigned)step, (int)sp.index(), (unsigned)targets.capacity, (unsigned)tickets,
                (unsigned)n_injected, L.n_dropped, (unsigned)spm.npart(), (unsigned)spm.maxnpart(),
                (unsigned)spp.npart(), (unsigned)spp.maxnpart(), setup.on_capacity.c_str());
              if (setup.on_capacity == "error") {
                write_ledger(sp, step, t_after);
                std::fflush(stdout);
                raise::Error(msg, HERE);
              }
              if (H.n_capacity_warnings < 5) {
                std::printf("%s (warning, continuing)\n", msg.c_str());
              }
              H.n_capacity_warnings += 1;
            }
            if (L.n_capped_events > 0) {
              const auto msg = fmt::format(
                "bh_proto: on_max_events: %llu ion-steps exceeded max_events_per_step = %d at step %u "
                "(species %d) [setup.on_max_events = %s]",
                L.n_capped_events, setup.max_events_per_step, (unsigned)step, (int)sp.index(),
                setup.on_max_events.c_str());
              if (setup.on_max_events == "error") {
                write_ledger(sp, step, t_after);
                std::fflush(stdout);
                raise::Error(msg, HERE);
              }
              if (H.n_max_events_warnings < 5) {
                std::printf("%s (warning, continuing)\n", msg.c_str());
              }
              H.n_max_events_warnings += 1;
            }
          } else {
            H.accumulate(L);
          }
        }
      }
      if (setup.test_gauss && stage_active && step == start_step) {
        write_gauss(domain, step, t_after, "after_first_launch");
      }
      if (row) {
        for (auto& sp : domain.species) {
          if (ion_mass.count(sp.index()) == 0) {
            continue;
          }
          write_ledger(sp, step, t_after);
          write_ions(sp, step, t_after);
        }
        if (setup.test_gauss) {
          write_gauss(domain, step, t_after, "step");
        }
      }
      if (last) {
        finish(domain);
      }
    }

    void write_ledger(const Particles<D, M::CoordType>& sp, timestep_t step, double t) {
      auto&             H    = ledgers[sp.index()];
      const std::string name = "hadronic_ledger_" + std::to_string((int)sp.index()) + ".txt";
      std::ofstream out(name, H.file_started ? std::ios::app : std::ios::trunc);
      if (!H.file_started) {
        out << "# bh_proto ledger, species " << (int)sp.index() << " (" << sp.label()
            << "), mu_d = " << std::setprecision(17) << ion_mass.at(sp.index())
            << ", mu_r = " << setup.mu_r << ", tau0 = " << setup.tau0
            << ", skindepth0 = " << skindepth0 << ", dt = " << dt << ", source = " << setup.source
            << ", spectrum = " << setup.spectrum << ", r_th_factor = " << setup.r_th_factor
            << ", f_r = " << setup.f_r << ", t_latency = " << t_latency
            << ", pair_weight = " << setup.pair_weight
            << ", max_events_per_step = " << setup.max_events_per_step << "\n";
        out << "# cumulative entries are summed over launches since the start of the segment; "
               "snapshots (debt_outstanding, pending_outstanding, sum_wK, npart) refer to the state "
               "after the step; energies in w m_e c^2, momenta in w m_e c; identity residual = "
               "E_drag + debt_initial - [(E_pairs - E_absorbed) + E_round + debt_outstanding + "
               "E_exported + E_lost_nopairs] (double-double on the host)\n";
        out << "# time is the time after the step, (step + 1) dt (code units l/c)\n";
        out << "# columns: step [1] | time [l/c] | E_drag [cumulative] | E_pairs [cumulative: sum "
               "w_pair,stored (E- + E+) from the stored momenta] | E_absorbed [cumulative: sum "
               "w_pair,stored eps_stored] | E_exported [cumulative] | E_round [cumulative] | "
               "E_lost_nopairs [cumulative] | pending_discarded [cumulative] | debt_initial "
               "[segment start] | P_drag_x | P_drag_y | P_drag_z | P_pairs_x | P_pairs_y | "
               "P_pairs_z | P_absorbed_x | P_absorbed_y | P_absorbed_z [cumulative] | "
               "debt_outstanding [snapshot] | pending_outstanding [snapshot] | residual [identity] "
               "| n_events | n_spawned | n_dropped | n_substeps | n_bounded | n_capped_loss | "
               "n_closing | n_capped | n_capped_events | n_above_table | n_latency | n_processed | "
               "n_applied | n_rng | n_k0_trigger [1, cumulative] | sum_wK [snapshot: sum over alive "
               "ions of w K in double from the stored u] | B_budget [cumulative single-precision "
               "rounding budget of the transfers] | npart_minus | npart_plus [snapshots]\n";
        H.file_started = true;
      }
      // the array snapshot is recomputed from the stored arrays (independent of the kernel)
      const double swk = sum_wK_from_arrays(sp, ion_mass.at(sp.index()));
      const auto&  C   = H.cumulative;
      out << std::setprecision(17) << step << " " << t << " " << C.E_drag.value() << " "
          << C.E_pairs.value() << " " << C.E_absorbed.value() << " " << C.E_exported.value() << " "
          << C.E_round.value() << " " << C.E_lost_nopairs.value() << " "
          << C.pending_discarded.value() << " " << H.debt_initial.value() << " " << C.P_drag[0]
          << " " << C.P_drag[1] << " " << C.P_drag[2] << " " << C.P_pairs[0] << " " << C.P_pairs[1]
          << " " << C.P_pairs[2] << " " << C.P_absorbed[0] << " " << C.P_absorbed[1] << " "
          << C.P_absorbed[2] << " " << C.debt_outstanding.value() << " "
          << C.pending_outstanding.value() << " " << H.identity_residual() << " " << C.n_events
          << " " << C.n_spawned << " " << C.n_dropped << " " << C.n_substeps << " " << C.n_bounded
          << " " << C.n_capped_loss << " " << C.n_closing << " " << C.n_capped << " "
          << C.n_capped_events << " " << C.n_above_table << " " << C.n_latency << " "
          << C.n_processed << " " << C.n_applied << " " << C.n_rng << " " << C.n_k0_trigger << " "
          << swk << " " << C.B_budget.value() << " " << pair_npart_minus << " " << pair_npart_plus
          << "\n";
    }

    npart_t pair_npart_minus { 0 }, pair_npart_plus { 0 };

    void write_ions(const Particles<D, M::CoordType>& sp, timestep_t step, double t) {
      auto&             H    = ledgers[sp.index()];
      const std::string name = "hadronic_ions_" + std::to_string((int)sp.index()) + ".txt";
      std::ofstream out(name, H.ions_started ? std::ios::app : std::ios::trunc);
      const double Bn = std::sqrt(setup.B[0] * setup.B[0] + setup.B[1] * setup.B[1] +
                                  setup.B[2] * setup.B[2]);
      if (!H.ions_started) {
        out << "# bh_proto per-ion output, species " << (int)sp.index() << ", mu_d = "
            << std::setprecision(17) << ion_mass.at(sp.index()) << ", mu_r = " << setup.mu_r
            << "\n";
        out << "# one block per output; block header: '# step S time T n N'; rows: "
               "ion [array index] | group [index into ions.K] | K_init [m_e c^2] | K "
               "[m_e c^2, double from the stored u] | Gamma_p [1] | |u| [m_e c, from the "
               "stored components] | pitch [u.B/(|u||B|), 0 when B = 0] | delta [m_e c^2, "
               "pending payload] | tag [1] | r [m_e c^2, debt payload] | K0 [m_e c^2, debt_k0 "
               "payload]\n";
        H.ions_started = true;
      }
      const npart_t n_out = std::min<npart_t>(sp.npart(),
                                              static_cast<npart_t>(setup.ions_K.size()) *
                                                static_cast<npart_t>(setup.ions_per_K));
      auto ux1 = Kokkos::create_mirror_view(sp.ux1);
      auto ux2 = Kokkos::create_mirror_view(sp.ux2);
      auto ux3 = Kokkos::create_mirror_view(sp.ux3);
      auto tag = Kokkos::create_mirror_view(sp.tag);
      auto pld = Kokkos::create_mirror_view(sp.pld_r);
      Kokkos::deep_copy(ux1, sp.ux1);
      Kokkos::deep_copy(ux2, sp.ux2);
      Kokkos::deep_copy(ux3, sp.ux3);
      Kokkos::deep_copy(tag, sp.tag);
      Kokkos::deep_copy(pld, sp.pld_r);
      const double mu_d = ion_mass.at(sp.index());
      out << "# step " << step << " time " << std::setprecision(17) << t << " n " << n_out << "\n";
      for (npart_t p = 0; p < n_out; ++p) {
        const double u1 = ux1(p), u2 = ux2(p), u3 = ux3(p);
        const double K  = kernel::hadronic::kinetic_energy(mu_d, u1, u2, u3);
        const double un = std::sqrt(u1 * u1 + u2 * u2 + u3 * u3);
        const double pitch = (Bn > 0.0 && un > 0.0)
                               ? (u1 * setup.B[0] + u2 * setup.B[1] + u3 * setup.B[2]) / (un * Bn)
                               : 0.0;
        const int    g = static_cast<int>(p) / setup.ions_per_K;
        out << p << " " << g << " " << std::setprecision(17) << setup.ions_K[g] << " " << K
            << " " << 1.0 + K / setup.mu_r << " " << un << " " << pitch << " "
            << static_cast<double>(pld(p, setup.payload_pending)) << " " << tag(p) << " "
            << static_cast<double>(pld(p, setup.payload_debt)) << " "
            << static_cast<double>(pld(p, setup.payload_debt_k0)) << "\n";
      }
    }

    /**
     * @brief Every pair (both leptons) of the two pair species, at the end
     */
    void write_pairs(Domain<S, M>& domain) {
      if (pair_minus == 0) {
        return;
      }
      auto& spm = domain.species[pair_minus - 1];
      auto& spp = domain.species[pair_plus - 1];
      std::ofstream out("hadronic_pairs.txt", std::ios::trunc);
      out << "# bh_proto pairs: species e- = " << (int)pair_minus << ", e+ = " << (int)pair_plus
          << "; one row per slot k of the two species (contiguous: slot k of e- and slot k of e+ "
             "form a pair when both were born by the kernel; background particles have payload "
             "birth time 0)\n";
      out << "# columns: k | i1- | dx1- | i1+ | dx1+ | w- | w+ | ux1- | ux2- | ux3- | ux1+ | ux2+ | ux3+ | "
             "tag- | tag+ | birth- | gamma_p- | eps- | birth+ | gamma_p+ | eps+ | i1_prev- | dx1_prev- "
             "| i1_prev+ | dx1_prev+\n";
      const npart_t n = std::min(spm.npart(), spp.npart());
      auto mirror = [](const auto& v) {
        auto h = Kokkos::create_mirror_view(v);
        Kokkos::deep_copy(h, v);
        return h;
      };
      auto i1m = mirror(spm.i1), i1p = mirror(spp.i1);
      auto dxm = mirror(spm.dx1), dxp = mirror(spp.dx1);
      auto i1pm = mirror(spm.i1_prev), i1pp = mirror(spp.i1_prev);
      auto dxpm = mirror(spm.dx1_prev), dxpp = mirror(spp.dx1_prev);
      auto wm = mirror(spm.weight), wp = mirror(spp.weight);
      auto u1m = mirror(spm.ux1), u2m = mirror(spm.ux2), u3m = mirror(spm.ux3);
      auto u1p = mirror(spp.ux1), u2p = mirror(spp.ux2), u3p = mirror(spp.ux3);
      auto tm = mirror(spm.tag), tp = mirror(spp.tag);
      auto pm = mirror(spm.pld_r), pp = mirror(spp.pld_r);
      out << "# n " << n << " npart_minus " << spm.npart() << " npart_plus " << spp.npart() << "\n";
      out << std::setprecision(17);
      for (npart_t k = 0; k < n; ++k) {
        out << k << " " << i1m(k) << " " << (double)dxm(k) << " " << i1p(k) << " " << (double)dxp(k)
            << " " << (double)wm(k) << " " << (double)wp(k) << " " << (double)u1m(k) << " "
            << (double)u2m(k) << " " << (double)u3m(k) << " " << (double)u1p(k) << " "
            << (double)u2p(k) << " " << (double)u3p(k) << " " << tm(k) << " " << tp(k) << " "
            << (double)pm(k, 0) << " " << (double)pm(k, 1) << " " << (double)pm(k, 2) << " "
            << (double)pp(k, 0) << " " << (double)pp(k, 1) << " " << (double)pp(k, 2) << " "
            << i1pm(k) << " " << (double)dxpm(k) << " " << i1pp(k) << " " << (double)dxpp(k) << "\n";
      }
    }

    /**
     * @brief T13: the discrete Gauss residual R_i = (E_{i+1/2} - E_{i-1/2}) -
     *        (q0 / (B0 V0)) rho_i in 1D with rho_i the first-order (CIC) node
     *        charge of every depositing charged species (the shape of the
     *        zigzag deposit), computed in double from the arrays; E in real_t
     */
    void write_gauss(Domain<S, M>& domain, timestep_t step, double t, const char* stage) {
      if constexpr (D == Dim::_1D) {
        const int n1 = static_cast<int>(domain.mesh.n_active(in::x1));
        std::vector<double> rho(n1, 0.0);
        double              max_abs_q = 0.0;
        for (auto& sp : domain.species) {
          if (!sp.deposit() || sp.charge() == 0.0f || sp.pusher() == ParticlePusher::NONE ||
              sp.npart() == 0) {
            continue;
          }
          auto i1 = Kokkos::create_mirror_view(sp.i1);
          auto dx = Kokkos::create_mirror_view(sp.dx1);
          auto w  = Kokkos::create_mirror_view(sp.weight);
          auto tg = Kokkos::create_mirror_view(sp.tag);
          Kokkos::deep_copy(i1, sp.i1);
          Kokkos::deep_copy(dx, sp.dx1);
          Kokkos::deep_copy(w, sp.weight);
          Kokkos::deep_copy(tg, sp.tag);
          const double q = static_cast<double>(sp.charge());
          for (npart_t p = 0; p < sp.npart(); ++p) {
            if (tg(p) == ParticleTag::dead) {
              continue;
            }
            const int    i  = ((i1(p) % n1) + n1) % n1;
            const double d  = static_cast<double>(dx(p));
            const double qw = q * static_cast<double>(w(p));
            rho[i]                  += qw * (1.0 - d);
            rho[(i + 1) % n1]       += qw * d;
            max_abs_q                = std::max(max_abs_q, std::abs(qw));
          }
        }
        auto em = Kokkos::create_mirror_view(domain.fields.em);
        Kokkos::deep_copy(em, domain.fields.em);
        double max_R = 0.0, max_dR = 0.0, max_rho = 0.0, max_E = 0.0, sum_R = 0.0, sum_rho = 0.0;
        int    arg_R = -1;
        const bool first = gauss_R0.empty();
        if (first) {
          gauss_R0.assign(n1, 0.0);
        }
        for (int i = 0; i < n1; ++i) {
          const double Ep  = static_cast<double>(em(i + N_GHOSTS, em::ex1));
          const double Em  = static_cast<double>(em(((i - 1 + n1) % n1) + N_GHOSTS, em::ex1));
          const double div = Ep - Em;
          const double R   = div - gauss_coeff * rho[i];
          if (first) {
            gauss_R0[i] = R; // the initial residual (a randomly loaded background does not satisfy Gauss's law)
          }
          if (std::abs(R) > max_R) {
            max_R = std::abs(R);
            arg_R = i;
          }
          max_dR   = std::max(max_dR, std::abs(R - gauss_R0[i]));
          max_rho  = std::max(max_rho, std::abs(gauss_coeff * rho[i]));
          max_E    = std::max(max_E, std::abs(Ep));
          sum_R   += R;
          sum_rho += rho[i];
        }
        std::ofstream out("hadronic_gauss.txt", gauss_started ? std::ios::app : std::ios::trunc);
        if (!gauss_started) {
          out << "# bh_proto Gauss residual (1D): R_i = (E_{i+1/2} - E_{i-1/2}) - (q0/(B0 V0)) rho_i, "
                 "rho_i = CIC node charge of the depositing charged species (double from the arrays), "
                 "E from the real_t field array; gauss_coeff = q0/(B0 V0) = "
              << std::setprecision(17) << gauss_coeff << "\n";
          out << "# columns: step | time | stage | max|R| | max|c rho| | max|E| | max|R - R0|/max|c rho| (R0 = the "
                 "residual at the first row) | argmax i of |R| | sum_i R | sum_i rho | max|q w| (per particle) | "
                 "npart e- | npart e+ | max|R - R0|\n";
          gauss_started = true;
        }
        npart_t nm = 0, np = 0;
        if (pair_minus != 0) {
          nm = domain.species[pair_minus - 1].npart();
          np = domain.species[pair_plus - 1].npart();
        }
        out << std::setprecision(17) << step << " " << t << " " << stage << " " << max_R << " "
            << max_rho << " " << max_E << " " << (max_rho > 0.0 ? max_dR / max_rho : 0.0) << " "
            << arg_R << " " << sum_R << " " << sum_rho << " " << max_abs_q << " " << nm << " " << np << " "
            << max_dR << "\n";
      } else {
        (void)domain;
        (void)step;
        (void)t;
        (void)stage;
      }
    }

    void finish(Domain<S, M>& domain) {
      if (pair_minus != 0) {
        pair_npart_minus = domain.species[pair_minus - 1].npart();
        pair_npart_plus  = domain.species[pair_plus - 1].npart();
      }
      // pair-side sums from the arrays (dd): sum w_s (gamma- + gamma+ - eps_s) over
      // kernel-born pairs (birth payload > 0), with the same per-pair expression as
      // the kernel's E_pairs
      kernel::hadronic::dd pairs_net, pairs_E, pairs_eps, pairs_w;
      unsigned long long   n_pairs_born = 0;
      if (pair_minus != 0) {
        auto& spm = domain.species[pair_minus - 1];
        auto& spp = domain.species[pair_plus - 1];
        const npart_t n = std::min(spm.npart(), spp.npart());
        auto mirror = [](const auto& v) {
          auto h = Kokkos::create_mirror_view(v);
          Kokkos::deep_copy(h, v);
          return h;
        };
        auto wm = mirror(spm.weight), wp = mirror(spp.weight);
        auto u1m = mirror(spm.ux1), u2m = mirror(spm.ux2), u3m = mirror(spm.ux3);
        auto u1p = mirror(spp.ux1), u2p = mirror(spp.ux2), u3p = mirror(spp.ux3);
        auto pm = mirror(spm.pld_r), pp = mirror(spp.pld_r);
        for (npart_t k = 0; k < n; ++k) {
          if (!(pm(k, 0) > ZERO) || !(pp(k, 0) > ZERO)) {
            continue; // background particle slot
          }
          const double ws   = static_cast<double>(wm(k));
          const double gm   = kernel::hadronic::lepton_gamma(u1m(k), u2m(k), u3m(k));
          const double gp   = kernel::hadronic::lepton_gamma(u1p(k), u2p(k), u3p(k));
          const double E_s  = gm + gp;
          const double eps  = static_cast<double>(pm(k, 2));
          pairs_E.add_prod(ws, E_s);
          pairs_eps.add_prod(ws, eps);
          pairs_net.add_prod(ws, E_s);
          pairs_net.add_prod(-ws, eps);
          pairs_w.add(ws);
          ++n_pairs_born;
        }
        write_pairs(domain);
      }
      for (auto& sp : domain.species) {
        if (ion_mass.count(sp.index()) == 0) {
          continue;
        }
        std::uint64_t h = 0xCBF29CE484222325ULL;
        const auto    n = static_cast<std::size_t>(sp.npart());
        bhproto::mix_bits(h, n);
        bhproto::checksum_1d(h, sp.i1, n);
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          bhproto::checksum_1d(h, sp.i2, n);
        }
        if constexpr (D == Dim::_3D) {
          bhproto::checksum_1d(h, sp.i3, n);
        }
        bhproto::checksum_1d(h, sp.dx1, n);
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          bhproto::checksum_1d(h, sp.dx2, n);
        }
        if constexpr (D == Dim::_3D) {
          bhproto::checksum_1d(h, sp.dx3, n);
        }
        bhproto::checksum_1d(h, sp.ux1, n);
        bhproto::checksum_1d(h, sp.ux2, n);
        bhproto::checksum_1d(h, sp.ux3, n);
        bhproto::checksum_1d(h, sp.weight, n);
        bhproto::checksum_1d(h, sp.tag, n);
        bhproto::checksum_2d(h, sp.pld_r, n);
        const double wK_final = sum_wK_from_arrays(sp, ion_mass.at(sp.index()));
        double       wK_init  = 0.0;
        for (const double v : initial_wK[sp.index()]) {
          wK_init += v;
        }
        const auto& H = ledgers[sp.index()];
        const auto& C = H.cumulative;
        const auto  debt_final = sum_w_payload_from_arrays(sp, static_cast<unsigned short>(setup.payload_debt));
        std::printf("bh_proto: checksum species %d npart %u : %s\n", (int)sp.index(),
                    (unsigned)sp.npart(), bhproto::hex(h).c_str());
        std::printf("bh_proto: closure species %d : sum w (K_init - K_final) = %.17g, "
                    "E_drag = %.17g, difference = %.6e, relative = %.6e\n",
                    (int)sp.index(), wK_init - wK_final, C.E_drag.value(),
                    (wK_init - wK_final) - C.E_drag.value(),
                    C.E_drag.value() != 0.0 ? ((wK_init - wK_final) - C.E_drag.value()) / C.E_drag.value() : 0.0);
        // T11 array identity: sum_pairs w_s (E- + E+ - eps_s) = sum_ions w (K_init - K_final)
        //                     - sum_ions w r_stored - E_exported - E_lost_nopairs [+ debt_initial]
        {
          kernel::hadronic::dd rhs;
          rhs.add(wK_init - wK_final);
          rhs.sub(debt_final);
          rhs.sub(C.E_exported);
          rhs.sub(C.E_lost_nopairs);
          rhs.add(H.debt_initial);
          kernel::hadronic::dd res = rhs;
          res.sub(pairs_net);
          std::printf("bh_proto: array identity species %d : sum_pairs w_s (E- + E+ - eps_s) = %.17g "
                      "(%llu kernel-born pairs, sum w_pair = %.17g, sum w_s E = %.17g, sum w_s eps = %.17g), "
                      "rhs = sum w dK - sum w r - E_exported - E_lost_nopairs + debt_initial = %.17g, "
                      "residual = %.17g, E_round (ledger) = %.17g, residual - E_round = %.6e, "
                      "B_budget = %.17g, |residual|/B = %.6e\n",
                      (int)sp.index(), pairs_net.value(), n_pairs_born, pairs_w.value(), pairs_E.value(),
                      pairs_eps.value(), rhs.value(), res.value(), C.E_round.value(),
                      res.value() - C.E_round.value(), C.B_budget.value(),
                      C.B_budget.value() > 0.0 ? std::abs(res.value()) / C.B_budget.value() : 0.0);
        }
        std::printf("bh_proto: ledger species %d : E_drag %.17g E_pairs %.17g E_absorbed %.17g "
                    "E_exported %.17g E_round %.17g E_lost_nopairs %.17g pending_discarded %.17g "
                    "debt_initial %.17g debt_outstanding %.17g pending_outstanding %.17g residual %.6e "
                    "P_drag %.17g %.17g %.17g P_pairs %.17g %.17g %.17g P_absorbed %.17g %.17g %.17g "
                    "n_events %llu n_spawned %llu n_dropped %llu n_substeps %llu n_bounded %llu "
                    "n_capped_loss %llu n_closing %llu n_capped %llu n_capped_events %llu "
                    "n_above_table %llu n_latency %llu n_processed %llu n_applied %llu n_rng %llu "
                    "n_k0_trigger %llu sum_wK %.17g B_budget %.17g npart_minus %u npart_plus %u "
                    "estimate_bare %.10g estimate_sized %.10g\n",
                    (int)sp.index(), C.E_drag.value(), C.E_pairs.value(), C.E_absorbed.value(),
                    C.E_exported.value(), C.E_round.value(), C.E_lost_nopairs.value(),
                    C.pending_discarded.value(), H.debt_initial.value(), C.debt_outstanding.value(),
                    C.pending_outstanding.value(), H.identity_residual(), C.P_drag[0], C.P_drag[1],
                    C.P_drag[2], C.P_pairs[0], C.P_pairs[1], C.P_pairs[2], C.P_absorbed[0],
                    C.P_absorbed[1], C.P_absorbed[2], C.n_events, C.n_spawned, C.n_dropped,
                    C.n_substeps, C.n_bounded, C.n_capped_loss, C.n_closing, C.n_capped,
                    C.n_capped_events, C.n_above_table, C.n_latency, C.n_processed, C.n_applied,
                    C.n_rng, C.n_k0_trigger, C.sum_wK, C.B_budget.value(), (unsigned)pair_npart_minus,
                    (unsigned)pair_npart_plus, estimate_bare, estimate_sized);
        if (any_absorbing) {
          std::printf("bh_proto: exported species %d : E_exported_host %.17g E_exported %.17g difference %.6e "
                      "pending_discarded_host %.17g pending_discarded %.17g difference %.6e n_dead_with_payload %llu "
                      "(host sums over dead-tagged ions before each launch, i.e. before the sort)\n",
                      (int)sp.index(), H.exported_host.value(), C.E_exported.value(),
                      H.exported_host.value() - C.E_exported.value(), H.discarded_host.value(),
                      C.pending_discarded.value(), H.discarded_host.value() - C.pending_discarded.value(),
                      H.n_dead_host);
        }
      }
      if (pair_minus != 0) {
        for (const spidx_t sidx : { pair_minus, pair_plus }) {
          auto&         sp = domain.species[sidx - 1];
          std::uint64_t h  = 0xCBF29CE484222325ULL;
          const auto    n  = static_cast<std::size_t>(sp.npart());
          bhproto::mix_bits(h, n);
          bhproto::checksum_1d(h, sp.i1, n);
          bhproto::checksum_1d(h, sp.dx1, n);
          bhproto::checksum_1d(h, sp.ux1, n);
          bhproto::checksum_1d(h, sp.ux2, n);
          bhproto::checksum_1d(h, sp.ux3, n);
          bhproto::checksum_1d(h, sp.weight, n);
          bhproto::checksum_1d(h, sp.tag, n);
          bhproto::checksum_2d(h, sp.pld_r, n);
          std::printf("bh_proto: checksum species %d npart %u : %s\n", (int)sidx, (unsigned)n,
                      bhproto::hex(h).c_str());
        }
      }
      // T10b: no live particle of any species carries a NaN payload
      {
        unsigned long long nan_live = 0, nan_tail = 0;
        for (auto& sp : domain.species) {
          if (sp.npld_r() == 0) {
            continue;
          }
          auto hp = Kokkos::create_mirror_view(sp.pld_r);
          auto ht = Kokkos::create_mirror_view(sp.tag);
          Kokkos::deep_copy(hp, sp.pld_r);
          Kokkos::deep_copy(ht, sp.tag);
          for (npart_t p = 0; p < sp.npart(); ++p) {
            for (unsigned short c = 0; c < sp.npld_r(); ++c) {
              if (std::isnan(static_cast<double>(hp(p, c)))) {
                if (ht(p) == ParticleTag::dead) {
                  continue;
                }
                ++nan_live;
              }
            }
          }
          for (npart_t p = sp.npart(); p < sp.maxnpart(); ++p) {
            for (unsigned short c = 0; c < sp.npld_r(); ++c) {
              if (std::isnan(static_cast<double>(hp(p, c)))) {
                ++nan_tail;
              }
            }
          }
        }
        std::printf("bh_proto: payload scan : %llu NaN payload entries on live particles, %llu in the "
                    "unused tails (tails are allowed to hold anything)\n", nan_live, nan_tail);
      }
      std::printf("bh_proto: checksum fields : %s\n",
                  bhproto::hex(bhproto::checksum_fields<D>(domain.fields.em)).c_str());
      std::fflush(stdout);
    }
  };

} // namespace user

#endif // PROBLEM_GENERATOR_H
