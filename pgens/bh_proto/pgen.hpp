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
#include "archetypes/hadronic/bh_event_host.h"
#include "archetypes/hadronic/bh_interpolate.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_sampler_device.h"
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

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

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

    /**
     * @brief Order-independent 64-bit checksum of the six field components over
     *        every cell (ghosts included): the wraparound sum of per-cell hashes
     *        (the cell's local index and its six values), so that the result does
     *        not depend on the traversal order; a multi-domain run sums the
     *        per-rank values (3.1)
     */
    template <Dimension D>
    inline auto checksum_fields(const ndfield_t<D, 6>& em) -> std::uint64_t {
      std::uint64_t total = 0;
      auto          hv    = Kokkos::create_mirror_view(em);
      Kokkos::deep_copy(hv, em);
      auto cell = [&](std::uint64_t idx, auto value_of) {
        std::uint64_t h = 0xCBF29CE484222325ULL;
        mix_bits(h, idx);
        for (std::size_t c = 0; c < 6; ++c) {
          mix_bits(h, bits_of(value_of(c)));
        }
        total += splitmix64(h);
      };
      if constexpr (D == Dim::_1D) {
        for (std::size_t i = 0; i < em.extent(0); ++i) {
          cell(i, [&](std::size_t c) { return hv(i, c); });
        }
      } else if constexpr (D == Dim::_2D) {
        for (std::size_t i = 0; i < em.extent(0); ++i) {
          for (std::size_t j = 0; j < em.extent(1); ++j) {
            cell(i * em.extent(1) + j, [&](std::size_t c) { return hv(i, j, c); });
          }
        }
      } else {
        for (std::size_t i = 0; i < em.extent(0); ++i) {
          for (std::size_t j = 0; j < em.extent(1); ++j) {
            for (std::size_t k = 0; k < em.extent(2); ++k) {
              cell((i * em.extent(1) + j) * em.extent(2) + k,
                   [&](std::size_t c) { return hv(i, j, k, c); });
            }
          }
        }
      }
      return total;
    }

    /**
     * @brief The ledger reduced over the local domains and (under MPI) over the
     *        ranks (3.1): the cumulative kernel ledger, the segment-start debt
     *        and the host-side export sums, all global; written by rank 0
     */
    struct GlobalLedger {
      kernel::hadronic::LossLedger C {};
      kernel::hadronic::dd         debt_initial {};
      kernel::hadronic::dd         exported_host {};
      kernel::hadronic::dd         discarded_host {};
      unsigned long long           n_dead_host { 0 };

      auto identity_residual() const -> double {
        kernel::hadronic::dd r;
        r.add(C.E_drag);
        r.add(debt_initial);
        r.sub(C.E_pairs);
        r.add(C.E_absorbed);
        r.sub(C.E_round);
        r.sub(C.E_stochastic);
        r.sub(C.debt_outstanding);
        r.sub(C.E_exported);
        r.sub(C.E_lost_nopairs);
        return r.value();
      }
    };

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
      unsigned long long           n_bound_warnings { 0 };      // event mode: on_bound = warn messages
      kernel::hadronic::dd         exported_host {};  // sum w r over dead ions, host, before each launch
      kernel::hadronic::dd         discarded_host {}; // sum w delta over dead ions, host, before each launch
      unsigned long long           n_dead_host { 0 }; // dead ions seen with a non-zero debt or pending request

      // snapshots are summed over the local domains of one step and replaced at the next
      kernel::hadronic::dd snap_pending {}, snap_debt {};
      double               snap_sum { 0.0 };
      unsigned long long   snap_alive { 0 };
      long long            snap_step { -1 };

      void accumulate(const kernel::hadronic::LossLedger& L, timestep_t step) {
        if (static_cast<long long>(step) != snap_step) {
          snap_pending = kernel::hadronic::dd {};
          snap_debt    = kernel::hadronic::dd {};
          snap_sum     = 0.0;
          snap_alive   = 0;
          snap_step    = static_cast<long long>(step);
        }
        snap_pending += L.pending_outstanding;
        snap_debt    += L.debt_outstanding;
        snap_sum     += L.sum_wK;
        snap_alive   += L.n_alive;
        cumulative += L;
        cumulative.pending_outstanding = snap_pending;
        cumulative.debt_outstanding    = snap_debt;
        cumulative.sum_wK              = snap_sum;
        cumulative.n_alive             = snap_alive;
      }

      /**
       * @brief The identity residual (II.2 step 6, spec v3.3), in dd:
       *        E_drag + debt_initial - [(E_pairs - E_absorbed) + E_round +
       *        E_stochastic + debt_outstanding + E_exported + E_lost_nopairs]
       */
      auto identity_residual() const -> double {
        kernel::hadronic::dd r;
        const auto&          C = cumulative;
        r.add(C.E_drag);
        r.add(debt_initial);
        r.sub(C.E_pairs);
        r.add(C.E_absorbed);
        r.sub(C.E_round);
        r.sub(C.E_stochastic);
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
    int           mpi_rank { 0 };
    int           mpi_size { 1 };
    unsigned int  n_domains_total { 1 };
    unsigned long long spawned_at_previous_poststep { 0 }; // production placement: n_spawned at the previous post-step

    // C-MPI-2: a rank-local failure under an "error" policy is recorded here and
    // acted upon at the common point at the end of HadronicStage (every rank,
    // every step): one MPI_Allreduce of the flag, then every rank writes its
    // ledger row (collective, consistent), rank 0 prints the gathered codes and
    // every rank aborts. The local branches never enter a collective.
    struct FailRecord {
      long long code { 0 };    // 0 none, 1 on_capacity, 2 on_max_events, 3 on_bound (event mode)
      long long step { -1 };
      long long species { 0 };
      long long count { 0 };   // n_dropped (1) or capped ion-steps (2) of the failing launch
    };
    FailRecord fail_local {};

    // tables
    std::shared_ptr<const arch::bh::PhotonBath> bath;
    arch::bh::Quadrature                        quadrature;
    arch::bh::HadronicTablesHost                tables_host;
    arch::bh::SamplerTables                     sampler_tables; // restframe spectrum (step 4)
    arch::bh::HadronicTables                    tables;

    // R6 Gauss-law sample (test.gauss): one residual evaluation of the domain
    struct GaussSample {
      double             max_R { 0.0 }, max_dR { 0.0 }, max_rho { 0.0 }, max_E { 0.0 }, sum_R { 0.0 }, sum_rho { 0.0 };
      double             max_abs_q { 0.0 }, ratio { 0.0 };
      int                arg_R { -1 };
      unsigned long long n_depositing { 0 };
      npart_t            nm { 0 }, np { 0 };
    };

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
    bool                                       event_restart_checked { false }; // step 5: payload scan done
    std::vector<double>                        gauss_R0;                 // Gauss residual per node at the first sample
    bool                                       gauss_first_birth_done { false };
    GaussSample                                gauss_prev {};             // production placement: previous post-step sample
    bool                                       gauss_prev_valid { false };
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
      // weights flag rule (audit F3, 2026-10-29; spec III.7): the flag may be off
      // only when every stored weight is 1 (not with the budget source, not with
      // pair_weight != 1 in the poisson source, not with ions.weight != 1)
      try {
        bhproto::check_weights_flag(setup, use_weights);
      } catch (const std::invalid_argument& e) {
        raise::Error(e.what(), HERE);
      }
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
      // R5 / decision (d): the prototype runs exactly one hadronic ion species (one
      // pair of lepton species, one ledger, one array closure)
      if (setup.species.size() != 1) {
        std::string listed;
        for (const int idx : setup.species) {
          listed += (listed.empty() ? "" : ", ") + std::to_string(idx);
        }
        raise::Error(fmt::format("bh_proto: setup.hadronic.species lists %zu species (%s); the prototype "
                                 "accepts exactly one hadronic ion species per run (decision (d))",
                                 setup.species.size(), listed.c_str()),
                     HERE);
      }
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

      // ---- MPI readiness (3.1) -------------------------------------------------
#if defined(MPI_ENABLED)
      MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
      // C-MPI-3: the guard is restricted to the post-step placement (the unsafe
      // harvest), for any rank count (a one-rank periodic self-exchange can refill
      // a dead slot too); the production placement is allowed with absorbing
      // boundaries and any number of domains (C-MPI-1 clears the remnants)
      raise::ErrorIf(bhproto::poststep_harvest_refused(any_absorbing, true, metadomain.ndomains(), setup.placement),
                     fmt::format("bh_proto: absorbing particle boundaries with hadronic.placement = \"poststep\" are "
                                 "refused in an MPI build (%u domain%s, %d rank%s): the particle exchange can refill "
                                 "a dead slot before the post-step hook harvests it (decision (e)); use "
                                 "hadronic.placement = \"production\"",
                                 metadomain.ndomains(), metadomain.ndomains() == 1 ? "" : "s", mpi_size,
                                 mpi_size == 1 ? "" : "s"),
                     HERE);
#endif
      n_domains_total = metadomain.ndomains();

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
          if (setup.spectrum == "restframe") {
            // the rest-frame sampler tables (step 4): built on the host, mirrored
            // to the device in single precision (contract section 3, <= 64 MB)
            sampler_tables = arch::bh::build_sampler_tables(*bath, tables_host);
            raise::ErrorIf(sampler_tables.bytes > 64u * 1024u * 1024u, "bh_proto: the rest-frame sampler tables exceed 64 MB", HERE);
            std::printf("bh_proto: rest-frame sampler tables: %d eps' nodes, %d x %d cells, %d cumulative nodes, "
                        "%s photon-energy table, %.1f MB on the device\n",
                        sampler_tables.n_k, sampler_tables.n_s, sampler_tables.n_tau, sampler_tables.n_cum,
                        sampler_tables.mono ? "no" : (std::to_string(sampler_tables.n_gamma) + " x " +
                                                      std::to_string(sampler_tables.n_eps)).c_str(),
                        static_cast<double>(sampler_tables.bytes) * 1.0e-6);
          }
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
        if (setup.event_mode()) {
          event_mode_report();
        }
        if (setup.pairs_enabled()) {
          capacity_estimate();
        }
      } else {
        std::printf("bh_proto: hadronic.enabled = false: no tables, no stage\n");
      }
      std::printf("bh_proto: real_t = %s, dt = %.17g, skindepth0 = %.17g, omegaB0 = %.17g, "
                  "max_steps = %u, stage %s, mode %s, source %s, spectrum %s, on_bound %s, "
                  "t_latency = %.17g (%s), L_min = %.17g, seed %s%llu, start_step %u\n",
                  sizeof(real_t) == 4 ? "float" : "double", dt, skindepth0, omegaB0, (unsigned)max_steps,
                  stage_active ? "active" : (setup.enabled ? "not launched (tau0 = 0)"
                                                           : "disabled"),
                  setup.mode.c_str(), setup.event_mode() ? "(none: event mode)" : setup.source.c_str(),
                  setup.spectrum.c_str(), setup.on_bound.c_str(), t_latency,
                  setup.t_latency_auto ? "auto" : "given", L_min,
                  seed_set ? "simulation.seed = " : "default ", (unsigned long long)base_seed,
                  (unsigned)start_step);
      std::printf("bh_proto: hadronic.placement = %s: the stage runs %s; MPI %s (%d rank%s, %u domain%s); "
                  "ledger reduced over the domains%s, written by rank 0\n",
                  setup.placement.c_str(),
                  hadronic_placement_production()
                    ? "at the production placement (srpic::HadronicInteractions: after the deposit, before "
                      "the particle exchange)"
                    : "at the post-step placement (CustomPostStep, after the exchange)",
#if defined(MPI_ENABLED)
                  "enabled",
#else
                  "not compiled",
#endif
                  mpi_size, mpi_size == 1 ? "" : "s", n_domains_total, n_domains_total == 1 ? "" : "s",
                  mpi_size > 1 ? " and over the ranks (MPI_Allreduce)" : "");
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
     * @brief Event mode (step 5, contract 2026-10-29 section 3): the startup
     *        report and the refusals, computed on the host tables of step 1.
     *
     *   a_max(Gamma) = f_max K / D_max,unweighted(Gamma) (bh_event_host.h):
     *   printed at gamma_p_typical, at the envelope ends, at the table edges
     *   and at every initial ion energy; the run is refused when pair_weight
     *   exceeds a_max anywhere on the active band: gamma_p_typical, the
     *   envelope sampled on a log grid, the initial energies (which are
     *   certainly in the run), and every table node from the lower edge up
     *   to the largest of these (the cooling path of every ion of the band;
     *   no monotonicity of a_max in Gamma is assumed, bh_event_host.h)
     *   [proposal of the contract]. Fact: with a <= a_max no outcome of the
     *   unweighted law at that Gamma can violate the per-event bound a D <=
     *   f_max K.
     *   lambda_max = max of coeff J dt / a over the same active band
     *   (contract section 3 says "over the table"; the table reaches the
     *   static-source envelope, Gamma_p = 92 for eps0 = 0.5, where J is
     *   17 times its value at Gamma_p = 6, so the table-wide maximum would
     *   refuse every run whose ions never reach the envelope: deviation,
     *   reported as an open question; the table-wide value is printed for
     *   information): refused at or above 1 (outside the intended
     *   frozen-hazard regime); above 0.1 the on_max_events policy applies
     *   (error refuses, warn continues) [proposal].
     *   ulp-energy ratio: min over the table of a <D> / (ulp(u) mu_d u /
     *   gamma_d) for each hadronic ion species; refused below 16 [proposal]:
     *   the per-event rounding residual must be a small fraction of the
     *   removed energy.
     *   Information only: the per-step mean loss over the bound, max over the
     *   table of coeff I dt / (f_max K) (the per-step condition of spec III.5
     *   is enforced by the kernel).
     */
    void event_mode_report() {
      namespace ev = arch::bh::event;
      const auto&  T       = tables_host;
      const double coeff   = setup.tau0 / skindepth0;
      const double a       = setup.pair_weight;
      const double eps_max = bath->eps_max();
      auto a_max_at = [&](double G) {
        return ev::a_max(G, eps_max, setup.mu_r, setup.f_max);
      };
      auto show = [&](const char* label, double G) {
        const double K  = setup.mu_r * (G - 1.0);
        const double Dm = ev::dmax_unweighted(G, eps_max);
        double       Dmean = 0.0, em = 0.0;
        arch::bh::interpolate_outcome_host(T, G, Dmean, em);
        const double J   = arch::bh::interpolate_rate_host(T, G);
        const double lam = coeff * J * dt / a;
        std::printf("bh_proto: event mode: %s: Gamma_p = %.10g, K = %.10g, D_max,unweighted = %.10g, "
                    "a_max = f_max K / D_max = %.6g (a <D> / (f_max K) = %.4e with <D> = %.6g), "
                    "lambda = R dt / a = %.4e\n",
                    label, G, K, Dm, a_max_at(G), Dmean > 0.0 ? a * Dmean / (setup.f_max * K) : 0.0, Dmean,
                    lam);
      };
      std::printf("bh_proto: event mode: pair_weight a = %.10g, f_max = %.10g, eps_max of the retained bath = "
                  "%.10g, max_events_per_step = %d, on_bound = %s, on_max_events = %s\n",
                  a, setup.f_max, eps_max, setup.max_events_per_step, setup.on_bound.c_str(),
                  setup.on_max_events.c_str());
      show("table lower edge", T.gamma_min);
      show("table upper edge", T.gamma_max);
      // the active band: every point is checked explicitly (a_max decreases
      // with Gamma, so the upper end binds; the explicit check does not rely on it)
      double      a_max_band = std::numeric_limits<double>::infinity();
      double      gamma_bind = 0.0;
      std::string where_bind;
      double      lambda_band = 0.0, gamma_lambda_band = 0.0;
      std::string where_lambda;
      auto band = [&](const char* label, double G) {
        const double am = a_max_at(G);
        if (am < a_max_band) {
          a_max_band = am;
          gamma_bind = G;
          where_bind = label;
        }
        const double lam = coeff * arch::bh::interpolate_rate_host(T, G) * dt / a;
        if (lam > lambda_band) {
          lambda_band       = lam;
          gamma_lambda_band = G;
          where_lambda      = label;
        }
      };
      if (setup.gamma_p_typical > 0.0) {
        show("gamma_p_typical", setup.gamma_p_typical);
        band("gamma_p_typical", setup.gamma_p_typical);
      }
      if (setup.has_envelope) {
        show("envelope gamma_min", setup.envelope_gamma_min);
        show("envelope gamma_max", setup.envelope_gamma_max);
        const int    nq = 64;
        const double l0 = std::log(setup.envelope_gamma_min), l1 = std::log(setup.envelope_gamma_max);
        for (int i = 0; i <= nq; ++i) {
          band("envelope", std::exp(l0 + (l1 - l0) * static_cast<double>(i) / nq));
        }
      }
      double band_top = setup.gamma_p_typical;
      if (setup.has_envelope) {
        band_top = std::max(band_top, setup.envelope_gamma_max);
      }
      for (std::size_t g = 0; g < setup.ions_K.size(); ++g) {
        const double G = 1.0 + setup.ions_K[g] / setup.mu_r;
        show(fmt::format("initial ions.K[%zu]", g).c_str(), G);
        band("initial ions.K", G);
        band_top = std::max(band_top, G);
      }
      // the cooling path: every table node below the top of the band
      std::size_t n_path = 0;
      for (std::size_t i = 0; i < T.size(); ++i) {
        if (T.gamma[i] <= band_top) {
          band("table node on the cooling path", T.gamma[i]);
          ++n_path;
        }
      }
      std::printf("bh_proto: event mode: a_max over the active band (top Gamma_p = %.10g, %zu table nodes on the "
                  "cooling path) = %.6g at Gamma_p = %.10g (%s): %s\n",
                  band_top, n_path, a_max_band, gamma_bind, where_bind.c_str(),
                  a <= a_max_band ? "pair_weight admitted" : "pair_weight EXCEEDS a_max");
      std::fflush(stdout);
      if (setup.test_bypass_amax) {
        std::printf("bh_proto: event mode: setup.test.bypass_amax = true: the refusal pair_weight > a_max is SKIPPED "
                    "(test hook for the on_bound policy; never a production setting)\n");
        std::fflush(stdout);
      } else {
        raise::ErrorIf(!(a <= a_max_band),
                       fmt::format("bh_proto: setup.pair_weight = %.10g exceeds a_max = f_max K / D_max,unweighted = "
                                   "%.6g at Gamma_p = %.10g (%s) on the active band: an event with a D > f_max K "
                                   "is possible there (contract section 3; lower pair_weight or raise f_max)",
                                   a, a_max_band, gamma_bind, where_bind.c_str()),
                       HERE);
      }
      // per ion species: the ulp-energy ratio and lambda_max over the table
      double lambda_max = 0.0, gamma_lambda = 0.0;
      for (const auto sp : ion_species) {
        const double mu_d = ion_mass.at(sp);
        const auto   n    = ev::table_numbers<real_t>(T, coeff, dt, a, mu_d);
        lambda_max        = n.lambda_max;
        gamma_lambda      = n.gamma_at_lambda_max;
        std::printf("bh_proto: event mode: species %d (mu_d = %.10g): min over the table of a <D> / ulp-energy = "
                    "%.6g at Gamma_p = %.10g (<D> = %.6g, ulp-energy = ulp(u) mu_d u / gamma_d = %.4e, real_t %s); "
                    "registered minimum %.0f: %s\n",
                    (int)sp, mu_d, n.ulp_ratio_min, n.gamma_at_ulp_ratio_min, n.dmean_at_min, n.dK_ulp_at_min,
                    sizeof(real_t) == 4 ? "float" : "double", ev::ULP_RATIO_MIN,
                    n.ulp_ratio_min >= ev::ULP_RATIO_MIN ? "admitted" : "BELOW");
        std::fflush(stdout);
        raise::ErrorIf(!(n.ulp_ratio_min >= ev::ULP_RATIO_MIN),
                       fmt::format("bh_proto: setup.pair_weight = %.10g: a <D> / ulp-energy = %.6g at Gamma_p = %.10g "
                                   "(species %d, mu_d = %.10g) is below the registered minimum %.0f: the per-event "
                                   "rounding of the stored momentum would not be small against the removed energy "
                                   "(contract section 2 e)",
                                   a, n.ulp_ratio_min, n.gamma_at_ulp_ratio_min, (int)sp, mu_d, ev::ULP_RATIO_MIN),
                       HERE);
      }
      // the per-step mean loss over the bound (information)
      double loss_ratio_max = 0.0, gamma_loss = 0.0;
      for (std::size_t i = 0; i < T.size(); ++i) {
        const double K = T.mu_r * (T.gamma[i] - 1.0);
        if (K > 0.0) {
          const double r = coeff * T.I[i] * dt / (setup.f_max * K);
          if (r > loss_ratio_max) {
            loss_ratio_max = r;
            gamma_loss     = T.gamma[i];
          }
        }
      }
      std::printf("bh_proto: event mode: lambda_max = max over the active band of coeff J dt / a = %.6g at Gamma_p = "
                  "%.10g (%s) (intended regime < 1; reported above %.1f under on_max_events = %s); over the whole "
                  "table %.6g at Gamma_p = %.10g (information: the table reaches the envelope); max over the table "
                  "of <D> R dt / (f_max K) = %.4e at Gamma_p = %.10g\n",
                  lambda_band, gamma_lambda_band, where_lambda.c_str(), ev::LAMBDA_WARN, setup.on_max_events.c_str(),
                  lambda_max, gamma_lambda, loss_ratio_max, gamma_loss);
      std::fflush(stdout);
      raise::ErrorIf(!(lambda_band < ev::LAMBDA_MAX),
                     fmt::format("bh_proto: event mode: lambda_max = coeff J dt / a = %.6g at Gamma_p = %.10g (%s) is "
                                 "not below 1: the frozen-hazard event count is outside its intended regime (reduce "
                                 "tau0 or dt, or raise pair_weight) [setup.on_max_events = %s]",
                                 lambda_band, gamma_lambda_band, where_lambda.c_str(), setup.on_max_events.c_str()),
                     HERE);
      if (lambda_band > ev::LAMBDA_WARN) {
        const auto msg = fmt::format(
          "bh_proto: event mode: lambda_max = %.6g at Gamma_p = %.10g (%s) exceeds %.1f: more than one interaction "
          "per step is frequent and the start-of-step rate is a coarser approximation [setup.on_max_events = %s]",
          lambda_band, gamma_lambda_band, where_lambda.c_str(), ev::LAMBDA_WARN, setup.on_max_events.c_str());
        if (setup.on_max_events == "error") {
          raise::Error(msg, HERE);
        }
        std::printf("%s (warning, continuing)\n", msg.c_str());
        std::fflush(stdout);
      }
    }

    /**
     * @brief Expected spawn rate per active ion per unit time at Gamma (III.8):
     *        budget: coeff I / r_th with r_th the actual minimum of its terms;
     *        poisson and event: coeff J / pair_weight (one macro-pair of weight
     *        a per interaction, interactions at the rate R = coeff J)
     */
    auto spawn_rate_at(double Gp) const -> double {
      const auto&  T     = tables_host;
      const double coeff = setup.tau0 / skindepth0;
      const double I     = arch::bh::interpolate_loss_host(T, Gp);
      if (!(I > 0.0)) {
        return 0.0;
      }
      if (setup.event_mode() || setup.source == "poisson") {
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
      // every local domain injects its own ions (the estimate and maxnpart are per domain)
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
        if (ht(p) != ParticleTag::alive) {
          continue; // dead remnants (absorbed or sent) and send-tagged copies are not counted
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
        if (ht(p) != ParticleTag::alive) {
          continue; // alive only (C-MPI-1): a transported copy's remnant is never counted
        }
        acc.add_prod(static_cast<double>(hw(p)), static_cast<double>(hp(p, slot)));
      }
      return acc;
    }

    /* -------------------------------------------------------------------- */
    /* the stage                                                             */
    /* -------------------------------------------------------------------- */
    /**
     * @brief The hadronic interaction stage on one local domain: the dead-ion
     *        harvest (host sums, non-MPI only), one launch of the kernel per
     *        hadronic ion species with the pair destinations, the host-side
     *        insertion bookkeeping and the failure policies. Called from
     *        CustomPostStep (placement = poststep) or by the engine after the
     *        deposit and before the particle exchange (placement = production,
     *        srpic::HadronicInteractions).
     */
    inline void HadronicStage(timestep_t step, simtime_t /*time*/, Domain<S, M>& domain) {
      const double t_after = static_cast<double>(step + 1) * dt; // t^{n+1}
      if (stage_active && setup.event_mode() && !event_restart_checked) {
        // contract section 3 (step 5): the event mode is stateless per ion and
        // cannot pay a pending request or a debt carried by a continuous-mode
        // checkpoint; the three hadronic slots of every alive ion must read 0.
        // Executed once per rank (one domain per rank), reduced over the ranks
        // before the refusal so that every rank aborts together.
        event_restart_checked = true;
        unsigned long long n_nonzero = 0;
        for (auto& sp : domain.species) {
          if (ion_mass.count(sp.index()) == 0 || sp.npart() == 0 || sp.npld_r() == 0) {
            continue;
          }
          auto ht = Kokkos::create_mirror_view(sp.tag);
          auto hp = Kokkos::create_mirror_view(sp.pld_r);
          Kokkos::deep_copy(ht, sp.tag);
          Kokkos::deep_copy(hp, sp.pld_r);
          const unsigned short slots[3] = { static_cast<unsigned short>(setup.payload_pending),
                                            static_cast<unsigned short>(setup.payload_debt),
                                            static_cast<unsigned short>(setup.payload_debt_k0) };
          for (npart_t p = 0; p < sp.npart(); ++p) {
            if (ht(p) == ParticleTag::dead) {
              continue;
            }
            for (const auto c : slots) {
              if (c < sp.npld_r() && hp(p, c) != ZERO) {
                ++n_nonzero;
                break;
              }
            }
          }
        }
#if defined(MPI_ENABLED)
        MPI_Allreduce(MPI_IN_PLACE, &n_nonzero, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        std::printf("bh_proto: event mode: payload scan at step %u (start_step %u): %llu alive ions carry a non-zero "
                    "hadronic payload (pending, debt or debt_k0): %s\n",
                    (unsigned)step, (unsigned)start_step, n_nonzero,
                    bhproto::event_restart_refused(setup.mode, n_nonzero) ? "REFUSED" : "clean");
        std::fflush(stdout);
        raise::ErrorIf(bhproto::event_restart_refused(setup.mode, n_nonzero),
                       fmt::format("bh_proto: mode = event entered with %llu alive ions carrying a non-zero hadronic "
                                   "payload (start_step %u): a continuous-mode checkpoint carries a pending request "
                                   "or a debt that the event mode cannot pay (contract section 3); restart in "
                                   "mode = continuous or start a fresh segment",
                                   n_nonzero, (unsigned)start_step),
                       HERE);
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
        prm.spectrum = setup.spectrum == "threshold" ? static_cast<int>(kernel::hadronic::PairSpectrum::Threshold)
                       : setup.spectrum == "restframe"
                         ? static_cast<int>(kernel::hadronic::PairSpectrum::RestFrame)
                         : static_cast<int>(kernel::hadronic::PairSpectrum::Delta);
        prm.r_th_factor         = setup.r_th_factor;
        prm.f_r                 = setup.f_r;
        prm.t_latency           = t_latency;
        prm.pair_weight         = setup.pair_weight;
        prm.max_events_per_step = setup.max_events_per_step;
        prm.kick                = setup.test_kick;
        prm.max_sampler_attempts = setup.test_max_sampler_attempts;
        prm.mode = setup.event_mode() ? static_cast<int>(kernel::hadronic::LossMode::Event)
                                      : static_cast<int>(kernel::hadronic::LossMode::Continuous);

        for (auto& sp : domain.species) {
          if (ion_mass.count(sp.index()) == 0 || sp.npart() == 0) {
            continue;
          }
          auto& H = ledgers[sp.index()];
          // T-exp: host-side sums over the ions tagged dead this step (absorbing
          // walls), taken before the launch and before the sort; the kernel's
          // E_exported / pending_discarded must reproduce them exactly.
          // R8 / decision (e): this harvest (and the kernel's dead-ion branch at the
          // post-step placement) is valid for non-MPI builds only: under MPI the
          // particle exchange of CommunicateParticles refills dead slots with
          // received particles before this hook runs, and the output path may
          // compact the arrays; in production the harvest happens at the production
          // placement (after the deposit, before CommunicateParticles), where the
          // absorbed ions are still in place
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
              targets, *pools[sp.index()], sampler_tables },
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
            pair_npart_minus = spm.npart(); // snapshots for the ledger row
            pair_npart_plus  = spp.npart();
            raise::ErrorIf(static_cast<unsigned long long>(n_injected) != L.n_spawned,
                           fmt::format("bh_proto: internal: n_injected %u != n_spawned %llu",
                                       (unsigned)n_injected, L.n_spawned),
                           HERE);
            H.accumulate(L, step);
            if (L.n_dropped > 0) {
              const auto msg = fmt::format(
                "bh_proto: on_capacity: pair capacity exhausted at step %u (species %d): "
                "capacity = %u, tickets = %u, inserted = %u, n_dropped = %llu this step; e- npart %u / "
                "maxnpart %u, e+ npart %u / maxnpart %u; the debt is kept [setup.on_capacity = %s]",
                (unsigned)step, (int)sp.index(), (unsigned)targets.capacity, (unsigned)tickets,
                (unsigned)n_injected, L.n_dropped, (unsigned)spm.npart(), (unsigned)spm.maxnpart(),
                (unsigned)spp.npart(), (unsigned)spp.maxnpart(), setup.on_capacity.c_str());
              if (setup.on_capacity == "error") {
                // C-MPI-2: no collective and no abort here (rank-local path); the
                // failure is recorded and acted upon at the end of the stage
                std::printf("%s (error: the run aborts after every rank wrote its ledger row)\n", msg.c_str());
                std::fflush(stdout);
                if (fail_local.code == 0) {
                  fail_local = FailRecord { 1, static_cast<long long>(step), static_cast<long long>(sp.index()),
                                            static_cast<long long>(L.n_dropped) };
                }
              } else {
                if (H.n_capacity_warnings < 5) {
                  std::printf("%s (warning, continuing)\n", msg.c_str());
                }
                H.n_capacity_warnings += 1;
              }
            }
            if (L.n_capped_events > 0) {
              const auto msg = fmt::format(
                "bh_proto: on_max_events: %llu ion-steps exceeded max_events_per_step = %d at step %u "
                "(species %d) [setup.on_max_events = %s]",
                L.n_capped_events, setup.max_events_per_step, (unsigned)step, (int)sp.index(),
                setup.on_max_events.c_str());
              if (setup.on_max_events == "error") {
                std::printf("%s (error: the run aborts after every rank wrote its ledger row)\n", msg.c_str());
                std::fflush(stdout);
                if (fail_local.code == 0) {
                  fail_local = FailRecord { 2, static_cast<long long>(step), static_cast<long long>(sp.index()),
                                            static_cast<long long>(L.n_capped_events) };
                }
              } else {
                if (H.n_max_events_warnings < 5) {
                  std::printf("%s (warning, continuing)\n", msg.c_str());
                }
                H.n_max_events_warnings += 1;
              }
            }
            // step 5: the per-event bound of the event mode (contract section 2 b):
            // the kernel skipped and counted the events with a D > f_max K; error
            // aborts at the common point, warn continues (the realised rate is
            // biased by the skipped fraction, which n_bounded / n_events measures)
            if (setup.event_mode() && L.n_bounded > 0) {
              const auto msg = fmt::format(
                "bh_proto: on_bound: %llu of %llu events had pair_weight D > f_max K at step %u (species %d) and "
                "were skipped (no pair, no recoil) [setup.on_bound = %s, pair_weight = %.10g, f_max = %.10g]",
                L.n_bounded, L.n_events, (unsigned)step, (int)sp.index(), setup.on_bound.c_str(),
                setup.pair_weight, setup.f_max);
              if (setup.on_bound == "error") {
                std::printf("%s (error: the run aborts after every rank wrote its ledger row)\n", msg.c_str());
                std::fflush(stdout);
                if (fail_local.code == 0) {
                  fail_local = FailRecord { 3, static_cast<long long>(step), static_cast<long long>(sp.index()),
                                            static_cast<long long>(L.n_bounded) };
                }
              } else {
                if (H.n_bound_warnings < 5) {
                  std::printf("%s (warning, continuing)\n", msg.c_str());
                }
                H.n_bound_warnings += 1;
              }
            }
          } else {
            H.accumulate(L, step);
          }
        }
        // ---- C-MPI-2: the common point, executed once per step by every rank in
        // both placements (the species loop above may be skipped on a rank with no
        // ions; this is not): reduce the failure flag, and if any rank failed,
        // every rank writes its ledger row, rank 0 prints the gathered codes and
        // every rank aborts
        {
          int any_failed = fail_local.code != 0 ? 1 : 0;
#if defined(MPI_ENABLED)
          MPI_Allreduce(MPI_IN_PLACE, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
          if (any_failed != 0) {
            for (auto& sp : domain.species) {
              if (ion_mass.count(sp.index()) != 0) {
                write_ledger(sp, step, t_after); // collective under MPI, consistent on every rank
              }
            }
            long long              local[4] = { fail_local.code, fail_local.step, fail_local.species, fail_local.count };
            std::vector<long long> all(4 * static_cast<std::size_t>(mpi_size), 0);
#if defined(MPI_ENABLED)
            MPI_Gather(local, 4, MPI_LONG_LONG, all.data(), 4, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
#else
            std::copy(local, local + 4, all.begin());
#endif
            if (mpi_rank == 0) {
              for (int r = 0; r < mpi_size; ++r) {
                const long long* rec = all.data() + 4 * r;
                if (rec[0] != 0) {
                  std::printf("bh_proto: failure on rank %d: %s at step %lld (species %lld), count %lld\n", r,
                              rec[0] == 1   ? "on_capacity (pair capacity exhausted; n_dropped)"
                              : rec[0] == 2 ? "on_max_events (ion-steps exceeding max_events_per_step)"
                                            : "on_bound (event mode: events with pair_weight D > f_max K; n_bounded)",
                              rec[1], rec[2], rec[3]);
                }
              }
              std::fflush(stdout);
            }
            raise::Error(fmt::format("bh_proto: %s on %s at step %u: every rank wrote its ledger row; the run aborts "
                                     "[setup.on_capacity = %s, setup.on_max_events = %s, setup.on_bound = %s]",
                                     fail_local.code == 1   ? "on_capacity: pair capacity exhausted"
                                     : fail_local.code == 2 ? "on_max_events: max_events_per_step exceeded"
                                     : fail_local.code == 3 ? "on_bound: an event with pair_weight D > f_max K"
                                                            : "a failure on another rank",
                                     fail_local.code != 0 ? "this rank" : "another rank", (unsigned)step,
                                     setup.on_capacity.c_str(), setup.on_max_events.c_str(), setup.on_bound.c_str()),
                         HERE);
          }
        }
      }
    }

    /**
     * @brief The placement switch read by the engine (traits::pgen::HasHadronicStage)
     */
    auto hadronic_placement_production() const -> bool {
      return setup.placement == "production";
    }

    inline void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      const double t_after = static_cast<double>(step + 1) * dt; // t^{n+1}
      const bool   last    = (step + 1 >= max_steps);
      const bool   row     = ((step + 1) % static_cast<timestep_t>(setup.ledger_every) == 0u) || last;
      // R6: the "before / after the first births" samples bracket the launch that
      // spawns the first pair: the pre-launch residual is computed every step until
      // then and written only when the launch that follows it spawns (post-step
      // placement; at the production placement the stage ran inside step_forward and
      // only the "after" sample of the step is available)
      GaussSample before_first {};
      bool        have_before = false;
      if (setup.test_gauss && !gauss_first_birth_done) {
        if (!hadronic_placement_production()) {
          before_first = gauss_sample(domain);
          have_before  = true;
        } else if (gauss_prev_valid) {
          // production placement: the sample taken at the end of the previous
          // post-step (no births yet) is the state before this step's stage
          before_first = gauss_prev;
          have_before  = true;
        }
      }
      unsigned long long spawned_before = 0;
      for (const auto& kv : ledgers) {
        spawned_before += kv.second.cumulative.n_spawned;
      }
      if (!hadronic_placement_production()) {
        HadronicStage(step, time, domain);
      } else if (setup.test_gauss && !gauss_first_birth_done) {
        // the stage ran at the production placement: count its spawns as "before" = the
        // previous step's total so that the first-birth step is still detected below
        spawned_before = spawned_at_previous_poststep;
      }
      if (setup.test_gauss && !gauss_first_birth_done) {
        unsigned long long spawned_after = 0;
        for (const auto& kv : ledgers) {
          spawned_after += kv.second.cumulative.n_spawned;
        }
        if (spawned_after > spawned_before) {
          // the first births happened in this launch: write the bracketing samples
          if (have_before) {
            write_gauss_sample(before_first, step, t_after, "before_first_births");
          }
          const auto after = gauss_sample(domain);
          write_gauss_sample(after, step, t_after, "after_first_births");
          raise::ErrorIf(after.n_depositing <= before_first.n_depositing,
                         "bh_proto: test.gauss: the depositing particle count did not increase at the first births",
                         HERE);
          std::printf("bh_proto: test.gauss: first births at step %u: depositing particles %llu -> %llu, "
                      "max|R - R0| / max|c rho| %.6e -> %.6e\n",
                      (unsigned)step, before_first.n_depositing, after.n_depositing, before_first.ratio, after.ratio);
          gauss_first_birth_done = true;
        }
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
      {
        unsigned long long spawned_total = 0;
        for (const auto& kv : ledgers) {
          spawned_total += kv.second.cumulative.n_spawned;
        }
        spawned_at_previous_poststep = spawned_total;
      }
      if (setup.test_gauss && !gauss_first_birth_done && hadronic_placement_production()) {
        gauss_prev       = gauss_sample(domain);
        gauss_prev_valid = true;
      }
      if (last) {
        finish(domain);
      }
    }

    /* ---------------------------------------------------------------------- */
    /* 3.1 MPI readiness: reductions, order-independent checksums, file names   */
    /* ---------------------------------------------------------------------- */

    /// sum of a 64-bit counter over the ranks (identity without MPI)
    auto mpi_sum_u64(unsigned long long x) const -> unsigned long long {
#if defined(MPI_ENABLED)
      MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
      return x;
    }

    auto mpi_sum_npart(npart_t n) const -> unsigned long long {
      return mpi_sum_u64(static_cast<unsigned long long>(n));
    }

    /// per-rank output file name: base[_rank<r>].txt (the rank suffix only under MPI with > 1 rank)
    auto per_rank_name(const std::string& base) const -> std::string {
      if (mpi_size > 1) {
        return base + "_rank" + std::to_string(mpi_rank) + ".txt";
      }
      return base + ".txt";
    }

    /**
     * @brief The ledger reduced over the ranks (3.1): every dd entry is reduced
     *        as its (hi, lo) pair of doubles with MPI_Allreduce(MPI_SUM) and
     *        renormalised, the momenta and sum_wK as doubles, the counters as
     *        64-bit unsigned integers. Without MPI the local ledger is returned
     *        (it is already the sum over the local domains).
     */
    auto reduce_ledger(const bhproto::HostLedger& H) const -> bhproto::GlobalLedger {
      bhproto::GlobalLedger G;
      G.C              = H.cumulative;
      G.debt_initial   = H.debt_initial;
      G.exported_host  = H.exported_host;
      G.discarded_host = H.discarded_host;
      G.n_dead_host    = H.n_dead_host;
#if defined(MPI_ENABLED)
      if (mpi_size > 1) {
        using kernel::hadronic::dd;
        dd* dds[] = { &G.C.E_drag,         &G.C.E_pairs,          &G.C.E_absorbed,
                      &G.C.E_exported,     &G.C.E_round,          &G.C.E_stochastic,
                      &G.C.E_lost_nopairs, &G.C.pending_discarded, &G.C.debt_outstanding,
                      &G.C.pending_outstanding, &G.C.B_budget,    &G.debt_initial,
                      &G.exported_host,    &G.discarded_host };
        constexpr int NDD = sizeof(dds) / sizeof(dds[0]);
        std::vector<double> buf(2 * NDD + 10, 0.0);
        for (int k = 0; k < NDD; ++k) {
          buf[2 * k]     = dds[k]->hi;
          buf[2 * k + 1] = dds[k]->lo;
        }
        for (int d = 0; d < 3; ++d) {
          buf[2 * NDD + d]     = G.C.P_drag[d];
          buf[2 * NDD + 3 + d] = G.C.P_pairs[d];
          buf[2 * NDD + 6 + d] = G.C.P_absorbed[d];
        }
        buf[2 * NDD + 9] = G.C.sum_wK;
        MPI_Allreduce(MPI_IN_PLACE, buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
        for (int k = 0; k < NDD; ++k) {
          dd r;
          r.add(buf[2 * k]);
          r.add(buf[2 * k + 1]);
          *dds[k] = r;
        }
        for (int d = 0; d < 3; ++d) {
          G.C.P_drag[d]     = buf[2 * NDD + d];
          G.C.P_pairs[d]    = buf[2 * NDD + 3 + d];
          G.C.P_absorbed[d] = buf[2 * NDD + 6 + d];
        }
        G.C.sum_wK = buf[2 * NDD + 9];
        unsigned long long* cnt[] = { &G.C.n_events,        &G.C.n_spawned,       &G.C.n_dropped,
                                      &G.C.n_substeps,      &G.C.n_bounded,       &G.C.n_capped_loss,
                                      &G.C.n_closing,       &G.C.n_closing_at_rest, &G.C.n_capped,
                                      &G.C.n_capped_events, &G.C.n_above_table,   &G.C.n_latency,
                                      &G.C.n_resolution,    &G.C.n_processed,     &G.C.n_alive,
                                      &G.C.n_applied,       &G.C.n_rng,           &G.C.n_k0_trigger,
                                      &G.C.n_sampler_attempts, &G.C.n_sampler_overflow, // round 2, finding 5
                                      &G.n_dead_host };
        constexpr int NC = sizeof(cnt) / sizeof(cnt[0]);
        unsigned long long cbuf[NC];
        for (int k = 0; k < NC; ++k) {
          cbuf[k] = *cnt[k];
        }
        MPI_Allreduce(MPI_IN_PLACE, cbuf, NC, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        for (int k = 0; k < NC; ++k) {
          *cnt[k] = cbuf[k];
        }
      }
#endif
      return G;
    }

    /**
     * @brief Order-independent 64-bit checksum of one species (T8, 3.1): the
     *        wraparound sum over the first npart slots of splitmix64 of a
     *        per-particle hash of (cell indices, displacements, momenta, weight,
     *        tag, payload row). It does not depend on the particle order, so the
     *        per-rank values can be summed over the ranks; it does depend on the
     *        local cell indices, so it is a per-decomposition reference.
     */
    auto checksum_species(const Particles<D, M::CoordType>& sp) const -> std::uint64_t {
      const auto n = static_cast<std::size_t>(sp.npart());
      auto mirror = [](const auto& v) {
        auto h = Kokkos::create_mirror_view(v);
        Kokkos::deep_copy(h, v);
        return h;
      };
      auto i1 = mirror(sp.i1);
      auto dx1 = mirror(sp.dx1);
      auto i2 = mirror(sp.i2);
      auto dx2 = mirror(sp.dx2);
      auto i3 = mirror(sp.i3);
      auto dx3 = mirror(sp.dx3);
      auto ux1 = mirror(sp.ux1);
      auto ux2 = mirror(sp.ux2);
      auto ux3 = mirror(sp.ux3);
      auto w  = mirror(sp.weight);
      auto tg = mirror(sp.tag);
      auto pl = mirror(sp.pld_r);
      const std::size_t npld = pl.extent(1);
      std::uint64_t total = 0;
      for (std::size_t p = 0; p < n; ++p) {
        std::uint64_t h = 0xCBF29CE484222325ULL;
        bhproto::mix_bits(h, bhproto::bits_of(i1(p)));
        bhproto::mix_bits(h, bhproto::bits_of(dx1(p)));
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          bhproto::mix_bits(h, bhproto::bits_of(i2(p)));
          bhproto::mix_bits(h, bhproto::bits_of(dx2(p)));
        }
        if constexpr (D == Dim::_3D) {
          bhproto::mix_bits(h, bhproto::bits_of(i3(p)));
          bhproto::mix_bits(h, bhproto::bits_of(dx3(p)));
        }
        bhproto::mix_bits(h, bhproto::bits_of(ux1(p)));
        bhproto::mix_bits(h, bhproto::bits_of(ux2(p)));
        bhproto::mix_bits(h, bhproto::bits_of(ux3(p)));
        bhproto::mix_bits(h, bhproto::bits_of(w(p)));
        bhproto::mix_bits(h, bhproto::bits_of(tg(p)));
        for (std::size_t c = 0; c < npld; ++c) {
          bhproto::mix_bits(h, bhproto::bits_of(pl(p, c)));
        }
        total += splitmix64(h);
      }
      return total;
    }

    void write_ledger(const Particles<D, M::CoordType>& sp, timestep_t step, double t) {
      auto&             H    = ledgers[sp.index()];
      // the array snapshot is recomputed from the stored arrays (independent of the kernel),
      // summed over ranks; the ledger is reduced over ranks (3.1) and written by rank 0
      double swk = sum_wK_from_arrays(sp, ion_mass.at(sp.index()));
#if defined(MPI_ENABLED)
      MPI_Allreduce(MPI_IN_PLACE, &swk, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
      const auto G   = reduce_ledger(H);
      const auto npm = mpi_sum_npart(pair_npart_minus), npp = mpi_sum_npart(pair_npart_plus);
      if (mpi_rank != 0) {
        return;
      }
      const std::string name = "hadronic_ledger_" + std::to_string((int)sp.index()) + ".txt";
      std::ofstream out(name, H.file_started ? std::ios::app : std::ios::trunc);
      if (!H.file_started) {
        out << "# bh_proto ledger, species " << (int)sp.index() << " (" << sp.label()
            << "), mu_d = " << std::setprecision(17) << ion_mass.at(sp.index())
            << ", mu_r = " << setup.mu_r << ", tau0 = " << setup.tau0
            << ", skindepth0 = " << skindepth0 << ", dt = " << dt << ", mode = " << setup.mode
            << ", source = " << (setup.event_mode() ? std::string("none (event mode)") : setup.source)
            << ", spectrum = " << setup.spectrum << ", on_bound = " << setup.on_bound
            << ", r_th_factor = " << setup.r_th_factor
            << ", f_r = " << setup.f_r << ", t_latency = " << t_latency
            << ", pair_weight = " << setup.pair_weight
            << ", max_events_per_step = " << setup.max_events_per_step << "\n";
        out << "# cumulative entries are summed over launches since the start of the segment; "
               "snapshots (debt_outstanding, pending_outstanding, sum_wK, npart) refer to the state "
               "after the step; energies in w m_e c^2, momenta in w m_e c; identity residual = "
               "E_drag + debt_initial - [(E_pairs - E_absorbed) + E_round + E_stochastic + "
               "debt_outstanding + E_exported + E_lost_nopairs] (double-double on the host)\n";
        out << "# time is the time after the step, (step + 1) dt (code units l/c)\n";
        out << "# columns: step [1] | time [l/c] | E_drag [cumulative] | E_pairs [cumulative: sum "
               "w_pair,stored (E- + E+) from the stored momenta] | E_absorbed [cumulative: sum "
               "w_pair,stored eps_stored] | E_exported [cumulative] | E_round [cumulative] | "
               "E_stochastic [cumulative: poisson, E_drag - sum W D, signed] | "
               "E_lost_nopairs [cumulative: none] | pending_discarded [cumulative] | debt_initial "
               "[segment start] | P_drag_x | P_drag_y | P_drag_z | P_pairs_x | P_pairs_y | "
               "P_pairs_z | P_absorbed_x | P_absorbed_y | P_absorbed_z [cumulative] | "
               "debt_outstanding [snapshot] | pending_outstanding [snapshot] | residual [identity] "
               "| n_events | n_spawned | n_dropped | n_substeps | n_bounded | n_capped_loss | "
               "n_closing | n_capped | n_capped_events | n_above_table | n_latency | n_processed | "
               "n_applied | n_rng | n_k0_trigger | n_closing_at_rest [1, cumulative] | sum_wK [snapshot: sum over alive "
               "ions of w K in double from the stored u] | B_budget [cumulative single-precision "
               "rounding budget of the transfers] | npart_minus | npart_plus [snapshots] | "
               "n_sampler_attempts | n_sampler_overflow [1, cumulative: restframe trials of the chains and "
               "packets whose rejection loop hit the bound; reduced over ranks; round 2]\n";
        H.file_started = true;
      }
      const auto& C = G.C;
      out << std::setprecision(17) << step << " " << t << " " << C.E_drag.value() << " "
          << C.E_pairs.value() << " " << C.E_absorbed.value() << " " << C.E_exported.value() << " "
          << C.E_round.value() << " " << C.E_stochastic.value() << " " << C.E_lost_nopairs.value() << " "
          << C.pending_discarded.value() << " " << G.debt_initial.value() << " " << C.P_drag[0]
          << " " << C.P_drag[1] << " " << C.P_drag[2] << " " << C.P_pairs[0] << " " << C.P_pairs[1]
          << " " << C.P_pairs[2] << " " << C.P_absorbed[0] << " " << C.P_absorbed[1] << " "
          << C.P_absorbed[2] << " " << C.debt_outstanding.value() << " "
          << C.pending_outstanding.value() << " " << G.identity_residual() << " " << C.n_events
          << " " << C.n_spawned << " " << C.n_dropped << " " << C.n_substeps << " " << C.n_bounded
          << " " << C.n_capped_loss << " " << C.n_closing << " " << C.n_capped << " "
          << C.n_capped_events << " " << C.n_above_table << " " << C.n_latency << " "
          << C.n_processed << " " << C.n_applied << " " << C.n_rng << " " << C.n_k0_trigger << " "
          << C.n_closing_at_rest << " " << swk << " " << C.B_budget.value() << " " << npm << " " << npp
          << " " << C.n_sampler_attempts << " " << C.n_sampler_overflow << "\n";
    }

    npart_t pair_npart_minus { 0 }, pair_npart_plus { 0 };

    void write_ions(const Particles<D, M::CoordType>& sp, timestep_t step, double t) {
      auto&             H    = ledgers[sp.index()];
      const std::string name = per_rank_name("hadronic_ions_" + std::to_string((int)sp.index()));
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
      std::ofstream out(per_rank_name("hadronic_pairs"), std::ios::trunc);
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
     * @brief T13: one sample of the discrete Gauss residual (1D): R_i = (E_{i+1/2} -
     *        E_{i-1/2}) - (q0 / (B0 V0)) rho_i with rho_i the first-order (CIC) node
     *        charge of every depositing charged species (the shape of the zigzag
     *        deposit), computed in double from the arrays; E in real_t; R0 is the
     *        residual of the first sample (a randomly loaded background does not
     *        satisfy Gauss's law; the deposit conserves R - R0)
     */
    GaussSample gauss_sample(Domain<S, M>& domain) {
      GaussSample g;
      if constexpr (D == Dim::_1D) {
        const int n1 = static_cast<int>(domain.mesh.n_active(in::x1));
        std::vector<double> rho(n1, 0.0);
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
            g.max_abs_q              = std::max(g.max_abs_q, std::abs(qw));
            g.n_depositing          += 1;
          }
        }
        auto em = Kokkos::create_mirror_view(domain.fields.em);
        Kokkos::deep_copy(em, domain.fields.em);
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
            gauss_R0[i] = R;
          }
          if (std::abs(R) > g.max_R) {
            g.max_R = std::abs(R);
            g.arg_R = i;
          }
          g.max_dR   = std::max(g.max_dR, std::abs(R - gauss_R0[i]));
          g.max_rho  = std::max(g.max_rho, std::abs(gauss_coeff * rho[i]));
          g.max_E    = std::max(g.max_E, std::abs(Ep));
          g.sum_R   += R;
          g.sum_rho += rho[i];
        }
        g.ratio = g.max_rho > 0.0 ? g.max_dR / g.max_rho : 0.0;
        if (pair_minus != 0) {
          g.nm = domain.species[pair_minus - 1].npart();
          g.np = domain.species[pair_plus - 1].npart();
        }
      } else {
        (void)domain;
      }
      return g;
    }

    void write_gauss_sample(const GaussSample& g, timestep_t step, double t, const char* stage) {
      std::ofstream out(per_rank_name("hadronic_gauss"), gauss_started ? std::ios::app : std::ios::trunc);
      if (!gauss_started) {
        out << "# bh_proto Gauss residual (1D): R_i = (E_{i+1/2} - E_{i-1/2}) - (q0/(B0 V0)) rho_i, "
               "rho_i = CIC node charge of the depositing charged species (double from the arrays), "
               "E from the real_t field array; gauss_coeff = q0/(B0 V0) = "
            << std::setprecision(17) << gauss_coeff << "\n";
        out << "# columns: step | time | stage | max|R| | max|c rho| | max|E| | max|R - R0|/max|c rho| (R0 = the "
               "residual at the first sample) | argmax i of |R| | sum_i R | sum_i rho | max|q w| (per particle) | "
               "npart e- | npart e+ | max|R - R0| | depositing particles\n";
        gauss_started = true;
      }
      out << std::setprecision(17) << step << " " << t << " " << stage << " " << g.max_R << " " << g.max_rho
          << " " << g.max_E << " " << g.ratio << " " << g.arg_R << " " << g.sum_R << " " << g.sum_rho << " "
          << g.max_abs_q << " " << g.nm << " " << g.np << " " << g.max_dR << " " << g.n_depositing << "\n";
    }

    void write_gauss(Domain<S, M>& domain, timestep_t step, double t, const char* stage) {
      write_gauss_sample(gauss_sample(domain), step, t, stage);
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
        auto tm = mirror(spm.tag), tp = mirror(spp.tag);
        for (npart_t k = 0; k < n; ++k) {
          if (tm(k) != ParticleTag::alive || tp(k) != ParticleTag::alive) {
            continue; // C-MPI-1: only alive leptons (a sent lepton's remnant is dead with zero payloads)
          }
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
      // the pair-side sums are global (3.1): hi and lo of each dd summed over the ranks
      auto reduce_dd = [this](kernel::hadronic::dd& d) {
#if defined(MPI_ENABLED)
        if (mpi_size > 1) {
          double buf[2] = { d.hi, d.lo };
          MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          kernel::hadronic::dd r;
          r.add(buf[0]);
          r.add(buf[1]);
          d = r;
        }
#else
        (void)d;
        (void)this;
#endif
      };
      reduce_dd(pairs_net);
      reduce_dd(pairs_E);
      reduce_dd(pairs_eps);
      reduce_dd(pairs_w);
      n_pairs_born = mpi_sum_u64(n_pairs_born);
      const auto npm_g = mpi_sum_npart(pair_npart_minus), npp_g = mpi_sum_npart(pair_npart_plus);
      for (auto& sp : domain.species) {
        if (ion_mass.count(sp.index()) == 0) {
          continue;
        }
        // order-independent checksum of every ion array, summed over the ranks (3.1, T8)
        const std::uint64_t h       = mpi_sum_u64(checksum_species(sp));
        const auto          npart_g = mpi_sum_npart(sp.npart());
        double wK_final = sum_wK_from_arrays(sp, ion_mass.at(sp.index()));
        double wK_init  = 0.0;
        for (const double v : initial_wK[sp.index()]) {
          wK_init += v;
        }
        auto debt_final = sum_w_payload_from_arrays(sp, static_cast<unsigned short>(setup.payload_debt));
#if defined(MPI_ENABLED)
        if (mpi_size > 1) {
          double buf[2] = { wK_final, wK_init };
          MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          wK_final = buf[0];
          wK_init  = buf[1];
        }
#endif
        reduce_dd(debt_final);
        const auto& H = ledgers[sp.index()];
        const auto  G = reduce_ledger(H); // the global ledger: identity residual over all ranks
        const auto& C = G.C;
        if (mpi_rank != 0) {
          continue;
        }
        std::printf("bh_proto: checksum species %d npart %llu : %s\n", (int)sp.index(), npart_g,
                    bhproto::hex(h).c_str());
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
          rhs.sub(C.E_stochastic);
          rhs.add(G.debt_initial);
          kernel::hadronic::dd res = rhs;
          res.sub(pairs_net);
          std::printf("bh_proto: array identity species %d : sum_pairs w_s (E- + E+ - eps_s) = %.17g "
                      "(%llu kernel-born pairs, sum w_pair = %.17g, sum w_s E = %.17g, sum w_s eps = %.17g), "
                      "rhs = sum w dK - sum w r - E_exported - E_lost_nopairs - E_stochastic + debt_initial = %.17g, "
                      "residual = %.17g, E_round (ledger) = %.17g, residual - E_round = %.6e, "
                      "B_budget = %.17g, |residual|/B = %.6e\n",
                      (int)sp.index(), pairs_net.value(), n_pairs_born, pairs_w.value(), pairs_E.value(),
                      pairs_eps.value(), rhs.value(), res.value(), C.E_round.value(),
                      res.value() - C.E_round.value(), C.B_budget.value(),
                      C.B_budget.value() > 0.0 ? std::abs(res.value()) / C.B_budget.value() : 0.0);
        }
        std::printf("bh_proto: ledger species %d : E_drag %.17g E_pairs %.17g E_absorbed %.17g "
                    "E_exported %.17g E_round %.17g E_stochastic %.17g E_lost_nopairs %.17g pending_discarded %.17g "
                    "debt_initial %.17g debt_outstanding %.17g pending_outstanding %.17g residual %.6e "
                    "P_drag %.17g %.17g %.17g P_pairs %.17g %.17g %.17g P_absorbed %.17g %.17g %.17g "
                    "n_events %llu n_spawned %llu n_dropped %llu n_substeps %llu n_bounded %llu "
                    "n_capped_loss %llu n_closing %llu n_capped %llu n_capped_events %llu "
                    "n_above_table %llu n_latency %llu n_processed %llu n_applied %llu n_rng %llu "
                    "n_k0_trigger %llu n_closing_at_rest %llu sum_wK %.17g B_budget %.17g npart_minus %u npart_plus %u "
                    "estimate_bare %.10g estimate_sized %.10g\n",
                    (int)sp.index(), C.E_drag.value(), C.E_pairs.value(), C.E_absorbed.value(),
                    C.E_exported.value(), C.E_round.value(), C.E_stochastic.value(), C.E_lost_nopairs.value(),
                    C.pending_discarded.value(), G.debt_initial.value(), C.debt_outstanding.value(),
                    C.pending_outstanding.value(), G.identity_residual(), C.P_drag[0], C.P_drag[1],
                    C.P_drag[2], C.P_pairs[0], C.P_pairs[1], C.P_pairs[2], C.P_absorbed[0],
                    C.P_absorbed[1], C.P_absorbed[2], C.n_events, C.n_spawned, C.n_dropped,
                    C.n_substeps, C.n_bounded, C.n_capped_loss, C.n_closing, C.n_capped,
                    C.n_capped_events, C.n_above_table, C.n_latency, C.n_processed, C.n_applied,
                    C.n_rng, C.n_k0_trigger, C.n_closing_at_rest, C.sum_wK, C.B_budget.value(), (unsigned)npm_g,
                    (unsigned)npp_g, estimate_bare, estimate_sized);
        if (setup.spectrum == "restframe") {
          std::printf("bh_proto: restframe sampler species %d : n_sampler_attempts %llu n_sampler_overflow %llu "
                      "attempts per pair %.3f\n",
                      (int)sp.index(), C.n_sampler_attempts, C.n_sampler_overflow,
                      C.n_spawned > 0 ? static_cast<double>(C.n_sampler_attempts) / C.n_spawned : 0.0);
        }
        if (any_absorbing) {
          std::printf("bh_proto: exported species %d : E_exported_host %.17g E_exported %.17g difference %.6e "
                      "pending_discarded_host %.17g pending_discarded %.17g difference %.6e n_dead_with_payload %llu "
                      "(host sums over dead-tagged ions before each launch, i.e. before the sort)\n",
                      (int)sp.index(), G.exported_host.value(), C.E_exported.value(),
                      G.exported_host.value() - C.E_exported.value(), G.discarded_host.value(),
                      C.pending_discarded.value(), G.discarded_host.value() - C.pending_discarded.value(),
                      G.n_dead_host);
        }
      }
      if (pair_minus != 0) {
        for (const spidx_t sidx : { pair_minus, pair_plus }) {
          auto&               sp = domain.species[sidx - 1];
          const std::uint64_t h  = mpi_sum_u64(checksum_species(sp));
          const auto          n  = mpi_sum_npart(sp.npart());
          if (mpi_rank == 0) {
            std::printf("bh_proto: checksum species %d npart %llu : %s\n", (int)sidx, n,
                        bhproto::hex(h).c_str());
          }
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
        nan_live = mpi_sum_u64(nan_live);
        nan_tail = mpi_sum_u64(nan_tail);
        if (mpi_rank == 0) {
          std::printf("bh_proto: payload scan : %llu NaN payload entries on live particles, %llu in the "
                      "unused tails (tails are allowed to hold anything)\n", nan_live, nan_tail);
        }
      }
      {
        // order-independent sum of per-cell hashes, summed over the ranks
        const std::uint64_t hf = mpi_sum_u64(bhproto::checksum_fields<D>(domain.fields.em));
        if (mpi_rank == 0) {
          std::printf("bh_proto: checksum fields : %s\n", bhproto::hex(hf).c_str());
        }
      }
      std::fflush(stdout);
    }
  };

} // namespace user

#endif // PROBLEM_GENERATOR_H
