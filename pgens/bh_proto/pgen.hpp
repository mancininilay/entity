/**
 * @file pgens/bh_proto/pgen.hpp
 * @brief Bethe-Heitler loss-only prototype (P2 step 2): a test problem
 *        generator that injects ions with prescribed kinetic energies in a
 *        periodic Cartesian box, optionally with uniform E and B fields, and
 *        launches the loss-only `BetheHeitler_kernel` on every ion species
 *        flagged hadronic at the end of each step (prototype placement of spec
 *        II.1: `CustomPostStep`, after `CommunicateParticles`).
 *
 * Outputs (in the working directory of the run):
 *   hadronic_ledger_<species>.txt  every `ledger.every` steps and at the last
 *                                  step: cumulative double ledger (II.2 step 6)
 *                                  with a header naming every column and unit
 *   hadronic_ions_<species>.txt    per-ion K, Gamma_p, |u|, pitch cosine, delta
 *   hadronic_table.txt             the startup table in the exchange format of
 *                                  step 1 (provenance; same code path as
 *                                  bh_tables_dump)
 *   stdout                         grid summary, per-K expectations, and at the
 *                                  last step the deterministic checksums of
 *                                  every ion array and of the six field arrays
 *
 * Payload slots: this tree declares real payloads per species with the input
 * key `n_payloads_real`; the generator requires every hadronic species to
 * declare at least max(payload_pending, payload_debt, payload_debt_k0) + 1
 * slots (3 with the defaults 0, 1, 2) and refuses otherwise. The slots of
 * newly injected ions are zeroed over [npart_before, npart_after) after each
 * injection (constraint P9, prototype recipe).
 *
 * Time-level convention: the loss acts on u^{n+1/2} after the push of step n;
 * ledger rows are stamped with the time after the step, (step + 1) dt.
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

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_table_io.h"
#include "archetypes/hadronic/bh_tables_device.h"
#include "framework/containers/particles.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "kernels/hadronic/bethe_heitler.hpp"
#include "kernels/injectors.hpp"

#include "bh_proto_params.h"

#include <Kokkos_Core.hpp>

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
      kernel::InjectParticle<D, Coord::Cartesian, T>(p,
                                                      i1, i2, i3,
                                                      dx1, dx2, dx3,
                                                      ux1, ux2, ux3,
                                                      phi, weight, tag, pld_i,
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
      bool                         file_started { false };
      bool                         ions_started { false };

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
    };

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

    const SimulationParams& params;
    bhproto::Setup          setup;
    UniformEMFields<D>      init_flds;

    // run constants
    double     dt { 0.0 };
    double     skindepth0 { 1.0 };
    double     omegaB0 { 1.0 };
    timestep_t max_steps { 0 };
    bool       stage_active { false }; // enabled and tau0 > 0
    bool       use_weights { false };

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
    std::uint64_t                              inject_seed { 0x5BD1E995ULL };

    PGen(const SimulationParams& p, const Metadomain<S, M>& metadomain)
      : params { p } {
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
        const auto runtime = params.template get<simtime_t>("simulation.runtime");
        max_steps = static_cast<timestep_t>(runtime / dt); // as engine.hpp
      }
      use_weights = params.template get<bool>("particles.use_weights");
      raise::ErrorIf(not use_weights,
                     "bh_proto: particles.use_weights must be true (the ions "
                     "are injected with explicit weights)",
                     HERE);
      raise::ErrorIf(M::CoordType != Coord::Cartesian,
                     "bh_proto: Cartesian metric required",
                     HERE);

      init_flds.e1 = static_cast<real_t>(setup.E[0]);
      init_flds.e2 = static_cast<real_t>(setup.E[1]);
      init_flds.e3 = static_cast<real_t>(setup.E[2]);
      init_flds.b1 = static_cast<real_t>(setup.B[0]);
      init_flds.b2 = static_cast<real_t>(setup.B[1]);
      init_flds.b3 = static_cast<real_t>(setup.B[2]);

      // ---- species checks --------------------------------------------------
      const auto& species = metadomain.species_params();
      const int   need_slots = std::max(setup.payload_pending,
                                        std::max(setup.payload_debt,
                                                 setup.payload_debt_k0)) + 1;
      for (const int idx : setup.species) {
        const ParticleSpecies* sp = nullptr;
        for (const auto& s : species) {
          if (static_cast<int>(s.index()) == idx) {
            sp = &s;
          }
        }
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
        if (sp->deposit()) {
          std::printf("bh_proto: warning: species %d deposits current; the uniform "
                      "fields of the validation boxes are exact only with deposit = "
                      "false\n", idx);
        }
        ion_species.push_back(static_cast<spidx_t>(idx));
        ion_mass[static_cast<spidx_t>(idx)] = static_cast<double>(sp->mass());
        ledgers[static_cast<spidx_t>(idx)]  = bhproto::HostLedger {};
      }

      // ---- tables ------------------------------------------------------------
      stage_active = setup.enabled && setup.tau0 > 0.0;
      if (setup.enabled) {
        try {
          bath = arch::bh::make_bath(setup.bath);
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
      } else {
        std::printf("bh_proto: hadronic.enabled = false: no tables, no stage\n");
      }
      std::printf("bh_proto: dt = %.17g, skindepth0 = %.17g, omegaB0 = %.17g, "
                  "max_steps = %u, stage %s\n",
                  dt, skindepth0, omegaB0, (unsigned)max_steps,
                  stage_active ? "active" : (setup.enabled ? "not launched (tau0 = 0)"
                                                           : "disabled"));
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
          std::printf("bh_proto: species %d K = %.10g (Gamma_p = %.10g, u = %.10g, mu_d = %.6g) %s: "
                      "I_exact = %.10e, I_table = %.10e, dK/step = %.10e, dK/K = %.4e, "
                      "ulp(u) = %.4e, dK_ulp = ulp mu_d u/gamma = %.4e, dK/dK_ulp = %.4e, "
                      "delta_th = %.4e (floor %.4e, eta_u K %.4e), steps per application ~ %.3g, "
                      "sub-cycling %s\n",
                      (int)sp, K, Gp, u, mu_d, where.c_str(), I_exact, I_table, dK, dK / K,
                      ulp, dK_ulp, dK_ulp > 0.0 ? dK / dK_ulp : 0.0, dth, floor,
                      setup.eta_u * K, dK > 0.0 ? std::ceil(dth / dK) : 0.0,
                      dK > setup.f_max * K ? "yes" : "no");
        }
      }
      std::fflush(stdout);
    }

    /* -------------------------------------------------------------------- */
    /* initial conditions                                                    */
    /* -------------------------------------------------------------------- */
    inline void InitPrtls(Domain<S, M>& domain) {
      raise::ErrorIf(domain.index() != 0u,
                     "bh_proto: multi-domain runs are out of scope in step 2",
                     HERE);
      const int n1 = static_cast<int>(domain.mesh.n_active(in::x1));
      const int n2 = (D == Dim::_2D || D == Dim::_3D)
                       ? static_cast<int>(domain.mesh.n_active(in::x2)) : 1;
      const int n3 = (D == Dim::_3D) ? static_cast<int>(domain.mesh.n_active(in::x3)) : 1;

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
          kern.pld_i = sp.pld_i; kern.Ks = Ks;
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
        // P9: every real payload slot of a newly injected ion is zero
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
     * @brief sum over alive ions of w K computed in double from the stored u
     */
    static auto sum_wK_from_arrays(const Particles<D, M::CoordType>& sp, double mu_d)
      -> double {
      auto ux1 = sp.ux1, ux2 = sp.ux2, ux3 = sp.ux3, wgt = sp.weight;
      auto tag = sp.tag;
      double total = 0.0;
      Kokkos::parallel_reduce("bh_proto::SumWK",
                              CreateParticleRangePolicy<Dim::_1D>({ 0u }, { sp.npart() }),
                              Lambda(prtlidx_t p, double& acc) {
                                if (tag(p) == ParticleTag::dead) {
                                  return;
                                }
                                acc += static_cast<double>(wgt(p)) *
                                       kernel::hadronic::kinetic_energy(
                                         mu_d, static_cast<double>(ux1(p)),
                                         static_cast<double>(ux2(p)),
                                         static_cast<double>(ux3(p)));
                              },
                              Kokkos::Sum<double>(total));
      return total;
    }

    /* -------------------------------------------------------------------- */
    /* the stage                                                             */
    /* -------------------------------------------------------------------- */
    inline void CustomPostStep(timestep_t step, simtime_t /*time*/, Domain<S, M>& domain) {
      const double t_after = static_cast<double>(step + 1) * dt; // t^{n+1}
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
        for (auto& sp : domain.species) {
          if (ion_mass.count(sp.index()) == 0 || sp.npart() == 0) {
            continue;
          }
          kernel::hadronic::LossLedger L;
          Kokkos::parallel_reduce(
            "bh_proto::BetheHeitler",
            CreateParticleRangePolicy<Dim::_1D>({ 0u }, { sp.npart() }),
            kernel::hadronic::BetheHeitler_kernel<M> {
              static_cast<const ParticleArrays&>(sp), ion_mass.at(sp.index()), tables, prm },
            Kokkos::Sum<kernel::hadronic::LossLedger>(L));
          ledgers[sp.index()].accumulate(L);
        }
      }
      const bool last = (step + 1 >= max_steps);
      if (((step + 1) % static_cast<timestep_t>(setup.ledger_every) == 0u) || last) {
        for (auto& sp : domain.species) {
          if (ion_mass.count(sp.index()) == 0) {
            continue;
          }
          write_ledger(sp, step, t_after);
          write_ions(sp, step, t_after);
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
            << ", skindepth0 = " << skindepth0 << ", dt = " << dt << "\n";
        out << "# cumulative entries are summed over launches since the start of the run; "
               "snapshots (pending_outstanding, sum_wK) refer to the state after the step\n";
        out << "# time is the time after the step, (step + 1) dt (code units l/c)\n";
        out << "# columns: step [1] | time [l/c] | E_drag [w m_e c^2, cumulative] | "
               "E_lost_nopairs [w m_e c^2, cumulative] | pending_discarded [w m_e c^2, "
               "cumulative] | pending_outstanding [w m_e c^2, snapshot] | P_drag_x [w m_e c, "
               "cumulative] | P_drag_y [w m_e c, cumulative] | P_drag_z [w m_e c, cumulative] "
               "| n_substeps [1, cumulative] | n_bounded [1, cumulative] | n_above_table [1, "
               "cumulative] | n_processed [1, cumulative] | sum_wK [w m_e c^2, snapshot: sum "
               "over alive ions of w K in double from the stored u]\n";
        H.file_started = true;
      }
      // the array snapshot is recomputed from the stored arrays (independent of the kernel)
      const double swk = sum_wK_from_arrays(sp, ion_mass.at(sp.index()));
      const auto&  C   = H.cumulative;
      out << std::setprecision(17) << step << " " << t << " " << C.E_drag << " "
          << C.E_lost_nopairs << " " << C.pending_discarded << " " << C.pending_outstanding
          << " " << C.P_drag[0] << " " << C.P_drag[1] << " " << C.P_drag[2] << " "
          << C.n_substeps << " " << C.n_bounded << " " << C.n_above_table << " "
          << C.n_processed << " " << swk << "\n";
    }

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
               "pending payload] | tag [1]\n";
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
            << static_cast<double>(pld(p, setup.payload_pending)) << " " << tag(p) << "\n";
      }
    }

    void finish(Domain<S, M>& domain) {
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
        const auto& C = ledgers[sp.index()].cumulative;
        std::printf("bh_proto: checksum species %d npart %u : %s\n", (int)sp.index(),
                    (unsigned)sp.npart(), bhproto::hex(h).c_str());
        std::printf("bh_proto: closure species %d : sum w (K_init - K_final) = %.17g, "
                    "E_drag = %.17g, difference = %.6e, relative = %.6e\n",
                    (int)sp.index(), wK_init - wK_final, C.E_drag,
                    (wK_init - wK_final) - C.E_drag,
                    C.E_drag != 0.0 ? ((wK_init - wK_final) - C.E_drag) / C.E_drag : 0.0);
        std::printf("bh_proto: ledger species %d : E_drag %.17g E_lost_nopairs %.17g "
                    "pending_discarded %.17g pending_outstanding %.17g P_drag %.17g %.17g %.17g "
                    "n_substeps %llu n_bounded %llu n_above_table %llu n_processed %llu "
                    "n_applied %llu\n",
                    (int)sp.index(), C.E_drag, C.E_lost_nopairs, C.pending_discarded,
                    C.pending_outstanding, C.P_drag[0], C.P_drag[1], C.P_drag[2],
                    C.n_substeps, C.n_bounded, C.n_above_table, C.n_processed, C.n_applied);
      }
      std::printf("bh_proto: checksum fields : %s\n",
                  bhproto::hex(bhproto::checksum_fields<D>(domain.fields.em)).c_str());
      std::fflush(stdout);
    }
  };

} // namespace user

#endif // PROBLEM_GENERATOR_H
