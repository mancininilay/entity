/**
 * Validation problem generator for the Bethe-Heitler emission policy.
 *
 * Periodic 2D box. `setup.n_ions` emitters of the species `setup.ion_species`
 * are placed at random positions (the same global set for any domain
 * decomposition) with momenta of one magnitude, set by the proton Lorentz
 * factor `setup.gamma_p`, isotropic or along +x1. If
 * `setup.neutralizer_species` is given, one particle of that species is
 * placed on top of every emitter, so that the initial charge density is zero
 * node by node.
 *
 * Every `setup.diag_interval` steps (after steps n, 2 n, ...) one line starting with "BHV" is printed
 * by the root rank with the sums over the emitters, the leptons, the policy
 * counters and, if `setup.gauss` is set, the residual of Gauss's law on the
 * nodes (the stencil of the divergence and the first-order node shape of the
 * zig-zag deposit). At the step `setup.dump_step` the lepton momenta are
 * written to binary files and a hash of the fields and particles is printed.
 *
 * `setup.negative_control = true` runs the same policy with the write of the
 * previous position of the new leptons left out (emitters must then use
 * `emission = "custom"`). `setup.stale_prev = true` fills the previous
 * positions of the unused lepton slots with random points of the domain, the
 * state that dead particles leave behind. `setup.recycle = true` counts and
 * removes the leptons after every step (long cooling runs).
 *
 * The file compiles against a tree without the policy (the hash and the sums
 * are then the only diagnostics).
 */

#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "kernels/injectors.hpp"

#if __has_include("archetypes/emission/bethe_heitler.h")
  #include "archetypes/emission/bethe_heitler.h"
  #define BHV_HAS_POLICY 1
#endif

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace user {
  using namespace ntt;

  namespace bhv {
    // counter-based generator: uniform in [0, 1) from (seed, particle, draw)
    inline auto uniform(std::uint64_t seed, std::uint64_t particle, std::uint64_t draw)
      -> double {
      std::uint64_t z = seed * 0x9E3779B97F4A7C15ULL +
                        particle * 0xBF58476D1CE4E5B9ULL +
                        draw * 0x94D049BB133111EBULL + 0x2545F4914F6CDD1DULL;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z = z ^ (z >> 31);
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      z = z ^ (z >> 31);
      return static_cast<double>(z >> 11) * 0x1.0p-53;
    }

    inline auto fnv(const void* data, std::size_t n, std::uint64_t h) -> std::uint64_t {
      const auto* p = static_cast<const unsigned char*>(data);
      for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
      }
      return h;
    }

    inline void reduce_sum(std::vector<double>& v) {
#if defined(MPI_ENABLED)
      std::vector<double> out(v.size(), 0.0);
      MPI_Allreduce(v.data(),
                    out.data(),
                    static_cast<int>(v.size()),
                    MPI_DOUBLE,
                    MPI_SUM,
                    MPI_COMM_WORLD);
      v = out;
#else
      (void)v;
#endif
    }

    inline auto is_root() -> bool {
#if defined(MPI_ENABLED)
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      return rank == 0;
#else
      return true;
#endif
    }

    inline auto rank() -> int {
#if defined(MPI_ENABLED)
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      return rank;
#else
      return 0;
#endif
    }

#if defined(BHV_HAS_POLICY)
    /**
     * The policy with the write of the previous position left out (negative
     * control of the charge test): the insertion of the built-in policies.
     */
    template <MetricClass M>
    struct NoPrevWrite {
      using base_t  = ::arch::emission::BetheHeitler<M>;
      using Payload = typename base_t::Payload;
      base_t base;

      NoPrevWrite(const base_t& base) : base { base } {}

      auto emitted_species_indices() const -> std::vector<spidx_t> {
        return base.emitted_species_indices();
      }

      auto numbers_injected() const -> std::vector<npart_t> {
        return base.numbers_injected();
      }

      Inline auto shouldEmit(const coord_t<M::PrtlDim>& x_Cd,
                             const coord_t<M::PrtlDim>& x_Ph,
                             const vec_t<Dim::_3D>&     u_Ph,
                             const vec_t<Dim::_3D>&     ep_Ph,
                             const vec_t<Dim::_3D>&     bp_Ph,
                             vec_t<Dim::_3D>&           delta_u_Ph,
                             Payload& payload) const -> Kokkos::pair<bool, bool> {
        return base.shouldEmit(x_Cd, x_Ph, u_Ph, ep_Ph, bp_Ph, delta_u_Ph, payload);
      }

      Inline void emit(const tuple_t<int, M::Dim>&      xi_Cd,
                       const tuple_t<prtldx_t, M::Dim>& dxi_Cd,
                       const vec_t<Dim::_3D>&,
                       real_t         weight,
                       real_t         phi,
                       const Payload& payload) const {
        const auto w = static_cast<real_t>(base.pair_weight) * weight;
        kernel::InjectParticle<M::Dim, M::CoordType, false>(
          base.electron_offset + payload.ticket,
          base.electron_i1,
          base.electron_i2,
          base.electron_i3,
          base.electron_dx1,
          base.electron_dx2,
          base.electron_dx3,
          base.electron_ux1,
          base.electron_ux2,
          base.electron_ux3,
          base.electron_phi,
          base.electron_weight,
          base.electron_tag,
          base.electron_pld_i,
          xi_Cd,
          dxi_Cd,
          { payload.u_electron[0], payload.u_electron[1], payload.u_electron[2] },
          w,
          phi);
        kernel::InjectParticle<M::Dim, M::CoordType, false>(
          base.positron_offset + payload.ticket,
          base.positron_i1,
          base.positron_i2,
          base.positron_i3,
          base.positron_dx1,
          base.positron_dx2,
          base.positron_dx3,
          base.positron_ux1,
          base.positron_ux2,
          base.positron_ux3,
          base.positron_phi,
          base.positron_weight,
          base.positron_tag,
          base.positron_pld_i,
          xi_Cd,
          dxi_Cd,
          { payload.u_positron[0], payload.u_positron[1], payload.u_positron[2] },
          w,
          phi);
      }
    };
#endif // BHV_HAS_POLICY
  } // namespace bhv

  // uniform magnetic field (units of B0)
  template <Dimension D>
  struct InitFields {
    InitFields(real_t bx, real_t by, real_t bz) : bx { bx }, by { by }, bz { bz } {}

    Inline auto bx1(const coord_t<D>&) const -> real_t {
      return bx;
    }

    Inline auto bx2(const coord_t<D>&) const -> real_t {
      return by;
    }

    Inline auto bx3(const coord_t<D>&) const -> real_t {
      return bz;
    }

  private:
    const real_t bx, by, bz;
  };

  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    static constexpr auto engines = ::traits::pgen::compatible_with<SimEngine::SRPIC> {};
    static constexpr auto metrics =
      ::traits::pgen::compatible_with<Metric::Minkowski> {};
    static constexpr auto dimensions = ::traits::pgen::compatible_with<Dim::_2D> {};

    const SimulationParams& params;

    const int    ion_species, neutralizer_species;
    const int    electron_species, positron_species;
    const int    n_ions, seed, diag_interval, dump_step;
    const bool   isotropic, gauss, negative_control, stale_prev, recycle;
    // leptons removed so far by `setup.recycle`: n, sum w, sum w gamma of
    // the electrons and of the positrons (this rank)
    double       recycled[6] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const double gamma_p, mass_ratio, ion_weight;
    const std::vector<ncells_t> resolution;
    double       K0 { 0.0 };
    InitFields<D> init_flds;

    PGen(const SimulationParams& p, const Metadomain<S, M>&)
      : params { p }
      , ion_species { p.template get<int>("setup.ion_species") }
      , neutralizer_species { p.template get<int>("setup.neutralizer_species", 0) }
      , electron_species { p.template get<int>("setup.electron_species", 0) }
      , positron_species { p.template get<int>("setup.positron_species", 0) }
      , n_ions { p.template get<int>("setup.n_ions") }
      , seed { p.template get<int>("setup.seed", 1) }
      , diag_interval { p.template get<int>("setup.diag_interval", 1) }
      , dump_step { p.template get<int>("setup.dump_step", -1) }
      , isotropic { p.template get<bool>("setup.isotropic", true) }
      , gauss { p.template get<bool>("setup.gauss", false) }
      , negative_control { p.template get<bool>("setup.negative_control", false) }
      , stale_prev { p.template get<bool>("setup.stale_prev", false) }
      , recycle { p.template get<bool>("setup.recycle", false) }
      , gamma_p { static_cast<double>(p.template get<real_t>("setup.gamma_p")) }
      , mass_ratio { static_cast<double>(
          p.template get<real_t>("setup.mass_ratio",
                                 static_cast<real_t>(1836.15267343))) }
      , ion_weight { static_cast<double>(
          p.template get<real_t>("setup.ion_weight", ONE)) }
      , resolution { p.template get<std::vector<ncells_t>>("grid.resolution") }
      , init_flds { p.template get<real_t>("setup.bx", ZERO),
                    p.template get<real_t>("setup.by", ZERO),
                    p.template get<real_t>("setup.bz", ZERO) } {}

    void InitPrtls(Domain<S, M>& domain) {
      auto& ions = domain.species[ion_species - 1];
      const auto mu_d = static_cast<double>(ions.mass());
      // kinetic energy [m_e c^2] of a proton of Lorentz factor gamma_p
      K0                 = mass_ratio * (gamma_p - 1.0);
      const double gamma = 1.0 + K0 / mu_d;
      const double u     = std::sqrt((gamma - 1.0) * (gamma + 1.0));

      const auto off  = domain.offset_ncells();
      const int  off1 = static_cast<int>(off[0]), off2 = static_cast<int>(off[1]);
      const int  n1   = static_cast<int>(domain.mesh.n_active(in::x1));
      const int  n2   = static_cast<int>(domain.mesh.n_active(in::x2));
      const double NX = static_cast<double>(resolution[0]);
      const double NY = static_cast<double>(resolution[1]);

      auto i1_h  = Kokkos::create_mirror_view(ions.i1);
      auto i2_h  = Kokkos::create_mirror_view(ions.i2);
      auto dx1_h = Kokkos::create_mirror_view(ions.dx1);
      auto dx2_h = Kokkos::create_mirror_view(ions.dx2);
      auto ux1_h = Kokkos::create_mirror_view(ions.ux1);
      auto ux2_h = Kokkos::create_mirror_view(ions.ux2);
      auto ux3_h = Kokkos::create_mirror_view(ions.ux3);
      auto wei_h = Kokkos::create_mirror_view(ions.weight);
      auto tag_h = Kokkos::create_mirror_view(ions.tag);
      npart_t n  = 0u;
      for (int g = 0; g < n_ions; ++g) {
        const auto   gid = static_cast<std::uint64_t>(g);
        const double xg  = NX * bhv::uniform(seed, gid, 0);
        const double yg  = NY * bhv::uniform(seed, gid, 1);
        const int    ig = static_cast<int>(xg), jg = static_cast<int>(yg);
        if (ig < off1 or ig >= off1 + n1 or jg < off2 or jg >= off2 + n2) {
          continue;
        }
        raise::ErrorIf(n >= ions.maxnpart(), "bh_validation: ion species full", HERE);
        double nx = 1.0, ny = 0.0, nz = 0.0;
        if (isotropic) {
          const double cos_t = 2.0 * bhv::uniform(seed, gid, 2) - 1.0;
          const double sin_t = std::sqrt(1.0 - cos_t * cos_t);
          const double phi   = 2.0 * constant::PI * bhv::uniform(seed, gid, 3);
          nx                 = sin_t * std::cos(phi);
          ny                 = sin_t * std::sin(phi);
          nz                 = cos_t;
        }
        i1_h(n)  = ig - off1;
        i2_h(n)  = jg - off2;
        dx1_h(n) = static_cast<prtldx_t>(xg - static_cast<double>(ig));
        dx2_h(n) = static_cast<prtldx_t>(yg - static_cast<double>(jg));
        // a float offset can round up to 1
        if (dx1_h(n) >= static_cast<prtldx_t>(1)) {
          dx1_h(n) = static_cast<prtldx_t>(0.999999);
        }
        if (dx2_h(n) >= static_cast<prtldx_t>(1)) {
          dx2_h(n) = static_cast<prtldx_t>(0.999999);
        }
        ux1_h(n) = static_cast<real_t>(u * nx);
        ux2_h(n) = static_cast<real_t>(u * ny);
        ux3_h(n) = static_cast<real_t>(u * nz);
        wei_h(n) = static_cast<real_t>(ion_weight);
        tag_h(n) = ParticleTag::alive;
        ++n;
      }
      Kokkos::deep_copy(ions.i1, i1_h);
      Kokkos::deep_copy(ions.i2, i2_h);
      Kokkos::deep_copy(ions.dx1, dx1_h);
      Kokkos::deep_copy(ions.dx2, dx2_h);
      Kokkos::deep_copy(ions.ux1, ux1_h);
      Kokkos::deep_copy(ions.ux2, ux2_h);
      Kokkos::deep_copy(ions.ux3, ux3_h);
      Kokkos::deep_copy(ions.weight, wei_h);
      Kokkos::deep_copy(ions.tag, tag_h);
      ions.set_npart(n);
      ions.set_counter(n);

      if (neutralizer_species > 0) {
        auto& neut = domain.species[neutralizer_species - 1];
        raise::ErrorIf(n > neut.maxnpart(),
                       "bh_validation: neutralizer species full",
                       HERE);
        auto ni1_h  = Kokkos::create_mirror_view(neut.i1);
        auto ni2_h  = Kokkos::create_mirror_view(neut.i2);
        auto ndx1_h = Kokkos::create_mirror_view(neut.dx1);
        auto ndx2_h = Kokkos::create_mirror_view(neut.dx2);
        auto nux1_h = Kokkos::create_mirror_view(neut.ux1);
        auto nux2_h = Kokkos::create_mirror_view(neut.ux2);
        auto nux3_h = Kokkos::create_mirror_view(neut.ux3);
        auto nwei_h = Kokkos::create_mirror_view(neut.weight);
        auto ntag_h = Kokkos::create_mirror_view(neut.tag);
        // charge-neutral with the emitters node by node
        const double w = ion_weight * std::abs(static_cast<double>(ions.charge()) /
                                               static_cast<double>(neut.charge()));
        for (npart_t p = 0u; p < n; ++p) {
          ni1_h(p)  = i1_h(p);
          ni2_h(p)  = i2_h(p);
          ndx1_h(p) = dx1_h(p);
          ndx2_h(p) = dx2_h(p);
          nux1_h(p) = ZERO;
          nux2_h(p) = ZERO;
          nux3_h(p) = ZERO;
          nwei_h(p) = static_cast<real_t>(w);
          ntag_h(p) = ParticleTag::alive;
        }
        Kokkos::deep_copy(neut.i1, ni1_h);
        Kokkos::deep_copy(neut.i2, ni2_h);
        Kokkos::deep_copy(neut.dx1, ndx1_h);
        Kokkos::deep_copy(neut.dx2, ndx2_h);
        Kokkos::deep_copy(neut.ux1, nux1_h);
        Kokkos::deep_copy(neut.ux2, nux2_h);
        Kokkos::deep_copy(neut.ux3, nux3_h);
        Kokkos::deep_copy(neut.weight, nwei_h);
        Kokkos::deep_copy(neut.tag, ntag_h);
        neut.set_npart(n);
        neut.set_counter(n);
      }
      if (stale_prev) {
        fill_stale_prev(domain, electron_species);
        fill_stale_prev(domain, positron_species);
      }
    }

    // previous positions of the unused slots of a species set to random
    // points of the domain, as dead particles would leave them
    void fill_stale_prev(Domain<S, M>& domain, int species) const {
      if (species <= 0) {
        return;
      }
      auto&      sp = domain.species[species - 1];
      const int  n1 = static_cast<int>(domain.mesh.n_active(in::x1));
      const int  n2 = static_cast<int>(domain.mesh.n_active(in::x2));
      auto i1_h  = Kokkos::create_mirror_view(sp.i1_prev);
      auto i2_h  = Kokkos::create_mirror_view(sp.i2_prev);
      auto dx1_h = Kokkos::create_mirror_view(sp.dx1_prev);
      auto dx2_h = Kokkos::create_mirror_view(sp.dx2_prev);
      Kokkos::deep_copy(i1_h, sp.i1_prev);
      Kokkos::deep_copy(i2_h, sp.i2_prev);
      Kokkos::deep_copy(dx1_h, sp.dx1_prev);
      Kokkos::deep_copy(dx2_h, sp.dx2_prev);
      const auto stream = static_cast<std::uint64_t>(seed) + 7919u * species +
                          104729u * domain.index();
      for (npart_t p = sp.npart(); p < sp.maxnpart(); ++p) {
        i1_h(p)  = static_cast<int>(n1 * bhv::uniform(stream, p, 10));
        i2_h(p)  = static_cast<int>(n2 * bhv::uniform(stream, p, 11));
        dx1_h(p) = static_cast<prtldx_t>(0.999 * bhv::uniform(stream, p, 12));
        dx2_h(p) = static_cast<prtldx_t>(0.999 * bhv::uniform(stream, p, 13));
      }
      Kokkos::deep_copy(sp.i1_prev, i1_h);
      Kokkos::deep_copy(sp.i2_prev, i2_h);
      Kokkos::deep_copy(sp.dx1_prev, dx1_h);
      Kokkos::deep_copy(sp.dx2_prev, dx2_h);
    }

#if defined(BHV_HAS_POLICY)
    auto EmissionPolicy(simtime_t, spidx_t, Domain<S, M>& domain) const {
      raise::ErrorIf(not negative_control,
                     "bh_validation: emission = custom needs "
                     "setup.negative_control = true",
                     HERE);
      const auto dt = params.template get<real_t>("algorithms.timestep.dt");
      const auto e_idx = params.template get<spidx_t>(
        "radiation.emission.bethe_heitler.electron_species");
      const auto p_idx = params.template get<spidx_t>(
        "radiation.emission.bethe_heitler.positron_species");
      auto& tables = ::arch::emission::bh::GetTables(params, dt);
      return bhv::NoPrevWrite<M> { ::arch::emission::BetheHeitler<M> {
        domain.species[e_idx - 1],
        domain.species[p_idx - 1],
        e_idx,
        p_idx,
        domain.species[ion_species - 1].mass(),
        dt,
        domain.index(),
        params,
        tables,
        domain.random_pool() } };
    }
#endif

    // sums over the alive particles of one species: n, sum w, sum w (K - K0),
    // sum w (K - K0)^2, sum w gamma, sum w K, with K = m (gamma - 1) [m_e c^2]
    void species_sums(const Particles<D, M::CoordType>& sp,
                      double                             K_ref,
                      std::vector<double>&               out) const {
      const auto   m  = static_cast<double>(sp.mass());
      const auto   ux1 = sp.ux1, ux2 = sp.ux2, ux3 = sp.ux3;
      const auto   weight = sp.weight;
      const auto   tag    = sp.tag;
      double       s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
      Kokkos::parallel_reduce(
        "BHVSums",
        sp.rangeActiveParticles(),
        KOKKOS_LAMBDA(prtlidx_t p,
                      double&   l0,
                      double&   l1,
                      double&   l2,
                      double&   l3,
                      double&   l4,
                      double&   l5) {
          if (tag(p) != ParticleTag::alive) {
            return;
          }
          const double a = static_cast<double>(ux1(p));
          const double b = static_cast<double>(ux2(p));
          const double c = static_cast<double>(ux3(p));
          const double u2    = a * a + b * b + c * c;
          const double gamma = math::sqrt(1.0 + u2);
          const double K     = m * u2 / (gamma + 1.0);
          const double w     = static_cast<double>(weight(p));
          l0 += 1.0;
          l1 += w;
          l2 += w * (K - K_ref);
          l3 += w * (K - K_ref) * (K - K_ref);
          l4 += w * gamma;
          l5 += w * K;
        },
        s0,
        s1,
        s2,
        s3,
        s4,
        s5);
      out = { s0, s1, s2, s3, s4, s5 };
    }

    // max over the nodes of |div E - kappa rho| and of |kappa rho|, |E|
    void gauss_residual(Domain<S, M>& domain, double& residual, double& rho_max, double& e_max) const {
      const int nx = static_cast<int>(resolution[0]);
      const int ny = static_cast<int>(resolution[1]);
      const auto off  = domain.offset_ncells();
      const int  off1 = static_cast<int>(off[0]), off2 = static_cast<int>(off[1]);
      const int  n1   = static_cast<int>(domain.mesh.n_active(in::x1));
      const int  n2   = static_cast<int>(domain.mesh.n_active(in::x2));
      const int  ng   = static_cast<int>(N_GHOSTS);
      const std::size_t nn = static_cast<std::size_t>(nx) * ny;
      // [ex1 | ex2 | rho]
      std::vector<double> buf(3 * nn, 0.0);
      auto em_h = Kokkos::create_mirror_view(domain.fields.em);
      Kokkos::deep_copy(em_h, domain.fields.em);
      for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
          const std::size_t g = static_cast<std::size_t>(off1 + i) +
                                static_cast<std::size_t>(nx) * (off2 + j);
          buf[g]      = static_cast<double>(em_h(i + ng, j + ng, em::ex1));
          buf[nn + g] = static_cast<double>(em_h(i + ng, j + ng, em::ex2));
        }
      }
      for (const auto& sp : domain.species) {
        const double q = static_cast<double>(sp.charge());
        if (q == 0.0 or sp.npart() == 0) {
          continue;
        }
        auto i1_h  = Kokkos::create_mirror_view(sp.i1);
        auto i2_h  = Kokkos::create_mirror_view(sp.i2);
        auto dx1_h = Kokkos::create_mirror_view(sp.dx1);
        auto dx2_h = Kokkos::create_mirror_view(sp.dx2);
        auto wei_h = Kokkos::create_mirror_view(sp.weight);
        auto tag_h = Kokkos::create_mirror_view(sp.tag);
        Kokkos::deep_copy(i1_h, sp.i1);
        Kokkos::deep_copy(i2_h, sp.i2);
        Kokkos::deep_copy(dx1_h, sp.dx1);
        Kokkos::deep_copy(dx2_h, sp.dx2);
        Kokkos::deep_copy(wei_h, sp.weight);
        Kokkos::deep_copy(tag_h, sp.tag);
        for (npart_t p = 0u; p < sp.npart(); ++p) {
          if (tag_h(p) != ParticleTag::alive) {
            continue;
          }
          const double w  = q * static_cast<double>(wei_h(p));
          const double fx = static_cast<double>(dx1_h(p));
          const double fy = static_cast<double>(dx2_h(p));
          const int    i  = ((off1 + i1_h(p)) % nx + nx) % nx, ip = (i + 1) % nx;
          const int    j  = ((off2 + i2_h(p)) % ny + ny) % ny, jp = (j + 1) % ny;
          buf[2 * nn + i + static_cast<std::size_t>(nx) * j]   += w * (1.0 - fx) * (1.0 - fy);
          buf[2 * nn + ip + static_cast<std::size_t>(nx) * j]  += w * fx * (1.0 - fy);
          buf[2 * nn + i + static_cast<std::size_t>(nx) * jp]  += w * (1.0 - fx) * fy;
          buf[2 * nn + ip + static_cast<std::size_t>(nx) * jp] += w * fx * fy;
        }
      }
      bhv::reduce_sum(buf);
      const double kappa = static_cast<double>(params.template get<real_t>("scales.q0")) /
                           (static_cast<double>(params.template get<real_t>("scales.B0")) *
                            static_cast<double>(params.template get<real_t>("scales.V0")));
      residual = 0.0;
      rho_max  = 0.0;
      e_max    = 0.0;
      for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
          const int         im = (i - 1 + nx) % nx, jm = (j - 1 + ny) % ny;
          const std::size_t g   = i + static_cast<std::size_t>(nx) * j;
          const std::size_t gim = im + static_cast<std::size_t>(nx) * j;
          const std::size_t gjm = i + static_cast<std::size_t>(nx) * jm;
          const double div = (buf[g] - buf[gim]) + (buf[nn + g] - buf[nn + gjm]);
          const double rho = kappa * buf[2 * nn + g];
          residual = std::max(residual, std::abs(div - rho));
          rho_max  = std::max(rho_max, std::abs(rho));
          e_max    = std::max(e_max, std::max(std::abs(buf[g]), std::abs(buf[nn + g])));
        }
      }
    }

    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& domain) {
      const auto istep = static_cast<int>(step);
#if defined(BHV_HAS_POLICY)
      if (istep == 0 and bhv::is_root() and
          params.contains("radiation.emission.bethe_heitler.electron_species")) {
        // nodes of the rate tables, for the reference curves of the analysis
        const auto& T = ::arch::emission::bh::GetTables(
          params,
          params.template get<real_t>("algorithms.timestep.dt"));
        std::FILE*  f = std::fopen("bhv_tables.txt", "w");
        if (f != nullptr) {
          std::fprintf(f, "# gamma_th %.17g gamma_min %.17g gamma_max %.17g n %zu\n",
                       T.host.gamma_th, T.host.gamma_min, T.host.gamma_max, T.host.size());
          std::fprintf(f, "# y gamma J I Dmean\n");
          for (std::size_t i = 0; i < T.host.size(); ++i) {
            std::fprintf(f, "%.17g %.17g %.17g %.17g %.17g\n", T.host.y[i], T.host.gamma[i],
                         T.host.J[i], T.host.I[i], T.host.Dmean[i]);
          }
          std::fclose(f);
        }
      }
#endif
      if (recycle and istep != dump_step) {
        // long cooling runs: the leptons are counted and removed every step
        std::vector<double> s;
        int                 k = 0;
        for (const int idx : { electron_species, positron_species }) {
          if (idx > 0 and idx != neutralizer_species) {
            auto& sp = domain.species[idx - 1];
            species_sums(sp, 0.0, s);
            recycled[3 * k + 0] += s[0];
            recycled[3 * k + 1] += s[1];
            recycled[3 * k + 2] += s[4];
            sp.set_npart(0);
          }
          ++k;
        }
      }
      // after n = istep + 1 steps, every `diag_interval` steps
      if (diag_interval > 0 and ((istep + 1) % diag_interval == 0 or istep == dump_step)) {
        const double K_ref = mass_ratio * (gamma_p - 1.0);
        // [ions(6) | electrons(6) | positrons(6) | neutralizer(6) | counters(6)]
        std::vector<double> all(30, 0.0);
        std::vector<double> s;
        species_sums(domain.species[ion_species - 1], K_ref, s);
        std::copy(s.begin(), s.end(), all.begin());
        if (electron_species > 0) {
          species_sums(domain.species[electron_species - 1], 0.0, s);
          std::copy(s.begin(), s.end(), all.begin() + 6);
        }
        if (positron_species > 0) {
          species_sums(domain.species[positron_species - 1], 0.0, s);
          std::copy(s.begin(), s.end(), all.begin() + 12);
        }
        if (neutralizer_species > 0) {
          species_sums(domain.species[neutralizer_species - 1], 0.0, s);
          std::copy(s.begin(), s.end(), all.begin() + 18);
        }
        // the recycled leptons count as created
        all[6]  += recycled[0];
        all[7]  += recycled[1];
        all[10] += recycled[2];
        all[12] += recycled[3];
        all[13] += recycled[4];
        all[16] += recycled[5];
#if defined(BHV_HAS_POLICY)
        {
          const auto c = ::arch::emission::bh::GetCounters();
          all[24]      = static_cast<double>(c.pairs);
          all[25]      = static_cast<double>(c.above_tables);
          all[26]      = static_cast<double>(c.probability_exceeded);
          all[27]      = static_cast<double>(c.loss_bound_skipped);
          all[28]      = static_cast<double>(c.capacity_dropped);
          all[29]      = c.absorbed_energy;
        }
#endif
        bhv::reduce_sum(all);
        double residual = 0.0, rho_max = 0.0, e_max = 0.0;
        if (gauss) {
          gauss_residual(domain, residual, rho_max, e_max);
        }
        if (bhv::is_root()) {
          std::printf("BHV step %d time %.17g K_ref %.17g", istep, static_cast<double>(time), K_ref);
          std::printf(" ion_n %.0f ion_w %.17g ion_dK %.17g ion_dK2 %.17g ion_K %.17g",
                      all[0], all[1], all[2], all[3], all[5]);
          std::printf(" el_n %.0f el_w %.17g el_E %.17g", all[6], all[7], all[10]);
          std::printf(" po_n %.0f po_w %.17g po_E %.17g", all[12], all[13], all[16]);
          std::printf(" ne_n %.0f ne_E %.17g", all[18], all[22]);
          std::printf(" pairs %.0f above %.0f pexc %.0f bounded %.0f dropped %.0f absorbed %.17g",
                      all[24], all[25], all[26], all[27], all[28], all[29]);
          if (gauss) {
            std::printf(" gauss_r %.9e rho_max %.9e e_max %.9e", residual, rho_max, e_max);
          }
          std::printf("\n");
          std::fflush(stdout);
        }
      }
      if (istep == dump_step) {
        dump(domain);
      }
    }

    // lepton momenta to binary files, and a hash of the fields and particles
    void dump(Domain<S, M>& domain) const {
      std::uint64_t h = 1469598103934665603ULL;
      {
        auto em_h = Kokkos::create_mirror_view(domain.fields.em);
        Kokkos::deep_copy(em_h, domain.fields.em);
        h = bhv::fnv(em_h.data(), em_h.span() * sizeof(real_t), h);
      }
      for (const auto& sp : domain.species) {
        const auto n = sp.npart();
        h            = bhv::fnv(&n, sizeof(n), h);
        auto i1_h  = Kokkos::create_mirror_view(sp.i1);
        auto i2_h  = Kokkos::create_mirror_view(sp.i2);
        auto dx1_h = Kokkos::create_mirror_view(sp.dx1);
        auto dx2_h = Kokkos::create_mirror_view(sp.dx2);
        auto ux1_h = Kokkos::create_mirror_view(sp.ux1);
        auto ux2_h = Kokkos::create_mirror_view(sp.ux2);
        auto ux3_h = Kokkos::create_mirror_view(sp.ux3);
        auto wei_h = Kokkos::create_mirror_view(sp.weight);
        auto tag_h = Kokkos::create_mirror_view(sp.tag);
        Kokkos::deep_copy(i1_h, sp.i1);
        Kokkos::deep_copy(i2_h, sp.i2);
        Kokkos::deep_copy(dx1_h, sp.dx1);
        Kokkos::deep_copy(dx2_h, sp.dx2);
        Kokkos::deep_copy(ux1_h, sp.ux1);
        Kokkos::deep_copy(ux2_h, sp.ux2);
        Kokkos::deep_copy(ux3_h, sp.ux3);
        Kokkos::deep_copy(wei_h, sp.weight);
        Kokkos::deep_copy(tag_h, sp.tag);
        h = bhv::fnv(i1_h.data(), n * sizeof(int), h);
        h = bhv::fnv(i2_h.data(), n * sizeof(int), h);
        h = bhv::fnv(dx1_h.data(), n * sizeof(prtldx_t), h);
        h = bhv::fnv(dx2_h.data(), n * sizeof(prtldx_t), h);
        h = bhv::fnv(ux1_h.data(), n * sizeof(real_t), h);
        h = bhv::fnv(ux2_h.data(), n * sizeof(real_t), h);
        h = bhv::fnv(ux3_h.data(), n * sizeof(real_t), h);
        h = bhv::fnv(wei_h.data(), n * sizeof(real_t), h);
        h = bhv::fnv(tag_h.data(), n * sizeof(short), h);
        const auto idx = static_cast<int>(sp.index());
        if (idx == electron_species or idx == positron_species or idx == ion_species) {
          const std::string name = "bhv_species_" + std::to_string(idx) + "_rank_" +
                                   std::to_string(bhv::rank()) + ".bin";
          std::FILE* f = std::fopen(name.c_str(), "wb");
          if (f != nullptr) {
            // header: n, sizeof(real_t); then ux1, ux2, ux3, weight; then tag
            const std::uint64_t hdr[2] = { static_cast<std::uint64_t>(n),
                                           static_cast<std::uint64_t>(sizeof(real_t)) };
            std::fwrite(hdr, sizeof(std::uint64_t), 2, f);
            std::fwrite(ux1_h.data(), sizeof(real_t), n, f);
            std::fwrite(ux2_h.data(), sizeof(real_t), n, f);
            std::fwrite(ux3_h.data(), sizeof(real_t), n, f);
            std::fwrite(wei_h.data(), sizeof(real_t), n, f);
            std::fwrite(tag_h.data(), sizeof(short), n, f);
            std::fclose(f);
          }
        }
      }
      std::printf("BHVHASH rank %d %016llx\n", bhv::rank(), static_cast<unsigned long long>(h));
      std::fflush(stdout);
    }
  };

} // namespace user

#endif // PROBLEM_GENERATOR_H
