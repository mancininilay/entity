/**
 * @file tests/kernels/injectors_payloads.cpp
 * @brief T10b (kernel level), constraint P9: every real payload slot of a newly
 *        injected particle is zero. The real payload arrays of the destination
 *        species are filled with NaN before the injection; after it, every slot
 *        in [offset, offset + n_injected) must be zero and every slot of the
 *        untouched tail must still be NaN (so the overload wrote exactly the
 *        injected slots). Exercised through
 *          - the `kernel::InjectParticle` overload taking the pld_r view
 *            (tracking and non-tracking instantiations, and a species with
 *            npld_r = 0 whose view is unallocated),
 *          - `kernel::UniformInjector_kernel`,
 *          - `kernel::NonUniformInjector_kernel`,
 *          - `kernel::GlobalInjector_kernel` (through a one-domain Metadomain),
 *        all in 1D Minkowski on the default execution space.
 *
 * Verdict lines: "PASS"/"FAIL" with the measured count; summary line at the end.
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include "archetypes/energy_dist.h"
#include "archetypes/spatial_dist.h"
#include "framework/containers/particles.h"
#include "framework/domain/metadomain.h"
#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;

namespace {

  int n_pass = 0, n_fail = 0;

  void verdict(bool ok, const std::string& what, double value, double expect) {
    std::printf("%s: %s : %.6g (expected %.6g)\n", ok ? "PASS" : "FAIL", what.c_str(),
                value, expect);
    if (ok) {
      ++n_pass;
    } else {
      ++n_fail;
    }
  }

  using M1 = Minkowski<Dim::_1D>;

  auto make_species(spidx_t idx, const char* label, float charge, npart_t maxnpart,
                    unsigned short npld_r, bool tracking) -> Particles<Dim::_1D, Coord::Cartesian> {
    return Particles<Dim::_1D, Coord::Cartesian> { idx,
                                                   label,
                                                   1.0f,
                                                   charge,
                                                   maxnpart,
                                                   0,
                                                   0,
                                                   ParticlePusher::BORIS,
                                                   tracking,
                                                   RadiativeDrag::NONE,
                                                   EmissionType::NONE,
                                                   npld_r,
                                                   static_cast<unsigned short>(tracking ? 1 : 0) };
  }

  void fill_nan(Particles<Dim::_1D, Coord::Cartesian>& sp) {
    if (sp.npld_r() > 0) {
      Kokkos::deep_copy(sp.pld_r, std::numeric_limits<real_t>::quiet_NaN());
    }
    Kokkos::fence();
  }

  /**
   * @brief count NaN payload entries in [p0, p1) and the zero entries there
   */
  void scan(const Particles<Dim::_1D, Coord::Cartesian>& sp, npart_t p0, npart_t p1,
            unsigned long long& n_nan, unsigned long long& n_zero, unsigned long long& n_total) {
    n_nan = n_zero = n_total = 0;
    if (sp.npld_r() == 0) {
      return;
    }
    auto h = Kokkos::create_mirror_view(sp.pld_r);
    Kokkos::deep_copy(h, sp.pld_r);
    for (npart_t p = p0; p < p1; ++p) {
      for (unsigned short c = 0; c < sp.npld_r(); ++c) {
        ++n_total;
        const double v = static_cast<double>(h(p, c));
        if (std::isnan(v)) {
          ++n_nan;
        } else if (v == 0.0) {
          ++n_zero;
        }
      }
    }
  }

  void check_species(const std::string& what, const Particles<Dim::_1D, Coord::Cartesian>& sp,
                     npart_t offset, npart_t n_injected) {
    unsigned long long nan_in, zero_in, tot_in, nan_tail, zero_tail, tot_tail;
    scan(sp, offset, offset + n_injected, nan_in, zero_in, tot_in);
    scan(sp, offset + n_injected, sp.maxnpart(), nan_tail, zero_tail, tot_tail);
    verdict(n_injected > 0, what + ": particles injected", static_cast<double>(n_injected), 1.0);
    verdict(nan_in == 0 && zero_in == tot_in,
            what + ": every real payload slot of the injected particles is zero (" +
              std::to_string(tot_in) + " slots)",
            static_cast<double>(zero_in), static_cast<double>(tot_in));
    verdict(nan_tail == tot_tail,
            what + ": the untouched tail still holds NaN (" + std::to_string(tot_tail) + " slots)",
            static_cast<double>(nan_tail), static_cast<double>(tot_tail));
  }

  /* ------------------------------------------------------------------------ */
  void test_overload_direct() {
    const npart_t maxn = 64;
    auto sp  = make_species(1, "e", -1.0f, maxn, 2, false);
    auto spt = make_species(2, "et", 1.0f, maxn, 3, true);
    auto sp0 = make_species(3, "e0", -1.0f, maxn, 0, false); // no real payloads
    fill_nan(sp);
    fill_nan(spt);
    const npart_t n = 10, offset = 5;
    {
      auto i1 = sp.i1, i2 = sp.i2, i3 = sp.i3;
      auto dx1 = sp.dx1, dx2 = sp.dx2, dx3 = sp.dx3;
      auto ux1 = sp.ux1, ux2 = sp.ux2, ux3 = sp.ux3, phi = sp.phi, w = sp.weight;
      auto tag = sp.tag;
      auto pldi = sp.pld_i;
      auto pldr = sp.pld_r;
      Kokkos::parallel_for(
        "direct", n, Lambda(npart_t k) {
          tuple_t<int, Dim::_1D>      xi { static_cast<int>(k) };
          tuple_t<prtldx_t, Dim::_1D> dxi { static_cast<prtldx_t>(0.25) };
          vec_t<Dim::_3D>             v { ONE, ZERO, ZERO };
          kernel::InjectParticle<Dim::_1D, Coord::Cartesian, false>(
            offset + k, i1, i2, i3, dx1, dx2, dx3, ux1, ux2, ux3, phi, w, tag, pldi, pldr, xi, dxi, v,
            static_cast<real_t>(2.0));
        });
      Kokkos::fence();
      check_species("InjectParticle overload (no tracking)", sp, offset, n);
    }
    {
      auto i1 = spt.i1, i2 = spt.i2, i3 = spt.i3;
      auto dx1 = spt.dx1, dx2 = spt.dx2, dx3 = spt.dx3;
      auto ux1 = spt.ux1, ux2 = spt.ux2, ux3 = spt.ux3, phi = spt.phi, w = spt.weight;
      auto tag = spt.tag;
      auto pldi = spt.pld_i;
      auto pldr = spt.pld_r;
      Kokkos::parallel_for(
        "direct_t", n, Lambda(npart_t k) {
          tuple_t<int, Dim::_1D>      xi { static_cast<int>(k) };
          tuple_t<prtldx_t, Dim::_1D> dxi { static_cast<prtldx_t>(0.5) };
          vec_t<Dim::_3D>             v { ZERO, ONE, ZERO };
          kernel::InjectParticle<Dim::_1D, Coord::Cartesian, true>(
            k, i1, i2, i3, dx1, dx2, dx3, ux1, ux2, ux3, phi, w, tag, pldi, pldr, xi, dxi, v, ONE, ZERO,
            0u, 100u + k);
        });
      Kokkos::fence();
      check_species("InjectParticle overload (tracking)", spt, 0, n);
      auto hi = Kokkos::create_mirror_view(spt.pld_i);
      Kokkos::deep_copy(hi, spt.pld_i);
      bool ok = true;
      for (npart_t k = 0; k < n; ++k) {
        ok = ok && hi(k, pldi::spcCtr) == 100u + k;
      }
      verdict(ok, "InjectParticle overload (tracking): species counter payload written", ok ? 1.0 : 0.0, 1.0);
    }
    {
      // npld_r = 0: the pld_r view is unallocated (extent(1) = 0) and the loop is a no-op
      auto i1 = sp0.i1, i2 = sp0.i2, i3 = sp0.i3;
      auto dx1 = sp0.dx1, dx2 = sp0.dx2, dx3 = sp0.dx3;
      auto ux1 = sp0.ux1, ux2 = sp0.ux2, ux3 = sp0.ux3, phi = sp0.phi, w = sp0.weight;
      auto tag = sp0.tag;
      auto pldi = sp0.pld_i;
      auto pldr = sp0.pld_r;
      verdict(pldr.extent(1) == 0, "npld_r = 0: pld_r view has extent(1) = 0",
              static_cast<double>(pldr.extent(1)), 0.0);
      Kokkos::parallel_for(
        "direct_0", n, Lambda(npart_t k) {
          tuple_t<int, Dim::_1D>      xi { static_cast<int>(k) };
          tuple_t<prtldx_t, Dim::_1D> dxi { static_cast<prtldx_t>(0.75) };
          vec_t<Dim::_3D>             v { ZERO, ZERO, ONE };
          kernel::InjectParticle<Dim::_1D, Coord::Cartesian, false>(
            k, i1, i2, i3, dx1, dx2, dx3, ux1, ux2, ux3, phi, w, tag, pldi, pldr, xi, dxi, v);
        });
      Kokkos::fence();
      auto ht = Kokkos::create_mirror_view(sp0.tag);
      Kokkos::deep_copy(ht, sp0.tag);
      bool ok = true;
      for (npart_t k = 0; k < n; ++k) {
        ok = ok && ht(k) == ParticleTag::alive;
      }
      verdict(ok, "npld_r = 0: injection through the overload completes (tags alive)", ok ? 1.0 : 0.0, 1.0);
    }
  }

  /* ------------------------------------------------------------------------ */
  void test_uniform_injector() {
    const std::vector<ncells_t> res { 32 };
    const boundaries_t<real_t>  ext { { ZERO, static_cast<real_t>(32.0) } };
    M1                          metric { res, ext, {} };
    const npart_t               maxn = 512;
    auto sp1 = make_species(1, "e-", -1.0f, maxn, 2, false);
    auto sp2 = make_species(2, "e+", 1.0f, maxn, 3, true);
    fill_nan(sp1);
    fill_nan(sp2);
    // a few particles already present (offset)
    sp1.set_npart(7);
    sp2.set_npart(7);
    random_number_pool_t pool { 12345u };
    array_t<real_t*>     xi_min { "xi_min", 1 }, xi_max { "xi_max", 1 };
    Kokkos::deep_copy(xi_min, static_cast<real_t>(N_GHOSTS));
    Kokkos::deep_copy(xi_max, static_cast<real_t>(32 + N_GHOSTS));
    const npart_t n = 100;
    using ED = arch::energy_dist::Cold<Dim::_1D>;
    Kokkos::parallel_for("UniformInjector", n,
                         kernel::UniformInjector_kernel<SimEngine::SRPIC, M1, ED, ED>(
                           sp1, sp2, 0u, metric, xi_min, xi_max, ED {}, ED {}, ONE, pool));
    Kokkos::fence();
    check_species("UniformInjector_kernel, species 1 (npld_r = 2)", sp1, 7, n);
    check_species("UniformInjector_kernel, species 2 (npld_r = 3, tracking)", sp2, 7, n);
  }

  /* ------------------------------------------------------------------------ */
  void test_nonuniform_injector() {
    const std::vector<ncells_t> res { 32 };
    const boundaries_t<real_t>  ext { { ZERO, static_cast<real_t>(32.0) } };
    M1                          metric { res, ext, {} };
    const npart_t               maxn = 4096;
    auto sp1 = make_species(1, "e-", -1.0f, maxn, 1, true);
    auto sp2 = make_species(2, "e+", 1.0f, maxn, 4, false);
    fill_nan(sp1);
    fill_nan(sp2);
    sp1.set_npart(3);
    sp2.set_npart(3);
    random_number_pool_t pool { 777u };
    using ED = arch::energy_dist::Cold<Dim::_1D>;
    using SD = arch::spatial_dist::Uniform<Dim::_1D>;
    auto kern = kernel::NonUniformInjector_kernel<SimEngine::SRPIC, M1, ED, ED, SD>(
      static_cast<real_t>(8.0), sp1, sp2, 0u, metric, ED {}, ED {}, SD {}, ONE, pool);
    Kokkos::parallel_for("NonUniformInjector",
                         CreateRangePolicy<Dim::_1D>({ N_GHOSTS }, { 32 + N_GHOSTS }), kern);
    Kokkos::fence();
    const auto n_inj = kern.number_injected();
    verdict(n_inj == 256, "NonUniformInjector_kernel: 8 ppc x 32 cells injected", static_cast<double>(n_inj), 256.0);
    check_species("NonUniformInjector_kernel, species 1 (npld_r = 1, tracking)", sp1, 3, n_inj);
    check_species("NonUniformInjector_kernel, species 2 (npld_r = 4)", sp2, 3, n_inj);
  }

  /* ------------------------------------------------------------------------ */
  void test_global_injector() {
    const std::vector<ncells_t> res { 32 };
    const boundaries_t<real_t>  ext { { ZERO, static_cast<real_t>(32.0) } };
    const boundaries_t<FldsBC>  fldsbc { { FldsBC::PERIODIC, FldsBC::PERIODIC } };
    const boundaries_t<PrtlBC>  prtlbc { { PrtlBC::PERIODIC, PrtlBC::PERIODIC } };
    const std::vector<int>      decomp { -1 };
    std::vector<ParticleSpecies> species;
    species.emplace_back(1, "e-", 1.0f, -1.0f, 1024, 0, 0, ParticlePusher::BORIS, false,
                         RadiativeDrag::NONE, EmissionType::NONE, 2, 0);
    species.emplace_back(2, "e+", 1.0f, 1.0f, 1024, 0, 0, ParticlePusher::BORIS, true,
                         RadiativeDrag::NONE, EmissionType::NONE, 3, 1);
    Metadomain<SimEngine::SRPIC, M1> md { 1u, decomp, res, ext, fldsbc, prtlbc, {}, species };
    const npart_t n = 50;
    std::map<std::string, std::vector<real_t>> data;
    for (npart_t k = 0; k < n; ++k) {
      data["x1"].push_back(static_cast<real_t>(0.5 + 31.0 * static_cast<double>(k) / n));
      data["ux1"].push_back(static_cast<real_t>(0.1));
      data["ux2"].push_back(ZERO);
      data["ux3"].push_back(ZERO);
    }
    md.runOnLocalDomains([&](auto& dom) {
      for (spidx_t s = 1; s <= 2; ++s) {
        auto& sp = dom.species[s - 1];
        fill_nan(sp);
        sp.set_npart(4);
        auto kern = kernel::GlobalInjector_kernel<SimEngine::SRPIC, M1>(sp, md.mesh().metric, dom, data,
                                                                        false);
        Kokkos::parallel_for("GlobalInjector", n, kern);
        Kokkos::fence();
        const auto n_inj = kern.number_injected();
        verdict(n_inj == n, "GlobalInjector_kernel, species " + std::to_string(s) + ": all injected",
                static_cast<double>(n_inj), static_cast<double>(n));
        check_species("GlobalInjector_kernel, species " + std::to_string(s) +
                        (s == 2 ? " (npld_r = 3, tracking)" : " (npld_r = 2)"),
                      sp, 4, n_inj);
      }
    });
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::printf("injectors_payloads: real_t = %s, execution space = %s\n",
                sizeof(real_t) == 4 ? "float" : "double", Kokkos::DefaultExecutionSpace::name());
    test_overload_direct();
    test_uniform_injector();
    test_nonuniform_injector();
    test_global_injector();
    std::printf("injectors_payloads summary: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0 || n_pass == 0) {
      rc = 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "injectors_payloads: exception: %s\n", e.what());
    rc = 2;
  }
  Kokkos::finalize();
  return rc;
}
