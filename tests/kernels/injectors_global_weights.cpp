/**
 * @file tests/kernels/injectors_global_weights.cpp
 * @brief [fork:hadronic/F1] `kernel::GlobalInjector_kernel` must write the
 *        supplied input weight of every injected particle (Codex 2's weight
 *        audit of 2026-10-29, finding F1: at 051bb896 the three dimensional
 *        branches read `weights(p)`, the destination species array at the
 *        *input* index p, instead of `in_wei(p)`).
 *
 *        Re-implementation of the audit's poisoned-destination reproducer as a
 *        ctest entry, through the actual wrapper `arch::InjectGlobally`:
 *          - the destination weight array is filled with 99 before the call
 *            (every slot, including the `offset` pre-existing particles and the
 *            untouched tail);
 *          - the input list carries weights 2 and 3 (the audit's values) and
 *            four more, with a unique `ux1 = k + 1` per input so that each
 *            injected slot can be matched back to its input whatever the order
 *            of the atomic slot reservation;
 *          - two of the inputs lie outside the domain in x1 and, from 2D on,
 *            one more lies outside in x2: they must be filtered, so the input
 *            index k and the destination slot `offset + index` differ;
 *          - the species already holds `offset = 5` particles (nonzero offset);
 *          - 1D, 2D and 3D Minkowski, a one-domain Metadomain;
 *          - weighted mode (`use_weights = true`: stored weight == input
 *            weight, exactly) and unit-weight mode (`use_weights = false`:
 *            stored weight == 1 whatever the input "weights" entry);
 *          - a non-tracking species with real payloads and a tracking species
 *            with integer payloads (both `InjectParticle` instantiations).
 *        Checked: the number injected equals the number of in-domain inputs;
 *        every injected slot carries its own input's weight (or 1); the
 *        pre-existing slots and the tail still hold 99; the injected tags are
 *        alive.
 *
 *        Expected at 051bb896: the weighted cases FAIL (99 stored instead of
 *        the inputs); after the fix every case passes.
 *
 * Verdict lines: "PASS"/"FAIL"; summary line at the end; exit 1 on any FAIL.
 */

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include "archetypes/particle_injector.h"
#include "framework/containers/particles.h"
#include "framework/domain/metadomain.h"
#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>

#include <cstdio>
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

  constexpr real_t POISON = static_cast<real_t>(99.0);

  template <Dimension D>
  void run_case(bool use_weights) {
    using M = Minkowski<D>;
    const auto Dn = static_cast<unsigned short>(D);
    const std::string tag = std::to_string(Dn) + "D, " +
                            (use_weights ? "weighted" : "unit-weight");

    const std::vector<ncells_t> res(Dn, 16);
    const boundaries_t<real_t>  ext(Dn, { ZERO, static_cast<real_t>(16.0) });
    const boundaries_t<FldsBC>  fldsbc(Dn, { FldsBC::PERIODIC, FldsBC::PERIODIC });
    const boundaries_t<PrtlBC>  prtlbc(Dn, { PrtlBC::PERIODIC, PrtlBC::PERIODIC });
    const std::vector<int>      decomp(Dn, -1);

    const npart_t maxn = 64;
    std::vector<ParticleSpecies> species;
    // species 1: no tracking, two real payloads; species 2: tracking, two integer payloads
    species.emplace_back(1, "e-", 1.0f, -1.0f, maxn, 0, 0, ParticlePusher::BORIS, false,
                         RadiativeDrag::NONE, EmissionType::NONE, 2, 0);
    species.emplace_back(2, "e+", 1.0f, 1.0f, maxn, 0, 0, ParticlePusher::BORIS, true,
                         RadiativeDrag::NONE, EmissionType::NONE, 0, 2);
    Metadomain<SimEngine::SRPIC, M> md { 1u, decomp, res, ext, fldsbc, prtlbc, {}, species };

    // the input list: k, x1, (x2), (x3), weight; two outside in x1, one outside in x2 (2D+)
    const std::vector<double> in_x1 { 2.0, -5.0, 3.0, 100.0, 5.0, 7.0, 9.0, 11.0 };
    const std::vector<double> in_x2 { 2.0, 2.0, 3.0, 3.0, 5.0, 20.0, 9.0, 11.0 };
    const std::vector<double> in_x3 { 2.0, 2.0, 3.0, 3.0, 5.0, 7.0, 9.0, 11.0 };
    const std::vector<double> in_w { 2.0, 1000.0, 3.0, 1000.0, 0.5, 4.0, 6.0, 8.0 };
    const npart_t             n_in = in_x1.size();

    std::map<std::string, std::vector<real_t>> data;
    std::vector<bool> inside(n_in, true);
    for (npart_t k = 0; k < n_in; ++k) {
      data["x1"].push_back(static_cast<real_t>(in_x1[k]));
      inside[k] = inside[k] && (in_x1[k] >= 0.0 && in_x1[k] < 16.0);
      if (Dn >= 2) {
        data["x2"].push_back(static_cast<real_t>(in_x2[k]));
        inside[k] = inside[k] && (in_x2[k] >= 0.0 && in_x2[k] < 16.0);
      }
      if (Dn >= 3) {
        data["x3"].push_back(static_cast<real_t>(in_x3[k]));
        inside[k] = inside[k] && (in_x3[k] >= 0.0 && in_x3[k] < 16.0);
      }
      data["ux1"].push_back(static_cast<real_t>(k + 1)); // unique id of the input
      data["ux2"].push_back(ZERO);
      data["ux3"].push_back(ZERO);
      data["weights"].push_back(static_cast<real_t>(in_w[k]));
    }
    npart_t n_expect = 0;
    for (npart_t k = 0; k < n_in; ++k) {
      n_expect += inside[k] ? 1 : 0;
    }

    md.runOnLocalDomains([&](auto& dom) {
      for (spidx_t s = 1; s <= 2; ++s) {
        auto&             sp   = dom.species[s - 1];
        const std::string what = tag + ", species " + std::to_string(s) +
                                 (s == 2 ? " (tracking)" : " (npld_r = 2)");
        const npart_t offset = 5;
        Kokkos::deep_copy(sp.weight, POISON); // poisoned destination, every slot
        Kokkos::deep_copy(sp.tag, ParticleTag::dead);
        Kokkos::fence();
        sp.set_npart(offset);

        arch::InjectGlobally<SimEngine::SRPIC, M>(md, dom, s, data, use_weights);
        Kokkos::fence();

        const npart_t n_inj = sp.npart() - offset;
        verdict(n_inj == n_expect, what + ": number injected = in-domain inputs",
                static_cast<double>(n_inj), static_cast<double>(n_expect));

        auto w_h   = Kokkos::create_mirror_view(sp.weight);
        auto ux1_h = Kokkos::create_mirror_view(sp.ux1);
        auto tag_h = Kokkos::create_mirror_view(sp.tag);
        Kokkos::deep_copy(w_h, sp.weight);
        Kokkos::deep_copy(ux1_h, sp.ux1);
        Kokkos::deep_copy(tag_h, sp.tag);

        // every injected slot carries its own input's weight (matched by ux1)
        std::vector<int> matched(n_in, 0);
        bool             ok_w = true, ok_match = true, ok_alive = true;
        double           first_bad = -1.0, first_bad_expect = -1.0;
        for (npart_t p = offset; p < offset + n_inj && p < maxn; ++p) {
          const auto id = static_cast<long>(static_cast<double>(ux1_h(p)) + 0.5) - 1;
          if (id < 0 || id >= static_cast<long>(n_in) || !inside[id]) {
            ok_match = false;
            continue;
          }
          ++matched[id];
          const real_t expect = use_weights ? static_cast<real_t>(in_w[id]) : ONE;
          if (w_h(p) != expect) {
            if (ok_w) {
              first_bad        = static_cast<double>(w_h(p));
              first_bad_expect = static_cast<double>(expect);
            }
            ok_w = false;
          }
          ok_alive = ok_alive && (tag_h(p) == ParticleTag::alive);
        }
        for (npart_t k = 0; k < n_in; ++k) {
          ok_match = ok_match && (matched[k] == (inside[k] ? 1 : 0));
        }
        verdict(ok_match, what + ": each in-domain input injected exactly once, none outside",
                ok_match ? 1.0 : 0.0, 1.0);
        verdict(ok_w,
                what + ": stored weight of every injected slot == " +
                  (use_weights ? "its input weight" : "1"),
                ok_w ? 1.0 : first_bad, ok_w ? 1.0 : first_bad_expect);
        verdict(ok_alive, what + ": injected tags alive", ok_alive ? 1.0 : 0.0, 1.0);

        // the pre-existing slots and the tail are untouched (still 99)
        bool ok_pre = true, ok_tail = true;
        for (npart_t p = 0; p < offset; ++p) {
          ok_pre = ok_pre && (w_h(p) == POISON);
        }
        for (npart_t p = offset + n_inj; p < maxn; ++p) {
          ok_tail = ok_tail && (w_h(p) == POISON);
        }
        verdict(ok_pre, what + ": the " + std::to_string(offset) + " pre-existing slots still hold 99",
                ok_pre ? 1.0 : 0.0, 1.0);
        verdict(ok_tail, what + ": the untouched tail still holds 99", ok_tail ? 1.0 : 0.0, 1.0);
      }
    });
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::printf("injectors_global_weights: real_t = %s, execution space = %s\n",
                sizeof(real_t) == 4 ? "float" : "double", Kokkos::DefaultExecutionSpace::name());
    run_case<Dim::_1D>(true);
    run_case<Dim::_1D>(false);
    run_case<Dim::_2D>(true);
    run_case<Dim::_2D>(false);
    run_case<Dim::_3D>(true);
    run_case<Dim::_3D>(false);
    std::printf("injectors_global_weights summary: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0 || n_pass == 0) {
      rc = 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "injectors_global_weights: exception: %s\n", e.what());
    rc = 2;
  }
  Kokkos::finalize();
  return rc;
}
