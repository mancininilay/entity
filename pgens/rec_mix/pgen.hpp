/**
 * @file pgens/rec_mix/pgen.hpp
 * @brief P1.3 `rec_mix`: relativistic reconnection in an electron-positron-ion
 *        plasma, Entity port of Tristan v2 `user_rec-mix.F90` (active ions)
 *        and `user_rec-mix_testprtl.F90` (passive test ions).
 *
 * Layouts (setup.layout):
 *   "active"   : 1,2,3 = e-, e+, ion of the sheet; 4,5,6 = e-, e+, ion of the
 *                lower inflow side (x2 < y0); 7,8,9 = same for the upper side.
 *   "testprtl" : 1,2 = sheet pairs; 4,5 / 7,8 = lower / upper pairs;
 *                3, 6, 9 = three passive ion masses (deposit = false),
 *                loaded on both sides and absent from the sheet.
 *
 * Coordinate map (P1.3 D29): (x1, x2, x3) = (y_T, x_T, -z_T); x1 = outflow,
 * x2 = inflow, sheet at x2 = y0. Grid must have unit cells: extent = ncells.
 *
 * Normalization (P13-D39): setup.sigma_hot, dup_cells and theta_up are the
 * hot p1.1 labels; sigma0 = sigma_hot (D + 5T/2), skindepth0 = d_up sqrt(F),
 * or the literal saved Tristan inputs (setup.sigma_input, c_omp) after a check.
 *
 * Test geometries (setup.geometry, setup.field_test): "closed_box" (doubly
 * periodic, two sheets at N2/4 and 3N2/4 with opposite currents, no injector,
 * padding or opening; CV = whole box: C1.14 closed-box drift); "pulse_x2",
 * "pulse_x1" (vacuum pulses towards the inflow or outflow walls: C1.5).
 *
 * CustomPostStep order (P1.3 tech spec II.2):
 *   (1) open the x1 boundaries at t >= t_open (also after a restart);
 *   (2) injector (11-cell average, Poisson/ceiling rounding, capacity guard,
 *       payload zeroing);
 *   (3) padding handling (u = -/+0.1 toward the wall, removal beyond
 *       padding - nfilter);
 *   (4) control-volume energy budget update;
 *   (5) masked spectra and moments, incl. per-species bulk-velocity energies;
 *   (6) full-resolution midplane rows.
 *
 * Outputs in <simulation.name>/usr/ (formats documented in README.md):
 *   recmix.meta.json, recmix.<step>.bin, cvbudget.bin, midline.<step>.bin,
 *   segments.jsonl, fields_manifest.jsonl, and with setup.validation_dump the
 *   exact particle state vdump.<step>.r<rank>.bin
 * Custom field outputs: "mixing" (M), "mask" (0/1), "count_<s>".
 *
 * [fork:P1.3/C1] Author: Claude (2026-10-04). Reviewer: Codex (A3).
 */

#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/pgen.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/numeric.h"

#include "archetypes/energy_dist.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

namespace user {
  using namespace ntt;

  namespace recmix {

    enum Role : int {
      SHEET  = 0,
      LOWER  = 1,
      UPPER  = 2,
      TRACER = 3
    };

    inline auto splitmix64(std::uint64_t z) -> std::uint64_t {
      z += 0x9E3779B97F4A7C15ULL;
      z  = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z  = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      return z ^ (z >> 31);
    }

    inline auto mpi_rank() -> int {
#if defined(MPI_ENABLED)
      int r { 0 };
      MPI_Comm_rank(MPI_COMM_WORLD, &r);
      return r;
#else
      return 0;
#endif
    }

    // sum a host vector over ranks onto rank 0 (in place)
    inline void reduce_to_root(std::vector<double>& v) {
#if defined(MPI_ENABLED)
      if (v.empty()) {
        return;
      }
      if (mpi_rank() == 0) {
        MPI_Reduce(MPI_IN_PLACE, v.data(), (int)v.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      } else {
        MPI_Reduce(v.data(), nullptr, (int)v.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      }
#else
      (void)v;
#endif
    }

    inline void reduce_to_root(std::vector<float>& v) {
#if defined(MPI_ENABLED)
      if (v.empty()) {
        return;
      }
      if (mpi_rank() == 0) {
        MPI_Reduce(MPI_IN_PLACE, v.data(), (int)v.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      } else {
        MPI_Reduce(v.data(), nullptr, (int)v.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      }
#else
      (void)v;
#endif
    }

    inline auto all_max(double x) -> double {
#if defined(MPI_ENABLED)
      double y { x };
      MPI_Allreduce(&x, &y, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      return y;
#else
      return x;
#endif
    }

    /* ------------------------------------------------------------------ */
    /* Field functors (2D, D29 map)                                       */
    /* ------------------------------------------------------------------ */
    // Initial fields. open: one sheet at y0, B_x1 = tanh((x2 - y0)/a).
    // closed (C1.14 closed box): two sheets at y1 = N2/4 and y2 = 3 N2/4 in a
    // doubly periodic box, B_x1 = tanh((x2 - y1)/a) - tanh((x2 - y2)/a) - 1.
    // bg scales the background (0 for the C1.5 vacuum tests). pulse: 0 none;
    // 1 two plane pulses along x2 (E_x3, B_x1), centred at c_lo (moving to -x2,
    // B_x1 = -E_x3) and c_hi (moving to +x2, B_x1 = +E_x3); 2 the same along x1
    // (E_x3, B_x2) centred at c_lo (moving to -x1, B_x2 = +E_x3) and c_hi (+x1,
    // B_x2 = -E_x3). Gaussian profile amp exp(-(x - c)^2/(2 w^2)), uniform in
    // the other direction: an exact vacuum plane wave up to the Yee dispersion.
    template <Dimension D>
    struct InitFields {
      InitFields(real_t a,
                 real_t y0,
                 real_t bguide,
                 real_t bg     = ONE,
                 bool   closed = false,
                 real_t y1     = ZERO,
                 real_t y2     = ZERO,
                 int    pulse  = 0,
                 real_t c_lo   = ZERO,
                 real_t c_hi   = ZERO,
                 real_t pw     = ONE,
                 real_t pamp   = ZERO)
        : a { a }
        , y0 { y0 }
        , bguide { bguide }
        , bg { bg }
        , closed { closed }
        , y1 { y1 }
        , y2 { y2 }
        , pulse { pulse }
        , c_lo { c_lo }
        , c_hi { c_hi }
        , pw { pw }
        , pamp { pamp } {}

      Inline auto g(real_t x, real_t c) const -> real_t {
        return pamp * math::exp(-HALF * SQR((x - c) / pw));
      }

      Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
        real_t b = closed ? (math::tanh((x_Ph[1] - y1) / a) -
                             math::tanh((x_Ph[1] - y2) / a) - ONE)
                          : math::tanh((x_Ph[1] - y0) / a);
        b *= bg;
        if (pulse == 1) {
          b += g(x_Ph[1], c_hi) - g(x_Ph[1], c_lo);
        }
        return b;
      }

      Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
        return (pulse == 2) ? (g(x_Ph[0], c_lo) - g(x_Ph[0], c_hi)) : ZERO;
      }

      Inline auto bx3(const coord_t<D>&) const -> real_t {
        return -bguide * bg; // D29: B_x3 = -B_z,T
      }

      Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
        if (pulse == 1) {
          return g(x_Ph[1], c_lo) + g(x_Ph[1], c_hi);
        }
        if (pulse == 2) {
          return g(x_Ph[0], c_lo) + g(x_Ph[0], c_hi);
        }
        return ZERO;
      }

    private:
      // non-const: re-assigned in the PGen constructor
      real_t a, y0, bguide, bg;
      bool   closed;
      real_t y1, y2;
      int    pulse;
      real_t c_lo, c_hi, pw, pamp;
    };

    // outflow (x1) walls, used after t_open: reconnected targets (I.3, D29)
    // (bg scales the targets: 0 for the C1.5 vacuum tests; the MATCH update
    // F <- s F + (1 - s) F_target is linear in F - F_target, kernels/fields_bcs.hpp)
    template <Dimension D>
    struct BoundaryFieldsInX1 {
      BoundaryFieldsInX1(real_t a,
                         real_t x0,
                         real_t y0,
                         real_t bguide,
                         real_t beta_rec,
                         real_t bg = ONE)
        : a { a }
        , x0 { x0 }
        , y0 { y0 }
        , bguide { bguide }
        , beta_rec { beta_rec }
        , bg { bg } {}

      Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
        return bg * math::tanh((x_Ph[1] - y0) / a);
      }

      Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
        return bg * beta_rec * math::tanh((x_Ph[0] - x0) / a);
      }

      Inline auto bx3(const coord_t<D>&) const -> real_t {
        return -bg * bguide;
      }

      // E.B = 0 upstream with E_x3 = -beta_rec, B_x3 = -bguide
      Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {
        return -bg * beta_rec * bguide * math::tanh((x_Ph[1] - y0) / a);
      }

      Inline auto ex2(const coord_t<D>&) const -> real_t {
        return ZERO;
      }

      Inline auto ex3(const coord_t<D>&) const -> real_t {
        return -bg * beta_rec; // D29: E_x3 = -E_z,T
      }

    private:
      const real_t a, x0, y0, bguide, beta_rec, bg;
    };

    // inflow (x2) walls: initial field, E -> 0
    template <Dimension D>
    struct BoundaryFieldsInX2 {
      BoundaryFieldsInX2(real_t a, real_t y0, real_t bguide, real_t bg = ONE)
        : a { a }
        , y0 { y0 }
        , bguide { bguide }
        , bg { bg } {}

      Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
        return bg * math::tanh((x_Ph[1] - y0) / a);
      }

      Inline auto bx2(const coord_t<D>&) const -> real_t {
        return ZERO;
      }

      Inline auto bx3(const coord_t<D>&) const -> real_t {
        return -bg * bguide;
      }

      Inline auto ex1(const coord_t<D>&) const -> real_t {
        return ZERO;
      }

      Inline auto ex2(const coord_t<D>&) const -> real_t {
        return ZERO;
      }

      Inline auto ex3(const coord_t<D>&) const -> real_t {
        return ZERO;
      }

    private:
      const real_t a, y0, bguide, bg;
    };

    /* ------------------------------------------------------------------ */
    /* Exact Maxwell-Juttner sampler for all temperatures.                */
    /* Kinetic energy e = gamma - 1 has density                            */
    /*   f(e) ~ (1 + e) sqrt(e) sqrt(e + 2) exp(-e/theta).                 */
    /* Envelope: sqrt(e + 2) <= sqrt(2) + sqrt(e), so                      */
    /*   f <= [sqrt2 e^1/2 + e + sqrt2 e^3/2 + e^2] exp(-e/theta),         */
    /* a mixture of Gamma(k, theta), k = 3/2, 2, 5/2, 3, with weights      */
    /* c_k Gamma(k) theta^k (c = sqrt2, 1, sqrt2, 1); acceptance           */
    /* sqrt(e + 2)/(sqrt2 + sqrt e) >= 0.707. Entity's stock sampler uses  */
    /* a non-relativistic Gaussian below theta = 0.5, which underestimates */
    /* the pressure of mildly relativistic sheet ions (measured: 23% at    */
    /* theta = 0.17).                                                      */
    /* The drift is applied with the flipping method (Zenitani 2015), as   */
    /* in Entity's energy_dist::Maxwellian.                                */
    /* ------------------------------------------------------------------ */
    template <class G>
    Inline auto unif(G& gen) -> double {
      double r = static_cast<double>(Random<real_t>(gen));
      while (r <= 0.0) {
        r = static_cast<double>(Random<real_t>(gen));
      }
      return r;
    }

    template <class G>
    Inline auto gauss2(G& gen) -> double { // sum of squares of 2 N(0,1)
      return -2.0 * math::log(unif(gen));
    }

    template <class G>
    Inline auto gauss1sq(G& gen) -> double { // square of one N(0,1)
      const double r = math::sqrt(-2.0 * math::log(unif(gen)));
      const double c = math::cos(constant::TWO_PI * unif(gen));
      return SQR(r * c);
    }

    template <class G>
    Inline void sampleJuttner(G& gen, double theta, double ud3, vec_t<Dim::_3D>& v) {
      double u { 0.0 };
      if (theta > 0.0) {
        const double w1 = constant::SQRT2 * 0.886226925452758 * math::pow(theta, 1.5);
        const double w2 = 1.0 * theta * theta;
        const double w3 = constant::SQRT2 * 1.329340388179137 * math::pow(theta, 2.5);
        const double w4 = 2.0 * theta * theta * theta;
        const double wt = w1 + w2 + w3 + w4;
        double       e { 0.0 };
        for (int it = 0; it < 1000; ++it) {
          const double r = unif(gen) * wt;
          if (r < w1) { // Gamma(3/2): chi2(3)/2
            e = 0.5 * theta * (gauss2(gen) + gauss1sq(gen));
          } else if (r < w1 + w2) { // Gamma(2)
            e = -theta * math::log(unif(gen) * unif(gen));
          } else if (r < w1 + w2 + w3) { // Gamma(5/2): chi2(5)/2
            e = 0.5 * theta * (gauss2(gen) + gauss2(gen) + gauss1sq(gen));
          } else { // Gamma(3)
            e = -theta * math::log(unif(gen) * unif(gen) * unif(gen));
          }
          const double acc = math::sqrt(e + 2.0) / (constant::SQRT2 + math::sqrt(e));
          if (unif(gen) <= acc) {
            break;
          }
        }
        u = math::sqrt(e * (e + 2.0));
      }
      // isotropic direction
      const double mu  = 2.0 * unif(gen) - 1.0;
      const double phi = constant::TWO_PI * unif(gen);
      const double st  = math::sqrt(1.0 - mu * mu);
      double       u1 = u * st * math::cos(phi), u2 = u * st * math::sin(phi), u3 = u * mu;
      // flipping-method boost along x3 by the four-velocity |ud3|, then orient
      if (ud3 != 0.0) {
        const double ad = (ud3 > 0.0) ? ud3 : -ud3;
        const double G0 = math::sqrt(1.0 + ad * ad);
        const double bd = ad / G0;
        const double gp = math::sqrt(1.0 + u1 * u1 + u2 * u2 + u3 * u3);
        if (-bd * u3 > gp * unif(gen)) {
          u3 = -u3;
        }
        u3 = G0 * (u3 + bd * gp);
        if (ud3 < 0.0) {
          u3 = -u3;
        }
      }
      v[0] = static_cast<real_t>(u1);
      v[1] = static_cast<real_t>(u2);
      v[2] = static_cast<real_t>(u3);
    }

    /* ------------------------------------------------------------------ */
    /* Box injector: Tristan's fillRegionWithThermalPlasma                */
    /*   ntrials positions uniform in the (local) box; optional sheet     */
    /*   acceptance cosh^-2 * (1 - exp(-r^2/(5a)^2)); co-located particles */
    /*   of 1 or 2 species, each from its own (drifting) Maxwellian.       */
    /* ------------------------------------------------------------------ */
    template <class M>
    struct BoxInject_kernel {
      static constexpr auto D = M::Dim;

      struct Arr {
        array_t<int*>      i1, i2, i3;
        array_t<prtldx_t*> dx1, dx2, dx3;
        array_t<real_t*>   ux1, ux2, ux3, phi, weight;
        array_t<short*>    tag;
        array_t<npart_t**> pld_i;
        array_t<real_t**>  pld_r;
        npart_t            offset { 0 }, cntr { 0 };
        bool               tracking { false };
        unsigned short     ntrk { 0 }, npld_i { 0 }, npld_r { 0 };
      };

      Arr              A, B;
      bool             two;
      real_t           x1min, x1max, x2min, x2max; // local code coords
      int              n1loc, n2loc;
      double           thA, thB, uA3, uB3;         // theta (own mass), drift u_x3
      bool             sheet_profile;
      real_t           a, x0, y0, dip2;            // global coords
      real_t           off1, off2;                 // local -> global
      npart_t          dom_idx;
      random_number_pool_t pool;
      array_t<npart_t> idx { "rm_inj_idx" };

      template <class P>
      static auto make_arr(P& sp) -> Arr {
        Arr r;
        r.i1       = sp.i1;
        r.i2       = sp.i2;
        r.i3       = sp.i3;
        r.dx1      = sp.dx1;
        r.dx2      = sp.dx2;
        r.dx3      = sp.dx3;
        r.ux1      = sp.ux1;
        r.ux2      = sp.ux2;
        r.ux3      = sp.ux3;
        r.phi      = sp.phi;
        r.weight   = sp.weight;
        r.tag      = sp.tag;
        r.pld_i    = sp.pld_i;
        r.pld_r    = sp.pld_r;
        r.offset   = sp.npart();
        r.cntr     = sp.counter();
        r.tracking = sp.use_tracking();
#if defined(MPI_ENABLED)
        r.ntrk = sp.use_tracking() ? 2 : 0;
#else
        r.ntrk = sp.use_tracking() ? 1 : 0;
#endif
        r.npld_i = sp.npld_i();
        r.npld_r = sp.npld_r();
        return r;
      }

      Inline void write(const Arr&                  s,
                        npart_t                     p,
                        const tuple_t<int, D>&      xi,
                        const tuple_t<prtldx_t, D>& dxi,
                        const vec_t<Dim::_3D>&      v,
                        npart_t                     k) const {
        if (s.tracking) {
          kernel::InjectParticle<D, M::CoordType, true>(p,
                                                        s.i1, s.i2, s.i3,
                                                        s.dx1, s.dx2, s.dx3,
                                                        s.ux1, s.ux2, s.ux3,
                                                        s.phi, s.weight, s.tag,
                                                        s.pld_i, xi, dxi, v,
                                                        ONE, ZERO, dom_idx,
                                                        s.cntr + k);
        } else {
          kernel::InjectParticle<D, M::CoordType, false>(p,
                                                         s.i1, s.i2, s.i3,
                                                         s.dx1, s.dx2, s.dx3,
                                                         s.ux1, s.ux2, s.ux3,
                                                         s.phi, s.weight, s.tag,
                                                         s.pld_i, xi, dxi, v,
                                                         ONE, ZERO);
        }
        // K4: zero every payload slot the generator (or later CH3/CH4) uses
        for (auto c { s.ntrk }; c < s.npld_i; ++c) {
          s.pld_i(p, c) = 0;
        }
        for (auto c { 0 }; c < s.npld_r; ++c) {
          s.pld_r(p, c) = ZERO;
        }
      }

      Inline void operator()(npart_t) const {
        if constexpr (D == Dim::_2D) {
          auto   gen = pool.get_state();
          real_t x1  = x1min + Random<real_t>(gen) * (x1max - x1min);
          real_t x2  = x2min + Random<real_t>(gen) * (x2max - x2min);
          bool   acc = true;
          if (sheet_profile) {
            const real_t X1 = x1 + off1, X2 = x2 + off2;
            const real_t ch = math::cosh((X2 - y0) / a);
            const real_t r2 = SQR(X1 - x0) + SQR(X2 - y0);
            const real_t f  = (ONE / SQR(ch)) * (ONE - math::exp(-r2 / dip2));
            acc             = (Random<real_t>(gen) < f);
          }
          if (not acc) {
            pool.free_state(gen);
            return;
          }
          vec_t<Dim::_3D> v { ZERO }, w { ZERO };
          sampleJuttner(gen, thA, uA3, v);
          if (two) {
            sampleJuttner(gen, thB, uB3, w);
          }
          pool.free_state(gen);
          int i1 = static_cast<int>(x1);
          int i2 = static_cast<int>(x2);
          i1     = (i1 > n1loc - 1) ? (n1loc - 1) : ((i1 < 0) ? 0 : i1);
          i2     = (i2 > n2loc - 1) ? (n2loc - 1) : ((i2 < 0) ? 0 : i2);
          prtldx_t d1 = static_cast<prtldx_t>(x1 - static_cast<real_t>(i1));
          prtldx_t d2 = static_cast<prtldx_t>(x2 - static_cast<real_t>(i2));
          d1 = (d1 >= static_cast<prtldx_t>(1)) ? static_cast<prtldx_t>(0.9999)
                                                : ((d1 < static_cast<prtldx_t>(0)) ? static_cast<prtldx_t>(0) : d1);
          d2 = (d2 >= static_cast<prtldx_t>(1)) ? static_cast<prtldx_t>(0.9999)
                                                : ((d2 < static_cast<prtldx_t>(0)) ? static_cast<prtldx_t>(0) : d2);
          const tuple_t<int, D>      xi { i1, i2 };
          const tuple_t<prtldx_t, D> dxi { d1, d2 };
          const npart_t k = Kokkos::atomic_fetch_add(&idx(), static_cast<npart_t>(1));
          write(A, A.offset + k, xi, dxi, v, k);
          if (two) {
            write(B, B.offset + k, xi, dxi, w, k);
          }
        }
      }
    };

    /* ------------------------------------------------------------------ */
    /* Masked spectra and moments of one species                         */
    /* ------------------------------------------------------------------ */
    namespace mom {
      enum : int {
        N = 0,   // sum w (masked)
        K,       // sum w m (gamma - 1)
        G,       // sum w gamma
        U1, U2, U3,
        B1, B2, B3,
        NUNDER, EUNDER, // clamped into the first bin: count, sum w m (gamma-1)
        NOVER, EOVER,   // clamped into the last bin
        NSKIP,          // gamma - 1 <= 0 (skipped from the log histogram)
        KB_OWN,         // sum over masked cells N_s m (gamma_b,s - 1), own bulk,
                        // with the sampling-variance correction of |beta_b|^2
        KB_ALL,         // same with the all-species mass-weighted bulk (p1.1 velx);
                        // NaN for non-depositing species (not in the velocity set)
        NTOT,           // sum w over the whole domain (unmasked)
        KTOT,           // sum w m (gamma - 1) over the whole domain
        KB_OWN_RAW,     // Kb_own without the variance correction (= stock V_s)
        NLOWOCC,        // masked N_s in cells whose own-species box has < bulk_nmin particles
        KB_OWN_W1,      // Kb_own with window 1 (3 x 3 box): window sensitivity (IC5)
        NLOWOCC_W1,     // Nlowocc with window 1
        KLOWOCC,        // masked sum w m (gamma - 1) in cells whose own-species box has
                        // < bulk_nmin particles (energy share of under-occupied boxes, RV1a)
        KBLOWOCC,       // Kb_own (corrected) assigned to those cells
        KLOWOCC_W1,     // KLOWOCC with window 1
        KBLOWOCC_W1,    // KBLOWOCC with window 1
        KB_OWN_HALF,    // Kb_own with the box means from the particles with dx1 < 1/2 only
                        // (about half), weighted by the full cell counts: position-
                        // subsampling sensitivity (RV1b; dx1 can correlate with velocity, A0-3b)
        NMOM
      };

      inline auto names() -> std::vector<std::string> {
        return { "N",      "K",     "G",      "U1",     "U2",     "U3",
                 "B1",     "B2",    "B3",     "Nunder", "Eunder", "Nover",
                 "Eover",  "Nskip", "Kb_own", "Kb_all", "Ntot",   "Ktot",
                 "Kb_own_raw", "Nlowocc", "Kb_own_w1", "Nlowocc_w1", "Klowocc",
                 "Kblowocc", "Klowocc_w1", "Kblowocc_w1", "Kb_own_half" };
      }
    } // namespace mom

    template <class M>
    struct MaskedMoments_kernel {
      using value_type = double[];
      using size_type  = int;
      static constexpr int NPART_MOM = mom::KB_OWN; // particle-side entries

      const int                    value_count { mom::NMOM };
      array_t<int*>                i1, i2;
      array_t<real_t*>             ux1, ux2, ux3, weight;
      array_t<short*>              tag;
      Kokkos::View<real_t**>       mix;
      Kokkos::Experimental::ScatterView<double*> hist;
      real_t                       thr;
      double                       mass, lnmin, lnmax;
      int                          nbins;

      Inline void init(double* v) const {
        for (int c = 0; c < mom::NMOM; ++c) {
          v[c] = 0.0;
        }
      }

      Inline void join(double* dst, const double* src) const {
        for (int c = 0; c < mom::NMOM; ++c) {
          dst[c] += src[c];
        }
      }

      Inline void operator()(prtlidx_t p, double* v) const {
        if (tag(p) != ParticleTag::alive) {
          return;
        }
        const double u1 = ux1(p), u2 = ux2(p), u3 = ux3(p), w = weight(p);
        const double usq = u1 * u1 + u2 * u2 + u3 * u3;
        const double gam = math::sqrt(1.0 + usq);
        const double e   = usq / (gam + 1.0); // gamma - 1, cancellation-free
        v[mom::NTOT] += w;
        v[mom::KTOT] += w * mass * e;
        if (not(mix(i1(p), i2(p)) > thr)) {
          return;
        }
        v[mom::N]  += w;
        v[mom::K]  += w * mass * e;
        v[mom::G]  += w * gam;
        v[mom::U1] += w * u1;
        v[mom::U2] += w * u2;
        v[mom::U3] += w * u3;
        v[mom::B1] += w * u1 / gam;
        v[mom::B2] += w * u2 / gam;
        v[mom::B3] += w * u3 / gam;
        if (not(e > 0.0)) {
          v[mom::NSKIP] += w;
          return;
        }
        // Tristan binning (user_rec-mix.F90:647-659): ln(E + 1e-8), clamped
        const double le  = math::log(e + 1e-8);
        int          bin = 0;
        if (le <= lnmin) {
          bin              = 0;
          v[mom::NUNDER]  += w;
          v[mom::EUNDER]  += w * mass * e;
        } else if (le >= lnmax) {
          bin             = nbins - 1;
          v[mom::NOVER]  += w;
          v[mom::EOVER]  += w * mass * e;
        } else {
          bin = static_cast<int>(math::ceil((le - lnmin) * nbins / (lnmax - lnmin))) - 1;
          bin = (bin < 0) ? 0 : ((bin > nbins - 1) ? nbins - 1 : bin);
        }
        auto h  = hist.access();
        h(bin) += w;
      }
    };

    // Bulk kinetic energy over masked cells: sum N_s(cell) (gamma_b(cell) - 1).
    // The bulk velocity is the top-hat box (half-width win) mean
    // b = sum(beta)/n from bckp components cn (n) and cb..cb+2 (sum beta);
    // N_s(cell) is component cs at the cell itself. If cq >= 0, component cq
    // holds sum |beta|^2 and |b|^2 is corrected by the sampling variance,
    // |b|^2 - s^2/n with s^2 = (sum|beta|^2 - n|b|^2)/(n - 1), an unbiased
    // estimate of the squared population mean (boxes with n < 2 uncorrected).
    // Results: v[0] corrected, v[1] raw, v[2] N_s in boxes with n < nmin.
    template <Dimension D>
    struct BulkEnergy_kernel {
      using value_type = double[];
      using size_type  = int;
      const int              value_count { 5 };
      ndfield_t<D, 6>        buf;
      Kokkos::View<real_t**> mix;
      real_t                 thr;
      int                    win, cn, cb, cs, cq;
      int                    ck { -1 }; // component with sum w (gamma - 1) per cell, or -1
      double                 nmin;

      // v: 0 bulk (corrected), 1 bulk (raw), 2 count, 3 kinetic, 4 bulk (corrected),
      // the last three over cells whose box holds fewer than nmin particles
      Inline void init(double* v) const {
        v[0] = v[1] = v[2] = v[3] = v[4] = 0.0;
      }

      Inline void join(double* d, const double* s) const {
        d[0] += s[0];
        d[1] += s[1];
        d[2] += s[2];
        d[3] += s[3];
        d[4] += s[4];
      }

      Inline void operator()(int i, int j, double* v) const {
        if (not(mix(i, j) > thr)) {
          return;
        }
        const int    I  = i + static_cast<int>(N_GHOSTS);
        const int    J  = j + static_cast<int>(N_GHOSTS);
        const double nc = buf(I, J, cs);
        if (not(nc > 0.0)) {
          return;
        }
        double n { 0.0 }, b1 { 0.0 }, b2 { 0.0 }, b3 { 0.0 }, bq { 0.0 };
        for (int dj = -win; dj <= win; ++dj) {
          for (int di = -win; di <= win; ++di) {
            n  += buf(I + di, J + dj, cn);
            b1 += buf(I + di, J + dj, cb);
            b2 += buf(I + di, J + dj, cb + 1);
            b3 += buf(I + di, J + dj, cb + 2);
            if (cq >= 0) {
              bq += buf(I + di, J + dj, cq);
            }
          }
        }
        if (not(n > 0.0)) {
          return;
        }
        const double cap = 1.0 - 1e-12;
        const double braw = (b1 * b1 + b2 * b2 + b3 * b3) / (n * n);
        double       bcor = braw;
        if (cq >= 0 and n > 1.5) {
          const double s2 = (bq - n * braw) / (n - 1.0);
          bcor            = braw - s2 / n;
          bcor            = (bcor > 0.0) ? bcor : 0.0;
        }
        const double kbc = nc * (1.0 / math::sqrt(1.0 - ((bcor < cap) ? bcor : cap)) - 1.0);
        v[0] += kbc;
        v[1] += nc * (1.0 / math::sqrt(1.0 - ((braw < cap) ? braw : cap)) - 1.0);
        if (n < nmin) {
          v[2] += nc;
          if (ck >= 0) {
            v[3] += buf(I, J, ck);
          }
          v[4] += kbc;
        }
      }
    };

  } // namespace recmix

  /* ==================================================================== */
  /* Problem generator                                                    */
  /* ==================================================================== */
  template <SimEngine::type S, class M>
  struct PGen {
    static constexpr auto D { M::Dim };
    static constexpr auto engines {
      ::traits::pgen::compatible_with<SimEngine::SRPIC> {}
    };
    static constexpr auto metrics {
      ::traits::pgen::compatible_with<Metric::Minkowski> {}
    };
    // 3D (planes instead of rows, 3D masks) is deferred to R1-3D
    static constexpr auto dimensions {
      ::traits::pgen::compatible_with<Dim::_2D> {}
    };

    const SimulationParams& params;
    Metadomain<S, M>&       metadomain;

    /* --- inputs ------------------------------------------------------- */
    std::string         layout, sheet_init;
    bool                active { true };
    // P13-D39: nominal hot labels; optional literal Tristan inputs
    double              sigma_hot { 0 }, q { 0 }, mu { 1 };
    double              ref_sigma_rtol { 1e-4 }, ref_comp_atol { 0.01 };
    std::string         norm_mode { "nominal" }, norm_label;
    // test geometries: closed box (C1.14 drift), vacuum pulse tests (C1.5)
    std::string         geometry { "open" }, field_test { "none" };
    bool                closed { false };
    int                 ftest { 0 }; // 0 none, 1 pulse_x2, 2 pulse_x1
    double              pulse_w { 5 }, pulse_off { 0 }, pulse_amp { 0.1 };
    bool                load_plasma { true };
    double              bg_field { 1 };
    bool                vdump { false }; // exact particle-state dump (A0-2, C1.12)
    std::vector<double> test_masses, test_temperatures;
    double              dup_cells { 5 }, theta_up { 0.01 }, eta { 3 };
    double              a_cells { 400 }, dip_width { 5 }, sheet_extent { 10 };
    double              b_guide { 0 }, beta_rec { 0.1 };
    double              t_open_tlc { 1.35 };
    int                 padding { 12 }, ds_abs { 10 }, nfilter { 0 };
    double              thr { 0.01 };
    int                 usr_interval { 500 }, line_interval { 100 };
    int                 cv_interval { 1 }, cv_margin { 10 };
    int                 line_halfwidth { 3 };
    double              spec_min { 1e-3 }, spec_max { 1e5 };
    int                 spec_num { 300 };
    int                 bulk_window { 2 }, inj_avg_cells { 11 };
    int                 bulk_nmin { 20 }; // box occupancy below which the bulk estimate is flagged
    bool                inject_on { true }, load_tracers { true }; // test switches
    int                 cv_slot_cfg { -1 }; // user int payload slot of the CV flag (-1: last)
    bool                allow_any_pusher { false };

    /* --- derived ------------------------------------------------------ */
    int    N1 { 0 }, N2 { 0 };
    double x0 { 0 }, y0 { 0 };
    double Dmass { 1 }, sigma0 { 0 }, skindepth0 { 0 }, larmor0 { 0 };
    double Hfac { 1 }, Fskin { 1 }, sigma_input_nom { 0 }, c_omp_nom { 0 };
    double y1 { 0 }, y2 { 0 }, yline { 0 }; // closed-box sheets; midline row centre
    bool   cv_whole { false };              // control volume = whole (periodic) box
    double ppc0 { 0 }, dt { 0 };
    double beta_d { 0 }, u_d { 0 }, T_cs { 0 }, t_open { 0 };
    double w_in { 0 }, w_out { 0 };
    int    cv1a { 0 }, cv1b { 0 }, cv2a { 0 }, cv2b { 0 };
    std::vector<int> role;     // per species (0-based)
    std::vector<int> mask_lo, mask_up, inj_count;

    /* --- state -------------------------------------------------------- */
    bool                   bc_opened { false };
    bool                   cv_started { false };
    std::uint64_t          host_seed { 0 };
    Kokkos::View<real_t**> m_mix;
    Kokkos::View<real_t***> m_cnt;
    std::vector<double>    cv_in_E, cv_out_E, cv_in_N, cv_out_N;
    double                 cv_flux_int { 0.0 }, cv_jwork_int { 0.0 };
    timestep_t             cv_last_step { 0 };
    std::filesystem::path  usr_dir;
    std::uint64_t          run_id { 0 }, start_label { 0 };
    timestep_t             last_field_noted { static_cast<timestep_t>(-1) };

    recmix::InitFields<D> init_flds;

    /* ------------------------------------------------------------------ */
    auto has(const std::string& k) const -> bool {
      return params.contains("setup." + k);
    }

    auto getReal(const std::string& k, double def) const -> double {
      if (not has(k)) {
        return def;
      }
      const auto& any = params.allVars().at("setup." + k);
      if (any.type() == typeid(real_t)) {
        return static_cast<double>(std::any_cast<real_t>(any));
      } else if (any.type() == typeid(int)) {
        return static_cast<double>(std::any_cast<int>(any));
      }
      raise::Error("rec_mix: setup." + k + " must be a number", HERE);
      return def;
    }

    auto getInt(const std::string& k, int def) const -> int {
      if (not has(k)) {
        return def;
      }
      const auto& any = params.allVars().at("setup." + k);
      if (any.type() == typeid(int)) {
        return std::any_cast<int>(any);
      } else if (any.type() == typeid(real_t)) {
        const auto v = static_cast<double>(std::any_cast<real_t>(any));
        raise::ErrorIf(std::abs(v - std::round(v)) > 1e-6,
                       "rec_mix: setup." + k + " must be an integer",
                       HERE);
        return static_cast<int>(std::lround(v));
      }
      raise::Error("rec_mix: setup." + k + " must be an integer", HERE);
      return def;
    }

    auto getStr(const std::string& k, const std::string& def) const -> std::string {
      return has(k) ? params.template get<std::string>("setup." + k) : def;
    }

    auto getRealVec(const std::string& k) const -> std::vector<double> {
      std::vector<double> out;
      if (not has(k)) {
        return out;
      }
      const auto& any = params.allVars().at("setup." + k);
      if (any.type() == typeid(std::vector<real_t>)) {
        for (auto v : std::any_cast<std::vector<real_t>>(any)) {
          out.push_back(v);
        }
      } else if (any.type() == typeid(std::vector<int>)) {
        for (auto v : std::any_cast<std::vector<int>>(any)) {
          out.push_back(v);
        }
      } else {
        raise::Error("rec_mix: setup." + k + " must be an array of numbers", HERE);
      }
      return out;
    }

    static void info(const std::string& msg) {
      if (recmix::mpi_rank() == 0) {
        std::cout << "[rec_mix] " << msg << std::endl;
      }
    }

    /* ------------------------------------------------------------------ */
    PGen(const SimulationParams& p, Metadomain<S, M>& m)
      : params { p }
      , metadomain { m }
      , init_flds { ONE, ZERO, ZERO } {
      layout     = getStr("layout", "active");
      sheet_init = getStr("sheet_init", "legacy");
      raise::ErrorIf(layout != "active" and layout != "testprtl",
                     "rec_mix: setup.layout must be \"active\" or \"testprtl\"",
                     HERE);
      raise::ErrorIf(sheet_init != "legacy" and sheet_init != "balanced",
                     "rec_mix: setup.sheet_init must be \"legacy\" or \"balanced\"",
                     HERE);
      active = (layout == "active");

      raise::ErrorIf(has("sigma"),
                     "rec_mix: setup.sigma is no longer accepted (P13-D39): the p1.1 labels "
                     "are hot magnetizations. Give sigma_hot (and optionally the literal "
                     "Tristan inputs sigma_input, c_omp); see make_input.py",
                     HERE);
      sigma_hot    = getReal("sigma_hot", 0.0);
      ref_sigma_rtol = getReal("ref_sigma_rtol", 1e-4);
      ref_comp_atol  = getReal("ref_comp_atol", 0.01);
      norm_label     = getStr("normalization_label", "");
      geometry       = getStr("geometry", "open");
      field_test     = getStr("field_test", "none");
      raise::ErrorIf(geometry != "open" and geometry != "closed_box",
                     "rec_mix: setup.geometry must be \"open\" or \"closed_box\"",
                     HERE);
      raise::ErrorIf(field_test != "none" and field_test != "pulse_x2" and
                       field_test != "pulse_x1",
                     "rec_mix: setup.field_test must be \"none\", \"pulse_x2\" or \"pulse_x1\"",
                     HERE);
      closed = (geometry == "closed_box");
      ftest  = (field_test == "pulse_x2") ? 1 : ((field_test == "pulse_x1") ? 2 : 0);
      raise::ErrorIf(closed and ftest != 0,
                     "rec_mix: closed_box and field_test are exclusive",
                     HERE);
      pulse_w   = getReal("pulse_width", 5.0);
      pulse_amp = getReal("pulse_amp", 0.1);
      vdump     = has("validation_dump") ? params.template get<bool>("setup.validation_dump")
                                         : false;
      q            = active ? getReal("q", 0.0) : 0.0;
      mu           = active ? getReal("mu", 1.0) : 1.0;
      dup_cells    = getReal("dup_cells", 5.0);
      theta_up     = getReal("theta_up", 0.01);
      eta          = getReal("cs_overdensity", 3.0);
      a_cells      = getReal("cs_width", 400.0);
      dip_width    = getReal("dip_width", 5.0);
      sheet_extent = getReal("sheet_extent", 10.0);
      b_guide      = getReal("b_guide", 0.0);
      beta_rec     = getReal("beta_rec_target", 0.1);
      t_open_tlc   = getReal("t_open_tlc", 1.35);
      padding      = getInt("injector_padding", 12);
      ds_abs       = getInt("ds_abs", 10);
      thr          = getReal("mixing_threshold", 0.01);
      usr_interval = getInt("usr_interval", 500);
      line_interval  = getInt("line_interval", 100);
      cv_interval    = getInt("cv_interval", 1);
      cv_margin      = getInt("cv_margin", 10);
      line_halfwidth = getInt("line_halfwidth", 3);
      spec_min       = getReal("spec_min", 1e-3);
      spec_max       = getReal("spec_max", 1e5);
      spec_num       = getInt("spec_num", 300);
      bulk_window    = getInt("bulk_window", 2);
      bulk_nmin      = getInt("bulk_nmin", 20);
      inj_avg_cells  = getInt("inj_avg_cells", 11);
      cv_slot_cfg    = getInt("cv_payload_slot", -1);
      allow_any_pusher = has("allow_any_pusher")
                           ? params.template get<bool>("setup.allow_any_pusher")
                           : false;
      inject_on      = has("inject") ? params.template get<bool>("setup.inject") : true;
      load_tracers   = has("load_tracers") ? params.template get<bool>("setup.load_tracers")
                                           : true;

      raise::ErrorIf(sigma_hot <= 0.0, "rec_mix: setup.sigma_hot must be > 0", HERE);
      if (closed or ftest != 0) {
        inject_on = false; // test layouts: no injector
      }
      if (ftest != 0) {
        load_plasma = false; // vacuum pulse test: no particles, zero background
        bg_field    = 0.0;
      }
      raise::ErrorIf(active and (q < 0.0 or q > 1.0),
                     "rec_mix: setup.q must be in [0, 1]",
                     HERE);
      raise::ErrorIf(active and q > 0.0 and mu <= 0.0,
                     "rec_mix: setup.mu must be > 0",
                     HERE);
      raise::ErrorIf(bulk_window < 0 or bulk_window > static_cast<int>(N_GHOSTS),
                     "rec_mix: setup.bulk_window must be in [0, N_GHOSTS]",
                     HERE);
      raise::ErrorIf(spec_min <= 0.0 or spec_max <= spec_min or spec_num < 2,
                     "rec_mix: bad spectral binning",
                     HERE);
      raise::ErrorIf(usr_interval < 1 or line_interval < 1 or cv_interval < 1,
                     "rec_mix: intervals must be >= 1",
                     HERE);
      raise::ErrorIf(usr_interval % cv_interval != 0,
                     "rec_mix: usr_interval must be a multiple of cv_interval",
                     HERE);

      /* grid: unit cells, origin at 0 -------------------------------------- */
      const auto& gm = metadomain.mesh();
      N1             = static_cast<int>(gm.n_active(in::x1));
      N2             = static_cast<int>(gm.n_active(in::x2));
      const auto e1  = gm.extent(in::x1);
      const auto e2  = gm.extent(in::x2);
      raise::ErrorIf(std::abs(e1.first) > 1e-6 or std::abs(e2.first) > 1e-6 or
                       std::abs(e1.second - N1) > 1e-3 or
                       std::abs(e2.second - N2) > 1e-3,
                     "rec_mix: grid.extent must be [[0, N1], [0, N2]] (unit "
                     "cells) to match the Tristan conventions",
                     HERE);
      raise::ErrorIf(N2 % 2 != 0, "rec_mix: N2 must be even", HERE);
      x0 = 0.5 * N1;
      y0 = 0.5 * N2;
      y1 = 0.25 * N2;
      y2 = 0.75 * N2;
      yline = closed ? y1 : y0;
      raise::ErrorIf(closed and N2 % 4 != 0, "rec_mix: closed_box needs N2 % 4 == 0", HERE);
      pulse_off = getReal("pulse_offset", 0.25 * ((ftest == 2) ? N1 : N2));

      /* algorithms ---------------------------------------------------------- */
      nfilter = static_cast<int>(
        params.template get<unsigned short>("algorithms.current_filters"));
      if (padding < nfilter + 4) {
        info(fmt::format("WARNING: injector_padding %d < nfilter + 4; set to %d",
                         padding,
                         nfilter + 4));
        padding = nfilter + 4;
      }
      dt   = static_cast<double>(params.template get<real_t>("algorithms.timestep.dt"));
      ppc0 = static_cast<double>(params.template get<real_t>("particles.ppc0"));

      /* units (tech spec III.1, P13-D39) -------------------------------------- */
      // nominal hot labels: sigma_hot = B^2/(4 pi w), w = n0 me c^2 (D + 5T/2) for a
      // common physical T = theta_up and Gamma = 5/3; d_up = dup_cells includes the
      // thermal and ion response with Nilay's factor F. Code inputs (Tristan's
      // fiducial cold electron-mass sigma, c_omp; n0 = ppc0, all upstream particles):
      // sigma0 = sigma_hot (D + 5T/2), skindepth0 = d_up sqrt(F). Testprtl: only
      // the depositing pairs (q = 0, D = 1). Literal inputs (the saved p1.1 values)
      // are used as given after a check within the saved precision.
      {
        const double T  = theta_up;
        const double th = 1.0 + 1.5 * T;
        Dmass           = active ? (2.0 - q + q * mu) / 2.0 : 1.0;
        Hfac            = Dmass + 2.5 * T;
        Fskin           = active ? ((2.0 - q) / th + q / (mu * th)) / 2.0 : 1.0 / th;
        sigma_input_nom = sigma_hot * Hfac;
        c_omp_nom       = dup_cells * std::sqrt(Fskin);
        const bool ls   = has("sigma_input"), lc = has("c_omp");
        raise::ErrorIf(ls != lc,
                       "rec_mix: give both setup.sigma_input and setup.c_omp (literal "
                       "inputs) or neither",
                       HERE);
        if (ls) {
          norm_mode       = "literal";
          sigma0          = getReal("sigma_input", 0.0);
          skindepth0      = getReal("c_omp", 0.0);
          const double ds = std::abs(sigma0 / sigma_input_nom - 1.0);
          const double dc = std::abs(skindepth0 - c_omp_nom);
          raise::ErrorIf(ds > ref_sigma_rtol or dc > ref_comp_atol,
                         fmt::format("rec_mix: literal inputs sigma_input = %.9g, c_omp = "
                                     "%.9g disagree with the hot labels (sigma_hot = %.9g, "
                                     "d_up = %.9g, T = %.9g give %.9g, %.9g): |rel| %.3g "
                                     "(tol %.3g), |abs| %.3g cells (tol %.3g)",
                                     sigma0,
                                     skindepth0,
                                     sigma_hot,
                                     dup_cells,
                                     T,
                                     sigma_input_nom,
                                     c_omp_nom,
                                     ds,
                                     ref_sigma_rtol,
                                     dc,
                                     ref_comp_atol),
                         HERE);
        } else {
          norm_mode  = "nominal";
          sigma0     = sigma_input_nom;
          skindepth0 = c_omp_nom;
        }
      }
      larmor0       = skindepth0 / std::sqrt(sigma0);
      const auto sd = static_cast<double>(params.template get<real_t>("scales.skindepth0"));
      const auto lr = static_cast<double>(params.template get<real_t>("scales.larmor0"));
      raise::ErrorIf(std::abs(sd / skindepth0 - 1.0) > 1e-4 or
                       std::abs(lr / larmor0 - 1.0) > 1e-4,
                     fmt::format("rec_mix: [scales] disagree with the physical "
                                 "inputs: need skindepth0 = %.9g, larmor0 = %.9g "
                                 "(sigma0 = %.9g); got %.9g, %.9g",
                                 skindepth0,
                                 larmor0,
                                 sigma0,
                                 sd,
                                 lr),
                     HERE);

      /* sheet --------------------------------------------------------------- */
      beta_d = (eta > 0.0) ? std::sqrt(sigma0) * skindepth0 / (a_cells * eta) : 0.0;
      raise::ErrorIf(beta_d >= 1.0, "rec_mix: sheet drift beta_d >= 1", HERE);
      u_d = beta_d / std::sqrt(1.0 - beta_d * beta_d);
      if (eta <= 0.0) {
        T_cs = 0.0; // no sheet
      } else if (sheet_init == "legacy") {
        T_cs = 0.5 * sigma0 * (1.0 + b_guide * b_guide) / (eta * Dmass);
      } else {
        // pressure balance with the drifting sheet: the transverse stress of a
        // population boosted along x3 is n_rest kT = n_lab kT / Gamma_d, so
        // eta n0 T_cs / Gamma_d = B0^2/8pi requires T_cs = sigma0 Gamma_d/(2 eta)
        T_cs = 0.5 * sigma0 * std::sqrt(1.0 + u_d * u_d) / eta;
      }
      {
        const double cc = (ftest == 2) ? x0 : y0;
        init_flds = recmix::InitFields<D> { static_cast<real_t>(a_cells),
                                            static_cast<real_t>(y0),
                                            static_cast<real_t>(b_guide),
                                            static_cast<real_t>(bg_field),
                                            closed,
                                            static_cast<real_t>(y1),
                                            static_cast<real_t>(y2),
                                            ftest,
                                            static_cast<real_t>(cc - pulse_off),
                                            static_cast<real_t>(cc + pulse_off),
                                            static_cast<real_t>(pulse_w),
                                            static_cast<real_t>(pulse_amp) };
      }

      t_open = has("t_open") ? getReal("t_open", 0.0) : t_open_tlc * N1;
      if (closed or ftest == 1) {
        t_open = -1.0; // never opened (closed box; x2 pulse test keeps x1 periodic)
      }

      /* boundaries ---------------------------------------------------------- */
      const auto fbc = params.template get<boundaries_t<FldsBC>>("grid.boundaries.fields");
      const auto pbc = params.template get<boundaries_t<PrtlBC>>("grid.boundaries.particles");
      {
        const bool resuming   = params.template get<bool>("checkpoint.is_resuming");
        const auto start_time = static_cast<double>(
          params.template get<simtime_t>("checkpoint.start_time"));
        const bool x1_open = (fbc[0].first == FldsBC::MATCH and fbc[0].second == FldsBC::MATCH and
                              pbc[0].first == PrtlBC::ABSORB and pbc[0].second == PrtlBC::ABSORB);
        const bool x1_per  = (fbc[0].first == FldsBC::PERIODIC and pbc[0].first == PrtlBC::PERIODIC);
        if (closed or ftest == 1) {
          raise::ErrorIf(not x1_per,
                         "rec_mix: closed_box and the pulse_x2 test need x1 PERIODIC",
                         HERE);
        } else if (ftest == 2) {
          // x1 pulse test: the outflow walls (MATCH + ABSORB, width w_out) from the
          // start; Entity registers the MATCH widths only if a wall is MATCH at start
          raise::ErrorIf(not x1_open, "rec_mix: the pulse_x1 test needs x1 MATCH/ABSORB", HERE);
          bc_opened = true;
          t_open    = -1.0;
        } else if (x1_open) {
          // restart past t_open with the open walls given in the restart input:
          // no step runs with periodic x1 (avoids the one-step transient, K5)
          raise::ErrorIf(not resuming or start_time < t_open,
                         "rec_mix: x1 MATCH/ABSORB at start is only allowed when resuming "
                         "past t_open",
                         HERE);
          bc_opened = true;
        } else {
          raise::ErrorIf(not x1_per,
                         "rec_mix: x1 must start PERIODIC (opened at t_open), or "
                         "MATCH/ABSORB when resuming past t_open",
                         HERE);
          if (resuming and start_time >= t_open) {
            info("WARNING: resuming past t_open with periodic x1 in the input: the walls "
                 "re-open at the first post-step (one step with periodic x1). Set x1 to "
                 "MATCH/ABSORB in the restart input to avoid it.");
          }
        }
      }
      const bool x2_per = (fbc[1].first == FldsBC::PERIODIC and pbc[1].first == PrtlBC::PERIODIC);
      if (closed or ftest == 2) {
        raise::ErrorIf(not x2_per,
                       "rec_mix: closed_box and the pulse_x1 test need x2 PERIODIC",
                       HERE);
      } else if (ftest == 1) {
        // x2 pulse test: MATCH, or CONDUCTOR as a reflecting control
        raise::ErrorIf(not((fbc[1].first == FldsBC::MATCH and fbc[1].second == FldsBC::MATCH) or
                           (fbc[1].first == FldsBC::CONDUCTOR and
                            fbc[1].second == FldsBC::CONDUCTOR)),
                       "rec_mix: the pulse_x2 test needs x2 MATCH (or CONDUCTOR, control)",
                       HERE);
      } else {
        raise::ErrorIf(fbc[1].first != FldsBC::MATCH or fbc[1].second != FldsBC::MATCH or
                         pbc[1].first != PrtlBC::ABSORB or pbc[1].second != PrtlBC::ABSORB,
                       "rec_mix: x2 must be MATCH (fields) and ABSORB (particles)",
                       HERE);
      }
      const bool any_match = (fbc[0].first == FldsBC::MATCH or fbc[0].second == FldsBC::MATCH or
                              fbc[1].first == FldsBC::MATCH or fbc[1].second == FldsBC::MATCH);
      if (closed or (ftest != 0 and not any_match)) {
        w_out = w_in = 0.0; // no absorbing layers (Entity registers ds only with a MATCH wall)
      } else {
        const auto ds = params.template get<boundaries_t<real_t>>("grid.boundaries.match.ds");
        w_out = std::max(ds[0].first, ds[0].second);
        w_in  = std::max(ds[1].first, ds[1].second);
      }
      if (not closed and std::abs(w_out - 2.0 * ds_abs) > 1e-6) {
        info(fmt::format("WARNING: match ds in x1 = %g differs from 2*ds_abs = %d (III.4)",
                         w_out,
                         2 * ds_abs));
      }
      if (not closed and std::abs(w_in - 1.5 * (padding - nfilter)) > 1e-6) {
        info(fmt::format("WARNING: match ds in x2 = %g differs from "
                         "1.5*(padding - nfilter) = %g (III.4)",
                         w_in,
                         1.5 * (padding - nfilter)));
      }

      /* control volume (III.13) ---------------------------------------------- */
      cv_whole = closed or (ftest != 0);
      if (cv_whole) {
        // test layouts: the whole box. Closed box (periodic): no surface, no
        // crossings, the records are the global energy budget. Field tests: the
        // CV only avoids an empty box; their records are not a budget (the
        // walls absorb, and no flux is integrated)
        cv1a = 0;
        cv1b = N1;
        cv2a = 0;
        cv2b = N2;
      } else {
        cv1a = static_cast<int>(std::ceil(w_out)) + cv_margin;
        cv1b = N1 - static_cast<int>(std::ceil(w_out)) - cv_margin;
        cv2a = padding + 1 + cv_margin;
        cv2b = N2 - padding - 1 - cv_margin;
      }
      raise::ErrorIf(cv1b <= cv1a or cv2b <= cv2a, "rec_mix: empty control volume", HERE);

      /* species -------------------------------------------------------------- */
      const auto& sp = metadomain.species_params();
      raise::ErrorIf(sp.size() != 9, "rec_mix: exactly 9 species are required", HERE);
      role.assign(9, recmix::SHEET);
      if (active) {
        role = { recmix::SHEET, recmix::SHEET, recmix::SHEET,
                 recmix::LOWER, recmix::LOWER, recmix::LOWER,
                 recmix::UPPER, recmix::UPPER, recmix::UPPER };
        mask_lo   = { 4, 5, 6 };
        mask_up   = { 7, 8, 9 };
        inj_count = { 4, 5, 6, 7, 8, 9 };
      } else {
        role = { recmix::SHEET,  recmix::SHEET, recmix::TRACER,
                 recmix::LOWER,  recmix::LOWER, recmix::TRACER,
                 recmix::UPPER,  recmix::UPPER, recmix::TRACER };
        mask_lo   = { 4, 5 };
        mask_up   = { 7, 8 };
        inj_count = { 4, 5, 7, 8 };
      }
      for (auto s = 0u; s < 9; ++s) {
        const auto& spc       = sp[s];
        const bool  is_lepton = (s % 3 != 2);
        const float exp_ch    = is_lepton ? ((s % 3 == 0) ? -1.0f : 1.0f) : 1.0f;
        raise::ErrorIf(std::abs(spc.charge() - exp_ch) > 1e-6f,
                       fmt::format("rec_mix: species %d must have charge %g", s + 1, exp_ch),
                       HERE);
        raise::ErrorIf(not allow_any_pusher and not(spc.pusher() & ParticlePusher::VAY),
                       fmt::format("rec_mix: species %d uses the %s pusher; the continuity "
                                   "numerics require Vay (D3). Set pusher = \"Vay\" (or "
                                   "setup.allow_any_pusher = true for tests)",
                                   s + 1,
                                   ParticlePusher::to_string(spc.pusher()).c_str()),
                       HERE);
        if (is_lepton) {
          raise::ErrorIf(std::abs(spc.mass() - 1.0f) > 1e-6f,
                         fmt::format("rec_mix: species %d must have mass 1", s + 1),
                         HERE);
          raise::ErrorIf(not spc.deposit(),
                         fmt::format("rec_mix: lepton species %d must deposit", s + 1),
                         HERE);
        } else if (active) {
          raise::ErrorIf(std::abs(spc.mass() / mu - 1.0) > 1e-5,
                         fmt::format("rec_mix: ion species %d must have mass mu = %g",
                                     s + 1,
                                     mu),
                         HERE);
          raise::ErrorIf(not spc.deposit(),
                         fmt::format("rec_mix: active ion species %d must deposit", s + 1),
                         HERE);
        } else {
          raise::ErrorIf(spc.deposit(),
                         fmt::format("rec_mix: tracer species %d must have deposit = false",
                                     s + 1),
                         HERE);
          test_masses.push_back(spc.mass());
        }
#if defined(MPI_ENABLED)
        const int ntrk = spc.use_tracking() ? 2 : 0;
#else
        const int ntrk = spc.use_tracking() ? 1 : 0;
#endif
        raise::ErrorIf(static_cast<int>(spc.npld_i()) - ntrk < 1,
                       fmt::format("rec_mix: species %d needs n_payloads_int >= 1 "
                                   "(control-volume flag)",
                                   s + 1),
                       HERE);
        raise::ErrorIf(cv_slot_cfg >= static_cast<int>(spc.npld_i()) - ntrk,
                       fmt::format("rec_mix: cv_payload_slot %d out of range for species %d",
                                   cv_slot_cfg,
                                   s + 1),
                       HERE);
      }
      if (not active) {
        test_temperatures = getRealVec("test_temperatures");
        if (test_temperatures.empty()) {
          test_temperatures.assign(3, theta_up);
        }
        raise::ErrorIf(test_temperatures.size() != 3,
                       "rec_mix: setup.test_temperatures needs 3 entries",
                       HERE);
        const auto tm = getRealVec("test_masses");
        if (not tm.empty()) {
          raise::ErrorIf(tm.size() != 3, "rec_mix: setup.test_masses needs 3 entries", HERE);
          for (auto k = 0u; k < 3; ++k) {
            raise::ErrorIf(std::abs(tm[k] / test_masses[k] - 1.0) > 1e-5,
                           "rec_mix: setup.test_masses disagree with the species masses",
                           HERE);
          }
        }
      }

      host_seed = params.template get<bool>("simulation.seed_set")
                    ? params.template get<std::uint64_t>("simulation.seed")
                    : constant::RandomSeed;

      cv_in_E.assign(9, 0.0);
      cv_out_E.assign(9, 0.0);
      cv_in_N.assign(9, 0.0);
      cv_out_N.assign(9, 0.0);

      usr_dir = std::filesystem::path(params.template get<std::string>("simulation.name")) /
                "usr";
      // provenance of every user output (Codex IC3): a 52-bit run id (exact in a
      // double) per process start, written into every file and record, and a
      // segment manifest; a file written by segment k at label s is valid iff
      // s <= start_label of every later segment
      start_label = params.template get<bool>("checkpoint.is_resuming")
                      ? static_cast<std::uint64_t>(
                          params.template get<timestep_t>("checkpoint.start_step"))
                      : 0;
      {
        std::uint64_t r = static_cast<std::uint64_t>(
          std::chrono::system_clock::now().time_since_epoch().count());
        r ^= static_cast<std::uint64_t>(std::random_device {}()) << 21;
        run_id = recmix::splitmix64(r) & ((1ULL << 52) - 1ULL);
        if (run_id == 0) {
          run_id = 1;
        }
#if defined(MPI_ENABLED)
        MPI_Bcast(&run_id, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
#endif
      }
      if (recmix::mpi_rank() == 0) {
        std::filesystem::create_directories(usr_dir);
        WriteMeta();
        WriteSegment();
      }
      Report();
    }

    /* ------------------------------------------------------------------ */
    void Report() const {
      std::ostringstream o;
      o << "layout = " << layout << ", sheet_init = " << sheet_init << ", geometry = "
        << geometry << ", field_test = " << field_test << "\n"
        << "  P13-D39 (" << norm_mode << (norm_label.empty() ? "" : ", " + norm_label)
        << "): sigma_hot = " << sigma_hot << ", T = " << theta_up << ", d_up = " << dup_cells
        << " cells, q = " << q << ", mu = " << mu << "\n"
        << "  D = rho/(n0 me) = " << Dmass << ", D + 5T/2 = " << Hfac << ", F = " << Fskin
        << "; nominal sigma_input = " << sigma_input_nom << ", c_omp = " << c_omp_nom << "\n"
        << "  sigma0 = sigma_input = " << sigma0 << ", skindepth0 = c_omp = " << skindepth0
        << ", larmor0 = " << larmor0 << "\n"
        << "  ppc0 = " << ppc0 << ", dt = " << dt << "\n"
        << "  sheet: a = " << a_cells << ", eta = " << eta << ", beta_d = " << beta_d
        << ", T_cs = " << T_cs << " me c^2 (" << sheet_init << ")\n"
        << "  N1 x N2 = " << N1 << " x " << N2 << ", t_lc = " << N1
        << ", t_open = " << t_open << "\n"
        << "  padding = " << padding << ", nfilter = " << nfilter
        << ", match ds (x1, x2) = (" << w_out << ", " << w_in << ")\n"
        << "  CV = [" << cv1a << ", " << cv1b << ") x [" << cv2a << ", " << cv2b << ")\n";
      if (not active) {
        o << "  tracer masses = " << test_masses[0] << ", " << test_masses[1] << ", "
          << test_masses[2] << "\n";
      }
      info(o.str());
    }

    void WriteSegment() const {
      std::ofstream f(usr_dir / "segments.jsonl", std::ios::app);
      const auto    now = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
      f << "{\"run_id\": " << run_id << ", \"start_label\": " << start_label
        << ", \"resuming\": "
        << (params.template get<bool>("checkpoint.is_resuming") ? "true" : "false")
        << ", \"unix_time\": " << now << ", \"simulation_name\": \""
        << params.template get<std::string>("simulation.name") << "\""
        << ", \"checkpoint_read_path\": \""
        << params.template get<std::string>("checkpoint.read_path") << "\""
#if defined(ENTITY_GIT_HASH)
        << ", \"entity_git_hash\": \"" << ENTITY_GIT_HASH << "\""
#endif
        << ", \"format_version\": 3}\n";
    }

    // Entity's own field files carry no writer id: record (label, run id) when
    // a field output containing the custom fields is written
    void NoteFieldOutput(timestep_t finished_step) {
      if (recmix::mpi_rank() != 0 or finished_step == last_field_noted) {
        return;
      }
      last_field_noted = finished_step;
      std::ofstream f(usr_dir / "fields_manifest.jsonl", std::ios::app);
      f << "{\"label\": " << finished_step + 1 << ", \"run_id\": " << run_id << "}\n";
    }

    void WriteMeta() const {
      const auto& sp = metadomain.species_params();
      std::ofstream f(usr_dir / "recmix.meta.json");
      f.precision(12);
      f << "{\n";
      f << "  \"format_version\": 4,\n"; // 4: P13-D39 inputs.sigma_hot, normalization, geometry
      f << "  \"run_id\": " << run_id << ",\n";
#if defined(ENTITY_GIT_HASH)
      f << "  \"entity_git_hash\": \"" << ENTITY_GIT_HASH << "\",\n";
#endif
      f << "  \"layout\": \"" << layout << "\",\n";
      f << "  \"sheet_init\": \"" << sheet_init << "\",\n";
      f << "  \"inputs\": {\"sigma_hot\": " << sigma_hot << ", \"q\": " << q << ", \"mu\": " << mu
        << ", \"dup_cells\": " << dup_cells << ", \"theta_up\": " << theta_up
        << ", \"cs_overdensity\": " << eta << ", \"cs_width\": " << a_cells
        << ", \"dip_width\": " << dip_width << ", \"b_guide\": " << b_guide
        << ", \"t_open\": " << t_open << ", \"injector_padding\": " << padding
        << ", \"ds_abs\": " << ds_abs << ", \"mixing_threshold\": " << thr
        << ", \"usr_interval\": " << usr_interval << ", \"line_interval\": " << line_interval
        << ", \"cv_interval\": " << cv_interval << ", \"cv_margin\": " << cv_margin
        << ", \"bulk_window\": " << bulk_window << ", \"inj_avg_cells\": " << inj_avg_cells
        << "},\n";
      f << "  \"derived\": {\"Dmass\": " << Dmass << ", \"sigma0\": " << sigma0
        << ", \"skindepth0\": " << skindepth0 << ", \"larmor0\": " << larmor0
        << ", \"ppc0\": " << ppc0 << ", \"dt\": " << dt << ", \"beta_d\": " << beta_d
        << ", \"T_cs\": " << T_cs << ", \"N1\": " << N1 << ", \"N2\": " << N2
        << ", \"x0\": " << x0 << ", \"y0\": " << y0 << ", \"nfilter\": " << nfilter
        << ", \"match_ds_x1\": " << w_out << ", \"match_ds_x2\": " << w_in
        << ", \"field_energy_per_cell_factor\": " << 0.5 * sigma0 * ppc0
        << ", \"poynting_factor\": " << sigma0 * ppc0 << "},\n";
      f << "  \"species\": [\n";
      const char* rnames[4] = { "sheet", "lower", "upper", "tracer" };
      for (auto s = 0u; s < sp.size(); ++s) {
        f << "    {\"index\": " << s + 1 << ", \"label\": \"" << sp[s].label()
          << "\", \"mass\": " << sp[s].mass() << ", \"charge\": " << sp[s].charge()
          << ", \"deposit\": " << (sp[s].deposit() ? "true" : "false")
          << ", \"pusher\": \"" << ParticlePusher::to_string(sp[s].pusher()) << "\""
          << ", \"maxnpart\": " << sp[s].maxnpart()
          << ", \"role\": \"" << rnames[role[s]] << "\"}" << (s + 1 < sp.size() ? "," : "")
          << "\n";
      }
      f << "  ],\n";
      f << "  \"mask\": {\"definition\": \"M = 2 min(D_lo, D_up)/(D_lo + D_up), D = "
           "particle count per cell of the side species, smoothing 0; mask = M > "
           "threshold\", \"lower_species\": [";
      for (auto k = 0u; k < mask_lo.size(); ++k) {
        f << mask_lo[k] << (k + 1 < mask_lo.size() ? ", " : "");
      }
      f << "], \"upper_species\": [";
      for (auto k = 0u; k < mask_up.size(); ++k) {
        f << mask_up[k] << (k + 1 < mask_up.size() ? ", " : "");
      }
      f << "], \"threshold\": " << thr << "},\n";
      f << "  \"spectra\": {\"quantity\": \"gamma - 1 (own mass)\", \"log\": true, "
           "\"min\": "
        << spec_min << ", \"max\": " << spec_max << ", \"num\": " << spec_num
        << ", \"binning\": \"idx = ceil((ln(E+1e-8) - ln(min)) * num / (ln(max) - "
           "ln(min))), clamped; centres exp(ln(min) + (i - 0.5) dln)\"},\n";
      f << "  \"moments\": [";
      const auto names = recmix::mom::names();
      for (auto k = 0u; k < names.size(); ++k) {
        f << "\"" << names[k] << "\"" << (k + 1 < names.size() ? ", " : "");
      }
      f << "],\n";
      f << "  \"bulk\": {\"Kb_own\": \"sum over masked cells of N_s m_s (gamma_b,s - "
           "1), beta_b,s = per-species mean 3-velocity in a (2w+1)^2 top-hat box\", "
           "\"Kb_all\": \"same with the mass-weighted mean 3-velocity of all "
           "depositing species (Tristan velx)\", \"window\": "
        << bulk_window << ", \"nmin\": " << bulk_nmin
        << ", \"Kb_own_raw\": \"uncorrected |b|^2 (equals the stock V_s estimate)\""
        << ", \"Nlowocc\": \"masked particles in own-species boxes with fewer than nmin particles\""
        << ", \"Kb_own_w1\": \"Kb_own with window 1 (window sensitivity)\""
        << ", \"Klowocc\": \"masked kinetic energy of the particles in under-occupied boxes\""
        << ", \"Kblowocc\": \"Kb_own assigned to those particles\""
        << ", \"Kb_own_half\": \"Kb_own with box means from the particles with dx1 < 1/2, "
           "weighted by the full cell counts (position-subsampling sensitivity; the in-cell "
           "position can correlate with velocity, so not a pure finite-sample test)\""
        << ", \"Klowocc_Kblowocc\": \"|Klowocc - Kblowocc| <= max(Klowocc, Kblowocc) bounds "
           "the recorded low-occupancy contribution, not its error against a physical random "
           "energy (box velocity assigned to the central-cell particles)\"},\n";
      f << "  \"cv\": {\"x1\": [" << cv1a << ", " << cv1b << "], \"x2\": [" << cv2a << ", "
        << cv2b << "], \"record_doubles\": " << 4 + 6 * 9 + 3
        << ", \"record_layout\": \"step, time, segment_start, run_id, 9 x (Ncv, Icv, inE, outE, "
           "inN, outN), U_EM, int Poynting out dt, int field-side current work dt\""
        << ", \"payload_int_slot\": " << cv_slot_cfg << "},\n";
      f << "  \"midline\": {\"rows_first\": " << static_cast<int>(yline) - line_halfwidth
        << ", \"nrows\": " << 2 * line_halfwidth
        << ", \"components\": [\"E1\", \"E2\", \"E3\", \"B1\", \"B2\", \"B3\"], "
           "\"location\": \"cell centres (same interpolation as the field output)\", "
           "\"node_line\": {\"j\": "
        << static_cast<int>(yline) << ", \"E3\": \"nodes (i, j0)\", \"B2\": \"(i+1/2, j0)\"}},\n";
      {
        // P13-D39 normalization and its provenance (tech spec III.1)
        const double T   = theta_up;
        const double th  = 1.0 + 1.5 * T;
        const double Fc  = active ? ((2.0 - q) / th + q / (mu + 1.5 * T)) / 2.0 : 1.0 / th;
        const double lep = active ? 1.0 + q / (mu * (2.0 - q)) : 1.0;
        const double due = skindepth0 / std::sqrt(Fskin);
        f << "  \"normalization\": {\"convention\": \"P13-D39 hot labels: sigma_input = "
             "sigma_hot (D + 5T/2), c_omp = d_up sqrt(F), F = [(2-q)/(1+3T/2) + "
             "q/(mu (1+3T/2))]/2 (testprtl: q = 0); sigma0 = sigma_input, skindepth0 = c_omp\""
          << ", \"mode\": \"" << norm_mode << "\", \"label\": \"" << norm_label << "\""
          << ", \"sigma_hot\": " << sigma_hot << ", \"T\": " << T
          << ", \"dup_cells_nominal\": " << dup_cells << ", \"D\": " << Dmass
          << ", \"enthalpy_factor\": " << Hfac << ", \"F\": " << Fskin
          << ", \"sigma_input_nominal\": " << sigma_input_nom
          << ", \"c_omp_nominal\": " << c_omp_nom << ", \"sigma_input\": " << sigma0
          << ", \"c_omp\": " << skindepth0
          << ", \"sigma_input_rel_diff\": " << sigma0 / sigma_input_nom - 1.0
          << ", \"c_omp_abs_diff\": " << skindepth0 - c_omp_nom
          << ", \"ref_sigma_rtol\": " << ref_sigma_rtol
          << ", \"ref_comp_atol\": " << ref_comp_atol
          << ", \"dup_cells_effective\": " << due
          << ", \"sigma_cold_total\": " << sigma0 / Dmass
          << ", \"sigma_e_cold_lepton\": " << (active ? 2.0 * sigma0 / (2.0 - q) : sigma0)
          << ", \"d_e_thermal\": " << due * std::sqrt(lep)
          << ", \"d_e_cold\": " << skindepth0 * std::sqrt(active ? 2.0 / (2.0 - q) : 1.0)
          << ", \"F_common_T\": " << Fc
          << ", \"c_omp_ratio_common_T_documented_not_used\": " << std::sqrt(Fc / Fskin)
          << "},\n";
      }
      f << "  \"geometry\": {\"geometry\": \"" << geometry << "\", \"field_test\": \""
        << field_test << "\", \"load_plasma\": " << (load_plasma ? "true" : "false")
        << ", \"inject\": " << (inject_on ? "true" : "false")
        << ", \"background_scale\": " << bg_field << ", \"y1\": " << y1 << ", \"y2\": " << y2
        << ", \"yline\": " << yline << ", \"cv_whole_box\": " << (cv_whole ? "true" : "false")
        << ", \"pulse_width\": " << pulse_w << ", \"pulse_offset\": " << pulse_off
        << ", \"pulse_amp\": " << pulse_amp << ", \"beta_rec_target\": " << beta_rec
        << ", \"validation_dump\": " << (vdump ? "true" : "false") << "}\n";
      f << "}\n";
    }

    /* ------------------------------------------------------------------ */
    auto MatchFieldsInX1(simtime_t) const -> recmix::BoundaryFieldsInX1<D> {
      return recmix::BoundaryFieldsInX1<D> { static_cast<real_t>(a_cells),
                                             static_cast<real_t>(x0),
                                             static_cast<real_t>(y0),
                                             static_cast<real_t>(b_guide),
                                             static_cast<real_t>(beta_rec),
                                             static_cast<real_t>(bg_field) };
    }

    auto MatchFieldsInX2(simtime_t) const -> recmix::BoundaryFieldsInX2<D> {
      return recmix::BoundaryFieldsInX2<D> { static_cast<real_t>(a_cells),
                                             static_cast<real_t>(y0),
                                             static_cast<real_t>(b_guide),
                                             static_cast<real_t>(bg_field) };
    }

    /* ------------------------------------------------------------------ */
    /* injection helpers                                                  */
    /* ------------------------------------------------------------------ */
    auto host_rng(const Domain<S, M>& dom, std::uint64_t step, std::uint64_t stream) const
      -> std::mt19937_64 {
      std::uint64_t z = recmix::splitmix64(host_seed ^ 0xA5A5A5A5DEADBEEFULL);
      z = recmix::splitmix64(z + static_cast<std::uint64_t>(dom.index()));
      z = recmix::splitmix64(z + step);
      z = recmix::splitmix64(z + stream);
      return std::mt19937_64 { z };
    }

    // Tristan: n < 10 -> Poisson(n), else ceiling(n)
    static auto tristan_round(double n, std::mt19937_64& rng) -> std::size_t {
      if (n <= 0.0) {
        return 0;
      }
      if (n < 10.0) {
        std::poisson_distribution<long long> pd(n);
        return static_cast<std::size_t>(pd(rng));
      }
      return static_cast<std::size_t>(std::ceil(n));
    }

    // inject ntrials co-located particles of species (sA[, sB]) in the global
    // box [X1a, X1b) x [X2a, X2b) intersected with the local domain
    auto InjectBox(Domain<S, M>&  dom,
                   int            sA,
                   int            sB,
                   double         ndens,
                   double         X1a,
                   double         X1b,
                   double         X2a,
                   double         X2b,
                   double         thA,
                   double         thB,
                   double         uA3,
                   double         uB3,
                   bool           sheet_profile,
                   std::mt19937_64& rng,
                   double         ysheet = -1.0) -> std::size_t {
      const auto off1 = static_cast<double>(dom.offset_ncells()[0]);
      const auto off2 = static_cast<double>(dom.offset_ncells()[1]);
      const auto n1   = static_cast<double>(dom.mesh.n_active(in::x1));
      const auto n2   = static_cast<double>(dom.mesh.n_active(in::x2));
      const auto x1a  = std::max(X1a, off1) - off1;
      const auto x1b  = std::min(X1b, off1 + n1) - off1;
      const auto x2a  = std::max(X2a, off2) - off2;
      const auto x2b  = std::min(X2b, off2 + n2) - off2;
      if (x1b <= x1a or x2b <= x2a or ndens <= 0.0) {
        return 0;
      }
      const auto ntrials = tristan_round(ndens * (x1b - x1a) * (x2b - x2a), rng);
      if (ntrials == 0) {
        return 0;
      }
      auto& spA = dom.species[sA - 1];
      raise::ErrorIf(spA.npart() + ntrials > spA.maxnpart(),
                     fmt::format("rec_mix: capacity guard: species %d needs %lu > "
                                 "maxnpart %lu on domain %d; increase maxnpart",
                                 sA,
                                 (unsigned long)(spA.npart() + ntrials),
                                 (unsigned long)spA.maxnpart(),
                                 dom.index()),
                     HERE);
      if (sB > 0) {
        auto& spB = dom.species[sB - 1];
        raise::ErrorIf(spB.npart() + ntrials > spB.maxnpart(),
                       fmt::format("rec_mix: capacity guard: species %d needs %lu > "
                                   "maxnpart %lu on domain %d; increase maxnpart",
                                   sB,
                                   (unsigned long)(spB.npart() + ntrials),
                                   (unsigned long)spB.maxnpart(),
                                   dom.index()),
                       HERE);
      }
      using K = recmix::BoxInject_kernel<M>;
      K kern {
        K::make_arr(spA),
        (sB > 0) ? K::make_arr(dom.species[sB - 1]) : K::make_arr(spA),
        (sB > 0),
        static_cast<real_t>(x1a),
        static_cast<real_t>(x1b),
        static_cast<real_t>(x2a),
        static_cast<real_t>(x2b),
        static_cast<int>(n1),
        static_cast<int>(n2),
        thA,
        thB,
        uA3,
        uB3,
        sheet_profile,
        static_cast<real_t>(a_cells),
        static_cast<real_t>(x0),
        static_cast<real_t>((ysheet >= 0.0) ? ysheet : y0), // sheet (and dip) centre
        static_cast<real_t>(SQR(dip_width * a_cells)),
        static_cast<real_t>(off1),
        static_cast<real_t>(off2),
        static_cast<npart_t>(dom.index()),
        dom.random_pool()
      };
      Kokkos::parallel_for("rec_mix::InjectBox", static_cast<npart_t>(ntrials), kern);
      auto n_h = Kokkos::create_mirror_view(kern.idx);
      Kokkos::deep_copy(n_h, kern.idx);
      const auto n = static_cast<npart_t>(n_h());
      spA.set_npart(spA.npart() + n);
      spA.set_counter(spA.counter() + n);
      if (sB > 0) {
        auto& spB = dom.species[sB - 1];
        spB.set_npart(spB.npart() + n);
        spB.set_counter(spB.counter() + n);
      }
      return n;
    }

    auto theta(int s) const -> double {
      return theta_up / static_cast<double>(metadomain.species_params()[s - 1].mass());
    }

    // max over ranks of npart/maxnpart per species (IC2 monitoring)
    auto FillFractions(const Domain<S, M>& dom, bool warn) const -> std::vector<double> {
      std::vector<double> fill(9, 0.0);
      for (auto s = 0u; s < 9; ++s) {
        fill[s] = static_cast<double>(dom.species[s].npart()) /
                  static_cast<double>(dom.species[s].maxnpart());
        fill[s] = recmix::all_max(fill[s]);
      }
      if (warn) {
        for (auto s = 0u; s < 9; ++s) {
          if (fill[s] > 0.8) {
            info(fmt::format("WARNING: species %d fills %.0f%% of maxnpart on some domain",
                             s + 1,
                             100.0 * fill[s]));
          }
        }
      }
      return fill;
    }

    // C1.3a: the load is exactly neutral cell by cell (co-located pairs)
    void LoadChargeCheck(const Domain<S, M>& dom) const {
      const auto n1 = static_cast<int>(dom.mesh.n_active(in::x1));
      const auto n2 = static_cast<int>(dom.mesh.n_active(in::x2));
      Kokkos::View<double**> rho("rec_mix::rho0", n1, n2);
      for (auto s = 0u; s < 9; ++s) {
        const auto& sp = dom.species[s];
        if (sp.npart() == 0 or not sp.deposit()) {
          continue;
        }
        const double q   = sp.charge();
        const auto   i1  = sp.i1;
        const auto   i2  = sp.i2;
        const auto   w   = sp.weight;
        const auto   tag = sp.tag;
        Kokkos::parallel_for(
          "rec_mix::LoadCharge",
          sp.rangeActiveParticles(),
          Lambda(prtlidx_t p) {
            if (tag(p) == ParticleTag::alive) {
              Kokkos::atomic_add(&rho(i1(p), i2(p)), q * static_cast<double>(w(p)));
            }
          });
      }
      double mx { 0.0 }, tot { 0.0 };
      Kokkos::parallel_reduce(
        "rec_mix::LoadChargeMax",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { n1, n2 }),
        Lambda(int i, int j, double& m) {
          const double a = math::abs(rho(i, j));
          m              = (a > m) ? a : m;
        },
        Kokkos::Max<double>(mx));
      Kokkos::parallel_reduce(
        "rec_mix::LoadChargeSum",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { n1, n2 }),
        Lambda(int i, int j, double& t) { t += rho(i, j); },
        tot);
      mx = recmix::all_max(mx);
      info(fmt::format("C1.3a load neutrality: max |net charge| per cell = %g e, domain "
                       "net charge = %g e (both 0 for co-located pairs)",
                       mx,
                       tot));
      raise::ErrorIf(mx > 0.0, "rec_mix: the load is not neutral cell by cell", HERE);
    }

    /* ------------------------------------------------------------------ */
    void InitPrtls(Domain<S, M>& dom) {
      if (not load_plasma) { // C1.5 vacuum pulse tests
        info("load: no plasma (field test)");
        return;
      }
      auto rng = host_rng(dom, 0xFFFFFFFFULL, 0);
      const double nup = 0.5 * ppc0;
      // upstream bands: open box, lower side [0, y0) and upper side [y0, N2);
      // closed box, lower side [0, y1) and [y2, N2), upper side [y1, y2), so
      // that each sheet separates the two sides (mixing mask)
      std::vector<std::array<double, 3>> bands; // {side, X2a, X2b}
      if (closed) {
        bands = { { 0.0, 0.0, y1 }, { 1.0, y1, y2 }, { 0.0, y2, static_cast<double>(N2) } };
      } else {
        bands = { { 0.0, 0.0, y0 }, { 1.0, y0, static_cast<double>(N2) } };
      }
      for (const auto& bd : bands) {
        const int    side = static_cast<int>(bd[0]);
        const double X2a  = bd[1];
        const double X2b  = bd[2];
        const int    se  = (side == 0) ? 4 : 7; // e-
        const int    sp  = se + 1;              // e+
        const int    si  = se + 2;              // ion
        if (active) {
          if (q < 1.0) {
            InjectBox(dom, se, sp, nup * (1.0 - q), 0.0, N1, X2a, X2b, theta(se),
                      theta(sp), 0.0, 0.0, false, rng);
          }
          if (q > 0.0) {
            InjectBox(dom, se, si, nup * q, 0.0, N1, X2a, X2b, theta(se), theta(si),
                      0.0, 0.0, false, rng);
          }
        } else {
          InjectBox(dom, se, sp, nup, 0.0, N1, X2a, X2b, theta(se), theta(sp), 0.0,
                    0.0, false, rng);
        }
      }
      // current sheet: species 1, 2 (and 3 if active); u_x3 = -sgn(Q) u_d (D29).
      // Closed box: the sheet at y2 carries the opposite current (+x3), each
      // sheet loaded within N2/4 of its centre.
      if (eta > 0.0) {
        const double ncs = nup * eta;
        const auto&  spp = metadomain.species_params();
        const auto   th  = [&](int s) {
          return T_cs / static_cast<double>(spp[s - 1].mass());
        };
        std::vector<std::array<double, 2>> sheets; // {centre, current sign}
        if (closed) {
          sheets = { { y1, 1.0 }, { y2, -1.0 } };
        } else {
          sheets = { { y0, 1.0 } };
        }
        for (const auto& sh : sheets) {
          const double yc  = sh[0];
          const double ext = closed ? 0.25 * N2 : sheet_extent * a_cells;
          const double X2a = std::max(0.0, yc - ext);
          const double X2b = std::min(static_cast<double>(N2), yc + ext);
          const auto drift = [&](int s) {
            return -sh[1] * static_cast<double>(spp[s - 1].charge() > 0 ? 1 : -1) * u_d;
          };
          if ((not active) or q < 1.0) {
            InjectBox(dom, 1, 2, ncs * (active ? (1.0 - q) : 1.0), 0.0, N1, X2a, X2b,
                      th(1), th(2), drift(1), drift(2), true, rng, yc);
          }
          if (active and q > 0.0) {
            InjectBox(dom, 1, 3, ncs * q, 0.0, N1, X2a, X2b, th(1), th(3), drift(1),
                      drift(3), true, rng, yc);
          }
        }
      }
      // passive tracers last, so that the depositing species draw the same
      // random numbers with or without tracers (test C0.4)
      if ((not active) and load_tracers) {
        for (const auto& bd : bands) {
          const double X2a = bd[1];
          const double X2b = bd[2];
          for (int k = 0; k < 3; ++k) {
            InjectBox(dom, 3 * (k + 1), -1, nup, 0.0, N1, X2a, X2b,
                      test_temperatures[k] / test_masses[k], 0.0, 0.0, 0.0, false, rng);
          }
        }
      }
      LoadChargeCheck(dom);
      const auto fill = FillFractions(dom, false);
      info(fmt::format("load: max fill npart/maxnpart per species = "
                       "%.3f %.3f %.3f | %.3f %.3f %.3f | %.3f %.3f %.3f",
                       fill[0], fill[1], fill[2], fill[3], fill[4], fill[5],
                       fill[6], fill[7], fill[8]));
    }

    /* ------------------------------------------------------------------ */
    /* per-step pieces                                                    */
    /* ------------------------------------------------------------------ */
    // number of alive particles of species `sp` with local i2 in [ja, jb)
    static auto CountInRows(const Particles<D, M::CoordType>& sp, int ja, int jb) -> double {
      if (sp.npart() == 0) {
        return 0.0;
      }
      const auto i2  = sp.i2;
      const auto tag = sp.tag;
      double     cnt { 0.0 };
      Kokkos::parallel_reduce(
        "rec_mix::CountInRows",
        sp.rangeActiveParticles(),
        Lambda(prtlidx_t p, double& acc) {
          if (tag(p) == ParticleTag::alive and i2(p) >= ja and i2(p) < jb) {
            acc += 1.0;
          }
        },
        cnt);
      return cnt;
    }

    void Inject(Domain<S, M>& dom, timestep_t step) {
      const int off2 = static_cast<int>(dom.offset_ncells()[1]);
      const int n2   = static_cast<int>(dom.mesh.n_active(in::x2));
      const int n1   = static_cast<int>(dom.mesh.n_active(in::x1));
      const double off1 = static_cast<double>(dom.offset_ncells()[0]);
      for (int side = 0; side < 2; ++side) {
        // injection cell (global) and averaging rows (global), Tristan 280-345
        const int jinj = (side == 0) ? padding : (N2 - padding - 1);
        if (jinj < off2 or jinj >= off2 + n2) {
          continue;
        }
        int ga = (side == 0) ? jinj : (jinj - inj_avg_cells + 1);
        int gb = (side == 0) ? (jinj + inj_avg_cells) : (jinj + 1);
        ga     = std::max(ga, off2);
        gb     = std::min(gb, off2 + n2);
        double cnt { 0.0 };
        for (const auto s : inj_count) {
          cnt += CountInRows(dom.species[s - 1], ga - off2, gb - off2);
        }
        const double dens = cnt / (static_cast<double>(gb - ga) * n1);
        if (not(ppc0 > dens)) {
          continue;
        }
        auto rng = host_rng(dom, static_cast<std::uint64_t>(step), 1 + side);
        const double X1a = off1, X1b = off1 + n1;
        const double X2a = jinj, X2b = jinj + 1.0;
        const double def = 0.5 * (ppc0 - dens);
        const int    se  = (side == 0) ? 4 : 7;
        if (active) {
          if (q < 1.0) {
            InjectBox(dom, se, se + 1, def * (1.0 - q), X1a, X1b, X2a, X2b, theta(se),
                      theta(se + 1), 0.0, 0.0, false, rng);
          }
          if (q > 0.0) {
            InjectBox(dom, se, se + 2, def * q, X1a, X1b, X2a, X2b, theta(se),
                      theta(se + 2), 0.0, 0.0, false, rng);
          }
        } else {
          InjectBox(dom, se, se + 1, def, X1a, X1b, X2a, X2b, theta(se), theta(se + 1),
                    0.0, 0.0, false, rng);
          for (int k = 0; k < 3; ++k) {
            InjectBox(dom, 3 * (k + 1), -1, def, X1a, X1b, X2a, X2b,
                      test_temperatures[k] / test_masses[k], 0.0, 0.0, 0.0, false, rng);
          }
        }
      }
    }

    void Padding(Domain<S, M>& dom) {
      const real_t off2 = static_cast<real_t>(dom.offset_ncells()[1]);
      const real_t pad  = static_cast<real_t>(padding);
      const real_t padf = static_cast<real_t>(padding - nfilter);
      const real_t n2g  = static_cast<real_t>(N2);
      const real_t u0   = static_cast<real_t>(0.1);
      for (auto& sp : dom.species) {
        if (sp.npart() == 0) {
          continue;
        }
        auto i2  = sp.i2;
        auto dx2 = sp.dx2;
        auto ux1 = sp.ux1;
        auto ux2 = sp.ux2;
        auto ux3 = sp.ux3;
        auto tag = sp.tag;
        Kokkos::parallel_for(
          "rec_mix::Padding",
          sp.rangeActiveParticles(),
          Lambda(prtlidx_t p) {
            if (tag(p) != ParticleTag::alive) {
              return;
            }
            const real_t X2 = static_cast<real_t>(i2(p)) + off2 + static_cast<real_t>(dx2(p));
            if (X2 < pad) {
              ux1(p) = ZERO;
              ux2(p) = -u0;
              ux3(p) = ZERO;
            } else if (X2 >= n2g - pad) {
              ux1(p) = ZERO;
              ux2(p) = u0;
              ux3(p) = ZERO;
            }
            if ((X2 < padf) or (X2 >= n2g - padf)) {
              tag(p) = ParticleTag::dead;
            }
          });
      }
    }

    /* --- control volume ------------------------------------------------ */
    // returns {in_E, out_E, in_N, out_N, inv_E, inv_N}; silent: flags only
    auto CVSpecies(Particles<D, M::CoordType>& sp,
                   const Domain<S, M>&         dom,
                   bool                        silent) const -> std::vector<double> {
      std::vector<double> res(6, 0.0);
      if (sp.npart() == 0) {
        return res;
      }
      // membership from the integer cell (the CV faces are cell faces): exact,
      // unlike float(i) + dx, which rounds up to the next face when 1 - dx is
      // below half an ulp of X (~3.8e-6 at X ~ 100; the A0-2 ambiguity, seen
      // as a spurious exit in the closed box)
      const int    off1 = static_cast<int>(dom.offset_ncells()[0]);
      const int    off2 = static_cast<int>(dom.offset_ncells()[1]);
      const int    a1 = cv1a, b1 = cv1b, a2 = cv2a, b2 = cv2b;
      const double mass = sp.mass();
#if defined(MPI_ENABLED)
      const int ntrk = sp.use_tracking() ? 2 : 0;
#else
      const int ntrk = sp.use_tracking() ? 1 : 0;
#endif
      const int slot = (cv_slot_cfg >= 0) ? ntrk + cv_slot_cfg
                                          : static_cast<int>(sp.npld_i()) - 1;
      auto i1 = sp.i1;
      auto i2 = sp.i2;
      auto ux1 = sp.ux1;
      auto ux2 = sp.ux2;
      auto ux3 = sp.ux3;
      auto w   = sp.weight;
      auto tag = sp.tag;
      auto pld = sp.pld_i;
      double r0 { 0 }, r1 { 0 }, r2 { 0 }, r3 { 0 }, r4 { 0 }, r5 { 0 };
      Kokkos::parallel_reduce(
        "rec_mix::CV",
        sp.rangeActiveParticles(),
        Lambda(prtlidx_t p, double& inE, double& outE, double& inN, double& outN,
               double& invE, double& invN) {
          if (tag(p) != ParticleTag::alive) {
            return;
          }
          const int    I1 = i1(p) + off1, I2 = i2(p) + off2;
          const bool   in_now = (I1 >= a1) and (I1 < b1) and (I2 >= a2) and (I2 < b2);
          const double u1 = ux1(p), u2 = ux2(p), u3 = ux3(p);
          const double usq = u1 * u1 + u2 * u2 + u3 * u3;
          const double e   = w(p) * mass * usq / (math::sqrt(1.0 + usq) + 1.0);
          const bool   was = (pld(p, slot) != 0);
          if (not silent) {
            if (in_now and not was) {
              inE += e;
              inN += w(p);
            } else if (was and not in_now) {
              outE += e;
              outN += w(p);
            }
          }
          if (in_now) {
            invE += e;
            invN += w(p);
          }
          pld(p, slot) = in_now ? 1 : 0;
        },
        r0, r1, r2, r3, r4, r5);
      res = { r0, r1, r2, r3, r4, r5 };
      return res;
    }

    // field energy in the CV (raw sum of squares of the Yee components owned
    // by the CV cells) and the outward Poynting integrand (raw e x b . n dl)
    // returns {U_raw, F_raw, WJ_raw}: sum of squares of the Yee components in
    // the CV (components on the CV boundary weighted 1/2, corners 1/4), the
    // outward Poynting integrand (e x b . n dl), and the energy given by the
    // current in the last Ampere update, sum ebar . dJ with dJ = coeff ppc0 J
    // (J as stored after Ampere) and ebar = E_new - (dCurl + dJ)/2 the
    // time-centred field of that update
    auto CVFields(const Domain<S, M>& dom) const -> std::vector<double> {
      const int off1 = static_cast<int>(dom.offset_ncells()[0]);
      const int off2 = static_cast<int>(dom.offset_ncells()[1]);
      const int n1   = static_cast<int>(dom.mesh.n_active(in::x1));
      const int n2   = static_cast<int>(dom.mesh.n_active(in::x2));
      const auto EM  = dom.fields.em;
      const int  G   = static_cast<int>(N_GHOSTS);
      // local index ranges of the CV
      const int la1 = std::max(cv1a, off1) - off1, lb1 = std::min(cv1b, off1 + n1) - off1;
      const int la2 = std::max(cv2a, off2) - off2, lb2 = std::min(cv2b, off2 + n2) - off2;
      double    U { 0.0 }, F { 0.0 }, WJ { 0.0 };
      // cells (inclusive of the far face nodes) owned by this domain; whole
      // periodic box: every owned component once, weight 1, no surface
      const bool whole = cv_whole;
      const int ca1 = whole ? 0 : std::max(cv1a, off1) - off1;
      const int cb1 = whole ? n1 : std::min(cv1b + 1, off1 + n1) - off1;
      const int ca2 = whole ? 0 : std::max(cv2a, off2) - off2;
      const int cb2 = whole ? n2 : std::min(cv2b + 1, off2 + n2) - off2;
      if (cb1 > ca1 and cb2 > ca2) {
        const int A1 = whole ? -2 : cv1a - off1, B1 = whole ? n1 + 2 : cv1b - off1;
        const int A2 = whole ? -2 : cv2a - off2, B2 = whole ? n2 + 2 : cv2b - off2;
        const auto   CUR  = dom.fields.cur;
        const double dtd  = static_cast<double>(dt);
        const double cJ   = -static_cast<double>(dt) *
                          static_cast<double>(params.template get<real_t>("scales.q0")) /
                          (static_cast<double>(params.template get<real_t>("scales.B0")) *
                           static_cast<double>(params.template get<real_t>("scales.V0"))) *
                          ppc0;
        Kokkos::parallel_reduce(
          "rec_mix::CVEnergy",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ ca1, ca2 }, { cb1, cb2 }),
          Lambda(int i, int j, double& acc, double& wj) {
            const int I = i + G, J = j + G;
            // weights along each axis: node-type location (integer index) or
            // half-type location (index + 1/2)
            const double wn1 = (i == A1 or i == B1) ? 0.5 : ((i > A1 and i < B1) ? 1.0 : 0.0);
            const double wh1 = (i >= A1 and i < B1) ? 1.0 : 0.0;
            const double wn2 = (j == A2 or j == B2) ? 0.5 : ((j > A2 and j < B2) ? 1.0 : 0.0);
            const double wh2 = (j >= A2 and j < B2) ? 1.0 : 0.0;
            // E1 (i+1/2, j), E2 (i, j+1/2), E3 (i, j), B1 (i, j+1/2), B2 (i+1/2, j), B3 (i+1/2, j+1/2)
            const double w[6] = { wh1 * wn2, wn1 * wh2, wn1 * wn2,
                                  wn1 * wh2, wh1 * wn2, wh1 * wh2 };
            double       sum { 0.0 };
            for (int c = 0; c < 6; ++c) {
              const double v = EM(I, J, c);
              sum += w[c] * v * v;
            }
            acc += sum;
            // curl part of the last Ampere update (kernels/ampere_mink.hpp, 2D,
            // unit cells: coeff1 = coeff2 = dt); B is unchanged since then
            const double dC[3] = {
              dtd * (EM(I, J, em::bx3) - EM(I, J - 1, em::bx3)),
              dtd * (EM(I - 1, J, em::bx3) - EM(I, J, em::bx3)),
              dtd * (EM(I, J - 1, em::bx1) - EM(I, J, em::bx1) + EM(I, J, em::bx2) -
                     EM(I - 1, J, em::bx2))
            };
            for (int c = 0; c < 3; ++c) {
              const double dJ = cJ * CUR(I, J, c);
              // time-centred E of the Ampere update: (E_old + E_new)/2
              const double eb = EM(I, J, c) - 0.5 * (dC[c] + dJ);
              wj += w[c] * eb * dJ;
            }
          },
          U,
          WJ);
      }
      if (whole) {
        return { U, F, WJ };
      }
      // x1 faces (node lines I = cv1a, cv1b), segments j in [cv2a, cv2b)
      for (int side = 0; side < 2; ++side) {
        const int Ig = (side == 0) ? cv1a : cv1b;
        if (Ig < off1 or Ig >= off1 + n1 or lb2 <= la2) {
          continue;
        }
        const double sgn = (side == 0) ? -1.0 : 1.0;
        const int    I   = Ig - off1 + G;
        double       f { 0.0 };
        Kokkos::parallel_reduce(
          "rec_mix::CVFluxX1",
          Kokkos::RangePolicy<>(la2, lb2),
          Lambda(int j, double& acc) {
            const int    J  = j + G;
            const double e2 = EM(I, J, em::ex2);
            const double b3 = 0.5 * (EM(I - 1, J, em::bx3) + EM(I, J, em::bx3));
            const double e3 = 0.5 * (EM(I, J, em::ex3) + EM(I, J + 1, em::ex3));
            const double b2 = 0.25 * (EM(I - 1, J, em::bx2) + EM(I, J, em::bx2) +
                                      EM(I - 1, J + 1, em::bx2) + EM(I, J + 1, em::bx2));
            acc += e2 * b3 - e3 * b2;
          },
          f);
        F += sgn * f;
      }
      // x2 faces (node lines J = cv2a, cv2b), segments i in [cv1a, cv1b)
      for (int side = 0; side < 2; ++side) {
        const int Jg = (side == 0) ? cv2a : cv2b;
        if (Jg < off2 or Jg >= off2 + n2 or lb1 <= la1) {
          continue;
        }
        const double sgn = (side == 0) ? -1.0 : 1.0;
        const int    J   = Jg - off2 + G;
        double       f { 0.0 };
        Kokkos::parallel_reduce(
          "rec_mix::CVFluxX2",
          Kokkos::RangePolicy<>(la1, lb1),
          Lambda(int i, double& acc) {
            const int    I  = i + G;
            const double e1 = EM(I, J, em::ex1);
            const double b3 = 0.5 * (EM(I, J - 1, em::bx3) + EM(I, J, em::bx3));
            const double e3 = 0.5 * (EM(I, J, em::ex3) + EM(I + 1, J, em::ex3));
            const double b1 = 0.25 * (EM(I, J - 1, em::bx1) + EM(I + 1, J - 1, em::bx1) +
                                      EM(I, J, em::bx1) + EM(I + 1, J, em::bx1));
            acc += e3 * b1 - e1 * b3;
          },
          f);
        F += sgn * f;
      }
      return { U, F, WJ };
    }

    void CVStep(Domain<S, M>& dom, timestep_t step, simtime_t time, bool write_now) {
      const bool silent = not cv_started;
      std::vector<double> inv(18, 0.0);
      for (auto s = 0u; s < 9; ++s) {
        const auto r = CVSpecies(dom.species[s], dom, silent);
        cv_in_E[s]  += r[0];
        cv_out_E[s] += r[1];
        cv_in_N[s]  += r[2];
        cv_out_N[s] += r[3];
        inv[2 * s]     = r[4];
        inv[2 * s + 1] = r[5];
      }
      const auto fl = CVFields(dom);
      const double U = fl[0], F = fl[1], WJ = fl[2];
      if (not silent) {
        // rectangle rule over the steps since the previous CV update
        const double nst = static_cast<double>(step - cv_last_step);
        cv_flux_int += F * dt * nst;
        cv_jwork_int += -WJ * nst;
      }
      cv_last_step = step;
      if (silent or write_now) {
        // record: step, time, segment_start, 0; per species (Ncv, Icv, inE, outE,
        // inN, outN); U_EM, int Poynting dt, field-side work of the (filtered)
        // current on the particles in the CV  [energies in me c^2 x weight]
        std::vector<double> rec;
        rec.push_back(static_cast<double>(step + 1));
        rec.push_back(static_cast<double>(time) + dt);
        rec.push_back(silent ? 1.0 : 0.0);
        rec.push_back(static_cast<double>(run_id)); // < 2^52, exact
        for (auto s = 0u; s < 9; ++s) {
          rec.push_back(inv[2 * s + 1]);
          rec.push_back(inv[2 * s]);
          rec.push_back(cv_in_E[s]);
          rec.push_back(cv_out_E[s]);
          rec.push_back(cv_in_N[s]);
          rec.push_back(cv_out_N[s]);
        }
        rec.push_back(0.5 * sigma0 * ppc0 * U);
        rec.push_back(sigma0 * ppc0 * cv_flux_int);
        rec.push_back(sigma0 * ppc0 * cv_jwork_int);
        // per-rank partial sums -> root; header entries are identical on all
        // ranks, so reduce only the payload
        std::vector<double> payload(rec.begin() + 4, rec.end());
        recmix::reduce_to_root(payload);
        if (recmix::mpi_rank() == 0) {
          std::copy(payload.begin(), payload.end(), rec.begin() + 4);
          std::ofstream f(usr_dir / "cvbudget.bin", std::ios::binary | std::ios::app);
          f.write(reinterpret_cast<const char*>(rec.data()), rec.size() * sizeof(double));
        }
        for (auto s = 0u; s < 9; ++s) {
          cv_in_E[s] = cv_out_E[s] = cv_in_N[s] = cv_out_N[s] = 0.0;
        }
        cv_flux_int  = 0.0;
        cv_jwork_int = 0.0;
      }
      cv_started = true;
    }

    /* --- mixing / mask ---------------------------------------------------- */
    void ComputeMixing(const Domain<S, M>& dom) {
      const auto n1 = dom.mesh.n_active(in::x1);
      const auto n2 = dom.mesh.n_active(in::x2);
      if (m_mix.extent(0) != n1 or m_mix.extent(1) != n2) {
        m_mix = Kokkos::View<real_t**>("rec_mix::mix", n1, n2);
        m_cnt = Kokkos::View<real_t***>("rec_mix::cnt", n1, n2, 2);
      }
      Kokkos::deep_copy(m_cnt, ZERO);
      auto cnt = m_cnt;
      for (int side = 0; side < 2; ++side) {
        for (const auto s : (side == 0) ? mask_lo : mask_up) {
          const auto& sp = dom.species[s - 1];
          if (sp.npart() == 0) {
            continue;
          }
          const auto i1  = sp.i1;
          const auto i2  = sp.i2;
          const auto tag = sp.tag;
          Kokkos::parallel_for(
            "rec_mix::MixCount",
            sp.rangeActiveParticles(),
            Lambda(prtlidx_t p) {
              if (tag(p) == ParticleTag::alive) {
                Kokkos::atomic_add(&cnt(i1(p), i2(p), side), ONE);
              }
            });
        }
      }
      auto mix = m_mix;
      Kokkos::parallel_for(
        "rec_mix::Mix",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { (int)n1, (int)n2 }),
        Lambda(int i, int j) {
          const double L = cnt(i, j, 0), U = cnt(i, j, 1);
          mix(i, j)      = (L + U > 0.0)
                             ? static_cast<real_t>(2.0 * (L < U ? L : U) / (L + U))
                             : ZERO;
        });
    }

    /* --- bulk-velocity deposits into bckp ----------------------------------- */
    // comps [c0, c0+4): sum w*wt, sum w*wt*beta_1..3 for the given species
    // c0..c0+3: sum w, sum w beta; cq: sum w |beta|^2; ck: sum w (gamma - 1);
    // half: only particles with dx1 < 1/2 (a subsample by in-cell position; it is
    // not guaranteed velocity-independent, since dx1 and velocity can correlate in
    // the evolved plasma: Codex A0-3b)
    void DepositBulk(Domain<S, M>& dom, const std::vector<int>& specs, bool mass_weighted, int c0,
                     int cq = -1, int ck = -1, bool half = false) {
      auto buf = dom.fields.bckp;
      const int G = static_cast<int>(N_GHOSTS);
      for (const auto s : specs) {
        const auto& sp = dom.species[s - 1];
        if (sp.npart() == 0) {
          continue;
        }
        const real_t wt  = mass_weighted ? static_cast<real_t>(sp.mass()) : ONE;
        const auto   i1  = sp.i1;
        const auto   i2  = sp.i2;
        const auto   ux1 = sp.ux1;
        const auto   ux2 = sp.ux2;
        const auto   ux3 = sp.ux3;
        const auto   w   = sp.weight;
        const auto   tag = sp.tag;
        const auto   dx1 = sp.dx1;
        Kokkos::parallel_for(
          "rec_mix::DepositBulk",
          sp.rangeActiveParticles(),
          Lambda(prtlidx_t p) {
            if (tag(p) != ParticleTag::alive) {
              return;
            }
            if (half and not(dx1(p) < HALF)) {
              return;
            }
            const real_t usq = SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p));
            const real_t ig  = ONE / math::sqrt(ONE + usq);
            const real_t c  = w(p) * wt;
            const int    I  = i1(p) + G, J = i2(p) + G;
            Kokkos::atomic_add(&buf(I, J, c0), c);
            Kokkos::atomic_add(&buf(I, J, c0 + 1), c * ux1(p) * ig);
            Kokkos::atomic_add(&buf(I, J, c0 + 2), c * ux2(p) * ig);
            Kokkos::atomic_add(&buf(I, J, c0 + 3), c * ux3(p) * ig);
            if (cq >= 0) {
              Kokkos::atomic_add(&buf(I, J, cq), c * usq * SQR(ig));
            }
            if (ck >= 0) {
              // gamma - 1 = u^2/(1 + gamma), without cancellation
              Kokkos::atomic_add(&buf(I, J, ck), w(p) * usq / (ONE + math::sqrt(ONE + usq)));
            }
          });
      }
    }

    void DepositCount(Domain<S, M>& dom, int s, int c) {
      auto buf = dom.fields.bckp;
      const int   G  = static_cast<int>(N_GHOSTS);
      const auto& sp = dom.species[s - 1];
      Kokkos::deep_copy(Kokkos::subview(buf, Kokkos::ALL, Kokkos::ALL, c), ZERO);
      if (sp.npart() == 0) {
        return;
      }
      const auto i1  = sp.i1;
      const auto i2  = sp.i2;
      const auto w   = sp.weight;
      const auto tag = sp.tag;
      Kokkos::parallel_for(
        "rec_mix::DepositCount",
        sp.rangeActiveParticles(),
        Lambda(prtlidx_t p) {
          if (tag(p) == ParticleTag::alive) {
            Kokkos::atomic_add(&buf(i1(p) + G, i2(p) + G, c), w(p));
          }
        });
    }

    auto BulkSum(Domain<S, M>& dom, int cn, int cb, int cs, int cq, int win, int ck = -1) const
      -> std::vector<double> {
      const auto n1 = static_cast<int>(dom.mesh.n_active(in::x1));
      const auto n2 = static_cast<int>(dom.mesh.n_active(in::x2));
      recmix::BulkEnergy_kernel<D> kern;
      kern.buf  = dom.fields.bckp;
      kern.mix  = m_mix;
      kern.thr  = static_cast<real_t>(thr);
      kern.win  = win;
      kern.cn   = cn;
      kern.cb   = cb;
      kern.cs   = cs;
      kern.cq   = cq;
      kern.ck   = ck;
      kern.nmin = static_cast<double>(bulk_nmin);
      Kokkos::View<double*, Kokkos::HostSpace> res("rec_mix::bulk", 5);
      Kokkos::parallel_reduce("rec_mix::BulkEnergy",
                              Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { n1, n2 }),
                              kern,
                              res);
      return { res(0), res(1), res(2), res(3), res(4) };
    }

    /* --- exact particle-state dump (validation only; A0-2, C1.12) ----------- */
    // Per rank, usr/vdump.<label>.r<rank>.bin: magic "P13VDP01", int64 label,
    // double time, uint64 run_id, int32 {rank, nspecies, sizeof(prtldx_t),
    // sizeof(real_t), off1, off2, n1, n2, domain index, number of ranks};
    // per species: int64 n_alive, int32
    // has_ids, int32 0, then int32 i1g[n], i2g[n] (global cell = local + offset),
    // prtldx_t dx1[n], dx2[n], real_t ux1[n], ux2[n], ux3[n], w[n], uint64 cv[n]
    // (the control-volume flag), and if has_ids uint64 idx[n], dom[n] (Entity
    // tracking: species counter at creation, creating domain; 0 without MPI).
    // The state is the one the moment kernels of the same post-step use, so the
    // cell index and the offset dx1 (the dx1 < 1/2 subsample) are exact; the
    // rounded output position X = i + dx cannot provide them (Codex A0-2).
    void ValidationDump(const Domain<S, M>& dom, timestep_t step, simtime_t time) const {
      const int  rank  = recmix::mpi_rank();
      const auto fname = usr_dir / fmt::format("vdump.%08lu.r%05d.bin",
                                               (unsigned long)(step + 1),
                                               rank);
      std::ofstream f(fname, std::ios::binary);
      raise::ErrorIf(not f.good(), "rec_mix: cannot open " + fname.string(), HERE);
      const char    magic[8] = { 'P', '1', '3', 'V', 'D', 'P', '0', '1' };
      f.write(magic, 8);
      const std::int64_t  st  = static_cast<std::int64_t>(step + 1);
      const double        tm  = static_cast<double>(time) + dt;
      const std::uint64_t rid = run_id;
      int                 nranks = 1;
#if defined(MPI_ENABLED)
      MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
      const std::int32_t  hdr[10] = { rank,
                                     9,
                                     static_cast<std::int32_t>(sizeof(prtldx_t)),
                                     static_cast<std::int32_t>(sizeof(real_t)),
                                     static_cast<std::int32_t>(dom.offset_ncells()[0]),
                                     static_cast<std::int32_t>(dom.offset_ncells()[1]),
                                     static_cast<std::int32_t>(dom.mesh.n_active(in::x1)),
                                     static_cast<std::int32_t>(dom.mesh.n_active(in::x2)),
                                     static_cast<std::int32_t>(dom.index()),
                                     static_cast<std::int32_t>(nranks) };
      f.write(reinterpret_cast<const char*>(&st), 8);
      f.write(reinterpret_cast<const char*>(&tm), 8);
      f.write(reinterpret_cast<const char*>(&rid), 8);
      f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
      const auto o1 = static_cast<std::int32_t>(dom.offset_ncells()[0]);
      const auto o2 = static_cast<std::int32_t>(dom.offset_ncells()[1]);
      for (auto s = 0u; s < 9; ++s) {
        const auto& sp = dom.species[s];
        const auto  np = sp.npart();
#if defined(MPI_ENABLED)
        const int ntrk = sp.use_tracking() ? 2 : 0;
#else
        const int ntrk = sp.use_tracking() ? 1 : 0;
#endif
        const int slot = (cv_slot_cfg >= 0) ? ntrk + cv_slot_cfg
                                            : static_cast<int>(sp.npld_i()) - 1;
        std::vector<std::int32_t>  i1v, i2v;
        std::vector<prtldx_t>      d1v, d2v;
        std::vector<real_t>        u1v, u2v, u3v, wv;
        std::vector<std::uint64_t> cvv, idv, dmv;
        if (np > 0) {
          const auto rng = std::make_pair(static_cast<std::size_t>(0),
                                          static_cast<std::size_t>(np));
          auto h_i1  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(sp.i1, rng));
          auto h_i2  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(sp.i2, rng));
          auto h_d1  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(sp.dx1, rng));
          auto h_d2  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(sp.dx2, rng));
          auto h_u1  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(sp.ux1, rng));
          auto h_u2  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(sp.ux2, rng));
          auto h_u3  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(sp.ux3, rng));
          auto h_w   = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(sp.weight, rng));
          auto h_tag = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(sp.tag, rng));
          auto h_pld = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           Kokkos::subview(sp.pld_i, rng, Kokkos::ALL));
          for (std::size_t p = 0; p < static_cast<std::size_t>(np); ++p) {
            if (h_tag(p) != ParticleTag::alive) {
              continue;
            }
            i1v.push_back(h_i1(p) + o1);
            i2v.push_back(h_i2(p) + o2);
            d1v.push_back(h_d1(p));
            d2v.push_back(h_d2(p));
            u1v.push_back(h_u1(p));
            u2v.push_back(h_u2(p));
            u3v.push_back(h_u3(p));
            wv.push_back(h_w(p));
            cvv.push_back(static_cast<std::uint64_t>(h_pld(p, slot)));
            if (sp.use_tracking()) {
              idv.push_back(static_cast<std::uint64_t>(h_pld(p, pldi::spcCtr)));
#if defined(MPI_ENABLED)
              dmv.push_back(static_cast<std::uint64_t>(h_pld(p, pldi::domIdx)));
#else
              dmv.push_back(0);
#endif
            }
          }
        }
        const std::int64_t n   = static_cast<std::int64_t>(i1v.size());
        const std::int32_t ids[2] = { sp.use_tracking() ? 1 : 0, 0 };
        f.write(reinterpret_cast<const char*>(&n), 8);
        f.write(reinterpret_cast<const char*>(ids), 8);
        const auto put = [&](const auto& v) {
          if (not v.empty()) {
            f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(v[0]));
          }
        };
        put(i1v);
        put(i2v);
        put(d1v);
        put(d2v);
        put(u1v);
        put(u2v);
        put(u3v);
        put(wv);
        put(cvv);
        if (sp.use_tracking()) {
          put(idv);
          put(dmv);
        }
      }
      raise::ErrorIf(not f.good(), "rec_mix: write error in " + fname.string(), HERE);
    }

    /* --- masked output ------------------------------------------------------ */
    void MaskedOutput(Domain<S, M>& dom, timestep_t step, simtime_t time) {
      if (vdump) {
        ValidationDump(dom, step, time);
      }
      ComputeMixing(dom);
      const int NM = recmix::mom::NMOM;
      std::vector<double> hist(9 * spec_num, 0.0), moms(9 * NM, 0.0);
      std::vector<double> ncells(2, 0.0);
      {
        auto   mix = m_mix;
        const real_t th = static_cast<real_t>(thr);
        double nm { 0.0 };
        const auto n1 = static_cast<int>(dom.mesh.n_active(in::x1));
        const auto n2 = static_cast<int>(dom.mesh.n_active(in::x2));
        Kokkos::parallel_reduce(
          "rec_mix::MaskCells",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { n1, n2 }),
          Lambda(int i, int j, double& acc) {
            if (mix(i, j) > th) {
              acc += 1.0;
            }
          },
          nm);
        ncells[0] = nm;
        ncells[1] = static_cast<double>(n1) * n2;
      }
      for (auto s = 0u; s < 9; ++s) {
        auto&                      sp = dom.species[s];
        Kokkos::View<double*>      hd("rec_mix::hist", spec_num);
        Kokkos::View<double*, Kokkos::HostSpace> res("rec_mix::moms", NM);
        if (sp.npart() > 0) {
          recmix::MaskedMoments_kernel<M> kern;
          kern.i1     = sp.i1;
          kern.i2     = sp.i2;
          kern.ux1    = sp.ux1;
          kern.ux2    = sp.ux2;
          kern.ux3    = sp.ux3;
          kern.weight = sp.weight;
          kern.tag    = sp.tag;
          kern.mix    = m_mix;
          kern.hist   = Kokkos::Experimental::ScatterView<double*>(hd);
          kern.thr    = static_cast<real_t>(thr);
          kern.mass   = sp.mass();
          kern.lnmin  = std::log(spec_min);
          kern.lnmax  = std::log(spec_max);
          kern.nbins  = spec_num;
          Kokkos::parallel_reduce("rec_mix::MaskedMoments",
                                  sp.rangeActiveParticles(),
                                  kern,
                                  res);
          Kokkos::Experimental::contribute(hd, kern.hist);
        } else {
          Kokkos::deep_copy(res, 0.0);
        }
        auto hh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hd);
        for (int b = 0; b < spec_num; ++b) {
          hist[s * spec_num + b] = hh(b);
        }
        for (int c = 0; c < recmix::mom::KB_OWN; ++c) {
          moms[s * NM + c] = res(c);
        }
        moms[s * NM + recmix::mom::NTOT] = res(recmix::mom::NTOT);
        moms[s * NM + recmix::mom::KTOT] = res(recmix::mom::KTOT);
      }
      // bulk-corrected energies (SM S2, P13-D32): own-species mean 3-velocity;
      // bckp comps 0: n, 1-3: sum beta, 4: sum |beta|^2 (halo-filled),
      // 5: sum (gamma - 1) in the own cell (energy share of under-occupied boxes)
      for (auto s = 0u; s < 9; ++s) {
        const int    sp1 = static_cast<int>(s + 1);
        const double m   = dom.species[s].mass();
        Kokkos::deep_copy(dom.fields.bckp, ZERO);
        DepositBulk(dom, { sp1 }, false, 0, 4, 5);
        metadomain.CommunicateBckp(dom, { 0, 5 });
        const auto r  = BulkSum(dom, 0, 1, 0, 4, bulk_window, 5);
        const auto r1 = BulkSum(dom, 0, 1, 0, 4, 1, 5);
        moms[s * NM + recmix::mom::KB_OWN]      = m * r[0];
        moms[s * NM + recmix::mom::KB_OWN_RAW]  = m * r[1];
        moms[s * NM + recmix::mom::NLOWOCC]     = r[2];
        moms[s * NM + recmix::mom::KLOWOCC]     = m * r[3];
        moms[s * NM + recmix::mom::KBLOWOCC]    = m * r[4];
        moms[s * NM + recmix::mom::KB_OWN_W1]   = m * r1[0];
        moms[s * NM + recmix::mom::NLOWOCC_W1]  = r1[2];
        moms[s * NM + recmix::mom::KLOWOCC_W1]  = m * r1[3];
        moms[s * NM + recmix::mom::KBLOWOCC_W1] = m * r1[4];
        // position-subsampling sensitivity: box means from the subsample dx1 < 1/2,
        // applied with the full cell counts (comp 5); a change against Kb_own
        // reflects the finite-sample dependence of the estimator and any
        // correlation of dx1 with velocity (A0-3b), so it is not a bias bound
        Kokkos::deep_copy(dom.fields.bckp, ZERO);
        DepositBulk(dom, { sp1 }, false, 0, 4, -1, true);
        metadomain.CommunicateBckp(dom, { 0, 5 });
        DepositCount(dom, sp1, 5);
        moms[s * NM + recmix::mom::KB_OWN_HALF] = m * BulkSum(dom, 0, 1, 5, 4, bulk_window)[0];
      }
      // ... and the mass-weighted velocity of all depositing species (Tristan
      // velx, helpers.F90 computeFluidVelocity); comps 0-3 mass-weighted sums,
      // comp 5: own count at the cell. Undefined (NaN) for non-depositing species.
      {
        std::vector<int> dep;
        for (auto s = 0u; s < 9; ++s) {
          if (dom.species[s].deposit() and dom.species[s].mass() > 0.0f) {
            dep.push_back(static_cast<int>(s + 1));
          }
        }
        Kokkos::deep_copy(dom.fields.bckp, ZERO);
        DepositBulk(dom, dep, true, 0);
        metadomain.CommunicateBckp(dom, { 0, 4 });
        for (auto s = 0u; s < 9; ++s) {
          if (not dom.species[s].deposit()) {
            moms[s * NM + recmix::mom::KB_ALL] = std::numeric_limits<double>::quiet_NaN();
            continue;
          }
          DepositCount(dom, s + 1, 5);
          moms[s * NM + recmix::mom::KB_ALL] = dom.species[s].mass() *
                                               BulkSum(dom, 0, 1, 5, -1, bulk_window)[1];
        }
      }
      recmix::reduce_to_root(hist);
      recmix::reduce_to_root(moms);
      recmix::reduce_to_root(ncells);
      const auto fill = FillFractions(dom, true);
      if (recmix::mpi_rank() == 0) {
        const auto fname = usr_dir / fmt::format("recmix.%08lu.bin", (unsigned long)(step + 1));
        std::ofstream f(fname, std::ios::binary);
        const char    magic[8] = { 'P', '1', '3', 'R', 'M', 'X', '0', '2' };
        f.write(magic, 8);
        const std::int64_t  st  = static_cast<std::int64_t>(step + 1);
        const double        tm  = static_cast<double>(time) + dt;
        const std::uint64_t rid = run_id;
        const std::int32_t  hdr[4] = { 9, spec_num, NM, 0 };
        f.write(reinterpret_cast<const char*>(&st), 8);
        f.write(reinterpret_cast<const char*>(&tm), 8);
        f.write(reinterpret_cast<const char*>(&rid), 8);
        f.write(reinterpret_cast<const char*>(hdr), 16);
        f.write(reinterpret_cast<const char*>(ncells.data()), 2 * sizeof(double));
        f.write(reinterpret_cast<const char*>(hist.data()), hist.size() * sizeof(double));
        f.write(reinterpret_cast<const char*>(moms.data()), moms.size() * sizeof(double));
        f.write(reinterpret_cast<const char*>(fill.data()), fill.size() * sizeof(double));
      }
    }

    /* --- midplane rows (D31) ------------------------------------------------- */
    void MidlineOutput(const Domain<S, M>& dom, timestep_t step, simtime_t time) const {
      const int nrows = 2 * line_halfwidth;
      const int jfst  = static_cast<int>(yline) - line_halfwidth;
      const int off1  = static_cast<int>(dom.offset_ncells()[0]);
      const int off2  = static_cast<int>(dom.offset_ncells()[1]);
      const int n1    = static_cast<int>(dom.mesh.n_active(in::x1));
      const int n2    = static_cast<int>(dom.mesh.n_active(in::x2));
      const int G     = static_cast<int>(N_GHOSTS);
      const auto EM   = dom.fields.em;
      // layout: rows[nrows][6][N1], then E3 node line [N1], B2 node line [N1]
      std::vector<float> out(static_cast<std::size_t>(nrows) * 6 * N1 + 2 * N1, 0.0f);
      Kokkos::View<real_t***> rows("rec_mix::rows", nrows, 6, n1);
      Kokkos::View<real_t**>  node("rec_mix::node", 2, n1);
      const int ra = std::max(jfst, off2), rb = std::min(jfst + nrows, off2 + n2);
      if (rb > ra) {
        Kokkos::parallel_for(
          "rec_mix::Midline",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, ra }, { n1, rb }),
          Lambda(int i, int jg) {
            const int I = i + G, J = jg - off2 + G, r = jg - jfst;
            // identical operations to kernels/fields_to_phys.hpp (2D)
            rows(r, 0, i) = INV_2 * (EM(I, J, em::ex1) + EM(I, J + 1, em::ex1));
            rows(r, 1, i) = INV_2 * (EM(I, J, em::ex2) + EM(I + 1, J, em::ex2));
            rows(r, 2, i) = INV_4 * (EM(I, J, em::ex3) + EM(I + 1, J, em::ex3) +
                                     EM(I, J + 1, em::ex3) + EM(I + 1, J + 1, em::ex3));
            rows(r, 3, i) = INV_2 * (EM(I, J, em::bx1) + EM(I + 1, J, em::bx1));
            rows(r, 4, i) = INV_2 * (EM(I, J, em::bx2) + EM(I, J + 1, em::bx2));
            rows(r, 5, i) = EM(I, J, em::bx3);
          });
      }
      const int  j0       = static_cast<int>(yline);
      const bool has_node = (j0 >= off2 and j0 < off2 + n2);
      if (has_node) {
        const int J = j0 - off2 + G;
        Kokkos::parallel_for(
          "rec_mix::NodeLine",
          Kokkos::RangePolicy<>(0, n1),
          Lambda(int i) {
            node(0, i) = EM(i + G, J, em::ex3);
            node(1, i) = EM(i + G, J, em::bx2);
          });
      }
      auto rows_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rows);
      auto node_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), node);
      for (int jg = ra; jg < rb; ++jg) {
        const int r = jg - jfst;
        for (int c = 0; c < 6; ++c) {
          for (int i = 0; i < n1; ++i) {
            out[(static_cast<std::size_t>(r) * 6 + c) * N1 + off1 + i] = rows_h(r, c, i);
          }
        }
      }
      if (has_node) {
        const std::size_t base = static_cast<std::size_t>(nrows) * 6 * N1;
        for (int i = 0; i < n1; ++i) {
          out[base + off1 + i]      = node_h(0, i);
          out[base + N1 + off1 + i] = node_h(1, i);
        }
      }
      recmix::reduce_to_root(out);
      if (recmix::mpi_rank() == 0) {
        const auto fname = usr_dir / fmt::format("midline.%08lu.bin", (unsigned long)(step + 1));
        std::ofstream f(fname, std::ios::binary);
        const char    magic[8] = { 'P', '1', '3', 'M', 'I', 'D', '0', '2' };
        f.write(magic, 8);
        const std::int64_t  st  = static_cast<std::int64_t>(step + 1);
        const double        tm  = static_cast<double>(time) + dt;
        const std::uint64_t rid = run_id;
        const std::int32_t  hdr[4] = { N1, nrows, 6, jfst };
        f.write(reinterpret_cast<const char*>(&st), 8);
        f.write(reinterpret_cast<const char*>(&tm), 8);
        f.write(reinterpret_cast<const char*>(&rid), 8);
        f.write(reinterpret_cast<const char*>(hdr), 16);
        f.write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
      }
    }

    /* ------------------------------------------------------------------ */
    void CustomPostStep(timestep_t step, simtime_t time, Domain<S, M>& dom) {
      // (1) open the outflow walls (after a restart past t_open: first post-step)
      if ((t_open > 0.0) and (not bc_opened) and (static_cast<double>(time) >= t_open)) {
        bc_opened = true;
        metadomain.setFldsBC(bc_in::Mx1, FldsBC::MATCH);
        metadomain.setPrtlBC(bc_in::Mx1, PrtlBC::ABSORB);
        metadomain.setFldsBC(bc_in::Px1, FldsBC::MATCH);
        metadomain.setPrtlBC(bc_in::Px1, PrtlBC::ABSORB);
        info(fmt::format("outflow boundaries opened at t = %g (step %lu)",
                         static_cast<double>(time),
                         (unsigned long)step));
      }
      // (2) injector, (3) padding
      if (inject_on) {
        Inject(dom, step);
      }
      if (not closed) {
        Padding(dom);
      }
      // (4) control-volume budget
      const bool usr_now = (step % static_cast<timestep_t>(usr_interval) == 0);
      if ((not cv_started) or (step % static_cast<timestep_t>(cv_interval) == 0) or usr_now) {
        CVStep(dom, step, time, usr_now);
      }
      // (5) masked spectra and moments
      if (usr_now) {
        MaskedOutput(dom, step, time);
      }
      // (6) midplane rows
      if (step % static_cast<timestep_t>(line_interval) == 0) {
        MidlineOutput(dom, step, time);
      }
    }

    /* ------------------------------------------------------------------ */
    void CustomFieldOutput(const std::string&  name,
                           ndfield_t<D, 6>&    buff,
                           cellidx_t           idx,
                           timestep_t          fstep,
                           simtime_t,
                           const Domain<S, M>& dom) {
      const int G = static_cast<int>(N_GHOSTS);
      NoteFieldOutput(fstep);
      if (name.rfind("count_", 0) == 0) {
        // raw (smoothing 0) particle count per cell of one species, for
        // independent checks of the in-code bulk energies (test C1.16)
        const int s = std::stoi(name.substr(6));
        raise::ErrorIf(s < 1 or s > 9, "rec_mix: bad custom field " + name, HERE);
        const auto& sp = dom.species[s - 1];
        Kokkos::parallel_for(
          "rec_mix::CountZero",
          Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            { 0, 0 },
            { (int)dom.mesh.n_active(in::x1), (int)dom.mesh.n_active(in::x2) }),
          Lambda(int i, int j) { buff(i + G, j + G, idx) = ZERO; });
        if (sp.npart() > 0) {
          const auto i1  = sp.i1;
          const auto i2  = sp.i2;
          const auto w   = sp.weight;
          const auto tag = sp.tag;
          Kokkos::parallel_for(
            "rec_mix::CountField",
            sp.rangeActiveParticles(),
            Lambda(prtlidx_t p) {
              if (tag(p) == ParticleTag::alive) {
                Kokkos::atomic_add(&buff(i1(p) + G, i2(p) + G, idx), w(p));
              }
            });
        }
        return;
      }
      ComputeMixing(dom);
      const auto   mix  = m_mix;
      const real_t th   = static_cast<real_t>(thr);
      const bool   mask = (name == "mask");
      raise::ErrorIf(name != "mask" and name != "mixing",
                     "rec_mix: unknown custom field " + name,
                     HERE);
      Kokkos::parallel_for(
        "rec_mix::CustomField",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
          { 0, 0 },
          { (int)dom.mesh.n_active(in::x1), (int)dom.mesh.n_active(in::x2) }),
        Lambda(int i, int j) {
          buff(i + G, j + G, idx) = mask ? ((mix(i, j) > th) ? ONE : ZERO) : mix(i, j);
        });
    }
  };

} // namespace user

#endif
