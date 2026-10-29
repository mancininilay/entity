/**
 * @file kernels/hadronic/bethe_heitler.hpp
 * @brief Bethe-Heitler ion interaction kernel (spec v3.2 II.2, mode
 *        `continuous`): continuous ion energy loss read from the startup
 *        tables with the precision-safe pending-loss application of III.3
 *        (step 2), the per-ion debt and packet mechanism of III.4 with the
 *        three sources `budget`, `poisson`, `none`, pair creation with the
 *        mean-energy spectra `delta` and `threshold` (III.4) and with the
 *        rest-frame sampler `restframe` (III.6, step 4: sampled lab photon
 *        energy, rest-frame photon energy, lepton energies and angles, exact
 *        rotation and boost; the D_max-weighted chain in the budget source,
 *        the unweighted chain in the poisson source; the pair momenta are no
 *        longer collinear, the absorbed photon's lab momentum enters
 *        P_absorbed and the sampled eps the pair payload slot 2), pair
 *        insertion into two dedicated species with atomic
 *        tickets (II.3), the hadronic RNG pool (II.4) and the full
 *        double-precision ledger with its identity (II.2 step 6).
 *
 * Conventions (all energy bookkeeping in double; the ledger energies in a
 * compensated double-double accumulator so that the identity closes to far
 * below double round-off whatever the summation order):
 *   K       = mu_d u^2 / (gamma_d + 1)          ion kinetic energy (m_e c^2)
 *   Gamma_p = 1 + K / mu_r                      physical proton Lorentz factor
 *   dK      = (tau0 / skindepth0) g(x) I(Gamma_p) dt   requested loss per step
 *   delta   = pld_r(p, pending)                 pending (unapplied) loss request
 *   r       = pld_r(p, debt)                    debt: energy removed from the
 *                                               ion and not yet paid to a pair
 *   K0      = pld_r(p, debt_k0)                 ion kinetic energy when the
 *                                               current debt started (0 if r = 0)
 *   application: K_new = max(K - delta, K_floor), u_new = sqrt(K_new (K_new +
 *   2 mu_d)) / mu_d along u-hat, stored as real_t; dK_applied = K(u_old) -
 *   K(u_new_stored) from the two stored values; delta -= dK_applied;
 *   budget: r += dK_applied, K0 <- K_new if r was 0.
 *   packets (budget): r_th = min(r_th_factor <D>(Gamma_p), f_r K,
 *   dK t_latency / dt [if t_latency > 0]); while (r >= r_th, or |K - K0| >
 *   f_r K0 once with r_packet = r) and fewer than max_events_per_step packets:
 *   r_packet = min(r, r_th); D = <D> (delta) or 2 Gamma_p - eps_mean
 *   (threshold); w_pair = w r_packet / D; both leptons at the ion's (i, dx)
 *   with gamma_e = (D + eps_mean) / 2 (delta) or Gamma_p (threshold) along
 *   u-hat; on success r -= r_packet; on capacity failure keep r and stop.
 *   Closing packet below the table edge: r_packet = r with the law at
 *   Gamma_min. Dead ions: E_exported += w r, pending_discarded += w delta,
 *   r = delta = K0 = 0.
 *   E_round: the single-precision rounding of every transfer, computed in
 *   double from the stored values: for each packet w r_packet - w_s (E_s -
 *   eps_s) with w_s the stored weight, E_s = gamma_- + gamma_+ from the stored
 *   momenta and eps_s the stored eps payload; plus w (r_exact - r_stored) for
 *   the rounding of the debt payload itself. With these, the identity
 *   E_drag + debt_initial = (E_pairs - E_absorbed) + E_round + debt_outstanding
 *   + E_exported + E_lost_nopairs holds to the accumulator's round-off.
 *   poisson: no debt; per step n ~ Poisson(R dt / pair_weight) events (R =
 *   coeff J(Gamma_p) g) from an exact sampler compared with the cap before any
 *   integer conversion, capped at max_events_per_step (n_capped_events),
 *   w_pair = pair_weight w, the unweighted law; E_stochastic accumulates the
 *   signed imbalance w dK_applied - sum_events w_pair D (expectation zero;
 *   decision (c)), so that the identity
 *   E_drag + debt_initial = (E_pairs - E_absorbed) + E_round + E_stochastic
 *   + debt_outstanding + E_exported + E_lost_nopairs holds for every source.
 *   none: loss only, E_lost_nopairs = E_drag.
 *   K0 (spec v3.3): set when r goes from 0 to positive, kept until r returns
 *   to 0; the K0 trigger pays the whole debt and has precedence over the
 *   regular trigger.
 *   mode = event (step 5, contract 2026-10-29 section 2): no steps 1 to 4,
 *   no pending request, no debt (the three payload slots are neither read
 *   nor written). Per ion and step: R = coeff J(Gamma_p) g; if R dt = 0
 *   return without RNG use; n ~ Poisson(R dt / a) from poisson_sample
 *   (compared with max_events_per_step before any integer conversion;
 *   clamped and counted in n_capped_events above it); for each event at the
 *   current state: the unweighted outcome (restframe: sample_restframe with
 *   weighted = false, overflow falls back to the mean-energy law and counts
 *   n_sampler_overflow; delta / threshold: outcome_law), the per-event bound
 *   a D <= f_max K_cur (violated: the event is skipped, no spawn, no recoil,
 *   n_bounded += 1; the host applies on_bound), the spawn with w_pair = a w
 *   (capacity failure: no recoil, n_dropped, the loop ends), the exact recoil
 *   in double K_new = K_cur - a D, u_new along the fixed u-hat stored as
 *   real_t, dK_applied = K_cur - K(u_new,stored) from the stored values,
 *   E_drag += w dK_applied, P_drag += w mu_d (u_old - u_new); E_round +=
 *   w dK_applied - w_pair D (the ion side: realised minus intended; the
 *   contract's section 2 e writes the first term with the opposite sign,
 *   which does not close the identity: fact by algebra, tested by E1) plus
 *   the pair term of spawn_pair_vec, so that E_drag = (E_pairs - E_absorbed)
 *   + E_round holds to the accumulator's round-off with E_stochastic =
 *   E_lost_nopairs = E_exported = debt = pending = 0. An ion that cools
 *   below the table's lower edge within the step has no further
 *   interaction in that step. n_events counts the drawn interactions
 *   (spawned + dropped + bounded).
 *
 * The functor is used with `Kokkos::parallel_reduce` and the `LossLedger`
 * value type (a struct of double-double accumulators, doubles and 64-bit
 * counters reduced with `Kokkos::Sum`); no real_t atomics are used anywhere;
 * integer atomics only for the spawn tickets.
 *
 * @implements
 *   - kernel::hadronic::dd
 *   - kernel::hadronic::HadronicParams
 *   - kernel::hadronic::PairTargets
 *   - kernel::hadronic::LossLedger
 *   - kernel::hadronic::BetheHeitler_kernel<>
 * @namespaces:
 *   - kernel::hadronic::
 */

#ifndef KERNELS_HADRONIC_BETHE_HEITLER_HPP
#define KERNELS_HADRONIC_BETHE_HEITLER_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"

#include "archetypes/hadronic/bh_sampler_device.h"
#include "archetypes/hadronic/bh_tables_device.h"
#include "framework/containers/particles.h"
#include "kernels/injectors.hpp"

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <limits>

namespace kernel::hadronic {
  using namespace ntt;

  /**
   * @brief Spatial profile g(x) of the photon bath (D5); only `uniform` here
   */
  enum class BathProfile : int {
    Uniform = 0
  };

  enum class PairSource : int {
    None    = 0,
    Budget  = 1,
    Poisson = 2
  };

  /**
   * @brief The loss operator (spec III.5, step 5): the continuous drag of
   *        II.2 steps 1 to 4, or the event (Monte Carlo) mode of the contract
   *        2026-10-29: n ~ Poisson(R dt / a) interactions per step, each with
   *        an unweighted outcome, one macro-pair of weight a w and the exact
   *        recoil a D; stateless per ion (the three payload slots stay zero)
   */
  enum class LossMode : int {
    Continuous = 0,
    Event      = 1
  };

  enum class PairSpectrum : int {
    Delta     = 0,
    Threshold = 1,
    RestFrame = 2 // the rest-frame sampler (step 4): sampled eps, eps', energies and angles
  };

  /**
   * @brief Compensated double-double accumulator (hi + lo), device-callable.
   *        Sums and products of doubles are accumulated with error-free
   *        transformations (two-sum, two-product via fma), so the result is
   *        correct to ~1e-32 relative to the largest partial sum, whatever the
   *        order of the reduction. Used for every ledger energy entry.
   */
  struct dd {
    double hi { 0.0 };
    double lo { 0.0 };

    KOKKOS_INLINE_FUNCTION
    dd() = default;

    KOKKOS_INLINE_FUNCTION
    explicit dd(double x) : hi { x }, lo { 0.0 } {}

    KOKKOS_INLINE_FUNCTION
    static void two_sum(double a, double b, double& s, double& e) {
      s               = a + b;
      const double bb = s - a;
      e               = (a - (s - bb)) + (b - bb);
    }

    KOKKOS_INLINE_FUNCTION
    static void two_prod(double a, double b, double& p, double& e) {
      p = a * b;
      e = Kokkos::fma(a, b, -p);
    }

    KOKKOS_INLINE_FUNCTION
    void add(double x) {
      double s, e;
      two_sum(hi, x, s, e);
      lo += e;
      two_sum(s, lo, hi, lo);
    }

    KOKKOS_INLINE_FUNCTION
    void add(const dd& o) {
      add(o.hi);
      add(o.lo);
    }

    KOKKOS_INLINE_FUNCTION
    void sub(const dd& o) {
      add(-o.hi);
      add(-o.lo);
    }

    /**
     * @brief += a * b with the product's rounding error carried
     */
    KOKKOS_INLINE_FUNCTION
    void add_prod(double a, double b) {
      double p, e;
      two_prod(a, b, p, e);
      add(p);
      add(e);
    }

    /**
     * @brief += (this dd scaled by a) ... used as a static helper: a * x
     */
    KOKKOS_INLINE_FUNCTION
    static auto prod(double a, const dd& x) -> dd {
      dd r;
      r.add_prod(a, x.hi);
      r.add_prod(a, x.lo);
      return r;
    }

    KOKKOS_INLINE_FUNCTION
    auto value() const -> double {
      return hi + lo;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator+=(const dd& o) -> dd& {
      add(o);
      return *this;
    }
  };

  /**
   * @brief Scalar parameters of the kernel (II.2)
   */
  struct HadronicParams {
    double tau0_over_skindepth0 { 0.0 };
    double mu_r { 1836.15 };
    double eta_u { 1.0e-4 };
    double pending_floor_ulp { 100.0 };
    double f_max { 0.01 };
    int    max_substeps { 64 };
    // payload indices into pld_r of the ion species
    unsigned short pld_pending { 0 };
    unsigned short pld_debt { 1 };
    unsigned short pld_debt_k0 { 2 };
    double         dt { 0.0 };
    double         time { 0.0 }; // the time stamped on newborn pairs (time after the step)
    int            profile { static_cast<int>(BathProfile::Uniform) };
    // "design off" (T5b only): the table dK is applied to u every step with no
    // pending accumulation (the remainder of each application is discarded)
    bool unsafe_application { false };
    // ---- loss operator (step 5) ---------------------------------------------
    int mode { static_cast<int>(LossMode::Continuous) };
    // ---- pair creation (step 3) ---------------------------------------------
    int    source { static_cast<int>(PairSource::None) };
    int    spectrum { static_cast<int>(PairSpectrum::Delta) };
    double r_th_factor { 10.0 };
    double f_r { 0.02 };
    double t_latency { 0.0 }; // code time; 0 = latency cap off
    double pair_weight { 1.0 }; // poisson
    int    max_events_per_step { 8 };
    // T13 only: antisymmetric longitudinal kick (in u, m_e c) of the two leptons
    // along the ion direction, deterministic from the ticket (no RNG): e- gets
    // +k u-hat, e+ gets -k u-hat with k = kick * (2 h - 1), h a hash of the
    // ticket in [0, 1); the kick energy is not rounding and lands in E_round
    double kick { 0.0 };
    // restframe: bound of the weighted chain's rejection loop per packet; an
    // exhausted loop falls back to the mean-energy law for that packet and is
    // counted in n_sampler_overflow
    int max_sampler_attempts { 100000 };
  };

  /**
   * @brief Pair destinations (II.3): the two dedicated species, their offsets
   *        (current npart), the capacity computed on the host before the
   *        launch, and the device spawn counter
   */
  struct PairTargets {
    ParticleArrays   minus;
    ParticleArrays   plus;
    npart_t          offset_m { 0 };
    npart_t          offset_p { 0 };
    npart_t          capacity { 0 };
    npart_t          counter_m { 0 };
    npart_t          counter_p { 0 };
    npart_t          domain_idx { 0 };
    bool             tracking_m { false };
    bool             tracking_p { false };
    array_t<npart_t> spawn_counter;
    bool             valid { false };
  };

  /**
   * @brief Per-launch ledger (II.2 step 6 and part IV)
   *
   * Cumulative entries are summed by the host over launches; `debt_outstanding`,
   * `pending_outstanding`, `sum_wK`, `n_alive` are snapshots of the launch.
   */
  struct LossLedger {
    dd E_drag;
    dd E_pairs;
    dd E_absorbed;
    dd E_exported;
    dd E_round;
    dd E_stochastic;   // poisson: E_drag - sum_events W D (signed, expectation zero); 0 otherwise
    dd E_lost_nopairs; // none: the applied loss (no pairs); 0 otherwise
    dd pending_discarded;
    double P_drag[3] { 0.0, 0.0, 0.0 };
    double P_pairs[3] { 0.0, 0.0, 0.0 };
    double P_absorbed[3] { 0.0, 0.0, 0.0 };
    // snapshots
    dd     debt_outstanding;
    dd     pending_outstanding;
    double sum_wK { 0.0 }; // sum of w K over alive and send-tagged ions after the launch
    // single-precision rounding budget of the transfers (T11): sum over pair
    // creations of w_pair E_pair 2^-23 plus sum over loss applications of w K 2^-22
    dd B_budget;
    // counters
    unsigned long long n_events { 0 };        // spawn attempts (packets or poisson events); event mode: drawn interactions (spawned + dropped + bounded)
    unsigned long long n_spawned { 0 };       // successful insertions (pairs)
    unsigned long long n_dropped { 0 };       // insertions refused for capacity
    unsigned long long n_substeps { 0 };
    unsigned long long n_bounded { 0 };       // continuous: substep exhaustion; event: events skipped by a D > f_max K
    unsigned long long n_capped_loss { 0 };   // steps on which the f_max K cap changed the applied loss
    unsigned long long n_closing { 0 };       // closing packets (below the table edge)
    unsigned long long n_closing_at_rest { 0 }; // closing packets of ions at u = 0 (emitted along +x)
    unsigned long long n_capped { 0 };        // budget: packet loops stopped by max_events_per_step with r >= r_th
    unsigned long long n_capped_events { 0 }; // poisson: steps on which the event count exceeded the cap
    unsigned long long n_above_table { 0 };
    unsigned long long n_latency { 0 };       // steps on which the pending floor exceeds the packet threshold
    unsigned long long n_resolution { 0 };
    unsigned long long n_processed { 0 };     // ions whose loss was evaluated from the table
    unsigned long long n_alive { 0 };         // snapshot: alive and send-tagged ions
    unsigned long long n_applied { 0 };       // applications to u this launch
    unsigned long long n_rng { 0 };           // random numbers drawn from the hadronic pool
    unsigned long long n_k0_trigger { 0 };    // packets spawned on the K0 trigger
    unsigned long long n_sampler_attempts { 0 }; // restframe: trials of the chains (accepted + rejected)
    unsigned long long n_sampler_overflow { 0 }; // restframe: packets whose rejection loop hit the bound

    KOKKOS_INLINE_FUNCTION
    LossLedger() = default;

    KOKKOS_INLINE_FUNCTION
    auto operator+=(const LossLedger& o) -> LossLedger& {
      E_drag            += o.E_drag;
      E_pairs           += o.E_pairs;
      E_absorbed        += o.E_absorbed;
      E_exported        += o.E_exported;
      E_round           += o.E_round;
      E_stochastic      += o.E_stochastic;
      E_lost_nopairs    += o.E_lost_nopairs;
      pending_discarded += o.pending_discarded;
      for (int d = 0; d < 3; ++d) {
        P_drag[d]     += o.P_drag[d];
        P_pairs[d]    += o.P_pairs[d];
        P_absorbed[d] += o.P_absorbed[d];
      }
      debt_outstanding    += o.debt_outstanding;
      pending_outstanding += o.pending_outstanding;
      sum_wK              += o.sum_wK;
      B_budget            += o.B_budget;
      n_events            += o.n_events;
      n_spawned           += o.n_spawned;
      n_dropped           += o.n_dropped;
      n_substeps          += o.n_substeps;
      n_bounded           += o.n_bounded;
      n_capped_loss       += o.n_capped_loss;
      n_closing           += o.n_closing;
      n_closing_at_rest   += o.n_closing_at_rest;
      n_capped            += o.n_capped;
      n_capped_events     += o.n_capped_events;
      n_above_table       += o.n_above_table;
      n_latency           += o.n_latency;
      n_resolution        += o.n_resolution;
      n_processed         += o.n_processed;
      n_alive             += o.n_alive;
      n_applied           += o.n_applied;
      n_rng               += o.n_rng;
      n_k0_trigger        += o.n_k0_trigger;
      n_sampler_attempts  += o.n_sampler_attempts;
      n_sampler_overflow  += o.n_sampler_overflow;
      return *this;
    }
  };

  /**
   * @brief Ion kinetic energy in double from three stored momentum components
   *        (the expression shared by the kernel and the array-based closure
   *        checks: K = mu u^2 / (gamma + 1), no cancellation)
   */
  Inline auto kinetic_energy(double mu, double u1, double u2, double u3) -> double {
    const double u2s = u1 * u1 + u2 * u2 + u3 * u3;
    const double g   = math::sqrt(1.0 + u2s);
    return mu * u2s / (g + 1.0);
  }

  /**
   * @brief Lepton Lorentz factor (mass 1) in double from the stored components;
   *        the expression shared by the kernel ledger and the array closure
   */
  Inline auto lepton_gamma(double u1, double u2, double u3) -> double {
    return math::sqrt(1.0 + u1 * u1 + u2 * u2 + u3 * u3);
  }

  /**
   * @brief ulp of a real_t value: nextafter(x, +inf) - x (x >= 0)
   */
  Inline auto ulp_real(real_t x) -> double {
    if (!math::isfinite(x)) {
      x = std::numeric_limits<real_t>::max();
    }
    if (x == std::numeric_limits<real_t>::max()) {
      // the spacing just below the largest finite value
      const real_t down = math::nextafter(x, static_cast<real_t>(0));
      return static_cast<double>(x) - static_cast<double>(down);
    }
    const real_t up = math::nextafter(x, std::numeric_limits<real_t>::infinity());
    return static_cast<double>(up) - static_cast<double>(x);
  }


  /**
   * @brief Exact Poisson deviate with mean lambda, capped: returns k in [0, kmax]
   *        or kmax + 1 meaning "the draw exceeds kmax" (the caller counts it and
   *        clamps). The comparison with the cap is made before any conversion
   *        to int, so no draw is ever converted when it does not fit (R2).
   *        lambda < 10: inversion by sequential search with the cap inside the
   *        loop (at most kmax + 1 terms). lambda >= 10: transformed rejection
   *        with squeeze, PTRS (Hoermann 1993, "The transformed rejection method
   *        for generating Poisson random variables"), exact. A non-finite or
   *        huge lambda (> 1e9) is reported as exceeding the cap without a
   *        draw. Every random number drawn is counted in n_rng.
   */
  Inline auto poisson_sample(random_generator_t& gen, double lambda, int kmax,
                             unsigned long long& n_rng) -> int {
    if (!math::isfinite(lambda) || lambda > 1.0e9) {
      return kmax + 1; // non-finite (a NaN included) or huge: exceeds the cap, no draw
    }
    if (!(lambda > 0.0)) {
      return 0;
    }
    if (lambda < 10.0) {
      const double u = gen.drand();
      n_rng += 1;
      double p = math::exp(-lambda), cum = p;
      int    k = 0;
      while (u > cum) {
        if (k >= kmax || k > 100000) {
          return kmax + 1;
        }
        ++k;
        p   *= lambda / static_cast<double>(k);
        cum += p;
      }
      return k;
    }
    const double slam     = math::sqrt(lambda);
    const double loglam   = math::log(lambda);
    const double b        = 0.931 + 2.53 * slam;
    const double a        = -0.059 + 0.02483 * b;
    const double invalpha = 1.1239 + 1.1328 / (b - 3.4);
    const double vr       = 0.9277 - 3.6224 / (b - 2.0);
    for (int it = 0; it < 10000; ++it) {
      const double U  = gen.drand() - 0.5;
      const double V  = gen.drand();
      n_rng          += 2;
      const double us = 0.5 - math::abs(U);
      const double kd = math::floor((2.0 * a / us + b) * U + lambda + 0.43);
      bool         accept = false;
      if (us >= 0.07 && V <= vr) {
        accept = true;
      } else if (kd < 0.0 || (us < 0.013 && V > us)) {
        accept = false;
      } else if (math::log(V) + math::log(invalpha) - math::log(a / (us * us) + b) <=
                 -lambda + kd * loglam - math::lgamma(kd + 1.0)) {
        accept = true;
      }
      if (accept) {
        if (kd > static_cast<double>(kmax)) {
          return kmax + 1; // compared in double: never converted
        }
        return static_cast<int>(kd);
      }
    }
    return kmax + 1; // rejection loop exhausted (probability negligible): treated as exceeding
  }

  /**
   * @brief splitmix64 finaliser (device-callable); used for the T13 kick only
   */
  Inline auto splitmix64_dev(std::uint64_t x) -> std::uint64_t {
    x += 0x9E3779B97F4A7C15ULL;
    x  = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x  = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
  }

  /**
   * @brief Bethe-Heitler kernel over the ions of one species
   * @tparam M Metric (Cartesian coordinates assumed for the pair insertion)
   */
  template <class M>
  struct BetheHeitler_kernel {
    static constexpr auto D = M::Dim;
    static constexpr auto C = M::CoordType;

    ParticleArrays                 prtls;
    const double                   mu_d;
    const arch::bh::HadronicTables tables;
    const HadronicParams           prm;
    PairTargets                    pairs;
    random_number_pool_t           pool;
    arch::bh::SamplerTables        sampler; // restframe spectrum (valid only when provided)

    /**
     * @brief Loss-only construction (source = none; no pair destinations)
     */
    BetheHeitler_kernel(const ParticleArrays&           arrays,
                        double                          ion_mass,
                        const arch::bh::HadronicTables& tbl,
                        const HadronicParams&           params)
      : prtls { arrays }
      , mu_d { ion_mass }
      , tables { tbl }
      , prm { params }
      , pairs {}
      , pool { 0x9E3779B97F4A7C15ULL }
      , sampler {} {}

    /**
     * @brief Full construction with the pair destinations and the hadronic pool
     */
    BetheHeitler_kernel(const ParticleArrays&           arrays,
                        double                          ion_mass,
                        const arch::bh::HadronicTables& tbl,
                        const HadronicParams&           params,
                        const PairTargets&              targets,
                        const random_number_pool_t&     hadronic_pool)
      : prtls { arrays }
      , mu_d { ion_mass }
      , tables { tbl }
      , prm { params }
      , pairs { targets }
      , pool { hadronic_pool }
      , sampler {} {}

    /**
     * @brief Full construction with the rest-frame sampler tables (spectrum = restframe)
     */
    BetheHeitler_kernel(const ParticleArrays&           arrays,
                        double                          ion_mass,
                        const arch::bh::HadronicTables& tbl,
                        const HadronicParams&           params,
                        const PairTargets&              targets,
                        const random_number_pool_t&     hadronic_pool,
                        const arch::bh::SamplerTables&  sampler_tables)
      : prtls { arrays }
      , mu_d { ion_mass }
      , tables { tbl }
      , prm { params }
      , pairs { targets }
      , pool { hadronic_pool }
      , sampler { sampler_tables } {}

    Inline auto restframe() const -> bool {
      return prm.spectrum == static_cast<int>(PairSpectrum::RestFrame) && sampler.valid;
    }

    /**
     * @brief One outcome of the rest-frame sampler for an ion moving along the
     *        unit vector d (simulation frame): the lab momenta of the two leptons
     *        and of the absorbed photon, rotated from the sampler's frame (z along
     *        d) with an orthonormal basis about d, the sampled lab photon energy
     *        and the ion loss D = E- + E+ - eps in double
     * @return false when the weighted chain did not accept within the bound
     */
    template <class Gen>
    Inline auto sample_restframe(double Gp, bool weighted, Gen& gen, LossLedger& L, double d1, double d2,
                                 double d3, double* vm, double* vp, double* kph, double& D, double& eps) const
      -> bool {
      arch::bh::SampledPair ev;
      const bool ok = sampler.draw(Gp, weighted, gen, L.n_rng, prm.max_sampler_attempts, ev);
      L.n_sampler_attempts += static_cast<unsigned long long>(ev.attempts);
      if (!ok) {
        return false;
      }
      // basis about d: e1 in the plane least aligned with d, e2 = d x e1
      double e1x, e1y, e1z;
      if (math::abs(d1) < 0.9) {
        e1x = 1.0 - d1 * d1;
        e1y = -d1 * d2;
        e1z = -d1 * d3;
      } else {
        e1x = -d2 * d1;
        e1y = 1.0 - d2 * d2;
        e1z = -d2 * d3;
      }
      const double l = math::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
      e1x /= l;
      e1y /= l;
      e1z /= l;
      const double e2x = d2 * e1z - d3 * e1y, e2y = d3 * e1x - d1 * e1z, e2z = d1 * e1y - d2 * e1x;
      vm[0] = ev.pm[0] * e1x + ev.pm[1] * e2x + ev.pm[2] * d1;
      vm[1] = ev.pm[0] * e1y + ev.pm[1] * e2y + ev.pm[2] * d2;
      vm[2] = ev.pm[0] * e1z + ev.pm[1] * e2z + ev.pm[2] * d3;
      vp[0] = ev.pp[0] * e1x + ev.pp[1] * e2x + ev.pp[2] * d1;
      vp[1] = ev.pp[0] * e1y + ev.pp[1] * e2y + ev.pp[2] * d2;
      vp[2] = ev.pp[0] * e1z + ev.pp[1] * e2z + ev.pp[2] * d3;
      kph[0] = ev.kph[0] * e1x + ev.kph[1] * e2x + ev.kph[2] * d1;
      kph[1] = ev.kph[0] * e1y + ev.kph[1] * e2y + ev.kph[2] * d2;
      kph[2] = ev.kph[0] * e1z + ev.kph[1] * e2z + ev.kph[2] * d3;
      D   = ev.D;
      eps = ev.eps;
      return true;
    }

    /**
     * @brief I(Gamma_p(K)) with the edge conventions (0 below, edge value above)
     */
    Inline auto loss_rate_at(double K) const -> double {
      const double Gp = 1.0 + K / prm.mu_r;
      return tables.I(Gp);
    }

    /**
     * @brief Outcome law at Gamma_p (mean-energy spectra): the ion loss D per
     *        interaction, the absorbed photon energy eps, the two lepton
     *        Lorentz factors
     */
    Inline void outcome_law(double Gp, double& Dv, double& eps, double& g_m, double& g_p) const {
      const double Dm = tables.Dmean(Gp);
      eps             = tables.EpsMean(Gp);
      if (prm.spectrum == static_cast<int>(PairSpectrum::Threshold)) {
        g_m = Gp;
        g_p = Gp;
        Dv  = 2.0 * Gp - eps;
      } else {
        Dv  = Dm;
        g_m = 0.5 * (Dm + eps);
        g_p = g_m;
      }
    }

    /**
     * @brief Write one lepton (II.3): position and prev = the ion's, momentum v,
     *        weight, tag, payloads (birth time, parent Gamma_p, eps; the rest 0)
     */
    Inline void write_lepton(ParticleArrays&       dst,
                             bool                  tracking,
                             npart_t               idx,
                             npart_t               species_cntr,
                             prtlidx_t             p,
                             const vec_t<Dim::_3D>& v,
                             real_t                w_s,
                             short                 tag,
                             double                Gp,
                             double                eps) const {
      tuple_t<int, D>      xi;
      tuple_t<prtldx_t, D> dxi;
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        xi[0]  = prtls.i1(p);
        dxi[0] = prtls.dx1(p);
      }
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        xi[1]  = prtls.i2(p);
        dxi[1] = prtls.dx2(p);
      }
      if constexpr (D == Dim::_3D) {
        xi[2]  = prtls.i3(p);
        dxi[2] = prtls.dx3(p);
      }
      if (tracking) {
        kernel::InjectParticle<D, C, true>(idx, dst.i1, dst.i2, dst.i3, dst.dx1, dst.dx2,
                                           dst.dx3, dst.ux1, dst.ux2, dst.ux3, dst.phi,
                                           dst.weight, dst.tag, dst.pld_i, xi, dxi, v, w_s,
                                           ZERO, pairs.domain_idx, species_cntr);
      } else {
        kernel::InjectParticle<D, C, false>(idx, dst.i1, dst.i2, dst.i3, dst.dx1, dst.dx2,
                                            dst.dx3, dst.ux1, dst.ux2, dst.ux3, dst.phi,
                                            dst.weight, dst.tag, dst.pld_i, xi, dxi, v, w_s,
                                            ZERO, pairs.domain_idx, species_cntr);
      }
      // *_prev = * (tile key; the next push rewrites it before any deposit, P2)
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        dst.i1_prev(idx)  = xi[0];
        dst.dx1_prev(idx) = dxi[0];
      }
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        dst.i2_prev(idx)  = xi[1];
        dst.dx2_prev(idx) = dxi[1];
      }
      if constexpr (D == Dim::_3D) {
        dst.i3_prev(idx)  = xi[2];
        dst.dx3_prev(idx) = dxi[2];
      }
      dst.tag(idx) = tag; // the parent's tag (alive or send-tagged)
      // every real payload slot of THIS destination written (R1: the two species may
      // declare different slot counts; each is zeroed from its own extent), then the
      // three defined slots
      const auto npld = dst.pld_r.extent(1);
      for (std::size_t c = 0; c < npld; ++c) {
        dst.pld_r(idx, c) = ZERO;
      }
      dst.pld_r(idx, 0) = static_cast<real_t>(prm.time);
      dst.pld_r(idx, 1) = static_cast<real_t>(Gp);
      dst.pld_r(idx, 2) = static_cast<real_t>(eps);
    }

    /**
     * @brief Spawn one pair (II.3): atomic ticket against the host capacity,
     *        both leptons written at offset + ticket, ledger entries from the
     *        stored values. `intended` is the energy the ion pays for this
     *        pair (w r_packet for budget, w_pair D for poisson) in dd; the
     *        rounding of the transfer goes to E_round.
     * @return true on success, false on capacity failure (n_dropped counted)
     */
    Inline auto spawn_pair(prtlidx_t   p,
                           LossLedger& L,
                           double      w_pair,
                           const dd&   intended,
                           double      d1,
                           double      d2,
                           double      d3, // unit direction of the ion
                           double      Gp,
                           double      eps,
                           double      g_m,
                           double      g_p,
                           short       tag) const -> bool {
      // collinear momenta along d (mean-energy spectra); the T13 kick is applied
      // in spawn_pair_vec from the ticket (kick_dir = d)
      const double um = math::sqrt((g_m > 1.0) ? (g_m - 1.0) * (g_m + 1.0) : 0.0);
      const double up = math::sqrt((g_p > 1.0) ? (g_p - 1.0) * (g_p + 1.0) : 0.0);
      const double vm[3]  = { um * d1, um * d2, um * d3 };
      const double vp[3]  = { up * d1, up * d2, up * d3 };
      const double kph[3] = { eps * d1, eps * d2, eps * d3 };
      const double dir[3] = { d1, d2, d3 };
      return spawn_pair_vec(p, L, w_pair, intended, vm, vp, kph, Gp, eps, tag, dir);
    }

    /**
     * @brief Spawn one pair with explicit lab momenta of the two leptons (double,
     *        simulation frame) and the absorbed photon's lab momentum kph; the
     *        ledger uses the stored (real_t) lepton momenta and the stored eps
     *        payload, P_absorbed the photon momentum in double
     */
    Inline auto spawn_pair_vec(prtlidx_t     p,
                               LossLedger&   L,
                               double        w_pair,
                               const dd&     intended,
                               const double* vm_d,
                               const double* vp_d,
                               const double* kph,
                               double        Gp,
                               double        eps,
                               short         tag,
                               const double* kick_dir = nullptr) const -> bool {
      L.n_events += 1;
      if (!pairs.valid) {
        L.n_dropped += 1;
        return false;
      }
      const npart_t ticket = Kokkos::atomic_fetch_add(&pairs.spawn_counter(),
                                                      static_cast<npart_t>(1));
      if (ticket >= pairs.capacity) {
        L.n_dropped += 1;
        return false;
      }
      const npart_t im = pairs.offset_m + ticket;
      const npart_t ip = pairs.offset_p + ticket;

      const real_t w_s = static_cast<real_t>(w_pair);
      // T13 kick (mean-energy spectra only): antisymmetric along kick_dir,
      // deterministic from the ticket
      double k = 0.0;
      if (kick_dir != nullptr && prm.kick != 0.0) {
        const double h = static_cast<double>(splitmix64_dev(static_cast<std::uint64_t>(ticket) ^
                                                            0xD1B54A32D192ED03ULL) >> 11) *
                         0x1.0p-53;
        k = prm.kick * (2.0 * h - 1.0);
      }
      const double kd[3] = { kick_dir ? kick_dir[0] : 0.0, kick_dir ? kick_dir[1] : 0.0,
                             kick_dir ? kick_dir[2] : 0.0 };
      vec_t<Dim::_3D> vm { static_cast<real_t>(vm_d[0] + k * kd[0]), static_cast<real_t>(vm_d[1] + k * kd[1]),
                           static_cast<real_t>(vm_d[2] + k * kd[2]) };
      vec_t<Dim::_3D> vp { static_cast<real_t>(vp_d[0] - k * kd[0]), static_cast<real_t>(vp_d[1] - k * kd[1]),
                           static_cast<real_t>(vp_d[2] - k * kd[2]) };
      ParticleArrays mm = pairs.minus;
      ParticleArrays pp = pairs.plus;
      write_lepton(mm, pairs.tracking_m, im, pairs.counter_m + ticket, p, vm, w_s, tag, Gp, eps);
      write_lepton(pp, pairs.tracking_p, ip, pairs.counter_p + ticket, p, vp, w_s, tag, Gp, eps);

      // ledger from the stored values
      const double wd    = static_cast<double>(w_s);
      const double gm_s  = lepton_gamma(static_cast<double>(vm[0]), static_cast<double>(vm[1]),
                                        static_cast<double>(vm[2]));
      const double gp_s  = lepton_gamma(static_cast<double>(vp[0]), static_cast<double>(vp[1]),
                                        static_cast<double>(vp[2]));
      const double E_s   = gm_s + gp_s;
      const double eps_s = static_cast<double>(static_cast<real_t>(eps));
      L.E_pairs.add_prod(wd, E_s);
      L.E_absorbed.add_prod(wd, eps_s);
      L.B_budget.add(wd * E_s * 0x1.0p-23);
      for (int c = 0; c < 3; ++c) {
        L.P_pairs[c] += wd * (static_cast<double>(vm[c]) + static_cast<double>(vp[c]));
      }
      // the absorbed photon's lab momentum (eps along d for the mean-energy
      // spectra, the sampled direction for restframe)
      L.P_absorbed[0] += wd * kph[0];
      L.P_absorbed[1] += wd * kph[1];
      L.P_absorbed[2] += wd * kph[2];
      // E_round += intended - w_s (E_s - eps_s)
      L.E_round.add(intended);
      L.E_round.add_prod(-wd, E_s);
      L.E_round.add_prod(wd, eps_s);
      L.n_spawned += 1;
      return true;
    }

    /**
     * @brief The event (Monte Carlo) loss mode for one alive or send-tagged
     *        ion (contract 2026-10-29 section 2; file header). Gamma_p
     *        arrives clamped to the table's upper edge; u > 0 is the stored
     *        momentum norm, K the kinetic energy from the stored components.
     */
    Inline void event_branch(prtlidx_t   p,
                             LossLedger& L,
                             short       tag,
                             double      w,
                             double      u1,
                             double      u2,
                             double      u3,
                             double      u,
                             double      K,
                             double      Gp) const {
      L.n_alive += 1;
      if (Gp < tables.gamma_min || !(u > 0.0)) {
        L.sum_wK += w * K; // below the lower edge (or at rest): no interaction, no RNG
        return;
      }
      const double g_x = 1.0; // uniform profile (D5)
      const double R   = prm.tau0_over_skindepth0 * g_x * tables.J(Gp);
      const double Rdt = R * prm.dt;
      L.n_processed += 1;
      if (Rdt == 0.0) {
        L.sum_wK += w * K; // R dt exactly 0: no RNG use (contract step 1)
        return;
      }
      const double a      = prm.pair_weight;
      const double lambda = Rdt / a; // double; non-finite or huge -> exceeds the cap
      auto         gen    = pool.get_state();
      int          n      = poisson_sample(gen, lambda, prm.max_events_per_step, L.n_rng);
      if (n > prm.max_events_per_step) {
        n = prm.max_events_per_step; // the draw exceeded the cap: clamped and counted (on_max_events)
        L.n_capped_events += 1;
      }
      // the fixed direction u-hat and the current state
      const double c1 = u1 / u, c2 = u2 / u, c3 = u3 / u;
      double       n1d = u1, n2d = u2, n3d = u3; // current stored components (double)
      double       K_cur  = K;
      double       Gp_cur = Gp;
      const double w_pair = a * w;
      const bool   rf     = restframe();
      for (int e = 0; e < n; ++e) {
        if (Gp_cur < tables.gamma_min) {
          break; // cooled below the lower edge within the step: no further interaction
        }
        // a. the unweighted outcome at the current state
        double Dv, eps, g_m, g_p;
        outcome_law(Gp_cur, Dv, eps, g_m, g_p);
        double vm[3], vp[3], kph[3], Ds = Dv, eps_s = eps;
        bool   have_vec = false;
        if (rf) {
          if (sample_restframe(Gp_cur, false, gen, L, c1, c2, c3, vm, vp, kph, Ds, eps_s)) {
            have_vec = true;
          } else {
            L.n_sampler_overflow += 1; // fall back to the mean-energy law for this event
            Ds    = Dv;
            eps_s = eps;
          }
        }
        // b. the per-event bound: the event is skipped (no spawn, no recoil) and counted
        if (a * Ds > prm.f_max * K_cur) {
          L.n_events  += 1;
          L.n_bounded += 1;
          continue;
        }
        // c. the pair, weight a w; the intended transfer w_pair D in dd
        dd intended;
        intended.add_prod(w_pair, Ds);
        const bool spawned = have_vec
                               ? spawn_pair_vec(p, L, w_pair, intended, vm, vp, kph, Gp_cur, eps_s, tag)
                               : spawn_pair(p, L, w_pair, intended, c1, c2, c3, Gp_cur, eps, g_m, g_p, tag);
        if (!spawned) {
          break; // capacity: no recoil (n_dropped counted by the spawn)
        }
        // d. the exact recoil in double, stored as real_t along u-hat
        double K_new = K_cur - a * Ds; // > 0 by b. when D > 0; >= K_cur when D <= 0
        if (K_new < 0.0) {
          K_new = 0.0; // unreachable for f_max < 1; a guard, never a sentinel
        }
        const double u_new = math::sqrt(K_new * (K_new + 2.0 * mu_d)) / mu_d;
        const real_t m1    = static_cast<real_t>(u_new * c1);
        const real_t m2    = static_cast<real_t>(u_new * c2);
        const real_t m3    = static_cast<real_t>(u_new * c3);
        prtls.ux1(p)       = m1;
        prtls.ux2(p)       = m2;
        prtls.ux3(p)       = m3;
        const double m1d = static_cast<double>(m1), m2d = static_cast<double>(m2), m3d = static_cast<double>(m3);
        const double K_stored   = kinetic_energy(mu_d, m1d, m2d, m3d);
        const double dK_applied = K_cur - K_stored;
        L.E_drag.add_prod(w, dK_applied);
        L.B_budget.add(w * K_cur * 0x1.0p-22);
        L.P_drag[0] += w * mu_d * (n1d - m1d);
        L.P_drag[1] += w * mu_d * (n2d - m2d);
        L.P_drag[2] += w * mu_d * (n3d - m3d);
        // e. the ion side of the rounding residual: realised minus intended (the
        //    pair side, intended minus stored, was added by the spawn)
        L.E_round.add_prod(w, dK_applied);
        L.E_round.sub(intended);
        L.n_applied += 1;
        // f. the new state
        n1d    = m1d;
        n2d    = m2d;
        n3d    = m3d;
        K_cur  = K_stored;
        Gp_cur = 1.0 + K_cur / prm.mu_r;
        if (Gp_cur > tables.gamma_max) {
          Gp_cur = tables.gamma_max;
        }
      }
      pool.free_state(gen);
      L.sum_wK += w * K_cur;
    }

    Inline void operator()(prtlidx_t p, LossLedger& L) const {
      const short  tag   = prtls.tag(p);
      const double w     = static_cast<double>(prtls.weight(p));
      double       delta = static_cast<double>(prtls.pld_r(p, prm.pld_pending));
      const bool   budget  = prm.source == static_cast<int>(PairSource::Budget);
      const bool   poisson = prm.source == static_cast<int>(PairSource::Poisson);
      // debt and K0 are read only with the budget source (the slots stay 0 otherwise)
      const double r_in  = budget ? static_cast<double>(prtls.pld_r(p, prm.pld_debt)) : 0.0;
      double       K0    = budget ? static_cast<double>(prtls.pld_r(p, prm.pld_debt_k0)) : 0.0;

      if (tag == ParticleTag::dead) {
        // absorbed this step (the request is never fulfilled, the debt is exported),
        // or a cleared remnant (zero payloads: a particle sent to another domain
        // leaves its slot dead with every payload zeroed by the send-buffer packer,
        // so nothing is exported for it; the receiver owns its debt)
        if (delta != 0.0) {
          L.pending_discarded.add_prod(w, delta);
          prtls.pld_r(p, prm.pld_pending) = ZERO;
        }
        if (budget && r_in != 0.0) {
          L.E_exported.add_prod(w, r_in);
          prtls.pld_r(p, prm.pld_debt)    = ZERO;
          prtls.pld_r(p, prm.pld_debt_k0) = ZERO;
        }
        return;
      }
      // alive or send-tagged: process

      // step 0: energies
      const real_t u1r = prtls.ux1(p), u2r = prtls.ux2(p), u3r = prtls.ux3(p);
      const double u1  = static_cast<double>(u1r);
      const double u2  = static_cast<double>(u2r);
      const double u3  = static_cast<double>(u3r);
      const double u2s = u1 * u1 + u2 * u2 + u3 * u3;
      const double u   = math::sqrt(u2s);
      const double gd  = math::sqrt(1.0 + u2s);
      const double K   = mu_d * u2s / (gd + 1.0);
      double       Gp  = 1.0 + K / prm.mu_r;
      // debt in exact arithmetic across this step
      dd r;
      r.add(r_in);

      if (Gp > tables.gamma_max) {
        Gp = tables.gamma_max; // clamp to the edge: edge value of the table
        L.n_above_table += 1;
      }
      if (prm.mode == static_cast<int>(LossMode::Event)) {
        event_branch(p, L, tag, w, u1, u2, u3, u, K, Gp);
        return;
      }
      if (Gp < tables.gamma_min) {
        // below the lower edge: no loss; a pending request is discarded; a debt
        // (ion decelerated by the fields) is paid in one closing packet with the
        // law at Gamma_min
        if (delta != 0.0) {
          L.pending_discarded.add_prod(w, delta);
          prtls.pld_r(p, prm.pld_pending) = ZERO;
        }
        if (budget && r_in > 0.0) {
          double Dv, eps, g_m, g_p;
          outcome_law(tables.gamma_min, Dv, eps, g_m, g_p);
          dd intended;
          intended.add_prod(w, r_in);
          // R4: an ion stopped exactly at u = 0 has no direction; the newborn pair is
          // emitted along +x (convention) so that the debt does not wait indefinitely
          const double c1 = u > 0.0 ? u1 / u : 1.0;
          const double c2 = u > 0.0 ? u2 / u : 0.0;
          const double c3 = u > 0.0 ? u3 / u : 0.0;
          if (u == 0.0) {
            L.n_closing_at_rest += 1;
          }
          bool spawned = false;
          if (restframe()) {
            // the weighted chain at the lower table edge
            auto   gen = pool.get_state();
            double vm[3], vp[3], kph[3], Ds, eps_s;
            const bool ok = sample_restframe(tables.gamma_min, true, gen, L, c1, c2, c3, vm, vp, kph, Ds, eps_s);
            pool.free_state(gen);
            if (ok) {
              spawned = spawn_pair_vec(p, L, w * r_in / Ds, intended, vm, vp, kph, Gp, eps_s, tag);
            } else {
              L.n_sampler_overflow += 1;
              spawned = spawn_pair(p, L, w * r_in / Dv, intended, c1, c2, c3, Gp, eps, g_m, g_p, tag);
            }
          } else {
            spawned = spawn_pair(p, L, w * r_in / Dv, intended, c1, c2, c3, Gp, eps, g_m, g_p, tag);
          }
          if (spawned) {
            r = dd {};
            K0 = 0.0;
            L.n_closing += 1;
          }
          const real_t r_s = static_cast<real_t>(r.value());
          dd           rounding = r;
          rounding.add(-static_cast<double>(r_s));
          L.E_round.add(dd::prod(w, rounding));
          prtls.pld_r(p, prm.pld_debt)    = r_s;
          prtls.pld_r(p, prm.pld_debt_k0) = static_cast<real_t>(K0);
          L.debt_outstanding.add_prod(w, static_cast<double>(r_s));
        } else if (budget) {
          L.debt_outstanding.add_prod(w, r_in);
        }
        L.n_alive += 1;
        L.sum_wK  += w * K;
        return;
      }

      // step 1: requested loss
      const double g_x   = 1.0; // uniform profile (D5)
      const double coeff = prm.tau0_over_skindepth0 * g_x;
      double       dK    = coeff * tables.I(Gp) * prm.dt;
      const double K_floor = prm.mu_r * (tables.gamma_min - 1.0);
      // a request that is not finite (overflowing coefficient) or exceeds
      // f_max K is sub-cycled; the substep count is compared in double before
      // any conversion (R2: no undefined double-to-int conversion)
      const bool finite_request = math::isfinite(dK);
      if (K > 0.0 && (!finite_request || dK > prm.f_max * K)) {
        // sub-cycle: RK2 (midpoint) on dK/dt = -coeff I(Gamma_p(K))
        const double n_req = finite_request ? math::ceil(dK / (prm.f_max * K))
                                            : std::numeric_limits<double>::infinity();
        int  n       = prm.max_substeps;
        bool bounded = true;
        if (finite_request && n_req <= static_cast<double>(prm.max_substeps)) {
          n       = static_cast<int>(n_req);
          bounded = false;
        }
        if (n < 1) {
          n = 1;
        }
        const double h  = prm.dt / static_cast<double>(n);
        double       Kc = K;
        for (int s = 0; s < n; ++s) {
          const double k1 = -coeff * loss_rate_at(Kc);
          double       Km = Kc + 0.5 * h * k1;
          if (Km < K_floor) {
            Km = K_floor;
          }
          const double k2 = -coeff * loss_rate_at(Km);
          Kc             += h * k2;
          if (Kc < K_floor) {
            Kc = K_floor;
          }
        }
        dK = K - Kc;
        if (!math::isfinite(dK)) {
          dK = K - K_floor; // the integrator clamps at the floor; never NaN
        }
        if (bounded) {
          // explicit clamp reading (decision 2026-10-25 (c), III.7): the loss
          // of max_substeps RK2 substeps is capped at f_max K; n_bounded counts
          // the exhaustion, n_capped_loss the steps on which the cap changed
          // the applied value
          if (dK > prm.f_max * K) {
            dK = prm.f_max * K;
            L.n_capped_loss += 1;
          }
          L.n_bounded += 1;
        }
        L.n_substeps += static_cast<unsigned long long>(n);
      }
      delta += dK;
      L.n_processed += 1;

      // packet threshold at the pre-application state (couples the application
      // threshold to the packet threshold, III.3)
      double r_th_pre = 0.0;
      if (budget && dK > 0.0) {
        const double a = prm.r_th_factor * tables.Dmean(Gp);
        const double b = prm.f_r * K;
        r_th_pre       = (a < b) ? a : b;
        if (prm.t_latency > 0.0) {
          const double c = dK * prm.t_latency / prm.dt;
          if (c < r_th_pre) {
            r_th_pre = c;
          }
        }
      }

      // step 2: precision-safe application. The ulp of the stored momentum
      // norm is taken from the double norm u rounded to real_t (R4: a real_t
      // sum of squares overflows for |u| ~ 1e20); a norm that is not
      // representable in real_t takes the ulp of the largest finite real_t
      const real_t ur = math::isfinite(static_cast<real_t>(u))
                          ? static_cast<real_t>(u)
                          : std::numeric_limits<real_t>::max();
      const double pending_floor = prm.pending_floor_ulp * ulp_real(ur) * mu_d *
                                   u / gd;
      double delta_th_base = prm.eta_u * K;
      if (budget && r_th_pre > 0.0 && r_th_pre < delta_th_base) {
        delta_th_base = r_th_pre;
      }
      if (budget && r_th_pre > 0.0 && pending_floor > r_th_pre) {
        L.n_latency += 1;
      }
      const double delta_th = (pending_floor > delta_th_base) ? pending_floor : delta_th_base;
      double K_cur = K;
      double n1d = u1, n2d = u2, n3d = u3; // current stored components (double)
      double dK_applied = 0.0;
      if (delta > 0.0 && delta >= delta_th && u > 0.0) {
        double K_new = K - delta;
        if (K_new < K_floor) {
          K_new = K_floor;
        }
        if (K_new > K) {
          K_new = K;
        }
        const double u_new = math::sqrt(K_new * (K_new + 2.0 * mu_d)) / mu_d;
        const double scale = u_new / u;
        const real_t n1 = static_cast<real_t>(scale * u1);
        const real_t n2 = static_cast<real_t>(scale * u2);
        const real_t n3 = static_cast<real_t>(scale * u3);
        prtls.ux1(p) = n1;
        prtls.ux2(p) = n2;
        prtls.ux3(p) = n3;
        n1d = static_cast<double>(n1);
        n2d = static_cast<double>(n2);
        n3d = static_cast<double>(n3);
        const double K_stored = kinetic_energy(mu_d, n1d, n2d, n3d);
        dK_applied = K - K_stored;
        delta -= dK_applied;
        if (prm.unsafe_application) {
          delta = 0.0; // design off: no pending accumulation
        }
        L.E_drag.add_prod(w, dK_applied);
        L.B_budget.add(w * K * 0x1.0p-22);
        if (budget) {
          if (r_in == 0.0) {
            K0 = K_stored;
          }
          r.add(dK_applied);
        } else if (poisson) {
          L.E_stochastic.add_prod(w, dK_applied);
        } else {
          L.E_lost_nopairs.add_prod(w, dK_applied);
        }
        L.P_drag[0] += w * mu_d * (u1 - n1d);
        L.P_drag[1] += w * mu_d * (u2 - n2d);
        L.P_drag[2] += w * mu_d * (u3 - n3d);
        L.n_applied += 1;
        K_cur = K_stored;
      } else if (prm.unsafe_application) {
        delta = 0.0;
      }
      const real_t delta_r = static_cast<real_t>(delta);
      prtls.pld_r(p, prm.pld_pending) = delta_r;
      L.pending_outstanding.add_prod(w, static_cast<double>(delta_r));
      L.sum_wK += w * K_cur;
      L.n_alive += 1;

      // steps 3-4: pair creation at the post-application state
      const double un = math::sqrt(n1d * n1d + n2d * n2d + n3d * n3d);
      double       Gp_cur = 1.0 + K_cur / prm.mu_r;
      if (Gp_cur > tables.gamma_max) {
        Gp_cur = tables.gamma_max;
      }
      const double d1 = un > 0.0 ? n1d / un : 1.0;
      const double d2 = un > 0.0 ? n2d / un : 0.0;
      const double d3 = un > 0.0 ? n3d / un : 0.0;

      if (budget) {
        if (dK > 0.0 && r.value() > 0.0 && un > 0.0) {
          double Dv, eps, g_m, g_p;
          outcome_law(Gp_cur, Dv, eps, g_m, g_p);
          double r_th = prm.r_th_factor * tables.Dmean(Gp_cur);
          if (prm.f_r * K_cur < r_th) {
            r_th = prm.f_r * K_cur;
          }
          if (prm.t_latency > 0.0) {
            const double c = dK * prm.t_latency / prm.dt;
            if (c < r_th) {
              r_th = c;
            }
          }
          bool k0_trigger = (K0 > 0.0) && (math::abs(K_cur - K0) > prm.f_r * K0);
          int  n_pk       = 0;
          const bool rf   = restframe();
          // a generator state for the restframe draws (no number is drawn from it
          // by the mean-energy spectra: acquiring it leaves the pool unchanged)
          auto gen = pool.get_state();
          if (r_th > 0.0) {
            while (true) {
              const double rv = r.value();
              if (!(rv > 0.0)) {
                break;
              }
              if (n_pk >= prm.max_events_per_step) {
                if (rv >= r_th) {
                  L.n_capped += 1;
                }
                break;
              }
              // R3 / decision (b): an active K0 trigger has precedence and pays the
              // whole debt; otherwise the regular trigger pays r_th
              double r_packet;
              bool   via_k0 = false;
              if (k0_trigger) {
                r_packet = rv;
                via_k0   = true;
              } else if (rv >= r_th) {
                r_packet = r_th;
              } else {
                break;
              }
              dd intended;
              intended.add_prod(w, r_packet);
              bool spawned;
              if (rf) {
                // the D_max-weighted chain: w_pair = w r_packet / D(z), D from the lab values
                double vm[3], vp[3], kph[3], Ds, eps_s;
                if (sample_restframe(Gp_cur, true, gen, L, d1, d2, d3, vm, vp, kph, Ds, eps_s)) {
                  spawned = spawn_pair_vec(p, L, w * r_packet / Ds, intended, vm, vp, kph, Gp_cur, eps_s, tag);
                } else {
                  L.n_sampler_overflow += 1;
                  spawned = spawn_pair(p, L, w * r_packet / Dv, intended, d1, d2, d3, Gp_cur, eps, g_m, g_p, tag);
                }
              } else {
                spawned = spawn_pair(p, L, w * r_packet / Dv, intended, d1, d2, d3, Gp_cur, eps, g_m, g_p, tag);
              }
              if (!spawned) {
                break; // capacity: keep the debt
              }
              r.add(-r_packet);
              if (via_k0) {
                L.n_k0_trigger += 1; // r is now 0: the loop ends and K0 is reset below
              }
              ++n_pk;
            }
          }
          pool.free_state(gen);
        }
        // store the debt; its rounding to real_t goes to E_round
        double rv = r.value();
        if (rv < 0.0 && rv > -1e-300) {
          rv = 0.0;
        }
        const real_t r_s = static_cast<real_t>(rv);
        if (r_s == ZERO) {
          K0 = 0.0; // K0 is reset only when the debt returns to 0 (never after a partial payment)
        }
        dd rounding = r;
        rounding.add(-static_cast<double>(r_s));
        L.E_round.add(dd::prod(w, rounding));
        prtls.pld_r(p, prm.pld_debt)    = r_s;
        prtls.pld_r(p, prm.pld_debt_k0) = static_cast<real_t>(K0);
        L.debt_outstanding.add_prod(w, static_cast<double>(r_s));
      } else if (poisson) {
        if (dK > 0.0 && un > 0.0 && prm.pair_weight > 0.0) {
          const double R      = coeff * tables.J(Gp_cur) * g_x;
          const double lambda = R * prm.dt / prm.pair_weight; // double; non-finite or huge -> exceeds the cap
          if (lambda > 0.0 || !math::isfinite(lambda)) {
            auto gen = pool.get_state();
            int  n   = poisson_sample(gen, lambda, prm.max_events_per_step, L.n_rng);
            if (n > prm.max_events_per_step) {
              n = prm.max_events_per_step; // the draw exceeded the cap: clamped and counted (on_max_events)
              L.n_capped_events += 1;
            }
            if (n > 0) {
              double Dv, eps, g_m, g_p;
              outcome_law(Gp_cur, Dv, eps, g_m, g_p);
              const double w_pair = prm.pair_weight * w;
              const bool   rf     = restframe();
              for (int e = 0; e < n; ++e) {
                dd   intended;
                bool spawned;
                if (rf) {
                  // the unweighted chain: the event's own D is the energy this pair costs
                  double vm[3], vp[3], kph[3], Ds, eps_s;
                  if (sample_restframe(Gp_cur, false, gen, L, d1, d2, d3, vm, vp, kph, Ds, eps_s)) {
                    intended.add_prod(w_pair, Ds);
                    spawned = spawn_pair_vec(p, L, w_pair, intended, vm, vp, kph, Gp_cur, eps_s, tag);
                  } else {
                    L.n_sampler_overflow += 1;
                    intended.add_prod(w_pair, Dv);
                    spawned = spawn_pair(p, L, w_pair, intended, d1, d2, d3, Gp_cur, eps, g_m, g_p, tag);
                  }
                } else {
                  intended.add_prod(w_pair, Dv);
                  spawned = spawn_pair(p, L, w_pair, intended, d1, d2, d3, Gp_cur, eps, g_m, g_p, tag);
                }
                if (!spawned) {
                  break;
                }
                // the drag not matched by this pair (signed, expectation zero)
                L.E_stochastic.sub(intended);
              }
            }
            pool.free_state(gen);
          }
        }
      }
    }
  };

} // namespace kernel::hadronic

namespace Kokkos {
  template <>
  struct reduction_identity<kernel::hadronic::LossLedger> {
    KOKKOS_FORCEINLINE_FUNCTION static kernel::hadronic::LossLedger sum() {
      return kernel::hadronic::LossLedger {};
    }
  };
} // namespace Kokkos

#endif // KERNELS_HADRONIC_BETHE_HEITLER_HPP
