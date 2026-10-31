/**
 * @file tests/kernels/bh_perf.cpp
 * @brief P2 step 7, the scaling harness at the kernel level (contract 2026-10-30 §1):
 *   variants M3: the production kernel against the two harness table variants of
 *           tests/kernels/perf/ (constants in place of the lookup; float tables with
 *           linear interpolation), continuous (none) and event (delta, a = 1) at
 *           Gamma_p0 = 3, N ions, 20 warm-up and 100 timed launches, medians; the
 *           lookup fraction and the float-lerp saving reported. PERF BUILD ONLY.
 *           Round 2 (Codex 1 F5): MATCHED WORK: before every launch the ion arrays are
 *           restored from a snapshot of the initial state and the RNG pool is re-seeded
 *           with the launch's seed, so every kernel sees the same ion states and the same
 *           random stream on every launch; the residual work mismatch (events drawn and
 *           pairs spawned per variant against the production kernel) is printed; the
 *           ledger identity (1e-12 x E_drag) is a verdict for EVERY row, warm-up launches
 *           included. Timed interval: the parallel_reduce (kernel + ledger reduction, the
 *           host-returning reducer) and the fence; pairs.finish() (the spawn-counter
 *           readback) is timed separately and reported; the production stage adds the
 *           post-step (ledger accumulation) which is not part of this kernel measurement.
 *   stride  M6: the stage run every n steps with dt_BH = n dt, n in {1, 4, 16}, at fixed
 *           physical time, for the event mode (delta, a = 1) and the continuous mode
 *           with its three sources (none; poisson a = 1; budget), N ions at Gamma_p0 =
 *           3 (BH_PERF_N, default 65536; the A100 job sets the contract's 2^22): the
 *           ACCUMULATED STAGE-LAUNCH TIME (kernel + reduction + spawn readback of every
 *           launch, fenced) per launch and per unit physical time, which is the stride
 *           saving (the whole-harness time, allocation and host readbacks included, is
 *           printed beside it and is not the saving: Codex 1 F9), <K> at the end
 *           against the ODE (event: the Jensen-corrected ODE, the contract's 4 s.e.
 *           verdict; continuous: the plain ODE, relative, REPORTED WITHOUT A VERDICT:
 *           a deterministic ensemble has no sampling error and the finite-step
 *           allowance is a contract clarification pending with Nilay), the ledger
 *           identity (1e-12 x E_drag) on every run. Measurement, not a registered
 *           physics test; the n = 16 event exceedance is an observed finite-step
 *           discrepancy, retained, and no admissibility rule is inferred from it.
 * The helpers (tables, ions, pairs, launch, cooling_run, reference) are those of
 * tests/kernels/bh_event.cpp, included with its main compiled out. Nothing here
 * touches the module's production source.
 */

#define BH_EVENT_NO_MAIN
#include "bh_event.cpp"

#include "perf/bh_perf_kernel_variant.hpp"
#include "perf/bh_perf_tables.h"

// F6 (profiling): NVTX ranges around the phases of every launch on CUDA builds (nsys --trace=nvtx picks them up;
// header-only NVTX3 of the CUDA toolkit, no library to link); no-ops elsewhere
#if defined(KOKKOS_ENABLE_CUDA)
  #include <nvtx3/nvToolsExt.h>
  #define BH_PERF_RANGE_PUSH(name) nvtxRangePushA(name)
  #define BH_PERF_RANGE_POP()      nvtxRangePop()
#else
  #define BH_PERF_RANGE_PUSH(name) ((void)0)
  #define BH_PERF_RANGE_POP()      ((void)0)
#endif

namespace {

  namespace khp = kernel::hadronic_perf;
  double identity_residual_rel(const kh::LossLedger& C);

  // F6: a configuration filter for the profiled invocations: `variants --mode event|continuous --kernel production|const|float`
  // runs ONE (mode, kernel) configuration so that a profiler's capture window covers exactly that configuration's launches
  std::string g_sel_mode, g_sel_kernel;

  bool selected(const char* mode, const char* kernel) {
    return (g_sel_mode.empty() || g_sel_mode == mode) && (g_sel_kernel.empty() || g_sel_kernel == kernel);
  }

  /* ------------------------------------------------------------------------ */
  /* M3: the production kernel against the two harness table variants           */
  /* ------------------------------------------------------------------------ */

  // one launch of any kernel type over the ions, fenced; the pairs recycled by the caller
  template <class KernelT>
  double timed_launch(const char* label, KernelT& kern, Ions& ions, Pairs& pairs, kh::LossLedger& L) {
    L = kh::LossLedger {};
    BH_PERF_RANGE_PUSH(label); // "<label>": the kernel + its ledger reduction (one parallel_reduce) + the fence
    const auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_reduce(label, CreateParticleRangePolicy<Dim::_1D>({ 0u }, { ions.n }), kern, Kokkos::Sum<kh::LossLedger>(L));
    Kokkos::fence();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    BH_PERF_RANGE_POP();
    (void)pairs;
    return s;
  }

  struct VariantRow {
    std::string kernel, mode;
    double      median_s, min_s, ns_per_ion, meanK_over_K0, identity_rel, identity_worst_launch, finish_median_s;
    unsigned long long events, spawned;
  };

  // the ion state restored before every launch (matched work, F5): momenta and the three payload slots
  struct IonSnapshot {
    array_t<real_t*>  ux1, ux2, ux3;
    array_t<real_t**> pld_r;

    explicit IonSnapshot(const Ions& ions)
      : ux1 { "snap_ux1", ions.n }
      , ux2 { "snap_ux2", ions.n }
      , ux3 { "snap_ux3", ions.n }
      , pld_r { "snap_pld_r", ions.n, 3 } {
      Kokkos::deep_copy(ux1, ions.arr.ux1);
      Kokkos::deep_copy(ux2, ions.arr.ux2);
      Kokkos::deep_copy(ux3, ions.arr.ux3);
      Kokkos::deep_copy(pld_r, ions.arr.pld_r);
    }

    void restore(Ions& ions) const {
      Kokkos::deep_copy(ions.arr.ux1, ux1);
      Kokkos::deep_copy(ions.arr.ux2, ux2);
      Kokkos::deep_copy(ions.arr.ux3, ux3);
      Kokkos::deep_copy(ions.arr.pld_r, pld_r);
      Kokkos::fence();
    }
  };

  template <class KernelT>
  VariantRow run_variant(const char* name, const char* label, Tables& T, KernelT& kern, Ions& ions, Pairs& pairs,
                         const IonSnapshot& snap, unsigned seed_base, int mode, int warm, int measured) {
    std::vector<double> t, tf;
    kh::LossLedger      L, C;
    double              worst_identity = 0.0;
    for (int l = 0; l < warm + measured; ++l) {
      // matched work: the same ion states and the same random stream on every launch, for every kernel
      BH_PERF_RANGE_PUSH("bh_perf_restore_and_reseed");
      snap.restore(ions);
      kern.pool = random_number_pool_t(seed_base + static_cast<unsigned>(l));
      pairs.recycle();
      auto targ  = pairs.targets();
      kern.pairs = targ;
      BH_PERF_RANGE_POP();
      const double s = timed_launch(label, kern, ions, pairs, L);
      BH_PERF_RANGE_PUSH("bh_perf_finish_readback");
      const auto   f0 = std::chrono::steady_clock::now();
      pairs.finish(targ); // the spawn-counter readback (deep_copy device -> host)
      const double f = std::chrono::duration<double>(std::chrono::steady_clock::now() - f0).count();
      BH_PERF_RANGE_POP();
      worst_identity = std::max(worst_identity, identity_residual_rel(L)); // every launch, warm-up included
      if (l >= warm) {
        t.push_back(s);
        tf.push_back(f);
        C += L;
      }
    }
    std::sort(t.begin(), t.end());
    std::sort(tf.begin(), tf.end());
    VariantRow r;
    r.kernel                = name;
    r.mode                  = mode == EVENT ? "event" : "continuous";
    r.median_s              = t[t.size() / 2];
    r.min_s                 = t.front();
    r.finish_median_s       = tf[tf.size() / 2];
    r.ns_per_ion            = 1.0e9 * r.median_s / static_cast<double>(ions.n);
    const auto Ks           = ions.K();
    double     m            = 0.0;
    for (const double k : Ks) {
      m += k;
    }
    const auto K0s          = snap.ux1; // the snapshot energy is the same for every ion: read one
    (void)K0s;
    r.meanK_over_K0         = m / static_cast<double>(ions.n); // filled by the caller relative to K0
    r.identity_rel          = identity_residual_rel(C);
    r.identity_worst_launch = worst_identity;
    r.events                = C.n_events;
    r.spawned               = C.n_spawned;
    (void)T;
    return r;
  }

  void test_variants(Tables& T, npart_t N) {
    const double Gp0 = 3.0, lambda1 = 0.05;
    const double coeff = coeff_for_lambda(T, Gp0, lambda1, 1.0);
    const int    warm = 20, measured = 100;
    std::printf("\n=== M3 variants: production kernel vs (i) constants in place of the lookup, (ii) float tables with linear "
                "interpolation; N = %u at Gamma_p0 = 3, lambda = %.3g, %d warm-up and %d timed launches, medians; MATCHED WORK "
                "(the ion state restored and the RNG pool re-seeded before every launch, identically for every kernel) ===\n",
                (unsigned)N, lambda1, warm, measured);
    std::printf("PERF BUILD (BH_PERF_VARIANT): the variant kernels are measurement code; none of their numbers is a physics "
                "result, and this program writes no output file. Timed interval: parallel_reduce (kernel + ledger reduction) and "
                "fence; pairs.finish() (spawn-counter readback) timed separately; the production stage's post-step is not included.\n");
    std::printf("(perf) M3 BH_PERF_N = %u\n", (unsigned)N);
    Ions                ions(N, 1.0);
    std::vector<double> K0v(N, K_of_gamma(Gp0));
    ions.set_all(K0v);
    const IonSnapshot   snap(ions);
    Pairs               pairs(static_cast<npart_t>(std::max<std::size_t>(200000, static_cast<std::size_t>(N) * 2)));
    const auto          Tc = khp::make_const_tables(T.host, Gp0);
    const auto          Tf = khp::make_float_tables(T.host);
    diag(fmt("M3 variant (i) constants at Gamma_p = 3: I = %.6g, J = %.6g, <D> = %.6g, eps = %.4g; variant (ii) float grid: %d nodes over "
             "[%.6f, %.4f]",
             Tc.I0, Tc.J0, Tc.D0, Tc.eps0, Tf.n, Tf.gamma_min, Tf.gamma_max));
    std::vector<VariantRow> rows;
    for (const int mode : { CONTINUOUS, EVENT }) {
      const char* mname = mode == EVENT ? "event" : "continuous";
      if (!selected(mname, "production") && !selected(mname, "const") && !selected(mname, "float")) {
        continue;
      }
      auto prm = params(coeff, DELTA, 1.0, 0.01, 8);
      prm.mode = mode;
      if (mode == CONTINUOUS) {
        prm.source = static_cast<int>(kh::PairSource::None);
      }
      random_number_pool_t pool(4242u);
      const unsigned       seed_base = 4242u + static_cast<unsigned>(1000 * mode);
      using KP = kh::BetheHeitler_kernel<Minkowski<Dim::_1D>>;
      using KC = khp::BetheHeitler_kernel_variant<Minkowski<Dim::_1D>, khp::PerfTablesConst>;
      using KF = khp::BetheHeitler_kernel_variant<Minkowski<Dim::_1D>, khp::PerfTablesFloat>;
      auto gate = [&](VariantRow& r) {
        r.meanK_over_K0 /= K0v[0];
        verdict(r.identity_rel <= 1.0e-12, fmt("M3 %s %s: ledger identity residual / E_drag over the timed launches", r.mode.c_str(), r.kernel.c_str()),
                r.identity_rel, 1.0e-12);
        verdict(r.identity_worst_launch <= 1.0e-12,
                fmt("M3 %s %s: worst per-launch ledger identity residual / E_drag (warm-up included)", r.mode.c_str(), r.kernel.c_str()),
                r.identity_worst_launch, 1.0e-12);
      };
      if (selected(mname, "production")) {
        KP kern { ions.arr, ions.mu_d, T.dev, prm, pairs.targets(), pool };
        rows.push_back(run_variant("production", mode == EVENT ? "bh_perf_production_event" : "bh_perf_production_continuous", T, kern,
                                   ions, pairs, snap, seed_base, mode, warm, measured));
        gate(rows.back());
      }
      if (selected(mname, "const")) {
        KC kern { ions.arr, ions.mu_d, Tc, prm, pairs.targets(), pool };
        rows.push_back(run_variant("(i) constants", mode == EVENT ? "bh_perf_variant_const_event" : "bh_perf_variant_const_continuous", T,
                                   kern, ions, pairs, snap, seed_base, mode, warm, measured));
        gate(rows.back());
      }
      if (selected(mname, "float")) {
        KF kern { ions.arr, ions.mu_d, Tf, prm, pairs.targets(), pool };
        rows.push_back(run_variant("(ii) float lerp", mode == EVENT ? "bh_perf_variant_float_event" : "bh_perf_variant_float_continuous", T,
                                   kern, ions, pairs, snap, seed_base, mode, warm, measured));
        gate(rows.back());
      }
    }
    std::printf("(perf) M3 launches per configuration: %d warm-up + %d timed = %d parallel_reduce launches of the kernel, %d configurations "
                "selected (profiler capture windows are per configuration)\n",
                warm, measured, warm + measured, static_cast<int>(rows.size()));
    std::printf("(perf) M3 %-10s %-16s %12s %12s %10s %12s %12s %10s %10s %10s\n", "mode", "kernel", "median ms", "min ms", "ns/ion",
                "<K>/K0", "identity", "events", "spawned", "finish ms");
    for (const auto& r : rows) {
      std::printf("(perf) M3 %-10s %-16s %12.4f %12.4f %10.3f %12.6f %12.2e %10llu %10llu %10.4f\n", r.mode.c_str(), r.kernel.c_str(),
                  1.0e3 * r.median_s, 1.0e3 * r.min_s, r.ns_per_ion, r.meanK_over_K0, r.identity_rel, r.events, r.spawned,
                  1.0e3 * r.finish_median_s);
    }
    // the residual work mismatch under matched states and streams: events drawn and pairs spawned per variant against the
    // production kernel (constants at Gamma_p = 3 reproduce the table's rate at the restored state; the float tables differ
    // by their rounding)
    for (const char* m : { "continuous", "event" }) {
      unsigned long long ep = 0, sp = 0;
      for (const auto& r : rows) {
        if (r.mode == m && r.kernel == "production") {
          ep = r.events;
          sp = r.spawned;
        }
      }
      for (const auto& r : rows) {
        if (r.mode == m && r.kernel != "production") {
          std::printf("(perf) M3 %s %s: residual work mismatch vs production under matched states: events %+.3e relative, spawned %+.3e "
                      "relative\n",
                      m, r.kernel.c_str(), ep > 0 ? (static_cast<double>(r.events) - static_cast<double>(ep)) / static_cast<double>(ep) : 0.0,
                      sp > 0 ? (static_cast<double>(r.spawned) - static_cast<double>(sp)) / static_cast<double>(sp) : 0.0);
        }
      }
    }
    for (const char* m : { "continuous", "event" }) {
      double tp = 0.0, tc = 0.0, tf = 0.0;
      for (const auto& r : rows) {
        if (r.mode == m) {
          if (r.kernel == "production") {
            tp = r.median_s;
          } else if (r.kernel == "(i) constants") {
            tc = r.median_s;
          } else {
            tf = r.median_s;
          }
        }
      }
      std::printf("(perf) M3 %s: lookup fraction (production - constants) / production = %.3f; float-lerp saving (production - float) / "
                  "production = %.3f [controlled: matched ion states and RNG streams per launch; limitation: the constants variant removes "
                  "the lookup's memory traffic AND its arithmetic together, and the float variant changes the interpolation coordinate "
                  "with the precision; harness variants, not physics]\n",
                  m, tp > 0.0 ? (tp - tc) / tp : 0.0, tp > 0.0 ? (tp - tf) / tp : 0.0);
    }
  }

  /* ------------------------------------------------------------------------ */
  /* F6: the RNG-pool share, measured in isolation                              */
  /* ------------------------------------------------------------------------ */
  // the only RNG work of the production kernel per ion is get_state / one or a few draws / free_state; this kernel does
  // exactly that (one draw, written to an array so that it is not optimised away) over N threads: its time is an
  // upper bound of the pool's share of the stage at equal thread count (attribution method of the README)
  struct RngPoolOnly {
    random_number_pool_t pool;
    array_t<real_t*>     out;

    Inline void operator()(const npart_t p) const {
      auto gen = pool.get_state();
      out(p)   = static_cast<real_t>(gen.drand(0.0, 1.0));
      pool.free_state(gen);
    }
  };

  void test_rngpool(npart_t N) {
    const int warm = 20, measured = 100;
    std::printf("\n=== M3 attribution: RNG pool acquire / one draw / release over N = %u threads, %d warm-up and %d timed launches "
                "(an upper bound of the pool's share of the stage) ===\n",
                (unsigned)N, warm, measured);
    array_t<real_t*>    out { "rng_out", N };
    std::vector<double> t;
    double              checksum = 0.0;
    for (int l = 0; l < warm + measured; ++l) {
      random_number_pool_t pool(9000u + static_cast<unsigned>(l));
      RngPoolOnly          k { pool, out };
      BH_PERF_RANGE_PUSH("bh_perf_rngpool_only");
      const auto t0 = std::chrono::steady_clock::now();
      Kokkos::parallel_for("bh_perf_rngpool_only", CreateParticleRangePolicy<Dim::_1D>({ 0u }, { N }), k);
      Kokkos::fence();
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      BH_PERF_RANGE_POP();
      if (l >= warm) {
        t.push_back(s);
      }
    }
    {
      auto h = Kokkos::create_mirror_view(out);
      Kokkos::deep_copy(h, out);
      for (npart_t p = 0; p < N; p += std::max<npart_t>(1, N / 1024)) {
        checksum += static_cast<double>(h(p));
      }
    }
    std::sort(t.begin(), t.end());
    std::printf("(perf) M3 rngpool: median %.4f ms, min %.4f ms, %.3f ns per thread (checksum %.6g)\n", 1.0e3 * t[t.size() / 2],
                1.0e3 * t.front(), 1.0e9 * t[t.size() / 2] / static_cast<double>(N), checksum);
    verdict(std::isfinite(checksum), "M3 rngpool: the draws are finite", checksum, 0.0);
  }

  struct StrideRow {
    int         mode, source;
    int         n;
    int         launches;
    double      seconds, launch_seconds, launch_seconds_per_launch, meanK_over_K0, z_or_rel, identity_rel;
    bool        identity_ok;
    std::string name;
  };

  constexpr int SRC_NONE    = static_cast<int>(kh::PairSource::None);
  constexpr int SRC_POISSON = static_cast<int>(kh::PairSource::Poisson);
  constexpr int SRC_BUDGET  = static_cast<int>(kh::PairSource::Budget);

  // M6 run: cooling_run of bh_event.cpp with the continuous source selectable and the stage-launch time accumulated
  // (every launch(): kernel + ledger reduction + spawn readback, fenced); the pairs are recycled outside the timing
  struct StrideRun {
    Series S;
    double launch_seconds { 0.0 };
  };

  StrideRun stride_run(Tables& T, int mode, int source, double coeff, double dt, double Gp0, npart_t N, int n_max,
                       int out_every, std::uint64_t seed, double eta_u) {
    const auto t0 = std::chrono::steady_clock::now();
    StrideRun           R;
    Series&             S = R.S;
    Ions                ions(N, 1.0);
    std::vector<double> K0v(N, K_of_gamma(Gp0));
    ions.set_all(K0v);
    S.N     = N;
    S.dt    = dt;
    S.coeff = coeff;
    S.a     = 1.0;
    S.K0    = K0v[0];
    Pairs pairs(static_cast<npart_t>(std::max<std::size_t>(200000, static_cast<std::size_t>(N) * 2)));
    auto  prm = params(coeff, DELTA, 1.0, 0.01, 8);
    prm.dt    = dt;
    prm.mode  = mode;
    if (mode == CONTINUOUS) {
      prm.source = source; // none, poisson or budget (r_th_factor 10, f_r 0.02, t_latency 0: the kernel defaults)
      prm.eta_u  = eta_u;
    }
    random_number_pool_t pool(seed);
    HostLedger           H;
    auto record = [&](int n) {
      const auto Ks = ions.K();
      double     m  = 0.0;
      for (const double k : Ks) {
        m += k;
      }
      m /= static_cast<double>(N);
      double v = 0.0;
      for (const double k : Ks) {
        v += (k - m) * (k - m);
      }
      v /= static_cast<double>(N - 1);
      S.t.push_back(static_cast<double>(n) * dt);
      S.meanK.push_back(m);
      S.varK.push_back(v);
    };
    record(0);
    int n = 0;
    while (n < n_max) {
      const auto l0 = std::chrono::steady_clock::now();
      H.add(launch(ions, pairs, T, prm, pool, static_cast<double>(n + 1) * dt)); // fenced inside; the ledger returns on the host
      R.launch_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - l0).count();
      pairs.recycle();
      ++n;
      if (n % out_every == 0) {
        record(n);
      }
    }
    if (S.t.back() != static_cast<double>(n) * dt) {
      record(n);
    }
    S.n_launch = n;
    S.C        = H.C;
    S.seconds  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return R;
  }

  double identity_residual_rel(const kh::LossLedger& C) {
    // E_drag - [(E_pairs - E_absorbed) + E_round + E_stochastic + debt_outstanding + E_exported + E_lost_nopairs]: the
    // full identity of bh_event.cpp's HostLedger::residual (the budget source leaves a debt snapshot at the end of a run;
    // the event mode and the none / poisson sources have debt and exports 0)
    kh::dd r;
    r.add(C.E_drag.value());
    r.sub(C.E_pairs);
    r.add(C.E_absorbed);
    r.sub(C.E_round);
    r.sub(C.E_stochastic);
    r.sub(C.debt_outstanding);
    r.sub(C.E_exported);
    r.sub(C.E_lost_nopairs);
    const double E = std::abs(C.E_drag.value());
    return E > 0.0 ? std::abs(r.value()) / E : std::abs(r.value());
  }

  void test_stride(Tables& T, npart_t N) {
    const double Gp0 = 3.0, lambda1 = 0.05, t_end = 1000.0;
    const double coeff = coeff_for_lambda(T, Gp0, lambda1, 1.0);
    const int    n1    = static_cast<int>(t_end / DT); // launches at n = 1
    std::printf("\n=== M6 stride: the stage every n steps with dt_BH = n dt, n in {1, 4, 16}, t_end = %g, Gamma_p0 = 3, N = %u, "
                "lambda = %.3g at n = 1; event (delta, a = 1) and continuous with source none / poisson (a = 1) / budget; "
                "the saving is the ratio of ACCUMULATED STAGE-LAUNCH times at equal physical time ===\n",
                t_end, (unsigned)N, lambda1);
    std::printf("(perf) M6 BH_PERF_N = %u\n", (unsigned)N);
    struct Cfg {
      int         mode, source;
      const char* name;
    };
    const std::vector<Cfg>  cfgs = { { EVENT, SRC_NONE, "event" }, { CONTINUOUS, SRC_NONE, "cont-none" },
                                     { CONTINUOUS, SRC_POISSON, "cont-poisson" }, { CONTINUOUS, SRC_BUDGET, "cont-budget" } };
    std::vector<StrideRow> rows;
    for (const auto& c : cfgs) {
      for (const int n : { 1, 4, 16 }) {
        const int  launches = n1 / n;
        const auto R = stride_run(T, c.mode, c.source, coeff, DT * n, Gp0, N, launches, std::max(1, launches / 10),
                                  77000u + static_cast<unsigned>(10 * n + c.mode + 100 * c.source), 1.0e-5);
        const Series& S = R.S;
        auto          d2  = std::function<double(double)>([&](double G) { return d2_delta(T, G); });
        const auto    Ref = reference(T, S, d2);
        StrideRow     row;
        row.mode                      = c.mode;
        row.source                    = c.source;
        row.name                      = c.name;
        row.n                         = n;
        row.launches                  = S.n_launch;
        row.seconds                   = S.seconds;
        row.launch_seconds            = R.launch_seconds;
        row.launch_seconds_per_launch = S.n_launch > 0 ? R.launch_seconds / S.n_launch : 0.0;
        row.meanK_over_K0             = S.meanK.back() / S.K0;
        row.identity_rel              = identity_residual_rel(S.C);
        row.identity_ok               = row.identity_rel <= 1.0e-12;
        const std::size_t i           = S.t.size() - 1;
        if (c.mode == EVENT) {
          const double se = std::sqrt(S.varK[i] / static_cast<double>(S.N));
          row.z_or_rel    = (S.meanK[i] - Ref.K_corr[i]) / se;
          verdict(std::abs(row.z_or_rel) <= 4.0,
                  fmt("M6 event n = %d: <K>(t = %g) vs the Jensen-corrected ODE, |diff| / s.e. (contract: 4 s.e.; an exceedance is an "
                      "observed finite-step discrepancy, retained)",
                      n, S.t[i]),
                  std::abs(row.z_or_rel), 4.0);
        } else {
          row.z_or_rel = S.meanK[i] / Ref.K_plain[i] - 1.0;
          diag(fmt("M6 %s n = %d: <K>(t = %g) / K_ODE,plain - 1 = %.3e [reported, no verdict: deterministic ensemble (Var K = %.2e K0^2), the "
                   "finite-step allowance is a contract clarification pending with Nilay]; dt_BH = %g",
                   c.name, n, S.t[i], row.z_or_rel, S.varK[i] / (S.K0 * S.K0), DT * n));
        }
        verdict(row.identity_ok, fmt("M6 %s n = %d: ledger identity residual / E_drag", c.name, n), row.identity_rel, 1.0e-12);
        std::printf("(perf) M6 %-12s n = %2d: %5d launches; stage-launch time %8.3f s (%.3f ms per launch, %.3f ms per unit time); "
                    "whole harness %8.3f s; <K>/K0 = %.6f, events %llu, pairs %llu, substeps %llu\n",
                    c.name, n, S.n_launch, R.launch_seconds, 1.0e3 * row.launch_seconds_per_launch, 1.0e3 * R.launch_seconds / t_end,
                    S.seconds, row.meanK_over_K0, S.C.n_events, S.C.n_spawned, S.C.n_substeps);
        rows.push_back(row);
      }
    }
    std::printf("(perf) M6 summary (saving = accumulated stage-launch time at n = 1 / at n, same physical time; the whole-harness ratio "
                "beside it is NOT the saving):\n");
    for (const auto& r : rows) {
      double t1 = 0.0, w1 = 0.0;
      for (const auto& q : rows) {
        if (q.mode == r.mode && q.source == r.source && q.n == 1) {
          t1 = q.launch_seconds;
          w1 = q.seconds;
        }
      }
      std::printf("(perf) M6 %-12s n = %2d: saving %.2f x (whole-harness ratio %.2f x), <K>/K0 %.6f, %s %+.3e, identity %.2e\n", r.name.c_str(),
                  r.n, r.launch_seconds > 0.0 ? t1 / r.launch_seconds : 0.0, r.seconds > 0.0 ? w1 / r.seconds : 0.0, r.meanK_over_K0,
                  r.mode == EVENT ? "z" : "rel", r.z_or_rel, r.identity_rel);
    }
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  try {
    std::printf("bh_perf: real_t = %s, Kokkos default execution space = %s\n", SINGLE ? "float" : "double",
                Kokkos::DefaultExecutionSpace::name());
    auto                     T = make_tables();
    std::vector<std::string> groups;
    for (int i = 1; i < argc; ++i) {
      const std::string a(argv[i]);
      if (a == "--mode" && i + 1 < argc) {
        g_sel_mode = argv[++i];
      } else if (a == "--kernel" && i + 1 < argc) {
        g_sel_kernel = argv[++i];
      } else if (argv[i][0] != '-') {
        groups.emplace_back(argv[i]);
      }
    }
    if (!g_sel_mode.empty() || !g_sel_kernel.empty()) {
      std::printf("bh_perf: configuration filter: mode '%s', kernel '%s'\n", g_sel_mode.c_str(), g_sel_kernel.c_str());
    }
    auto want = [&groups](const char* g) {
      return groups.empty() || std::find(groups.begin(), groups.end(), std::string(g)) != groups.end();
    };
    npart_t N = 65536;
    if (const char* env = std::getenv("BH_PERF_N")) {
      const long v = std::atol(env);
      if (v > 0 && v <= 100000000L) {
        N = static_cast<npart_t>(v);
      }
    }
    std::printf("bh_perf: BH_PERF_N = %u (the A100 authorisation fixes 4194304 for M3 and M6; the job asserts this line)\n", (unsigned)N);
    if (want("stride")) {
      test_stride(T, N);
    }
    if (want("variants")) {
      test_variants(T, N);
    }
    if (want("rngpool")) {
      test_rngpool(N);
    }
    std::printf("bh_perf summary: %d passed, %d failed\n", n_pass, n_fail);
    if (n_fail > 0 || n_pass == 0) {
      rc = 1;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_perf: exception: %s\n", e.what());
    rc = 2;
  }
  Kokkos::finalize();
  return rc;
}
