/**
 * @file tests/kernels/bh_perf.cpp
 * @brief P2 step 7, the scaling harness at the kernel level (contract 2026-10-30 §1):
 *   variants M3: the production kernel against the two harness table variants of
 *           tests/kernels/perf/ (constants in place of the lookup; float tables with
 *           linear interpolation), continuous (none) and event (delta, a = 1) at
 *           Gamma_p0 = 3, N ions, 20 warm-up and 100 timed launches, medians; the
 *           lookup fraction and the float-lerp saving reported. PERF BUILD ONLY.
 *   stride  M6: the stage run every n steps with dt_BH = n dt, n in {1, 4, 16}, at fixed
 *           physical time, for the event mode (delta, a = 1) and the continuous mode
 *           (source none), N ions at Gamma_p0 = 3 (BH_PERF_N, default 65536; the A100
 *           job sets 2^22): wall time per launch and per unit physical time (the
 *           saving n gives), <K> at the end against the ODE (event: the
 *           Jensen-corrected ODE, 4 s.e.; continuous: the plain ODE, relative), the
 *           ledger identity (1e-12 x E_drag) on every run. Measurement, not a
 *           registered physics test: the only verdicts are the identity and the
 *           4 s.e. cooling check of the contract.
 * The helpers (tables, ions, pairs, launch, cooling_run, reference) are those of
 * tests/kernels/bh_event.cpp, included with its main compiled out. Nothing here
 * touches the module's production source.
 */

#define BH_EVENT_NO_MAIN
#include "bh_event.cpp"

#include "perf/bh_perf_kernel_variant.hpp"
#include "perf/bh_perf_tables.h"

namespace {

  namespace khp = kernel::hadronic_perf;
  double identity_residual_rel(const kh::LossLedger& C);

  /* ------------------------------------------------------------------------ */
  /* M3: the production kernel against the two harness table variants           */
  /* ------------------------------------------------------------------------ */

  // one launch of any kernel type over the ions, fenced; the pairs recycled by the caller
  template <class KernelT>
  double timed_launch(const char* label, KernelT& kern, Ions& ions, Pairs& pairs, kh::LossLedger& L) {
    L = kh::LossLedger {};
    const auto t0 = std::chrono::steady_clock::now();
    Kokkos::parallel_reduce(label, CreateParticleRangePolicy<Dim::_1D>({ 0u }, { ions.n }), kern, Kokkos::Sum<kh::LossLedger>(L));
    Kokkos::fence();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    (void)pairs;
    return s;
  }

  struct VariantRow {
    std::string kernel, mode;
    double      median_s, min_s, ns_per_ion, meanK_over_K0, identity_rel;
    unsigned long long events;
  };

  template <class KernelT>
  VariantRow run_variant(const char* name, const char* label, Tables& T, KernelT& kern, Ions& ions, Pairs& pairs,
                         const std::vector<double>& K0v, int mode, int warm, int measured) {
    ions.set_all(K0v);
    std::vector<double> t;
    kh::LossLedger      L, C;
    for (int l = 0; l < warm + measured; ++l) {
      pairs.recycle();
      auto targ = pairs.targets();
      kern.pairs = targ;
      const double s = timed_launch(label, kern, ions, pairs, L);
      pairs.finish(targ);
      if (l >= warm) {
        t.push_back(s);
        C += L;
      }
    }
    std::sort(t.begin(), t.end());
    VariantRow r;
    r.kernel        = name;
    r.mode          = mode == EVENT ? "event" : "continuous";
    r.median_s      = t[t.size() / 2];
    r.min_s         = t.front();
    r.ns_per_ion    = 1.0e9 * r.median_s / static_cast<double>(ions.n);
    const auto Ks   = ions.K();
    double     m    = 0.0;
    for (const double k : Ks) {
      m += k;
    }
    r.meanK_over_K0 = m / static_cast<double>(ions.n) / K0v[0];
    r.identity_rel  = identity_residual_rel(C);
    r.events        = C.n_events;
    (void)T;
    return r;
  }

  void test_variants(Tables& T, npart_t N) {
    const double Gp0 = 3.0, lambda1 = 0.05;
    const double coeff = coeff_for_lambda(T, Gp0, lambda1, 1.0);
    const int    warm = 20, measured = 100;
    std::printf("\n=== M3 variants: production kernel vs (i) constants in place of the lookup, (ii) float tables with linear "
                "interpolation; N = %u at Gamma_p0 = 3, lambda = %.3g, %d warm-up and %d timed launches, medians ===\n",
                (unsigned)N, lambda1, warm, measured);
    std::printf("PERF BUILD (BH_PERF_VARIANT): the variant kernels are measurement code; none of their numbers is a physics "
                "result, and this program writes no output file.\n");
    Ions                ions(N, 1.0);
    std::vector<double> K0v(N, K_of_gamma(Gp0));
    Pairs               pairs(static_cast<npart_t>(std::max<std::size_t>(200000, static_cast<std::size_t>(N) * 2)));
    const auto          Tc = khp::make_const_tables(T.host, Gp0);
    const auto          Tf = khp::make_float_tables(T.host);
    diag(fmt("M3 variant (i) constants at Gamma_p = 3: I = %.6g, J = %.6g, <D> = %.6g, eps = %.4g; variant (ii) float grid: %d nodes over "
             "[%.6f, %.4f]",
             Tc.I0, Tc.J0, Tc.D0, Tc.eps0, Tf.n, Tf.gamma_min, Tf.gamma_max));
    std::vector<VariantRow> rows;
    for (const int mode : { CONTINUOUS, EVENT }) {
      auto prm = params(coeff, DELTA, 1.0, 0.01, 8);
      prm.mode = mode;
      if (mode == CONTINUOUS) {
        prm.source = static_cast<int>(kh::PairSource::None);
      }
      random_number_pool_t pool(4242u);
      using KP = kh::BetheHeitler_kernel<Minkowski<Dim::_1D>>;
      using KC = khp::BetheHeitler_kernel_variant<Minkowski<Dim::_1D>, khp::PerfTablesConst>;
      using KF = khp::BetheHeitler_kernel_variant<Minkowski<Dim::_1D>, khp::PerfTablesFloat>;
      {
        KP kern { ions.arr, ions.mu_d, T.dev, prm, pairs.targets(), pool };
        rows.push_back(run_variant("production", mode == EVENT ? "bh_perf_production_event" : "bh_perf_production_continuous", T, kern,
                                   ions, pairs, K0v, mode, warm, measured));
        verdict(rows.back().identity_rel <= 1.0e-12, fmt("M3 %s production kernel: ledger identity residual / E_drag", rows.back().mode.c_str()),
                rows.back().identity_rel, 1.0e-12);
      }
      {
        KC kern { ions.arr, ions.mu_d, Tc, prm, pairs.targets(), pool };
        rows.push_back(run_variant("(i) constants", mode == EVENT ? "bh_perf_variant_const_event" : "bh_perf_variant_const_continuous", T,
                                   kern, ions, pairs, K0v, mode, warm, measured));
      }
      {
        KF kern { ions.arr, ions.mu_d, Tf, prm, pairs.targets(), pool };
        rows.push_back(run_variant("(ii) float lerp", mode == EVENT ? "bh_perf_variant_float_event" : "bh_perf_variant_float_continuous", T,
                                   kern, ions, pairs, K0v, mode, warm, measured));
      }
    }
    std::printf("(perf) M3 %-10s %-16s %12s %12s %10s %12s %12s %10s\n", "mode", "kernel", "median ms", "min ms", "ns/ion", "<K>/K0", "identity",
                "events");
    for (const auto& r : rows) {
      std::printf("(perf) M3 %-10s %-16s %12.4f %12.4f %10.3f %12.6f %12.2e %10llu\n", r.mode.c_str(), r.kernel.c_str(), 1.0e3 * r.median_s,
                  1.0e3 * r.min_s, r.ns_per_ion, r.meanK_over_K0, r.identity_rel, r.events);
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
                  "production = %.3f [harness variants, not physics]\n",
                  m, tp > 0.0 ? (tp - tc) / tp : 0.0, tp > 0.0 ? (tp - tf) / tp : 0.0);
    }
  }

  struct StrideRow {
    int    mode;
    int    n;
    int    launches;
    double seconds, seconds_per_launch, meanK_over_K0, z_or_rel, identity_rel;
    bool   identity_ok;
  };

  double identity_residual_rel(const kh::LossLedger& C) {
    // E_drag - [(E_pairs - E_absorbed) + E_round + E_stochastic + E_lost_nopairs] (debt and exports are 0 here)
    kh::dd r;
    r.add(C.E_drag.value());
    r.sub(C.E_pairs);
    r.add(C.E_absorbed);
    r.sub(C.E_round);
    r.sub(C.E_stochastic);
    r.sub(C.E_lost_nopairs);
    const double E = std::abs(C.E_drag.value());
    return E > 0.0 ? std::abs(r.value()) / E : std::abs(r.value());
  }

  void test_stride(Tables& T, npart_t N) {
    const double Gp0 = 3.0, lambda1 = 0.05, t_end = 1000.0;
    const double coeff = coeff_for_lambda(T, Gp0, lambda1, 1.0);
    const int    n1    = static_cast<int>(t_end / DT); // launches at n = 1
    std::printf("\n=== M6 stride: the stage every n steps with dt_BH = n dt, n in {1, 4, 16}, t_end = %g, Gamma_p0 = 3, N = %u, "
                "lambda = %.3g at n = 1 (event delta a = 1; continuous none) ===\n",
                t_end, (unsigned)N, lambda1);
    std::vector<StrideRow> rows;
    for (const int mode : { EVENT, CONTINUOUS }) {
      for (const int n : { 1, 4, 16 }) {
        const int  launches = n1 / n;
        const auto S = cooling_run(T, mode, DELTA, 1.0, 0.01, coeff, DT * n, Gp0, N, 0.0, launches, std::max(1, launches / 10),
                                   77000u + static_cast<unsigned>(10 * n + mode), 1.0e-5);
        auto       d2 = std::function<double(double)>([&](double G) { return d2_delta(T, G); });
        const auto R  = reference(T, S, d2);
        StrideRow  row;
        row.mode               = mode;
        row.n                  = n;
        row.launches           = S.n_launch;
        row.seconds            = S.seconds;
        row.seconds_per_launch = S.n_launch > 0 ? S.seconds / S.n_launch : 0.0;
        row.meanK_over_K0      = S.meanK.back() / S.K0;
        row.identity_rel       = identity_residual_rel(S.C);
        row.identity_ok        = row.identity_rel <= 1.0e-12;
        const std::size_t i    = S.t.size() - 1;
        if (mode == EVENT) {
          const double se = std::sqrt(S.varK[i] / static_cast<double>(S.N));
          row.z_or_rel    = (S.meanK[i] - R.K_corr[i]) / se;
          verdict(std::abs(row.z_or_rel) <= 4.0,
                  fmt("M6 event n = %d: <K>(t = %g) vs the Jensen-corrected ODE, |diff| / s.e. (contract: 4 s.e.)", n, S.t[i]),
                  std::abs(row.z_or_rel), 4.0);
        } else {
          row.z_or_rel = S.meanK[i] / R.K_plain[i] - 1.0;
          diag(fmt("M6 continuous n = %d: <K>(t = %g) / K_ODE,plain - 1 = %.3e (deterministic; the Euler step dt_BH = %g against dt_BH = 0.5)",
                   n, S.t[i], row.z_or_rel, DT * n));
        }
        verdict(row.identity_ok, fmt("M6 %s n = %d: ledger identity residual / E_drag", mode == EVENT ? "event" : "continuous", n),
                row.identity_rel, 1.0e-12);
        std::printf("(perf) M6 %-10s n = %2d: %5d launches in %8.3f s (%.3f ms per launch, %.3f ms per unit time), <K>/K0 = %.6f, "
                    "events %llu, pairs %llu\n",
                    mode == EVENT ? "event" : "continuous", n, S.n_launch, S.seconds, 1.0e3 * row.seconds_per_launch,
                    1.0e3 * S.seconds / t_end, row.meanK_over_K0, S.C.n_events, S.C.n_spawned);
        rows.push_back(row);
      }
    }
    std::printf("(perf) M6 summary (saving = stage time at n = 1 / stage time at n, same physical time):\n");
    for (const auto& r : rows) {
      double t1 = 0.0;
      for (const auto& q : rows) {
        if (q.mode == r.mode && q.n == 1) {
          t1 = q.seconds;
        }
      }
      std::printf("(perf) M6 %-10s n = %2d: saving %.2f x, <K>/K0 %.6f, %s %+.3e, identity %.2e\n", r.mode == EVENT ? "event" : "continuous",
                  r.n, r.seconds > 0.0 ? t1 / r.seconds : 0.0, r.meanK_over_K0, r.mode == EVENT ? "z" : "rel", r.z_or_rel, r.identity_rel);
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
      if (argv[i][0] != '-') {
        groups.emplace_back(argv[i]);
      }
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
    if (want("stride")) {
      test_stride(T, N);
    }
    if (want("variants")) {
      test_variants(T, N);
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
