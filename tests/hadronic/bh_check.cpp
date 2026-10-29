/**
 * @file tests/hadronic/bh_check.cpp
 * @brief Host-only helper for the run-based tests of the Bethe-Heitler
 *        loss-only prototype (bash scripts under tests/hadronic/). Reads the
 *        files written by the bh_proto problem generator and the run logs, and
 *        performs the pre-registered comparisons in double precision. Uses only
 *        the standard library and the step-1 host headers (no Kokkos).
 *
 * Subcommands (all print "PASS"/"FAIL" lines and exit 0 on pass, 1 on fail,
 * 2 on a usage or file error):
 *
 *   tau0  --eps0 E --mu_r MR --mu_d MD --gamma_p G --dt DT --skindepth0 S --ratio R
 *         prints tau0 such that dK / dK_ulp = R at Gamma_p = G, with dK_ulp =
 *         ulp_float(|u|) mu_d u / gamma_d (single-precision ulp, whatever the
 *         build), and the number of steps until K has dropped by 1% (RK4 of
 *         the exact rate), as: "tau0 <v> steps <n> dK_ulp <v> dK <v>"
 *
 *   t5b   --single F --double F --tol T
 *         compares K of ion 0 at the last block of two hadronic_ions files
 *         (single and double builds); passes if |K_s/K_d - 1| <= T; also
 *         prints the max over all blocks and K_eff = K - delta
 *
 *   t5b-off --single F --double F
 *         prints K_single/K_double - 1 at the last block (the design-off
 *         demonstration table; no verdict)
 *
 *   t7a   --run F --ref F --tol T --pitch_tol P [--eta_u E --floor_ulp F --mu_d M --real float|double]
 *         K(t) of ion 0 in the B != 0 run vs the B = 0 run at every block
 *         (relative, <= T), the pitch cosine constant to P over the run, and
 *         with --eta_u the stored-K envelope |delta| <= eta_u K + pending_floor
 *
 *   t7b   --ions F --eps0 E --mu_r MR --mu_d MD --K0 K --E Ex --omegaB0 W
 *         --tau0 T --skindepth0 S [--tol TOL] [--eta_u E --floor_ulp F --real float|double]
 *         RK4 of dK/dt = omegaB0 E beta_d(K) - (tau0/skindepth0) I(Gamma_p(K))
 *         against K(t) of ion 0 at every block; prints the max relative error
 *         and the end-time error "end_error <v>"; with --tol the max is a verdict
 *
 *   kick  --ions F --E Ex --omegaB0 W --mu_d MD
 *         diagnostic: mean du per step of ion 0 (no loss) vs (q/m) omegaB0 E dt
 *
 *   order --e1 A --e2 B --e4 C
 *         prints the two convergence orders log2(A/B), log2(B/C); passes if
 *         both are within 1.0 +- 0.3
 *
 *   t8    --a LOG --b LOG --c LOG --ledger_b F --ledger_c F
 *         identical "bh_proto: checksum" lines across the three logs; the two
 *         ledgers identically zero in every column but step, time, sum_wK
 *
 *   reject --log F --key K
 *         the log names setup.K in an error message
 *
 *   step 3 (pair plumbing):
 *   ledger --file F [--tol T] [--expect_dropped 0|1] [--expect_npart N] [--min_spawned N] [--expect_rng N]
 *          [--min_sampler_overflow N]
 *          [--expect_events N] [--expect_capped_events N]
 *          [--expect_nopairs 1] [--min_bounded N] [--expect_bounded 0] [--min_events N]
 *          (step 5: n_bounded = events skipped by the per-event bound of the event mode)
 *         last row of a hadronic_ledger file: identity residual <= T (1e-12) x E_drag, optional counts
 *   abort  --log F --key K [--step_min A --step_max B]
 *         the log carries the K message "... at step N" (on_capacity abort), N in [A, B]
 *   t12b   --log F [--measured 1]
 *         prints the startup estimate (bare and sized) from the log; with --measured compares the final
 *         npart of the pair species with the bare estimate (factor 2)
 *   gauss  --test F --control F [--eps E --steps N]
 *         hadronic_gauss files: max|R - R0|/max|c rho| of the pair run within a factor 2 of the control
 *         run's measured residual (N x E, the a-priori round-off floor, when that residual is exactly
 *         zero), no growth trend, the first births within that level
 *   det    --a LOG --b LOG --expect same|differ
 *         identical (or different) "bh_proto: checksum" lines
 *   texp   --log F
 *         E_exported and pending_discarded equal the host sums over the dead ions (1e-12), identity residual
 *   cmp    --a F --b F [--tol T]
 *          compares two hadronic_ledger files row by row (T14b placement test):
 *          same number of data rows, every counter column (step, n_*, npart)
 *          exactly equal, every real column |a - b| <= T max(1, |a|, |b|)
 *          (default T = 1e-12); prints the largest relative difference
 */

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_interpolate.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  int n_fail = 0;

  void verdict(bool ok, const std::string& what, double value, double tol) {
    std::printf("%s: %s : %.6e (tolerance %.3e)\n", ok ? "PASS" : "FAIL", what.c_str(), value, tol);
    if (!ok) {
      ++n_fail;
    }
  }

  using Args = std::map<std::string, std::string>;

  Args parse_args(int argc, char** argv, int start) {
    Args a;
    for (int i = start; i < argc; ++i) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0 || i + 1 >= argc) {
        throw std::invalid_argument("expected --key value, got '" + k + "'");
      }
      a[k.substr(2)] = argv[++i];
    }
    return a;
  }

  double num(const Args& a, const std::string& k) {
    if (!a.count(k)) {
      throw std::invalid_argument("missing --" + k);
    }
    double v;
    if (!arch::bh::parse_number_strict(a.at(k), v)) {
      throw std::invalid_argument("--" + k + ": '" + a.at(k) + "' is not a finite number");
    }
    return v;
  }

  const std::string& str(const Args& a, const std::string& k) {
    if (!a.count(k)) {
      throw std::invalid_argument("missing --" + k);
    }
    return a.at(k);
  }

  /* ---- hadronic_ions file -------------------------------------------------- */
  struct IonRow {
    int    ion, group;
    double K_init, K, Gamma_p, u, pitch, delta;
    int    tag;
  };

  struct Block {
    long                step;
    double              time;
    std::vector<IonRow> rows;
  };

  std::vector<Block> read_ions(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot open " + path);
    }
    std::vector<Block> blocks;
    std::string        line;
    while (std::getline(in, line)) {
      if (line.empty()) {
        continue;
      }
      if (line[0] == '#') {
        if (line.rfind("# step ", 0) == 0) {
          Block b;
          std::istringstream ss(line.substr(2));
          std::string w;
          ss >> w >> b.step >> w >> b.time;
          blocks.push_back(b);
        }
        continue;
      }
      if (blocks.empty()) {
        throw std::runtime_error(path + ": data row before the first block header");
      }
      IonRow r;
      std::istringstream ss(line);
      if (!(ss >> r.ion >> r.group >> r.K_init >> r.K >> r.Gamma_p >> r.u >> r.pitch >> r.delta >> r.tag)) {
        throw std::runtime_error(path + ": malformed row '" + line + "'");
      }
      blocks.back().rows.push_back(r);
    }
    if (blocks.empty()) {
      throw std::runtime_error(path + ": no blocks");
    }
    return blocks;
  }

  /* ---- ledger file ----------------------------------------------------------- */
  struct LedgerRow {
    std::vector<double> cols;
  };

  std::vector<LedgerRow> read_ledger(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot open " + path);
    }
    std::vector<LedgerRow> rows;
    std::string            line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      LedgerRow r;
      std::istringstream ss(line);
      double v;
      while (ss >> v) {
        r.cols.push_back(v);
      }
      // step 2 wrote 15 columns; step 3 writes 43, step 4 round 2 45 (sum_wK is column 39, 0-based)
      if (r.cols.size() != 15 && r.cols.size() != 43 && r.cols.size() != 45) {
        throw std::runtime_error(path + ": expected 15, 43 or 45 columns, got " + std::to_string(r.cols.size()));
      }
      rows.push_back(r);
    }
    return rows;
  }

  std::vector<std::string> grep_lines(const std::string& path, const std::string& needle) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot open " + path);
    }
    std::vector<std::string> out;
    std::string              line;
    while (std::getline(in, line)) {
      if (line.find(needle) != std::string::npos) {
        out.push_back(line);
      }
    }
    return out;
  }

  /* ---- physics helpers -------------------------------------------------------- */
  struct Rates {
    std::shared_ptr<const arch::bh::PhotonBath> bath;
    std::unique_ptr<arch::bh::BetheHeitlerRates> rates;
    arch::bh::HadronicTablesHost                 T;
    double                                       mu_r;

    Rates(double eps0, double mu_r_) : mu_r { mu_r_ } {
      arch::bh::BathSpec s;
      s.type = "mono";
      s.eps0 = eps0;
      bath   = arch::bh::make_bath(s);
      rates  = std::make_unique<arch::bh::BetheHeitlerRates>(bath);
      T      = arch::bh::build_tables(*rates, mu_r);
    }

    double I_exact(double K) const {
      const double G = 1.0 + K / mu_r;
      if (!(G > T.gamma_th)) {
        return 0.0;
      }
      return rates->at_gamma(G).I;
    }

    double I_table(double K) const {
      return arch::bh::interpolate_loss_host(T, 1.0 + K / mu_r);
    }
  };

  double u_of_K(double K, double mu) {
    return std::sqrt(K * (K + 2.0 * mu)) / mu;
  }

  double rk4(double K0, double t_end, double h, const std::function<double(double)>& f) {
    double K = K0, t = 0.0;
    while (t < t_end * (1.0 - 1e-15)) {
      const double hh = std::min(h, t_end - t);
      const double k1 = f(K), k2 = f(K + 0.5 * hh * k1), k3 = f(K + 0.5 * hh * k2), k4 = f(K + hh * k3);
      K += hh / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
      t += hh;
    }
    return K;
  }

  /**
   * @brief ulp of |u| in the precision of the run that wrote the file
   */
  double ulp_of(double u, bool run_is_double) {
    if (run_is_double) {
      return std::nextafter(u, std::numeric_limits<double>::infinity()) - u;
    }
    const float uf = static_cast<float>(u);
    return static_cast<double>(std::nextafter(uf, std::numeric_limits<float>::infinity())) - static_cast<double>(uf);
  }

  /**
   * @brief stored-K envelope (decision (a)): |delta| <= eta_u K + pending_floor
   *        for every ion at every block
   */
  void envelope_check(const std::string& name, const std::vector<Block>& blocks, double mu_d, double eta_u,
                      double floor_ulp, bool run_is_double) {
    double worst = 0.0;
    for (const auto& b : blocks) {
      for (const auto& r : b.rows) {
        const double gd    = std::sqrt(1.0 + r.u * r.u);
        const double floor = floor_ulp * ulp_of(r.u, run_is_double) * mu_d * r.u / gd;
        worst              = std::max(worst, std::abs(r.delta) / (eta_u * r.K + floor));
      }
    }
    verdict(worst <= 1.0, "envelope [" + name + "]: max |K_stored - K_eff| / (eta_u K + pending_floor), all ions, all blocks",
            worst, 1.0);
  }

  /* ---- subcommands ------------------------------------------------------------- */
  int cmd_tau0(const Args& a) {
    const Rates  R(num(a, "eps0"), num(a, "mu_r"));
    const double mu_d = num(a, "mu_d"), G = num(a, "gamma_p"), dt = num(a, "dt");
    const double skin = num(a, "skindepth0"), ratio = num(a, "ratio");
    const double K    = R.mu_r * (G - 1.0);
    const double u    = u_of_K(K, mu_d);
    const double gd   = std::sqrt(1.0 + u * u);
    const float  uf   = static_cast<float>(u);
    const double ulp  = static_cast<double>(std::nextafter(uf, std::numeric_limits<float>::infinity())) -
                       static_cast<double>(uf);
    const double dK_ulp = ulp * mu_d * u / gd;
    const double I_tab  = R.I_table(K);
    const double dK     = ratio * dK_ulp;
    const double tau0   = dK * skin / (I_tab * dt);
    // steps until K has dropped by 1% (exact rate, RK4 with step dt)
    const double coeff = tau0 / skin;
    double       Kc    = K;
    long         n     = 0;
    while (Kc > 0.99 * K && n < 500000000L) {
      Kc = rk4(Kc, dt, dt, [&](double KK) { return -coeff * R.I_exact(KK); });
      ++n;
    }
    std::printf("tau0 %.17g steps %ld dK_ulp %.17g dK %.17g ulp %.17g u %.17g\n", tau0, n, dK_ulp, dK, ulp, u);
    return 0;
  }

  int cmd_t5b(const Args& a, bool with_verdict) {
    const auto S = read_ions(str(a, "single"));
    const auto Dd = read_ions(str(a, "double"));
    if (S.size() != Dd.size()) {
      std::printf("(diag) block counts differ: single %zu, double %zu (comparing the last)\n", S.size(), Dd.size());
    }
    const auto& ls = S.back().rows.at(0);
    const auto& ld = Dd.back().rows.at(0);
    const double rel     = ls.K / ld.K - 1.0;
    const double rel_eff = (ls.K - ls.delta) / (ld.K - ld.delta) - 1.0;
    const double expected_drop = (ld.K_init - ld.K) / ld.K_init;
    const double realised      = (ls.K_init - ls.K) / (ld.K_init - ld.K);
    double max_all = 0.0;
    const std::size_t nb = std::min(S.size(), Dd.size());
    for (std::size_t b = 0; b < nb; ++b) {
      if (S[b].step == Dd[b].step) {
        max_all = std::max(max_all, std::abs(S[b].rows.at(0).K / Dd[b].rows.at(0).K - 1.0));
      }
    }
    std::printf("(diag) t5b: last block step %ld: K_single %.17g K_double %.17g K_single/K_double - 1 = %+.6e, "
                "K_eff ratio - 1 = %+.6e, double-build drop %.6e, realised/expected loss %.6f, max over blocks %.6e\n",
                S.back().step, ls.K, ld.K, rel, rel_eff, expected_drop, realised, max_all);
    std::printf("T5B_ROW %+.6e %+.6e %.6f\n", rel, rel_eff, realised);
    if (with_verdict) {
      const double tol = num(a, "tol");
      verdict(std::abs(rel) <= tol, "T5b cross-build: |K_single/K_double - 1| at the end time", std::abs(rel), tol);
    }
    return 0;
  }

  int cmd_t7a(const Args& a) {
    const auto run = read_ions(str(a, "run"));
    const auto ref = read_ions(str(a, "ref"));
    const double tol = num(a, "tol"), ptol = num(a, "pitch_tol");
    if (run.size() != ref.size()) {
      throw std::runtime_error("t7a: block counts differ");
    }
    double max_rel = 0.0, max_rel_eff = 0.0, pitch_min = 1e300, pitch_max = -1e300;
    for (std::size_t b = 0; b < run.size(); ++b) {
      if (run[b].step != ref[b].step) {
        throw std::runtime_error("t7a: block steps differ");
      }
      for (std::size_t p = 0; p < run[b].rows.size(); ++p) {
        const auto& r = run[b].rows[p];
        const auto& s = ref[b].rows.at(p);
        max_rel     = std::max(max_rel, std::abs(r.K / s.K - 1.0));
        max_rel_eff = std::max(max_rel_eff, std::abs((r.K - r.delta) / (s.K - s.delta) - 1.0));
        pitch_min   = std::min(pitch_min, r.pitch);
        pitch_max   = std::max(pitch_max, r.pitch);
      }
    }
    std::printf("(diag) t7a: %zu blocks, final K(B) %.17g K(0) %.17g, max |K_B/K_0 - 1| %.6e (K_eff %.6e), pitch range [%.6e, %.6e]\n",
                run.size(), run.back().rows[0].K, ref.back().rows[0].K, max_rel, max_rel_eff, pitch_min, pitch_max);
    verdict(max_rel <= tol, "T7a: K(t) with B vs B = 0 at every output time (all ions), relative", max_rel, tol);
    verdict(pitch_max - pitch_min <= ptol, "T7a: pitch cosine u.B/(|u||B|) constant over the run (max - min)",
            pitch_max - pitch_min, ptol);
    if (a.count("eta_u")) {
      const bool dbl = a.count("real") && str(a, "real") == "double";
      envelope_check("T7a B run", run, num(a, "mu_d"), num(a, "eta_u"), num(a, "floor_ulp"), dbl);
      envelope_check("T7a B = 0 run", ref, num(a, "mu_d"), num(a, "eta_u"), num(a, "floor_ulp"), dbl);
    }
    return 0;
  }

  int cmd_t7b(const Args& a) {
    const Rates  R(num(a, "eps0"), num(a, "mu_r"));
    const double mu_d = num(a, "mu_d"), K0 = num(a, "K0"), E = num(a, "E"), w0 = num(a, "omegaB0");
    const double coeff = num(a, "tau0") / num(a, "skindepth0");
    const auto   blocks = read_ions(str(a, "ions"));
    auto f = [&](double K) {
      const double u    = u_of_K(K, mu_d);
      const double beta = u / std::sqrt(1.0 + u * u);
      return w0 * E * beta - coeff * R.I_exact(K);
    };
    auto f_tab = [&](double K) {
      const double u    = u_of_K(K, mu_d);
      const double beta = u / std::sqrt(1.0 + u * u);
      return w0 * E * beta - coeff * R.I_table(K);
    };
    double max_rel = 0.0, end_rel = 0.0, max_rel_half = 0.0, end_rel_half = 0.0;
    double max_eff_tab = 0.0, end_eff_tab = 0.0, end_eff_exact = 0.0;
    double K_ref = K0, t_ref = 0.0, K_tab = K0; // RK4 continued block to block
    for (const auto& b : blocks) {
      const double dt_run = b.time / static_cast<double>(b.step + 1);
      if (b.time > t_ref) {
        K_ref = rk4(K_ref, b.time - t_ref, dt_run / 8.0, f);
        K_tab = rk4(K_tab, b.time - t_ref, dt_run / 8.0, f_tab);
        t_ref = b.time;
      }
      {
        const double Keff = b.rows.at(0).K - b.rows.at(0).delta;
        const double rt   = std::abs(Keff / K_tab - 1.0);
        max_eff_tab       = std::max(max_eff_tab, rt);
        end_eff_tab       = rt;
        end_eff_exact     = std::abs(Keff / K_ref - 1.0);
      }
      // the stored u after step n is u^{n+1/2}: the half-step convention
      const double K_ref_half = rk4(K_ref, 0.5 * dt_run, dt_run / 8.0, [&](double K) { return -f(K); });
      const double Ksim   = b.rows.at(0).K;
      const double rel    = std::abs(Ksim / K_ref - 1.0);
      const double relh   = std::abs(Ksim / K_ref_half - 1.0);
      max_rel      = std::max(max_rel, rel);
      max_rel_half = std::max(max_rel_half, relh);
      end_rel      = rel;
      end_rel_half = relh;
    }
    std::printf("(diag) t7b: %zu blocks, final K_sim %.17g, max relative error vs RK4 (t = (n+1) dt) %.6e, "
                "end %.6e; with the half-step convention (t = (n+1/2) dt): max %.6e, end %.6e\n",
                blocks.size(), blocks.back().rows[0].K, max_rel, end_rel, max_rel_half, end_rel_half);
    std::printf("(diag) t7b: K_eff = K - delta vs the interpolated-table RK4: max %.6e, end %.6e; K_eff vs exact RK4 at the "
                "end %.6e (the lag and the interpolation contributions removed / kept)\n",
                max_eff_tab, end_eff_tab, end_eff_exact);
    std::printf("T7B_END %.17g %.17g %.17g\n", end_rel, end_rel_half, end_eff_tab);
    if (a.count("tol")) {
      const double tol = num(a, "tol");
      verdict(max_rel <= tol, "T7b: K(t) vs RK4 of (field work - loss) at every output time (finest dt)", max_rel, tol);
    }
    if (a.count("eta_u")) {
      const bool   dbl   = a.count("real") && str(a, "real") == "double";
      const double eta_u = num(a, "eta_u"), floor_ulp = num(a, "floor_ulp");
      envelope_check("T7b run", blocks, mu_d, eta_u, floor_ulp, dbl);
      // the ratio eta_u K / pending_floor at the end of the run (must exceed 1 in the registered range)
      const auto&  r  = blocks.back().rows.at(0);
      const double gd = std::sqrt(1.0 + r.u * r.u);
      const double fl = floor_ulp * ulp_of(r.u, dbl) * mu_d * r.u / gd;
      std::printf("(diag) t7b: eta_u K / pending_floor at the end = %.3f (eta_u %.3g, K %.6g, floor %.4g)\n",
                  eta_u * r.K / fl, eta_u, r.K, fl);
      std::printf("T7B_FLOOR_RATIO %.17g\n", eta_u * r.K / fl);
    }
    return 0;
  }

  /**
   * @brief Diagnostic: mean du per step of ion 0 between the first and the last
   *        block (no loss) against the requested (q/m) omegaB0 E dt
   */
  int cmd_kick(const Args& a) {
    const auto   blocks = read_ions(str(a, "ions"));
    const double E = num(a, "E"), w0 = num(a, "omegaB0"), mu_d = num(a, "mu_d");
    const auto&  first = blocks.front();
    const auto&  last  = blocks.back();
    const double dt    = last.time / static_cast<double>(last.step + 1);
    const long   nsteps = last.step - first.step;
    if (nsteps <= 0) {
      throw std::runtime_error("kick: need two blocks");
    }
    const double du_measured  = (last.rows.at(0).u - first.rows.at(0).u) / static_cast<double>(nsteps);
    const double du_requested = w0 * E * dt / mu_d;
    const float  uf  = static_cast<float>(first.rows.at(0).u);
    const double ulp = static_cast<double>(std::nextafter(uf, std::numeric_limits<float>::infinity())) - uf;
    std::printf("pusher kick: measured du/step %.10g, requested (q/m) omegaB0 E dt = %.10g, ratio - 1 = %+.4e; "
                "half-kick / ulp_float(u_init) = %.4f (ulp %.4g, u %.6g -> %.6g over %ld steps)\n",
                du_measured, du_requested, du_measured / du_requested - 1.0, 0.5 * du_requested / ulp, ulp,
                first.rows.at(0).u, last.rows.at(0).u, nsteps);
    return 0;
  }

  int cmd_order(const Args& a) {
    const double e1 = num(a, "e1"), e2 = num(a, "e2"), e4 = num(a, "e4");
    const double o1 = std::log2(e1 / e2), o2 = std::log2(e2 / e4);
    std::printf("(diag) orders: log2(e1/e2) = %.4f, log2(e2/e4) = %.4f (errors %.6e %.6e %.6e)\n", o1, o2, e1, e2, e4);
    verdict(std::abs(o1 - 1.0) <= 0.3 && std::abs(o2 - 1.0) <= 0.3,
            "dt-convergence order within 1.0 +- 0.3 for both ratios (value: first ratio)", o1, 0.3);
    return 0;
  }

  int cmd_t8(const Args& a) {
    const auto la = grep_lines(str(a, "a"), "bh_proto: checksum");
    const auto lb = grep_lines(str(a, "b"), "bh_proto: checksum");
    const auto lc = grep_lines(str(a, "c"), "bh_proto: checksum");
    bool same = !la.empty() && la.size() == lb.size() && la.size() == lc.size();
    for (std::size_t i = 0; same && i < la.size(); ++i) {
      same = (la[i] == lb[i]) && (la[i] == lc[i]);
    }
    for (const auto& l : la) {
      std::printf("(diag) run a: %s\n", l.c_str());
    }
    for (const auto& l : lb) {
      std::printf("(diag) run b: %s\n", l.c_str());
    }
    for (const auto& l : lc) {
      std::printf("(diag) run c: %s\n", l.c_str());
    }
    verdict(same, "T8: identical checksum lines (every ion array, the six field arrays) across the three runs",
            static_cast<double>(la.size()), 0.0);
    for (const char* which : { "ledger_b", "ledger_c" }) {
      const auto rows = read_ledger(str(a, which));
      double     maxabs = 0.0;
      for (const auto& r : rows) {
        const std::size_t sum_wK_col = (r.cols.size() >= 43) ? 39 : r.cols.size() - 1;
        for (std::size_t c = 2; c < r.cols.size(); ++c) { // all but step, time, sum_wK
          if (c == sum_wK_col) {
            continue;
          }
          maxabs = std::max(maxabs, std::abs(r.cols[c]));
        }
      }
      verdict(!rows.empty() && maxabs == 0.0,
              std::string("T8: ledger identically zero (") + which + ", every column but step, time, sum_wK)",
              maxabs, 0.0);
    }
    return 0;
  }

  int cmd_reject(const Args& a) {
    const std::string key = "setup." + str(a, "key");
    const auto        lines = grep_lines(str(a, "log"), key);
    bool              named = false;
    for (const auto& l : lines) {
      if (l.find("bh_proto") != std::string::npos || l.find("Error") != std::string::npos) {
        named = true;
        std::printf("(diag) %s\n", l.substr(0, 300).c_str());
      }
    }
    verdict(named, "rejection: the error message names " + key, static_cast<double>(lines.size()), 0.0);
    return 0;
  }

  /* ---- step 3: ledger file (last row) ------------------------------------- */
  constexpr std::size_t LEDGER_COLS = 45; // step-4 round-2 ledger file (n_sampler_attempts, n_sampler_overflow)

  struct LedgerLast {
    std::vector<double> c;
  };

  LedgerLast read_last_ledger_row(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot open " + path);
    }
    std::string line, last;
    while (std::getline(in, line)) {
      if (!line.empty() && line[0] != '#') {
        last = line;
      }
    }
    if (last.empty()) {
      throw std::runtime_error("no data row in " + path);
    }
    LedgerLast         r;
    std::istringstream ss(last);
    double             v;
    while (ss >> v) {
      r.c.push_back(v);
    }
    if (r.c.size() != LEDGER_COLS) {
      throw std::runtime_error("ledger row has " + std::to_string(r.c.size()) + " columns, expected " +
                               std::to_string(LEDGER_COLS));
    }
    return r;
  }

  // column indices of the step-3 ledger file (round 2: 43 columns; E_stochastic after E_round,
  // n_closing_at_rest after n_k0_trigger); step 4 round 2: 45, the sampler counters appended
  enum LedgerCol : int {
    L_STEP = 0, L_TIME = 1, L_EDRAG = 2, L_EPAIRS = 3, L_EABS = 4, L_EEXP = 5, L_EROUND = 6, L_ESTOCH = 7,
    L_ELOST = 8, L_PDISC = 9, L_DINIT = 10, L_DOUT = 20, L_POUT = 21, L_RES = 22, L_NEVENTS = 23, L_NSPAWNED = 24,
    L_NDROPPED = 25, L_NBOUNDED = 27, L_NCLOSING = 29, L_NCAPPED = 30, L_NCAPEV = 31, L_NRNG = 36, L_NK0 = 37, L_NCLOSE0 = 38,
    L_SUMWK = 39, L_B = 40, L_NPM = 41, L_NPP = 42, L_NSATT = 43, L_NSOVF = 44
  };

  int cmd_ledger(const Args& a) {
    const auto   r   = read_last_ledger_row(str(a, "file"));
    const double tol = a.count("tol") ? num(a, "tol") : 1.0e-12;
    const double E   = std::abs(r.c[L_EDRAG]);
    std::printf("LEDGER_ROW step %.0f n_events %.0f n_spawned %.0f n_dropped %.0f n_closing %.0f n_capped %.0f "
                "n_capped_events %.0f n_rng %.0f npart_minus %.0f npart_plus %.0f E_drag %.17g E_pairs %.17g "
                "E_absorbed %.17g E_exported %.17g E_round %.17g E_stochastic %.17g E_lost_nopairs %.17g "
                "debt_outstanding %.17g residual %.6e B %.17g n_sampler_attempts %.0f n_sampler_overflow %.0f\n",
                r.c[L_STEP], r.c[L_NEVENTS], r.c[L_NSPAWNED], r.c[L_NDROPPED], r.c[L_NCLOSING], r.c[L_NCAPPED],
                r.c[L_NCAPEV], r.c[L_NRNG], r.c[L_NPM], r.c[L_NPP], r.c[L_EDRAG], r.c[L_EPAIRS], r.c[L_EABS],
                r.c[L_EEXP], r.c[L_EROUND], r.c[L_ESTOCH], r.c[L_ELOST], r.c[L_DOUT], r.c[L_RES], r.c[L_B],
                r.c[L_NSATT], r.c[L_NSOVF]);
    verdict(std::abs(r.c[L_RES]) <= tol * E, "ledger identity residual / E_drag (last row)",
            E > 0.0 ? std::abs(r.c[L_RES]) / E : std::abs(r.c[L_RES]), tol);
    if (a.count("expect_dropped")) {
      const bool want = num(a, "expect_dropped") > 0.5;
      verdict(want ? r.c[L_NDROPPED] > 0.0 : r.c[L_NDROPPED] == 0.0,
              want ? "ledger: n_dropped > 0" : "ledger: n_dropped = 0", r.c[L_NDROPPED], want ? 1.0 : 0.0);
    }
    if (a.count("expect_npart")) {
      const double n = num(a, "expect_npart");
      verdict(r.c[L_NPM] == n && r.c[L_NPP] == n, "ledger: npart of both pair species = " + std::to_string((long)n),
              r.c[L_NPM], n);
    }
    if (a.count("min_spawned")) {
      verdict(r.c[L_NSPAWNED] >= num(a, "min_spawned"), "ledger: n_spawned >= " + a.at("min_spawned"), r.c[L_NSPAWNED],
              num(a, "min_spawned"));
    }
    if (a.count("expect_rng")) {
      const double n = num(a, "expect_rng");
      verdict(r.c[L_NRNG] == n, "ledger: n_rng = " + a.at("expect_rng"), r.c[L_NRNG], n);
    }
    if (a.count("expect_events")) {
      const double n = num(a, "expect_events");
      verdict(r.c[L_NEVENTS] == n && r.c[L_NSPAWNED] == n, "ledger: n_events = n_spawned = " + a.at("expect_events"),
              r.c[L_NEVENTS], n);
    }
    if (a.count("expect_capped_events")) {
      const double n = num(a, "expect_capped_events");
      verdict(r.c[L_NCAPEV] == n, "ledger: n_capped_events = " + a.at("expect_capped_events"), r.c[L_NCAPEV], n);
    }
    if (a.count("min_sampler_overflow")) {
      verdict(r.c[L_NSOVF] >= num(a, "min_sampler_overflow"),
              "ledger: n_sampler_overflow >= " + a.at("min_sampler_overflow"), r.c[L_NSOVF],
              num(a, "min_sampler_overflow"));
      verdict(r.c[L_NSATT] >= r.c[L_NSOVF], "ledger: n_sampler_attempts >= n_sampler_overflow", r.c[L_NSATT],
              r.c[L_NSOVF]);
    }
    if (a.count("min_bounded")) {
      verdict(r.c[L_NBOUNDED] >= num(a, "min_bounded"), "ledger: n_bounded >= " + a.at("min_bounded"), r.c[L_NBOUNDED],
              num(a, "min_bounded"));
    }
    if (a.count("expect_bounded")) {
      const double n = num(a, "expect_bounded");
      verdict(r.c[L_NBOUNDED] == n, "ledger: n_bounded = " + a.at("expect_bounded"), r.c[L_NBOUNDED], n);
    }
    if (a.count("min_events")) {
      verdict(r.c[L_NEVENTS] >= num(a, "min_events"), "ledger: n_events >= " + a.at("min_events"), r.c[L_NEVENTS],
              num(a, "min_events"));
    }
    if (a.count("expect_nopairs") && num(a, "expect_nopairs") > 0.5) {
      verdict(r.c[L_NSPAWNED] == 0.0 && r.c[L_ELOST] == r.c[L_EDRAG], "ledger (none): no spawns and E_lost_nopairs = E_drag",
              r.c[L_ELOST] - r.c[L_EDRAG], 0.0);
    }
    return 0;
  }

  /* ---- step 3: abort message ---------------------------------------------- */
  int cmd_abort(const Args& a) {
    const auto lines = grep_lines(str(a, "log"), str(a, "key"));
    bool       found = false;
    long       step  = -1;
    for (const auto& l : lines) {
      const auto pos = l.find("at step ");
      if (pos != std::string::npos) {
        step  = std::atol(l.c_str() + pos + 8);
        found = true;
        std::printf("(diag) %s\n", l.substr(0, 400).c_str());
      }
    }
    verdict(found, "abort: the log names " + str(a, "key") + " with the step", static_cast<double>(step), 0.0);
    if (a.count("step_min") && a.count("step_max")) {
      verdict(found && step >= num(a, "step_min") && step <= num(a, "step_max"),
              "abort: the exhaustion step lies in [" + a.at("step_min") + ", " + a.at("step_max") + "]",
              static_cast<double>(step), num(a, "step_max"));
    }
    return 0;
  }

  /* ---- step 3: startup estimate ------------------------------------------- */
  double after(const std::string& l, const std::string& key) {
    const auto pos = l.find(key);
    if (pos == std::string::npos) {
      throw std::runtime_error("'" + key + "' not found in: " + l.substr(0, 200));
    }
    return std::atof(l.c_str() + pos + key.size());
  }

  int cmd_t12b(const Args& a) {
    const auto est = grep_lines(str(a, "log"), "capacity estimate (III.8)");
    if (est.empty()) {
      throw std::runtime_error("no capacity estimate line in the log");
    }
    const double sized = after(est.front(), "expected macro-particles per pair species = ");
    const double bare  = after(est.front(), "(bare: ");
    std::printf("T12B_BARE %.10g T12B_SIZED %.10g\n", bare, sized);
    if (a.count("measured") && num(a, "measured") > 0.5) {
      const auto led = grep_lines(str(a, "log"), "bh_proto: ledger species");
      if (led.empty()) {
        throw std::runtime_error("no ledger line in the log");
      }
      const double n = after(led.front(), "npart_minus ");
      std::printf("T12B_MEASURED %.0f\n", n);
      verdict(n >= 0.5 * bare && n <= 2.0 * bare,
              "T12b: measured spawn count within a factor 2 of the estimate (bare " + std::to_string(bare) + ")",
              n / bare, 2.0);
    }
    return 0;
  }

  /* ---- step 3: Gauss residual --------------------------------------------- */
  struct GaussRow {
    long        step;
    double      time;
    std::string stage;
    double      maxR, maxrho, maxE, ratio, sumR, sumrho, maxq, maxdR;
    int         argmax;
    double      nm, np, ndep { 0.0 };
  };

  std::vector<GaussRow> read_gauss(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot open " + path);
    }
    std::vector<GaussRow> rows;
    std::string           line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::istringstream ss(line);
      GaussRow           r;
      ss >> r.step >> r.time >> r.stage >> r.maxR >> r.maxrho >> r.maxE >> r.ratio >> r.argmax >> r.sumR >> r.sumrho >>
        r.maxq >> r.nm >> r.np >> r.maxdR >> r.ndep;
      rows.push_back(r);
    }
    return rows;
  }

  int cmd_gauss(const Args& a) {
    const auto test = read_gauss(str(a, "test"));
    const auto ctrl = read_gauss(str(a, "control"));
    auto max_ratio = [](const std::vector<GaussRow>& rows, const std::string& stage, std::size_t from, std::size_t to) {
      double m = 0.0;
      std::size_t k = 0;
      for (const auto& r : rows) {
        if (r.stage != stage) {
          continue;
        }
        if (k >= from && k < to) {
          m = std::max(m, r.ratio);
        }
        ++k;
      }
      return m;
    };
    auto count_stage = [](const std::vector<GaussRow>& rows, const std::string& stage) {
      std::size_t k = 0;
      for (const auto& r : rows) {
        k += (r.stage == stage);
      }
      return k;
    };
    const auto nt = count_stage(test, "step"), nc = count_stage(ctrl, "step");
    if (nt < 4 || nc < 4) {
      throw std::runtime_error("too few 'step' rows in the gauss files");
    }
    const double t_all = max_ratio(test, "step", 0, nt), c_all = max_ratio(ctrl, "step", 0, nc);
    // the control run's round-off level: the measured control residual; when that is exactly zero
    // (as happens in the double build) the a-priori accumulated round-off of the field update over
    // the run (n_steps x epsilon of the build's real_t) replaces it; --eps and --steps give the numbers
    const double eps_build = a.count("eps") ? num(a, "eps") : 0.0;
    const double n_steps   = a.count("steps") ? num(a, "steps") : 0.0;
    // the brief's criterion compares with the measured control run; only when that residual is
    // exactly zero (degenerate) is the a-priori floor used instead
    const double level     = c_all > 0.0 ? c_all : n_steps * eps_build;
    std::printf("(diag) control level = %s (measured control %.6e, a-priori floor n_steps x eps = %.0f x %.3e = %.6e); "
                "test / measured control = %s\n",
                c_all > 0.0 ? "the measured control residual" : "the a-priori floor (measured control exactly zero)", c_all,
                n_steps, eps_build, n_steps * eps_build, c_all > 0.0 ? std::to_string(t_all / c_all).c_str() : "undefined");
    for (const auto& r : test) {
      if (r.stage != "step") {
        std::printf("(diag) test %s: max|R| %.6e, max|c rho| %.6e, max|R - R0| %.6e, ratio %.6e, npart e- %.0f e+ %.0f\n",
                    r.stage.c_str(), r.maxR, r.maxrho, r.maxdR, r.ratio, r.nm, r.np);
      }
    }
    std::printf("(diag) test: %zu step rows, max over rows of max|R - R0| / max|c rho| = %.6e (last row %.6e, pairs e- %.0f); "
                "control: %zu rows, %.6e (last %.6e)\n",
                nt, t_all, test.back().ratio, test.back().nm, nc, c_all, ctrl.back().ratio);
    verdict(t_all <= 2.0 * level && level > 0.0,
            "T13: max|R - R0| / max|c rho| of the pair run within a factor 2 of the control run's round-off level "
            "(test / level)",
            level > 0.0 ? t_all / level : t_all, 2.0);
    // no growth trend: the last quarter of the rows against the first quarter
    const std::size_t q = std::max<std::size_t>(1, nt / 4);
    const double      t_first = max_ratio(test, "step", 0, q), t_last = max_ratio(test, "step", nt - q, nt);
    verdict(t_last <= 2.0 * std::max(t_first, level), "T13: no growth trend (last quarter / max(first quarter, level))",
            t_last / std::max(t_first, level), 2.0);
    // the births right after the first spawning launch do not change the residual beyond the control
    // level (R6: the samples bracket the launch that spawned the first pair; the depositing count rises)
    const GaussRow* before = nullptr;
    const GaussRow* after  = nullptr;
    for (const auto& r : test) {
      if (r.stage == "before_first_births") {
        before = &r;
      }
      if (r.stage == "after_first_births") {
        after = &r;
      }
    }
    verdict(before != nullptr && after != nullptr && after->step == before->step,
            "T13: the before/after samples bracket the first spawning launch (same step)",
            before && after ? static_cast<double>(after->step) : -1.0, before ? static_cast<double>(before->step) : -1.0);
    if (before && after) {
      verdict(after->ndep > before->ndep, "T13: the depositing particle count increased at the first births",
              after->ndep - before->ndep, 1.0);
      verdict(after->ratio <= 2.0 * level, "T13: residual right after the first births within a factor 2 of the control level",
              level > 0.0 ? after->ratio / level : after->ratio, 2.0);
    }
    return 0;
  }

  /* ---- step 3: determinism ------------------------------------------------ */
  int cmd_det(const Args& a) {
    const auto la = grep_lines(str(a, "a"), "bh_proto: checksum");
    const auto lb = grep_lines(str(a, "b"), "bh_proto: checksum");
    const bool want_same = str(a, "expect") == "same";
    bool       same      = la.size() == lb.size() && !la.empty();
    for (std::size_t i = 0; same && i < la.size(); ++i) {
      same = la[i] == lb[i];
    }
    for (const auto& l : la) {
      std::printf("(diag) a: %s\n", l.c_str());
    }
    for (const auto& l : lb) {
      std::printf("(diag) b: %s\n", l.c_str());
    }
    verdict(want_same ? same : !same,
            want_same ? "determinism: every checksum line identical (same seed)"
                      : "determinism: the checksum lines differ (different seeds)",
            same ? 1.0 : 0.0, want_same ? 1.0 : 0.0);
    return 0;
  }

  /* ---- step 3 round 2: placement comparison of two ledger files ---------- */
  std::vector<std::vector<double>> read_ledger_rows(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot open " + path);
    }
    std::vector<std::vector<double>> rows;
    std::string                      line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::vector<double> r;
      std::istringstream  ss(line);
      double              v;
      while (ss >> v) {
        r.push_back(v);
      }
      if (r.size() != LEDGER_COLS) {
        throw std::runtime_error("ledger row has " + std::to_string(r.size()) + " columns, expected " +
                                 std::to_string(LEDGER_COLS) + " in " + path);
      }
      rows.push_back(r);
    }
    if (rows.empty()) {
      throw std::runtime_error("no data row in " + path);
    }
    return rows;
  }

  int cmd_cmp(const Args& a) {
    const auto   A   = read_ledger_rows(str(a, "a"));
    const auto   B   = read_ledger_rows(str(a, "b"));
    const double tol = a.count("tol") ? num(a, "tol") : 1.0e-12;
    verdict(A.size() == B.size(), "placement: the two ledger files have the same number of rows",
            static_cast<double>(A.size()), static_cast<double>(B.size()));
    if (A.size() != B.size()) {
      return 0;
    }
    auto is_counter = [](std::size_t c) {
      return c == L_STEP || (c >= L_NEVENTS && c <= L_NCLOSE0) || c == L_NPM || c == L_NPP || c == L_NSATT ||
             c == L_NSOVF;
    };
    double      worst = 0.0, worst_cnt = 0.0;
    std::size_t worst_row = 0, worst_col = 0;
    for (std::size_t r = 0; r < A.size(); ++r) {
      for (std::size_t c = 0; c < LEDGER_COLS; ++c) {
        const double x = A[r][c], y = B[r][c];
        if (is_counter(c)) {
          worst_cnt = std::max(worst_cnt, std::abs(x - y));
        } else {
          const double d = std::abs(x - y) / std::max({ 1.0, std::abs(x), std::abs(y) });
          if (d > worst) {
            worst     = d;
            worst_row = r;
            worst_col = c;
          }
        }
      }
    }
    std::printf("CMP rows %zu largest relative difference %.6e at row %zu column %zu (a %.17g, b %.17g); "
                "n_spawned a %.0f b %.0f; E_pairs a %.17g b %.17g\n",
                A.size(), worst, worst_row, worst_col, A[worst_row][worst_col], B[worst_row][worst_col],
                A.back()[L_NSPAWNED], B.back()[L_NSPAWNED], A.back()[L_EPAIRS], B.back()[L_EPAIRS]);
    verdict(worst_cnt == 0.0, "placement: every counter column (step, n_*, npart) identical in every row", worst_cnt,
            0.0);
    verdict(worst <= tol, "placement: every real column identical to round-off, |a - b| / max(1, |a|, |b|)", worst,
            tol);
    verdict(A.back()[L_NSPAWNED] > 0.0, "placement: pairs were spawned (the comparison is not vacuous)",
            A.back()[L_NSPAWNED], 1.0);
    return 0;
  }

  /* ---- step 3: exported debt ---------------------------------------------- */
  int cmd_texp(const Args& a) {
    const auto lines = grep_lines(str(a, "log"), "bh_proto: exported species");
    if (lines.empty()) {
      throw std::runtime_error("no 'exported species' line in the log");
    }
    const auto&  l       = lines.front();
    const double eh      = after(l, "E_exported_host ");
    const double e       = after(l, "E_exported ");
    const double ph      = after(l, "pending_discarded_host ");
    const double pd      = after(l, "pending_discarded ");
    const double n_dead  = after(l, "n_dead_with_payload ");
    std::printf("(diag) %s\n", l.substr(0, 400).c_str());
    verdict(eh > 0.0 && std::abs(e - eh) <= 1.0e-12 * eh, "T-exp: E_exported = host sum w r over the dead ions, relative",
            eh > 0.0 ? std::abs(e - eh) / eh : 1.0, 1.0e-12);
    // the pending request may carry a signed rounding residual, so the scale is |ph|
    const double scale_pd = std::max(std::abs(ph), 1.0e-300);
    verdict(std::abs(pd - ph) <= 1.0e-12 * scale_pd,
            "T-exp: pending_discarded = host sum w delta (signed), |difference| / |host sum|",
            std::abs(pd - ph) / scale_pd, 1.0e-12);
    verdict(n_dead > 0.0, "T-exp: dead ions with a payload were seen by the hook before the sort", n_dead, 1.0);
    const auto led = grep_lines(str(a, "log"), "bh_proto: ledger species");
    if (!led.empty()) {
      const double res = after(led.front(), "residual ");
      const double E   = after(led.front(), "E_drag ");
      verdict(std::abs(res) <= 1.0e-12 * E, "T-exp: ledger identity with the exported term, residual / E_drag",
              std::abs(res) / E, 1.0e-12);
    }
    return 0;
  }

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bh_check <subcommand> [--key value ...]\n");
    return 2;
  }
  try {
    const std::string cmd = argv[1];
    const auto        a   = parse_args(argc, argv, 2);
    if (cmd == "tau0") {
      cmd_tau0(a);
    } else if (cmd == "t5b") {
      cmd_t5b(a, true);
    } else if (cmd == "t5b-off") {
      cmd_t5b(a, false);
    } else if (cmd == "t7a") {
      cmd_t7a(a);
    } else if (cmd == "t7b") {
      cmd_t7b(a);
    } else if (cmd == "kick") {
      cmd_kick(a);
    } else if (cmd == "order") {
      cmd_order(a);
    } else if (cmd == "t8") {
      cmd_t8(a);
    } else if (cmd == "reject") {
      cmd_reject(a);
    } else if (cmd == "ledger") {
      cmd_ledger(a);
    } else if (cmd == "abort") {
      cmd_abort(a);
    } else if (cmd == "t12b") {
      cmd_t12b(a);
    } else if (cmd == "gauss") {
      cmd_gauss(a);
    } else if (cmd == "det") {
      cmd_det(a);
    } else if (cmd == "texp") {
      cmd_texp(a);
    } else if (cmd == "cmp") {
      cmd_cmp(a);
    } else {
      std::fprintf(stderr, "bh_check: unknown subcommand '%s'\n", cmd.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_check: %s\n", e.what());
    return 2;
  }
  return n_fail > 0 ? 1 : 0;
}
