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
      if (r.cols.size() != 15) {
        throw std::runtime_error(path + ": expected 15 columns, got " + std::to_string(r.cols.size()));
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
        for (std::size_t c = 2; c + 1 < r.cols.size(); ++c) { // all but step, time, sum_wK
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
