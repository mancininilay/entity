/**
 * @file tools/bh_sampler_dump.cpp
 * @brief Command-line producer of the step-4 fixture files (contract revision
 *        1.0 section 4), producer `entity-cpp`
 *
 * Usage:
 *   bh_sampler_dump qk_table --out FILE [--status STATUS] [--threads N]
 *   bh_sampler_dump moments  --bath NAME --out FILE [--seed S] [--n N]
 *                            [--n-weighted-attempts N] [--status STATUS] [--threads N]
 *   bh_sampler_dump hist     --bath NAME --gamma G --out FILE [--seed S] [--n N]
 *                            [--n-weighted-attempts N] [--status STATUS]
 *   bh_sampler_dump ka2008   --bath NAME --gamma G --out FILE [--status STATUS]
 *
 * `qk_table` (section 4 item 1): the eps' nodes of the step-1 grid (64 per
 * decade in log10(eps' - 2) from 1e-4 while eps' <= 1000, then eps' = 1000),
 * q_k from the rest-frame machinery, q_k from Chodorowski eq. 3.7, <E'_->,
 * <p' cos theta'>, sigma~ from the integral of the energy law (the A1 value)
 * and Maximon's sigma~. Deterministic quadrature (bh_restframe.h).
 *
 * `moments` (item 2): at every Gamma_p node of the bath's rate table, N
 * unweighted draws (default 1e6): <D> with its standard error, <D^2>, <E->,
 * <E+>, <cos theta-> (lab, to u^), <eps>, min D, max D; from the weighted
 * chain, with a budget of `--n-weighted-attempts` trials per node (default
 * 1e6; the acceptance efficiency falls to 1e-3 at the upper table edge, so
 * 1e6 accepted draws per node are not affordable there, see the report):
 * E_p[1/D] (I/J) with its standard error, the acceptance efficiency, 1/min D.
 * Seeds: SplitMix64(seed + i) unweighted and SplitMix64(seed + 500000 + i)
 * weighted at node i.
 *
 * `hist` (item 3): at one Gamma_p, N draws of each chain (the weighted chain
 * within the attempts budget): 200 log-spaced bins in E- and in E+ over the
 * kinematic lab range [1, Gamma (E'_max + beta p'_max)], 200 linear bins in
 * cos theta- and cos theta+ (lab, to u^), 200 linear bins in D over
 * [D_min, D_max] of the bath at this Gamma_p; counts of the unweighted chain
 * and 1/D-weighted, normalised weights of the weighted chain.
 *
 * `ka2008` (item 4): the lab e- spectrum of the UR anchor (bh_lab_spectrum.h,
 * Kelner and Aharonian 2008 eqs. 58 to 64 as the exact boost of the
 * rest-frame laws, by quadrature) as a CDF on the 200 log-spaced E- bins of
 * the `hist` files at the same Gamma_p (mono baths only).
 *
 * Bath names: mono05, mono03, powerlaw, blackbody, mono05_mu100 (the five
 * step-1 acceptance baths). Option errors exit with 2, runtime errors with 1.
 */

#include "archetypes/hadronic/bh_restframe.h"
#include "archetypes/hadronic/bh_lab_spectrum.h"
#include "archetypes/hadronic/bh_sampler_host.h"
#include "archetypes/hadronic/bh_sampler_io.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  struct Options {
    std::string   command;
    std::string   out;
    std::string   bath;
    std::string   status { "cpp_sampler_candidate" };
    double        gamma { 0.0 };
    std::uint64_t seed { 20261026ULL };
    std::size_t   n { 1000000 };
    std::size_t   n_weighted_attempts { 1000000 };
    unsigned      threads { 0 };
  };

  std::size_t parse_count(const std::string& key, const std::string& val) {
    double v = 0.0;
    if (!arch::bh::parse_number_strict(val, v) || !(v >= 1.0) || v != std::floor(v) || v > 1.0e12) {
      throw std::invalid_argument(key + " must be a positive integer");
    }
    return static_cast<std::size_t>(v);
  }

  Options parse(const std::vector<std::string>& t) {
    Options o;
    if (t.empty()) {
      throw std::invalid_argument("missing command (qk_table | moments | hist)");
    }
    o.command = t[0];
    if (o.command != "qk_table" && o.command != "moments" && o.command != "hist" && o.command != "ka2008") {
      throw std::invalid_argument("unknown command '" + o.command + "'");
    }
    for (std::size_t i = 1; i < t.size(); ++i) {
      const std::string& key = t[i];
      if (i + 1 >= t.size()) {
        throw std::invalid_argument("option " + key + " needs a value");
      }
      const std::string& val = t[++i];
      if (key == "--out") {
        o.out = val;
      } else if (key == "--status") {
        o.status = val;
      } else if (key == "--bath") {
        o.bath = val;
      } else if (key == "--gamma") {
        double v = 0.0;
        if (!arch::bh::parse_number_strict(val, v) || !(v > 1.0)) {
          throw std::invalid_argument("--gamma must be a number > 1");
        }
        o.gamma = v;
      } else if (key == "--seed") {
        o.seed = static_cast<std::uint64_t>(parse_count(key, val));
      } else if (key == "--n") {
        o.n = parse_count(key, val);
      } else if (key == "--n-weighted-attempts") {
        o.n_weighted_attempts = parse_count(key, val);
      } else if (key == "--threads") {
        o.threads = static_cast<unsigned>(parse_count(key, val));
      } else {
        throw std::invalid_argument("unknown option '" + key + "'");
      }
    }
    if (o.out.empty()) {
      throw std::invalid_argument("--out is required");
    }
    if (o.command != "qk_table" && o.bath.empty()) {
      throw std::invalid_argument("--bath is required");
    }
    if ((o.command == "hist" || o.command == "ka2008") && !(o.gamma > 1.0)) {
      throw std::invalid_argument("--gamma is required");
    }
    return o;
  }

  std::vector<std::pair<std::string, std::string>> common_header(const std::string& format, const Options& o) {
    using arch::bh::json_string;
    return {
      { "format", json_string(format) },
      { "format_revision", "1" },
      { "producer", json_string("entity-cpp") },
      { "version", json_string(ENTITY_GIT_HASH) },
      { "status", json_string(o.status) },
    };
  }

  std::string bath_json(const arch::bh::PhotonBath& bath) {
    std::string b = "{\"type\": " + arch::bh::json_string(bath.type());
    for (const auto& [k, v] : bath.parameters()) {
      b += ", " + arch::bh::json_string(k) + ": " + v;
    }
    b += "}";
    return b;
  }

  std::string tables_json(const arch::bh::rf::RestFrameTables& T, const arch::bh::rf::SamplerCumulatives& C,
                          const arch::bh::rf::BathEpsTables& B) {
    return "{\"eps_prime_nodes\": " + std::to_string(T.n_k()) +
           ", \"eps_prime_nodes_per_decade\": " + std::to_string(T.nodes_per_decade) +
           ", \"n_s\": " + std::to_string(T.n_s) + ", \"n_tau\": " + std::to_string(T.n_tau) +
           ", \"energy_variable\": \"E' = 1 + (eps' - 2) sin^2(pi s/2)\"" +
           ", \"angle_variable\": \"t = ln(1 - cos theta' + (E' - p')/p'), tau = (t - ln w)/(ln(2 + w) - ln w)\"" +
           ", \"interpolation\": \"mixture of the bracketing nodes' CDFs, linear in ln(eps' - 2) and in s; "
           "piecewise-constant density inside a cell\"" +
           ", \"cumulatives_h\": " + arch::bh::json_number(C.h()) + ", \"cumulatives_nodes\": " +
           std::to_string(C.size()) + ", \"n_eps\": " + std::to_string(B.mono ? 0 : B.n_eps) +
           ", \"p_closed_min\": " + arch::bh::json_number(T.p_closed_min) + "}";
  }

  int run_qk_table(const Options& o) {
    using namespace arch::bh;
    const rf::RestFrameQuadrature q;
    const std::vector<double>     grid = rf::qk_grid();
    std::vector<rf::Moments>      mom(grid.size());
    rf::parallel_for_index(
      grid.size(), [&](std::size_t i) { mom[i] = rf::moments(grid[i], q); }, o.threads);
    FixtureTable t;
    t.header = common_header("bh-qk-table", o);
    t.header.push_back({ "grid",
                         "{\"variable\": \"log10(eps_prime - 2)\", \"nodes_per_decade\": 64, "
                         "\"delta_min\": 1e-4, \"eps_prime_max\": 1000, \"endpoint_appended\": true}" });
    t.header.push_back(
      { "laws",
        "{\"energy\": \"Blumenthal 1970 eq. 10 (= MOK 1969 3D-1000), Z = 1 unscreened\", "
        "\"angular\": \"Sauter-Gluckstern-Hull, MOK 1969 3D-2000, Z = 1 unscreened; signs and "
        "normalisation fixed against the numerical integral of Heitler 1954 eq. 26.6\", "
        "\"q_k\": \"eps_prime - 2 <p cos theta>_h\", \"normalisation\": \"the laws' own integrals (sigma_from_g, "
        "int h); Maximon's sigma~ reported in sigma_maximon\"}" });
    t.header.push_back({ "quadrature",
                         "{\"gl_order\": " + std::to_string(q.gl_order) +
                           ", \"energy_panels\": " + std::to_string(q.energy_panels) +
                           ", \"energy_variable\": \"E = 1 + (eps_prime - 2) sin^2(pi s / 2)\"" +
                           ", \"h_t_angle\": " + json_number(q.h_t_angle) +
                           ", \"angle_variable\": \"t = ln(1 - cos theta + (E - p)/p)\", \"columns_format\": \"%.12e\"}" });
    t.header.push_back({ "node_count", std::to_string(grid.size()) });
    t.columns = { "eps_prime", "q_k", "q_k_eq37", "mean_E_minus", "mean_p_cos_theta", "sigma_from_g",
                  "sigma_maximon" };
    for (std::size_t i = 0; i < grid.size(); ++i) {
      const auto& m = mom[i];
      t.rows.push_back(
        { grid[i], m.q_k, q_k(grid[i]), m.mean_E, m.mean_pcos, m.sigma_g, m.sigma_maximon });
    }
    write_fixture_file(o.out, t);
    const FixtureContents c = read_fixture_file(o.out);
    std::printf("bh_sampler_dump qk_table: %zu nodes written to %s (data_sha256 %s)\n", c.rows.size(), o.out.c_str(),
                c.data_sha256.c_str());
    return 0;
  }

  struct Setup {
    arch::bh::rf::CanonicalBath                         cb;
    std::shared_ptr<const arch::bh::PhotonBath>         bath;
    std::shared_ptr<arch::bh::BetheHeitlerRates>        rates;
    arch::bh::HadronicTablesHost                        H;
    std::shared_ptr<const arch::bh::rf::RestFrameTables>    T;
    std::shared_ptr<const arch::bh::rf::SamplerCumulatives> cum;
    std::shared_ptr<arch::bh::rf::BathEpsTables>            B;
    std::shared_ptr<arch::bh::rf::HostSampler>              S;
  };

  Setup make_setup(const Options& o, const std::vector<double>* extra_gammas) {
    using namespace arch::bh;
    Setup s;
    s.cb    = rf::canonical_bath(o.bath);
    s.bath  = make_bath(s.cb.spec);
    s.rates = std::make_shared<BetheHeitlerRates>(s.bath);
    s.H     = build_tables(*s.rates, s.cb.mu_r);
    s.T     = std::make_shared<const rf::RestFrameTables>(rf::RestFrameTables::build(16, 256, 256, 8, o.threads));
    s.cum   = std::make_shared<const rf::SamplerCumulatives>();
    if (extra_gammas) {
      // the eps tables at the requested Gamma (and a neighbour) only
      s.B = std::make_shared<rf::BathEpsTables>(rf::BathEpsTables::build_at(*s.bath, *extra_gammas, *s.cum));
    } else {
      s.B = std::make_shared<rf::BathEpsTables>(rf::BathEpsTables::build(*s.bath, s.H, *s.cum, 2048, 8, o.threads));
    }
    s.S = std::make_shared<rf::HostSampler>(s.bath, s.H, s.T, s.cum, s.B);
    return s;
  }

  int run_moments(const Options& o) {
    using namespace arch::bh;
    const auto t0 = std::chrono::steady_clock::now();
    Setup      s  = make_setup(o, nullptr);
    const std::size_t ng = s.H.size();
    struct Row {
      double Dmean, Dse, D2, Em, Ep, cosm, eps, Dmin, Dmax;
      double Einv, Einv_se, eff, invminD;
      std::size_t n_acc, n_att;
    };
    std::vector<Row> rows(ng);
    rf::parallel_for_index(
      ng,
      [&](std::size_t i) {
        const double g = s.H.gamma[i];
        Row          r {};
        {
          rf::SplitMix64 rng(o.seed + i);
          double sD = 0, sD2 = 0, sEm = 0, sEp = 0, sc = 0, se = 0, mn = 1e300, mx = -1e300;
          for (std::size_t n = 0; n < o.n; ++n) {
            const auto e = s.S->draw(g, false, rng);
            sD += e.D;
            sD2 += e.D * e.D;
            sEm += e.lab_m.E;
            sEp += e.lab_p.E;
            const double pm = std::sqrt(e.lab_m.p.x * e.lab_m.p.x + e.lab_m.p.y * e.lab_m.p.y + e.lab_m.p.z * e.lab_m.p.z);
            sc += e.lab_m.p.z / pm;
            se += e.eps;
            mn = std::min(mn, e.D);
            mx = std::max(mx, e.D);
          }
          const double N = static_cast<double>(o.n);
          r.Dmean        = sD / N;
          r.Dse          = std::sqrt(std::max(0.0, sD2 / N - r.Dmean * r.Dmean) / N);
          r.D2           = sD2 / N;
          r.Em           = sEm / N;
          r.Ep           = sEp / N;
          r.cosm         = sc / N;
          r.eps          = se / N;
          r.Dmin         = mn;
          r.Dmax         = mx;
        }
        {
          rf::SplitMix64 rng(o.seed + 500000 + i);
          double         s1 = 0, s2 = 0, mn = 1e300;
          std::size_t    n = 0, att = 0;
          while (att < o.n_weighted_attempts && n < o.n) {
            const auto e = s.S->draw(g, true, rng);
            att += static_cast<std::size_t>(e.attempts);
            const double y = 1.0 / e.D;
            s1 += y;
            s2 += y * y;
            mn = std::min(mn, e.D);
            ++n;
          }
          const double m = s1 / n;
          r.Einv         = m * s.H.Dmean[i];
          r.Einv_se      = std::sqrt(std::max(0.0, s2 / n - m * m) / n) * s.H.Dmean[i];
          r.eff          = static_cast<double>(n) / att;
          r.invminD      = 1.0 / mn;
          r.n_acc        = n;
          r.n_att        = att;
        }
        rows[i] = r;
      },
      o.threads);
    FixtureTable t;
    t.header = common_header("bh-sampler-moments", o);
    t.header.push_back({ "bath", json_string(o.bath) });
    t.header.push_back({ "photon_field", bath_json(*s.bath) });
    t.header.push_back({ "physical_mass_ratio", json_number(s.cb.mu_r) });
    t.header.push_back({ "gamma_th", json_number(s.H.gamma_th) });
    t.header.push_back({ "seed", std::to_string(o.seed) });
    t.header.push_back({ "n_unweighted", std::to_string(o.n) });
    t.header.push_back({ "n_weighted_attempts_budget", std::to_string(o.n_weighted_attempts) });
    t.header.push_back({ "rng", json_string("SplitMix64(seed + i) unweighted, SplitMix64(seed + 500000 + i) weighted, "
                                            "node i; uniform = (bits >> 11 + 0.5) 2^-53") });
    t.header.push_back({ "chains",
                         json_string("unweighted: eps ~ n/eps^2 Phi(eps'_hi), eps'|eps ~ eps' sigma~ on [2, eps'_hi], "
                                     "E'- ~ g, angles ~ h independently, rotation and boost along u^ = z; weighted: "
                                     "eps ~ n/eps^2 F(eps'_hi), eps'|eps ~ eps' sigma~ D_max, unweighted conditionals, "
                                     "accept with D/D_max") });
    t.header.push_back({ "tables", tables_json(*s.T, *s.cum, *s.B) });
    t.header.push_back({ "Dmean_table_definition", json_string("I/J of the step-1 rate table at the node") });
    t.header.push_back({ "node_count", std::to_string(ng) });
    t.columns = { "gamma_p", "beta_p", "J", "I", "Dmean_table", "n_unweighted", "Dmean_mc", "Dmean_se", "D2mean",
                  "Emean_minus", "Emean_plus", "cos_minus_lab_mean", "eps_mean_mc", "D_min", "D_max",
                  "n_weighted_accepted", "n_weighted_attempts", "Einv_Dmean", "Einv_Dmean_se", "efficiency",
                  "inv_min_D_weighted" };
    for (std::size_t i = 0; i < ng; ++i) {
      const auto& r = rows[i];
      t.rows.push_back({ s.H.gamma[i], s.H.beta[i], s.H.J[i], s.H.I[i], s.H.Dmean[i], static_cast<double>(o.n),
                         r.Dmean, r.Dse, r.D2, r.Em, r.Ep, r.cosm, r.eps, r.Dmin, r.Dmax,
                         static_cast<double>(r.n_acc), static_cast<double>(r.n_att), r.Einv, r.Einv_se, r.eff,
                         r.invminD });
    }
    write_fixture_file(o.out, t);
    const FixtureContents c = read_fixture_file(o.out);
    std::printf("bh_sampler_dump moments %s: %zu nodes written to %s in %.0f s (data_sha256 %s)\n", o.bath.c_str(),
                c.rows.size(), o.out.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), c.data_sha256.c_str());
    return 0;
  }

  int run_ka2008(const Options& o) {
    using namespace arch::bh;
    const auto t0 = std::chrono::steady_clock::now();
    const auto cb = rf::canonical_bath(o.bath);
    if (cb.spec.type != "mono") {
      throw std::runtime_error("ka2008: mono baths only");
    }
    const double g = o.gamma, eps0 = cb.spec.eps0;
    double       E_lo, E_hi;
    rf::lab_energy_range(g, eps0, E_lo, E_hi);
    const int           NB = 200;
    std::vector<double> edges(NB + 1);
    for (int i = 0; i <= NB; ++i) {
      edges[i] = E_lo * std::pow(E_hi / E_lo, static_cast<double>(i) / NB);
    }
    const rf::LabSpectrumQuadrature q;
    double                          mean = 0.0, mean_r = 0.0;
    const std::vector<double>       F  = rf::lab_electron_cdf(g, eps0, edges, q, &mean);
    const std::vector<double>       Fr = rf::lab_electron_cdf(g, eps0, edges, q.refined(), &mean_r);
    double                          conv = 0.0;
    for (std::size_t i = 0; i < F.size(); ++i) {
      conv = std::max(conv, std::abs(F[i] - Fr[i]));
    }
    const double gb = std::sqrt((g - 1.0) * (g + 1.0));
    FixtureTable t;
    t.header = common_header("bh-ka2008-cdf", o);
    t.header.push_back({ "bath", json_string(o.bath) });
    t.header.push_back({ "photon_field", "{\"type\": \"mono\", \"eps0\": " + json_number(eps0) + "}" });
    t.header.push_back({ "physical_mass_ratio", json_number(cb.mu_r) });
    t.header.push_back({ "gamma_p", json_number(g) });
    t.header.push_back({ "beta_p", json_number(gb / g) });
    t.header.push_back({ "eps_prime_hi", json_number(eps0 * (g + gb)) });
    t.header.push_back({ "applicability_eps_prime_hi_gamma_over_mu_r", json_number(eps0 * (g + gb) * g / cb.mu_r) });
    t.header.push_back(
      { "law", json_string("Kelner and Aharonian 2008 eqs. 58-64 (equation numbers from the brief): the "
                           "Sauter-Gluckstern-Hull single-lepton law (Blumenthal 1970 eq. 11) in the proton rest "
                           "frame, photon energy eps' = Gamma eps (1 - beta cos theta), lepton boosted with the exact "
                           "(Gamma, beta); CDF of the lab e- energy by quadrature (bh_lab_spectrum.h)") });
    t.header.push_back({ "quadrature",
                         "{\"gl_order\": " + std::to_string(q.gl_order) + ", \"h_t_k\": " + json_number(q.h_t_k) +
                           ", \"s_panels\": " + std::to_string(q.s_panels) + ", \"h_t_angle\": " +
                           json_number(q.h_t_angle) + ", \"delta_min\": " + json_number(q.delta_min) +
                           ", \"max_cdf_change_under_refinement\": " + json_number(conv) + "}" });
    t.header.push_back({ "mean_E_minus_lab", json_number(mean) });
    t.header.push_back({ "mean_E_minus_lab_refined", json_number(mean_r) });
    t.header.push_back({ "bins", "{\"E\": {\"n\": 200, \"spacing\": \"log\", \"lo\": " + json_number(E_lo) +
                                   ", \"hi\": " + json_number(E_hi) + "}}" });
    t.header.push_back({ "node_count", std::to_string(NB) });
    t.columns = { "E_minus_lo", "E_minus_hi", "cdf_at_hi", "probability_in_bin" };
    for (int i = 0; i < NB; ++i) {
      t.rows.push_back({ edges[i], edges[i + 1], F[i + 1], F[i + 1] - F[i] });
    }
    write_fixture_file(o.out, t);
    const FixtureContents c = read_fixture_file(o.out);
    std::printf("bh_sampler_dump ka2008 %s Gamma %g: %zu bins written to %s in %.0f s; <E-> = %.6f, refinement change "
                "%.2e (data_sha256 %s)\n",
                o.bath.c_str(), g, c.rows.size(), o.out.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), mean, conv,
                c.data_sha256.c_str());
    return 0;
  }

  int run_hist(const Options& o) {
    using namespace arch::bh;
    const auto          t0 = std::chrono::steady_clock::now();
    const double        g  = o.gamma;
    std::vector<double> gammas { g, g * (1.0 + 1.0e-6) };
    Setup               s = make_setup(o, &gammas);
    if (!(g > s.H.gamma_th)) {
      throw std::runtime_error("hist: Gamma_p below the bath's threshold " + std::to_string(s.H.gamma_th));
    }
    const double gb = std::sqrt((g - 1.0) * (g + 1.0)), beta = gb / g, A = g + gb;
    const double eps_max = s.bath->eps_max();
    const double khi     = eps_max * A; // eps'_hi at eps_max
    if (khi > constants::K_MAX) {
      throw std::runtime_error("hist: eps'_hi = " + std::to_string(khi) + " exceeds the kernel cap 1000");
    }
    // kinematic ranges
    const double Epmax = khi - 1.0, ppmax = rf::momentum(Epmax);
    const double E_lo = 1.0, E_hi = g * Epmax + gb * ppmax;
    // D range over the bath: scan eps and eps'
    double Dlo = 1.0e300, Dhi = -1.0e300;
    {
      const double eps_min = s.bath->is_mono() ? eps_max : std::max(s.bath->eps_min(), 2.0 / A);
      for (int i = 0; i <= 2000; ++i) {
        const double eps = eps_min + (eps_max - eps_min) * i / 2000.0;
        const double kh  = eps * A;
        if (!(kh > 2.0)) {
          continue;
        }
        for (int j = 0; j <= 2000; ++j) {
          const double k = 2.0 + (kh - 2.0) * j / 2000.0;
          double       a, b;
          rf::HostSampler::bounds(g, gb, eps, k, a, b);
          Dlo = std::min(Dlo, a);
          Dhi = std::max(Dhi, b);
        }
      }
    }
    const int           NB = 200;
    std::vector<double> edgesE(NB + 1), edgesC(NB + 1), edgesD(NB + 1);
    for (int i = 0; i <= NB; ++i) {
      edgesE[i] = E_lo * std::pow(E_hi / E_lo, static_cast<double>(i) / NB);
      edgesC[i] = -1.0 + 2.0 * i / NB;
      edgesD[i] = Dlo + (Dhi - Dlo) * i / NB;
    }
    auto bin_log = [&](double v) {
      const int i = static_cast<int>(std::floor(std::log(v / E_lo) / std::log(E_hi / E_lo) * NB));
      return std::min(NB - 1, std::max(0, i));
    };
    auto bin_lin = [&](double v, double lo, double hi) {
      const int i = static_cast<int>(std::floor((v - lo) / (hi - lo) * NB));
      return std::min(NB - 1, std::max(0, i));
    };
    std::vector<double> nEm(NB, 0), nEp(NB, 0), nCm(NB, 0), nCp(NB, 0), nD(NB, 0);
    std::vector<double> wEm(NB, 0), wEp(NB, 0), wCm(NB, 0), wCp(NB, 0), wD(NB, 0);
    std::size_t         n_acc = 0, n_att = 0;
    {
      rf::SplitMix64 rng(o.seed);
      for (std::size_t n = 0; n < o.n; ++n) {
        const auto e = s.S->draw(g, false, rng);
        auto       cosl = [](const rf::FourMomentum& q) {
          return q.p.z / std::sqrt(q.p.x * q.p.x + q.p.y * q.p.y + q.p.z * q.p.z);
        };
        nEm[bin_log(e.lab_m.E)] += 1.0;
        nEp[bin_log(e.lab_p.E)] += 1.0;
        nCm[bin_lin(cosl(e.lab_m), -1.0, 1.0)] += 1.0;
        nCp[bin_lin(cosl(e.lab_p), -1.0, 1.0)] += 1.0;
        nD[bin_lin(e.D, Dlo, Dhi)] += 1.0;
      }
    }
    {
      rf::SplitMix64 rng(o.seed + 500000);
      double         wsum = 0.0;
      while (n_att < o.n_weighted_attempts && n_acc < o.n) {
        const auto e = s.S->draw(g, true, rng);
        n_att += static_cast<std::size_t>(e.attempts);
        ++n_acc;
        const double w = 1.0 / e.D;
        wsum += w;
        auto cosl = [](const rf::FourMomentum& q) {
          return q.p.z / std::sqrt(q.p.x * q.p.x + q.p.y * q.p.y + q.p.z * q.p.z);
        };
        wEm[bin_log(e.lab_m.E)] += w;
        wEp[bin_log(e.lab_p.E)] += w;
        wCm[bin_lin(cosl(e.lab_m), -1.0, 1.0)] += w;
        wCp[bin_lin(cosl(e.lab_p), -1.0, 1.0)] += w;
        wD[bin_lin(e.D, Dlo, Dhi)] += w;
      }
      for (int i = 0; i < NB; ++i) {
        wEm[i] /= wsum;
        wEp[i] /= wsum;
        wCm[i] /= wsum;
        wCp[i] /= wsum;
        wD[i] /= wsum;
      }
    }
    FixtureTable t;
    t.header = common_header("bh-sampler-hist", o);
    t.header.push_back({ "bath", json_string(o.bath) });
    t.header.push_back({ "photon_field", bath_json(*s.bath) });
    t.header.push_back({ "physical_mass_ratio", json_number(s.cb.mu_r) });
    t.header.push_back({ "gamma_p", json_number(g) });
    t.header.push_back({ "beta_p", json_number(beta) });
    t.header.push_back({ "gamma_th", json_number(s.H.gamma_th) });
    t.header.push_back({ "eps_prime_hi_max", json_number(khi) });
    t.header.push_back({ "seed", std::to_string(o.seed) });
    t.header.push_back({ "n_unweighted", std::to_string(o.n) });
    t.header.push_back({ "n_weighted_accepted", std::to_string(n_acc) });
    t.header.push_back({ "n_weighted_attempts", std::to_string(n_att) });
    t.header.push_back({ "rng", json_string("SplitMix64(seed) unweighted, SplitMix64(seed + 500000) weighted") });
    t.header.push_back({ "bins",
                         "{\"E\": {\"n\": 200, \"spacing\": \"log\", \"lo\": " + json_number(E_lo) +
                           ", \"hi\": " + json_number(E_hi) + "}, \"cos\": {\"n\": 200, \"spacing\": \"linear\", \"lo\": -1, \"hi\": 1}, "
                           "\"D\": {\"n\": 200, \"spacing\": \"linear\", \"lo\": " +
                           json_number(Dlo) + ", \"hi\": " + json_number(Dhi) + "}}" });
    t.header.push_back({ "weights", json_string("unweighted chain: counts; weighted chain: sum of 1/D per bin divided "
                                                "by the total sum of 1/D (packet weights, normalised)") });
    t.header.push_back({ "lab_frame", json_string("u^ = +z; cos theta relative to u^; lepton energies in m_e c^2") });
    t.header.push_back({ "tables", tables_json(*s.T, *s.cum, *s.B) });
    t.header.push_back({ "node_count", std::to_string(NB) });
    t.columns = { "E_minus_lo", "E_minus_hi", "n_unw_E_minus", "w_wt_E_minus", "E_plus_lo", "E_plus_hi", "n_unw_E_plus",
                  "w_wt_E_plus", "cos_minus_lo", "cos_minus_hi", "n_unw_cos_minus", "w_wt_cos_minus", "cos_plus_lo",
                  "cos_plus_hi", "n_unw_cos_plus", "w_wt_cos_plus", "D_lo", "D_hi", "n_unw_D", "w_wt_D" };
    for (int i = 0; i < NB; ++i) {
      t.rows.push_back({ edgesE[i], edgesE[i + 1], nEm[i], wEm[i], edgesE[i], edgesE[i + 1], nEp[i], wEp[i], edgesC[i],
                         edgesC[i + 1], nCm[i], wCm[i], edgesC[i], edgesC[i + 1], nCp[i], wCp[i], edgesD[i],
                         edgesD[i + 1], nD[i], wD[i] });
    }
    write_fixture_file(o.out, t);
    const FixtureContents c = read_fixture_file(o.out);
    std::printf("bh_sampler_dump hist %s Gamma %g: %zu bins written to %s in %.0f s; weighted %zu accepted of %zu "
                "attempts (data_sha256 %s)\n",
                o.bath.c_str(), g, c.rows.size(), o.out.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), n_acc, n_att,
                c.data_sha256.c_str());
    return 0;
  }

} // namespace

int main(int argc, char** argv) {
  Options o;
  try {
    o = parse(std::vector<std::string>(argv + 1, argv + argc));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_sampler_dump: option error: %s\n", e.what());
    std::fprintf(stderr,
                 "usage: bh_sampler_dump qk_table --out FILE [--status S] [--threads N]\n"
                 "       bh_sampler_dump moments --bath NAME --out FILE [--seed S] [--n N] [--n-weighted-attempts N]\n"
                 "       bh_sampler_dump hist --bath NAME --gamma G --out FILE [--seed S] [--n N] "
                 "[--n-weighted-attempts N]\n"
                 "       bh_sampler_dump ka2008 --bath NAME --gamma G --out FILE\n");
    return 2;
  }
  try {
    if (o.command == "qk_table") {
      return run_qk_table(o);
    }
    if (o.command == "moments") {
      return run_moments(o);
    }
    if (o.command == "ka2008") {
      return run_ka2008(o);
    }
    return run_hist(o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_sampler_dump: error: %s\n", e.what());
    return 1;
  }
}
