/**
 * @file tools/bh_sampler_dump.cpp
 * @brief Command-line producer of the step-4 fixture files (contract revision
 *        1.0 section 4), producer `entity-cpp`
 *
 * Usage:
 *   bh_sampler_dump qk_table --out FILE [--status STATUS] [--threads N]
 *   bh_sampler_dump moments  --bath NAME --out FILE [--seed S] [--n N]
 *                            [--n-weighted N] [--status STATUS] [--threads N]
 *   bh_sampler_dump hist     --bath NAME --gamma G --out FILE [--seed S] [--n N]
 *                            [--n-weighted N] [--bins-from FILE] [--status STATUS] [--threads N]
 *   bh_sampler_dump ka2008   --bath NAME --gamma G --out FILE [--status STATUS]
 *
 * `qk_table` (section 4 item 1): the eps' nodes of the step-1 grid (64 per
 * decade in log10(eps' - 2) from 1e-4 while eps' <= 1000, then eps' = 1000),
 * q_k from the rest-frame machinery, q_k from Chodorowski eq. 3.7, <E'_->,
 * <p' cos theta'>, sigma~ from the integral of the energy law (the A1 value)
 * and Maximon's sigma~. Deterministic quadrature (bh_restframe.h).
 *
 * `moments` (item 2): at every Gamma_p node of the bath's rate table, N
 * unweighted draws (default 1e6): <D>, <D^2>, <E->, <E+>, <cos theta-> (lab,
 * to u^), <eps>, each with its standard error (round 2, finding 7), min D,
 * max D; the weighted chain run to a fixed number of accepted packets
 * (`--n-weighted`, default 1e5; round 2, the contract's fixed N; the attempts
 * used are reported): E_p[1/D] with its standard error, times I/J and times
 * <D>_MC (delta-method s.e.), the acceptance efficiency with its s.e., 1/min D;
 * and the deterministic <D> of the density the chain samples (Dmean_sampled).
 * Seeds: SplitMix64(seed + i) unweighted and SplitMix64(seed + 500000 + i)
 * weighted at node i.
 *
 * `hist` (item 3): at one Gamma_p, N unweighted draws (default 1e6) and a
 * fixed number of accepted weighted packets (`--n-weighted`, default 1e6),
 * in 64 chunks with seeds of their own (thread-count independent): 200
 * log-spaced bins in E- and in E+ over the kinematic lab range
 * [1, Gamma (E'_max + beta p'_max)], 200 linear bins in cos theta- and
 * cos theta+ (lab, to u^), 200 linear bins in D over [D_min, D_max] of the
 * bath at this Gamma_p, or the edges of another fixture (`--bins-from`); the
 * layout of the bh-sampler-fixture histograms (quantity_id, bin, lo, hi,
 * unweighted_count, weighted_packet_count, sum_1_over_D,
 * sum_1_over_D_squared, normalized_packet_weight).
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

#include <algorithm>
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
    std::size_t   n_weighted { 0 };                       // accepted packets; 0: the command's default
    std::size_t   max_weighted_attempts { 100000000000ULL }; // a safety cap only (exceeding it is an error)
    std::string   bins_from;                              // hist: take the bin edges from this fixture
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
      } else if (key == "--n-weighted") {
        o.n_weighted = parse_count(key, val);
      } else if (key == "--max-weighted-attempts") {
        o.max_weighted_attempts = parse_count(key, val);
      } else if (key == "--bins-from") {
        o.bins_from = val;
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
    if (o.n_weighted == 0) {
      // the contract's fixed N: 1e5 accepted packets per node for the moments, 1e6 for a histogram
      o.n_weighted = o.command == "moments" ? 100000 : 1000000;
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
        "{\"energy\": \"Motz, Olsen and Koch 1969 3D-1000 (Heitler 1954 p. 258 eq. 8), Z = 1 unscreened\", "
        "\"angular\": \"Sauter-Gluckstern-Hull, MOK 1969 3D-2000 = Blumenthal 1970 eq. 10 (definitions eq. 11), "
        "Z = 1 unscreened; checked against the numerical integral of MOK 3D-3000 (Heitler 1954 p. 257 eq. 6)\", "
        "\"verification\": \"algebra, signs and normalisation checked against the PDFs by the referee (Codex 1, "
        "2026-10-27)\", "
        "\"q_k\": \"eps_prime - 2 <p cos theta>_h, computed as 2 <(E - p) + p (1 - cos theta)>_h (no cancellation; "
        "round 2)\", \"normalisation\": \"the laws' own integrals (sigma_from_g, "
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

  // running mean and variance (Welford 1962): a constant sample returns the constant exactly and a zero
  // variance, so the zero-variance columns of a mono bath (eps = eps0) are exact
  struct Welford {
    double n { 0.0 }, mean { 0.0 }, m2 { 0.0 };

    void add(double x) {
      n += 1.0;
      const double d = x - mean;
      mean += d / n;
      m2 += d * (x - mean);
    }

    double se() const {
      return n > 1.0 ? std::sqrt(std::max(0.0, m2 / (n - 1.0)) / n) : 0.0;
    }
  };

  int run_moments(const Options& o) {
    using namespace arch::bh;
    const auto t0 = std::chrono::steady_clock::now();
    Setup      s  = make_setup(o, nullptr);
    const std::size_t ng = s.H.size();
    // <D> of the density the unweighted chain samples (round 2, finding 7), deterministic
    const rf::QkFineGrid    qk16 = rf::table_qk_fine_grid(*s.T, 16, o.threads);
    const rf::SampledChainD SD(*s.cum, [&](double k) { return qk16(k); }, 16);
    struct Row {
      double      Dsampled;
      Welford     D, D2, Em, Ep, cosm, eps;
      double      Dmin, Dmax;
      Welford     invD;
      std::size_t n_acc, n_att;
      double      minDw;
    };
    std::vector<Row> rows(ng);
    rf::parallel_for_index(
      ng,
      [&](std::size_t i) {
        const double g = s.H.gamma[i];
        Row          r {};
        r.Dsampled = SD.Dmean_unweighted(*s.B, g, 16);
        {
          rf::SplitMix64 rng(o.seed + i);
          r.Dmin = 1.0e300;
          r.Dmax = -1.0e300;
          for (std::size_t n = 0; n < o.n; ++n) {
            const auto   e  = s.S->draw(g, false, rng);
            const double pm = std::sqrt(e.lab_m.p.x * e.lab_m.p.x + e.lab_m.p.y * e.lab_m.p.y +
                                        e.lab_m.p.z * e.lab_m.p.z);
            r.D.add(e.D);
            r.D2.add(e.D * e.D);
            r.Em.add(e.lab_m.E);
            r.Ep.add(e.lab_p.E);
            r.cosm.add(e.lab_m.p.z / pm);
            r.eps.add(e.eps);
            r.Dmin = std::min(r.Dmin, e.D);
            r.Dmax = std::max(r.Dmax, e.D);
          }
        }
        {
          rf::SplitMix64 rng(o.seed + 500000 + i);
          r.minDw = 1.0e300;
          r.n_acc = r.n_att = 0;
          while (r.n_acc < o.n_weighted) {
            const auto e = s.S->draw(g, true, rng);
            r.n_att += static_cast<std::size_t>(e.attempts);
            if (r.n_att > o.max_weighted_attempts) {
              throw std::runtime_error("moments: the weighted chain exceeded --max-weighted-attempts at Gamma " +
                                       std::to_string(g));
            }
            r.invD.add(1.0 / e.D);
            r.minDw = std::min(r.minDw, e.D);
            ++r.n_acc;
          }
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
    t.header.push_back({ "n_weighted_accepted", std::to_string(o.n_weighted) });
    t.header.push_back({ "rng", json_string("SplitMix64(seed + i) unweighted, SplitMix64(seed + 500000 + i) weighted, "
                                            "node i; uniform = (bits >> 11 + 0.5) 2^-53") });
    t.header.push_back({ "chains",
                         json_string("unweighted: eps ~ n/eps^2 Phi(eps'_hi), eps'|eps ~ eps' sigma~ on [2, eps'_hi], "
                                     "E'- ~ g, angles ~ h independently, rotation and boost along u^ = z; weighted: "
                                     "eps ~ n/eps^2 F(eps'_hi), eps'|eps ~ eps' sigma~ D_max, unweighted conditionals, "
                                     "accept with D/D_max, run to a fixed number of accepted packets") });
    t.header.push_back({ "tables", tables_json(*s.T, *s.cum, *s.B) });
    t.header.push_back({ "Dmean_table_definition", json_string("I/J of the step-1 rate table at the node") });
    t.header.push_back(
      { "Dmean_sampled_definition",
        json_string("<D> of the density the unweighted chain samples (photon table, eps' interpolant, the tables' "
                    "q_k on 16 sub-nodes per eps' interval), by quadrature (rf::SampledChainD, Gauss-Legendre 16): "
                    "the value Dmean_mc converges to") });
    t.header.push_back(
      { "standard_errors",
        json_string("every Monte Carlo mean carries <column>_se = sqrt(s^2/N), s^2 the unbiased sample variance "
                    "(Welford's update); Einv_Dmean_mc_se by the delta method from the independent unweighted and "
                    "weighted chains; efficiency_se = efficiency sqrt((1 - efficiency)/n_weighted_accepted)") });
    std::string det = "[\"gamma_p\", \"beta_p\", \"J\", \"I\", \"Dmean_table\", \"Dmean_sampled\"";
    if (s.B->mono) {
      det += ", \"eps_mean_mc\"";
    }
    det += "]";
    t.header.push_back({ "deterministic_columns", det });
    if (s.B->mono) {
      t.header.push_back(
        { "zero_variance_columns",
          json_string("eps_mean_mc: eps = eps0 exactly in a mono bath; Welford's update returns eps0 exactly and "
                      "eps_mean_mc_se = 0; compare at 1e-12 relative, not by a standard-error criterion") });
    }
    t.header.push_back({ "node_count", std::to_string(ng) });
    t.columns = { "gamma_p", "beta_p", "J", "I", "Dmean_table", "Dmean_sampled", "n_unweighted", "Dmean_mc",
                  "Dmean_se", "D2mean", "D2mean_se", "Emean_minus", "Emean_minus_se", "Emean_plus", "Emean_plus_se",
                  "cos_minus_lab_mean", "cos_minus_lab_mean_se", "eps_mean_mc", "eps_mean_mc_se", "D_min", "D_max",
                  "n_weighted_accepted", "n_weighted_attempts", "Einv", "Einv_se", "Einv_Dmean", "Einv_Dmean_se",
                  "Einv_Dmean_mc", "Einv_Dmean_mc_se", "efficiency", "efficiency_se", "inv_min_D_weighted" };
    std::size_t att_total = 0, att_max = 0;
    for (std::size_t i = 0; i < ng; ++i) {
      const auto&  r   = rows[i];
      const double a   = r.invD.mean, sa = r.invD.se(), b = r.D.mean, sb = r.D.se();
      const double eff = static_cast<double>(r.n_acc) / static_cast<double>(r.n_att);
      att_total += r.n_att;
      att_max = std::max(att_max, r.n_att);
      t.rows.push_back({ s.H.gamma[i], s.H.beta[i], s.H.J[i], s.H.I[i], s.H.Dmean[i], r.Dsampled,
                         static_cast<double>(o.n), r.D.mean, r.D.se(), r.D2.mean, r.D2.se(), r.Em.mean, r.Em.se(),
                         r.Ep.mean, r.Ep.se(), r.cosm.mean, r.cosm.se(), r.eps.mean, r.eps.se(), r.Dmin, r.Dmax,
                         static_cast<double>(r.n_acc), static_cast<double>(r.n_att), a, sa, a * s.H.Dmean[i],
                         sa * s.H.Dmean[i], a * b, std::sqrt(b * b * sa * sa + a * a * sb * sb), eff,
                         eff * std::sqrt(std::max(0.0, 1.0 - eff) / static_cast<double>(r.n_acc)), 1.0 / r.minDw });
    }
    t.header.push_back({ "n_weighted_attempts_total", std::to_string(att_total) });
    write_fixture_file(o.out, t);
    const FixtureContents c = read_fixture_file(o.out);
    std::printf("bh_sampler_dump moments %s: %zu nodes written to %s in %.0f s; weighted chain %zu accepted per node, "
                "%zu attempts in total, at most %zu at one node (data_sha256 %s)\n",
                o.bath.c_str(), c.rows.size(), o.out.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), o.n_weighted, att_total,
                att_max, c.data_sha256.c_str());
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
                           "Sauter-Gluckstern-Hull single-lepton law (MOK 3D-2000 = Blumenthal 1970 eq. 10) in the proton rest "
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
    const int                        NQ = 5;
    const char* const                qname[NQ] = { "Eminus", "Eplus", "cosminus", "cosplus", "D" };
    std::vector<std::vector<double>> edges(NQ);
    std::string                      bins_json;
    if (!o.bins_from.empty()) {
      // the bin edges of another fixture in the bh-sampler-fixture layout (quantity_id, bin, lo, hi, ...)
      const FixtureContents ref = read_fixture_file(o.bins_from);
      auto col = [&](const std::string& n) {
        for (std::size_t c = 0; c < ref.columns.size(); ++c) {
          if (ref.columns[c] == n) {
            return c;
          }
        }
        throw std::runtime_error("hist: --bins-from file has no column " + n);
      };
      const std::size_t cq = col("quantity_id"), cl = col("lo"), ch = col("hi");
      for (const auto& row : ref.rows) {
        const int q = static_cast<int>(row[cq]);
        if (q < 0 || q >= NQ) {
          throw std::runtime_error("hist: --bins-from quantity_id out of range");
        }
        if (edges[q].empty()) {
          edges[q].push_back(row[cl]);
        }
        edges[q].push_back(row[ch]);
      }
      for (int q = 0; q < NQ; ++q) {
        if (edges[q].size() < 2 || !std::is_sorted(edges[q].begin(), edges[q].end())) {
          throw std::runtime_error("hist: --bins-from edges missing or not increasing");
        }
      }
      bins_json = "{\"source\": " + json_string(o.bins_from) + "}";
    } else {
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
      const int NB = 200;
      for (int q = 0; q < NQ; ++q) {
        edges[q].resize(NB + 1);
      }
      for (int i = 0; i <= NB; ++i) {
        edges[0][i] = edges[1][i] = E_lo * std::pow(E_hi / E_lo, static_cast<double>(i) / NB);
        edges[2][i] = edges[3][i] = -1.0 + 2.0 * i / NB;
        edges[4][i]               = Dlo + (Dhi - Dlo) * i / NB;
      }
      bins_json = "{\"E\": {\"n\": 200, \"spacing\": \"log\", \"lo\": " + json_number(E_lo) + ", \"hi\": " +
                  json_number(E_hi) +
                  "}, \"cos\": {\"n\": 200, \"spacing\": \"linear\", \"lo\": -1, \"hi\": 1}, \"D\": {\"n\": 200, "
                  "\"spacing\": \"linear\", \"lo\": " +
                  json_number(Dlo) + ", \"hi\": " + json_number(Dhi) + "}}";
    }
    // values outside the edges are counted in the first or last bin (and reported)
    auto bin_of = [&](int q, double v, std::size_t& clamped) {
      const auto& e  = edges[q];
      const int   nb = static_cast<int>(e.size()) - 1;
      if (v < e.front() || v > e.back()) {
        ++clamped;
      }
      const int i = static_cast<int>(std::upper_bound(e.begin(), e.end(), v) - e.begin()) - 1;
      return std::min(nb - 1, std::max(0, i));
    };
    auto values = [](const rf::PairEvent& e, double (&v)[NQ]) {
      auto cosl = [](const rf::FourMomentum& q) {
        return q.p.z / std::sqrt(q.p.x * q.p.x + q.p.y * q.p.y + q.p.z * q.p.z);
      };
      v[0] = e.lab_m.E;
      v[1] = e.lab_p.E;
      v[2] = cosl(e.lab_m);
      v[3] = cosl(e.lab_p);
      v[4] = e.D;
    };
    // 64 chunks with seeds of their own, merged in chunk order: the result does not depend on the thread count
    const std::size_t NC = 64;
    struct Chunk {
      std::vector<std::vector<double>> nu, nw, sw, sw2;
      std::size_t                      att { 0 }, clamped { 0 };
    };
    std::vector<Chunk> chunks(NC);
    rf::parallel_for_index(
      NC,
      [&](std::size_t c) {
        Chunk& ch = chunks[c];
        for (auto* v : { &ch.nu, &ch.nw, &ch.sw, &ch.sw2 }) {
          v->assign(NQ, {});
          for (int q = 0; q < NQ; ++q) {
            (*v)[q].assign(edges[q].size() - 1, 0.0);
          }
        }
        const std::size_t nu = o.n / NC + (c < o.n % NC ? 1 : 0);
        const std::size_t nw = o.n_weighted / NC + (c < o.n_weighted % NC ? 1 : 0);
        double            v[NQ];
        {
          rf::SplitMix64 rng(o.seed + 1000003ULL * c);
          for (std::size_t n = 0; n < nu; ++n) {
            values(s.S->draw(g, false, rng), v);
            for (int q = 0; q < NQ; ++q) {
              ch.nu[q][bin_of(q, v[q], ch.clamped)] += 1.0;
            }
          }
        }
        {
          rf::SplitMix64 rng(o.seed + 500000ULL + 1000003ULL * c);
          for (std::size_t n = 0; n < nw; ++n) {
            const auto e = s.S->draw(g, true, rng);
            ch.att += static_cast<std::size_t>(e.attempts);
            if (ch.att > o.max_weighted_attempts / NC + 1) {
              throw std::runtime_error("hist: the weighted chain exceeded --max-weighted-attempts");
            }
            values(e, v);
            const double w = 1.0 / e.D;
            for (int q = 0; q < NQ; ++q) {
              const int i = bin_of(q, v[q], ch.clamped);
              ch.nw[q][i] += 1.0;
              ch.sw[q][i] += w;
              ch.sw2[q][i] += w * w;
            }
          }
        }
      },
      o.threads);
    // merge in chunk order
    std::vector<std::vector<double>> nu(NQ), nw(NQ), sw(NQ), sw2(NQ);
    std::size_t                      n_att = 0, clamped = 0;
    for (int q = 0; q < NQ; ++q) {
      nu[q].assign(edges[q].size() - 1, 0.0);
      nw[q] = sw[q] = sw2[q] = nu[q];
    }
    for (const auto& ch : chunks) {
      n_att += ch.att;
      clamped += ch.clamped;
      for (int q = 0; q < NQ; ++q) {
        for (std::size_t i = 0; i < nu[q].size(); ++i) {
          nu[q][i] += ch.nu[q][i];
          nw[q][i] += ch.nw[q][i];
          sw[q][i] += ch.sw[q][i];
          sw2[q][i] += ch.sw2[q][i];
        }
      }
    }
    double W = 0.0, W2 = 0.0;
    for (std::size_t i = 0; i < sw[0].size(); ++i) {
      W += sw[0][i];
      W2 += sw2[0][i];
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
    t.header.push_back({ "n_weighted_accepted", std::to_string(o.n_weighted) });
    t.header.push_back({ "n_weighted_attempts", std::to_string(n_att) });
    t.header.push_back({ "weighted_ESS", json_number(W * W / W2) });
    t.header.push_back({ "values_outside_edges", std::to_string(clamped) });
    t.header.push_back({ "rng", json_string("64 chunks c = 0..63 (n/64 draws each, remainder to the first): "
                                            "SplitMix64(seed + 1000003 c) unweighted, SplitMix64(seed + 500000 + "
                                            "1000003 c) weighted; merged in chunk order") });
    t.header.push_back({ "quantities", "[\"Eminus\", \"Eplus\", \"cosminus\", \"cosplus\", \"D\"]" });
    t.header.push_back({ "bins", bins_json });
    t.header.push_back({ "weights", json_string("weighted chain: packet weight 1/D; normalized_packet_weight = "
                                                "sum_1_over_D of the bin / the total sum of 1/D") });
    t.header.push_back({ "lab_frame", json_string("u^ = +z; cos theta relative to u^; lepton energies in m_e c^2; "
                                                  "values outside the edges counted in the first or last bin") });
    t.header.push_back({ "tables", tables_json(*s.T, *s.cum, *s.B) });
    std::size_t nrows = 0;
    for (int q = 0; q < NQ; ++q) {
      nrows += nu[q].size();
    }
    t.header.push_back({ "node_count", std::to_string(nrows) });
    t.columns = { "quantity_id", "bin", "lo", "hi", "unweighted_count", "weighted_packet_count", "sum_1_over_D",
                  "sum_1_over_D_squared", "normalized_packet_weight" };
    for (int q = 0; q < NQ; ++q) {
      for (std::size_t i = 0; i < nu[q].size(); ++i) {
        t.rows.push_back({ static_cast<double>(q), static_cast<double>(i), edges[q][i], edges[q][i + 1], nu[q][i],
                           nw[q][i], sw[q][i], sw2[q][i], sw[q][i] / W });
      }
    }
    write_fixture_file(o.out, t);
    const FixtureContents c = read_fixture_file(o.out);
    std::printf("bh_sampler_dump hist %s Gamma %g: %zu rows (%s) written to %s in %.0f s; weighted %zu accepted of %zu "
                "attempts, ESS %.0f; %zu values outside the edges (data_sha256 %s)\n",
                o.bath.c_str(), g, c.rows.size(), qname[0], o.out.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), o.n_weighted, n_att,
                W * W / W2, clamped, c.data_sha256.c_str());
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
                 "       bh_sampler_dump moments --bath NAME --out FILE [--seed S] [--n N] [--n-weighted N]\n"
                 "       bh_sampler_dump hist --bath NAME --gamma G --out FILE [--seed S] [--n N] "
                 "[--n-weighted N] [--bins-from FILE]\n"
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
