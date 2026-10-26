/**
 * @file tools/bh_sampler_dump.cpp
 * @brief Command-line producer of the step-4 fixture files (contract revision
 *        1.0 section 4), producer `entity-cpp`
 *
 * Usage:
 *   bh_sampler_dump qk_table --out FILE [--status STATUS] [--threads N]
 *
 * `qk_table` (section 4 item 1): the eps' nodes of the step-1 grid (64 per
 * decade in log10(eps' - 2) from 1e-4 while eps' <= 1000, then eps' = 1000),
 * q_k from the rest-frame machinery, q_k from Chodorowski eq. 3.7, <E'_->,
 * <p' cos theta'>, sigma~ from the integral of the energy law (the A1 value)
 * and Maximon's sigma~. Deterministic quadrature (bh_restframe.h).
 *
 * Option errors exit with 2, runtime errors with 1.
 */

#include "archetypes/hadronic/bh_restframe.h"
#include "archetypes/hadronic/bh_sampler_io.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

  struct Options {
    std::string command;
    std::string out;
    std::string status { "cpp_sampler_candidate" };
    unsigned    threads { 0 };
  };

  Options parse(const std::vector<std::string>& t) {
    Options o;
    if (t.empty()) {
      throw std::invalid_argument("missing command (qk_table)");
    }
    o.command = t[0];
    if (o.command != "qk_table") {
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
      } else if (key == "--threads") {
        double v = 0.0;
        if (!arch::bh::parse_number_strict(val, v) || !(v >= 1.0) || v != std::floor(v)) {
          throw std::invalid_argument("--threads must be a positive integer");
        }
        o.threads = static_cast<unsigned>(v);
      } else {
        throw std::invalid_argument("unknown option '" + key + "'");
      }
    }
    if (o.out.empty()) {
      throw std::invalid_argument("--out is required");
    }
    return o;
  }

  int run_qk_table(const Options& o) {
    using namespace arch::bh;
    const rf::RestFrameQuadrature q;
    const std::vector<double>  grid = rf::qk_grid();
    std::vector<rf::Moments>   mom(grid.size());
    rf::parallel_for_index(
      grid.size(),
      [&](std::size_t i) { mom[i] = rf::moments(grid[i], q); },
      o.threads);
    FixtureTable t;
    t.header = {
      { "format", json_string("bh-qk-table") },
      { "format_revision", "1" },
      { "producer", json_string("entity-cpp") },
      { "version", json_string(ENTITY_GIT_HASH) },
      { "status", json_string(o.status) },
      { "grid",
        "{\"variable\": \"log10(eps_prime - 2)\", \"nodes_per_decade\": 64, "
        "\"delta_min\": 1e-4, \"eps_prime_max\": 1000, \"endpoint_appended\": true}" },
      { "laws",
        "{\"energy\": \"Blumenthal 1970 eq. 10 (= MOK 1969 3D-1000), Z = 1 unscreened\", "
        "\"angular\": \"Sauter-Gluckstern-Hull, MOK 1969 3D-2000, Z = 1 unscreened; signs and "
        "normalisation fixed against the numerical integral of Heitler 1954 eq. 26.6\", "
        "\"q_k\": \"eps_prime - 2 <p cos theta>_h\", \"normalisation\": \"Maximon 1968 sigma~\"}" },
      { "quadrature",
        "{\"gl_order\": " + std::to_string(q.gl_order) +
          ", \"energy_panels\": " + std::to_string(q.energy_panels) +
          ", \"energy_variable\": \"E = 1 + (eps_prime - 2) sin^2(pi s / 2)\"" +
          ", \"h_t_angle\": " + json_number(q.h_t_angle) +
          ", \"angle_variable\": \"t = ln(1 - cos theta + (E - p)/p)\", \"columns_format\": \"%.12e\"}" },
      { "node_count", std::to_string(grid.size()) },
    };
    t.columns = { "eps_prime", "q_k", "q_k_eq37", "mean_E_minus", "mean_p_cos_theta",
                  "sigma_from_g", "sigma_maximon" };
    for (std::size_t i = 0; i < grid.size(); ++i) {
      const auto& m = mom[i];
      t.rows.push_back({ grid[i], m.q_k, q_k(grid[i]), m.mean_E, m.mean_pcos, m.sigma_g,
                         m.sigma_maximon });
    }
    write_fixture_file(o.out, t);
    // read back through the validating reader
    const FixtureContents c = read_fixture_file(o.out);
    std::printf("bh_sampler_dump qk_table: %zu nodes written to %s (data_sha256 %s)\n",
                c.rows.size(), o.out.c_str(), c.data_sha256.c_str());
    return 0;
  }

} // namespace

int main(int argc, char** argv) {
  Options o;
  try {
    o = parse(std::vector<std::string>(argv + 1, argv + argc));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_sampler_dump: option error: %s\n", e.what());
    std::fprintf(stderr, "usage: bh_sampler_dump qk_table --out FILE [--status S] [--threads N]\n");
    return 2;
  }
  try {
    return run_qk_table(o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_sampler_dump: error: %s\n", e.what());
    return 1;
  }
}
