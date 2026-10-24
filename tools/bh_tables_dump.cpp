/**
 * @file tools/bh_tables_dump.cpp
 * @brief Builds the Bethe-Heitler startup tables for a photon bath and writes
 *        them in the exchange-file format of contract revision 1.3 section 4.
 *
 * Usage:
 *   bh_tables_dump --type mono --eps0 0.5 [--mu_r 1836.15] [--gamma_p_max G]
 *                  [--h 0.02] [--status cpp_candidate] --out table.txt
 *   bh_tables_dump --type powerlaw --eps_min 0.1 --eps_max 1.5 --index 2 ...
 *   bh_tables_dump --type blackbody --theta 0.1 --eps_min 0 --eps_max 1.9 ...
 *   bh_tables_dump --type table --file spectrum.txt [--eps_min a] [--eps_max b] ...
 *
 * All table quantities are computed in double precision.
 */

#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_table_io.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

namespace {
  void usage() {
    std::fprintf(
      stderr,
      "usage: bh_tables_dump --type {mono|powerlaw|blackbody|table} [bath "
      "parameters]\n"
      "                      [--mu_r 1836.15] [--gamma_p_max G] [--h 0.02]\n"
      "                      [--status cpp_candidate] --out FILE\n"
      "  mono:      --eps0 E\n"
      "  powerlaw:  --eps_min A --eps_max B --index P\n"
      "  blackbody: --theta T --eps_min A --eps_max B\n"
      "  table:     --file F [--eps_min A] [--eps_max B]\n");
  }
} // namespace

int main(int argc, char** argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--", 0) != 0) {
      usage();
      return 2;
    }
    a = a.substr(2);
    if (i + 1 >= argc) {
      usage();
      return 2;
    }
    args[a] = argv[++i];
  }
  auto has = [&args](const std::string& k) {
    return args.count(k) > 0;
  };
  auto num = [&args](const std::string& k, double def) {
    return args.count(k) ? std::strtod(args[k].c_str(), nullptr) : def;
  };
  if (!has("type") || !has("out")) {
    usage();
    return 2;
  }
  try {
    arch::bh::BathSpec spec;
    spec.type                = args["type"];
    spec.eps0                = num("eps0", 0.0);
    spec.eps_min             = num("eps_min", -1.0);
    spec.eps_max             = num("eps_max", -1.0);
    spec.index               = num("index", 0.0);
    spec.theta               = num("theta", 0.0);
    spec.file                = has("file") ? args["file"] : "";
    const double mu_r        = num("mu_r", 1836.15);
    const double gamma_p_max = num("gamma_p_max", 0.0);

    arch::bh::Quadrature q;
    q.h_log_delta = num("h", q.h_log_delta);
    q.h_log_eps   = num("h", q.h_log_eps);

    const auto                        bath = arch::bh::make_bath(spec);
    const arch::bh::BetheHeitlerRates rates(bath, q);
    const auto T = arch::bh::build_tables(rates, mu_r, gamma_p_max);
    arch::bh::validate_tables(T);

    arch::bh::TableFileMeta meta;
    meta.status = has("status") ? args["status"] : "cpp_candidate";
#ifdef ENTITY_GIT_HASH
    meta.commit = ENTITY_GIT_HASH;
#endif
    arch::bh::write_table_file(args["out"], T, *bath, q, meta);

    const auto back = arch::bh::read_table_file(args["out"]);
    std::printf(
      "%s: %zu nodes (fine %zu, coarse %zu), gamma_th = %.10g, "
      "gamma_min = %.10g, gamma_max = %.10g, eps'_max = %.6g, "
      "I_peak = %.10e, clipped (n, e) = (%.6e, %.6e), sha256 = %s%s\n",
      args["out"].c_str(),
      T.size(),
      T.n_fine,
      T.n_coarse,
      T.gamma_th,
      T.gamma_min,
      T.gamma_max,
      T.eps_prime_max,
      T.I_peak,
      T.clipped_number_fraction,
      T.clipped_energy_fraction,
      back.sha256_data.c_str(),
      back.sha_ok ? "" : " (HASH MISMATCH)");
    return back.sha_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_tables_dump: %s\n", e.what());
    return 1;
  }
}
