/**
 * @file tools/bh_tables_dump.cpp
 * @brief Builds the Bethe-Heitler startup tables for a photon bath and writes
 *        them in the exchange-file format of contract revision 1.4 section 2.
 *
 * Usage:
 *   bh_tables_dump --type mono --eps0 0.5 [--mu_r 1836.15] [--gamma_p_max G|auto]
 *                  [--h 0.02] [--status cpp_candidate] --out table.txt
 *   bh_tables_dump --type powerlaw --eps_min 0.1 --eps_max 1.5 --index 2 ...
 *   bh_tables_dump --type blackbody --theta 0.1 --eps_min 0 --eps_max 1.9 ...
 *   bh_tables_dump --type table --file spectrum.txt [--eps_min a] [--eps_max b] ...
 *
 * Every numeric option must convert completely to a finite number in its
 * admissible range (see bh_options.h); a conversion failure exits with 2.
 * All table quantities are computed in double precision.
 */

#include "archetypes/hadronic/bh_born_qk.h"
#include "archetypes/hadronic/bh_options.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_table_io.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace {
  void usage() {
    std::fprintf(
      stderr,
      "usage: bh_tables_dump --type {mono|powerlaw|blackbody|table} [bath "
      "parameters]\n"
      "                      [--mu_r 1836.15] [--gamma_p_max G|auto] [--h "
      "0.02]\n"
      "                      [--status cpp_candidate] --out FILE\n"
      "  mono:      --eps0 E                       (0 < E < 2)\n"
      "  powerlaw:  --eps_min A --eps_max B --index P\n"
      "  blackbody: --theta T --eps_min A --eps_max B\n"
      "  table:     --file F [--eps_min A] [--eps_max B]\n"
      "  --qk_source born | fit (default born): q_k from the Born machinery (contract 1.5) or the eq. 3.7 fit\n");
  }
} // namespace

int main(int argc, char** argv) {
  arch::bh::DumpOptions opt;
  try {
    opt = arch::bh::parse_dump_options(
      std::vector<std::string>(argv + 1, argv + argc));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_tables_dump: %s\n", e.what());
    usage();
    return 2;
  }
  try {
    arch::bh::Quadrature q;
    q.h_log_delta = opt.h;
    q.h_log_eps   = opt.h;
    q.qk_source   = opt.qk_source;
    if (q.qk_source == arch::bh::QkSource::Born) {
      const auto& BT = arch::bh::born::table();
      std::printf("bh_tables_dump: qk_source = born-3D2000: %zu q_k nodes built in %.3f s\n", BT.size(),
                  BT.build_seconds());
    } else {
      std::printf("bh_tables_dump: qk_source = chodorowski-3.7\n");
    }

    const auto                        bath = arch::bh::make_bath(opt.spec);
    const arch::bh::BetheHeitlerRates rates(bath, q);
    const auto T = arch::bh::build_tables(rates, opt.mu_r, opt.gamma_p_max);
    arch::bh::validate_tables(T);

    arch::bh::TableFileMeta meta;
    meta.status = opt.status;
#ifdef ENTITY_GIT_HASH
    meta.version = ENTITY_GIT_HASH;
#endif
    arch::bh::write_table_file(opt.out, T, *bath, q, meta);

    // read back: the reader throws on any schema or hash violation
    const auto back = arch::bh::read_table_file(opt.out);
    std::printf(
      "%s: %zu nodes (fine %zu, coarse %zu), gamma_th = %.10g, "
      "gamma_min = %.10g, gamma_max = %.10g, eps'_max = %.6g, "
      "I_peak = %.10e at gamma = %.10g, discarded (n, e) = (%.6e, %.6e), "
      "data_sha256 = %s\n",
      opt.out.c_str(),
      T.size(),
      T.n_fine,
      T.n_coarse,
      T.gamma_th,
      T.gamma_min,
      T.gamma_max,
      T.eps_prime_max,
      T.I_peak,
      T.gamma_at_I_peak,
      T.clipped_number_fraction,
      T.clipped_energy_fraction,
      back.data_sha256.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bh_tables_dump: %s\n", e.what());
    return 1;
  }
}
