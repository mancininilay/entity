/**
 * @file tests/archetypes/bh_fixtures.cpp
 * @brief Contract 1.4 item 8 inside ctest: the C++ Bethe-Heitler rates against
 *        the frozen Python reference tables (`bhref` 0.1.0, fixtures under
 *        tests/data/hadronic/bhref-0.1.0/, the fit path), and, since step 6
 *        (contract 1.5), the Born path (`qk_source = born`) against the bhref
 *        0.3.0 frozen tables (tests/data/hadronic/bhref-0.3.0/, frozen 2026-10-30,
 *        replaced by the frozen release at the freeze). For each of the five files: the
 *        file hash against SHA256SUMS and the header's data_sha256 (verified by
 *        the reader), the bath rebuilt from the header, `at_gamma` evaluated at
 *        every reference gamma_p, the contract tolerances with masks and peak
 *        from the reference (|dJ/J|, |dI/I| <= 1e-6 where I_ref > 1e-3 I_peak_ref,
 *        <= 1e-3 where I_ref > 1e-9 I_peak_ref), and the C++ node placement
 *        against the reference nodes (y to 1e-9). Host-only, no Kokkos.
 */

#include "archetypes/hadronic/bh_born_qk.h"
#include "archetypes/hadronic/bh_physics.h"
#include "archetypes/hadronic/bh_table_io.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef BH_TEST_DATA_DIR
  #error "BH_TEST_DATA_DIR must be defined"
#endif

namespace {

  int n_pass = 0, n_fail = 0;

  void verdict(bool ok, const std::string& what, double value, double tol) {
    std::printf("%s: %s : %.6e (tolerance %.3e)\n", ok ? "PASS" : "FAIL", what.c_str(), value, tol);
    ok ? ++n_pass : ++n_fail;
  }

  std::map<std::string, std::string> read_sha256sums(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot open " + path);
    }
    std::map<std::string, std::string> out;
    std::string                        hash, name;
    while (in >> hash >> name) {
      out[name] = hash;
    }
    return out;
  }

  double json_number(const arch::bh::JsonValue& obj, const char* key) {
    const auto* v = obj.find(key);
    if (v == nullptr || v->type != arch::bh::JsonValue::Type::Number) {
      throw std::runtime_error(std::string("photon_field: missing numeric member ") + key);
    }
    return v->number;
  }

  arch::bh::BathSpec bath_from_header(const arch::bh::TableFileContents& F) {
    const auto it = F.header.find("photon_field");
    if (it == F.header.end()) {
      throw std::runtime_error("header has no photon_field");
    }
    const auto obj = arch::bh::parse_json(it->second);
    const auto* type = obj.find("type");
    if (type == nullptr || type->type != arch::bh::JsonValue::Type::String) {
      throw std::runtime_error("photon_field.type missing");
    }
    arch::bh::BathSpec s;
    s.type = type->string;
    if (s.type == "mono") {
      s.eps0 = json_number(obj, "eps0");
    } else if (s.type == "powerlaw") {
      s.eps_min = json_number(obj, "eps_min");
      s.eps_max = json_number(obj, "eps_max");
      s.index   = json_number(obj, "index");
    } else if (s.type == "blackbody") {
      s.theta   = json_number(obj, "theta");
      s.eps_min = json_number(obj, "eps_min");
      s.eps_max = json_number(obj, "eps_max");
    } else {
      throw std::runtime_error("photon_field.type '" + s.type + "' not handled by the fixture test");
    }
    return s;
  }

  struct Reference {
    std::string        dir;           // under BH_TEST_DATA_DIR
    std::string        status_prefix; // expected header status
    std::string        qk_key;        // expected header qk_source (absent reads as chodorowski-3.7)
    arch::bh::QkSource source;        // the C++ path compared
  };

  void compare_file(const Reference& ref, const std::string& dir, const std::string& name, const std::string& expected_hash) {
    const std::string path = dir + "/" + name;
    std::printf("---- %s [%s, C++ qk_source = %s] ----\n", name.c_str(), ref.dir.c_str(), arch::bh::qk_source_name(ref.source));
    // file-level manifest
    const std::string file_hash = arch::bh::sha256_of_file(path);
    verdict(file_hash == expected_hash, name + ": file sha256 matches SHA256SUMS", file_hash == expected_hash, 1.0);
    // the reader verifies data_sha256 and the schema (throws otherwise)
    const auto F = arch::bh::read_table_file(path);
    verdict(F.producer == "bhref" && F.status.rfind(ref.status_prefix, 0) == 0 && F.qk_source == ref.qk_key,
            name + ": producer bhref, status " + ref.status_prefix + "..., qk_source " + ref.qk_key +
              " (data_sha256 verified by the reader; found status " + F.status + ", qk_source " + F.qk_source + ")",
            1.0, 1.0);

    // the C++ rates for the bath of the header
    const auto                        spec = bath_from_header(F);
    const auto                        bath = arch::bh::make_bath(spec);
    arch::bh::Quadrature              q;
    q.qk_source = ref.source;
    const arch::bh::BetheHeitlerRates rates(bath, q);
    // the other path, for the contract-1.5 statements (J and eps_mean unchanged, the fit budget of I)
    arch::bh::Quadrature              q_other;
    q_other.qk_source = ref.source == arch::bh::QkSource::Born ? arch::bh::QkSource::Fit : arch::bh::QkSource::Born;
    const arch::bh::BetheHeitlerRates rates_other(bath, q_other);
    const double gamma_p_max = F.header.count("gamma_p_max_input") &&
                                   arch::bh::parse_json(F.header.at("gamma_p_max_input")).type ==
                                     arch::bh::JsonValue::Type::Number
                                 ? arch::bh::parse_json(F.header.at("gamma_p_max_input")).number
                                 : 0.0;
    const auto T = arch::bh::build_tables(rates, F.physical_mass_ratio, gamma_p_max);

    // grid: same node count and y to 1e-9
    double max_dy = 0.0;
    if (T.size() == F.node_count) {
      for (std::size_t i = 0; i < T.size(); ++i) {
        max_dy = std::max(max_dy, std::abs(T.y[i] - F.y[i]));
      }
    } else {
      max_dy = 1.0e300;
    }
    verdict(T.size() == F.node_count && max_dy <= 1.0e-9,
            name + ": C++ node placement reproduces the reference nodes (" + std::to_string(T.size()) + " vs " +
              std::to_string(F.node_count) + " nodes, max |dy|)",
            max_dy, 1.0e-9);
    std::printf("(diag) %s: gamma_th C++ %.15g ref %.15g, gamma_min %.15g / %.15g, gamma_max %.15g / %.15g, "
                "I_peak %.12e / %.12e\n",
                name.c_str(), T.gamma_th, F.gamma_th, T.gamma_min, F.gamma_min, T.gamma_max, F.gamma_max,
                T.I_peak, F.I_peak);

    // rates at every reference node, masks and peak from the reference
    double max_core_J = 0.0, max_core_I = 0.0, max_tail_J = 0.0, max_tail_I = 0.0, max_all_J = 0.0,
           max_all_I = 0.0, max_D = 0.0, max_eps = 0.0;
    double      max_dJ_paths = 0.0, max_deps_paths = 0.0, max_fit_budget = 0.0, gamma_fit_budget = 0.0;
    std::size_t n_core = 0, n_tail = 0;
    for (std::size_t i = 0; i < F.node_count; ++i) {
      const auto   r  = rates.at_gamma(F.gamma[i]);
      const auto   ro = rates_other.at_gamma(F.gamma[i]);
      max_dJ_paths    = std::max(max_dJ_paths, std::abs(r.J / ro.J - 1.0));
      max_deps_paths  = std::max(max_deps_paths, std::abs(r.eps_mean / ro.eps_mean - 1.0));
      if (F.I[i] > 1.0e-3 * F.I_peak) {
        const double fb = std::abs(r.I / ro.I - 1.0);
        if (fb > max_fit_budget) {
          max_fit_budget   = fb;
          gamma_fit_budget = F.gamma[i];
        }
      }
      const double dJ = std::abs(r.J / F.J[i] - 1.0);
      const double dI = std::abs(r.I / F.I[i] - 1.0);
      max_all_J       = std::max(max_all_J, dJ);
      max_all_I       = std::max(max_all_I, dI);
      if (F.Dmean[i] > 0.0) {
        max_D = std::max(max_D, std::abs(r.Dmean / F.Dmean[i] - 1.0));
      }
      if (F.eps_mean[i] > 0.0) {
        max_eps = std::max(max_eps, std::abs(r.eps_mean / F.eps_mean[i] - 1.0));
      }
      if (F.I[i] > 1.0e-3 * F.I_peak) {
        max_core_J = std::max(max_core_J, dJ);
        max_core_I = std::max(max_core_I, dI);
        ++n_core;
      }
      if (F.I[i] > 1.0e-9 * F.I_peak) {
        max_tail_J = std::max(max_tail_J, dJ);
        max_tail_I = std::max(max_tail_I, dI);
        ++n_tail;
      }
    }
    verdict(max_core_J <= 1.0e-6, name + ": |dJ/J| where I_ref > 1e-3 I_peak_ref (" + std::to_string(n_core) + " nodes)",
            max_core_J, 1.0e-6);
    verdict(max_core_I <= 1.0e-6, name + ": |dI/I| where I_ref > 1e-3 I_peak_ref", max_core_I, 1.0e-6);
    verdict(max_tail_J <= 1.0e-3, name + ": |dJ/J| where I_ref > 1e-9 I_peak_ref (" + std::to_string(n_tail) + " nodes)",
            max_tail_J, 1.0e-3);
    verdict(max_tail_I <= 1.0e-3, name + ": |dI/I| where I_ref > 1e-9 I_peak_ref", max_tail_I, 1.0e-3);
    std::printf("(diag) %s: over all %zu nodes |dJ/J| %.3e, |dI/I| %.3e; |dDmean/Dmean| %.3e, |deps_mean/eps_mean| %.3e\n",
                name.c_str(), F.node_count, max_all_J, max_all_I, max_D, max_eps);
    if (ref.source == arch::bh::QkSource::Born) {
      // contract 1.5: J and eps_mean do not depend on q_k; I changes by the fit budget
      verdict(max_dJ_paths <= 1.0e-12, name + ": J of the Born path = J of the fit path at every reference node",
              max_dJ_paths, 1.0e-12);
      verdict(max_deps_paths <= 1.0e-12, name + ": eps_mean of the Born path = eps_mean of the fit path at every node",
              max_deps_paths, 1.0e-12);
      verdict(max_fit_budget <= 1.0e-2,
              name + ": A5-fit-budget [labelled anchor]: |I_Born/I_fit - 1| where I_ref > 1e-3 I_peak (worst at Gamma_p " +
                std::to_string(gamma_fit_budget) + ")",
              max_fit_budget, 1.0e-2);
    }
  }

} // namespace

int main() {
  const std::vector<Reference> refs = {
    { "bhref-0.1.0", "frozen:0.1.0", "chodorowski-3.7", arch::bh::QkSource::Fit },
    { "bhref-0.3.0", "frozen:0.3.0", "born-3D2000", arch::bh::QkSource::Born },
  };
  try {
    for (const auto& ref : refs) {
      const std::string dir  = std::string(BH_TEST_DATA_DIR) + "/" + ref.dir;
      const auto        sums = read_sha256sums(dir + "/SHA256SUMS");
      std::printf("==== reference %s (C++ %s) ====\n", ref.dir.c_str(), arch::bh::qk_source_name(ref.source));
      for (const char* name : { "mono05", "mono03", "powerlaw", "blackbody", "mono05_mu100" }) {
        const std::string file = std::string(name) + ".txt";
        if (!sums.count(file)) {
          throw std::runtime_error(file + " not listed in " + dir + "/SHA256SUMS");
        }
        compare_file(ref, dir, file, sums.at(file));
      }
    }
  } catch (const std::exception& e) {
    std::printf("FAIL: exception: %s\n", e.what());
    ++n_fail;
  }
  std::printf("bh_fixtures summary: %d passed, %d failed\n", n_pass, n_fail);
  return n_fail > 0 ? 1 : 0;
}
