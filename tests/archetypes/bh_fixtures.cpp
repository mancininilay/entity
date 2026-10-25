/**
 * @file tests/archetypes/bh_fixtures.cpp
 * @brief Contract 1.4 item 8 inside ctest: the C++ Bethe-Heitler rates against
 *        the frozen Python reference tables (`bhref` 0.1.0, fixtures under
 *        tests/data/hadronic/bhref-0.1.0/). For each of the five files: the
 *        file hash against SHA256SUMS and the header's data_sha256 (verified by
 *        the reader), the bath rebuilt from the header, `at_gamma` evaluated at
 *        every reference gamma_p, the contract tolerances with masks and peak
 *        from the reference (|dJ/J|, |dI/I| <= 1e-6 where I_ref > 1e-3 I_peak_ref,
 *        <= 1e-3 where I_ref > 1e-9 I_peak_ref), and the C++ node placement
 *        against the reference nodes (y to 1e-9). Host-only, no Kokkos.
 */

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

  void compare_file(const std::string& dir, const std::string& name, const std::string& expected_hash) {
    const std::string path = dir + "/" + name;
    std::printf("---- %s ----\n", name.c_str());
    // file-level manifest
    const std::string file_hash = arch::bh::sha256_of_file(path);
    verdict(file_hash == expected_hash, name + ": file sha256 matches SHA256SUMS", file_hash == expected_hash, 1.0);
    // the reader verifies data_sha256 and the schema (throws otherwise)
    const auto F = arch::bh::read_table_file(path);
    verdict(F.producer == "bhref" && F.status == "frozen:0.1.0",
            name + ": producer bhref, status frozen:0.1.0 (data_sha256 verified by the reader)", 1.0, 1.0);

    // the C++ rates for the bath of the header
    const auto                        spec = bath_from_header(F);
    const auto                        bath = arch::bh::make_bath(spec);
    const arch::bh::BetheHeitlerRates rates(bath);
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
    std::size_t n_core = 0, n_tail = 0;
    for (std::size_t i = 0; i < F.node_count; ++i) {
      const auto   r  = rates.at_gamma(F.gamma[i]);
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
  }

} // namespace

int main() {
  const std::string dir = std::string(BH_TEST_DATA_DIR) + "/bhref-0.1.0";
  try {
    const auto sums = read_sha256sums(dir + "/SHA256SUMS");
    for (const char* name : { "mono05", "mono03", "powerlaw", "blackbody", "mono05_mu100" }) {
      const std::string file = std::string(name) + ".txt";
      if (!sums.count(file)) {
        throw std::runtime_error(file + " not listed in SHA256SUMS");
      }
      compare_file(dir, file, sums.at(file));
    }
  } catch (const std::exception& e) {
    std::printf("FAIL: exception: %s\n", e.what());
    ++n_fail;
  }
  std::printf("bh_fixtures summary: %d passed, %d failed\n", n_pass, n_fail);
  return n_fail > 0 ? 1 : 0;
}
