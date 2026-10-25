/**
 * @file tests/hadronic/bh_proto_params.cpp
 * @brief Rejection tests of the bh_proto `[setup]` reader (brief section 2,
 *        "Rejection tests"): every parameter check refuses with a message
 *        naming the key, and a valid setup parses with the documented values.
 *        Host-only, no Kokkos.
 */

#include "../../pgens/bh_proto/bh_proto_params.h"

#include <toml11/toml.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

  int n_pass = 0, n_fail = 0;

  const char* VALID = R"(
hadronic.enabled = true
hadronic.species = [1]
tau0 = 1.0
physical_mass_ratio = 1836.15
photon_field.type = "mono"
photon_field.eps0 = 0.5
eta_u = 1e-4
pending_floor_ulp = 100
f_max = 0.01
max_substeps = 64
ions.K = [9180.75, 3672.3]
ions.per_K = 2
ions.direction = "+x"
ions.weight = 1.0
fields.B = [0.0, 0.0, 0.0]
ledger.every = 5
)";

  toml::value parse_toml(const std::string& text) {
    return toml::parse_str(text);
  }

  /**
   * @brief expect parse_setup to throw with a message naming setup.<key>
   */
  void expect_reject(const std::string& label, const std::string& text, const std::string& key) {
    try {
      const auto s = user::bhproto::parse_setup(parse_toml(text));
      (void)s;
      std::printf("FAIL: %s: accepted (expected a refusal naming setup.%s)\n", label.c_str(), key.c_str());
      ++n_fail;
    } catch (const std::invalid_argument& e) {
      const std::string msg = e.what();
      if (msg.find("setup." + key) != std::string::npos) {
        std::printf("PASS: %s: refused: %s\n", label.c_str(), msg.c_str());
        ++n_pass;
      } else {
        std::printf("FAIL: %s: refused but the message does not name setup.%s: %s\n", label.c_str(),
                    key.c_str(), msg.c_str());
        ++n_fail;
      }
    }
  }

  std::string replace_line(const std::string& text, const std::string& key, const std::string& new_line) {
    std::string out;
    std::size_t pos = 0;
    bool        done = false;
    while (pos < text.size()) {
      const auto  nl   = text.find('\n', pos);
      const auto  end  = nl == std::string::npos ? text.size() : nl;
      std::string line = text.substr(pos, end - pos);
      if (line.rfind(key + " ", 0) == 0 || line.rfind(key + "=", 0) == 0) {
        line = new_line;
        done = true;
      }
      out += line + "\n";
      pos  = end + 1;
    }
    if (!done) {
      out += new_line + "\n";
    }
    return out;
  }

  std::string drop_line(const std::string& text, const std::string& key) {
    std::string out;
    std::size_t pos = 0;
    while (pos < text.size()) {
      const auto  nl   = text.find('\n', pos);
      const auto  end  = nl == std::string::npos ? text.size() : nl;
      std::string line = text.substr(pos, end - pos);
      if (!(line.rfind(key + " ", 0) == 0 || line.rfind(key + "=", 0) == 0)) {
        out += line + "\n";
      }
      pos = end + 1;
    }
    return out;
  }

} // namespace

int main() {
  try {
    // the valid setup parses with the given values
    {
      const auto s = user::bhproto::parse_setup(parse_toml(VALID));
      const bool ok = s.enabled && s.species.size() == 1 && s.species[0] == 1 && s.tau0 == 1.0 &&
                      s.mu_r == 1836.15 && s.bath.type == "mono" && s.bath.eps0 == 0.5 && s.eta_u == 1e-4 &&
                      s.pending_floor_ulp == 100.0 && s.f_max == 0.01 && s.max_substeps == 64 &&
                      s.ions_K.size() == 2 && s.ions_K[0] == 9180.75 && s.ions_per_K == 2 &&
                      s.ions_direction == "+x" && s.ions_weight == 1.0 && s.ledger_every == 5 &&
                      s.payload_pending == 0 && s.payload_debt == 1 && s.payload_debt_k0 == 2 &&
                      !s.allow_unsafe_application && !s.design_off() && s.gamma_p_max == 0.0;
      std::printf("%s: valid setup parses with the documented values\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    // defaults when the optional keys are absent
    {
      std::string t = VALID;
      for (const char* k : { "eta_u", "pending_floor_ulp", "f_max", "max_substeps", "ions.weight", "fields.B",
                             "ledger.every" }) {
        t = drop_line(t, k);
      }
      const auto s  = user::bhproto::parse_setup(parse_toml(t));
      const bool ok = s.eta_u == 1e-4 && s.pending_floor_ulp == 100.0 && s.f_max == 0.01 && s.max_substeps == 64 &&
                      s.ions_weight == 1.0 && s.B[0] == 0.0 && s.B[2] == 0.0 && s.ledger_every == 1 && s.table_dump;
      std::printf("%s: optional keys take the documented defaults\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    // disabled stage: only the ion keys are required
    {
      const auto s = user::bhproto::parse_setup(parse_toml(
        "hadronic.enabled = false\nhadronic.species = [1]\nions.K = [1.0]\nions.per_K = 1\nions.direction = \"isotropic\"\n"));
      const bool ok = !s.enabled && s.ions_direction == "isotropic";
      std::printf("%s: disabled stage needs only the ion keys\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    // integers accepted where floats are expected (exact conversion)
    {
      const auto s = user::bhproto::parse_setup(parse_toml(replace_line(VALID, "tau0", "tau0 = 2")));
      const bool ok = s.tau0 == 2.0;
      std::printf("%s: a TOML integer is accepted for a float key (tau0 = 2)\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    // design off accepted only with the explicit flag
    {
      std::string t = replace_line(VALID, "eta_u", "eta_u = 0.0");
      t             = replace_line(t, "pending_floor_ulp", "pending_floor_ulp = 0.0");
      t             = replace_line(t, "allow_unsafe_application", "allow_unsafe_application = true");
      const auto s  = user::bhproto::parse_setup(parse_toml(t));
      const bool ok = s.design_off() && s.allow_unsafe_application;
      std::printf("%s: design off accepted with allow_unsafe_application = true\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }

    // ---- the pre-registered rejections -------------------------------------
    expect_reject("negative tau0", replace_line(VALID, "tau0", "tau0 = -1.0"), "tau0");
    expect_reject("tau0 not a number", replace_line(VALID, "tau0", "tau0 = \"1.0\""), "tau0");
    expect_reject("mu_r <= 0", replace_line(VALID, "physical_mass_ratio", "physical_mass_ratio = 0.0"),
                  "physical_mass_ratio");
    expect_reject("mu_r negative", replace_line(VALID, "physical_mass_ratio", "physical_mass_ratio = -5"),
                  "physical_mass_ratio");
    expect_reject("eps0 >= 2 (mono)", replace_line(VALID, "photon_field.eps0", "photon_field.eps0 = 2.0"),
                  "photon_field.eps0");
    {
      std::string t = replace_line(VALID, "photon_field.type", "photon_field.type = \"powerlaw\"");
      t             = drop_line(t, "photon_field.eps0");
      t += "photon_field.eps_min = 0.1\nphoton_field.eps_max = 2.0\nphoton_field.index = 2.0\n";
      expect_reject("eps_max >= 2 (powerlaw)", t, "photon_field.eps_max");
    }
    {
      std::string t = replace_line(VALID, "photon_field.type", "photon_field.type = \"blackbody\"");
      t             = drop_line(t, "photon_field.eps0");
      t += "photon_field.theta = 0.1\nphoton_field.eps_max = 2.5\n";
      expect_reject("eps_max >= 2 (blackbody)", t, "photon_field.eps_max");
    }
    expect_reject("f_max = 0", replace_line(VALID, "f_max", "f_max = 0.0"), "f_max");
    expect_reject("f_max > 1", replace_line(VALID, "f_max", "f_max = 1.5"), "f_max");
    expect_reject("max_substeps = 0", replace_line(VALID, "max_substeps", "max_substeps = 0"), "max_substeps");
    expect_reject("max_substeps not an integer", replace_line(VALID, "max_substeps", "max_substeps = 2.0"),
                  "max_substeps");
    expect_reject("unknown photon_field.type", replace_line(VALID, "photon_field.type", "photon_field.type = \"laser\""),
                  "photon_field.type");
    expect_reject("missing photon_field.type", drop_line(VALID, "photon_field.type"), "photon_field.type");
    expect_reject("allow_unsafe_application absent with eta_u = 0", replace_line(VALID, "eta_u", "eta_u = 0.0"),
                  "allow_unsafe_application");
    expect_reject("allow_unsafe_application absent with pending_floor_ulp = 0",
                  replace_line(VALID, "pending_floor_ulp", "pending_floor_ulp = 0"), "allow_unsafe_application");
    expect_reject("allow_unsafe_application = false with eta_u = 0",
                  replace_line(VALID, "eta_u", "eta_u = 0.0\nallow_unsafe_application = false"),
                  "allow_unsafe_application");
    expect_reject("ions.K empty", replace_line(VALID, "ions.K", "ions.K = []"), "ions.K");
    expect_reject("ions.K non-positive", replace_line(VALID, "ions.K", "ions.K = [9180.75, 0.0]"), "ions.K");
    expect_reject("ions.K negative", replace_line(VALID, "ions.K", "ions.K = [-1.0]"), "ions.K");
    expect_reject("ions.K not an array", replace_line(VALID, "ions.K", "ions.K = 5.0"), "ions.K");
    expect_reject("ions.K missing", drop_line(VALID, "ions.K"), "ions.K");
    expect_reject("ions.K with a string", replace_line(VALID, "ions.K", "ions.K = [1.0, \"a\"]"), "ions.K");
    expect_reject("non-boolean where a boolean is required (hadronic.enabled = 1)",
                  replace_line(VALID, "hadronic.enabled", "hadronic.enabled = 1"), "hadronic.enabled");
    expect_reject("non-boolean where a boolean is required (hadronic.enabled = \"true\")",
                  replace_line(VALID, "hadronic.enabled", "hadronic.enabled = \"true\""), "hadronic.enabled");
    expect_reject("non-boolean allow_unsafe_application",
                  replace_line(VALID, "allow_unsafe_application", "allow_unsafe_application = 1"),
                  "allow_unsafe_application");
    expect_reject("hadronic.enabled missing", drop_line(VALID, "hadronic.enabled"), "hadronic.enabled");
    expect_reject("hadronic.species empty", replace_line(VALID, "hadronic.species", "hadronic.species = []"),
                  "hadronic.species");
    expect_reject("hadronic.species index 0", replace_line(VALID, "hadronic.species", "hadronic.species = [0]"),
                  "hadronic.species");
    expect_reject("hadronic.species duplicate", replace_line(VALID, "hadronic.species", "hadronic.species = [1, 1]"),
                  "hadronic.species");
    expect_reject("ions.per_K = 0", replace_line(VALID, "ions.per_K", "ions.per_K = 0"), "ions.per_K");
    expect_reject("ions.direction unknown", replace_line(VALID, "ions.direction", "ions.direction = \"-x\""),
                  "ions.direction");
    expect_reject("ions.weight <= 0", replace_line(VALID, "ions.weight", "ions.weight = 0.0"), "ions.weight");
    expect_reject("fields.B wrong length", replace_line(VALID, "fields.B", "fields.B = [1.0, 0.0]"), "fields.B");
    expect_reject("ledger.every = 0", replace_line(VALID, "ledger.every", "ledger.every = 0"), "ledger.every");
    expect_reject("eta_u negative", replace_line(VALID, "eta_u", "eta_u = -1e-4"), "eta_u");
    expect_reject("pending_floor_ulp negative", replace_line(VALID, "pending_floor_ulp", "pending_floor_ulp = -1"),
                  "pending_floor_ulp");
    expect_reject("gamma_p_max <= 1", replace_line(VALID, "gamma_p_max", "gamma_p_max = 1.0"), "gamma_p_max");
    expect_reject("gamma_p_max bad string", replace_line(VALID, "gamma_p_max", "gamma_p_max = \"max\""), "gamma_p_max");
    expect_reject("payload slots not distinct", replace_line(VALID, "payload_debt", "payload_debt = 0"),
                  "payload_pending");
    // R1 regressions: 64-bit values validated before any narrowing
    expect_reject("payload_pending = 4294967295 (would narrow to -1 / 65535)",
                  VALID + std::string("payload_pending = 4294967295\n"), "payload_pending");
    expect_reject("payload_pending = 4294967296 with payload_debt = 0 (would collapse to slot 0)",
                  VALID + std::string("payload_pending = 4294967296\npayload_debt = 0\n"), "payload_pending");
    expect_reject("ledger.every = 4294967296 (would narrow to 0)",
                  replace_line(VALID, "ledger.every", "ledger.every = 4294967296"), "ledger.every");
    expect_reject("ledger.every = 2147483648 (INT_MAX + 1)",
                  replace_line(VALID, "ledger.every", "ledger.every = 2147483648"), "ledger.every");
    expect_reject("payload slot 65535 (count would overflow unsigned short)",
                  VALID + std::string("payload_debt_k0 = 65535\n"), "payload_debt_k0");
    expect_reject("ions.per_K = 2147483648", replace_line(VALID, "ions.per_K", "ions.per_K = 2147483648"), "ions.per_K");
    expect_reject("ions.per_K x ions.K.size() > INT_MAX", replace_line(VALID, "ions.per_K", "ions.per_K = 1073741824"),
                  "ions.per_K");
    expect_reject("max_substeps = 4294967296", replace_line(VALID, "max_substeps", "max_substeps = 4294967296"),
                  "max_substeps");
    {
      const auto s = user::bhproto::parse_setup(parse_toml(
        replace_line(VALID + std::string("payload_debt_k0 = 65534\n"), "ledger.every", "ledger.every = 2147483647")));
      const bool ok = s.payload_debt_k0 == 65534 && s.ledger_every == 2147483647;
      std::printf("%s: the largest admissible slot (65534) and ledger.every = INT_MAX are accepted\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    expect_reject("unknown key", VALID + std::string("tau = 1.0\n"), "tau");
    expect_reject("unknown nested key", VALID + std::string("hadronic.enable = true\n"), "hadronic.enable");
    expect_reject("tau0 missing when enabled", drop_line(VALID, "tau0"), "tau0");
    expect_reject("mu_r missing when enabled", drop_line(VALID, "physical_mass_ratio"), "physical_mass_ratio");
    expect_reject("table bath without file",
                  drop_line(replace_line(VALID, "photon_field.type", "photon_field.type = \"table\""),
                            "photon_field.eps0"),
                  "photon_field.file");
  } catch (const std::exception& e) {
    std::printf("FAIL: unexpected exception: %s\n", e.what());
    ++n_fail;
  }
  std::printf("bh_proto_params summary: %d passed, %d failed\n", n_pass, n_fail);
  return n_fail > 0 ? 1 : 0;
}
