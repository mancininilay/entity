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
pair_species = [2, 3]
gamma_p_typical = 6.0
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

int main(int argc, char** argv) {
  const std::string mpi_inputs = argc > 1 ? argv[1] : "";
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
    // ---- step 3 (pair plumbing): defaults and rejections -------------------------------------
    {
      const auto s = user::bhproto::parse_setup(parse_toml(VALID));
      const bool ok = s.source == "budget" && s.spectrum == "delta" && s.pair_minus == 2 && s.pair_plus == 3 &&
                      s.r_th_factor == 10.0 && s.f_r == 0.02 && s.t_latency_auto && s.pair_weight == 1.0 &&
                      s.max_events_per_step == 8 && s.on_capacity == "error" && s.on_capacity_estimate == "error" &&
                      s.on_max_events == "error" && s.gamma_p_typical == 6.0 && s.active_fraction == 1.0 &&
                      !s.has_envelope && s.c_layer == 3.0 && s.comm_headroom == 0.2 && !s.test_nan_payloads &&
                      s.test_kick == 0.0 && !s.test_gauss && s.background_ppc == 0.0 && s.background_n_global == 0;
      std::printf("%s: step-3 defaults (source budget, spectrum delta, pair_species [2, 3], r_th_factor 10, f_r 0.02, "
                  "t_latency auto, pair_weight 1, max_events_per_step 8, on_* error, active_fraction 1, c_layer 3, "
                  "comm_headroom 0.2, test hooks off)\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    {
      const auto s = user::bhproto::parse_setup(parse_toml(
        drop_line(drop_line(VALID, "pair_species"), "gamma_p_typical") + std::string("source = \"none\"\n")));
      const bool ok = s.source == "none" && s.pair_minus == 0 && !s.pairs_enabled();
      std::printf("%s: source = none accepted without pair_species and without gamma_p_typical\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    {
      const auto s = user::bhproto::parse_setup(parse_toml(
        drop_line(VALID, "gamma_p_typical") + std::string("source = \"poisson\"\npair_weight = 2.5\nt_latency = 3.0\n"
                                                           "spectrum = \"threshold\"\nmax_events_per_step = 3\n")));
      const bool ok = s.source == "poisson" && s.pair_weight == 2.5 && !s.t_latency_auto && s.t_latency == 3.0 &&
                      s.spectrum == "threshold" && s.max_events_per_step == 3 && s.pairs_enabled();
      std::printf("%s: poisson accepted without gamma_p_typical; t_latency number, threshold spectrum, "
                  "max_events_per_step read\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    {
      const auto s = user::bhproto::parse_setup(parse_toml(
        drop_line(VALID, "gamma_p_typical") + std::string("envelope.index = 1.5\nenvelope.gamma_min = 3.0\n"
                                                           "envelope.gamma_max = 12.0\nt_latency = \"auto\"\n"
                                                           "on_capacity = \"warn\"\nactive_fraction = 0.5\n")));
      const bool ok = s.has_envelope && s.envelope_index == 1.5 && s.envelope_gamma_min == 3.0 &&
                      s.envelope_gamma_max == 12.0 && s.t_latency_auto && s.on_capacity == "warn" &&
                      s.active_fraction == 0.5 && s.gamma_p_typical == 0.0;
      std::printf("%s: envelope accepted in place of gamma_p_typical; t_latency = \"auto\"; on_capacity = warn\n",
                  ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    expect_reject("source unknown", VALID + std::string("source = \"exact\"\n"), "source");
    expect_reject("spectrum unknown", VALID + std::string("spectrum = \"table\"\n"), "spectrum");
    expect_reject("pair_species missing with budget", drop_line(VALID, "pair_species"), "pair_species");
    expect_reject("pair_species missing with poisson",
                  drop_line(VALID, "pair_species") + std::string("source = \"poisson\"\n"), "pair_species");
    expect_reject("pair_species with one element", replace_line(VALID, "pair_species", "pair_species = [2]"), "pair_species");
    expect_reject("pair_species index 0", replace_line(VALID, "pair_species", "pair_species = [0, 3]"), "pair_species");
    expect_reject("pair_species index 256", replace_line(VALID, "pair_species", "pair_species = [2, 256]"), "pair_species");
    expect_reject("pair_species equal to the ion species", replace_line(VALID, "pair_species", "pair_species = [1, 3]"),
                  "pair_species");
    expect_reject("pair_species equal to each other", replace_line(VALID, "pair_species", "pair_species = [2, 2]"),
                  "pair_species");
    expect_reject("pair_species not integers", replace_line(VALID, "pair_species", "pair_species = [2.0, 3.0]"),
                  "pair_species");
    expect_reject("r_th_factor = 0", VALID + std::string("r_th_factor = 0.0\n"), "r_th_factor");
    expect_reject("r_th_factor negative", VALID + std::string("r_th_factor = -1.0\n"), "r_th_factor");
    expect_reject("f_r = 0", VALID + std::string("f_r = 0.0\n"), "f_r");
    expect_reject("f_r = 1", VALID + std::string("f_r = 1.0\n"), "f_r");
    expect_reject("t_latency negative", VALID + std::string("t_latency = -0.5\n"), "t_latency");
    expect_reject("t_latency not auto", VALID + std::string("t_latency = \"car\"\n"), "t_latency");
    expect_reject("pair_weight = 0", VALID + std::string("pair_weight = 0.0\n"), "pair_weight");
    expect_reject("max_events_per_step = 0", VALID + std::string("max_events_per_step = 0\n"), "max_events_per_step");
    expect_reject("max_events_per_step = 4294967296", VALID + std::string("max_events_per_step = 4294967296\n"),
                  "max_events_per_step");
    expect_reject("on_capacity invalid", VALID + std::string("on_capacity = \"abort\"\n"), "on_capacity");
    expect_reject("on_capacity_estimate invalid", VALID + std::string("on_capacity_estimate = \"ignore\"\n"),
                  "on_capacity_estimate");
    expect_reject("on_max_events invalid", VALID + std::string("on_max_events = true\n"), "on_max_events");
    expect_reject("gamma_p_typical = 1", replace_line(VALID, "gamma_p_typical", "gamma_p_typical = 1.0"), "gamma_p_typical");
    expect_reject("neither gamma_p_typical nor envelope with budget", drop_line(VALID, "gamma_p_typical"), "gamma_p_typical");
    expect_reject("active_fraction = 0", VALID + std::string("active_fraction = 0.0\n"), "active_fraction");
    expect_reject("active_fraction > 1", VALID + std::string("active_fraction = 1.5\n"), "active_fraction");
    expect_reject("envelope incomplete", VALID + std::string("envelope.index = 1.0\n"), "envelope.index");
    expect_reject("envelope.gamma_min = 1", VALID + std::string("envelope.index = 1.0\nenvelope.gamma_min = 1.0\n"
                                                                 "envelope.gamma_max = 5.0\n"), "envelope.gamma_min");
    expect_reject("envelope.gamma_min >= gamma_max", VALID + std::string("envelope.index = 1.0\nenvelope.gamma_min = 5.0\n"
                                                                          "envelope.gamma_max = 5.0\n"), "envelope.gamma_max");
    expect_reject("c_layer = 0", VALID + std::string("c_layer = 0.0\n"), "c_layer");
    expect_reject("comm_headroom negative", VALID + std::string("comm_headroom = -0.1\n"), "comm_headroom");
    expect_reject("test.kick negative", VALID + std::string("test.kick = -1.0\n"), "test.kick");
    expect_reject("test.nan_payloads not boolean", VALID + std::string("test.nan_payloads = 1\n"), "test.nan_payloads");
    expect_reject("background.ppc negative", VALID + std::string("background.ppc = -1.0\n"), "background.ppc");
    expect_reject("background.temperature negative", VALID + std::string("background.temperature = -1.0\n"),
                  "background.temperature");
    expect_reject("background.n_global negative", VALID + std::string("background.n_global = -1\n"), "background.n_global");
    expect_reject("background.n_global = 4294967296", VALID + std::string("background.n_global = 4294967296\n"),
                  "background.n_global");
    expect_reject("photon_field.eps_max = 2 (powerlaw)",
                  replace_line(replace_line(VALID, "photon_field.type", "photon_field.type = \"powerlaw\""),
                               "photon_field.eps0", "photon_field.eps_min = 0.1\nphoton_field.eps_max = 2.0\nphoton_field.index = 2.0"),
                  "photon_field.eps_max");
    // ---- weights flag rule (audit F3, 2026-10-29; spec III.7): check_weights_flag ----------
    {
      auto expect_flag = [&](const std::string& label, const std::string& text, bool use_weights,
                             bool expect_refused, const std::string& key) {
        try {
          const auto s = user::bhproto::parse_setup(parse_toml(text));
          user::bhproto::check_weights_flag(s, use_weights);
          if (expect_refused) {
            std::printf("FAIL: flag rule: %s: accepted (expected a refusal naming %s)\n", label.c_str(), key.c_str());
            ++n_fail;
          } else {
            std::printf("PASS: flag rule: %s: accepted\n", label.c_str());
            ++n_pass;
          }
        } catch (const std::invalid_argument& e) {
          const std::string msg = e.what();
          if (!expect_refused) {
            std::printf("FAIL: flag rule: %s: refused (expected acceptance): %s\n", label.c_str(), msg.c_str());
            ++n_fail;
          } else if (msg.find("particles.use_weights") != std::string::npos && msg.find(key) != std::string::npos) {
            std::printf("PASS: flag rule: %s: refused: %s\n", label.c_str(), msg.c_str());
            ++n_pass;
          } else {
            std::printf("FAIL: flag rule: %s: refused but the message does not name particles.use_weights and %s: %s\n",
                        label.c_str(), key.c_str(), msg.c_str());
            ++n_fail;
          }
        }
      };
      const std::string poisson = drop_line(VALID, "gamma_p_typical") + std::string("source = \"poisson\"\n");
      const std::string none    = drop_line(drop_line(VALID, "pair_species"), "gamma_p_typical") +
                               std::string("source = \"none\"\n");
      // refusals with the flag off
      expect_flag("use_weights = false with source = budget", VALID, false, true, "setup.source = budget");
      expect_flag("use_weights = false with source = poisson and pair_weight = 2.5",
                  poisson + "pair_weight = 2.5\n", false, true, "setup.pair_weight = 2.5");
      expect_flag("use_weights = false with source = poisson and pair_weight = 1e-20",
                  poisson + "pair_weight = 1e-20\n", false, true, "setup.pair_weight");
      expect_flag("use_weights = false with ions.weight = 2 (source = none)",
                  replace_line(none, "ions.weight", "ions.weight = 2.0"), false, true, "setup.ions.weight = 2");
      expect_flag("use_weights = false with ions.weight = 0.5 (poisson, pair_weight = 1)",
                  replace_line(poisson, "ions.weight", "ions.weight = 0.5"), false, true, "setup.ions.weight = 0.5");
      // accepted with the flag off: every stored weight is 1
      expect_flag("use_weights = false with source = poisson and pair_weight = 1 (default)", poisson, false, false, "");
      expect_flag("use_weights = false with source = poisson and pair_weight = 1 (explicit)",
                  poisson + "pair_weight = 1.0\n", false, false, "");
      expect_flag("use_weights = false with source = none", none, false, false, "");
      expect_flag("use_weights = false with the stage disabled (ions.weight = 1)",
                  "hadronic.enabled = false\nhadronic.species = [1]\nions.K = [1.0]\nions.per_K = 1\nions.direction = \"+x\"\n",
                  false, false, "");
      // nothing refused with the flag on
      expect_flag("use_weights = true with source = budget", VALID, true, false, "");
      expect_flag("use_weights = true with source = poisson and pair_weight = 2.5",
                  poisson + "pair_weight = 2.5\n", true, false, "");
      expect_flag("use_weights = true with ions.weight = 2", replace_line(VALID, "ions.weight", "ions.weight = 2.0"),
                  true, false, "");
    }
    // C-MPI-3 (round 3): the construction-time guard predicate on every combination of
    // (absorbing, MPI build, domain count, placement): refused only for absorbing boundaries
    // at the post-step placement in an MPI build, for any domain count
    {
      int n_refused = 0, n_checked = 0;
      for (const bool absorbing : { false, true }) {
        for (const bool mpi : { false, true }) {
          for (const unsigned int nd : { 1u, 4u }) {
            for (const std::string placement : { "poststep", "production" }) {
              const bool refused = user::bhproto::poststep_harvest_refused(absorbing, mpi, nd, placement);
              const bool expect  = absorbing && mpi && placement == "poststep";
              ++n_checked;
              n_refused += refused ? 1 : 0;
              if (refused == expect) {
                ++n_pass;
              } else {
                ++n_fail;
                std::printf("FAIL: guard predicate (absorbing %d, mpi %d, ndomains %u, %s): refused %d, expected %d\n",
                            (int)absorbing, (int)mpi, nd, placement.c_str(), (int)refused, (int)expect);
              }
            }
          }
        }
      }
      std::printf("PASS: guard predicate: %d combinations checked, %d refused (absorbing + MPI + poststep, ndomains 1 "
                  "and 4), production placement never refused\n", n_checked, n_refused);
    }
    // the prepared MPI inputs (pgens/bh_proto/inputs/mpi/) are accepted at parse time with the
    // production placement
    if (!mpi_inputs.empty()) {
      for (const std::string name : { "g3_asymmetric_capacity_2ranks", "t14b_absorb_production_4ranks" }) {
        const std::string path = mpi_inputs + "/" + name + ".toml";
        try {
          const auto  file  = toml::parse(path);
          const auto& setup = toml::find(file, "setup");
          const auto  s     = user::bhproto::parse_setup(setup);
          const auto  nd    = toml::find<unsigned int>(file, "simulation", "domain", "number");
          const bool  ok    = s.placement == "production" && s.enabled && s.pairs_enabled() && nd >= 2 &&
                           !user::bhproto::poststep_harvest_refused(true, true, nd, s.placement);
          if (ok) {
            ++n_pass;
            std::printf("PASS: MPI input %s accepted at parse time (production placement, %u domains, source %s)\n",
                        name.c_str(), nd, s.source.c_str());
          } else {
            ++n_fail;
            std::printf("FAIL: MPI input %s parsed but is not a production-placement multi-domain setup\n", name.c_str());
          }
        } catch (const std::exception& e) {
          ++n_fail;
          std::printf("FAIL: MPI input %s rejected: %s\n", name.c_str(), e.what());
        }
      }
    } else {
      std::printf("(no inputs directory given: the prepared MPI inputs are not parsed)\n");
    }
  } catch (const std::exception& e) {
    std::printf("FAIL: unexpected exception: %s\n", e.what());
    ++n_fail;
  }
  std::printf("bh_proto_params summary: %d passed, %d failed\n", n_pass, n_fail);
  return n_fail > 0 ? 1 : 0;
}
