/**
 * @file tests/hadronic/bh_proto_params.cpp
 * @brief Rejection tests of the bh_proto `[setup]` reader (brief section 2,
 *        "Rejection tests"): every parameter check refuses with a message
 *        naming the key, and a valid setup parses with the documented values.
 *        Host-only, no Kokkos.
 */

#include "../../pgens/bh_proto/bh_proto_params.h"

#include "archetypes/hadronic/bh_event_host.h"

#include <toml11/toml.hpp>

#include <cmath>
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
                      s.on_max_events == "error" && s.on_bound == "error" && s.mode == "continuous" &&
                      !s.event_mode() && s.gamma_p_typical == 6.0 && s.active_fraction == 1.0 &&
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
    // ---- step 5 (event mode, contract 2026-10-29 section 3): T23 additions -------------------
    {
      const auto s  = user::bhproto::parse_setup(parse_toml(VALID + std::string("mode = \"event\"\n")));
      const bool ok = s.mode == "event" && s.event_mode() && s.pairs_enabled() && s.source == "none" &&
                      s.pair_minus == 2 && s.pair_plus == 3 && s.on_bound == "error" && s.pair_weight == 1.0 &&
                      s.f_max == 0.01 && s.max_events_per_step == 8;
      std::printf("%s: mode = event accepted without source: event_mode, pairs_enabled, source none, on_bound error, "
                  "pair_weight 1, f_max 0.01, max_events_per_step 8\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    for (const char* spelling : { "Event", "EVENT", "eVeNt" }) {
      const auto s  = user::bhproto::parse_setup(parse_toml(VALID + std::string("mode = \"") + spelling + "\"\n"));
      const bool ok = s.mode == "event" && s.event_mode();
      std::printf("%s: mode = \"%s\" is accepted case-insensitively and stored as \"event\"\n", ok ? "PASS" : "FAIL",
                  spelling);
      ok ? ++n_pass : ++n_fail;
    }
    {
      const auto s  = user::bhproto::parse_setup(parse_toml(VALID + std::string("mode = \"Continuous\"\n")));
      const bool ok = s.mode == "continuous" && !s.event_mode() && s.source == "budget";
      std::printf("%s: mode = \"Continuous\" is stored as \"continuous\" and keeps source = budget\n",
                  ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    {
      // the continuous-mode keys are accepted and ignored with mode = event
      std::string t = replace_line(VALID, "eta_u", "eta_u = 1e-3");
      t             = replace_line(t, "pending_floor_ulp", "pending_floor_ulp = 10");
      t             = replace_line(t, "f_max", "f_max = 0.02");
      const auto s  = user::bhproto::parse_setup(parse_toml(
        t + std::string("mode = \"event\"\nr_th_factor = 5.0\nf_r = 0.05\nt_latency = 2.0\n"
                        "on_bound = \"warn\"\npair_weight = 2.0\nspectrum = \"restframe\"\n")));
      const bool ok = s.event_mode() && s.on_bound == "warn" && s.pair_weight == 2.0 && s.f_max == 0.02 &&
                      s.spectrum == "restframe" && s.r_th_factor == 5.0 && s.f_r == 0.05 && s.t_latency == 2.0 &&
                      s.eta_u == 1e-3 && s.pending_floor_ulp == 10.0;
      std::printf("%s: event mode accepts the continuous keys, on_bound = warn, pair_weight, f_max, spectrum\n",
                  ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    {
      const auto s = user::bhproto::parse_setup(parse_toml(
        drop_line(VALID, "gamma_p_typical") + std::string("mode = \"event\"\nenvelope.index = 2.0\n"
                                                           "envelope.gamma_min = 2.5\nenvelope.gamma_max = 8.0\n")));
      const bool ok = s.event_mode() && s.has_envelope && s.gamma_p_typical == 0.0;
      std::printf("%s: event mode accepts the envelope in place of gamma_p_typical\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    expect_reject("mode unknown", VALID + std::string("mode = \"exact\"\n"), "mode");
    expect_reject("mode not a string", VALID + std::string("mode = 1\n"), "mode");
    expect_reject("mode = event with source = poisson", VALID + std::string("mode = \"event\"\nsource = \"poisson\"\n"),
                  "source");
    expect_reject("mode = event with source = budget", VALID + std::string("mode = \"event\"\nsource = \"budget\"\n"),
                  "source");
    expect_reject("mode = event with source = none", VALID + std::string("mode = \"event\"\nsource = \"none\"\n"),
                  "source");
    expect_reject("mode = event without pair_species", drop_line(VALID, "pair_species") + std::string("mode = \"event\"\n"),
                  "pair_species");
    expect_reject("mode = event without gamma_p_typical or envelope",
                  drop_line(VALID, "gamma_p_typical") + std::string("mode = \"event\"\n"), "gamma_p_typical");
    expect_reject("mode = event with pair_weight = 0", VALID + std::string("mode = \"event\"\npair_weight = 0.0\n"),
                  "pair_weight");
    expect_reject("on_bound invalid", VALID + std::string("on_bound = \"abort\"\n"), "on_bound");
    expect_reject("on_bound not a string", VALID + std::string("on_bound = 1\n"), "on_bound");
    // the restart predicate (contract section 3): refused only in the event mode with a non-zero payload
    {
      const bool ok = !user::bhproto::event_restart_refused("event", 0) &&
                      user::bhproto::event_restart_refused("event", 1) &&
                      user::bhproto::event_restart_refused("event", 1000000ULL) &&
                      !user::bhproto::event_restart_refused("continuous", 5);
      std::printf("%s: event_restart_refused: event with 0 payloads clean, event with >= 1 refused, continuous never\n",
                  ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    // the closed-form D_max,unweighted (spec III.2) against a brute-force maximum over the
    // kinematic domain (eps' in [eps'_lo, eps'_hi], |p'_z| <= sqrt(eps'^2 - 4)) at fixed eps:
    // the grid maximum never exceeds the closed form and approaches it with the grid
    {
      namespace ev = arch::bh::event;
      bool ok = true;
      for (const double eps : { 0.3, 0.5, 1.0 }) {
        for (const double G : { 2.5, 3.0, 6.0, 10.0, 50.0 }) {
          const double gb    = std::sqrt((G - 1.0) * (G + 1.0));
          const double k_hi  = G * eps + gb * eps;
          const double k_lo  = std::max(2.0, G * eps - gb * eps);
          const double Dref  = ev::dmax_unweighted(G, eps);
          if (!(k_hi >= 2.0)) {
            ok = ok && Dref == 0.0;
            continue;
          }
          double Dgrid = -std::numeric_limits<double>::infinity();
          const int nk = 4000, np = 400;
          for (int i = 0; i <= nk; ++i) {
            const double k  = k_lo + (k_hi - k_lo) * static_cast<double>(i) / nk;
            const double pm = std::sqrt(std::max(0.0, (k - 2.0) * (k + 2.0)));
            for (int j = 0; j <= np; ++j) {
              const double pz = -pm + 2.0 * pm * static_cast<double>(j) / np;
              Dgrid           = std::max(Dgrid, G * k + gb * pz - eps); // D = Gamma (eps' + beta p'_z) - eps
            }
          }
          const bool here = Dgrid <= Dref * (1.0 + 1e-12) && Dgrid >= Dref * (1.0 - 1e-6);
          if (!here) {
            std::printf("FAIL: D_max closed form at eps = %g, Gamma = %g: grid %.12g, closed form %.12g\n", eps, G, Dgrid,
                        Dref);
          }
          ok = ok && here;
        }
      }
      // the contract's orientation numbers (mono eps0 = 0.5, mu_r = 1836.15, f_max = 0.01) are
      // reported, not asserted: they were computed with <D>, not D_max (see the report)
      std::printf("%s: D_max,unweighted closed form equals the brute-force maximum over (eps', p'_z) at fixed eps "
                  "(15 cases, 4001 x 401 grid); a_max(3) = %.4g, a_max(6) = %.4g, a_max(10) = %.4g at eps0 = 0.5, "
                  "mu_r = 1836.15, f_max = 0.01\n",
                  ok ? "PASS" : "FAIL", ev::a_max(3.0, 0.5, 1836.15, 0.01), ev::a_max(6.0, 0.5, 1836.15, 0.01),
                  ev::a_max(10.0, 0.5, 1836.15, 0.01));
      ok ? ++n_pass : ++n_fail;
    }
    // the shape of a_max(Gamma) (documented in bh_event_host.h; the pgen's active-band check does not
    // rely on it): monotone decrease above the threshold for eps <= 1, and for Gamma >= 2 in every
    // case; for eps = 1.9 the maximum lies between the threshold and Gamma = 2 (measured here)
    {
      namespace ev = arch::bh::event;
      bool ok = true;
      for (const double eps : { 0.3, 0.5, 1.0, 1.9 }) {
        const double gth  = 1.0 / eps + 0.25 * eps;
        double       prev = std::numeric_limits<double>::infinity(), am_max = 0.0, g_max = 0.0;
        bool         mono_all = true, mono_above2 = true;
        for (int i = 1; i <= 400; ++i) {
          const double G  = gth * std::pow(100.0 / gth, static_cast<double>(i) / 400);
          const double am = ev::a_max(G, eps, 1836.15, 0.01);
          if (am > am_max) {
            am_max = am;
            g_max  = G;
          }
          if (!(am <= prev)) {
            mono_all = false;
            if (G >= 2.0) {
              mono_above2 = false;
            }
          }
          prev = am;
        }
        const bool here = mono_above2 && (eps > 1.0 || mono_all) && (eps <= 1.0 || (g_max > gth && g_max < 2.0));
        std::printf("%s: a_max(Gamma) at eps = %g: Gamma_th = %.6g, maximum %.6g at Gamma = %.6g, monotone decrease "
                    "from the threshold: %s, for Gamma >= 2: %s\n",
                    here ? "PASS" : "FAIL", eps, gth, am_max, g_max, mono_all ? "yes" : "no", mono_above2 ? "yes" : "no");
        ok = ok && here;
      }
      ok ? ++n_pass : ++n_fail;
    }
    // the ulp-energy of the stored momentum: float against double at the same K (ratio 2^29 for the
    // same binade), and the double value against ulp(u) mu u / gamma from std::nextafter directly
    {
      namespace ev = arch::bh::event;
      bool ok = true;
      for (const double mu_d : { 1.0, 1836.15 }) {
        for (const double K : { 3672.3, 9180.75, 1.0e5 }) {
          const double ef = ev::ulp_energy<float>(K, mu_d);
          const double ed = ev::ulp_energy<double>(K, mu_d);
          const double u  = std::sqrt(K * (K + 2.0 * mu_d)) / mu_d;
          const double gd = std::sqrt(1.0 + u * u);
          const double ref = (std::nextafter(u, std::numeric_limits<double>::infinity()) - u) * mu_d * u / gd;
          const double ratio = ef / ed;
          const bool   here  = ed > 0.0 && std::fabs(ed / ref - 1.0) < 1e-12 && ratio > std::ldexp(1.0, 28) &&
                            ratio < std::ldexp(1.0, 30);
          if (!here) {
            std::printf("FAIL: ulp_energy at K = %g, mu_d = %g: float %.6e, double %.6e, ref %.6e, ratio %.6g\n", K,
                        mu_d, ef, ed, ref, ratio);
          }
          ok = ok && here;
        }
      }
      std::printf("%s: ulp_energy<float> / ulp_energy<double> in (2^28, 2^30) and the double value matches nextafter "
                  "directly (6 cases)\n", ok ? "PASS" : "FAIL");
      ok ? ++n_pass : ++n_fail;
    }
    // table_numbers on a synthetic host table: lambda_max and the ulp ratio pick the right nodes,
    // nodes with J = 0 or <D> = 0 are skipped, and the double instantiation is exercised
    {
      namespace ev = arch::bh::event;
      arch::bh::HadronicTablesHost T;
      T.mu_r  = 1836.15;
      T.gamma = { 2.2, 3.0, 6.0, 10.0, 30.0 };
      T.J     = { 0.0, 1.0e-3, 4.0e-3, 2.0e-3, 5.0e-3 };
      T.Dmean = { 0.0, 6.0, 12.0, 20.0, 60.0 };
      T.I     = { 0.0, 6.0e-3, 4.8e-2, 4.0e-2, 3.0e-1 };
      const double coeff = 2.0, dt = 0.5, a = 3.0;
      const auto nf = ev::table_numbers<float>(T, coeff, dt, a, 1.0);
      const auto nd = ev::table_numbers<double>(T, coeff, dt, a, 1.0);
      // lambda at the nodes: 0, 1e-3/3, 4e-3/3, 2e-3/3, 5e-3/3 -> max at Gamma = 30
      const bool lam_ok = std::fabs(nf.lambda_max - coeff * 5.0e-3 * dt / a) < 1e-15 && nf.gamma_at_lambda_max == 30.0 &&
                          nd.lambda_max == nf.lambda_max && nd.gamma_at_lambda_max == 30.0;
      // the ulp ratio a <D> / ulp_energy(K) at the four interacting nodes; the minimum is checked
      // against a direct evaluation and must be at the node where a <D> / dK_ulp is smallest
      double ref_min_f = std::numeric_limits<double>::infinity(), ref_min_d = ref_min_f;
      double gf = 0.0, gdbl = 0.0;
      for (std::size_t i = 1; i < T.gamma.size(); ++i) {
        const double K  = T.mu_r * (T.gamma[i] - 1.0);
        const double rf = a * T.Dmean[i] / ev::ulp_energy<float>(K, 1.0);
        const double rd = a * T.Dmean[i] / ev::ulp_energy<double>(K, 1.0);
        if (rf < ref_min_f) { ref_min_f = rf; gf = T.gamma[i]; }
        if (rd < ref_min_d) { ref_min_d = rd; gdbl = T.gamma[i]; }
      }
      const bool ulp_ok = nf.ulp_ratio_min == ref_min_f && nf.gamma_at_ulp_ratio_min == gf &&
                          nd.ulp_ratio_min == ref_min_d && nd.gamma_at_ulp_ratio_min == gdbl &&
                          nd.ulp_ratio_min / nf.ulp_ratio_min > std::ldexp(1.0, 28) && nf.nodes == 5;
      const bool ok = lam_ok && ulp_ok;
      std::printf("%s: table_numbers (synthetic table): lambda_max = %.6g at Gamma = %g (expected %.6g at 30); ulp ratio "
                  "min float %.6g at Gamma = %g, double %.6g at Gamma = %g (J = 0 and <D> = 0 nodes skipped)\n",
                  ok ? "PASS" : "FAIL", nf.lambda_max, nf.gamma_at_lambda_max, coeff * 5.0e-3 * dt / a, nf.ulp_ratio_min,
                  nf.gamma_at_ulp_ratio_min, nd.ulp_ratio_min, nd.gamma_at_ulp_ratio_min);
      ok ? ++n_pass : ++n_fail;
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
