/**
 * @file pgens/bh_proto/bh_proto_params.h
 * @brief Strict reader of the `[setup]` table of the bh_proto problem generator
 *        (Bethe-Heitler prototype, P2 steps 2 and 3).
 *
 * The reader works on the raw TOML value (`SimulationParams::data()`), not on
 * the flattened `setup.*` container, so that nested tables (`hadronic.enabled`,
 * `photon_field.type`, `ions.K`), empty arrays and wrong types are seen and
 * refused with a message naming the key and the value. Every numeric key is
 * range-checked against its destination type before any cast. No sentinel
 * value can result from a failed conversion: a key is either absent (and takes
 * its documented default, when it has one) or converts completely, or the
 * reader throws.
 *
 * Keys (all under `[setup]`; floats accept TOML integers):
 *   hadronic.enabled            bool, required
 *   hadronic.species            array of exactly one int >= 1, required (the ion species; the
 *                               generator refuses more than one, decision (d))
 *   hadronic.placement          "poststep" | "production" (default poststep): where the engine
 *                               launches the stage (CustomPostStep, or after the deposit and
 *                               before the particle exchange through srpic::HadronicInteractions)
 *   tau0                        float >= 0 (0 = stage not launched), required when enabled
 *   physical_mass_ratio         float > 0, required when enabled
 *   photon_field.type           "mono" | "powerlaw" | "blackbody" | "table", required when enabled
 *   photon_field.eps0 / eps_min / eps_max / index / theta / file   as bh_tables_dump
 *   gamma_p_max                 float > 1 or the string "auto" (default auto)
 *   eta_u                       float >= 0 (default 1e-4)
 *   pending_floor_ulp           float >= 0 (default 100)
 *   f_max                       float in (0, 1] (default 0.01; decision of 2026-10-25)
 *   max_substeps                int in [1, 1e6] (default 64)
 *   allow_unsafe_application    bool (default false); required true when eta_u = 0 or pending_floor_ulp = 0
 *   payload_pending / payload_debt / payload_debt_k0   int in [0, 65534], distinct (default 0, 1, 2)
 *   source                      "budget" | "poisson" | "none" (default budget)
 *   spectrum                    "delta" | "threshold" (default delta)
 *   pair_species                array [i_minus, i_plus] of int in [1, 255], distinct, not an ion
 *                               species; required unless source = none
 *   r_th_factor                 float > 0 (default 10)
 *   f_r                         float in (0, 1) (default 0.02)
 *   t_latency                   "auto" | float >= 0 (default auto; 0 = cap off)
 *   pair_weight                 float > 0 (default 1; poisson)
 *   max_events_per_step         int in [1, 1e6] (default 8)
 *   on_capacity / on_capacity_estimate / on_max_events   "error" | "warn" (default error)
 *   gamma_p_typical             float > 1 or 0 (default 0 = use the envelope)
 *   active_fraction             float in (0, 1] (default 1)
 *   envelope.index              float (finite), envelope.gamma_min > 1, envelope.gamma_max > gamma_min
 *                               (all three or none); with source = budget one of gamma_p_typical
 *                               or the envelope is required
 *   c_layer                     float > 0 (default 3)
 *   comm_headroom               float >= 0 (default 0.2)
 *   ions.K                      non-empty array of float > 0 (kinetic energies, m_e c^2), required
 *   ions.per_K                  int >= 1 with per_K * ions.K.size() <= INT_MAX, required
 *   ions.direction              "+x" | "isotropic", required
 *   ions.weight                 float > 0 (default 1)
 *   fields.B, fields.E          arrays of three floats (default zero)
 *   ledger.every                int in [1, INT_MAX] (default 1)
 *   table_dump                  bool (default true): write the table file next to the run output
 *   test.nan_payloads           bool (default false): fill every real payload slot of every
 *                               species with NaN before any injection (T10b)
 *   test.kick                   float >= 0 (default 0): antisymmetric transverse kick of the
 *                               leptons (T13)
 *   test.gauss                  bool (default false): write the discrete Gauss residual (T13)
 *   background.ppc              float >= 0 (default 0): a neutral background of the two pair
 *                               species injected through arch::InjectNonUniform (stock path)
 *   background.temperature      float >= 0 (default 0)
 *   background.n_global         int in [0, INT_MAX] (default 0): particles per pair species
 *                               injected through arch::InjectGlobally (stock path, weight 1)
 *
 * @namespaces:
 *   - user::bhproto::
 */

#ifndef PGENS_BH_PROTO_PARAMS_H
#define PGENS_BH_PROTO_PARAMS_H

#include "archetypes/hadronic/bh_physics.h"

#include <toml11/toml.hpp>

#include <array>
#include <climits>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace user::bhproto {

  struct Setup {
    bool               enabled { false };
    std::vector<int>   species;
    double             tau0 { 0.0 };
    double             mu_r { 1836.15 };
    arch::bh::BathSpec bath;
    double             gamma_p_max { 0.0 }; // 0 = auto
    double             eta_u { 1.0e-4 };
    double             pending_floor_ulp { 100.0 };
    double             f_max { 0.01 };
    int                max_substeps { 64 };
    bool               allow_unsafe_application { false };
    int                payload_pending { 0 };
    int                payload_debt { 1 };
    int                payload_debt_k0 { 2 };
    // pair creation
    std::string        source { "budget" };
    std::string        spectrum { "delta" };
    int                pair_minus { 0 }; // 0 = not given (source = none)
    int                pair_plus { 0 };
    double             r_th_factor { 10.0 };
    double             f_r { 0.02 };
    bool               t_latency_auto { true };
    double             t_latency { 0.0 };
    double             pair_weight { 1.0 };
    int                max_events_per_step { 8 };
    std::string        on_capacity { "error" };
    std::string        on_capacity_estimate { "error" };
    std::string        on_max_events { "error" };
    std::string        placement { "poststep" }; // "poststep" | "production" (hadronic.placement)
    double             gamma_p_typical { 0.0 };
    double             active_fraction { 1.0 };
    bool               has_envelope { false };
    double             envelope_index { 0.0 };
    double             envelope_gamma_min { 0.0 };
    double             envelope_gamma_max { 0.0 };
    double             c_layer { 3.0 };
    double             comm_headroom { 0.2 };
    // ions
    std::vector<double> ions_K;
    int                ions_per_K { 0 };
    std::string        ions_direction;
    double             ions_weight { 1.0 };
    std::array<double, 3> B { 0.0, 0.0, 0.0 };
    std::array<double, 3> E { 0.0, 0.0, 0.0 };
    int                ledger_every { 1 };
    bool               table_dump { true };
    // test hooks
    bool               test_nan_payloads { false };
    double             test_kick { 0.0 };
    bool               test_gauss { false };
    double             background_ppc { 0.0 };
    double             background_temperature { 0.0 };
    int                background_n_global { 0 };

    auto design_off() const -> bool {
      return eta_u == 0.0 && pending_floor_ulp == 0.0;
    }

    auto pairs_enabled() const -> bool {
      return enabled && source != "none";
    }
  };

  namespace detail {

    inline auto fmt_double(double v) -> std::string {
      std::ostringstream ss;
      ss.precision(17);
      ss << v;
      return ss.str();
    }

    inline auto type_name(const toml::value& v) -> std::string {
      if (v.is_boolean()) {
        return "boolean";
      }
      if (v.is_integer()) {
        return "integer";
      }
      if (v.is_floating()) {
        return "float";
      }
      if (v.is_string()) {
        return "string";
      }
      if (v.is_array()) {
        return "array";
      }
      if (v.is_table()) {
        return "table";
      }
      return "unknown";
    }

    [[noreturn]]
    inline void fail(const std::string& key, const std::string& what) {
      throw std::invalid_argument("bh_proto: setup." + key + " " + what);
    }

    /**
     * @brief Resolve a dotted key through nested tables; nullptr if absent
     */
    inline auto find_path(const toml::value& root, const std::string& key)
      -> const toml::value* {
      const toml::value* cur = &root;
      std::size_t        start = 0;
      while (true) {
        const auto        dot  = key.find('.', start);
        const std::string part = key.substr(
          start,
          dot == std::string::npos ? std::string::npos : dot - start);
        if (!cur->is_table()) {
          return nullptr;
        }
        if (!cur->contains(part)) {
          return nullptr;
        }
        cur = &cur->at(part);
        if (dot == std::string::npos) {
          return cur;
        }
        start = dot + 1;
      }
    }

    inline auto get_bool(const toml::value& root, const std::string& key) -> bool {
      const auto* v = find_path(root, key);
      if (v == nullptr) {
        fail(key, "is required (boolean) and was not given");
      }
      if (!v->is_boolean()) {
        fail(key, "must be a boolean, got a " + type_name(*v));
      }
      return v->as_boolean();
    }

    inline auto get_bool_or(const toml::value& root,
                            const std::string& key,
                            bool               def) -> bool {
      const auto* v = find_path(root, key);
      if (v == nullptr) {
        return def;
      }
      if (!v->is_boolean()) {
        fail(key, "must be a boolean, got a " + type_name(*v));
      }
      return v->as_boolean();
    }

    inline auto as_number(const toml::value& v, const std::string& key) -> double {
      double x;
      if (v.is_integer()) {
        x = static_cast<double>(v.as_integer());
      } else if (v.is_floating()) {
        x = v.as_floating();
      } else {
        fail(key, "must be a number, got a " + type_name(v));
      }
      if (!std::isfinite(x)) {
        fail(key, "must be finite, got " + fmt_double(x));
      }
      return x;
    }

    inline auto get_number(const toml::value& root, const std::string& key) -> double {
      const auto* v = find_path(root, key);
      if (v == nullptr) {
        fail(key, "is required (number) and was not given");
      }
      return as_number(*v, key);
    }

    inline auto get_number_or(const toml::value& root,
                              const std::string& key,
                              double             def) -> double {
      const auto* v = find_path(root, key);
      if (v == nullptr) {
        return def;
      }
      return as_number(*v, key);
    }

    inline auto as_int(const toml::value& v, const std::string& key) -> long long {
      if (!v.is_integer()) {
        fail(key, "must be an integer, got a " + type_name(v));
      }
      return static_cast<long long>(v.as_integer());
    }

    inline auto get_int(const toml::value& root, const std::string& key) -> long long {
      const auto* v = find_path(root, key);
      if (v == nullptr) {
        fail(key, "is required (integer) and was not given");
      }
      return as_int(*v, key);
    }

    inline auto get_int_or(const toml::value& root,
                           const std::string& key,
                           long long          def) -> long long {
      const auto* v = find_path(root, key);
      if (v == nullptr) {
        return def;
      }
      return as_int(*v, key);
    }

    /**
     * @brief Integer key validated against [lo, hi] on the 64-bit value, before
     *        any narrowing (R1): the destination type's range is the caller's
     *        `hi`, and the value is returned as the destination type only after
     *        the check
     */
    inline auto get_int_in(const toml::value& root,
                           const std::string& key,
                           long long          lo,
                           long long          hi,
                           bool               required,
                           long long          def = 0) -> int {
      const auto* v = find_path(root, key);
      long long   n;
      if (v == nullptr) {
        if (required) {
          fail(key, "is required (integer) and was not given");
        }
        n = def;
      } else {
        n = as_int(*v, key);
      }
      if (n < lo || n > hi) {
        fail(key,
             "= " + std::to_string(n) + " must lie in [" + std::to_string(lo) + ", " +
               std::to_string(hi) + "]");
      }
      static_assert(sizeof(int) >= 4, "int narrower than 32 bits");
      if (hi > static_cast<long long>(std::numeric_limits<int>::max())) {
        fail(key, "internal: destination range exceeds int");
      }
      return static_cast<int>(n);
    }

    inline auto get_string(const toml::value& root, const std::string& key)
      -> std::string {
      const auto* v = find_path(root, key);
      if (v == nullptr) {
        fail(key, "is required (string) and was not given");
      }
      if (!v->is_string()) {
        fail(key, "must be a string, got a " + type_name(*v));
      }
      return std::string(v->as_string());
    }

    inline auto get_string_or(const toml::value& root,
                              const std::string& key,
                              const std::string& def) -> std::string {
      const auto* v = find_path(root, key);
      if (v == nullptr) {
        return def;
      }
      if (!v->is_string()) {
        fail(key, "must be a string, got a " + type_name(*v));
      }
      return std::string(v->as_string());
    }

    inline auto get_policy_or(const toml::value& root,
                              const std::string& key) -> std::string {
      const auto s = get_string_or(root, key, "error");
      if (s != "error" && s != "warn") {
        fail(key, "= '" + s + "' must be \"error\" or \"warn\"");
      }
      return s;
    }

    inline auto get_number_array(const toml::value& root, const std::string& key)
      -> std::vector<double> {
      const auto* v = find_path(root, key);
      if (v == nullptr) {
        fail(key, "is required (array of numbers) and was not given");
      }
      if (!v->is_array()) {
        fail(key, "must be an array of numbers, got a " + type_name(*v));
      }
      std::vector<double> out;
      std::size_t         i = 0;
      for (const auto& e : v->as_array()) {
        out.push_back(as_number(e, key + "[" + std::to_string(i) + "]"));
        ++i;
      }
      return out;
    }

    inline auto get_vec3_or_zero(const toml::value& root, const std::string& key)
      -> std::array<double, 3> {
      std::array<double, 3> out { 0.0, 0.0, 0.0 };
      if (find_path(root, key) == nullptr) {
        return out;
      }
      const auto arr = get_number_array(root, key);
      if (arr.size() != 3) {
        fail(key,
             "must have exactly three components, got " +
               std::to_string(arr.size()));
      }
      for (std::size_t d = 0; d < 3; ++d) {
        out[d] = arr[d];
      }
      return out;
    }

    inline void require_in_open_0_2(const std::string& key, double v) {
      if (!(v > 0.0 && v < 2.0)) {
        fail(key, "= " + fmt_double(v) + " must lie in (0, 2)");
      }
    }

    /**
     * @brief Collect every leaf key (dotted) of a table
     */
    inline void collect_keys(const toml::value&      t,
                             const std::string&      prefix,
                             std::vector<std::string>& out) {
      for (const auto& [k, v] : t.as_table()) {
        const std::string full = prefix.empty() ? std::string(k) : prefix + "." + std::string(k);
        if (v.is_table()) {
          collect_keys(v, full, out);
        } else {
          out.push_back(full);
        }
      }
    }

  } // namespace detail

  /**
   * @brief Parse and validate the `[setup]` table
   * @param setup the `setup` table of the input (may be an empty table)
   * @throws std::invalid_argument with a message naming `setup.<key>`
   */
  /**
   * @brief C-MPI-3: whether the construction-time guard refuses the run.
   *        The dead-ion harvest of the post-step placement (host sums over the
   *        dead-tagged ions in the post-step hook, and the kernel's dead branch
   *        there) is unsafe wherever a receive can refill a dead slot before the
   *        hook runs: that is any MPI build, with one rank too (the periodic
   *        self-exchange goes through the same send and receive buffers), for
   *        any domain count; the non-MPI exchange never receives, so it is safe
   *        at any domain count. The production placement harvests before the
   *        exchange and is safe everywhere (C-MPI-1 clears the remnants of the
   *        transported particles).
   * @param absorbing any particle boundary absorbing
   * @param mpi_build compiled with MPI_ENABLED
   * @param ndomains the number of domains (reported in the message; not part
   *                 of the predicate, see above)
   * @param placement "poststep" | "production"
   */
  inline auto poststep_harvest_refused(bool               absorbing,
                                       bool               mpi_build,
                                       unsigned int       ndomains,
                                       const std::string& placement) -> bool {
    (void)ndomains;
    return absorbing && mpi_build && placement == "poststep";
  }

  inline auto parse_setup(const toml::value& setup) -> Setup {
    using namespace detail;
    if (!setup.is_table()) {
      throw std::invalid_argument("bh_proto: [setup] must be a table");
    }

    static const std::set<std::string> known = {
      "hadronic.enabled",      "hadronic.species",
      "hadronic.placement",
      "tau0",                  "physical_mass_ratio",
      "photon_field.type",     "photon_field.eps0",
      "photon_field.eps_min",  "photon_field.eps_max",
      "photon_field.index",    "photon_field.theta",
      "photon_field.file",     "gamma_p_max",
      "eta_u",                 "pending_floor_ulp",
      "f_max",                 "max_substeps",
      "allow_unsafe_application",
      "payload_pending",       "payload_debt",
      "payload_debt_k0",       "ions.K",
      "ions.per_K",            "ions.direction",
      "ions.weight",           "fields.B",
      "fields.E",              "ledger.every",
      "table_dump",
      "source",                "spectrum",
      "pair_species",          "r_th_factor",
      "f_r",                   "t_latency",
      "pair_weight",           "max_events_per_step",
      "on_capacity",           "on_capacity_estimate",
      "on_max_events",         "gamma_p_typical",
      "active_fraction",       "envelope.index",
      "envelope.gamma_min",    "envelope.gamma_max",
      "c_layer",               "comm_headroom",
      "test.nan_payloads",     "test.kick",
      "test.gauss",            "background.ppc",
      "background.temperature", "background.n_global"
    };
    {
      std::vector<std::string> keys;
      collect_keys(setup, "", keys);
      for (const auto& k : keys) {
        if (known.count(k) == 0) {
          fail(k, "is not a key of the bh_proto problem generator");
        }
      }
    }

    Setup s;
    s.enabled = get_bool(setup, "hadronic.enabled");
    s.placement = get_string_or(setup, "hadronic.placement", "poststep");
    if (s.placement != "poststep" && s.placement != "production") {
      fail("hadronic.placement", "= '" + s.placement + "' must be \"poststep\" or \"production\"");
    }

    // ions (always required: the generator injects them regardless of the stage)
    s.ions_K = get_number_array(setup, "ions.K");
    if (s.ions_K.empty()) {
      fail("ions.K", "must be a non-empty array of positive kinetic energies");
    }
    for (std::size_t i = 0; i < s.ions_K.size(); ++i) {
      if (!(s.ions_K[i] > 0.0)) {
        fail("ions.K[" + std::to_string(i) + "]",
             "= " + fmt_double(s.ions_K[i]) + " must be > 0");
      }
    }
    // ions.per_K times ions.K.size() must fit the particle index type (npart_t,
    // 32-bit unsigned): bound per_K by INT_MAX / ions.K.size() before any cast
    s.ions_per_K = get_int_in(setup, "ions.per_K", 1,
                              static_cast<long long>(std::numeric_limits<int>::max()) /
                                static_cast<long long>(s.ions_K.size()),
                              true);
    s.ions_direction = get_string(setup, "ions.direction");
    if (s.ions_direction != "+x" && s.ions_direction != "isotropic") {
      fail("ions.direction",
           "= '" + s.ions_direction + "' must be \"+x\" or \"isotropic\"");
    }
    s.ions_weight = get_number_or(setup, "ions.weight", 1.0);
    if (!(s.ions_weight > 0.0)) {
      fail("ions.weight", "= " + fmt_double(s.ions_weight) + " must be > 0");
    }
    s.B = get_vec3_or_zero(setup, "fields.B");
    s.E = get_vec3_or_zero(setup, "fields.E");
    s.ledger_every = get_int_in(setup, "ledger.every", 1, std::numeric_limits<int>::max(), false, 1);
    s.table_dump = get_bool_or(setup, "table_dump", true);

    // test hooks (independent of the stage)
    s.test_nan_payloads = get_bool_or(setup, "test.nan_payloads", false);
    s.test_kick         = get_number_or(setup, "test.kick", 0.0);
    if (!(s.test_kick >= 0.0)) {
      fail("test.kick", "= " + fmt_double(s.test_kick) + " must be >= 0");
    }
    s.test_gauss     = get_bool_or(setup, "test.gauss", false);
    s.background_ppc = get_number_or(setup, "background.ppc", 0.0);
    if (!(s.background_ppc >= 0.0)) {
      fail("background.ppc", "= " + fmt_double(s.background_ppc) + " must be >= 0");
    }
    s.background_temperature = get_number_or(setup, "background.temperature", 0.0);
    if (!(s.background_temperature >= 0.0)) {
      fail("background.temperature",
           "= " + fmt_double(s.background_temperature) + " must be >= 0");
    }
    s.background_n_global = get_int_in(setup, "background.n_global", 0,
                                       std::numeric_limits<int>::max(), false, 0);

    // the ion species of the problem (ions are injected into them; the stage
    // runs on them when enabled)
    {
      const auto* v = find_path(setup, "hadronic.species");
      if (v == nullptr) {
        fail("hadronic.species", "is required (array of species indices >= 1)");
      }
      if (!v->is_array()) {
        fail("hadronic.species", "must be an array of integers, got a " + type_name(*v));
      }
      std::size_t i = 0;
      for (const auto& e : v->as_array()) {
        const auto idx = as_int(e, "hadronic.species[" + std::to_string(i) + "]");
        if (idx < 1 || idx > 255) {
          fail("hadronic.species[" + std::to_string(i) + "]",
               "= " + std::to_string(idx) + " must be a species index >= 1");
        }
        for (const auto prev : s.species) {
          if (prev == idx) {
            fail("hadronic.species", "lists species " + std::to_string(idx) + " twice");
          }
        }
        s.species.push_back(static_cast<int>(idx));
        ++i;
      }
      if (s.species.empty()) {
        fail("hadronic.species", "must list at least one species");
      }
    }

    if (!s.enabled) {
      return s;
    }

    s.tau0 = get_number(setup, "tau0");
    if (!(s.tau0 >= 0.0)) {
      fail("tau0", "= " + fmt_double(s.tau0) + " must be >= 0 (0 = stage not launched)");
    }
    s.mu_r = get_number(setup, "physical_mass_ratio");
    if (!(s.mu_r > 0.0)) {
      fail("physical_mass_ratio", "= " + fmt_double(s.mu_r) + " must be > 0");
    }

    // photon field, the same names and ranges as bh_tables_dump
    s.bath.type = get_string(setup, "photon_field.type");
    auto has = [&setup](const std::string& k) {
      return find_path(setup, k) != nullptr;
    };
    if (s.bath.type == "mono") {
      s.bath.eps0 = get_number(setup, "photon_field.eps0");
      require_in_open_0_2("photon_field.eps0", s.bath.eps0);
    } else if (s.bath.type == "powerlaw") {
      s.bath.eps_min = get_number(setup, "photon_field.eps_min");
      s.bath.eps_max = get_number(setup, "photon_field.eps_max");
      s.bath.index   = get_number(setup, "photon_field.index");
      require_in_open_0_2("photon_field.eps_min", s.bath.eps_min);
      require_in_open_0_2("photon_field.eps_max", s.bath.eps_max);
      if (!(s.bath.eps_max > s.bath.eps_min)) {
        fail("photon_field.eps_max", "must exceed photon_field.eps_min");
      }
    } else if (s.bath.type == "blackbody") {
      s.bath.theta   = get_number(setup, "photon_field.theta");
      s.bath.eps_max = get_number(setup, "photon_field.eps_max");
      s.bath.eps_min = get_number_or(setup, "photon_field.eps_min", 0.0);
      if (!(s.bath.theta > 0.0)) {
        fail("photon_field.theta", "= " + fmt_double(s.bath.theta) + " must be > 0");
      }
      if (!(s.bath.eps_min >= 0.0 && s.bath.eps_min < 2.0)) {
        fail("photon_field.eps_min", "= " + fmt_double(s.bath.eps_min) + " must lie in [0, 2)");
      }
      require_in_open_0_2("photon_field.eps_max", s.bath.eps_max);
    } else if (s.bath.type == "table") {
      s.bath.file = get_string(setup, "photon_field.file");
      if (s.bath.file.empty()) {
        fail("photon_field.file", "must not be empty");
      }
      if (has("photon_field.eps_min")) {
        s.bath.eps_min = get_number(setup, "photon_field.eps_min");
        require_in_open_0_2("photon_field.eps_min", s.bath.eps_min);
      }
      if (has("photon_field.eps_max")) {
        s.bath.eps_max = get_number(setup, "photon_field.eps_max");
        require_in_open_0_2("photon_field.eps_max", s.bath.eps_max);
      }
    } else {
      fail("photon_field.type",
           "= '" + s.bath.type + "' is unknown (mono, powerlaw, blackbody, table)");
    }

    // table upper edge
    {
      const auto* v = find_path(setup, "gamma_p_max");
      if (v != nullptr) {
        if (v->is_string()) {
          if (std::string(v->as_string()) != "auto") {
            fail("gamma_p_max", "must be a number > 1 or the string \"auto\"");
          }
          s.gamma_p_max = 0.0;
        } else {
          s.gamma_p_max = as_number(*v, "gamma_p_max");
          if (!(s.gamma_p_max > 1.0)) {
            fail("gamma_p_max", "= " + fmt_double(s.gamma_p_max) + " must be > 1");
          }
        }
      }
    }

    // loss application
    s.eta_u = get_number_or(setup, "eta_u", 1.0e-4);
    if (!(s.eta_u >= 0.0)) {
      fail("eta_u", "= " + fmt_double(s.eta_u) + " must be >= 0");
    }
    s.pending_floor_ulp = get_number_or(setup, "pending_floor_ulp", 100.0);
    if (!(s.pending_floor_ulp >= 0.0)) {
      fail("pending_floor_ulp", "= " + fmt_double(s.pending_floor_ulp) + " must be >= 0");
    }
    s.f_max = get_number_or(setup, "f_max", 0.01);
    if (!(s.f_max > 0.0 && s.f_max <= 1.0)) {
      fail("f_max", "= " + fmt_double(s.f_max) + " must lie in (0, 1]");
    }
    s.max_substeps = get_int_in(setup, "max_substeps", 1, 1000000LL, false, 64);
    s.allow_unsafe_application = get_bool_or(setup, "allow_unsafe_application", false);
    if ((s.eta_u == 0.0 || s.pending_floor_ulp == 0.0) && !s.allow_unsafe_application) {
      fail("allow_unsafe_application",
           "must be set to true when eta_u = 0 or pending_floor_ulp = 0 "
           "(the precision-safe application is switched off; T5b only)");
    }

    // payload indices: the kernel addresses the slot as unsigned short and the
    // species slot count is an unsigned short, so a slot index must lie in
    // [0, USHRT_MAX - 1] (the count max(index) + 1 then fits the type without
    // overflow); validated on the 64-bit value before the distinctness check
    // and before any cast (R1)
    {
      const long long slot_max = static_cast<long long>(USHRT_MAX) - 1;
      s.payload_pending = get_int_in(setup, "payload_pending", 0, slot_max, false, 0);
      s.payload_debt    = get_int_in(setup, "payload_debt", 0, slot_max, false, 1);
      s.payload_debt_k0 = get_int_in(setup, "payload_debt_k0", 0, slot_max, false, 2);
      if (s.payload_pending == s.payload_debt || s.payload_pending == s.payload_debt_k0 ||
          s.payload_debt == s.payload_debt_k0) {
        fail("payload_pending", "payload_debt and payload_debt_k0 must be three distinct slots");
      }
    }

    // ---- pair creation (step 3) ----------------------------------------------
    s.source = get_string_or(setup, "source", "budget");
    if (s.source != "budget" && s.source != "poisson" && s.source != "none") {
      fail("source", "= '" + s.source + "' must be \"budget\", \"poisson\" or \"none\"");
    }
    s.spectrum = get_string_or(setup, "spectrum", "delta");
    if (s.spectrum != "delta" && s.spectrum != "threshold") {
      fail("spectrum", "= '" + s.spectrum + "' must be \"delta\" or \"threshold\"");
    }
    {
      const auto* v = find_path(setup, "pair_species");
      if (v == nullptr) {
        if (s.source != "none") {
          fail("pair_species",
               "is required with source = \"" + s.source +
                 "\" (array [i_minus, i_plus] of species indices >= 1)");
        }
      } else {
        if (!v->is_array() || v->as_array().size() != 2) {
          fail("pair_species", "must be an array of exactly two species indices [i_minus, i_plus]");
        }
        const auto a = as_int(v->as_array()[0], "pair_species[0]");
        const auto b = as_int(v->as_array()[1], "pair_species[1]");
        if (a < 1 || a > 255) {
          fail("pair_species[0]", "= " + std::to_string(a) + " must be a species index in [1, 255]");
        }
        if (b < 1 || b > 255) {
          fail("pair_species[1]", "= " + std::to_string(b) + " must be a species index in [1, 255]");
        }
        if (a == b) {
          fail("pair_species", "lists the same species twice (" + std::to_string(a) + ")");
        }
        for (const auto ion : s.species) {
          if (ion == a || ion == b) {
            fail("pair_species",
                 "lists species " + std::to_string(ion) + " which is a hadronic ion species");
          }
        }
        s.pair_minus = static_cast<int>(a);
        s.pair_plus  = static_cast<int>(b);
      }
    }
    s.r_th_factor = get_number_or(setup, "r_th_factor", 10.0);
    if (!(s.r_th_factor > 0.0)) {
      fail("r_th_factor", "= " + fmt_double(s.r_th_factor) + " must be > 0");
    }
    s.f_r = get_number_or(setup, "f_r", 0.02);
    if (!(s.f_r > 0.0 && s.f_r < 1.0)) {
      fail("f_r", "= " + fmt_double(s.f_r) + " must lie in (0, 1)");
    }
    {
      const auto* v = find_path(setup, "t_latency");
      if (v == nullptr) {
        s.t_latency_auto = true;
      } else if (v->is_string()) {
        if (std::string(v->as_string()) != "auto") {
          fail("t_latency", "must be a number >= 0 or the string \"auto\"");
        }
        s.t_latency_auto = true;
      } else {
        s.t_latency_auto = false;
        s.t_latency      = as_number(*v, "t_latency");
        if (!(s.t_latency >= 0.0)) {
          fail("t_latency", "= " + fmt_double(s.t_latency) + " must be >= 0 (0 = cap off)");
        }
      }
    }
    s.pair_weight = get_number_or(setup, "pair_weight", 1.0);
    if (!(s.pair_weight > 0.0)) {
      fail("pair_weight", "= " + fmt_double(s.pair_weight) + " must be > 0");
    }
    s.max_events_per_step = get_int_in(setup, "max_events_per_step", 1, 1000000LL, false, 8);
    s.on_capacity          = get_policy_or(setup, "on_capacity");
    s.on_capacity_estimate = get_policy_or(setup, "on_capacity_estimate");
    s.on_max_events        = get_policy_or(setup, "on_max_events");
    s.gamma_p_typical      = get_number_or(setup, "gamma_p_typical", 0.0);
    if (!(s.gamma_p_typical == 0.0 || s.gamma_p_typical > 1.0)) {
      fail("gamma_p_typical",
           "= " + fmt_double(s.gamma_p_typical) + " must be > 1, or 0 to use the envelope");
    }
    s.active_fraction = get_number_or(setup, "active_fraction", 1.0);
    if (!(s.active_fraction > 0.0 && s.active_fraction <= 1.0)) {
      fail("active_fraction", "= " + fmt_double(s.active_fraction) + " must lie in (0, 1]");
    }
    {
      const bool hi = has("envelope.index"), hmin = has("envelope.gamma_min"),
                 hmax = has("envelope.gamma_max");
      if (hi || hmin || hmax) {
        if (!(hi && hmin && hmax)) {
          fail("envelope.index", "envelope.gamma_min and envelope.gamma_max must all be given together");
        }
        s.has_envelope       = true;
        s.envelope_index     = get_number(setup, "envelope.index");
        s.envelope_gamma_min = get_number(setup, "envelope.gamma_min");
        s.envelope_gamma_max = get_number(setup, "envelope.gamma_max");
        if (!(s.envelope_gamma_min > 1.0)) {
          fail("envelope.gamma_min", "= " + fmt_double(s.envelope_gamma_min) + " must be > 1");
        }
        if (!(s.envelope_gamma_max > s.envelope_gamma_min)) {
          fail("envelope.gamma_max",
               "= " + fmt_double(s.envelope_gamma_max) + " must exceed envelope.gamma_min = " +
                 fmt_double(s.envelope_gamma_min));
        }
      }
    }
    if (s.source == "budget" && s.gamma_p_typical == 0.0 && !s.has_envelope) {
      fail("gamma_p_typical",
           "or an envelope (envelope.index, envelope.gamma_min, envelope.gamma_max) is "
           "required with source = \"budget\" (startup capacity estimate, III.8)");
    }
    s.c_layer = get_number_or(setup, "c_layer", 3.0);
    if (!(s.c_layer > 0.0)) {
      fail("c_layer", "= " + fmt_double(s.c_layer) + " must be > 0");
    }
    s.comm_headroom = get_number_or(setup, "comm_headroom", 0.2);
    if (!(s.comm_headroom >= 0.0)) {
      fail("comm_headroom", "= " + fmt_double(s.comm_headroom) + " must be >= 0");
    }
    return s;
  }

} // namespace user::bhproto

#endif // PGENS_BH_PROTO_PARAMS_H
