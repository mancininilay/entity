/**
 * @file pgens/bh_proto/bh_proto_params.h
 * @brief Strict reader of the `[setup]` table of the bh_proto problem generator
 *        (Bethe-Heitler loss-only prototype, P2 step 2).
 *
 * The reader works on the raw TOML value (`SimulationParams::data()`), not on
 * the flattened `setup.*` container, so that nested tables (`hadronic.enabled`,
 * `photon_field.type`, `ions.K`), empty arrays and wrong types are seen and
 * refused with a message naming the key and the value. Every numeric key is
 * range-checked. No sentinel value can result from a failed conversion: a key
 * is either absent (and takes its documented default, when it has one) or
 * converts completely, or the reader throws.
 *
 * Keys (all under `[setup]`; floats accept TOML integers):
 *   hadronic.enabled            bool, required
 *   hadronic.species            non-empty array of int >= 1, required (the ion species)
 *   tau0                        float >= 0 (0 = stage not launched), required when enabled
 *   physical_mass_ratio         float > 0, required when enabled
 *   photon_field.type           "mono" | "powerlaw" | "blackbody" | "table", required when enabled
 *   photon_field.eps0 / eps_min / eps_max / index / theta / file   as bh_tables_dump
 *   gamma_p_max                 float > 1 or the string "auto" (default auto)
 *   eta_u                       float >= 0 (default 1e-4)
 *   pending_floor_ulp           float >= 0 (default 100)
 *   f_max                       float in (0, 1] (default 0.1)
 *   max_substeps                int >= 1 (default 64)
 *   allow_unsafe_application    bool (default false); required true when eta_u = 0 or pending_floor_ulp = 0
 *   payload_pending / payload_debt / payload_debt_k0   int >= 0, distinct (default 0, 1, 2)
 *   ions.K                      non-empty array of float > 0 (kinetic energies, m_e c^2), required
 *   ions.per_K                  int >= 1, required
 *   ions.direction              "+x" | "isotropic", required
 *   ions.weight                 float > 0 (default 1)
 *   fields.B, fields.E          arrays of three floats (default zero)
 *   ledger.every                int >= 1 (default 1)
 *   table_dump                  bool (default true): write the table file next to the run output
 *
 * @namespaces:
 *   - user::bhproto::
 */

#ifndef PGENS_BH_PROTO_PARAMS_H
#define PGENS_BH_PROTO_PARAMS_H

#include "archetypes/hadronic/bh_physics.h"

#include <toml11/toml.hpp>

#include <array>
#include <cmath>
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
    double             f_max { 0.1 };
    int                max_substeps { 64 };
    bool               allow_unsafe_application { false };
    int                payload_pending { 0 };
    int                payload_debt { 1 };
    int                payload_debt_k0 { 2 };
    std::vector<double> ions_K;
    int                ions_per_K { 0 };
    std::string        ions_direction;
    double             ions_weight { 1.0 };
    std::array<double, 3> B { 0.0, 0.0, 0.0 };
    std::array<double, 3> E { 0.0, 0.0, 0.0 };
    int                ledger_every { 1 };
    bool               table_dump { true };

    auto design_off() const -> bool {
      return eta_u == 0.0 && pending_floor_ulp == 0.0;
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
  inline auto parse_setup(const toml::value& setup) -> Setup {
    using namespace detail;
    if (!setup.is_table()) {
      throw std::invalid_argument("bh_proto: [setup] must be a table");
    }

    static const std::set<std::string> known = {
      "hadronic.enabled",      "hadronic.species",
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
      "table_dump"
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
    {
      const auto n = get_int(setup, "ions.per_K");
      if (n < 1 || n > 100000000LL) {
        fail("ions.per_K", "= " + std::to_string(n) + " must be >= 1");
      }
      s.ions_per_K = static_cast<int>(n);
    }
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
    {
      const auto n = get_int_or(setup, "ledger.every", 1);
      if (n < 1) {
        fail("ledger.every", "= " + std::to_string(n) + " must be >= 1");
      }
      s.ledger_every = static_cast<int>(n);
    }
    s.table_dump = get_bool_or(setup, "table_dump", true);

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
    s.f_max = get_number_or(setup, "f_max", 0.1);
    if (!(s.f_max > 0.0 && s.f_max <= 1.0)) {
      fail("f_max", "= " + fmt_double(s.f_max) + " must lie in (0, 1]");
    }
    {
      const auto n = get_int_or(setup, "max_substeps", 64);
      if (n < 1 || n > 1000000LL) {
        fail("max_substeps", "= " + std::to_string(n) + " must be >= 1");
      }
      s.max_substeps = static_cast<int>(n);
    }
    s.allow_unsafe_application = get_bool_or(setup, "allow_unsafe_application", false);
    if ((s.eta_u == 0.0 || s.pending_floor_ulp == 0.0) && !s.allow_unsafe_application) {
      fail("allow_unsafe_application",
           "must be set to true when eta_u = 0 or pending_floor_ulp = 0 "
           "(the precision-safe application is switched off; T5b only)");
    }

    // payload indices
    {
      const auto p = get_int_or(setup, "payload_pending", 0);
      const auto d = get_int_or(setup, "payload_debt", 1);
      const auto k = get_int_or(setup, "payload_debt_k0", 2);
      if (p < 0) {
        fail("payload_pending", "= " + std::to_string(p) + " must be >= 0");
      }
      if (d < 0) {
        fail("payload_debt", "= " + std::to_string(d) + " must be >= 0");
      }
      if (k < 0) {
        fail("payload_debt_k0", "= " + std::to_string(k) + " must be >= 0");
      }
      if (p == d || p == k || d == k) {
        fail("payload_pending", "payload_debt and payload_debt_k0 must be three distinct slots");
      }
      s.payload_pending = static_cast<int>(p);
      s.payload_debt    = static_cast<int>(d);
      s.payload_debt_k0 = static_cast<int>(k);
    }
    return s;
  }

} // namespace user::bhproto

#endif // PGENS_BH_PROTO_PARAMS_H
