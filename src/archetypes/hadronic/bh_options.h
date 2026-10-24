/**
 * @file archetypes/hadronic/bh_options.h
 * @brief Strict command-line option parsing for the table generator
 *        (`tools/bh_tables_dump`), kept in a header so that the test suite can
 *        exercise it. Every numeric option must convert completely to a finite
 *        number and lie in its admissible range; the automatic upper edge is
 *        selected only by omitting `--gamma_p_max` or by the word `auto`.
 *
 * @namespaces:
 *   - arch::bh::
 */

#ifndef ARCHETYPES_HADRONIC_BH_OPTIONS_H
#define ARCHETYPES_HADRONIC_BH_OPTIONS_H

#include "archetypes/hadronic/bh_physics.h"

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace arch::bh {

  struct DumpOptions {
    BathSpec    spec;
    double      mu_r { 1836.15 };
    double      gamma_p_max { 0.0 }; // 0 = auto (only via omission or "auto")
    double      h { 0.02 };
    std::string status { "cpp_candidate" };
    std::string out;
  };

  /**
   * @brief Parse `--key value` tokens (argv without the program name)
   * @throws std::invalid_argument with a message naming the offending option
   */
  inline DumpOptions parse_dump_options(const std::vector<std::string>& tokens) {
    static const std::vector<std::string> known = {
      "type", "eps0", "eps_min",     "eps_max", "index",  "theta",
      "file", "mu_r", "gamma_p_max", "h",       "status", "out"
    };
    std::map<std::string, std::string> args;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      const std::string& a = tokens[i];
      if (a.rfind("--", 0) != 0 || a.size() == 2) {
        throw std::invalid_argument(
          "expected an option of the form --key, got '" + a + "'");
      }
      const std::string key = a.substr(2);
      if (std::find(known.begin(), known.end(), key) == known.end()) {
        throw std::invalid_argument("unknown option --" + key);
      }
      if (i + 1 >= tokens.size()) {
        throw std::invalid_argument("option --" + key + " needs a value");
      }
      if (args.count(key)) {
        throw std::invalid_argument("option --" + key + " given twice");
      }
      args[key] = tokens[++i];
    }
    auto has = [&args](const std::string& k) {
      return args.count(k) > 0;
    };
    auto num = [&args](const std::string& k) {
      double v = 0.0;
      if (!parse_number_strict(args.at(k), v)) {
        throw std::invalid_argument(
          "option --" + k + ": '" + args.at(k) + "' is not a finite number");
      }
      return v;
    };
    auto require = [&has](const std::string& k, const std::string& type) {
      if (!has(k)) {
        throw std::invalid_argument(
          "photon_field.type " + type + " requires --" + k);
      }
    };
    auto in_open_0_2 = [](const std::string& k, double v) {
      if (!(v > 0.0 && v < 2.0)) {
        throw std::invalid_argument(
          "option --" + k + " = " + std::to_string(v) + " must lie in (0, 2)");
      }
    };

    DumpOptions o;
    if (!has("type")) {
      throw std::invalid_argument("--type is required");
    }
    if (!has("out")) {
      throw std::invalid_argument("--out is required");
    }
    o.spec.type = args["type"];
    o.out       = args["out"];
    if (has("status")) {
      o.status = args["status"];
    }
    if (o.spec.type == "mono") {
      require("eps0", "mono");
      o.spec.eps0 = num("eps0");
      in_open_0_2("eps0", o.spec.eps0);
    } else if (o.spec.type == "powerlaw") {
      require("eps_min", "powerlaw");
      require("eps_max", "powerlaw");
      require("index", "powerlaw");
      o.spec.eps_min = num("eps_min");
      o.spec.eps_max = num("eps_max");
      o.spec.index   = num("index");
      in_open_0_2("eps_min", o.spec.eps_min);
      in_open_0_2("eps_max", o.spec.eps_max);
    } else if (o.spec.type == "blackbody") {
      require("theta", "blackbody");
      require("eps_max", "blackbody");
      o.spec.theta   = num("theta");
      o.spec.eps_max = num("eps_max");
      o.spec.eps_min = has("eps_min") ? num("eps_min") : 0.0;
      if (!(o.spec.theta > 0.0)) {
        throw std::invalid_argument("option --theta must be > 0");
      }
      if (!(o.spec.eps_min >= 0.0 && o.spec.eps_min < 2.0)) {
        throw std::invalid_argument("option --eps_min must lie in [0, 2)");
      }
      in_open_0_2("eps_max", o.spec.eps_max);
    } else if (o.spec.type == "table") {
      require("file", "table");
      o.spec.file = args["file"];
      if (has("eps_min")) {
        o.spec.eps_min = num("eps_min");
        in_open_0_2("eps_min", o.spec.eps_min);
      }
      if (has("eps_max")) {
        o.spec.eps_max = num("eps_max");
        in_open_0_2("eps_max", o.spec.eps_max);
      }
    } else {
      throw std::invalid_argument("unknown photon_field.type '" + o.spec.type +
                                  "' (mono, powerlaw, blackbody, table)");
    }
    if (has("mu_r")) {
      o.mu_r = num("mu_r");
      if (!(o.mu_r > 0.0)) {
        throw std::invalid_argument("option --mu_r must be > 0");
      }
    }
    if (has("gamma_p_max") && args["gamma_p_max"] != "auto") {
      o.gamma_p_max = num("gamma_p_max");
      if (!(o.gamma_p_max > 1.0)) {
        throw std::invalid_argument(
          "option --gamma_p_max must be > 1 or the word auto");
      }
    }
    if (has("h")) {
      o.h = num("h");
      if (!(o.h > 0.0)) {
        throw std::invalid_argument("option --h must be > 0");
      }
    }
    return o;
  }

} // namespace arch::bh

#endif // ARCHETYPES_HADRONIC_BH_OPTIONS_H
