/**
 * @file archetypes/hadronic/bh_sampler_io.h
 * @brief Writer and minimal reader of the step-4 fixture files (contract
 *        revision 1.0 section 4): the contract-1.4 header style, `# key:
 *        value` with strict JSON values, `columns`, `data_sha256` over the
 *        literal data block, one-space `%.12e` columns.
 *
 * Host-only, standard library plus the hashing and JSON helpers of
 * bh_table_io.h.
 *
 * @namespaces:
 *   - arch::bh::
 */

#ifndef ARCHETYPES_HADRONIC_BH_SAMPLER_IO_H
#define ARCHETYPES_HADRONIC_BH_SAMPLER_IO_H

#include "archetypes/hadronic/bh_table_io.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace arch::bh {

  /**
   * @brief A fixture table: ordered header (key, JSON text) pairs written
   *        before `columns` and `data_sha256`, the column names, the rows
   */
  struct FixtureTable {
    std::vector<std::pair<std::string, std::string>> header;
    std::vector<std::string>                         columns;
    std::vector<std::vector<double>>                 rows;
  };

  /// the data block: every row at %.12e, one space between columns, newline
  inline std::string format_fixture_data(const FixtureTable& t) {
    std::string out;
    out.reserve(t.rows.size() * (t.columns.size() * 20 + 1));
    for (std::size_t r = 0; r < t.rows.size(); ++r) {
      if (t.rows[r].size() != t.columns.size()) {
        throw std::invalid_argument("format_fixture_data: row " + std::to_string(r) +
                                    " has " + std::to_string(t.rows[r].size()) +
                                    " values for " + std::to_string(t.columns.size()) +
                                    " columns");
      }
      for (std::size_t c = 0; c < t.rows[r].size(); ++c) {
        if (!std::isfinite(t.rows[r][c])) {
          throw std::invalid_argument("format_fixture_data: non-finite value in row " +
                                      std::to_string(r) + ", column " + t.columns[c]);
        }
        if (c) {
          out += ' ';
        }
        out += format_e12(t.rows[r][c]);
      }
      out += '\n';
    }
    return out;
  }

  /// the whole file: header lines, `columns`, `data_sha256`, data block
  inline std::string format_fixture_file(const FixtureTable& t) {
    const std::string  data = format_fixture_data(t);
    const std::string  sha  = Sha256::digest(data);
    std::ostringstream h;
    for (const auto& [key, json] : t.header) {
      if (key == "columns" || key == "data_sha256") {
        throw std::invalid_argument("format_fixture_file: '" + key +
                                    "' is written by the formatter, not the header");
      }
      h << "# " << key << ": " << json << "\n";
    }
    std::string cols = "[";
    for (std::size_t i = 0; i < t.columns.size(); ++i) {
      cols += (i ? ", " : "") + json_string(t.columns[i]);
    }
    cols += "]";
    h << "# columns: " << cols << "\n";
    h << "# data_sha256: " << json_string(sha) << "\n";
    return h.str() + data;
  }

  inline void write_fixture_file(const std::string& path, const FixtureTable& t) {
    const std::string content = format_fixture_file(t);
    std::ofstream     out(path, std::ios::binary);
    if (!out) {
      throw std::runtime_error("write_fixture_file: cannot open " + path);
    }
    out << content;
  }

  /**
   * @brief Minimal reader: header values as raw JSON text (validated as strict
   *        JSON), `columns` decoded, the data block hashed literally and
   *        compared with `data_sha256`, rows parsed as finite numbers
   */
  struct FixtureContents {
    std::map<std::string, std::string> header; // key -> raw JSON text
    std::vector<std::string>           columns;
    std::vector<std::vector<double>>   rows;
    std::string                        data_sha256;
  };

  inline FixtureContents parse_fixture_file(const std::string& text,
                                            const std::string& name = "fixture") {
    FixtureContents out;
    std::size_t     pos = 0;
    // header: lines starting with '#'
    while (pos < text.size() && text[pos] == '#') {
      const std::size_t eol  = text.find('\n', pos);
      const std::string line = text.substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
      pos                    = eol == std::string::npos ? text.size() : eol + 1;
      const std::size_t colon = line.find(':');
      if (line.size() < 2 || line[1] != ' ' || colon == std::string::npos) {
        throw std::runtime_error(name + ": malformed header line '" + line + "'");
      }
      const std::string key  = line.substr(2, colon - 2);
      std::string       json = line.substr(colon + 1);
      while (!json.empty() && json.front() == ' ') {
        json.erase(json.begin());
      }
      // strict JSON validation (throws on anything that is not JSON)
      parse_json(json);
      if (out.header.count(key)) {
        throw std::runtime_error(name + ": duplicate header key '" + key + "'");
      }
      out.header[key] = json;
    }
    const std::string data = text.substr(pos);
    const auto        it_sha = out.header.find("data_sha256");
    if (it_sha == out.header.end()) {
      throw std::runtime_error(name + ": missing data_sha256");
    }
    out.data_sha256 = json_unescape(it_sha->second);
    if (Sha256::digest(data) != out.data_sha256) {
      throw std::runtime_error(name + ": data_sha256 mismatch");
    }
    const auto it_cols = out.header.find("columns");
    if (it_cols == out.header.end()) {
      throw std::runtime_error(name + ": missing columns");
    }
    {
      const JsonValue cols = parse_json(it_cols->second);
      if (cols.type != JsonValue::Type::Array) {
        throw std::runtime_error(name + ": columns is not an array");
      }
      for (const auto& c : cols.array) {
        if (c.type != JsonValue::Type::String) {
          throw std::runtime_error(name + ": column name is not a string");
        }
        out.columns.push_back(c.string);
      }
    }
    // rows
    std::istringstream in(data);
    std::string        line;
    std::size_t        lineno = 0;
    while (std::getline(in, line)) {
      ++lineno;
      std::istringstream  ls(line);
      std::vector<double> row;
      std::string         tok;
      while (ls >> tok) {
        double v = 0.0;
        if (!parse_number_strict(tok, v)) {
          throw std::runtime_error(name + ": bad number '" + tok + "' in data line " +
                                   std::to_string(lineno));
        }
        row.push_back(v);
      }
      if (row.size() != out.columns.size()) {
        throw std::runtime_error(name + ": data line " + std::to_string(lineno) + " has " +
                                 std::to_string(row.size()) + " values for " +
                                 std::to_string(out.columns.size()) + " columns");
      }
      out.rows.push_back(std::move(row));
    }
    return out;
  }

  inline FixtureContents read_fixture_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("read_fixture_file: cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_fixture_file(ss.str(), path);
  }

} // namespace arch::bh

#endif // ARCHETYPES_HADRONIC_BH_SAMPLER_IO_H
