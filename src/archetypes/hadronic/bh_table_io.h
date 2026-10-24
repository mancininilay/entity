/**
 * @file archetypes/hadronic/bh_table_io.h
 * @brief Exchange-file format of the Bethe-Heitler startup tables
 *        (contract revision 1.4, section 2) and a self-contained SHA-256
 *
 * Contract 1.4 schema (section 2): UTF-8 text; header lines `# key: value`
 * with strict JSON values, one key per line, no blank lines; the data block
 * starts at the first line not beginning with `#` and runs to the end of the
 * file; each data line is seven numbers at `%.12e` separated by one space and
 * terminated by a newline; `data_sha256` is the SHA-256 of every byte of the
 * data block. The reader hashes the literal bytes and validates the content.
 *
 * @namespaces:
 *   - arch::bh::
 */

#ifndef ARCHETYPES_HADRONIC_BH_TABLE_IO_H
#define ARCHETYPES_HADRONIC_BH_TABLE_IO_H

#include "archetypes/hadronic/bh_physics.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace arch::bh {

  // ------------------------------------------------------------------------ //
  // SHA-256 (FIPS 180-4), self-contained
  // ------------------------------------------------------------------------ //
  class Sha256 {
  public:
    Sha256() {
      reset();
    }

    void reset() {
      m_h       = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
      m_len     = 0;
      m_buf_len = 0;
    }

    void update(const void* data, std::size_t n) {
      const auto* p  = static_cast<const std::uint8_t*>(data);
      m_len         += n;
      while (n > 0) {
        const std::size_t take = std::min(n,
                                          static_cast<std::size_t>(64 - m_buf_len));
        std::memcpy(m_buf.data() + m_buf_len, p, take);
        m_buf_len += take;
        p         += take;
        n         -= take;
        if (m_buf_len == 64) {
          transform(m_buf.data());
          m_buf_len = 0;
        }
      }
    }

    void update(const std::string& s) {
      update(s.data(), s.size());
    }

    std::string hex_digest() {
      const std::uint64_t bits = m_len * 8;
      const std::uint8_t  one  = 0x80;
      update(&one, 1);
      const std::uint8_t zero = 0;
      while (m_buf_len != 56) {
        update(&zero, 1);
      }
      std::uint8_t lenb[8];
      for (int i = 0; i < 8; ++i) {
        lenb[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
      }
      update(lenb, 8);
      std::string out;
      char        hex[9];
      for (const std::uint32_t v : m_h) {
        std::snprintf(hex, sizeof(hex), "%08x", v);
        out += hex;
      }
      return out;
    }

    static std::string digest(const std::string& s) {
      Sha256 h;
      h.update(s);
      return h.hex_digest();
    }

  private:
    static std::uint32_t rotr(std::uint32_t x, int n) {
      return (x >> n) | (x << (32 - n));
    }

    void transform(const std::uint8_t* block) {
      static constexpr std::uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
        0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
        0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
        0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
        0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
        0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
      };
      std::uint32_t w[64];
      for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24) |
               (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
               (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) |
               static_cast<std::uint32_t>(block[4 * i + 3]);
      }
      for (int i = 16; i < 64; ++i) {
        const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^
                                 (w[i - 15] >> 3);
        const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^
                                 (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
      }
      std::uint32_t a = m_h[0], b = m_h[1], c = m_h[2], d = m_h[3], e = m_h[4],
                    f = m_h[5], g = m_h[6], h = m_h[7];
      for (int i = 0; i < 64; ++i) {
        const std::uint32_t S1  = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const std::uint32_t ch  = (e & f) ^ (~e & g);
        const std::uint32_t t1  = h + S1 + ch + K[i] + w[i];
        const std::uint32_t S0  = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t t2  = S0 + maj;
        h                       = g;
        g                       = f;
        f                       = e;
        e                       = d + t1;
        d                       = c;
        c                       = b;
        b                       = a;
        a                       = t1 + t2;
      }
      m_h[0] += a;
      m_h[1] += b;
      m_h[2] += c;
      m_h[3] += d;
      m_h[4] += e;
      m_h[5] += f;
      m_h[6] += g;
      m_h[7] += h;
    }

    std::array<std::uint32_t, 8> m_h {};
    std::array<std::uint8_t, 64> m_buf {};
    std::uint64_t                m_len { 0 };
    std::size_t                  m_buf_len { 0 };
  };

  // ------------------------------------------------------------------------ //
  // Minimal JSON helpers for the header values
  // ------------------------------------------------------------------------ //

  /**
   * @brief Decode a JSON string literal (with its quotes); throws otherwise
   */
  inline std::string json_unescape(const std::string& raw) {
    if (raw.size() < 2 || raw.front() != '"' || raw.back() != '"') {
      throw std::runtime_error("expected a JSON string, got " + raw);
    }
    std::string out;
    for (std::size_t i = 1; i + 1 < raw.size(); ++i) {
      const char c = raw[i];
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i + 2 >= raw.size()) {
        throw std::runtime_error("dangling escape in JSON string " + raw);
      }
      const char e = raw[++i];
      switch (e) {
        case '"':
          out += '"';
          break;
        case '\\':
          out += '\\';
          break;
        case '/':
          out += '/';
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u': {
          if (i + 4 >= raw.size()) {
            throw std::runtime_error("truncated \\u escape in " + raw);
          }
          const unsigned long cp = std::stoul(raw.substr(i + 1, 4), nullptr, 16);
          i += 4;
          if (cp < 0x80) {
            out += static_cast<char>(cp);
          } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
          }
          break;
        }
        default:
          throw std::runtime_error(
            std::string("invalid escape \\") + e + " in " + raw);
      }
    }
    return out;
  }

  /**
   * @brief Strict JSON number (finite); throws otherwise
   */
  inline double json_to_number(const std::string& raw, const std::string& key) {
    double v = 0.0;
    if (!parse_number_strict(raw, v) || raw == "nan" || raw == "inf") {
      throw std::runtime_error(
        "header key " + key + ": '" + raw + "' is not a finite JSON number");
    }
    return v;
  }

  /**
   * @brief JSON array of strings, e.g. ["a", "b"]; throws otherwise
   */
  inline std::vector<std::string> json_string_array(const std::string& raw) {
    auto trim = [](std::string s) {
      const auto a = s.find_first_not_of(" \t");
      const auto b = s.find_last_not_of(" \t");
      return (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
    };
    const std::string t = trim(raw);
    if (t.size() < 2 || t.front() != '[' || t.back() != ']') {
      throw std::runtime_error("expected a JSON array, got " + raw);
    }
    std::vector<std::string> out;
    std::string              inner = t.substr(1, t.size() - 2);
    std::size_t              pos   = 0;
    while (pos < inner.size()) {
      const auto q1 = inner.find('"', pos);
      if (q1 == std::string::npos) {
        break;
      }
      std::size_t q2 = q1 + 1;
      while (q2 < inner.size() && !(inner[q2] == '"' && inner[q2 - 1] != '\\')) {
        ++q2;
      }
      if (q2 >= inner.size()) {
        throw std::runtime_error("unterminated string in JSON array " + raw);
      }
      out.push_back(json_unescape(inner.substr(q1, q2 - q1 + 1)));
      pos = q2 + 1;
    }
    return out;
  }

  /**
   * @brief JSON number for the header; non-finite metadata is an error
   */
  inline std::string json_number_checked(double v, const std::string& key) {
    if (!std::isfinite(v)) {
      throw std::runtime_error(
        "table header: " + key + " is not finite and cannot be written");
    }
    return json_number(v);
  }

  // ------------------------------------------------------------------------ //
  // Writer (contract 1.4 exchange schema)
  // ------------------------------------------------------------------------ //
  struct TableFileMeta {
    std::string producer { "entity-cpp" };
    std::string status { "cpp_candidate" };
    std::string version { "unknown" }; // git commit of the producer
  };

  inline const std::vector<std::string>& canonical_columns() {
    static const std::vector<std::string> cols = { "gamma_p", "beta_p", "y",
                                                   "J",       "I",      "Dmean",
                                                   "eps_mean" };
    return cols;
  }

  inline std::string format_e12(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.12e", v);
    return std::string(buf);
  }

  /**
   * @brief SHA-256 of the bytes of a file
   */
  inline std::string sha256_of_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("sha256_of_file: cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return Sha256::digest(ss.str());
  }

  /**
   * @brief The data block: one line per node, seven numbers at %.12e separated
   *        by one space, each line terminated by a newline
   */
  inline std::string format_table_data(const HadronicTablesHost& T) {
    std::string out;
    out.reserve(T.size() * 140);
    for (std::size_t i = 0; i < T.size(); ++i) {
      out += format_e12(T.gamma[i]);
      out += ' ';
      out += format_e12(T.beta[i]);
      out += ' ';
      out += format_e12(T.y[i]);
      out += ' ';
      out += format_e12(T.J[i]);
      out += ' ';
      out += format_e12(T.I[i]);
      out += ' ';
      out += format_e12(T.Dmean[i]);
      out += ' ';
      out += format_e12(T.eps_mean[i]);
      out += '\n';
    }
    return out;
  }

  /**
   * @brief The whole file (header + data block) as a string
   */
  inline std::string format_table_file(const HadronicTablesHost& T,
                                       const PhotonBath&         bath,
                                       const Quadrature&         q,
                                       const TableFileMeta&      meta) {
    const std::string  data = format_table_data(T);
    const std::string  sha  = Sha256::digest(data);
    std::ostringstream h;
    auto line = [&h](const std::string& key, const std::string& json) {
      h << "# " << key << ": " << json << "\n";
    };
    auto num = [](const std::string& key, double v) {
      return json_number_checked(v, key);
    };
    line("format", json_string("bh-rate-table"));
    line("format_revision", "1");
    line("producer", json_string(meta.producer));
    line("version", json_string(meta.version));
    line("status", json_string(meta.status));
    line("physical_mass_ratio", num("physical_mass_ratio", T.mu_r));
    {
      std::string b = "{\"type\": " + json_string(bath.type());
      for (const auto& [k, v] : bath.parameters()) {
        b += ", " + json_string(k) + ": " + v;
      }
      if (!bath.source_file().empty()) {
        b += ", \"file_sha256\": " +
             json_string(sha256_of_file(bath.source_file()));
      }
      b += "}";
      line("photon_field", b);
    }
    line("gamma_p_max_input",
         T.gamma_p_max_input > 0.0 ? num("gamma_p_max_input", T.gamma_p_max_input)
                                   : "null");
    line("eps_prime_max", num("eps_prime_max", T.eps_prime_max));
    line("gamma_envelope", num("gamma_envelope", T.gamma_envelope));
    line("gamma_th", num("gamma_th", T.gamma_th));
    line("gamma_min", num("gamma_min", T.gamma_min));
    line("gamma_max", num("gamma_max", T.gamma_max));
    line("y_min", num("y_min", T.y_min));
    line("y_max", num("y_max", T.y_max));
    line("grid_variable", json_string("log10(gamma_p - gamma_th)"));
    line("lower_edge_fraction",
         num("lower_edge_fraction", constants::LOWER_EDGE_FRACTION));
    line("first_decade_intervals_per_decade",
         std::to_string(T.nodes_per_decade_fine));
    line("other_intervals_per_decade", std::to_string(T.nodes_per_decade_coarse));
    line("endpoint_coalescing",
         num("endpoint_coalescing", constants::ENDPOINT_COALESCING));
    line("first_decade_node_count", std::to_string(T.n_fine));
    line("node_count", std::to_string(T.size()));
    line("I_peak", num("I_peak", T.I_peak));
    line("gamma_at_I_peak", num("gamma_at_I_peak", T.gamma_at_I_peak));
    line("number_normalization",
         num("number_normalization", bath.number_normalization()));
    line("mean_photon_energy", num("mean_photon_energy", bath.mean_eps()));
    line("discarded_number_fraction",
         num("discarded_number_fraction", T.clipped_number_fraction));
    line("discarded_energy_fraction",
         num("discarded_energy_fraction", T.clipped_energy_fraction));
    line("quadrature",
         "{\"delta0\": " + num("quadrature.delta0", q.delta0) +
           ", \"h_log_delta\": " + num("quadrature.h_log_delta", q.h_log_delta) +
           ", \"h_log_eps\": " + num("quadrature.h_log_eps", q.h_log_eps) +
           ", \"gl_order\": " + std::to_string(q.gl_order) + ", \"k_max\": " +
           num("quadrature.k_max", q.k_max) + ", \"columns_format\": \"%.12e\"}");
    {
      std::string cols = "[";
      for (std::size_t i = 0; i < canonical_columns().size(); ++i) {
        cols += (i ? ", " : "") + json_string(canonical_columns()[i]);
      }
      cols += "]";
      line("columns", cols);
    }
    line("data_sha256", json_string(sha));
    return h.str() + data;
  }

  inline void write_table_file(const std::string&        path,
                               const HadronicTablesHost& T,
                               const PhotonBath&         bath,
                               const Quadrature&         q,
                               const TableFileMeta&      meta) {
    const std::string content = format_table_file(T, bath, q, meta);
    std::ofstream     out(path, std::ios::binary);
    if (!out) {
      throw std::runtime_error("write_table_file: cannot open " + path);
    }
    out << content;
  }

  // ------------------------------------------------------------------------ //
  // Reader: hashes the literal data bytes, then validates everything
  // ------------------------------------------------------------------------ //
  struct TableFileContents {
    std::map<std::string, std::string> header; // key -> raw JSON text
    // decoded required metadata
    std::string format, producer, version, status, grid_variable;
    long        format_revision { 0 };
    double      physical_mass_ratio { 0.0 };
    double      eps_prime_max { 0.0 }, gamma_envelope { 0.0 };
    double      gamma_th { 0.0 }, gamma_min { 0.0 }, gamma_max { 0.0 };
    double      y_min { 0.0 }, y_max { 0.0 };
    double      lower_edge_fraction { 0.0 }, endpoint_coalescing { 0.0 };
    long        first_decade_intervals_per_decade { 0 };
    long        other_intervals_per_decade { 0 };
    std::size_t first_decade_node_count { 0 }, node_count { 0 };
    double      I_peak { 0.0 }, gamma_at_I_peak { 0.0 };
    double      number_normalization { 0.0 }, mean_photon_energy { 0.0 };
    double      discarded_number_fraction { 0.0 };
    double      discarded_energy_fraction { 0.0 };
    std::vector<std::string> columns;
    std::string              data_sha256; // from the header, verified
    // columns
    std::vector<double>      gamma, beta, y, J, I, Dmean, eps_mean;
  };

  inline const std::vector<std::string>& required_header_keys() {
    static const std::vector<std::string> keys = {
      "format",
      "format_revision",
      "producer",
      "version",
      "status",
      "physical_mass_ratio",
      "photon_field",
      "gamma_p_max_input",
      "eps_prime_max",
      "gamma_envelope",
      "gamma_th",
      "gamma_min",
      "gamma_max",
      "y_min",
      "y_max",
      "grid_variable",
      "lower_edge_fraction",
      "first_decade_intervals_per_decade",
      "other_intervals_per_decade",
      "endpoint_coalescing",
      "first_decade_node_count",
      "node_count",
      "I_peak",
      "gamma_at_I_peak",
      "number_normalization",
      "mean_photon_energy",
      "discarded_number_fraction",
      "discarded_energy_fraction",
      "quadrature",
      "columns",
      "data_sha256"
    };
    return keys;
  }

  /**
   * @brief Parse and validate a table file given as bytes
   * @throws std::runtime_error on any violation of the schema, including a
   *         missing or mismatched data_sha256
   */
  inline TableFileContents parse_table_file(
    const std::string& bytes,
    const std::string& name = "<memory>") {
    auto fail = [&name](const std::string& what) {
      throw std::runtime_error("table file " + name + ": " + what);
    };
    TableFileContents c;
    // --- header: lines starting with '#', up to the first line that does not
    std::size_t       pos    = 0;
    std::size_t       lineno = 0;
    while (pos < bytes.size() && bytes[pos] == '#') {
      const auto nl = bytes.find('\n', pos);
      if (nl == std::string::npos) {
        fail("header line " + std::to_string(lineno + 1) + " lacks a newline");
      }
      ++lineno;
      const std::string line  = bytes.substr(pos, nl - pos);
      const auto        colon = line.find(':');
      if (colon == std::string::npos) {
        fail("header line " + std::to_string(lineno) + " is not '# key: value'");
      }
      auto trim = [](std::string s) {
        const auto a = s.find_first_not_of(" \t");
        const auto b = s.find_last_not_of(" \t\r");
        return (a == std::string::npos) ? std::string() : s.substr(a, b - a + 1);
      };
      const std::string key = trim(line.substr(1, colon - 1));
      const std::string val = trim(line.substr(colon + 1));
      if (key.empty() || val.empty()) {
        fail("header line " + std::to_string(lineno) + " has an empty key or value");
      }
      if (c.header.count(key)) {
        fail("header key " + key + " repeated");
      }
      c.header[key] = val;
      pos           = nl + 1;
    }
    // --- data block: every remaining byte, verbatim
    const std::string data = bytes.substr(pos);
    if (data.empty()) {
      fail("empty data block");
    }
    if (data.back() != '\n') {
      fail("data block does not end with a newline");
    }
    for (const std::string& k : required_header_keys()) {
      if (!c.header.count(k)) {
        fail("missing required header key " + k);
      }
    }
    c.data_sha256            = json_unescape(c.header.at("data_sha256"));
    const std::string digest = Sha256::digest(data);
    if (c.data_sha256 != digest) {
      fail("data_sha256 mismatch: header " + c.data_sha256 + ", data " + digest);
    }
    // --- decode the required metadata
    c.format = json_unescape(c.header.at("format"));
    if (c.format != "bh-rate-table") {
      fail("format is " + c.format + ", expected bh-rate-table");
    }
    c.format_revision = static_cast<long>(
      json_to_number(c.header.at("format_revision"), "format_revision"));
    if (c.format_revision != 1) {
      fail("format_revision " + std::to_string(c.format_revision) +
           " not supported");
    }
    c.producer      = json_unescape(c.header.at("producer"));
    c.version       = json_unescape(c.header.at("version"));
    c.status        = json_unescape(c.header.at("status"));
    c.grid_variable = json_unescape(c.header.at("grid_variable"));
    auto numkey     = [&c](const std::string& k) {
      return json_to_number(c.header.at(k), k);
    };
    c.physical_mass_ratio               = numkey("physical_mass_ratio");
    c.eps_prime_max                     = numkey("eps_prime_max");
    c.gamma_envelope                    = numkey("gamma_envelope");
    c.gamma_th                          = numkey("gamma_th");
    c.gamma_min                         = numkey("gamma_min");
    c.gamma_max                         = numkey("gamma_max");
    c.y_min                             = numkey("y_min");
    c.y_max                             = numkey("y_max");
    c.lower_edge_fraction               = numkey("lower_edge_fraction");
    c.endpoint_coalescing               = numkey("endpoint_coalescing");
    c.first_decade_intervals_per_decade = static_cast<long>(
      numkey("first_decade_intervals_per_decade"));
    c.other_intervals_per_decade = static_cast<long>(
      numkey("other_intervals_per_decade"));
    c.I_peak                    = numkey("I_peak");
    c.gamma_at_I_peak           = numkey("gamma_at_I_peak");
    c.number_normalization      = numkey("number_normalization");
    c.mean_photon_energy        = numkey("mean_photon_energy");
    c.discarded_number_fraction = numkey("discarded_number_fraction");
    c.discarded_energy_fraction = numkey("discarded_energy_fraction");
    {
      const double n1 = numkey("first_decade_node_count");
      const double n2 = numkey("node_count");
      if (n1 < 0.0 || n2 < 0.0 || n1 != std::floor(n1) || n2 != std::floor(n2)) {
        fail("node counts must be non-negative integers");
      }
      c.first_decade_node_count = static_cast<std::size_t>(n1);
      c.node_count              = static_cast<std::size_t>(n2);
    }
    if (c.header.at("gamma_p_max_input") != "null") {
      numkey("gamma_p_max_input");
    }
    c.columns = json_string_array(c.header.at("columns"));
    if (c.columns != canonical_columns()) {
      fail("columns differ from the canonical list");
    }
    // --- parse the data bytes separately from the hash
    std::size_t start = 0, row = 0;
    while (start < data.size()) {
      const auto        nl   = data.find('\n', start);
      const std::string line = data.substr(start, nl - start);
      ++row;
      start = nl + 1;
      std::istringstream  ss(line);
      std::string         tok;
      std::vector<double> vals;
      while (ss >> tok) {
        double v = 0.0;
        if (!parse_number_strict(tok, v)) {
          fail("data row " + std::to_string(row) + ": token '" + tok +
               "' is not a finite number");
        }
        vals.push_back(v);
      }
      if (vals.size() != 7) {
        fail("data row " + std::to_string(row) +
             ": expected 7 columns, found " + std::to_string(vals.size()));
      }
      c.gamma.push_back(vals[0]);
      c.beta.push_back(vals[1]);
      c.y.push_back(vals[2]);
      c.J.push_back(vals[3]);
      c.I.push_back(vals[4]);
      c.Dmean.push_back(vals[5]);
      c.eps_mean.push_back(vals[6]);
    }
    // --- content validation
    if (c.gamma.size() != c.node_count) {
      fail("node_count " + std::to_string(c.node_count) + " differs from the " +
           std::to_string(c.gamma.size()) + " data rows");
    }
    if (c.gamma.size() < 2) {
      fail("fewer than two nodes");
    }
    for (std::size_t i = 0; i < c.gamma.size(); ++i) {
      if (!(c.J[i] > 0.0) || !(c.I[i] > 0.0)) {
        fail("J or I not strictly positive at row " + std::to_string(i + 1));
      }
      if (i > 0 && !(c.gamma[i] > c.gamma[i - 1])) {
        fail("gamma_p not strictly increasing at row " + std::to_string(i + 1));
      }
    }
    return c;
  }

  inline TableFileContents read_table_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("read_table_file: cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_table_file(ss.str(), path);
  }

} // namespace arch::bh

#endif // ARCHETYPES_HADRONIC_BH_TABLE_IO_H
