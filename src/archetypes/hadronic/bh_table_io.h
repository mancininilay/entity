/**
 * @file archetypes/hadronic/bh_table_io.h
 * @brief Exchange-file format of the Bethe-Heitler startup tables
 *        (contract revision 1.3, section 4) and a self-contained SHA-256
 *
 * File layout: UTF-8 text; header lines `# key: value` with JSON values;
 * then whitespace-separated columns `gamma_p beta_p y J I Dmean eps_mean`
 * at `%.12e`. The header carries the SHA-256 of the exact data block (every
 * byte from the first data line to the final newline).
 *
 * @namespaces:
 *   - arch::bh::
 */

#ifndef ARCHETYPES_HADRONIC_BH_TABLE_IO_H
#define ARCHETYPES_HADRONIC_BH_TABLE_IO_H

#include "archetypes/hadronic/bh_physics.h"

#include <array>
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
  // Writer
  // ------------------------------------------------------------------------ //
  struct TableFileMeta {
    std::string producer { "entity-cpp" };
    std::string status { "cpp_candidate" };
    std::string commit { "unknown" };
  };

  inline std::string format_e12(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.12e", v);
    return std::string(buf);
  }

  /**
   * @brief The data block: one line per node, 7 columns at %.12e
   */
  inline std::string format_table_data(const HadronicTablesHost& T) {
    std::string out;
    out.reserve(T.size() * 120);
    for (std::size_t i = 0; i < T.size(); ++i) {
      out += format_e12(T.gamma[i]);
      out += "  ";
      out += format_e12(T.beta[i]);
      out += "  ";
      out += format_e12(T.y[i]);
      out += "  ";
      out += format_e12(T.J[i]);
      out += "  ";
      out += format_e12(T.I[i]);
      out += "  ";
      out += format_e12(T.Dmean[i]);
      out += "  ";
      out += format_e12(T.eps_mean[i]);
      out += "\n";
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
    auto str = [](const std::string& s) {
      return "\"" + s + "\"";
    };
    line("format", str("entity-bh-table"));
    line("format_revision", str("contract-1.3"));
    line("producer", str(meta.producer));
    line("status", str(meta.status));
    line("commit", str(meta.commit));
    {
      std::string b = "{\"type\": " + str(bath.type());
      for (const auto& [k, v] : bath.parameters()) {
        b += ", \"" + k + "\": " + v;
      }
      b += "}";
      line("bath", b);
    }
    line("physical_mass_ratio", json_number(T.mu_r));
    line("gamma_p_max",
         T.gamma_p_max_input > 0.0 ? json_number(T.gamma_p_max_input) : "null");
    line("eps_prime_max", json_number(T.eps_prime_max));
    line("gamma_envelope", json_number(T.gamma_envelope));
    line("gamma_th", json_number(T.gamma_th));
    line("gamma_min", json_number(T.gamma_min));
    line("gamma_max", json_number(T.gamma_max));
    line("y_min", json_number(T.y_min));
    line("y_max", json_number(T.y_max));
    line("grid_variable", str("log10(gamma_p - gamma_th)"));
    line("nodes_per_decade_fine", std::to_string(T.nodes_per_decade_fine));
    line("nodes_per_decade_coarse", std::to_string(T.nodes_per_decade_coarse));
    line("n_fine", std::to_string(T.n_fine));
    line("n_coarse", std::to_string(T.n_coarse));
    line("n_nodes", std::to_string(T.size()));
    line("I_peak", json_number(T.I_peak));
    line("lower_edge_fraction", json_number(constants::LOWER_EDGE_FRACTION));
    line("clipped_number_fraction", json_number(T.clipped_number_fraction));
    line("clipped_energy_fraction", json_number(T.clipped_energy_fraction));
    line("quadrature",
         "{\"delta0\": " + json_number(q.delta0) +
           ", \"h_log_delta\": " + json_number(q.h_log_delta) +
           ", \"h_log_eps\": " + json_number(q.h_log_eps) +
           ", \"gl_order\": " + std::to_string(q.gl_order) +
           ", \"k_max\": " + json_number(q.k_max) + "}");
    line("columns", "[\"gamma_p\", \"beta_p\", \"y\", \"J\", \"I\", \"Dmean\", \"eps_mean\"]");
    line("format_columns", str("%.12e"));
    line("sha256_data", str(sha));
    return h.str() + data;
  }

  inline void write_table_file(const std::string&        path,
                               const HadronicTablesHost& T,
                               const PhotonBath&         bath,
                               const Quadrature&         q,
                               const TableFileMeta&      meta) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
      throw std::runtime_error("write_table_file: cannot open " + path);
    }
    out << format_table_file(T, bath, q, meta);
  }

  // ------------------------------------------------------------------------ //
  // Reader (header map + columns; verifies the SHA-256 of the data block)
  // ------------------------------------------------------------------------ //
  struct TableFileContents {
    std::map<std::string, std::string> header; // key -> raw JSON text
    std::vector<double>                gamma, beta, y, J, I, Dmean, eps_mean;
    std::string                        sha256_data; // recomputed
    bool                               sha_ok { false };
  };

  inline TableFileContents read_table_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("read_table_file: cannot open " + path);
    }
    TableFileContents c;
    std::string       line, data;
    while (std::getline(in, line)) {
      if (line.empty()) {
        continue;
      }
      if (line[0] == '#') {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
          continue;
        }
        std::string key  = line.substr(1, colon - 1);
        std::string val  = line.substr(colon + 1);
        auto        trim = [](std::string& s) {
          const auto a = s.find_first_not_of(" \t");
          const auto b = s.find_last_not_of(" \t\r");
          s = (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
        };
        trim(key);
        trim(val);
        c.header[key] = val;
        continue;
      }
      data += line;
      data += "\n";
      std::istringstream ss(line);
      double             g, b, yy, J, I, D, e;
      if (!(ss >> g >> b >> yy >> J >> I >> D >> e)) {
        throw std::runtime_error("read_table_file: malformed data line");
      }
      c.gamma.push_back(g);
      c.beta.push_back(b);
      c.y.push_back(yy);
      c.J.push_back(J);
      c.I.push_back(I);
      c.Dmean.push_back(D);
      c.eps_mean.push_back(e);
    }
    c.sha256_data = Sha256::digest(data);
    const auto it = c.header.find("sha256_data");
    c.sha_ok      = (it != c.header.end()) &&
               (it->second == "\"" + c.sha256_data + "\"");
    return c;
  }

} // namespace arch::bh

#endif // ARCHETYPES_HADRONIC_BH_TABLE_IO_H
