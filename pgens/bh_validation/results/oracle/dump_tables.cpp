// Oracle dump of the Bethe-Heitler startup tables (host side, no Kokkos).
// Compiled once against the fork headers at dev/hadronic 72427ff6 and once
// against the headers of the emission-policy branch: the two outputs must be
// byte-identical. Usage: dump_tables > out.txt
#include BH_HEADER_PHYSICS
#include BH_HEADER_BORN
#include BH_HEADER_SAMPLER_HOST
#include BH_HEADER_EVENT
#include BH_HEADER_INTERPOLATE

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace arch::bh;

static std::uint64_t fnv(const void* data, std::size_t n, std::uint64_t h = 1469598103934665603ULL) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}
template <class T>
static std::uint64_t fnv_vec(const std::vector<T>& v) {
  return fnv(v.data(), v.size() * sizeof(T));
}

static void dump_bath(const std::string& name, const BathSpec& spec, double mu_r, double gamma_p_max, bool full_nodes) {
  const auto bath = make_bath(spec);
  Quadrature q;
#if defined(BH_HAS_QK_SOURCE)
  q.qk_source = QkSource::Born;
#endif
  const BetheHeitlerRates  rates(bath, q);
  const HadronicTablesHost H = build_tables(rates, mu_r, gamma_p_max);
  validate_tables(H);
  std::printf("== bath %s mu_r %a gamma_p_max %a\n", name.c_str(), mu_r, gamma_p_max);
  std::printf("gamma_th %a gamma_min %a gamma_max %a y_min %a y_max %a n %zu n_fine %zu I_peak %a\n", H.gamma_th,
              H.gamma_min, H.gamma_max, H.y_min, H.y_max, H.size(), H.n_fine, H.I_peak);
  std::printf("hash gamma %016llx J %016llx I %016llx Dmean %016llx eps_mean %016llx y %016llx\n",
              (unsigned long long)fnv_vec(H.gamma), (unsigned long long)fnv_vec(H.J), (unsigned long long)fnv_vec(H.I),
              (unsigned long long)fnv_vec(H.Dmean), (unsigned long long)fnv_vec(H.eps_mean),
              (unsigned long long)fnv_vec(H.y));
  const std::size_t stride = full_nodes ? 1 : 16;
  for (std::size_t i = 0; i < H.size(); i += stride) {
    std::printf("node %zu %a %a %a %a %a\n", i, H.gamma[i], H.J[i], H.I[i], H.Dmean[i], H.eps_mean[i]);
  }
  // host interpolation and the admissible pair weight at a few Lorentz factors
  for (const double g : { 2.2, 3.0, 6.0, 10.0, 30.0, 100.0, 300.0 }) {
    if (g <= H.gamma_min || g >= H.gamma_max) {
      continue;
    }
    double Dm = 0.0, em = 0.0;
    interpolate_outcome_host(H, g, Dm, em);
    std::printf("interp %a rate %a loss %a Dmean %a eps_mean %a a_max %a\n", g, interpolate_rate_host(H, g),
                interpolate_loss_host(H, g), Dm, em, event::a_max(g, bath->eps_cut(), mu_r, 0.04));
  }
  // the photon-energy tables of the sampler
  const rf::SamplerCumulatives C;
  const rf::BathEpsTables      B = rf::BathEpsTables::build(*bath, H, C, 2048, 8, 0);
  std::printf("eps tables mono %d n_eps %d n_gamma %zu hash y %016llx cdf_u %016llx\n", (int)B.mono, B.n_eps,
              B.gamma.size(), (unsigned long long)fnv_vec(B.y), (unsigned long long)fnv_vec(B.cdf_u));
}

int main() {
  // the rest-frame tables and the cumulative do not depend on the bath
  const rf::RestFrameTables    T = rf::RestFrameTables::build(16, 256, 256, 8, 0);
  const rf::SamplerCumulatives C;
  std::printf("restframe n_k %zu n_s %d n_tau %d hash t_k %016llx cdf_E %016llx cdf_A %016llx\n", T.n_k(), T.n_s, T.n_tau,
              (unsigned long long)fnv_vec(T.t_k), (unsigned long long)fnv_vec(T.cdf_E),
              (unsigned long long)fnv_vec(T.cdf_A));
  std::printf("cumulative n %zu delta0 %a kmax %a hash t %016llx lnPhi %016llx\n", C.size(), C.delta0(), C.kmax(),
              (unsigned long long)fnv_vec(C.t()), (unsigned long long)fnv_vec(C.lnPhi()));
  const auto& BT = born::table();
  std::printf("born nodes %zu slope %a q(2.5) %a q(4) %a q(30) %a q(900) %a\n", BT.size(), BT.threshold_slope(),
              BT.q_k(2.5), BT.q_k(4.0), BT.q_k(30.0), BT.q_k(900.0));

  BathSpec mono05;
  mono05.type = "mono";
  mono05.eps0 = 0.5;
  dump_bath("mono05", mono05, 1836.15, 0.0, true);
  dump_bath("mono05_g12", mono05, 1836.15267343, 12.0, true);
  BathSpec mono005;
  mono005.type = "mono";
  mono005.eps0 = 0.05;
  dump_bath("mono005", mono005, 1836.15, 0.0, false);
  BathSpec pl;
  pl.type    = "powerlaw";
  pl.eps_min = 0.1;
  pl.eps_max = 1.5;
  pl.index   = 2.0;
  dump_bath("powerlaw", pl, 1836.15, 0.0, false);
  BathSpec bb;
  bb.type    = "blackbody";
  bb.theta   = 0.1;
  bb.eps_min = 0.0;
  bb.eps_max = 1.5;
  dump_bath("blackbody", bb, 1836.15, 0.0, false);
  return 0;
}
