#include <quadmath.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <initializer_list>
namespace std { inline __float128 log(__float128 x){return logq(x);} inline __float128 sqrt(__float128 x){return sqrtq(x);}
  inline __float128 atanh(__float128 x){return atanhq(x);} inline __float128 asinh(__float128 x){return asinhq(x);}
  inline __float128 abs(__float128 x){return fabsq(x);} }
#include "archetypes/hadronic/bh_restframe.h"
using namespace arch::bh::rf;
typedef __float128 Q;
// quad references: the general-region formulas in quad (stable sub-expressions; quad has 113 bits)
Q ref_g(Q k, Q Ep) {
  Q Em = k - Ep; if (Ep > Em) std::swap(Ep, Em);
  Q xp = Ep - 1, xm = Em - 1, a = xp*(xp+2), b = xm*(xm+2), pp = sqrtq(a), pm = sqrtq(b);
  return pp*pm*stable::energy_Q_general<Q>(Ep, Em, pp, pm, a, b)/(a*b)/(k*k*k);
}
Q ref_h(Q k, Q Ep, Q omc) {
  Q Em = k - Ep, xp = Ep - 1, xm = Em - 1, pp = sqrtq(xp*(xp+2)), pm = sqrtq(xm*(xm+2));
  return -pp*pm*stable::angle_K_general<Q>(Ep, Em, pp, pm, omc, -1, 1, 1)/(4*k*k*k);
}
int main() {
  const double pps[] = {1e-4, 1e-3, 1e-2, 0.099, 0.101, 0.149, 0.151, 0.3, 1.0, 3.0, 30.0, 300.0, 1000.0};
  std::printf("law 1, |double/ref - 1| (rows p+, cols p-); ref = quad general form\n%8s", "");
  for (double pm : pps) std::printf(" %8.0e", pm); std::printf("\n");
  double w1d = 0, w1l = 0;
  for (double pp : pps) { std::printf("%8.0e", pp);
    for (double pm : pps) {
      const double Ep = std::sqrt(1 + pp*pp), Em = std::sqrt(1 + pm*pm), k = Ep + Em;
      // inputs as doubles; quad reference from the same (k, Ep) doubles
      const double r = (double)ref_g((Q)k, (Q)Ep);
      const double d = dsigma_dE_T<double>(k, Ep), l = (double)dsigma_dE_T<long double>(k, Ep);
      const double ed = std::fabs(d/r - 1), el = std::fabs(l/r - 1);
      w1d = std::max(w1d, ed); w1l = std::max(w1l, el);
      std::printf(" %8.1e", ed);
    } std::printf("\n"); }
  std::printf("law 1 worst: double %.2e, long double %.2e\n", w1d, w1l);
  std::printf("\nlaw 3, max over omc of |double - ref| / (int |h| domc / 2, peak-resolving) (rows p+, cols p-)\n%8s", "");
  for (double pm : pps) std::printf(" %8.0e", pm); std::printf("\n");
  double w3d = 0, w3l = 0;
  for (double pp : pps) { std::printf("%8.0e", pp);
    for (double pm : pps) {
      const double Ep = std::sqrt(1 + pp*pp), Em = std::sqrt(1 + pm*pm), k = Ep + Em;
      RestFrameQuadrature qq; qq.h_t_angle = 0.1; qq.gl_order = 24;
      const double hm = 0.5 * integrate_angle(Ep, [&](double om) { return std::fabs((double)ref_h((Q)k, (Q)Ep, (Q)om)); }, qq);
      double ed = 0, el = 0;
      for (double omc : {1e-12, 1e-8, 1e-5, 1e-3, 0.03, 0.3, 0.7, 1.0, 1.4, 1.8, 1.99, 2.0}) {
        const double r = (double)ref_h((Q)k, (Q)Ep, (Q)omc);
        ed = std::max(ed, std::fabs(dsigma_dE_dcos_T<double>(k, Ep, omc) - r)/hm);
        el = std::max(el, std::fabs((double)dsigma_dE_dcos_T<long double>(k, Ep, omc) - r)/hm);
      }
      w3d = std::max(w3d, ed); w3l = std::max(w3l, el);
      std::printf(" %8.1e", ed);
    } std::printf("\n"); }
  std::printf("law 3 worst: double %.2e, long double %.2e\n", w3d, w3l);
  const double k = 2.0001113973859996, E = 1.0000002682044309;
  std::printf("referee point: quad %.10e  stable double %.10e  round-1 double %.10e\n", (double)ref_g((Q)k,(Q)E), dsigma_dE_T<double>(k,E), dsigma_dE_round1_T<double>(k,E));
}
