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
int main() {
  const double pps[] = {1e-4, 1e-2, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 3.0, 30.0, 300.0, 999.0};
  const double pms[] = {1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.15, 0.3, 1.0, 30.0, 999.0};
  
  std::printf("law 3 (production dispatch) in double: L1-relative error int|dh| / int|h| and first-moment error; rows p+, cols p-\n%8s", "");
  for (double pm : pms) std::printf(" %8.0e", pm); std::printf("\n");
  double wl1 = 0, wm1 = 0;
  for (double pp : pps) { std::printf("%8.0e", pp);
    for (double pm : pms) {
      const double Ep = std::sqrt(1+pp*pp), Em = std::sqrt(1+pm*pm), k = Ep + Em;
      double num = 0, den = 0, m1d = 0, m1q = 0;
      // explicit loop to accumulate several sums in one pass
      const double w = angular_scale(Ep), t0 = std::log(w), t1 = std::log(2.0 + w);
      const arch::bh::GaussLegendre gl(24); const int np = (int)std::ceil((t1 - t0)/0.1);
      for (int p = 0; p < np; ++p) { const double a = t0 + (t1-t0)*p/np, b = t0 + (t1-t0)*(p+1)/np;
        for (int i = 0; i < gl.order(); ++i) { const double t = 0.5*(a+b) + 0.5*(b-a)*gl.x[i], et = std::exp(t);
          const double om = std::min(2.0, std::max(0.0, et - w)), wt = 0.5*(b-a)*gl.w[i]*et;
          const Q Epq = (Q)Ep, Emq = (Q)k - Epq, xp = Epq - 1, xm = Emq - 1, ppq = sqrtq(xp*(xp+2)), pmq = sqrtq(xm*(xm+2));
          const double r = (double)(-ppq*pmq*stable::angle_K_general<Q>(Epq, Emq, ppq, pmq, (Q)om, -1, 1, 1)/(4*(Q)k*(Q)k*(Q)k));
          const double d = dsigma_dE_dcos_T<double>(k, Ep, om);
          num += wt*std::fabs(d - r); den += wt*std::fabs(r); m1d += wt*(1-om)*d; m1q += wt*(1-om)*r; } }
      const double l1 = num/den, em = std::fabs(m1d - m1q)/den;
      wl1 = std::max(wl1, l1); wm1 = std::max(wm1, em);
      std::printf(" %8.1e", l1);
    } std::printf("\n"); }
  std::printf("worst L1-relative %.2e, worst first-moment error / int h %.2e\n", wl1, wm1);
}
