// Codex 1: hand-specified witnesses for P13-D43, run on CPU and CUDA.
#include "metrics/minkowski.h"
#include "../../pgens/rec_mix/pgen.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace user::recmix;

static void require(bool value, const char* message) {
  if (not value) throw std::runtime_error(message);
}

static void check_segments() {
  require(segments_aligned(0, 2080, 13), "production geometry");
  require(segments_aligned(2080, 2080, 13), "second production domain");
  require(segments_aligned(26, 26, 13), "alternative decomposition");
  require(not segments_aligned(1, 26, 13), "misaligned origin refused");
  require(not segments_aligned(0, 27, 13), "partial segment refused");
  require(not segments_aligned(0, 26, 0), "zero width refused by segment helper");

  // Width 13: faces 0,13,26,39; rows [12,23), alive particles only.
  const std::vector<int> xs {0,12,13,25,26,38,-1,39,5,5,5,5};
  const std::vector<int> ys {12,22,12,22,12,22,15,15,11,23,15,15};
  array_t<int*> x("x", xs.size()), y("y", xs.size());
  array_t<short*> tag("tag", xs.size());
  auto hx=Kokkos::create_mirror_view(x), hy=Kokkos::create_mirror_view(y);
  auto ht=Kokkos::create_mirror_view(tag);
  for (std::size_t p=0; p<xs.size(); ++p) {
    hx(p)=xs[p]; hy(p)=ys[p]; ht(p)=p==10 ? ntt::ParticleTag::dead : ntt::ParticleTag::alive;
  }
  Kokkos::deep_copy(x,hx); Kokkos::deep_copy(y,hy); Kokkos::deep_copy(tag,ht);
  array_t<unsigned long long*> counts("counts",3);
  Kokkos::deep_copy(counts,0ULL);
  Kokkos::parallel_for("hand count", xs.size(), CountSegments_kernel{x,y,tag,counts,12,23,13,39});
  auto hc=Kokkos::create_mirror_view(counts); Kokkos::deep_copy(hc,counts);
  require(hc(0)==3 and hc(1)==2 and hc(2)==2, "alive strip/segment counts");
  // A second species must add, not reset, the depositing count.
  Kokkos::parallel_for("second species", xs.size(), CountSegments_kernel{x,y,tag,counts,12,23,13,39});
  Kokkos::deep_copy(hc,counts);
  require(hc(0)==6 and hc(1)==4 and hc(2)==4, "counts add across species");

  // 13*11*32=4576 particles. Equal excess and deficit cancel only in a long average.
  require(segment_pair_deficit(4576,11,13,32)==0, "target density");
  require(segment_pair_deficit(4862,11,13,32)==0, "excess clipped");
  require(segment_pair_deficit(4290,11,13,32)==1, "short deficit");
  require(segment_pair_deficit(9152,11,26,32)==0, "long controller cancellation");
  require(segment_pair_deficit(0,11,13,32)==16, "empty strip");

  // Skipped segments must not receive any trials, including at repeated ends.
  array_t<npart_t*> ends("ends",4);
  auto he=Kokkos::create_mirror_view(ends);
  he(0)=0; he(1)=2; he(2)=2; he(3)=5; Kokkos::deep_copy(ends,he);
  array_t<int*> assigned("assigned",5);
  Kokkos::parallel_for("assignment",5,KOKKOS_LAMBDA(int p) {
    assigned(p)=segment_for_trial(p,ends);
  });
  auto ha=Kokkos::create_mirror_view(assigned); Kokkos::deep_copy(ha,assigned);
  for(int p=0;p<5;++p) require(ha(p)==(p<2?1:3), "exclusive segment prefix");

  // Exercise the actual paired injection kernel: only segments 1 and 3
  // receive particles; charge-paired positions and initialized payloads agree.
  using Kernel = BoxInject_kernel<metric::Minkowski<Dim::_2D>>;
  auto make_particles = []() {
    Kernel::Arr v;
    v.i1=array_t<int*>("i1",5); v.i2=array_t<int*>("i2",5);
    v.dx1=array_t<prtldx_t*>("dx1",5); v.dx2=array_t<prtldx_t*>("dx2",5);
    v.ux1=array_t<real_t*>("u1",5); v.ux2=array_t<real_t*>("u2",5);
    v.ux3=array_t<real_t*>("u3",5); v.weight=array_t<real_t*>("w",5);
    v.tag=array_t<short*>("tag",5);
    v.pld_i=array_t<npart_t**>("payload_i",5,1);
    v.pld_r=array_t<real_t**>("payload_r",5,1);
    v.npld_i=1; v.npld_r=1;
    Kokkos::deep_copy(v.pld_i,123U); Kokkos::deep_copy(v.pld_r,real_t(123));
    return v;
  };
  Kernel kern;
  kern.A=make_particles(); kern.B=make_particles(); kern.two=true;
  kern.x1min=0; kern.x1max=52; kern.x2min=12; kern.x2max=13;
  kern.n1loc=52; kern.n2loc=40;
  kern.thA=.01; kern.thB=.0004; kern.uA3=0; kern.uB3=0;
  kern.sheet_profile=false; kern.a=1; kern.x0=0; kern.y0=20; kern.dip2=1;
  kern.off1=0; kern.off2=0; kern.dom_idx=0;
  kern.pool=random_number_pool_t(17);
  kern.segment_ends=ends; kern.segment_width=13;
  Kokkos::parallel_for("batched paired injection",5,kern);
  auto produced=Kokkos::create_mirror_view(kern.idx); Kokkos::deep_copy(produced,kern.idx);
  require(produced()==5,"batch count");
  auto xa=Kokkos::create_mirror_view(kern.A.i1), xb=Kokkos::create_mirror_view(kern.B.i1);
  auto da=Kokkos::create_mirror_view(kern.A.dx1), db=Kokkos::create_mirror_view(kern.B.dx1);
  auto ya=Kokkos::create_mirror_view(kern.A.i2), yb=Kokkos::create_mirror_view(kern.B.i2);
  auto ia=Kokkos::create_mirror_view(kern.A.pld_i), ib=Kokkos::create_mirror_view(kern.B.pld_i);
  auto ra=Kokkos::create_mirror_view(kern.A.pld_r), rb=Kokkos::create_mirror_view(kern.B.pld_r);
  auto ta=Kokkos::create_mirror_view(kern.A.tag), tb=Kokkos::create_mirror_view(kern.B.tag);
  Kokkos::deep_copy(xa,kern.A.i1); Kokkos::deep_copy(xb,kern.B.i1);
  Kokkos::deep_copy(da,kern.A.dx1); Kokkos::deep_copy(db,kern.B.dx1);
  Kokkos::deep_copy(ya,kern.A.i2); Kokkos::deep_copy(yb,kern.B.i2);
  Kokkos::deep_copy(ia,kern.A.pld_i); Kokkos::deep_copy(ib,kern.B.pld_i);
  Kokkos::deep_copy(ra,kern.A.pld_r); Kokkos::deep_copy(rb,kern.B.pld_r);
  Kokkos::deep_copy(ta,kern.A.tag); Kokkos::deep_copy(tb,kern.B.tag);
  int per_segment[4]={0,0,0,0};
  for(int p=0;p<5;++p) {
    require(xa(p)>=0 and xa(p)<52 and da(p)>=0 and da(p)<1,"position in domain");
    ++per_segment[xa(p)/13];
    require(xa(p)==xb(p) and da(p)==db(p) and ya(p)==12 and yb(p)==12,"co-located injection");
    require(ta(p)==ntt::ParticleTag::alive and tb(p)==ntt::ParticleTag::alive,"paired live particles");
    require(ia(p,0)==0 and ib(p,0)==0 and ra(p,0)==0 and rb(p,0)==0,"payload reset");
  }
  require(per_segment[0]==0 and per_segment[1]==2 and
          per_segment[2]==0 and per_segment[3]==3,"actual batched segment occupancy");

  std::mt19937_64 rng(431);
  require(tristan_round(0,rng)==0, "zero count");
  require(tristan_round(10,rng)==10 and tristan_round(10.1,rng)==11, "ceiling branch");
  // Analytical Poisson mean/variance, neither generated by the implementation.
  double sum=0, sum2=0;
  constexpr int samples=100000;
  for(int i=0;i<samples;++i) {
    const double n=tristan_round(1.3,rng); sum+=n; sum2+=n*n;
  }
  const double mean=sum/samples, var=sum2/samples-mean*mean;
  require(std::abs(mean-1.3)<0.03 and std::abs(var-1.3)<0.05, "Poisson moments");
  std::cout << "segment witnesses pass; Poisson mean=" << mean << ", variance=" << var << '\n';
}

auto main(int argc, char** argv) -> int {
  Kokkos::initialize(argc,argv);
  int status=0;
  try { check_segments(); }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; status=1; }
  Kokkos::finalize();
  return status;
}
