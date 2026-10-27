# Round-2 scratch programs (not built by CMake, not part of ctest)

Development-time validation programs of round 2, kept for reproducibility of
the numbers quoted in the step-4 report (R2 stage 1). They use GCC's
`__float128` (libquadmath) as a 113-bit reference and are compiled by hand:

```
g++ -O2 -std=c++17 -I src docs/hadronic/r2-scratch/stage1_validate_against_float128.cpp -o v1 -lquadmath && ./v1
g++ -O2 -std=c++17 -I src docs/hadronic/r2-scratch/stage1_law3_L1_against_float128.cpp -o v2 -lquadmath && ./v2
```

The first maps law 1 (relative error) and law 3 (maximum pointwise error over
the angle divided by the peak-resolving angular mean of |h|) in double and long
double over (p+, p-); the second maps law 3's L1-relative error over the angle
(the angular-CDF metric) and its first-moment error. Both define
`std::log`, `std::sqrt`, `std::atanh`, `std::asinh`, `std::abs` for
`__float128` locally (a scratch-only device).
