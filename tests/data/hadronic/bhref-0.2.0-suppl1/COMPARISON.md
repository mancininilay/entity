# Exact closure comparison for the log-D supplement

Codex 1, 2026-10-27. This records the statistic used in the C++ round-2 closure, not a new tolerance. The supplement provides the reference observations needed for Claude 3 F8; it does not itself close the C++ implementation check.

Use the stored bin edges exactly. For each bath/Gamma pair, merge adjacent bins from left to right until **all four** accumulated counts reach 100: C++ unweighted events, bhref unweighted events, C++ weighted packets, bhref weighted packets. Start the next bin and repeat. Join an incomplete final tail to its previous bin. Sum counts, weights and squared weights during merging. Use this same common partition for both chain comparisons. If only one merged bin remains, report insufficient resolution, not a pass.

For each chain separately, let a_j and b_j be the two samples' merged sums of weights, and a2_j and b2_j their sums of squared weights. For unweighted events both equal the count. For weighted packets use w=1/D: **a2_j is sum(w_i^2), not (sum w_i)^2**, and not squared normalized bin mass. Define

    A = sum_j a_j; B = sum_j b_j
    p_j = a_j/A; q_j = b_j/B
    v_j = a2_j/A^2; u_j = b2_j/B^2
    h_j = (p_j+q_j)/2; H_j = cumulative_sum(h)_j
    delta_j = cumulative_sum(p-q)_j
    J = {j: 0 < H_j < 1-1e-12}
    T_AD = sum_{j in J} delta_j^2 h_j/[H_j(1-H_j)].

This is the right-edge discrete Anderson-Darling statistic; do not insert a sample-size factor or replace h_j with a uniform bin width. Calibrate its null conditionally on the observed bin masses and squared weights. In each replicate independently generate z_j ~ Normal(0,v_j) and y_j ~ Normal(0,u_j), independently across bins and samples. Center each sample's multiplier vector to account for normalization:

    zc_j = z_j - p_j sum_l z_l
    yc_j = y_j - q_j sum_l y_l
    delta*_j = cumulative_sum(zc-yc)_j
    T* = sum_{j in J} (delta*_j)^2 h_j/[H_j(1-H_j)].

Use **19,999 replicates** and p=(1+number of T* >= T_AD)/20,000. A comparison passes iff **p >= 0.01**. The empirical 99th percentile is a diagnostic (NumPy's default linear interpolation), not an additional pass rule. Report ESS_A=A^2/sum(a2), ESS_B=B^2/sum(b2), T_AD, p, the percentile, and merged-bin count. A universal KS cutoff based only on those two ESS values does not implement this calibration: local weight dispersion determines the covariance.

`comparison_reference.py` contains the **unchanged** `merge` and `calibrated` functions extracted from the closure script (original SHA-256 `6a1ab4de07350f7b096043ff3c517fb174e93a87efa1010642961cdff9e640eb`). Replay reproduces every returned field of the original mono05 Gamma=3 weighted-D comparison, including p=0.01805 and 150 merged bins; see `evidence/comparison-replay.json`. The additional KS quantities returned by that function were diagnostics; AD supplies the verdict.

Closure RNG: NumPy `default_rng(seed)` (PCG64), batches `[1000]*19+[999]`, drawing the full z array then the full y array within each batch. Seed = 2026102701 + 10*case + 5*chain + quantity. Cases were mono05 at {2.23125,3,10,100}, then powerlaw at {1.09375,3,10,100}; chain=0 unweighted or 1 weighted; D quantity=4. Thus for the new D comparisons:

| Bath | Gamma | Unweighted null seed | Weighted null seed |
|---|---:|---:|---:|
| mono05 | 10 | 2026102725 | 2026102730 |
| mono05 | 100 | 2026102735 | 2026102740 |
| powerlaw | 10 | 2026102765 | 2026102770 |
| powerlaw | 100 | 2026102775 | 2026102780 |

These are **null-calibration seeds**, distinct from the event seeds in the fixtures. C++ must document its Gaussian generator; the same integer seed in a different generator does not reproduce the same normal stream or p-value bit for bit. Matching the method requires the above covariance, centering, quadrature, count and rule; bitwise replay additionally requires the same random stream and operation order.

The closure applied the registered >=95% aggregate requirement across 80 chain/quantity/bath/Gamma comparisons and listed all failures. For the F8 continuation, substitute these eight log-D comparisons for the corresponding eight linear-D comparisons in that family; retain the other 72 and report every failure. Do not count both resolutions to enlarge the denominator. The original linear-D fixtures and the historical 80/80 result remain unchanged.
