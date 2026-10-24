# C++ candidate tables for the Bethe-Heitler rates step (P2 step 1)

These five files are **candidates for the independent comparison** of contract
section 5 item 8 (`bhref` versus the C++ startup tables, same bath, same
grid). They are **not fixtures**: no test in this repository reads them, and
nothing is validated against them. The frozen fixtures of the specification
(II.5, "frozen bhref output") will be added only once the comparison has been
performed and the step-1 checkpoint has passed.

Produced by `tools/bh_tables_dump` (built by the fork's CMake) from the source
at the commit recorded in each file's `# version:` header line; the physics is
`src/archetypes/hadronic/bh_physics.h`, the format
`src/archetypes/hadronic/bh_table_io.h`. Every quantity is computed in double
precision regardless of the build's `real_t`.

| file | bath | `physical_mass_ratio` | command |
| --- | --- | --- | --- |
| `mono05.txt` | mono, eps0 = 0.5 | 1836.15 | `bh_tables_dump --type mono --eps0 0.5 --mu_r 1836.15 --out mono05.txt` |
| `mono03.txt` | mono, eps0 = 0.3 | 1836.15 | `bh_tables_dump --type mono --eps0 0.3 --mu_r 1836.15 --out mono03.txt` |
| `powerlaw.txt` | powerlaw, index 2 on [0.1, 1.5] | 1836.15 | `bh_tables_dump --type powerlaw --eps_min 0.1 --eps_max 1.5 --index 2 --mu_r 1836.15 --out powerlaw.txt` |
| `blackbody.txt` | blackbody, theta = 0.1 on [0, 1.9] | 1836.15 | `bh_tables_dump --type blackbody --theta 0.1 --eps_min 0 --eps_max 1.9 --mu_r 1836.15 --out blackbody.txt` |
| `mono05_mu100.txt` | mono, eps0 = 0.5 | 100 | `bh_tables_dump --type mono --eps0 0.5 --mu_r 100 --out mono05_mu100.txt` |

File format (contract revision 1.4, section 2): UTF-8 text; header lines
`# key: value` with strict JSON values, one key per line in the order of the
schema, no blank lines anywhere; the data block starts at the first line not
beginning with `#` and runs to the end of the file; each data line is the seven
numbers `gamma_p beta_p y J I Dmean eps_mean` at `%.12e` separated by one
space and terminated by a newline, including the last. `y = log10(gamma_p -
gamma_th)`. The header key `data_sha256` is the SHA-256 of every byte of the
data block; `grep -v '^#' FILE | sha256sum` reproduces it. `status` is
`cpp_candidate`.

Grid convention (recorded in the header): nodes at `y_min + i/256` for
`i = 0..256`, then `y_min + 1 + j/64` for `j >= 1` while `y < y_max - 1e-9`
(`endpoint_coalescing`), then `y_max` appended with `gamma_p` set exactly to
`gamma_max`. `y_min` solves `I(y) = 1e-12 I_peak` (`lower_edge_fraction`);
`gamma_max` is the Lorentz factor at which `eps'_hi = gamma_p eps_max (1 +
beta)` equals `eps_prime_max = min(0.05 mu_r, 1000)`, or `gamma_p_max_input`
if given and smaller.

See `docs/hadronic/2026-10-24-step1-cpp-tables-report.md` for the acceptance
results, the measured numbers and the referee resolution log.
