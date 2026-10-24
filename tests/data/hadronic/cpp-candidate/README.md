# C++ candidate tables for the Bethe-Heitler rates step (P2 step 1)

These five files are **candidates for the independent comparison** of contract
revision 1.3 section 5 item 8 (`bhref` versus the C++ startup tables, same
bath, same grid). They are **not fixtures**: no test in this repository reads
them, and nothing is validated against them. The frozen fixtures of the
specification (II.5, "frozen bhref output") will be added only once the
comparison has been performed and the step-1 checkpoint has passed.

Produced by `tools/bh_tables_dump` (built by the fork's CMake) from the source
at the commit recorded in each file's `# commit:` header line; the physics is
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

File format (contract section 4): UTF-8 text, header lines `# key: value`
with JSON values, then whitespace-separated columns
`gamma_p beta_p y J I Dmean eps_mean` at `%.12e`. `y = log10(gamma_p -
gamma_th)`. The header key `sha256_data` is the SHA-256 of the data block,
i.e. of every byte from the first data line to the final newline; it can be
checked with `grep -v '^#' FILE | sha256sum`. `status` is `cpp_candidate`.

Grid convention used here (the contract does not fix the remainder handling):
nodes at `y_min + i/256` for `i = 0..256`, then `y_min + 1 + j/64`, and the
upper edge `y_max` appended as the last node (the final interval is shorter
than 1/64 of a decade). `y_min` is the solution of `I(y) = 1e-12 I_peak` found
by bisection; `gamma_max` is the Lorentz factor at which `eps'_hi = Gamma
eps_max (1 + beta)` equals `min(0.05 mu_r, 1000)`.

See `docs/hadronic/2026-10-24-step1-cpp-tables-report.md` for the acceptance
results and the measured numbers.
