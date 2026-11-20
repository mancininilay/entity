# Reproductions on upstream 1.6.0rc (861b19a7)

None needs the Bethe-Heitler code. All were run on a clean detached
checkout of 861b19a7 (Apple clang 21, Kokkos 5.0.1 OpenMP, `output=OFF`,
double precision); the programs are compiled against the libraries of that
tree with its own flags (`build_repro.py <build tree> <source> <output>`).
Bounds checking is `-DKOKKOS_ENABLE_DEBUG_BOUNDS_CHECK` in `CMAKE_CXX_FLAGS`.

- `repro_sort_inactive_axis.cpp <1|2|3>`: `Particles::SortSpatially` on one
  live particle. `sort::PositionToTileIndex` reads `i2` and `i3` before the
  dispatch on the dimension. With bounds checking it aborts in 1D and 2D and
  completes in 3D, with and without `tiled_deposit` (`sort_up-o0-bounds*`);
  a plain build completes (`sort_up-o0-double_*`). The stock test
  `FRAMEWORK::particles_sort` (2D) aborts in the same way
  (`stock_particles_sort_up-o0-bounds-double.txt`).
- `repro_deposit_3d.cpp`: continuity of the deposit for one particle in 3D.
  With the zig-zag deposit the residual is a b c / 16 for a displacement
  (a, b, c) inside a cell (6.00009e-5, 7.50059e-6, 9.37500e-7 at dt = 0.2,
  0.1, 0.05 for a velocity (0.6, 0.5, 0.4)), and round-off when the particle
  moves in a plane. With the Esirkepov deposit of order 1 it is 1e-17
  (`deposit_3d_up-o0-double.txt`, `deposit_3d_up-o1-double.txt`).
- `emission_overflow_*.txt`: `tests/archetypes/repro_emission_overflow.cpp`
  on the same pristine tree (the unguarded insertion of the synchrotron
  policy), with and without bounds checking.
- `repro_inject_previous.cpp`: the contract of `kernel::InjectParticle` for
  charged particles. Two species of charge -1 and +1 in 2D; slot 0 of each
  holds the previous position of an earlier particle (two different points
  of one cell); one particle of each is inserted at the middle of the cell
  and the currents are deposited without a push. The neutral pair has not
  moved, yet the largest in-plane current is 0.6 and the largest |div J| is
  1.0; with the previous position set to the position both are exactly zero
  (`inject_previous_up-o0-double.txt`, `inject_previous_up-o0-bounds-double.txt`).
  Nothing upstream inserts charged particles this way today: this is the
  behaviour a policy that creates them has to know about.

Build trees: `up-o0-double` (default), `up-o1-double` (`deposit=esirkepov`,
`shape_order=1`), `up-o0-bounds-double` (bounds checking),
`up-o0-bounds-tiled-double` (bounds checking, `tiled_deposit=ON`).
