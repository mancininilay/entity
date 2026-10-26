# Inputs prepared for the MPI runs (Rusty G3 job), not run in the cloud rounds

These inputs are accepted at parse time by the bh_proto generator (checked by
`HADRONIC::bh_proto_params` on the committed files) and are meant for
`mpirun -np <ranks> entity.xc -input <file>` on an MPI build
(`-D mpi=ON`). Each file states its expected outcome in its header comment.

- `g3_asymmetric_capacity_2ranks.toml`: C-MPI-2 negative test. Two domains in
  x, absorbing x walls, production placement, 16 ions per domain drifting at c
  along +x with the budget source. Rank 1 (downstream) receives rank 0's ions
  in addition to its own, so its pair capacity (maxnpart = 40 per species) is
  exhausted while rank 0's is not. Expected: with `on_capacity = error` the
  run aborts through the common point at the end of the stage; both ranks
  write the ledger row of the abort step (rank 0's file carries the reduced
  ledger), the log carries one `failure on rank 1: on_capacity ...` line and
  the abort message on every rank; the run terminates (no hang). Run under a
  timeout (`timeout 300 mpirun -np 2 ...`): a hang is the failure mode this
  test guards against.
- `t14b_absorb_production_4ranks.toml`: T14b-lite. Four domains in x,
  absorbing x walls, production placement (C-MPI-3 allows it; the post-step
  placement is refused at construction in an MPI build). Expected: the
  reduced ledger closes with the exported term (`bh_check texp --log` and
  `bh_check ledger --file hadronic_ledger_1.txt` on rank 0's output),
  `E_exported_host = E_exported` over the ranks, every absorbed ion harvested
  exactly once (C-MPI-1: the remnants of the transported ions carry no debt).
