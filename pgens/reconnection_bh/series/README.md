# Step 1 of the resolved series: the resolution pair (prepared, not run)

Nothing here has run on a GPU and no job is authorised. A referee reads the branch before any run
uses it.

The pair: the reconnection setup of the shakedown at ten times its optical depth (0.14333), with
the corrected replenishing rule (`setup.injector = 1`), on four ranks, on the present grid
(4096 x 2048 cells) and at twice the resolution (8192 x 4096 cells, the time step halves with the
cell).

| File | What |
|---|---|
| `derive_inputs_s1.py` | derives every input here from `../r-low.toml` (or `../toy-low.toml`) by listed edits |
| `s1-n-a.toml`, `s1-n-b.toml`, `s1-n-c.toml` | the present grid, three realisations (`setup.seed` absent, 1, 2), 3000 time units |
| `s1-f-a.toml`, `s1-f-b.toml`, `s1-f-c.toml` | twice the resolution, three realisations, 3000 time units |
| `toy-s1.toml` | the same edits on the toy box of stage A (200 steps), to exercise every key together on a laptop |

Open, and not decided by these files:

1. **Realisations against length.** `derive_inputs_s1.py --runtime 9000` writes one input of 9000
   time units per grid instead of three of 3000.
2. **The decomposition.** The files carry `[2, 2]`; `--decomposition 4,1` writes `[4, 1]` and the
   capacities that go with it.
3. **The order of the species.** The files keep the order of the shakedown (emitter listed before
   the lepton species), for which the registered check scripts are written. With the lepton
   species listed first (`setup.primary_species = [3, 4]`) the charge check can see a missing
   write of the previous position of a new pair; the check scripts then need the species indices
   as inputs, which is not done.

Capacities per rank follow the rule and the numbers stated at the top of `derive_inputs_s1.py`.
The draft of the pre-registration is in the master folder of the project, not in this repository.
