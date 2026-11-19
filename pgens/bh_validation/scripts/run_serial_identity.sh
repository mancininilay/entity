#!/usr/bin/env bash
# V10, off is off: the same input (no species uses the policy; fields, deposit
# and a moving plasma on) run with a Serial build of the branch and of
# upstream; the BHV lines and the hash of the fields and particles must be
# identical.
#   run_serial_identity.sh <builds dir> <output root> <make_input.py>
set -uo pipefail
BUILDS="$1"; ROOT="$2"; MK="$3"
rc=0
for prec in single double; do
  for order in leptons_first ions_first; do
    for tree in branch upstream; do
      dir="$ROOT/${order}_${prec}/$tree"
      mkdir -p "$dir"
      python3 "$MK" "$dir/input.toml" name=v10 order=$order policy=off steps=200 diag_interval=20 dump_step=199 > "$dir/make_input.log"
      ( cd "$dir" && "$BUILDS/serial-$tree-$prec/src/entity.xc" -input input.toml > run.log 2>&1 )
      grep "^BHV" "$dir/run.log" > "$dir/bhv.txt"
    done
    a="$ROOT/${order}_${prec}/branch"; b="$ROOT/${order}_${prec}/upstream"
    if cmp -s "$a/bhv.txt" "$b/bhv.txt" && [ -s "$a/bhv.txt" ] && grep -q BHVHASH "$a/bhv.txt" \
       && cmp -s "$a/bhv_species_1_rank_0.bin" "$b/bhv_species_1_rank_0.bin" \
       && cmp -s "$a/bhv_species_2_rank_0.bin" "$b/bhv_species_2_rank_0.bin" \
       && cmp -s "$a/bhv_species_3_rank_0.bin" "$b/bhv_species_3_rank_0.bin"; then
      echo "PASS V10 $order $prec: $(wc -l < "$a/bhv.txt" | tr -d ' ') diagnostic lines, particle dumps and $(grep BHVHASH "$a/bhv.txt") identical to upstream"
    else
      echo "FAIL V10 $order $prec: branch and upstream differ"; rc=1
    fi
  done
done
exit $rc
