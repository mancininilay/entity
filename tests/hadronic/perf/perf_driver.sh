#!/usr/bin/env bash
# P2 step 7, scaling harness (contract 2026-10-30 §1 M1, M2, M5): runs bh_proto through entity.xc for every
# (mode, N_ion, variant, repeat), reads Entity's timers from the diagnostics blocks and the ledger from
# hadronic_ledger_1.txt, and writes one CSV row per run plus a medians CSV. Bash 3.2 compatible; fails when no
# run was executed or when a ledger identity does not close.
#
# Usage: perf_driver.sh <entity.xc> <bh_check> <inputs dir (tests/hadronic/perf/inputs)> <workdir>
# Environment (defaults in brackets):
#   PERF_N        space-separated ion counts            [262144 1048576 4194304 16777216]
#   PERF_MODES    subset of "none poisson budget event" [all four]
#   PERF_VARIANTS subset of "warm cold"                 [both]   warm: Gamma_p = 3; cold: Gamma_p = 2 (below the table)
#   PERF_REPEATS  repeats per configuration             [3]
#   PERF_REPEATS_TOP repeats at the LARGEST N of PERF_N  [PERF_REPEATS] (the F3 budget cut: 2 at 2^24 only)
#   PERF_QK_SOURCE  qk_source of the engine runs         [born] (the production default; "fit" is the proposal of
#                                                        round 2 §F3 that removes the ~7 s Born-table startup of every
#                                                        process without changing the stage's work: Nilay decides)
#   PERF_WARMUP   warm-up steps (multiple of PERF_DIAG) [100]
#   PERF_MEASURED measured steps (multiple of PERF_DIAG)[500]
#   PERF_DIAG     diagnostics interval in steps         [1]   (round 2, Codex 1 F4: every measured step is a sample;
#                                                            the per-step timers are SUMMED over all PERF_MEASURED steps)
#   PERF_M5       1: also the M5 reference runs (deposit on): hadronic off; poisson a = 1 (pair-loaded); budget [1]
#                 Round 2 (Codex 1 F8): the pair-loaded runs (poisson, budget) write a checkpoint at their last step and
#                 a RESTART run with hadronic DISABLED (the checkpoint's own setup block edited: hadronic.enabled = false)
#                 and deposit on measures the stock pusher + deposit on the exact end-loaded population (weights,
#                 phase space and payloads from the checkpoint): rows m5poisson_pic and m5budget_pic. The pair-loaded runs
#                 themselves (hadronic on, growing population) stay as diagnostics. A zero-pair budget population is
#                 reported as it is (t_latency = 0 in the perf budget input).
#   PERF_LABEL    free label written in every row (e.g. "mac-openmp-smoke" or "a100")             [unlabelled]
# The ledger row (and the per-ion dump of bh_proto's post-step, an O(N) host pass with text output) is written
# once, at the last step (ledger.every = total steps): the "Custom" timer then holds the stage alone, not the
# diagnostics output; the startup capacity estimate is set to warn (the actual capacity policy stays error).
# Timer semantics (fact, engine.hpp: the timers are reset after every step and printed at steps that are multiples
# of PERF_DIAG): every printed block holds the timings of ONE step, the step at which it is printed. With the
# default PERF_DIAG = 1 every step is printed, the first PERF_WARMUP blocks and the final one (finish(): the ledger
# row and the per-ion dump) are discarded and the remaining PERF_MEASURED blocks are SUMMED, so the per-step time is
# the exact mean over the measured window (no phase sampling: Codex 1's F4). The driver REFUSES a log whose block
# count is not exactly PERF_WARMUP + PERF_MEASURED + 1 or whose values are not finite positive numbers (a run with
# missing or truncated timer output is a failed run, not a row of nan). "Custom" is the hadronic stage at the
# production placement (srpic.hpp) plus the generator's CustomPostStep (a measured proxy for the stage, the
# post-step being the ledger accumulation only, the dump and the row being in the excluded final block);
# "ParticlePusher" and "CurrentDeposit" are the stock pusher and deposit. "Timestep duration" is NOT a per-step
# value: progressbar.h averages a history of up to 1000 steps, so the driver reports it as progress_avg_ms (the
# printed running average at the last measured block, carrying the warm-up history) and never as the step cost;
# the whole-step cost is the sum of the named timers in the block ("Total"), reported as total_ms_per_step.
# Times in the CSV are milliseconds per step and ns per ion-step.
source "$(dirname "$0")/../common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
[ -n "$WD" ] || { echo "usage: perf_driver.sh <entity.xc> <bh_check> <inputs> <workdir>"; exit 2; }
PERF_N="${PERF_N:-262144 1048576 4194304 16777216}"
PERF_MODES="${PERF_MODES:-none poisson budget event}"
PERF_VARIANTS="${PERF_VARIANTS:-warm cold}"
PERF_REPEATS="${PERF_REPEATS:-3}"
PERF_REPEATS_TOP="${PERF_REPEATS_TOP:-$PERF_REPEATS}"
PERF_QK_SOURCE="${PERF_QK_SOURCE:-born}"
case "$PERF_QK_SOURCE" in born|fit) ;; *) echo "FAIL: PERF_QK_SOURCE must be born or fit"; exit 1 ;; esac
N_TOP=$(for n in $PERF_N; do echo $n; done | sort -n | tail -1)
PERF_WARMUP="${PERF_WARMUP:-100}"
PERF_MEASURED="${PERF_MEASURED:-500}"
PERF_DIAG="${PERF_DIAG:-1}"
PERF_M5="${PERF_M5:-1}"
PERF_LABEL="${PERF_LABEL:-unlabelled}"
DT=0.5
MU_R=1836.15
mkdir -p "$WD"
CSV="$WD/perf_runs.csv"
MED="$WD/perf_medians.csv"
echo "label,tag,mode,variant,N,repeat,samples,custom_ms_per_step,custom_ns_per_ion_step,pusher_ms_per_step,deposit_ms_per_step,total_ms_per_step,progress_avg_ms,npart_ion_end,npart_pairs_end,ledger_ok,timers_ok,exit" > "$CSV"
n_run=0; n_fail=0

# one extra diagnostics block at the end: the last step runs bh_proto's finish() (final summary, array identity, the
# per-ion dump: O(N) host work) inside the Custom timer, so the final block is excluded from the measured mean
steps_total=$((PERF_WARMUP + PERF_MEASURED + PERF_DIAG))
if [ $((PERF_WARMUP % PERF_DIAG)) -ne 0 ] || [ $((PERF_MEASURED % PERF_DIAG)) -ne 0 ]; then
  echo "FAIL: PERF_WARMUP and PERF_MEASURED must be multiples of PERF_DIAG"; exit 1
fi
blocks_skip=$((PERF_WARMUP / PERF_DIAG))
runtime=$(awk -v s="$steps_total" -v dt="$DT" 'BEGIN { printf "%.1f", s * dt }')

# timer_per_step <log> <name> <blocks to skip> <measured steps>: the mean per-step time in ms of a timer over the
# measured window: the blocks after the first <skip> and before the last one are summed and divided by their number,
# which MUST equal <measured steps>; prints "nan" when the count differs or a value is not a finite positive number.
# Entity's diagnostics table joins the name, a dot leader and the value ("Custom.........177.58 ms"); runs of two
# or more dots are turned into a space before parsing; units ns, us (printed as the micro sign), ms, s.
timer_per_step() {
  sed -e 's/µ/u/g' -e 's/\.\.\.*/ /g' "$1" | awk -v name="$2" -v skip="$3" -v steps="$4" '
    $1 == name && NF >= 3 {
      if ($2 ~ /^[0-9]+(\.[0-9]+)?$/ && ($3 == "ns" || $3 == "us" || $3 == "ms" || $3 == "s")) {
        v = $2 + 0; u = $3;
        if (u == "ns") v = v / 1.0e6; else if (u == "us") v = v / 1000.0; else if (u == "s") v = v * 1000.0;
        n++; val[n] = v
      } else { bad++ }
    }
    END {
      if (bad > 0 || n != skip + steps + 1) { printf "nan"; exit }
      sum = 0; for (i = skip + 1; i <= skip + steps; i++) { if (!(val[i] >= 0)) { printf "nan"; exit } sum += val[i] }
      printf "%.6f", sum / steps }'
}
# count_blocks <log>: the number of diagnostics blocks (lines starting with "Timestep duration")
count_blocks() {
  grep -c "^Timestep duration" "$1"
}
# progress_avg_ms <log> <blocks to skip> <measured steps>: the printed "Timestep duration" of the LAST measured block
# (a running average over up to 1000 steps, progressbar.h: reported, never used as the step cost)
progress_avg_ms() {
  sed 's/µ/u/g' "$1" | awk -v skip="$2" -v steps="$3" '
    $1 == "Timestep" && $2 == "duration:" {
      v = 0
      for (i = 3; i <= NF; i++) {
        if ($i ~ /^[0-9]+s$/) v += substr($i, 1, length($i) - 1) * 1000.0
        else if ($i ~ /^[0-9]+ms$/) v += substr($i, 1, length($i) - 2) + 0
        else if ($i ~ /^[0-9]+us$/) v += substr($i, 1, length($i) - 2) / 1000.0
      }
      n++; if (n == skip + steps) last = v
    }
    END { if (n >= skip + steps) printf "%.6f", last; else printf "nan" }'
}
# species_count <log> <label>: the particle count of a species in the last diagnostics block (Entity prints it
# in %.2e: approximate; the exact pair counts come from the ledger row, see pairs_from_ledger)
species_count() {
  sed -e 's/µ/u/g' -e 's/\.\.\.*/ /g' "$1" | awk -v lab="($2)" '
    $1 == "species" && $3 == lab { last = $4 }
    END { if (last == "") print 0; else printf "%.0f", last + 0 }'
}
# pairs_from_ledger <ledger file>: npart_minus + npart_plus of the last row (columns 42 and 43, 1-based)
pairs_from_ledger() {
  grep -v '^#' "$1" 2>/dev/null | tail -1 | awk '{ if (NF >= 43) printf "%.0f", $42 + $43; else print 0 }'
}

make_input() {
  # make_input <name> <mode> <N> <K> <deposit> <hadronic> [ckpt interval] -> $WD/<name>/input.toml
  local name="$1" mode="$2" N="$3" K="$4" dep="$5" had="$6" ckpt="${7:-100000000}"
  local dir="$WD/$name"; rm -rf "$dir"; mkdir -p "$dir"
  local maxion maxpair modeblock
  maxion=$(awk -v n="$N" 'BEGIN { printf "%.1f", n * 1.05 + 64 }')
  maxpair=$(awk -v n="$N" 'BEGIN { printf "%.1f", n * 1.5 + 1024 }')
  modeblock="$INP/mode_$mode.toml"
  awk -v name="$name" -v runtime="$runtime" -v nion="$N" -v kion="$K" -v maxion="$maxion" -v maxpair="$maxpair" -v ckpt="$ckpt" \
      -v dep="$dep" -v had="$had" -v diag="$PERF_DIAG" -v ledger="$steps_total" -v gtyp="3.0" -v modefile="$modeblock" -v qk="$PERF_QK_SOURCE" '
    { line = $0
      gsub(/@NAME@/, name, line); gsub(/@RUNTIME@/, runtime, line); gsub(/@NION@/, nion, line); gsub(/@KION@/, kion, line)
      gsub(/@MAXNPART_ION@/, maxion, line); gsub(/@MAXNPART_PAIR@/, maxpair, line); gsub(/@DEPOSIT@/, dep, line)
      gsub(/@HADRONIC@/, had, line); gsub(/@DIAG@/, diag, line); gsub(/@LEDGER_EVERY@/, ledger, line); gsub(/@GAMMA_TYP@/, gtyp, line)
      gsub(/@QK_SOURCE@/, qk, line); gsub(/@CKPT_INTERVAL@/, ckpt, line)
      if (line ~ /@MODE_BLOCK@/) { while ((getline ml < modefile) > 0) print ml; close(modefile) } else { print line } }' \
      "$INP/perf_template.toml" > "$dir/input.toml"
}

run_one() {
  # run_one <mode> <variant> <N> <repeat> <deposit> <hadronic> <tag> [ckpt]: with "ckpt" the run writes a checkpoint at its
  # last step (checkpoint.interval = total steps), for the M5 restart of F8
  local mode="$1" var="$2" N="$3" rep="$4" dep="$5" had="$6" tag="$7" ckpt="${8:-}"
  local K name rc ok custom pusher deposit total nion npm npp ns
  if [ "$var" = "cold" ]; then K=$(awk -v m="$MU_R" 'BEGIN { printf "%.6f", m * (2.0 - 1.0) }'); else K=$(awk -v m="$MU_R" 'BEGIN { printf "%.6f", m * (3.0 - 1.0) }'); fi
  name="${tag}_${mode}_${var}_N${N}_r${rep}"
  # the checkpoint is written at the last MEASURED step (the final step runs finish() and the engine does not reach a
  # checkpoint at the step count itself); the restart then continues from that population
  if [ "$ckpt" = "ckpt" ]; then make_input "$name" "$mode" "$N" "$K" "$dep" "$had" "$((PERF_WARMUP + PERF_MEASURED))"; else make_input "$name" "$mode" "$N" "$K" "$dep" "$had"; fi
  ( cd "$WD/$name" && "$EXE" -input input.toml > run.log 2>&1 ); rc=$?
  n_run=$((n_run + 1))
  custom=$(timer_per_step "$WD/$name/run.log" "Custom" "$blocks_skip" "$PERF_MEASURED")
  pusher=$(timer_per_step "$WD/$name/run.log" "ParticlePusher" "$blocks_skip" "$PERF_MEASURED")
  deposit=$(timer_per_step "$WD/$name/run.log" "CurrentDeposit" "$blocks_skip" "$PERF_MEASURED")
  total=$(timer_per_step "$WD/$name/run.log" "Total" "$blocks_skip" "$PERF_MEASURED")
  progress=$(progress_avg_ms "$WD/$name/run.log" "$blocks_skip" "$PERF_MEASURED")
  nblocks=$(count_blocks "$WD/$name/run.log")
  timers_ok=1
  if [ "$nblocks" -ne $((blocks_skip + PERF_MEASURED + 1)) ] || [ "$custom" = "nan" ] || [ "$pusher" = "nan" ] || [ "$total" = "nan" ]; then
    timers_ok=0
  fi
  nion=$(species_count "$WD/$name/run.log" "ion")
  if [ "$had" = "true" ]; then npm=$(pairs_from_ledger "$WD/$name/hadronic_ledger_1.txt"); npp=0
  else npm=$(species_count "$WD/$name/run.log" "e-"); npp=$(species_count "$WD/$name/run.log" "e+"); fi
  ok=1
  if [ $rc -ne 0 ]; then ok=0; echo "FAIL: $name: entity.xc exit $rc"; n_fail=$((n_fail + 1)); fi
  if [ $rc -eq 0 ] && [ $timers_ok -ne 1 ]; then
    ok=0; echo "FAIL: $name: timer output incomplete or non-finite ($nblocks diagnostics blocks, expected $((blocks_skip + PERF_MEASURED + 1)); Custom $custom, pusher $pusher, total $total)"; n_fail=$((n_fail + 1))
  fi
  if [ "$N" -le 0 ] || [ "$PERF_MEASURED" -le 0 ]; then ok=0; echo "FAIL: $name: N and PERF_MEASURED must be positive"; n_fail=$((n_fail + 1)); fi
  ledger_ok=1
  if [ "$had" = "true" ] && [ $rc -eq 0 ]; then
    if [ "$mode" = "event" ]; then
      "$CHECK" ledger --file "$WD/$name/hadronic_ledger_1.txt" --expect_identity 1 > "$WD/$name/ledger_check.log" 2>&1 || ledger_ok=0
    else
      "$CHECK" ledger --file "$WD/$name/hadronic_ledger_1.txt" > "$WD/$name/ledger_check.log" 2>&1 || ledger_ok=0
    fi
    if [ $ledger_ok -eq 0 ]; then ok=0; echo "FAIL: $name: ledger identity (see ledger_check.log)"; n_fail=$((n_fail + 1)); fi
  fi
  ns=$(awk -v c="$custom" -v n="$N" 'BEGIN { if (c == "nan" || n == 0) print "nan"; else printf "%.4f", c * 1.0e6 / n }')
  echo "$PERF_LABEL,$tag,$mode,$var,$N,$rep,$PERF_MEASURED,$custom,$ns,$pusher,$deposit,$total,$progress,$nion,$((npm + npp)),$ok,$timers_ok,$rc" >> "$CSV"
  echo "run $name: Custom $custom ms/step ($ns ns per ion-step), pusher $pusher, deposit $deposit, total $total ms (progress avg $progress ms), pairs $((npm + npp)), ok $ok, timers_ok $timers_ok, exit $rc"
}

# run_restart_pic <source run name> <tag> <mode> <N> <rep>: the M5 reference on the end-loaded population (F8). The source
# run (deposit on, hadronic on) wrote <name>.ckpt/ at its last step; its checkpoint directory is copied, the setup block of
# the checkpoint's meta file (which Entity's resume takes as the immutable setup: simulation.cpp) gets hadronic.enabled =
# false, and entity.xc -restart continues for steps_total more steps with the stock pusher + deposit only. The timers
# are read exactly as for a fresh run (warm-up discarded, measured steps summed). The population (ions + pairs with their
# weights, phase space and payloads) is the source run's end state: the species counts of the last block are reported.
run_restart_pic() {
  local src="$1" tag="$2" mode="$3" N="$4" rep="$5"
  local name="${tag}_${mode}_warm_N${N}_r${rep}" dir ckdir meta step rc custom pusher deposit total progress nblocks timers_ok ok nion npm npp
  dir="$WD/$name"; rm -rf "$dir"; mkdir -p "$dir"
  ckdir=$(ls -d "$WD/$src"/*.ckpt 2>/dev/null | head -1)
  n_run=$((n_run + 1))
  if [ -z "$ckdir" ]; then
    echo "FAIL: $name: no checkpoint directory under $WD/$src (the source run must write one at its last step)"; n_fail=$((n_fail + 1))
    echo "$PERF_LABEL,$tag,$mode,warm,$N,$rep,$PERF_MEASURED,nan,nan,nan,nan,nan,nan,0,0,0,0,3" >> "$CSV"; return
  fi
  cp -R "$ckdir" "$dir/restart.ckpt"
  meta=$(ls "$dir/restart.ckpt"/meta-*.toml 2>/dev/null | sort | tail -1)
  if [ -z "$meta" ]; then
    echo "FAIL: $name: no meta file in $ckdir"; n_fail=$((n_fail + 1))
    echo "$PERF_LABEL,$tag,$mode,warm,$N,$rep,$PERF_MEASURED,nan,nan,nan,nan,nan,nan,0,0,0,0,3" >> "$CSV"; return
  fi
  # the setup block of the checkpoint's meta is what the resumed run takes: hadronic off there
  sed -e 's/^\( *hadronic\.enabled *= *\)true/\1false/' "$meta" > "$meta.tmp" && mv "$meta.tmp" "$meta"
  grep -q 'hadronic\.enabled *= *false' "$meta" || { echo "FAIL: $name: could not disable hadronic in $meta"; n_fail=$((n_fail + 1)); return; }
  step=$(ls "$dir/restart.ckpt" | sed -n 's/^step-\([0-9]*\)\.bp$/\1/p' | sort -n | tail -1)
  # the restart input: the source run's input with the runtime extended by steps_total steps and the read path set
  # the restart input: the source run's input with hadronic disabled (the generator re-reads its [setup] from the input
  # file as well), the runtime extended by steps_total steps, the read path set, no further checkpoint
  awk -v rt="$(awk -v s="$steps_total" -v st="$step" -v dt="$DT" 'BEGIN { printf "%.1f", (s + st) * dt }')" -v rd="$dir/restart.ckpt" '
    /^ *runtime *=/ { print "  runtime = " rt; next }
    /^ *hadronic\.enabled *=/ { print "  hadronic.enabled    = false"; next }
    /^\[checkpoint\]/ { print; print "  read_path = \"" rd "\""; print "  write_path = \"" rd "-cont\""; print "  interval = 100000000"; skip = 1; next }
    skip == 1 && /^ *interval *=/ { skip = 0; next }
    { print }' "$WD/$src/input.toml" > "$dir/input.toml"
  ( cd "$dir" && "$EXE" -input input.toml -restart > run.log 2>&1 ); rc=$?
  custom=$(timer_per_step "$dir/run.log" "Custom" "$blocks_skip" "$PERF_MEASURED")
  pusher=$(timer_per_step "$dir/run.log" "ParticlePusher" "$blocks_skip" "$PERF_MEASURED")
  deposit=$(timer_per_step "$dir/run.log" "CurrentDeposit" "$blocks_skip" "$PERF_MEASURED")
  total=$(timer_per_step "$dir/run.log" "Total" "$blocks_skip" "$PERF_MEASURED")
  progress=$(progress_avg_ms "$dir/run.log" "$blocks_skip" "$PERF_MEASURED")
  nblocks=$(count_blocks "$dir/run.log")
  timers_ok=1; ok=1
  # a resumed run prints its blocks from start_step: the same count as a fresh run of steps_total steps
  if [ "$nblocks" -ne $((blocks_skip + PERF_MEASURED + 1)) ] || [ "$pusher" = "nan" ] || [ "$total" = "nan" ]; then timers_ok=0; fi
  nion=$(species_count "$dir/run.log" "ion"); npm=$(species_count "$dir/run.log" "e-"); npp=$(species_count "$dir/run.log" "e+")
  if [ $rc -ne 0 ]; then ok=0; echo "FAIL: $name: entity.xc -restart exit $rc"; n_fail=$((n_fail + 1)); fi
  if [ $rc -eq 0 ] && [ $timers_ok -ne 1 ]; then ok=0; echo "FAIL: $name: timer output incomplete ($nblocks blocks, expected $((blocks_skip + PERF_MEASURED + 1)))"; n_fail=$((n_fail + 1)); fi
  if [ $rc -eq 0 ] && grep -q "hadronic.enabled *= *true\|bh_proto: table:" "$dir/run.log"; then ok=0; echo "FAIL: $name: the restarted run still has the hadronic stage enabled"; n_fail=$((n_fail + 1)); fi
  echo "$PERF_LABEL,$tag,$mode,warm,$N,$rep,$PERF_MEASURED,$custom,nan,$pusher,$deposit,$total,$progress,$nion,$((npm + npp)),$ok,$timers_ok,$rc" >> "$CSV"
  echo "run $name (restart of $src at step $step, hadronic off): pusher $pusher ms/step, deposit $deposit, total $total ms, ions $nion, pairs $((npm + npp)), ok $ok, timers_ok $timers_ok, exit $rc"
}

for N in $PERF_N; do
  reps="$PERF_REPEATS"; [ "$N" = "$N_TOP" ] && reps="$PERF_REPEATS_TOP"
  for mode in $PERF_MODES; do
    for var in $PERF_VARIANTS; do
      rep=1
      while [ $rep -le "$reps" ]; do
        run_one "$mode" "$var" "$N" "$rep" false true m1
        rep=$((rep + 1))
      done
    done
  done
  if [ "$PERF_M5" = "1" ]; then
    rep=1
    while [ $rep -le "$reps" ]; do
      run_one none warm "$N" "$rep" true false m5pic             # stock pusher + deposit, hadronic off, ions only (the PIC floor)
      run_one poisson warm "$N" "$rep" true true m5poisson ckpt  # pair-loaded population (a = 1), deposit on, hadronic on: diagnostic
      run_one budget warm "$N" "$rep" true true m5budget ckpt    # the contract's budget run with deposit on (t_latency = 0): diagnostic
      run_restart_pic "m5poisson_poisson_warm_N${N}_r${rep}" m5poisson_pic poisson "$N" "$rep"  # F8: the END population, hadronic off
      run_restart_pic "m5budget_budget_warm_N${N}_r${rep}" m5budget_pic budget "$N" "$rep"
      rep=$((rep + 1))
    done
  fi
done

# medians over repeats per (label, mode, variant, N, tag) on custom_ms_per_step, pusher, deposit, total
echo "label,tag_mode_variant_N,runs,median_custom_ms_per_step,median_custom_ns_per_ion_step,median_pusher_ms_per_step,median_deposit_ms_per_step,median_total_ms_per_step,all_ok" > "$MED"
tail -n +2 "$CSV" | awk -F, '
  { key = $1 "," $2 "_" $3 "_" $4 "_" $5; if (!(key in seen)) { order[++k] = key; seen[key] = 1 }
    c[key] = c[key] " " $8; ns[key] = ns[key] " " $9; p[key] = p[key] " " $10; d[key] = d[key] " " $11; t[key] = t[key] " " $12
    n[key]++; if ($16 != 1 || $17 != 1) bad[key] = 1 }
  function median(s,   a, m, i, tmp, j) { m = split(s, a, " "); for (i = 2; i <= m; i++) { tmp = a[i]; j = i - 1; while (j >= 1 && a[j] + 0 > tmp + 0) { a[j+1] = a[j]; j-- } a[j+1] = tmp }
    if (m % 2 == 1) return a[(m + 1) / 2]; return (a[m/2] + a[m/2 + 1]) / 2 }
  END { for (i = 1; i <= k; i++) { key = order[i]; printf "%s,%d,%s,%s,%s,%s,%s,%d\n", key, n[key], median(c[key]), median(ns[key]), median(p[key]), median(d[key]), median(t[key]), (key in bad) ? 0 : 1 } }' >> "$MED"
echo "---- medians ($MED)"; cat "$MED"
if [ $n_run -eq 0 ]; then echo "FAIL: perf_driver.sh executed no run"; n_fail=$((n_fail + 1)); fi
echo "perf_driver.sh summary: $n_run runs executed, $n_fail failed checks, label $PERF_LABEL"
[ $n_fail -eq 0 ] && [ $n_run -gt 0 ]
