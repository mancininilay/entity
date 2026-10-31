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
#   PERF_DIAG     diagnostics interval in steps         [10]  (every PERF_DIAG-th step is a timing sample)
#   PERF_M5       1: also the M5 reference runs (deposit on): hadronic off; poisson a = 1 (pair-loaded); budget [1]
#   PERF_LABEL    free label written in every row (e.g. "mac-openmp-smoke" or "a100")             [unlabelled]
# The ledger row (and the per-ion dump of bh_proto's post-step, an O(N) host pass with text output) is written
# once, at the last step (ledger.every = total steps): the "Custom" timer then holds the stage alone, not the
# diagnostics output; the startup capacity estimate is set to warn (the actual capacity policy stays error).
# Timer semantics (fact, engine.hpp: the timers are reset after every step and printed at steps that are multiples
# of PERF_DIAG): every printed block holds the timings of ONE step, the step at which it is printed, so the blocks
# are per-step samples taken every PERF_DIAG steps. The first PERF_WARMUP/PERF_DIAG samples and the final one
# (finish()) are discarded; the mean of the remaining samples is the per-step time. "Custom" is the hadronic stage
# at the production placement (srpic.hpp), "ParticlePusher" and "CurrentDeposit" the stock pusher and deposit, and
# "Timestep duration" the whole step. Times in the CSV are milliseconds per step and ns per ion-step.
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
PERF_DIAG="${PERF_DIAG:-10}"
PERF_M5="${PERF_M5:-1}"
PERF_LABEL="${PERF_LABEL:-unlabelled}"
DT=0.5
MU_R=1836.15
mkdir -p "$WD"
CSV="$WD/perf_runs.csv"
MED="$WD/perf_medians.csv"
echo "label,tag,mode,variant,N,repeat,samples,custom_ms_per_step,custom_ns_per_ion_step,pusher_ms_per_step,deposit_ms_per_step,step_ms,npart_ion_end,npart_pairs_end,ledger_ok,exit" > "$CSV"
n_run=0; n_fail=0

# one extra diagnostics block at the end: the last step runs bh_proto's finish() (final summary, array identity, the
# per-ion dump: O(N) host work) inside the Custom timer, so the final block is excluded from the measured mean
steps_total=$((PERF_WARMUP + PERF_MEASURED + PERF_DIAG))
if [ $((PERF_WARMUP % PERF_DIAG)) -ne 0 ] || [ $((PERF_MEASURED % PERF_DIAG)) -ne 0 ]; then
  echo "FAIL: PERF_WARMUP and PERF_MEASURED must be multiples of PERF_DIAG"; exit 1
fi
blocks_skip=$((PERF_WARMUP / PERF_DIAG))
runtime=$(awk -v s="$steps_total" -v dt="$DT" 'BEGIN { printf "%.1f", s * dt }')

# timer_per_step <log> <name> <blocks to skip> <measured steps>: the mean per-step time in ms of a timer
# Entity's diagnostics table joins the name, a dot leader and the value ("Custom.........177.58 ms"); runs of two
# or more dots are turned into a space before parsing; units ns, us (printed as the micro sign), ms, s.
timer_per_step() {
  sed -e 's/µ/u/g' -e 's/\.\.\.*/ /g' "$1" | awk -v name="$2" -v skip="$3" -v steps="$4" '
    $1 == name && NF >= 3 {
      if ($2 ~ /^[0-9]+(\.[0-9]+)?$/ && ($3 == "ns" || $3 == "us" || $3 == "ms" || $3 == "s")) {
        v = $2 + 0; u = $3;
        if (u == "ns") v = v / 1.0e6; else if (u == "us") v = v / 1000.0; else if (u == "s") v = v * 1000.0;
        n++; if (n > skip) { sum += v; v_last = v }
      }
    }
    END { if (n > skip + 1) printf "%.6f", (sum - v_last) / (n - skip - 1); else printf "nan" }'
}
# step_ms <log> <blocks to skip>: the mean "Timestep duration" (ms) over the same samples
step_ms() {
  sed 's/µ/u/g' "$1" | awk -v skip="$2" '
    $1 == "Timestep" && $2 == "duration:" {
      v = 0
      for (i = 3; i <= NF; i++) {
        if ($i ~ /^[0-9]+s$/) v += substr($i, 1, length($i) - 1) * 1000.0
        else if ($i ~ /^[0-9]+ms$/) v += substr($i, 1, length($i) - 2) + 0
        else if ($i ~ /^[0-9]+us$/) v += substr($i, 1, length($i) - 2) / 1000.0
      }
      n++; if (n > skip) { sum += v; v_last = v }
    }
    END { if (n > skip + 1) printf "%.6f", (sum - v_last) / (n - skip - 1); else printf "nan" }'
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
  # make_input <name> <mode> <N> <K> <deposit> <hadronic> -> $WD/<name>/input.toml
  local name="$1" mode="$2" N="$3" K="$4" dep="$5" had="$6"
  local dir="$WD/$name"; rm -rf "$dir"; mkdir -p "$dir"
  local maxion maxpair modeblock
  maxion=$(awk -v n="$N" 'BEGIN { printf "%.1f", n * 1.05 + 64 }')
  maxpair=$(awk -v n="$N" 'BEGIN { printf "%.1f", n * 1.5 + 1024 }')
  modeblock="$INP/mode_$mode.toml"
  awk -v name="$name" -v runtime="$runtime" -v nion="$N" -v kion="$K" -v maxion="$maxion" -v maxpair="$maxpair" \
      -v dep="$dep" -v had="$had" -v diag="$PERF_DIAG" -v ledger="$steps_total" -v gtyp="3.0" -v modefile="$modeblock" -v qk="$PERF_QK_SOURCE" '
    { line = $0
      gsub(/@NAME@/, name, line); gsub(/@RUNTIME@/, runtime, line); gsub(/@NION@/, nion, line); gsub(/@KION@/, kion, line)
      gsub(/@MAXNPART_ION@/, maxion, line); gsub(/@MAXNPART_PAIR@/, maxpair, line); gsub(/@DEPOSIT@/, dep, line)
      gsub(/@HADRONIC@/, had, line); gsub(/@DIAG@/, diag, line); gsub(/@LEDGER_EVERY@/, ledger, line); gsub(/@GAMMA_TYP@/, gtyp, line)
      gsub(/@QK_SOURCE@/, qk, line)
      if (line ~ /@MODE_BLOCK@/) { while ((getline ml < modefile) > 0) print ml; close(modefile) } else { print line } }' \
      "$INP/perf_template.toml" > "$dir/input.toml"
}

run_one() {
  # run_one <mode> <variant> <N> <repeat> <deposit> <hadronic> <tag>
  local mode="$1" var="$2" N="$3" rep="$4" dep="$5" had="$6" tag="$7"
  local K name rc ok custom pusher deposit total nion npm npp ns
  if [ "$var" = "cold" ]; then K=$(awk -v m="$MU_R" 'BEGIN { printf "%.6f", m * (2.0 - 1.0) }'); else K=$(awk -v m="$MU_R" 'BEGIN { printf "%.6f", m * (3.0 - 1.0) }'); fi
  name="${tag}_${mode}_${var}_N${N}_r${rep}"
  make_input "$name" "$mode" "$N" "$K" "$dep" "$had"
  ( cd "$WD/$name" && "$EXE" -input input.toml > run.log 2>&1 ); rc=$?
  n_run=$((n_run + 1))
  custom=$(timer_per_step "$WD/$name/run.log" "Custom" "$blocks_skip" "$PERF_MEASURED")
  pusher=$(timer_per_step "$WD/$name/run.log" "ParticlePusher" "$blocks_skip" "$PERF_MEASURED")
  deposit=$(timer_per_step "$WD/$name/run.log" "CurrentDeposit" "$blocks_skip" "$PERF_MEASURED")
  total=$(step_ms "$WD/$name/run.log" "$blocks_skip")
  nion=$(species_count "$WD/$name/run.log" "ion")
  if [ "$had" = "true" ]; then npm=$(pairs_from_ledger "$WD/$name/hadronic_ledger_1.txt"); npp=0
  else npm=$(species_count "$WD/$name/run.log" "e-"); npp=$(species_count "$WD/$name/run.log" "e+"); fi
  ok=1
  if [ $rc -ne 0 ]; then ok=0; echo "FAIL: $name: entity.xc exit $rc"; n_fail=$((n_fail + 1)); fi
  if [ "$had" = "true" ] && [ $rc -eq 0 ]; then
    if [ "$mode" = "event" ]; then
      "$CHECK" ledger --file "$WD/$name/hadronic_ledger_1.txt" --expect_identity 1 > "$WD/$name/ledger_check.log" 2>&1 || ok=0
    else
      "$CHECK" ledger --file "$WD/$name/hadronic_ledger_1.txt" > "$WD/$name/ledger_check.log" 2>&1 || ok=0
    fi
    if [ $ok -eq 0 ]; then echo "FAIL: $name: ledger identity (see ledger_check.log)"; n_fail=$((n_fail + 1)); fi
  fi
  ns=$(awk -v c="$custom" -v n="$N" 'BEGIN { if (c == "nan" || n == 0) print "nan"; else printf "%.4f", c * 1.0e6 / n }')
  echo "$PERF_LABEL,$tag,$mode,$var,$N,$rep,$((PERF_MEASURED / PERF_DIAG)),$custom,$ns,$pusher,$deposit,$total,$nion,$((npm + npp)),$ok,$rc" >> "$CSV"
  echo "run $name: Custom $custom ms/step ($ns ns per ion-step), pusher $pusher, deposit $deposit, step $total ms, pairs $((npm + npp)), ledger_ok $ok, exit $rc"
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
      run_one none warm "$N" "$rep" true false m5pic        # stock pusher + deposit, hadronic off
      run_one poisson warm "$N" "$rep" true true m5poisson  # pair-loaded population (a = 1, physical pair count), deposit on
      run_one budget warm "$N" "$rep" true true m5budget    # the contract's budget run with deposit on (few packets with t_latency = 0)
      rep=$((rep + 1))
    done
  fi
done

# medians over repeats per (label, mode, variant, N, tag) on custom_ms_per_step, pusher, deposit, total
echo "label,tag_mode_variant_N,runs,median_custom_ms_per_step,median_custom_ns_per_ion_step,median_pusher_ms_per_step,median_deposit_ms_per_step,median_step_ms,ledger_all_ok" > "$MED"
tail -n +2 "$CSV" | awk -F, '
  { key = $1 "," $2 "_" $3 "_" $4 "_" $5; if (!(key in seen)) { order[++k] = key; seen[key] = 1 }
    c[key] = c[key] " " $8; ns[key] = ns[key] " " $9; p[key] = p[key] " " $10; d[key] = d[key] " " $11; t[key] = t[key] " " $12
    n[key]++; if ($15 != 1) bad[key] = 1 }
  function median(s,   a, m, i, tmp, j) { m = split(s, a, " "); for (i = 2; i <= m; i++) { tmp = a[i]; j = i - 1; while (j >= 1 && a[j] + 0 > tmp + 0) { a[j+1] = a[j]; j-- } a[j+1] = tmp }
    if (m % 2 == 1) return a[(m + 1) / 2]; return (a[m/2] + a[m/2 + 1]) / 2 }
  END { for (i = 1; i <= k; i++) { key = order[i]; printf "%s,%d,%s,%s,%s,%s,%s,%d\n", key, n[key], median(c[key]), median(ns[key]), median(p[key]), median(d[key]), median(t[key]), (key in bad) ? 0 : 1 } }' >> "$MED"
echo "---- medians ($MED)"; cat "$MED"
if [ $n_run -eq 0 ]; then echo "FAIL: perf_driver.sh executed no run"; n_fail=$((n_fail + 1)); fi
echo "perf_driver.sh summary: $n_run runs executed, $n_fail failed checks, label $PERF_LABEL"
[ $n_fail -eq 0 ] && [ $n_run -gt 0 ]
