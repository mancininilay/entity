#!/usr/bin/env bash
# P2 step 7, scaling harness (contract 2026-10-30 §1 M1, M2, M5): runs bh_proto through entity.xc for every
# (mode, N_ion, variant, repeat), reads Entity's timers from the diagnostics blocks and the ledger from
# hadronic_ledger_1.txt, and writes one CSV row per run plus a medians CSV. Bash 3.2 compatible.
# Exit status (Codex 1's closure C1, 2026-10-31): 0 every run valid; 1 a SCIENTIFIC failure only (an executed ledger
# identity check failed: a "FAIL: " line); 3 any INFRASTRUCTURE failure (an "INFRA: " line: entity.xc non-zero exit,
# missing or malformed timer output, a missing checkpoint or meta file, a restart that kept the stage enabled, bh_check
# unable to read a ledger, no run executed, an invalid configuration), which takes precedence over 1.
#
# Usage: perf_driver.sh <entity.xc> <bh_check> <inputs dir (tests/hadronic/perf/inputs)> <workdir>
# Environment (defaults in brackets):
#   PERF_N        space-separated ion counts            [262144 1048576 4194304 16777216]
#   PERF_MODES    subset of "none poisson budget event" [all four]
#   PERF_VARIANTS subset of "warm cold"                 [both]   warm: Gamma_p = 3; cold: Gamma_p = 2 (below the table)
#   PERF_REPEATS  repeats per configuration             [3]
#   PERF_REPEATS_TOP repeats at the LARGEST N of PERF_N  [PERF_REPEATS] (the F3 budget cut: 2 at 2^24 only)
#   PERF_QK_SOURCE  qk_source of the engine runs         [born] (the production default). The A100 job sets "fit"
#                                                        (Nilay's decision of 2026-10-31): no ~7 s Born-table startup
#                                                        per process; same code path and table layout, the dynamic work
#                                                        following the table values (equal cost expected, not identical
#                                                        by construction: Codex 1's C4)
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
#   PERF_CHECK_INPUTS_ONLY  1: generate the selector probe input, check it (below) and exit (0 or 3) without any run   [0]
#   PERF_REPEAT_FIRST, PERF_REPEAT_LAST  the repeat indices run at every N   [1 and the N's registered count]
#   PERF_M1       1: the M1/M2 runs (PERF_MODES x PERF_VARIANTS) at every N; 0: none (an M5-only part)   [1]
#   PERF_M5_SETS  subset of "pic poisson budget": the M5 groups run per repeat [all three]; "poisson" and "budget" are each
#                 a source run and its restart (the restart reads that run's checkpoint, so the pair is never split)
#   PERF_COMBINE  "<csv>:<csv>:...": no run; the medians of the concatenated perf_runs.csv files (a colon-separated list:
#                 paths may hold spaces) are written to <workdir>/perf_medians.csv, which must not exist
# Round 4 (budgets from B7398318): the campaign is split into one-hour parts (rusty/perf_budget.py --plan), each a slice of
# (N, repeats, modes, M5 groups) whose rows carry their true repeat index; PERF_COMBINE recomputes the medians over the
# parts with the rule below. No measured quantity changes: the same processes, steps and timers as one long run.
# The q_k selector (round 4, Codex 1's B1 of B7398318: the template had no @QK_SOURCE@, so all 28 generated inputs of that
# job omitted qk_source and the engine ran on the parser's default, Born, while fit had been requested). Three checks, each
# an INFRASTRUCTURE failure (a wrong effective input, never a scientific result): (1) before any run, a probe input is
# generated from the template and must hold exactly one line qk_source = "<PERF_QK_SOURCE>" (else the driver refuses the
# whole campaign, exit 3, nothing run); (2) every generated input is checked the same way before its run (else that run is
# not started); (3) every hadronic-on run's log must print exactly one startup line "bh_proto: qk_source = <PERF_QK_SOURCE>:"
# (pgens/bh_proto/pgen.hpp) and no other source. The effective source read from the log is the last CSV column qk_source
# ("n/a" for the hadronic-off runs, which build no table; "missing" when no startup line was found).
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
# Times in the CSV are milliseconds per step and ns per ion-step. npart_pairs_end counts the pair-species macroparticles
# (e- + e+, i.e. leptons; physical pairs are half of it): exact from the ledger row for hadronic-on runs, the %.2e-rounded
# diagnostics counts otherwise (the restart rows; npart_ion_end is always the rounded diagnostics count, e.g. 65500 for
# 65536 ions).
# Validity (Codex 1's closure C2, 2026-10-31): EVERY timing column of a row is gated in BOTH paths (fresh runs and
# the M5 restarts): Custom, pusher, deposit, Total and progress_avg_ms must each have exactly the measured window's
# samples, all finite (else the run is an infrastructure failure, timers_ok = 0); a deposit-enabled run must also have
# measured a positive deposit. In the CSV, "nan" means a missing or invalid value (the row is a failed run) and "n/a"
# means not applicable: custom_ns_per_ion_step of a restart row, whose hadronic stage is disabled, so its Custom timer
# (reported, gated) is the generator's post-step and not a stage cost per ion-step.
source "$(dirname "$0")/../common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
[ -n "$WD" ] || { echo "usage: perf_driver.sh <entity.xc> <bh_check> <inputs> <workdir>"; exit 2; }
PERF_N="${PERF_N:-262144 1048576 4194304 16777216}"
PERF_MODES="${PERF_MODES:-none poisson budget event}"
PERF_VARIANTS="${PERF_VARIANTS:-warm cold}"
PERF_REPEATS="${PERF_REPEATS:-3}"
PERF_REPEATS_TOP="${PERF_REPEATS_TOP:-$PERF_REPEATS}"
PERF_QK_SOURCE="${PERF_QK_SOURCE:-born}"
case "$PERF_QK_SOURCE" in born|fit) ;; *) echo "INFRA: PERF_QK_SOURCE must be born or fit"; exit 3 ;; esac
N_TOP=$(for n in $PERF_N; do echo $n; done | sort -n | tail -1)
PERF_WARMUP="${PERF_WARMUP:-100}"
PERF_MEASURED="${PERF_MEASURED:-500}"
PERF_DIAG="${PERF_DIAG:-1}"
PERF_M5="${PERF_M5:-1}"
PERF_LABEL="${PERF_LABEL:-unlabelled}"
PERF_REPEAT_FIRST="${PERF_REPEAT_FIRST:-1}"
PERF_M1="${PERF_M1:-1}"
PERF_M5_SETS="${PERF_M5_SETS:-pic poisson budget}"
for g in $PERF_M5_SETS; do case "$g" in pic|poisson|budget) ;; *) echo "INFRA: PERF_M5_SETS: unknown group '$g'"; exit 3 ;; esac; done
DT=0.5
MU_R=1836.15
mkdir -p "$WD"
CSV="$WD/perf_runs.csv"
MED="$WD/perf_medians.csv"
COMBINE_LATER=0; [ -n "${PERF_COMBINE:-}" ] && COMBINE_LATER=1
[ "$COMBINE_LATER" = "1" ] || echo "label,tag,mode,variant,N,repeat,samples,custom_ms_per_step,custom_ns_per_ion_step,pusher_ms_per_step,deposit_ms_per_step,total_ms_per_step,progress_avg_ms,npart_ion_end,npart_pairs_end,ledger_ok,timers_ok,exit,qk_source" > "$CSV"
n_run=0; n_fail=0; n_infra=0   # n_fail: scientific failures (FAIL lines); n_infra: infrastructure failures (INFRA lines)

# one extra diagnostics block at the end: the last step runs bh_proto's finish() (final summary, array identity, the
# per-ion dump: O(N) host work) inside the Custom timer, so the final block is excluded from the measured mean
steps_total=$((PERF_WARMUP + PERF_MEASURED + PERF_DIAG))
if [ $((PERF_WARMUP % PERF_DIAG)) -ne 0 ] || [ $((PERF_MEASURED % PERF_DIAG)) -ne 0 ]; then
  echo "INFRA: PERF_WARMUP and PERF_MEASURED must be multiples of PERF_DIAG"; exit 3
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
# deposit_measured <deposit enabled: true|false> <deposit ms per step>: true unless the deposit is enabled and its
# measured time is not a positive number (a deposit-enabled run whose CurrentDeposit is zero did not measure it)
deposit_measured() {
  [ "$1" = "true" ] || return 0
  awk -v d="$2" 'BEGIN { exit !(d ~ /^[0-9]+(\.[0-9]+)?$/ && d + 0 > 0) }'
}
# pairs_from_ledger <ledger file>: npart_minus + npart_plus of the last row (columns 42 and 43, 1-based)
pairs_from_ledger() {
  grep -v '^#' "$1" 2>/dev/null | tail -1 | awk '{ if (NF >= 43) printf "%.0f", $42 + $43; else print 0 }'
}
# selector_ok <input.toml>: 0 when the input holds exactly one qk_source line and its value is PERF_QK_SOURCE (B1)
selector_ok() {
  [ "$(grep -c '^ *qk_source *=' "$1")" = "1" ] && grep -Eq '^ *qk_source *= *"'"$PERF_QK_SOURCE"'" *$' "$1"
}
# runtime_qk_source <log>: the source of the startup line "bh_proto: qk_source = <src>: ..." (pgen.hpp); "missing" when
# there is none, "multiple:<a,b>" when there is more than one
runtime_qk_source() {
  sed -n 's/^bh_proto: qk_source = \([a-z]*\): .*/\1/p' "$1" 2>/dev/null | awk '
    { v[++n] = $1 } END { if (n == 0) print "missing"; else if (n == 1) print v[1]; else { s = v[1]; for (i = 2; i <= n; i++) s = s "," v[i]; print "multiple:" s } }'
}

# write_medians <out> <csv...>: medians over repeats per (label, mode, variant, N, tag) on custom_ms_per_step, pusher,
# deposit, total (the rows of every csv, header lines skipped); with MED_LABEL set, every row's label is replaced by it (the
# parts of one campaign carry their own job labels)
write_medians() {
  local out="$1"; shift
echo "label,tag_mode_variant_N,runs,median_custom_ms_per_step,median_custom_ns_per_ion_step,median_pusher_ms_per_step,median_deposit_ms_per_step,median_total_ms_per_step,all_ok" > "$out"
for c in "$@"; do tail -n +2 "$c"; done | awk -F, -v lab="${MED_LABEL:-}" '
  { key = (lab != "" ? lab : $1) "," $2 "_" $3 "_" $4 "_" $5; if (!(key in seen)) { order[++k] = key; seen[key] = 1 }
    c[key] = c[key] " " $8; ns[key] = ns[key] " " $9; p[key] = p[key] " " $10; d[key] = d[key] " " $11; t[key] = t[key] " " $12
    n[key]++; if ($16 != 1 || $17 != 1) bad[key] = 1 }
  function median(s,   a, m, i, tmp, j, na) { m = split(s, a, " "); na = 0; for (i = 1; i <= m; i++) { if (a[i] == "n/a") na++; if (a[i] == "nan") return "nan" }
    if (na == m) return "n/a"; if (na > 0) return "nan"
    for (i = 2; i <= m; i++) { tmp = a[i]; j = i - 1; while (j >= 1 && a[j] + 0 > tmp + 0) { a[j+1] = a[j]; j-- } a[j+1] = tmp }
    if (m % 2 == 1) return a[(m + 1) / 2]; return (a[m/2] + a[m/2 + 1]) / 2 }
  END { for (i = 1; i <= k; i++) { key = order[i]; printf "%s,%d,%s,%s,%s,%s,%s,%d\n", key, n[key], median(c[key]), median(ns[key]), median(p[key]), median(d[key]), median(t[key]), (key in bad) ? 0 : 1 } }' >> "$out"
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
  n_run=$((n_run + 1))
  # B1 (2): the generated input must carry the requested selector; otherwise the run is not started
  if ! selector_ok "$WD/$name/input.toml"; then
    echo "INFRA: $name: the generated input does not hold exactly one qk_source = \"$PERF_QK_SOURCE\" (run not started)"; n_infra=$((n_infra + 1))
    echo "$PERF_LABEL,$tag,$mode,$var,$N,$rep,$PERF_MEASURED,nan,nan,nan,nan,nan,nan,0,0,0,0,3,missing" >> "$CSV"; return
  fi
  ( cd "$WD/$name" && "$EXE" -input input.toml > run.log 2>&1 ); rc=$?
  custom=$(timer_per_step "$WD/$name/run.log" "Custom" "$blocks_skip" "$PERF_MEASURED")
  pusher=$(timer_per_step "$WD/$name/run.log" "ParticlePusher" "$blocks_skip" "$PERF_MEASURED")
  deposit=$(timer_per_step "$WD/$name/run.log" "CurrentDeposit" "$blocks_skip" "$PERF_MEASURED")
  total=$(timer_per_step "$WD/$name/run.log" "Total" "$blocks_skip" "$PERF_MEASURED")
  progress=$(progress_avg_ms "$WD/$name/run.log" "$blocks_skip" "$PERF_MEASURED")
  nblocks=$(count_blocks "$WD/$name/run.log")
  timers_ok=1
  if [ "$nblocks" -ne $((blocks_skip + PERF_MEASURED + 1)) ] || [ "$custom" = "nan" ] || [ "$pusher" = "nan" ] || [ "$deposit" = "nan" ] \
     || [ "$total" = "nan" ] || [ "$progress" = "nan" ] || ! deposit_measured "$dep" "$deposit"; then
    timers_ok=0
  fi
  nion=$(species_count "$WD/$name/run.log" "ion")
  if [ "$had" = "true" ]; then npm=$(pairs_from_ledger "$WD/$name/hadronic_ledger_1.txt"); npp=0
  else npm=$(species_count "$WD/$name/run.log" "e-"); npp=$(species_count "$WD/$name/run.log" "e+"); fi
  ok=1
  if [ $rc -ne 0 ]; then ok=0; echo "INFRA: $name: entity.xc exit $rc"; n_infra=$((n_infra + 1)); fi
  if [ $rc -eq 0 ] && [ $timers_ok -ne 1 ]; then
    ok=0; echo "INFRA: $name: timer output incomplete or non-finite ($nblocks diagnostics blocks, expected $((blocks_skip + PERF_MEASURED + 1)); Custom $custom, pusher $pusher, deposit $deposit (deposit on: $dep), total $total, progress $progress)"; n_infra=$((n_infra + 1))
  fi
  if [ "$N" -le 0 ] || [ "$PERF_MEASURED" -le 0 ]; then ok=0; echo "INFRA: $name: N and PERF_MEASURED must be positive"; n_infra=$((n_infra + 1)); fi
  # B1 (3): the source the engine actually used, from its startup line (a hadronic-off run builds no table)
  qk_eff="n/a"
  if [ "$had" = "true" ]; then
    qk_eff=$(runtime_qk_source "$WD/$name/run.log")
    if [ $rc -eq 0 ] && [ "$qk_eff" != "$PERF_QK_SOURCE" ]; then
      ok=0; echo "INFRA: $name: the engine ran with qk_source '$qk_eff' (startup line), requested '$PERF_QK_SOURCE'"; n_infra=$((n_infra + 1))
    fi
  fi
  ledger_ok=1
  if [ "$had" = "true" ] && [ $rc -eq 0 ]; then
    lrc=0
    if [ "$mode" = "event" ]; then
      "$CHECK" ledger --file "$WD/$name/hadronic_ledger_1.txt" --expect_identity 1 > "$WD/$name/ledger_check.log" 2>&1 || lrc=$?
    else
      "$CHECK" ledger --file "$WD/$name/hadronic_ledger_1.txt" > "$WD/$name/ledger_check.log" 2>&1 || lrc=$?
    fi
    # bh_check: 1 = an executed check failed (scientific), 2 = it could not read the ledger (infrastructure)
    if [ $lrc -eq 1 ]; then ledger_ok=0; ok=0; echo "FAIL: $name: ledger identity (see ledger_check.log)"; n_fail=$((n_fail + 1))
    elif [ $lrc -ne 0 ]; then ledger_ok=0; ok=0; echo "INFRA: $name: bh_check could not check the ledger (exit $lrc, see ledger_check.log)"; n_infra=$((n_infra + 1)); fi
  fi
  ns=$(awk -v c="$custom" -v n="$N" 'BEGIN { if (c == "nan" || n == 0) print "nan"; else printf "%.4f", c * 1.0e6 / n }')
  echo "$PERF_LABEL,$tag,$mode,$var,$N,$rep,$PERF_MEASURED,$custom,$ns,$pusher,$deposit,$total,$progress,$nion,$((npm + npp)),$ok,$timers_ok,$rc,$qk_eff" >> "$CSV"
  echo "run $name: Custom $custom ms/step ($ns ns per ion-step), pusher $pusher, deposit $deposit, total $total ms (progress avg $progress ms), leptons (e- + e+) $((npm + npp)), qk_source $qk_eff, ok $ok, timers_ok $timers_ok, exit $rc"
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
  # the source run's checkpoint by its exact name (<simulation name>.ckpt): a glob would also match stray copies such as
  # "<name> 2.ckpt" (found on the Mac smoke: empty duplicates created by the iCloud-synchronised folder)
  ckdir="$WD/$src/$src.ckpt"
  n_run=$((n_run + 1))
  if [ ! -d "$ckdir" ]; then
    echo "INFRA: $name: no checkpoint directory under $WD/$src (the source run must write one at its last step)"; n_infra=$((n_infra + 1))
    echo "$PERF_LABEL,$tag,$mode,warm,$N,$rep,$PERF_MEASURED,nan,nan,nan,nan,nan,nan,0,0,0,0,3,n/a" >> "$CSV"; return
  fi
  cp -R "$ckdir" "$dir/restart.ckpt"
  meta=$(ls "$dir/restart.ckpt"/meta-*.toml 2>/dev/null | sort | tail -1)
  if [ -z "$meta" ]; then
    echo "INFRA: $name: no meta file in $ckdir"; n_infra=$((n_infra + 1))
    echo "$PERF_LABEL,$tag,$mode,warm,$N,$rep,$PERF_MEASURED,nan,nan,nan,nan,nan,nan,0,0,0,0,3,n/a" >> "$CSV"; return
  fi
  # the setup block of the checkpoint's meta is what the resumed run takes: hadronic off there
  sed -e 's/^\( *hadronic\.enabled *= *\)true/\1false/' "$meta" > "$meta.tmp" && mv "$meta.tmp" "$meta"
  grep -q 'hadronic\.enabled *= *false' "$meta" || { echo "INFRA: $name: could not disable hadronic in $meta"; n_infra=$((n_infra + 1)); return; }
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
  # a resumed run prints its blocks from start_step: the same count as a fresh run of steps_total steps; every timing
  # column is gated as for a fresh run (C2), the deposit (on in every restart) must be measured
  if [ "$nblocks" -ne $((blocks_skip + PERF_MEASURED + 1)) ] || [ "$custom" = "nan" ] || [ "$pusher" = "nan" ] || [ "$deposit" = "nan" ] \
     || [ "$total" = "nan" ] || [ "$progress" = "nan" ] || ! deposit_measured true "$deposit"; then timers_ok=0; fi
  nion=$(species_count "$dir/run.log" "ion"); npm=$(species_count "$dir/run.log" "e-"); npp=$(species_count "$dir/run.log" "e+")
  if [ $rc -ne 0 ]; then ok=0; echo "INFRA: $name: entity.xc -restart exit $rc"; n_infra=$((n_infra + 1)); fi
  if [ $rc -eq 0 ] && [ $timers_ok -ne 1 ]; then ok=0; echo "INFRA: $name: timer output incomplete or non-finite ($nblocks blocks, expected $((blocks_skip + PERF_MEASURED + 1)); Custom $custom, pusher $pusher, deposit $deposit, total $total, progress $progress)"; n_infra=$((n_infra + 1)); fi
  if [ $rc -eq 0 ] && grep -q "hadronic.enabled *= *true\|bh_proto: table:" "$dir/run.log"; then ok=0; echo "INFRA: $name: the restarted run still has the hadronic stage enabled"; n_infra=$((n_infra + 1)); fi
  echo "$PERF_LABEL,$tag,$mode,warm,$N,$rep,$PERF_MEASURED,$custom,n/a,$pusher,$deposit,$total,$progress,$nion,$((npm + npp)),$ok,$timers_ok,$rc,n/a" >> "$CSV"
  echo "run $name (restart of $src at step $step, hadronic off): pusher $pusher ms/step, deposit $deposit, total $total ms, ions $nion, leptons (e- + e+) $((npm + npp)) [%.2e-rounded diagnostics], ok $ok, timers_ok $timers_ok, exit $rc"
}

# PERF_COMBINE: the medians over the parts' CSVs, written to a NEW file (an existing medians file is never overwritten)
if [ "$COMBINE_LATER" = "1" ]; then
  [ -e "$MED" ] && { echo "INFRA: PERF_COMBINE: $MED exists (no overwrite)"; exit 3; }
  IFS=: read -r -a COMBINE_FILES <<< "$PERF_COMBINE"
  for c in "${COMBINE_FILES[@]}"; do [ -s "$c" ] || { echo "INFRA: PERF_COMBINE: $c is missing or empty"; exit 3; }; done
  # the parts must measure one commit: the job labels a100-<job>-<40-hex commit> must agree on the commit
  commits=$(for c in "${COMBINE_FILES[@]}"; do tail -n +2 "$c"; done | cut -d, -f1 | sed -n 's/^a100-[^-]*-\([0-9a-f]\{40\}\)$/\1/p' | sort -u)
  [ "$(echo "$commits" | grep -c .)" -le 1 ] || { echo "INFRA: PERF_COMBINE: the parts measure different commits: $(echo $commits)"; exit 3; }
  echo "combined labels: $(for c in "${COMBINE_FILES[@]}"; do tail -n +2 "$c"; done | cut -d, -f1 | sort -u | tr '\n' ' ')"
  MED_LABEL="$PERF_LABEL" write_medians "$MED" "${COMBINE_FILES[@]}"
  echo "---- combined medians ($MED) of ${#COMBINE_FILES[@]} CSV files"; cat "$MED"; exit 0
fi

# B1 (1): before any run, a probe input generated from the template must carry the requested selector; a template without
# the placeholder (or with a fixed value) refuses the whole campaign here, before an hour is spent on the wrong tables
make_input selector_probe event 1 "$(awk -v m="$MU_R" 'BEGIN { printf "%.6f", m * 2.0 }')" false true
if ! selector_ok "$WD/selector_probe/input.toml"; then
  echo "INFRA: the input template $INP/perf_template.toml does not yield exactly one qk_source = \"$PERF_QK_SOURCE\" (probe $WD/selector_probe/input.toml): campaign refused, no run started"
  echo "perf_driver.sh summary: 0 runs executed, 0 scientific failures, 1 infrastructure failures, label $PERF_LABEL"
  exit 3
fi
echo "selector probe: the generated input holds qk_source = \"$PERF_QK_SOURCE\" ($WD/selector_probe/input.toml)"
[ "${PERF_CHECK_INPUTS_ONLY:-0}" = "1" ] && exit 0

for N in $PERF_N; do
  reps="$PERF_REPEATS"; [ "$N" = "$N_TOP" ] && reps="$PERF_REPEATS_TOP"
  [ -n "${PERF_REPEAT_LAST:-}" ] && reps="$PERF_REPEAT_LAST"
  [ "$PERF_M1" = "1" ] && for mode in $PERF_MODES; do
    for var in $PERF_VARIANTS; do
      rep="$PERF_REPEAT_FIRST"
      while [ $rep -le "$reps" ]; do
        run_one "$mode" "$var" "$N" "$rep" false true m1
        rep=$((rep + 1))
      done
    done
  done
  if [ "$PERF_M5" = "1" ]; then
    rep="$PERF_REPEAT_FIRST"
    while [ $rep -le "$reps" ]; do
      case " $PERF_M5_SETS " in *" pic "*)
        run_one none warm "$N" "$rep" true false m5pic ;;            # stock pusher + deposit, hadronic off, ions only (the PIC floor)
      esac
      case " $PERF_M5_SETS " in *" poisson "*)
        run_one poisson warm "$N" "$rep" true true m5poisson ckpt    # pair-loaded population (a = 1), deposit on, hadronic on: diagnostic
        run_restart_pic "m5poisson_poisson_warm_N${N}_r${rep}" m5poisson_pic poisson "$N" "$rep" ;;  # F8: the END population, hadronic off
      esac
      case " $PERF_M5_SETS " in *" budget "*)
        run_one budget warm "$N" "$rep" true true m5budget ckpt      # the contract's budget run with deposit on (t_latency = 0): diagnostic
        run_restart_pic "m5budget_budget_warm_N${N}_r${rep}" m5budget_pic budget "$N" "$rep" ;;
      esac
      rep=$((rep + 1))
    done
  fi
done

write_medians "$MED" "$CSV"
echo "---- medians ($MED)"; cat "$MED"
if [ $n_run -eq 0 ]; then echo "INFRA: perf_driver.sh executed no run"; n_infra=$((n_infra + 1)); fi
echo "perf_driver.sh summary: $n_run runs executed, $n_fail scientific failures, $n_infra infrastructure failures, label $PERF_LABEL"
[ $n_infra -eq 0 ] || exit 3
[ $n_fail -eq 0 ] || exit 1
exit 0
