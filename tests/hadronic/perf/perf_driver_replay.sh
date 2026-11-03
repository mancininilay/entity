#!/usr/bin/env bash
# P2 step 7 round 3 (Codex 1's closure C1/C2, 2026-10-31): negative controls of perf_driver.sh, registered as
# HADRONIC::perf_driver_replay. NOT a simulation: the REAL logs, ledgers and checkpoints of the HADRONIC::perf_smoke run
# (13 engine runs, fixture perf_smoke_output) are replayed through the unmodified driver by a stand-in for entity.xc that
# prints the archived run.log of the run with the same name (mutated per control) and copies its ledger and checkpoint;
# bh_check is the real one. Codex 1's controls (control, missing deposit, nan deposit, missing Custom, truncated output)
# plus a zero deposit in the deposit-enabled rows, a ledger whose identity residual is perturbed (scientific: exit 1,
# FAIL lines only) and a ledger bh_check cannot read (infrastructure). Each control has a pre-registered expectation:
# the driver's exit code, the number of FAIL and INFRA lines, and (control) every timing column finite with the two
# restart rows' custom_ns_per_ion_step = n/a. Bash 3.2 compatible.
# Round 4 (Codex 1's B1 of B7398318, the q_k selector): the control's 13 generated inputs must each hold exactly one
# qk_source = "born" and its CSV must report the effective source (born on the 10 hadronic-on rows, n/a on the 3 others);
# negative controls: fit requested while the replayed engine logs say born (B7398318's situation: 10 INFRA), the startup
# line removed from the logs (10 INFRA), a template WITHOUT the placeholder (the campaign refused before any run: 1 INFRA,
# no run directory), and a template with a fixed qk_source = "born" while fit is requested (refused the same way).
# Usage: perf_driver_replay.sh <bh_check> <inputs dir> <perf_smoke workdir> <scratch dir>
set -u
CHECK="$1"; INP="$2"; SMOKE="$(cd "$3" && pwd)"; SCR="$4"
HERE="$(cd "$(dirname "$0")" && pwd)"
case "$SCR" in /*) ;; *) SCR="$PWD/$SCR" ;; esac
n_bad=0; n_check=0
for d in m1_none_warm m1_none_cold m1_poisson_warm m1_poisson_cold m1_budget_warm m1_budget_cold m1_event_warm m1_event_cold \
         m5pic_none_warm m5poisson_poisson_warm m5budget_budget_warm m5poisson_pic_poisson_warm m5budget_pic_budget_warm; do
  [ -s "$SMOKE/${d}_N65536_r1/run.log" ] || { echo "replay: the perf_smoke output $SMOKE/${d}_N65536_r1/run.log is missing"; exit 2; }
done
rm -rf "$SCR"; mkdir -p "$SCR"
# the stand-in for entity.xc: run in the driver's run directory (its basename is the run name)
cat > "$SCR/replay-entity.sh" <<'FAKE'
#!/usr/bin/env bash
name="$(basename "$PWD")"; src="$REPLAY_SMOKE/$name"
for f in "$src"/hadronic_ledger_*.txt; do
  [ -e "$f" ] || continue
  case "$REPLAY_MODE" in
    bad-ledger)        awk '/^#/ { print; next } { $23 = sprintf("%.6e", $3 * 1.0e-6 + 1.0e-3); print }' "$f" > "$(basename "$f")" ;;  # cold runs: E_drag = 0
    unreadable-ledger) awk '/^#/ { print; next } { NF = NF - 1; print }' "$f" > "$(basename "$f")" ;;
    *)                 cp "$f" . ;;
  esac
done
[ -d "$src/$name.ckpt" ] && cp -R "$src/$name.ckpt" "$name.ckpt"
case "$REPLAY_MODE" in
  missing-deposit) grep -v 'CurrentDeposit' "$src/run.log" ;;
  nan-deposit)     sed 's/^\( *CurrentDeposit\.*\)[0-9.]* [a-zµ]*s/\1nan ms/' "$src/run.log" ;;
  zero-deposit)    sed 's/^\( *CurrentDeposit\.*\)[0-9.]* [a-zµ]*s/\10.00 ns/' "$src/run.log" ;;
  missing-custom)  grep -v '^ *Custom\.' "$src/run.log" ;;
  missing-startup) grep -v '^bh_proto: qk_source = ' "$src/run.log" ;;
  short-timers)    awk '/^Timestep duration/ { n++ } n <= 13 { print }' "$src/run.log" ;;
  *)               cat "$src/run.log" ;;
esac
exit 0
FAKE
chmod +x "$SCR/replay-entity.sh"
export REPLAY_SMOKE="$SMOKE"

# control <mode> <expected exit> <expected FAIL lines> <expected INFRA lines> [requested qk_source [inputs dir]]
control() {
  local mode="$1" ex="$2" ef="$3" ei="$4" qk="${5:-born}" inp="${6:-$INP}" rc nf ni
  REPLAY_MODE="$mode" PERF_N=65536 PERF_REPEATS=1 PERF_WARMUP=20 PERF_MEASURED=100 PERF_DIAG=1 PERF_M5=1 PERF_LABEL="replay-$mode" \
    PERF_QK_SOURCE="$qk" bash "$HERE/perf_driver.sh" "$SCR/replay-entity.sh" "$CHECK" "$inp" "$SCR/$mode" > "$SCR/$mode.log" 2>&1
  rc=$?
  nf=$(grep -c '^FAIL: ' "$SCR/$mode.log"); ni=$(grep -c '^INFRA: ' "$SCR/$mode.log")
  n_check=$((n_check + 1))
  if [ "$rc" = "$ex" ] && [ "$nf" = "$ef" ] && [ "$ni" = "$ei" ]; then
    echo "REPLAY ok: $mode: exit $rc, $nf FAIL, $ni INFRA lines"
  else
    echo "REPLAY MISMATCH: $mode: exit $rc (expected $ex), $nf FAIL (expected $ef), $ni INFRA (expected $ei); log $SCR/$mode.log"; n_bad=$((n_bad + 1))
  fi
}
control control 0 0 0
control missing-deposit 3 0 13
control nan-deposit 3 0 13
control zero-deposit 3 0 5          # m5pic, m5poisson, m5budget and the two restarts have the deposit on; the eight m1 rows legitimately 0
control missing-custom 3 0 13
control short-timers 3 0 13
control bad-ledger 1 10 0           # the ten hadronic-on runs: an executed identity check fails, nothing else
control unreadable-ledger 3 0 10    # bh_check exits 2 on a row with a missing column: infrastructure
# B1: the q_k selector (the template copies: the real inputs with the template's qk_source line removed or fixed)
mkdir -p "$SCR/inputs-no-placeholder" "$SCR/inputs-fixed-born"
cp "$INP"/*.toml "$SCR/inputs-no-placeholder/"; cp "$INP"/*.toml "$SCR/inputs-fixed-born/"
grep -v '^ *qk_source *=' "$INP/perf_template.toml" > "$SCR/inputs-no-placeholder/perf_template.toml"
sed 's/^\( *qk_source *= *\)"@QK_SOURCE@"/\1"born"/' "$INP/perf_template.toml" > "$SCR/inputs-fixed-born/perf_template.toml"
control wrong-source 3 0 10 fit     # fit requested, the engine logs say born (B7398318): every hadronic-on run
control missing-startup 3 0 10      # no startup line in the hadronic-on logs: the effective source is unknown
control no-placeholder 3 0 1 born "$SCR/inputs-no-placeholder"   # refused before any run, whatever the request
control fixed-born 3 0 1 fit "$SCR/inputs-fixed-born"            # a fixed value is not the requested selector
n_check=$((n_check + 1))
n_dirs=$(find "$SCR/no-placeholder" "$SCR/fixed-born" -name run.log 2>/dev/null | wc -l | tr -d ' ')
if [ "$n_dirs" = "0" ] && [ -s "$SCR/no-placeholder/selector_probe/input.toml" ] && [ ! -e "$SCR/no-placeholder/perf_runs.csv" -o "$(wc -l < "$SCR/no-placeholder/perf_runs.csv" | tr -d ' ')" = "1" ]; then
  echo "REPLAY ok: no-placeholder and fixed-born: no run started (no run.log), the probe input retained"
else echo "REPLAY MISMATCH: a refused campaign started $n_dirs run(s) or lost its probe input"; n_bad=$((n_bad + 1)); fi
n_check=$((n_check + 1))
n_in=0; n_good=0
for f in "$SCR"/control/*/input.toml; do
  case "$f" in */selector_probe/*) continue ;; esac
  n_in=$((n_in + 1))
  [ "$(grep -c '^ *qk_source *=' "$f")" = "1" ] && grep -Eq '^ *qk_source *= *"born" *$' "$f" && n_good=$((n_good + 1))
done
if [ "$n_in" = "13" ] && [ "$n_good" = "13" ]; then echo "REPLAY ok: control: 13/13 generated inputs hold exactly one qk_source = \"born\""
else echo "REPLAY MISMATCH: control: $n_good of $n_in generated inputs hold the selector (13 expected)"; n_bad=$((n_bad + 1)); fi
n_check=$((n_check + 1))
if awk -F, 'NR > 1 { if ($2 ~ /^m1$|^m5poisson$|^m5budget$/) { if ($19 == "born") on++ } else if ($19 == "n/a") off++; n++ }
            END { exit !(n == 13 && on == 10 && off == 3) }' "$SCR/control/perf_runs.csv"; then
  echo "REPLAY ok: control CSV: qk_source = born on the 10 hadronic-on rows, n/a on the 3 others"
else echo "REPLAY MISMATCH: control CSV qk_source column ($SCR/control/perf_runs.csv)"; n_bad=$((n_bad + 1)); fi

# the control's CSV: every timing column finite in every row; the restart rows carry n/a (not nan) in custom_ns_per_ion_step
n_check=$((n_check + 1))
bad=$(awk -F, 'NR > 1 { for (i = 8; i <= 13; i++) if (i != 9 && $i !~ /^[0-9]+(\.[0-9]+)?$/) b++
                        if ($2 ~ /_pic$/) { if ($9 != "n/a") b++ } else if ($9 !~ /^[0-9]+(\.[0-9]+)?$/) b++
                        if ($16 != 1 || $17 != 1) b++; n++ }
               END { if (n != 13) b++; print b + 0 }' "$SCR/control/perf_runs.csv")
if [ "$bad" = "0" ]; then echo "REPLAY ok: control CSV: 13 rows, every timing column finite, restart rows custom_ns_per_ion_step = n/a"
else echo "REPLAY MISMATCH: control CSV has $bad invalid entries ($SCR/control/perf_runs.csv)"; n_bad=$((n_bad + 1)); fi
n_check=$((n_check + 1))
if awk -F, 'NR > 1 && $2 ~ /_pic_/ { if ($5 != "n/a") b++; n++ } END { exit !(n == 2 && b + 0 == 0) }' "$SCR/control/perf_medians.csv"; then
  echo "REPLAY ok: control medians: the two restart rows' median_custom_ns_per_ion_step = n/a"
else echo "REPLAY MISMATCH: control medians ($SCR/control/perf_medians.csv)"; n_bad=$((n_bad + 1)); fi
# round 4 (one-hour parts): a slice (one mode and variant, the poisson M5 pair only) runs exactly its 3 processes; the
# medians over the parts (PERF_COMBINE) count a key once per row across the CSVs; a combine never overwrites an existing
# medians file and refuses parts of different commits
REPLAY_MODE=control PERF_N=65536 PERF_REPEATS=1 PERF_WARMUP=20 PERF_MEASURED=100 PERF_DIAG=1 PERF_M5=1 PERF_LABEL=replay-slice \
  PERF_MODES=poisson PERF_VARIANTS=warm PERF_M5_SETS=poisson PERF_QK_SOURCE=born \
  bash "$HERE/perf_driver.sh" "$SCR/replay-entity.sh" "$CHECK" "$INP" "$SCR/slice" > "$SCR/slice.log" 2>&1
rc=$?
n_check=$((n_check + 1))
tags=$(tail -n +2 "$SCR/slice/perf_runs.csv" | cut -d, -f2,3,4 | tr '\n' ' ')
if [ "$rc" = "0" ] && [ "$tags" = "m1,poisson,warm m5poisson,poisson,warm m5poisson_pic,poisson,warm " ]; then
  echo "REPLAY ok: slice (poisson warm, M5 poisson pair): exit 0, exactly its 3 processes"
else echo "REPLAY MISMATCH: slice: exit $rc, rows '$tags'"; n_bad=$((n_bad + 1)); fi
PERF_COMBINE="$SCR/control/perf_runs.csv:$SCR/slice/perf_runs.csv" PERF_LABEL=replay-combined \
  bash "$HERE/perf_driver.sh" "$SCR/replay-entity.sh" "$CHECK" "$INP" "$SCR/combined" > "$SCR/combined.log" 2>&1
rc=$?
n_check=$((n_check + 1))
if [ "$rc" = "0" ] && grep -q '^replay-combined,m1_poisson_warm_65536,2,' "$SCR/combined/perf_medians.csv" \
   && grep -q '^replay-combined,m5pic_none_warm_65536,1,' "$SCR/combined/perf_medians.csv" && [ ! -e "$SCR/combined/perf_runs.csv" ]; then
  echo "REPLAY ok: combined medians over two parts: m1 poisson warm from 2 rows, m5pic from 1; no run, no CSV written"
else echo "REPLAY MISMATCH: combined medians (exit $rc, $SCR/combined)"; n_bad=$((n_bad + 1)); fi
cp "$SCR/control/perf_medians.csv" "$SCR/control-medians-before.csv"
PERF_COMBINE="$SCR/control/perf_runs.csv" bash "$HERE/perf_driver.sh" "$SCR/replay-entity.sh" "$CHECK" "$INP" "$SCR/control" > "$SCR/combine-overwrite.log" 2>&1
rc=$?
n_check=$((n_check + 1))
if [ "$rc" = "3" ] && cmp -s "$SCR/control/perf_medians.csv" "$SCR/control-medians-before.csv" && [ "$(wc -l < "$SCR/control/perf_runs.csv" | tr -d ' ')" = "14" ]; then
  echo "REPLAY ok: combine into a directory with medians: refused (exit 3), its CSV and medians untouched"
else echo "REPLAY MISMATCH: combine-overwrite (exit $rc)"; n_bad=$((n_bad + 1)); fi
mkdir -p "$SCR/commits"
for c in 1111111111111111111111111111111111111111 2222222222222222222222222222222222222222; do
  sed "2,\$ s/^[^,]*,/a100-9-$c,/" "$SCR/slice/perf_runs.csv" > "$SCR/commits/$c.csv"
done
PERF_COMBINE="$SCR/commits/1111111111111111111111111111111111111111.csv:$SCR/commits/2222222222222222222222222222222222222222.csv" \
  bash "$HERE/perf_driver.sh" "$SCR/replay-entity.sh" "$CHECK" "$INP" "$SCR/commits-combined" > "$SCR/combine-commits.log" 2>&1
rc=$?
n_check=$((n_check + 1))
if [ "$rc" = "3" ] && grep -q "different commits" "$SCR/combine-commits.log"; then echo "REPLAY ok: combine of parts at two commits refused (exit 3)"
else echo "REPLAY MISMATCH: combine of two commits (exit $rc)"; n_bad=$((n_bad + 1)); fi
echo "perf_driver_replay summary: $n_check checks, $n_bad mismatches"
[ "$n_bad" -eq 0 ]
