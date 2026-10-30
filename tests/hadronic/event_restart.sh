#!/usr/bin/env bash
# Restart refusal of the event mode at the run level (P2 step 5, contract section 3; E8 (iv)):
#   A: a continuous-mode (budget) run writes checkpoints every 40 steps (Entity labels them 41 and 81; ions
#      with non-zero debt and pending payloads);
#   B: resuming A with mode = event (-continue, same name and checkpoint path; tau0 = 0.4 so that the
#      lambda_max rule admits the input) is refused with a message naming the non-zero hadronic payloads
#      and the start step of the resume (the latest checkpoint, > 40);
#   C: resuming A in the continuous mode completes (control: the checkpoint is readable);
#   D: an event-mode run checkpointed every 40 steps, then resumed in the event mode: the payload scan
#      reports every slot zero ("clean") and the run completes.
# Bash 3.2 compatible; fails when no run was executed. Usage: event_restart.sh <entity.xc> <bh_check> <inputs> <workdir>
source "$(dirname "$0")/common.sh"
EXE="$1"; CHECK="$2"; INP="$3"; WD="$4"
n_run=0
resume_in() {
  # resume_in <dir> <input> <logname>: run in <dir> (where the checkpoints are) with -continue
  local dir="$1" inp="$2" log="$3"
  ( cd "$dir" && "$EXE" -input "$inp" -continue > "$log" 2>&1 )
  return $?
}
# A
run_entity "$EXE" "$INP/event_restart_a.toml" "$WD/A"; rc=$?; n_run=$((n_run + 1))
check_rc $rc "run A (continuous, budget, checkpoints at 40 and 80) completes"
if [ $rc -eq 0 ]; then
  ls "$WD/A" | grep -q "ckpt\|checkpoint" && echo "ok: run A wrote a checkpoint directory: $(ls -d "$WD"/A/*ckpt* 2>/dev/null | head -1)" \
    || { echo "FAIL: run A wrote no checkpoint directory"; report_fail=1; }
  # the debt and pending payloads of the ions at the end of A are non-zero for at least one ion (per-ion output)
  nz=$(grep -v '^#' "$WD/A/hadronic_ions_1.txt" | awk '($8 != 0 || $10 != 0) {c++} END {print c+0}')
  if [ "${nz:-0}" -gt 0 ]; then
    echo "PASS: run A: $nz ion rows with a non-zero pending or debt payload in the per-ion output"
  else
    echo "FAIL: run A: no ion row with a non-zero payload (the refusal of B would be vacuous)"; report_fail=1
  fi
fi
# B: refused
if [ $rc -eq 0 ]; then
  resume_in "$WD/A" "$INP/event_restart_b.toml" "runB.log"; rcb=$?; n_run=$((n_run + 1))
  if [ $rcb -eq 0 ]; then
    echo "FAIL: run B (resume in the event mode) was accepted (exit 0)"; report_fail=1
  else
    echo "PASS: run B (resume in the event mode) refused (exit $rcb)"
    grep -q "mode = event entered with [1-9][0-9]* alive ions carrying a non-zero hadronic payload" "$WD/A/runB.log" \
      && echo "PASS: run B: the message names the non-zero hadronic payloads: $(grep -m1 -o 'mode = event entered with [0-9]* alive ions carrying a non-zero hadronic payload (start_step [0-9]*)' "$WD/A/runB.log")" \
      || { echo "FAIL: run B: the refusal message is missing or names no payload count"; report_fail=1; }
    sb=$(grep -o "start_step [0-9]*" "$WD/A/runB.log" | head -1 | awk '{print $2}')
    if [ -n "$sb" ] && [ "$sb" -gt 40 ] 2>/dev/null; then
      echo "PASS: run B resumed from the latest checkpoint (start_step $sb > 40)"
    else
      echo "FAIL: run B did not resume from a checkpoint (start_step '$sb')"; report_fail=1
    fi
    grep -q "payload scan at step [0-9]* (start_step [0-9]*): [1-9][0-9]* alive ions.*REFUSED" "$WD/A/runB.log" \
      && echo "PASS: run B: the payload scan line says REFUSED" \
      || { echo "FAIL: run B: no REFUSED payload scan line"; report_fail=1; }
  fi
  # C: control resume in the continuous mode
  resume_in "$WD/A" "$INP/event_restart_c.toml" "runC.log"; rcc=$?; n_run=$((n_run + 1))
  check_rc $rcc "run C (resume A in the continuous mode) completes"
fi
# D: event mode checkpointed and resumed in the event mode
run_entity "$EXE" "$INP/event_restart_d.toml" "$WD/D"; rcd=$?; n_run=$((n_run + 1))
check_rc $rcd "run D (event mode, checkpoints at 40 and 80) completes"
if [ $rcd -eq 0 ]; then
  grep -q "payload scan at step 0 .*0 alive ions carry a non-zero hadronic payload.*clean" "$WD/D/run.log" \
    && echo "PASS: run D: the payload scan at step 0 is clean" || { echo "FAIL: run D: no clean payload scan at step 0"; report_fail=1; }
  resume_in "$WD/D" "$INP/event_restart_d2.toml" "runD2.log"; rcd2=$?; n_run=$((n_run + 1))
  check_rc $rcd2 "run D2 (resume D in the event mode) completes"
  if [ $rcd2 -eq 0 ]; then
    sd=$(grep -o "start_step [0-9]*" "$WD/D/runD2.log" | head -1 | awk '{print $2}')
    grep -q "payload scan at step [0-9]* (start_step [1-9][0-9]*): 0 alive ions carry a non-zero hadronic payload.*clean" "$WD/D/runD2.log" \
      && echo "PASS: run D2: the payload scan at the resume (start_step $sd) is clean" \
      || { echo "FAIL: run D2: no clean payload scan at the resume"; report_fail=1; }
    "$CHECK" ledger --file "$WD/D/hadronic_ledger_1.txt" --expect_bounded 0 || report_fail=1
  fi
fi
if [ $n_run -eq 0 ]; then
  echo "FAIL: event_restart.sh executed no run"; report_fail=1
fi
echo "event_restart.sh summary: $n_run runs executed, report_fail = $report_fail"
[ $report_fail -eq 0 ] && [ $n_run -gt 0 ]
