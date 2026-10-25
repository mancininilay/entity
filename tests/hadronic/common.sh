# shared helpers for the bh_proto run-based tests (sourced by the scripts)
set -u
run_entity() {
  # run_entity <entity.xc> <input.toml> <workdir> : runs in <workdir>, log in <workdir>/run.log
  local exe="$1" inp="$2" wd="$3"
  rm -rf "$wd" && mkdir -p "$wd" || return 2
  ( cd "$wd" && "$exe" -input "$inp" > run.log 2>&1 )
  return $?
}
report_fail=0
check_rc() {
  # check_rc <rc> <what>
  if [ "$1" -ne 0 ]; then
    echo "FAIL: $2 (exit code $1)"
    report_fail=1
  else
    echo "ok: $2"
  fi
}
