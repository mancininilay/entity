#!/usr/bin/env bash
# Stage C, after the runs (Rusty login node, no Slurm, niced): the registered checks.
#   analyze_c.sh <R-mid> <R-low-b> <R-low-c> <R-off-b> <R-off-c>      (run directories)
# 1. checks 1, 2, 4 and 5 of R-mid with the scripts of the commit of stage B (run_checks.sh,
#    R-mid in the place of R-low, the original R-off as the run without the policy; its check 3
#    is the pairwise form of stage B and is kept for information);
# 2. check 3 of stage C (check3_ensemble.py);
# 3. check 6 (check6_series.py on R-mid, check6_health.py against the series of R-low);
# 4. for information: check 4 of R-mid against the two new runs without the policy, and
#    checks 1 and 4 of the members.
# The raw extractions go to a temporary directory under stage-c/scratch, removed at the end.
set -uo pipefail
MID="$1"; LOWB="$2"; LOWC="$3"; OFFB="$4"; OFFC="$5"
R="$HOME/ceph/bh-reconnection-claude1-20261121"; C="$R/stage-c"
OFF="$R/runs/r-off-7520718"; LOW="$R/runs/r-low-7520720"
bld=$(cat "$R/BUILD") || exit 1
out="$C/analysis-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$out" || exit 1
CACHE=$(mktemp -d "$C/scratch/bhrec-cache.XXXXXX") || exit 1
trap 'rm -rf "$CACHE"' EXIT
[ -x "$C/bp2raw-login" ] || { echo "no bp2raw wrapper"; exit 1; }
module load python/3.11.11 >/dev/null 2>&1
S="$R/src/pgens/reconnection_bh/scripts"
N="$C/scripts"
J="$R/src/pgens/reconnection_bh/data/j_table_mono18.txt"
export PYTHONDONTWRITEBYTECODE=1
log="$out/analysis.log"
{
  hostname; date -u
  python3 -c "import sys, numpy; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"
  echo "commit of the build and of the registered scripts $(cat "$R/COMMIT"); build $bld"
  for d in "$OFF" "$LOW" "$MID" "$LOWB" "$LOWC" "$OFFB" "$OFFC"; do echo "run: $d (exit status $(cat "$d/exit-status" 2>/dev/null))"; done
  (cd "$S" && sha256sum *.py *.sh); sha256sum "$J"
  (cd "$N" && sha256sum *.py); (cd "$C" && sha256sum -c SCRIPTS.sha256)
} > "$log" 2>&1
grep -q "FAILED" "$log" && { echo "a script of stage C is not the registered one" >> "$log"; echo ANALYSIS DONE >> "$log"; exit 1; }
nice -n 10 bash "$S/run_checks.sh" python3 "$C/bp2raw-login" "$CACHE" "$OFF" "$MID" "$J" "$out/checks-r-mid" >> "$log" 2>&1
echo "run_checks (R-mid against R-off) exit $?" >> "$log"
PYTHONPATH="$S" nice -n 10 python3 -B "$N/check3_ensemble.py" "$C/bp2raw-login" "$CACHE" --off "$OFF" "$OFFB" "$OFFC" --low "$LOW" "$LOWB" "$LOWC" --mid "$MID" > "$out/check3_ensemble.txt" 2>> "$log"
echo "check3_ensemble exit $?" >> "$log"
PYTHONPATH="$S" nice -n 10 python3 -B "$N/check6_series.py" "$C/bp2raw-login" "$CACHE" "$MID" "$out/r-mid-core-series" > /dev/null 2>> "$log"
echo "check6_series exit $?" >> "$log"
PYTHONPATH="$S" python3 -B "$N/check6_health.py" "$out/r-mid-core-series.npz" "$C/reference/r-low-7520720-core-series.npz" "$MID" "$LOW" "$LOWB" "$LOWC" > "$out/check6.txt" 2>> "$log"
echo "check6_health exit $?" >> "$log"
mkdir -p "$out/information"
for pair in "$OFFB:$MID" "$OFFC:$MID" "$OFF:$LOWB" "$OFF:$LOWC" "$OFFB:$LOWB" "$OFFC:$LOWC"; do
  o="${pair%%:*}"; l="${pair##*:}"; tag="$(basename "$l")-against-$(basename "$o")"
  python3 -B "$S/check4_gauss.py" "$C/bp2raw-login" "$CACHE" "$o" "$l" > "$out/information/check4-$tag.txt" 2>&1
  echo "check4 $tag exit $?" >> "$log"
  python3 -B "$S/check1_counters.py" "$C/bp2raw-login" "$CACHE" "$l" "$o" > "$out/information/check1-$tag.txt" 2>&1
  echo "check1 $tag exit $?" >> "$log"
done
du -sh "$CACHE" >> "$log" 2>&1
date -u >> "$log"
echo ANALYSIS DONE >> "$log"
echo "$out"
