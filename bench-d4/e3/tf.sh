#!/bin/bash
# tf.sh - E3 accuracy: the release's teacher-forced corpus (basalt-work/corpus/set.json, 19 items = 19 requests, 19437 positions,
# score_strata.py = the tray GPTQ args + DFlash2 spec 6, serve, adaptive placement), e2 exe. Arms: e2_A, e2_B (noise floor), d90.
# compare.py: per position, 95% bootstrap over the 19 items (each item is its own request = per answer).
source /d/Projects/strata-dec3/bench-d3/gpu_free.sh
LK=/d/Projects/Strata-data/bench-tools/gpu_lock.sh; W=/d/Projects/basalt-work; PY=$W/.venv/Scripts/python.exe
O=D:/Projects/strata-dec3/bench-d4/e3/tf; mkdir -p /d/Projects/strata-dec3/bench-d4/e3/tf
E=D:/Projects/strata-dec3/bench-d4/e2/strata.exe
run() {  # NAME SCORE_ENV
  local t0=$(date +%s)
  echo "# $1 [$2] $(date +%T) $(nvidia-smi --query-gpu=power.limit,clocks.mem --format=csv,noheader)"
  ( cd $W && DF_EXE=$E SCORE_ENV="$2" timeout 900 bash $LK e3-tf bash -c 'gpu_free && exec "$@"' _ \
      $PY score_strata.py corpus/set.json $O/$1.tsv > /d/Projects/strata-dec3/bench-d4/e3/tf/$1.out 2>&1 )
  echo "$1 rc=$? $(( $(date +%s) - t0 )) s | $(grep -a -c 'off the forced' /d/Projects/strata-dec3/bench-d4/e3/tf/$1.out) items | $(grep -a -o -E 'expert cache [0-9]+ slots' $O/$1.tsv.engine.log | tail -1)"
  sleep 3
}
for a in ${ARMS:-e2_A d90 e2_B}; do
  case $a in e2_*) run $a "";; d90) run d90 "STRATA_ADAPT_DECAY=0.9";; esac
done
cd $W
$PY compare.py $O/e2_A.tsv strata:e2_B=$O/e2_B.tsv strata:d90=$O/d90.tsv
$PY compare.py tf/strata_A.tsv strata:e2_A=$O/e2_A.tsv strata:e2_B=$O/e2_B.tsv strata:d90=$O/d90.tsv
