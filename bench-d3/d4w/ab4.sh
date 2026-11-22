#!/bin/bash
# ab4.sh (D4: ab.sh + A_ARGS / B_ARGS extra engine args, per layer-window counts kept) NAME A_EXE B_EXE - decode round 3 A/B: engine --serve with the GPTQ tray profile's args (rel04/dec_client.py: the
# profile's own mtp folder + this tree's serve Auto), one greedy GEN per arm, arms interleaved (pair i: AB, then BA, ...).
# env: A_ENV / B_ENV (STRATA_* sets for each arm), PAIRS (3), FIXTURES ("struct prose"), GEN_<fixture> (tokens).
# Out: bench-d3/ab_NAME.txt (one line per arm + a summary); logs bench-d3/log/ab_NAME_<arm>_<fx>_<i>.log.
source /d/Projects/Strata-data/rel04/common.sh
source /d/Projects/strata-dec3/bench-d3/gpu_free.sh   # gpu_free without the zombie tray PID
D=/d/Projects/strata-dec3/bench-d3; TREE=D:/Projects/strata-dec3
name=$1; EA=$2; EB=$3
[ -x "$EA" ] && [ -x "$EB" ] || { echo "usage: ab.sh NAME A_EXE B_EXE" >&2; exit 2; }
OUT=$D/ab_$name.txt; mkdir -p $D/log
declare -A F=([conv]=D:/Projects/basalt-work/fixtures/conv.txt [struct]=D:/Projects/basalt-work/fixtures/struct.txt
              [prose]=D:/Projects/basalt-work/fixtures/prose.txt [uk]=D:/Projects/Strata-data/mtp/uk.txt)
declare -A N=([conv]=${GEN_conv:-1200} [struct]=${GEN_struct:-1500} [prose]=${GEN_prose:-1200} [uk]=${GEN_uk:-1200})
one() {  # one ARM FIXTURE I
  local arm=$1 fx=$2 i=$3 exe env="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
  local add
  if [ $arm = A ]; then exe=$EA; env="$env $A_ENV"; add="$A_ARGS"; else exe=$EB; env="$env $B_ENV"; add="$B_ARGS"; fi
  local log=D:/Projects/strata-dec3/bench-d3/log/ab_${name}_${arm}_${fx}_$i.log
  local out=$(PF_ADD="$add" PF_EXE=$exe PF_GEN=${N[$fx]} PF_ENV="$env" PF_LOG=$log timeout 600 bash $LK d3-ab bash -c 'gpu_free && exec "$@"' _ \
              $PY -B /d/Projects/strata-dec3/bench-d3/d4w/dec_client4.py gptq $TREE ${F[$fx]} 2>&1 | grep -a -v "^waiting" | tail -1)
  local tps=$(grep -a -o -E '[0-9]+ generated in [0-9]+ ms \([0-9.]+ tok/s\)' $log | tail -1)
  local tw=$(grep -a 'strata decode timing:' $log | tail -1 | sed -E 's/.*decode timing: //; s/; per layer-window.*//; s/ \+ commit.*draft/ draft/')
  local pl=$(grep -a 'strata decode timing:' $log | tail -1 | sed -E 's/.*per layer-window: //')
  echo "$arm $fx $i | $tps | $tw | $pl" | tee -a $OUT
  [ -n "$tps" ] || echo "   no result: ${out:0:200}" | tee -a $OUT
  sleep 3
}
echo "# ab $name $(date '+%F %T') A=$EA [$A_ENV] [$A_ARGS] B=$EB [$B_ENV] [$B_ARGS]" | tee -a $OUT
pw | tee -a $OUT
for fx in ${FIXTURES:-struct prose}; do
  for i in $(seq 1 ${PAIRS:-3}); do
    if [ $((i % 2)) = 1 ]; then one A $fx $i; one B $fx $i; else one B $fx $i; one A $fx $i; fi
  done
  pw | tee -a $OUT
done
$PY -B $D/ab_sum.py $OUT "$name" | tee -a $OUT
