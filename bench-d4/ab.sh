#!/bin/bash
# ab.sh NAME A_EXE B_EXE - bench-d3/ab.sh for round 4 (E1): + A_ADD / B_ADD (extra engine args per arm, e.g. "--spec 12",
# through r4/dec_client4.py's PF_ADD), the slot count, the draft ms and ms/token in each line.
# env: A_ENV / B_ENV, A_ADD / B_ADD, PAIRS (3), FIXTURES ("struct prose"), GEN_<fixture>.
# Out: bench-d4/ab_NAME.txt; logs bench-d4/log/ab_NAME_<arm>_<fx>_<i>.log.
source /d/Projects/Strata-data/rel04/common.sh
source /d/Projects/strata-dec3/bench-d3/gpu_free.sh
D=/d/Projects/strata-dec3/bench-d4; TREE=D:/Projects/strata-dec3
name=$1; EA=$2; EB=$3
[ -x "$EA" ] && [ -x "$EB" ] || { echo "usage: ab.sh NAME A_EXE B_EXE" >&2; exit 2; }
OUT=$D/ab_$name.txt; mkdir -p $D/log
declare -A F=([conv]=D:/Projects/basalt-work/fixtures/conv.txt [struct]=D:/Projects/basalt-work/fixtures/struct.txt
              [prose]=D:/Projects/basalt-work/fixtures/prose.txt [uk]=D:/Projects/Strata-data/mtp/uk.txt)
declare -A N=([conv]=${GEN_conv:-1200} [struct]=${GEN_struct:-1500} [prose]=${GEN_prose:-1200} [uk]=${GEN_uk:-1200})
one() {  # one ARM FIXTURE I
  local arm=$1 fx=$2 i=$3 exe add env="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
  if [ $arm = A ]; then exe=$EA; env="$env $A_ENV"; add=$A_ADD; else exe=$EB; env="$env $B_ENV"; add=$B_ADD; fi
  local log=D:/Projects/strata-dec3/bench-d4/log/ab_${name}_${arm}_${fx}_$i.log
  local out=$(PF_EXE=$exe PF_GEN=${N[$fx]} PF_ENV="$env" PF_ADD="$add" PF_LOG=$log timeout 600 bash $LK d4-ab bash -c 'gpu_free && exec "$@"' _ \
              $PY -B /d/Projects/gpu-probe/r4/dec_client4.py gptq $TREE ${F[$fx]} 2>&1 | grep -a -v "^waiting" | tail -1)
  local tps=$(grep -a -o -E '[0-9]+ generated in [0-9]+ ms \([0-9.]+ tok/s\)' $log | tail -1)
  local tw=$(grep -a 'strata decode timing:' $log | tail -1 | sed -E 's/.*decode timing: //; s/; per layer-window.*//; s/ \+ commit.*draft/ draft/')
  local sl=$(grep -a -o -E 'expert cache [0-9]+ slots' $log | tail -1 | grep -o -E '[0-9]+')
  local wu=$(grep -a -o -E 'window up to [0-9]+ tokens, [0-9.]+ MiB' $log | tail -1)
  local fr=$(grep -a -o -E 'only [0-9]+ MiB of VRAM free at the start' $log | tail -1 | grep -o -E '[0-9]+')
  echo "$arm $fx $i | $tps | $tw | slots $sl | $wu | free ${fr:-ok}" | tee -a $OUT
  [ -n "$tps" ] || echo "   no result: ${out:0:200}" | tee -a $OUT
  sleep 3
}
echo "# ab $name $(date '+%F %T') A=$EA [$A_ENV] [$A_ADD] B=$EB [$B_ENV] [$B_ADD]" | tee -a $OUT
pw | tee -a $OUT
for fx in ${FIXTURES:-struct prose}; do
  for i in $(seq 1 ${PAIRS:-3}); do
    if [ $((i % 2)) = 1 ]; then one A $fx $i; one B $fx $i; else one B $fx $i; one A $fx $i; fi
  done
  pw | tee -a $OUT
done
$PY -B $D/ab_sum.py $OUT "$name" | tee -a $OUT
