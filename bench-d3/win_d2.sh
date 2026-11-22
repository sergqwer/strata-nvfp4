#!/bin/bash
# win_d2.sh OUT [PAIRS] - D2 window level: CLI fixed placement (exact.sh, STRATA_SPEC_FIXED=1, MTP arm M, plus
# --expert-cache auto so the pool is not the whole window), d1 vs d2 interleaved (ABBA), prose 600 + struct 600 tokens.
# exact.sh takes the GPU lock itself: never run this inside gpu_lock.sh.
source /d/Projects/strata-dec3/bench-d3/gpu_free.sh
X=/d/Projects/strata-dec-int/bench-int/exact.sh; F=D:/Projects/basalt-work/fixtures; RAW=/d/Projects/Strata-data/dflash-check/raw
D=D:/Projects/strata-dec3/bench-d3; OUT=$1; P=${2:-3}
declare -A E=([d1]=$D/d1/strata.exe [d2]=$D/d2/strata.exe)
nvidia-smi --query-gpu=power.limit,clocks.mem,clocks.max.memory --format=csv,noheader | tee -a $OUT
run() {  # run ARM FX I
  local tag=d3_d2w_$1_$2_$3
  gpu_free || return 1
  local line=$(STRATA_SPEC_FIXED=1 bash $X $tag M ${E[$1]} $F/$2.txt 600 --expert-cache auto)
  local vw=$(grep -a "verify window" $RAW/$tag.log | tail -1 | sed -E 's/.*verify window +//; s/; CPU.*//')
  local ws=$(grep -a "speculation" $RAW/$tag.log | tail -1 | grep -a -o -E "[0-9]+ rounds")
  echo "$1 $2 $3 | $ws | ${line#*| *| *| } | $vw" | tee -a $OUT
}
for fx in prose struct; do
  for i in $(seq 1 $P); do
    if [ $((i % 2)) = 1 ]; then run d1 $fx $i; run d2 $fx $i; else run d2 $fx $i; run d1 $fx $i; fi
  done
done
nvidia-smi --query-gpu=power.limit,clocks.mem --format=csv,noheader | tee -a $OUT
