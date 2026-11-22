#!/bin/bash
# sweep_head.sh OUT [ROUNDS] - E4 (c): gdn_parity (bitwise gdn_mid vs the chain + cold-L2 timing T 1/3/8) over
# STRATA_GDN_MID_HEAD (HEADS, default "0 1 2 3 7"), D2's default stream (DEPTH 1, PIECES 4), heads interleaved per round.
# Run inside gpu_lock.  Under DEPTH > 0 only bits 0 (the up-front piece) and 1 (the L2 prefetches) act: 7 = 3.
B=${B:-/d/Projects/strata-dec3/build-dev/gdn_parity.exe}
OUT=$1; R=${2:-4}
nvidia-smi --query-gpu=power.limit,clocks.mem,clocks.max.memory,temperature.gpu --format=csv,noheader > $OUT
for r in $(seq 1 $R); do
  for h in ${HEADS:-0 1 2 3 7}; do
    res=$(STRATA_GDN_PARITY_REPS=${REPS:-50} STRATA_GDN_MID_HEAD=$h timeout 180 $B 2>&1 | tr -d '\r')
    t=$(echo "$res" | grep -a "gdn_mid timing" | sed -E 's/.*T=([0-9]+).*gdn_mid ([0-9.]+) us/\1:\2/' | tr '\n' ' ')
    bw=$(echo "$res" | grep -a "gdn_mid ==" | grep -q bitwise && echo bitwise || echo DIFF)
    nb=$(echo "$res" | grep -a -c '\*\*\*')
    echo "r$r head $h  $t $bw *** $nb" >> $OUT
  done
done
nvidia-smi --query-gpu=power.limit,clocks.mem,temperature.gpu --format=csv,noheader >> $OUT
