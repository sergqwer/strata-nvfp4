#!/bin/bash
# final_d2.sh OUT - D2 kernel level, the shipped default against the old path: gdn_parity (bitwise + cold-L2 T 1/3/8),
# 4 interleaved rounds; also split 0 both ways.  Run inside gpu_lock.
B=/d/Projects/strata-dec3/build-dev/gdn_parity.exe
OUT=$1
nvidia-smi --query-gpu=power.limit,clocks.mem,clocks.max.memory,temperature.gpu --format=csv,noheader > $OUT
for r in 1 2 3 4; do
  for arm in "old:STRATA_GDN_MID_DEPTH=0" "new:STRATA_GDN_MID_X=1" "old_split0:STRATA_GDN_MID_DEPTH=0 STRATA_GDN_MID_SPLIT=0" "new_split0:STRATA_GDN_MID_SPLIT=0"; do
    name=${arm%%:*}; envs=${arm#*:}
    res=$(env STRATA_GDN_PARITY_REPS=50 STRATA_GDN_MID_HEAD=2 $envs timeout 120 $B 2>&1 | tr -d '\r')
    t=$(echo "$res" | grep -a "gdn_mid timing" | sed -E 's/.*T=([0-9]+).*gdn_mid ([0-9.]+) us/\1:\2/' | tr '\n' ' ')
    bw=$(echo "$res" | grep -a "gdn_mid ==" | sed -E 's/.*(bitwise|NO).*\(([0-9]+) cases\)/\1 \2/')
    fails=$(echo "$res" | grep -a -c '\*\*\*')
    echo "r$r $name  $t | $bw | *** lines $fails" >> $OUT
  done
done
nvidia-smi --query-gpu=power.limit,clocks.mem,temperature.gpu --format=csv,noheader >> $OUT
