#!/bin/bash
# sweep_d2.sh OUT [ROUNDS] - D2 kernel-level sweep: gdn_parity (bitwise gdn_mid vs the chain + cold-L2 timing T 1/3/8) over
# STRATA_GDN_MID_DEPTH / _PIECES / _DEPTHZ / _RZ (CFG = "d:n[:dz[:rz]] ..."), the engine's HEAD=2, configs interleaved per
# round.  Run inside gpu_lock.  Summary: python sweep_sum.py OUT
B=${B:-/d/Projects/strata-dec3/build-dev/gdn_parity.exe}
OUT=$1; R=${2:-3}
CFG=${CFG:-"0:4 1:1 2:1 1:2 2:2 1:4 2:4 3:4 4:4 2:8 3:8 4:8 6:8 8:8 2:5 4:5"}
nvidia-smi --query-gpu=power.limit,clocks.mem,clocks.max.memory,temperature.gpu --format=csv,noheader > $OUT
for r in $(seq 1 $R); do
  for c in $CFG; do
    IFS=: read -r d n dz rz <<< "$c"
    dz=${dz:-$d}; rz=${rz:-0}
    res=$(STRATA_GDN_PARITY_REPS=${REPS:-50} STRATA_GDN_MID_HEAD=${HEAD:-2} STRATA_GDN_MID_DEPTH=$d STRATA_GDN_MID_PIECES=$n \
          STRATA_GDN_MID_DEPTHZ=$dz STRATA_GDN_MID_RZ=$rz timeout 120 $B 2>&1 | tr -d '\r')
    t=$(echo "$res" | grep -a "gdn_mid timing" | sed -E 's/.*T=([0-9]+).*gdn_mid ([0-9.]+) us/\1:\2/' | tr '\n' ' ')
    bw=$(echo "$res" | grep -a "gdn_mid ==" | grep -q bitwise && echo bitwise || echo DIFF)
    echo "r$r depth $d pieces $n depthz $dz rz $rz  $t $bw" >> $OUT
  done
done
nvidia-smi --query-gpu=power.limit,clocks.mem,temperature.gpu --format=csv,noheader >> $OUT
