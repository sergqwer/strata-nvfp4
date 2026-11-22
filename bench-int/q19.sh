#!/bin/bash
# queue 19: the attention front per variant (nsys, prose): gate rows at 0 / 1 / 2, q proj whole
H=/d/Projects/strata-dec-int/bench-int; cd $H
I=D:/Projects/strata-dec-int/build-dev/strata.exe; CI=$H/pf_client_int.py; F=D:/Projects/basalt-work/fixtures
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
for arm in "g0|STRATA_QPROJ_GATE_AT=0" "g1|STRATA_QPROJ_GATE_AT=1" "g2|STRATA_QPROJ_GATE_AT=2" "qw|STRATA_QPROJ_HALF=0" "kp0|STRATA_QPROJ_HALF=0 STRATA_KV_POST=0"; do
  IFS='|' read -r t e <<< "$arm"
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD $e" bash nsys_dec.sh v_$t strata $F/prose.txt 600 > nsys/v_$t.out 2>&1
  echo "$t: $(python /d/Projects/strata-dec-host/bench-host/nsys_front_layer.py nsys/v_$t.sqlite strata | tail -2 | tr '\n' ' ') | $(grep -a -o 'decode timing: [0-9]* windows, avg T [0-9.]*, [0-9.]* tokens/window, [0-9.]* ms/window' nsys/v_$t.engine.log | tail -1)"
  rm -f nsys/v_$t.nsys-rep
done > front_var.txt 2>&1
echo Q19-DONE
