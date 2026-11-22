#!/bin/bash
# queue 24: gate rows on a capped grid (STRATA_QPROJ_GATE_GRID) - nsys fronts
H=/d/Projects/strata-dec-int/bench-int; cd $H
I=D:/Projects/strata-dec-int/build-dev/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"; F=D:/Projects/basalt-work/fixtures
Q="STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=3"
V=("c24:$Q STRATA_QPROJ_GATE_GRID=24" "c48:$Q STRATA_QPROJ_GATE_GRID=48" "c80:$Q STRATA_QPROJ_GATE_GRID=80" "c48p:$Q STRATA_QPROJ_GATE_GRID=48 STRATA_GRAPH_PRIO=1" "c48a1:STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=1 STRATA_QPROJ_GATE_GRID=48")
for v in "${V[@]}"; do t=${v%%:*}; e=${v#*:}
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD $e" bash nsys_dec.sh h_$t strata $F/prose.txt 600 > nsys/h_$t.out 2>&1
  echo "$t: $(python /d/Projects/strata-dec-host/bench-host/nsys_front_layer.py nsys/h_$t.sqlite strata | tail -2 | tr '\n' ' ')"
  rm -f nsys/h_$t.nsys-rep
done > front_r8c.txt 2>&1; echo Q24-DONE
