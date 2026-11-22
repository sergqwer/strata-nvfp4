#!/bin/bash
# queue 23: + STRATA_GRAPH_PRIO=1 (indexer branch high priority) on whole and on the best split; identity, nsys, serve
H=/d/Projects/strata-dec-int/bench-int; cd $H
I=D:/Projects/strata-dec-int/build-dev/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"; F=D:/Projects/basalt-work/fixtures
T=D:/Projects/Strata-data/mtp/rt-tuned; MTP=(--mtp $T --mtp-head proxy --mtp-cum auto --spec 8)
SMP=(--temperature 1.0 --top-p 0.95 --top-k 20 --seed 42)
G=${GBEST:-4}
export STRATA_SPEC_FIXED=1 STRATA_GRAPH_PRIO=1
for fx in conv struct; do n=400; [ $fx = struct ] && n=2000
  bash exact.sh n14_p_D_$fx D $I $F/$fx.txt $n
  bash exact.sh n14_p_P8_$fx M $I $F/$fx.txt $n "${MTP[@]}"
  STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=$G bash exact.sh n14_pg_D_$fx D $I $F/$fx.txt $n
done > exact14.txt 2>&1
bash exact.sh n14_p_Ds_prose D $I $F/prose.txt 600 "${SMP[@]}" >> exact14.txt 2>&1
unset STRATA_SPEC_FIXED STRATA_GRAPH_PRIO; echo EXACT-DONE
V=("w:" "wp:STRATA_GRAPH_PRIO=1" "gp:STRATA_GRAPH_PRIO=1 STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=$G" "g3p:STRATA_GRAPH_PRIO=1 STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=3")
for v in "${V[@]:1}"; do t=${v%%:*}; e=${v#*:}
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD $e" bash nsys_dec.sh h_$t strata $F/prose.txt 600 > nsys/h_$t.out 2>&1
  echo "$t: $(python /d/Projects/strata-dec-host/bench-host/nsys_front_layer.py nsys/h_$t.sqlite strata | tail -2 | tr '\n' ' ')"
  rm -f nsys/h_$t.nsys-rep
done > front_r8p.txt 2>&1; echo NSYS-DONE
for r in 1 2; do for fx in struct conv prose uk; do for v in "${V[@]}"; do t=${v%%:*}; e=${v#*:}
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD $e" FIXTURES=$fx bash serve.sh f_${t}_r$r strata
done; done; done > serve_r8p.txt 2>&1; echo SERVE-DONE
echo Q23-DONE
