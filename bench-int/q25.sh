#!/bin/bash
# queue 25: capped gate grid - identity + serve A/B unprofiled
H=/d/Projects/strata-dec-int/bench-int; cd $H
I=D:/Projects/strata-dec-int/build-dev/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"; F=D:/Projects/basalt-work/fixtures
T=D:/Projects/Strata-data/mtp/rt-tuned; MTP=(--mtp $T --mtp-head proxy --mtp-cum auto --spec 8)
SMP=(--temperature 1.0 --top-p 0.95 --top-k 20 --seed 42)
Q="STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=3"
( export STRATA_SPEC_FIXED=1 STRATA_GRAPH_PRIO=1 STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=3 STRATA_QPROJ_GATE_GRID=48
for fx in conv struct; do n=400; [ $fx = struct ] && n=2000
  bash exact.sh n15_c48p_D_$fx D $I $F/$fx.txt $n
  bash exact.sh n15_c48p_P8_$fx M $I $F/$fx.txt $n "${MTP[@]}"
done
bash exact.sh n15_c48p_Ds_prose D $I $F/prose.txt 600 "${SMP[@]}" ) > exact15.txt 2>&1; echo EXACT-DONE
V=("w:" "wp:STRATA_GRAPH_PRIO=1" "c48:$Q STRATA_QPROJ_GATE_GRID=48" "c48p:$Q STRATA_QPROJ_GATE_GRID=48 STRATA_GRAPH_PRIO=1" "c64p:$Q STRATA_QPROJ_GATE_GRID=64 STRATA_GRAPH_PRIO=1")
for r in 1 2; do for fx in struct conv prose uk; do for v in "${V[@]}"; do t=${v%%:*}; e=${v#*:}
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD $e" FIXTURES=$fx bash serve.sh g_${t}_r$r strata
done; done; done > serve_r8c.txt 2>&1; echo SERVE-DONE
echo Q25-DONE
