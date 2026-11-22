#!/bin/bash
# queue 22: attention-layer stream layout - q|gate split placements (STRATA_QPROJ_HALF=1, GATE_AT 0/1/3) vs whole:
# identity (fixed costs), nsys front, serve ms/window unprofiled, 4 fixtures x 2 rounds
H=/d/Projects/strata-dec-int/bench-int; cd $H
I=D:/Projects/strata-dec-int/build-dev/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"; F=D:/Projects/basalt-work/fixtures
T=D:/Projects/Strata-data/mtp/rt-tuned; MTP=(--mtp $T --mtp-head proxy --mtp-cum auto --spec 8)
SMP=(--temperature 1.0 --top-p 0.95 --top-k 20 --seed 42)
export STRATA_SPEC_FIXED=1
for g in 1 3; do
  for fx in conv struct; do n=400; [ $fx = struct ] && n=2000
    STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=$g bash exact.sh n13_g${g}_D_$fx D $I $F/$fx.txt $n
    STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=$g bash exact.sh n13_g${g}_P8_$fx M $I $F/$fx.txt $n "${MTP[@]}"
  done
  STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=$g bash exact.sh n13_g${g}_Ds_prose D $I $F/prose.txt 600 "${SMP[@]}"
done > exact13.txt 2>&1
unset STRATA_SPEC_FIXED; echo EXACT-DONE
V=("w:" "g0:STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=0" "g1:STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=1" "g3:STRATA_QPROJ_HALF=1 STRATA_QPROJ_GATE_AT=3")
for v in "${V[@]}"; do t=${v%%:*}; e=${v#*:}
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD $e" bash nsys_dec.sh h_$t strata $F/prose.txt 600 > nsys/h_$t.out 2>&1
  echo "$t: $(python /d/Projects/strata-dec-host/bench-host/nsys_front_layer.py nsys/h_$t.sqlite strata | tail -2 | tr '\n' ' ')"
  rm -f nsys/h_$t.nsys-rep
done > front_r8.txt 2>&1; echo NSYS-DONE
for r in 1 2; do for fx in struct conv prose uk; do for v in "${V[@]}"; do t=${v%%:*}; e=${v#*:}
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD $e" FIXTURES=$fx bash serve.sh e_${t}_r$r strata
done; done; done > serve_r8.txt 2>&1; echo SERVE-DONE
echo Q22-DONE
