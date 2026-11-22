#!/bin/bash
# queue 3: nsys of both engines at T=8, stage profiles, then the L2PF / adapt / PDL exactness and A/B (build-c)
H=/d/Projects/strata-dec-int/bench-int
until grep -q SERVE-DONE $H/serve_1.txt; do sleep 10; done
BC=D:/Projects/strata-dec-int/build-c/strata.exe; F=D:/Projects/basalt-work/fixtures
MTP=(--mtp D:/Projects/Strata-data/mtp/rt-tuned --mtp-head proxy --mtp-cum 0.43:0.31 --spec 8)
PF_EXE=$BC PF_DRAFT=mtp PF_ENV="STRATA_DECODE_TIMING=1" bash $H/nsys_dec.sh int_struct strata $F/struct.txt 1500 "${MTP[@]}"
bash $H/nsys_dec.sh basalt_struct basalt $F/struct.txt 1500
echo NSYS-DONE
ROUNDS=1 bash $H/ab.sh prof "P|$BC|STRATA_VERIFY_PROFILE=1| BP|||--profile,--wait-stats" "conv struct"
echo PROF-DONE
for e in STRATA_L2PF=64 STRATA_MTP_PDL=1 STRATA_DF_PDL=1; do
  env $e bash $H/exact.sh ie_${e%%=*}_P8_struct M $BC $F/struct.txt 2000 "${MTP[@]}" >> $H/exact_b.txt
done
bash $H/ab.sh l2 "A|$BC|| L64|$BC|STRATA_L2PF=64| G0|$BC||--adapt-min-gain,0 MP|$BC|STRATA_MTP_PDL=1| DP|$BC|STRATA_DF_PDL=1|" "struct conv"
echo Q3-DONE
