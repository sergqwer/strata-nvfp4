#!/bin/bash
H=/d/Projects/strata-dec-int/bench-int
until grep -q Q1-DONE $H/q1.log; do sleep 10; done
BC=D:/Projects/strata-dec-int/build-c/strata.exe; F=D:/Projects/basalt-work/fixtures
MTP=(--mtp D:/Projects/Strata-data/mtp/rt-tuned --mtp-head proxy --mtp-cum 0.43:0.31 --spec 8)
STRATA_MTP_PDL=1 bash $H/exact.sh ie_mpdl_P8_struct M $BC $F/struct.txt 2000 "${MTP[@]}" >> $H/exact_l2pf.txt
STRATA_DF_PDL=1 bash $H/exact.sh ie_dpdl_P8_struct M $BC $F/struct.txt 2000 "${MTP[@]}" >> $H/exact_l2pf.txt
bash $H/ab.sh pdl "A|$BC|| MP|$BC|STRATA_MTP_PDL=1| DP|$BC|STRATA_DF_PDL=1|" "struct conv"
PF_EXE=$BC PF_DRAFT=mtp PF_ENV="STRATA_DECODE_TIMING=1" bash $H/nsys_dec.sh int_struct strata $F/struct.txt 1500 "${MTP[@]}"
bash $H/nsys_dec.sh basalt_struct basalt $F/struct.txt 1500
echo Q2-DONE
