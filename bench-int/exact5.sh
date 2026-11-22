#!/bin/bash
# exact4 with fixed draft costs (STRATA_SPEC_FIXED=1, NOTES-host R2.6); exactness on the new defaults (sm_120 table: hc8, GDN TC/CONVL2/NP, equal chunks; CSPLIT): drafters, head, overlaps
# and CSPLIT must not change the int build's output; int with STRATA_NVFP4_SM120=0 STRATA_MMVQ_CSPLIT=0 must equal base
export STRATA_SPEC_FIXED=1
X=/d/Projects/strata-dec-int/bench-int/exact.sh
B=D:/Projects/strata-dec/build-dev/strata.exe; I=D:/Projects/strata-dec-int/build-dev/strata.exe
T=D:/Projects/Strata-data/mtp/rt-tuned; F=D:/Projects/basalt-work/fixtures
MTP=(--mtp $T --mtp-head proxy --mtp-cum auto --spec 8)
SMP=(--temperature 1.0 --top-p 0.95 --top-k 20 --seed 42)
for fx in conv struct; do
  n=400; [ $fx = struct ] && n=2000
  bash $X n5_int_D_$fx D $I $F/$fx.txt $n
  bash $X n5_int_N_$fx N $I $F/$fx.txt $n
  bash $X n5_int_P8_$fx M $I $F/$fx.txt $n "${MTP[@]}"
  STRATA_HEAD_PLANES=0 bash $X n5_int_P8np_$fx M $I $F/$fx.txt $n "${MTP[@]}"
  STRATA_MTP_OVERLAP=0 bash $X n5_int_P8ov0_$fx M $I $F/$fx.txt $n "${MTP[@]}"
  STRATA_MMVQ_CSPLIT=1 bash $X n5_int_P8cs1_$fx M $I $F/$fx.txt $n "${MTP[@]}"
  bash $X n5_int_M_$fx M $I $F/$fx.txt $n
  STRATA_NVFP4_SM120=0 bash $X n5_int_off_D_$fx D $I $F/$fx.txt $n
done
bash $X n5_int_Ds_prose D $I $F/prose.txt 600 "${SMP[@]}"
bash $X n5_int_Ms_prose M $I $F/prose.txt 600 "${SMP[@]}"
bash $X n5_int_P8s_prose M $I $F/prose.txt 600 "${SMP[@]}" "${MTP[@]}"
echo EXACT2-DONE
