#!/bin/bash
# the integration's exactness set (fixed placement): greedy ids base vs int across drafters, one seeded sampled run
X=/d/Projects/strata-dec-int/bench-int/exact.sh
B=D:/Projects/strata-dec/build-dev/strata.exe; I=D:/Projects/strata-dec-int/build-dev/strata.exe
T=D:/Projects/Strata-data/mtp/rt-tuned; F=D:/Projects/basalt-work/fixtures
MTP=(--mtp $T --mtp-head proxy --mtp-cum 0.43:0.31 --spec 8)
SMP=(--temperature 1.0 --top-p 0.95 --top-k 20 --seed 42)
for fx in conv struct; do
  n=400; [ $fx = struct ] && n=2000
  bash $X ie_base_D_$fx D $B $F/$fx.txt $n
  bash $X ie_int_D_$fx D $I $F/$fx.txt $n
  bash $X ie_int_N_$fx N $I $F/$fx.txt $n
  bash $X ie_int_P8_$fx M $I $F/$fx.txt $n "${MTP[@]}"
  STRATA_HEAD_PLANES=0 bash $X ie_int_P8np_$fx M $I $F/$fx.txt $n "${MTP[@]}"
done
bash $X ie_base_Ds_prose D $B $F/prose.txt 600 "${SMP[@]}"
bash $X ie_int_Ds_prose D $I $F/prose.txt 600 "${SMP[@]}"
bash $X ie_base_Ms_prose M $B $F/prose.txt 600 "${SMP[@]}"
bash $X ie_int_Ms_prose M $I $F/prose.txt 600 "${SMP[@]}"
echo EXACT-DONE
