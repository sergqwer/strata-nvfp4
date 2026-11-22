#!/bin/bash
# round 7: STRATA_ATTN_STAGED=1 (staged INT8 chunk kernel) must leave every output unchanged (fixed draft costs); + long prompts
export STRATA_SPEC_FIXED=1
X=/d/Projects/strata-dec-int/bench-int/exact.sh
I=D:/Projects/strata-dec-int/build-dev/strata.exe
T=D:/Projects/Strata-data/mtp/rt-tuned; F=D:/Projects/basalt-work/fixtures
MTP=(--mtp $T --mtp-head proxy --mtp-cum auto --spec 8)
SMP=(--temperature 1.0 --top-p 0.95 --top-k 20 --seed 42)
for fx in conv struct; do
  n=400; [ $fx = struct ] && n=2000
  STRATA_ATTN_STAGED=1 bash $X n12_st_D_$fx D $I $F/$fx.txt $n
  STRATA_ATTN_STAGED=1 bash $X n12_st_N_$fx N $I $F/$fx.txt $n
  STRATA_ATTN_STAGED=1 bash $X n12_st_P8_$fx M $I $F/$fx.txt $n "${MTP[@]}"
  STRATA_ATTN_STAGED=1 bash $X n12_st_M_$fx M $I $F/$fx.txt $n
done
STRATA_ATTN_STAGED=1 bash $X n12_st_Ds_prose D $I $F/prose.txt 600 "${SMP[@]}"
STRATA_ATTN_STAGED=1 bash $X n12_st_P8s_prose M $I $F/prose.txt 600 "${SMP[@]}" "${MTP[@]}"
for L in 16k 64k; do for a in 0 1; do
  STRATA_ATTN_STAGED=$a bash $X n12_long${L}_s$a M $I $F/prefill_$L.txt 400
done; done
echo EXACT12-DONE
