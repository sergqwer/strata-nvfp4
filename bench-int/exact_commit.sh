#!/bin/bash
# pre-commit identity: a subset of exact17 on the rebuilt dec/int exe (sources = dec/front patch merged)
export STRATA_SPEC_FIXED=1
X=/d/Projects/strata-dec-int/bench-int/exact.sh
I=D:/Projects/strata-dec-int/build-dev/strata.exe
T=D:/Projects/Strata-data/mtp/rt-tuned; F=D:/Projects/basalt-work/fixtures
MTP=(--mtp $T --mtp-head proxy --mtp-cum auto --spec 8)
SMP=(--temperature 1.0 --top-p 0.95 --top-k 20 --seed 42)
bash $X c_D_conv D $I $F/conv.txt 400
bash $X c_P8_conv M $I $F/conv.txt 400 "${MTP[@]}"
STRATA_NVFP4_SM120=0 bash $X c_off_D_conv D $I $F/conv.txt 400
bash $X c_P8_struct M $I $F/struct.txt 2000 "${MTP[@]}"
bash $X c_Ds_prose D $I $F/prose.txt 600 "${SMP[@]}"
echo EXACT-COMMIT-DONE
