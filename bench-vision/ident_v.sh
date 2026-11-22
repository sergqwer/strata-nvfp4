#!/bin/bash
# ident_v.sh EXE TAGP - rel04/ident.sh's def16 case and the decode conv set on EXE (exact.sh, CLI fixed placement, STRATA_SPEC_FIXED=1)
source /d/Projects/strata-vision-gpu/bench-vision/run.sh --defs-only
E=$1; P=$2; X=/d/Projects/strata-dec-int/bench-int/exact.sh; W=D:/Projects/basalt-work/fixtures; raw=/d/Projects/Strata-data/dflash-check/raw
OFF="STRATA_QSA_I8=0 STRATA_DENSE_F16C=0 STRATA_KV_PREFETCH=0 STRATA_PF_HC_INJ=0 STRATA_PF_DM16H=0 STRATA_PF_SHARED_LATE=0 STRATA_PF_KV_UNSTAGE_SIDE=0 STRATA_PF_UPMIX_SKIP32=0 STRATA_PF_REFILL_BG=0 STRATA_GEMM_WARM_FILE=0 STRATA_PF_GRAPH=0"
export STRATA_SPEC_FIXED=1
pw
b=D:/Projects/strata-vision-gpu/bench-vision/out/fl_${P}_def16.bin; rm -f $b
gpu_free2 && line=$(env STRATA_DUMP_FIRST_LOGITS=$b bash $X ${P}_def16 N $E $W/prefill_16k.txt 48 --max-context 140000)
echo "${P}_def16 | ${line##*|} | logits $(md5sum < $b | cut -c1-10)"
T=D:/Projects/Strata-data/mtp/rt-tuned
gpu_free2 && env $OFF bash $X ${P}_c_D_conv D $E $W/conv.txt 400
gpu_free2 && env $OFF bash $X ${P}_c_P8_conv M $E $W/conv.txt 400 --mtp $T --mtp-head proxy --mtp-cum auto --spec 8
pw
