#!/bin/bash
# serve_all.sh ROUND - part A's serve A/B, interleaved per fixture: base+DFlash2, int+DFlash2, int+MTP8, int+MTP8+hc8, Basalt
R=$1; H=/d/Projects/strata-dec-int/bench-int; OUT=$H/serve_$R.txt
BASE=D:/Projects/strata-dec/build-dev/strata.exe; INT=D:/Projects/strata-dec-int/build-dev/strata.exe; T=D:/Projects/Strata-data/mtp/rt-tuned
ENVS="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
MTP="--mtp $T --mtp-head proxy --mtp-cum 0.43:0.31 --spec 8"
for fx in ${FIXTURES:-conv struct prose uk}; do
  for arm in ${ARMS:-bDF2 iDF2 iMTP8 iMTP8hc B8}; do
    case $arm in
      bDF2)  PF_EXE=$BASE PF_ENV="$ENVS" FIXTURES=$fx bash $H/serve.sh ${R}_bDF2 strata ;;
      iDF2)  PF_EXE=$INT PF_ENV="$ENVS" FIXTURES=$fx bash $H/serve.sh ${R}_iDF2 strata ;;
      iMTP8) PF_EXE=$INT PF_DRAFT=mtp PF_ENV="$ENVS" FIXTURES=$fx bash $H/serve.sh ${R}_iMTP8 strata $MTP ;;
      iMTP8hc) PF_EXE=$INT PF_DRAFT=mtp PF_ENV="$ENVS STRATA_HC_Q8=1" FIXTURES=$fx bash $H/serve.sh ${R}_iMTP8hc strata $MTP ;;
      B8)    FIXTURES=$fx bash $H/serve.sh ${R}_B8 basalt ;;
    esac
  done
done 2>&1 | tee -a $OUT
echo SERVE-DONE >> $OUT
