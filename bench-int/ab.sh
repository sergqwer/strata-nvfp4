#!/bin/bash
# ab.sh ROUND "ARM1 ARM2 ..." [fixtures] - serve A/B, interleaved per fixture and round.  ARM = NAME|EXE|ENV,..|FLAG,..
# (MTP8 flags on every arm; NAME starting with D = DFlash2 instead).  Lines -> bench-int/ab_ROUND.txt
R=$1; ARMS=$2; FX=${3:-struct conv}; H=/d/Projects/strata-dec-int/bench-int; OUT=$H/ab_$R.txt
T=D:/Projects/Strata-data/mtp/rt-tuned
MTP="--mtp $T --mtp-head proxy --mtp-cum 0.43:0.31 --spec 8"
for rr in ${ROUNDS:-1 2}; do
for fx in $FX; do
  for a in $ARMS; do
    IFS='|' read -r name exe envs flags <<< "$a"; envs=${envs//,/ }; flags=${flags//,/ }
    if [ ${name:0:1} = B ]; then FIXTURES=$fx bash $H/serve.sh ${R}_${name}_r$rr basalt $flags
    elif [ ${name:0:1} = D ]; then PF_EXE=$exe PF_ENV="STRATA_DECODE_TIMING=1 $envs" FIXTURES=$fx bash $H/serve.sh ${R}_${name}_r$rr strata $flags
    else PF_EXE=$exe PF_DRAFT=mtp PF_ENV="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1 $envs" FIXTURES=$fx bash $H/serve.sh ${R}_${name}_r$rr strata $MTP $flags; fi
  done
done
done 2>&1 | tee -a $OUT
echo AB-DONE >> $OUT
