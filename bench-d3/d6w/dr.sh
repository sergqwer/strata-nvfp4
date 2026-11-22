#!/bin/bash
# dr.sh TAG EXE FX N [ENV...] - D6 draft check: CLI fixed placement (int exact.sh), tray MTP flags, STRATA_MTP_DUMP per run;
# prints the speculation line, draft timing and dump md5.  exact.sh takes the GPU lock itself.
source /d/Projects/strata-dec3/bench-d3/gpu_free.sh
tag=$1; EXE=$2; fx=$3; n=$4; shift 4
X=/d/Projects/strata-dec-int/bench-int/exact.sh; F=D:/Projects/basalt-work/fixtures; RAW=/d/Projects/Strata-data/dflash-check/raw
D=D:/Projects/strata-dec3/bench-d3/d6w; dump=$D/dump_$tag.txt; rm -f $dump
gpu_free || exit 1
line=$(env "$@" STRATA_SPEC_FIXED=1 STRATA_MTP_DUMP=$dump bash $X d6_$tag M $EXE $F/$fx.txt $n --mtp D:/Projects/Strata-data/mtp/rt-tuned --mtp-head proxy --mtp-cum auto --spec 8)
echo "$tag | $line | dump $(md5sum < $dump | cut -c1-10) $(wc -l < $dump)"
grep -a "draft timing\|mtp profile" $RAW/d6_$tag.log | tail -4 | cut -c1-400
