#!/bin/bash
# rep.sh TAG EXE CTX WIN [ENV...] - E5: CLI fixed placement (exact.sh arm M + the tray MTP flags), long16_prose, 400 tokens,
# --mtp-window WIN (0 = not passed). Out line -> bench-d4/e5/rep.txt
source /d/Projects/Strata-data/rel04/common.sh
source /d/Projects/strata-dec3/bench-d3/gpu_free.sh
tag=$1; EXE=$2; CTX=$3; WIN=$4; shift 4; XENV="$*"
X=/d/Projects/strata-dec-int/bench-int/exact.sh; W=D:/Projects/basalt-work/fixtures
MTP=(--mtp D:/Projects/Strata-data/mtp/rt-tuned --mtp-head proxy --mtp-cum auto --spec 8)
[ "$WIN" != 0 ] && MTP+=(--mtp-window $WIN)
gpu_free || exit 1
line=$(env $XENV STRATA_SPEC_FIXED=1 bash $X e5_$tag M $EXE $W/${FX:-long16_prose}.txt ${GEN:-400} "${MTP[@]}" --max-context $CTX)
dk=$(grep -a -o "draft kv: .*" /d/Projects/Strata-data/dflash-check/raw/e5_$tag.log | head -1 | cut -c1-90)
echo "$tag | ctx $CTX win $WIN [$XENV] | $line | $dk" | tee -a /d/Projects/strata-dec3/bench-d4/e5/rep.txt
