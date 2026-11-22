#!/bin/bash
# ident.sh TAG EXE [ENV...] - decode round 3 identity set (rel04/ident.sh's method: exact.sh = CLI fixed placement, arm N,
# 48 greedy tokens, ctx 140000, STRATA_SPEC_FIXED=1, md5 of the text + of the first-token logits; then exact_commit.sh's
# decode set).  exact.sh takes the GPU lock itself: never run this inside gpu_lock.sh.  ENV: STRATA_* for every run.
# Reference (plan): def16 93c4688100 / e9d907e164, def64 a13d259b7f / 5b1479d38d; conv D + P8 26c5e96e8e, off 293b92d3b9,
# struct 30b9f645ca, prose 10df564211.  Out: bench-d3/ident_TAG.txt
source /d/Projects/Strata-data/rel04/common.sh
source /d/Projects/strata-dec3/bench-d3/gpu_free.sh   # gpu_free without the zombie tray PID
D=/d/Projects/strata-dec3/bench-d3; tag=$1; EXE=$2; shift 2; XENV="$*"
[ -x "$EXE" ] || { echo "usage: ident.sh TAG EXE [ENV...]" >&2; exit 2; }
OUT=$D/ident_$tag.txt; mkdir -p $D/log
X=/d/Projects/strata-dec-int/bench-int/exact.sh; W=D:/Projects/basalt-work/fixtures
raw=/d/Projects/Strata-data/dflash-check/raw
echo "# ident $tag $(date '+%F %T') $EXE [$XENV]" | tee -a $OUT
pw | tee -a $OUT
idr() {  # idr NAME FIXTURE
  local b=D:/Projects/strata-dec3/bench-d3/log/fl_${tag}_$1.bin; rm -f $b
  gpu_free || return 1
  line=$(env $XENV STRATA_SPEC_FIXED=1 STRATA_DUMP_FIRST_LOGITS=$b bash $X d3_${tag}_$1 N $EXE $W/$2.txt 48 --max-context 140000)
  echo "$1 | ${line##*|} | logits $(md5sum < $b 2>/dev/null | cut -c1-10) | $(grep -a -o 'prefill  *[0-9]* tokens in [0-9.]* ms' $raw/d3_${tag}_$1.log | tail -1)" | tee -a $OUT
}
idr def16 prefill_16k
idr def64 prefill_64k
# as rel04: the decode set with the prefill switches of 0.2.15-nvfp4.4 =0, the drafter from Strata-data/mtp
OFF="STRATA_QSA_I8=0 STRATA_DENSE_F16C=0 STRATA_KV_PREFETCH=0 STRATA_PF_HC_INJ=0 STRATA_PF_DM16H=0 STRATA_PF_SHARED_LATE=0 STRATA_PF_KV_UNSTAGE_SIDE=0 STRATA_PF_UPMIX_SKIP32=0 STRATA_PF_REFILL_BG=0 STRATA_GEMM_WARM_FILE=0 STRATA_PF_GRAPH=0"
T=D:/Projects/Strata-data/mtp/rt-tuned; F=$W
MTP=(--mtp $T --mtp-head proxy --mtp-cum auto --spec 8)
SMP=(--temperature 1.0 --top-p 0.95 --top-k 20 --seed 42)
dc() { gpu_free || return 1; env $OFF $XENV STRATA_SPEC_FIXED=1 bash $X "$@" | tee -a $OUT; }
dc d3_${tag}_D_conv D $EXE $F/conv.txt 400
dc d3_${tag}_P8_conv M $EXE $F/conv.txt 400 "${MTP[@]}"
STRATA_NVFP4_SM120=0 dc d3_${tag}_off_D_conv D $EXE $F/conv.txt 400
dc d3_${tag}_P8_struct M $EXE $F/struct.txt 2000 "${MTP[@]}"
dc d3_${tag}_Ds_prose D $EXE $F/prose.txt 600 "${SMP[@]}"
pw | tee -a $OUT
echo IDENT-DONE | tee -a $OUT
