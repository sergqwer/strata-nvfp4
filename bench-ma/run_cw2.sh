#!/bin/bash
# run_cw.sh ARCH - the cheap wins on one emulated arch with the patched PTX engine:
#   x   = the portable set off (= the release), exact switches by hand: must equal the release check's md5s
#   d   = the defaults (portable set on)
cd /d/Projects/strata-multiarch/bench-ma
a=$1; exe=D:/Projects/strata-multiarch/build-ptx$a/strata.exe
EX="STRATA_ATTN_STAGED=1 STRATA_PF_GROUP_SCRATCH=1 STRATA_PF_HC_REGION=2 STRATA_KV_PREFETCH=1 STRATA_PF_GRAPH=7 STRATA_GRAPH_PRIO=1 STRATA_ROUTE_DB=2"
[ $a = 75 ] && EX="$EX STRATA_SM75_PERF=1"
echo "# $(date +%T) cheap wins e$a" >> results.txt
bash ma_test2.sh pf  p${a}x $exe STRATA_EMULATE_CC=$a STRATA_NVFP4_PORTABLE=0 $EX
bash ma_test2.sh mtp p${a}x $exe STRATA_EMULATE_CC=$a STRATA_NVFP4_PORTABLE=0 $EX
bash ma_test2.sh all p${a}d $exe STRATA_EMULATE_CC=$a
