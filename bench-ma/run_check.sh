#!/bin/bash
cd /d/Projects/strata-multiarch/bench-ma
REL=D:/Projects/strata-rel0215/build-release/strata.exe
B=D:/Projects/strata-multiarch
echo "# $(date +%T) release check" >> results.txt
DF_TIMEOUT=600 bash ma_test.sh pf rel_def $REL
DF_TIMEOUT=600 bash ma_test.sh all rel_off $REL STRATA_NVFP4_SM120=0
DF_TIMEOUT=600 bash ma_test.sh pf rel_offfp4 $REL STRATA_NVFP4_SM120=0 STRATA_PF_FP4G=0
for a in 86 89 75; do
  bash ma_test.sh all e$a $B/build-ptx$a/strata.exe STRATA_EMULATE_CC=$a
done
echo "# $(date +%T) CHECK-DONE" >> results.txt
