#!/bin/bash
# EARLY_READ on a card with fewer SMs (STRATA_EMULATE_SMS=84 = RTX 5080), the patched sm_120 build, Auto's MTP decode
cd /d/Projects/strata-multiarch/bench-ma
until grep -q "NEXT-DONE" results.txt; do sleep 20; done
E=D:/Projects/strata-multiarch/build-rel120/strata.exe
echo "# $(date +%T) EARLY_READ with 84 SMs held free (STRATA_EMULATE_SMS=84, a hog on the other 86)" >> results.txt
DF_TIMEOUT=900 bash ma_test.sh mtp h84force $E STRATA_EMULATE_SMS=84 STRATA_EARLY_READ_FIT=0
DF_TIMEOUT=900 bash ma_test.sh mtp h84fix $E STRATA_EMULATE_SMS=84
DF_TIMEOUT=900 bash ma_test.sh mtp h84off $E STRATA_EMULATE_SMS=84 STRATA_EARLY_READ=0
grep -a -h "early read\|EARLY_READ\|EMULATE_SMS" /d/Projects/Strata-data/dflash-check/raw/ma_h84*_mtp.log | cut -c1-220 >> results.txt
echo "# $(date +%T) HOG-DONE" >> results.txt
