#!/bin/bash
cd /d/Projects/strata-multiarch/bench-ma
B=D:/Projects/strata-multiarch; E=$B/build-rel120/strata.exe
echo "# $(date +%T) patched e75 defaults (Turing fix check)" >> results.txt
bash ma_test2.sh pf p75d $B/build-ptx75/strata.exe STRATA_EMULATE_CC=75
bash ma_test2.sh mtp p75d $B/build-ptx75/strata.exe STRATA_EMULATE_CC=75
echo "# $(date +%T) patched sm_120 build, defaults: must be the release's bits" >> results.txt
DF_TIMEOUT=600 bash ma_test2.sh pf p120 $E
DF_TIMEOUT=600 bash ma_test2.sh mtp p120 $E
echo "# $(date +%T) EARLY_READ with 84 SMs (STRATA_EMULATE_SMS=84)" >> results.txt
DF_TIMEOUT=600 bash ma_test2.sh mtp h84force $E STRATA_EMULATE_SMS=84 STRATA_EARLY_READ_FIT=0
DF_TIMEOUT=600 bash ma_test2.sh mtp h84fix $E STRATA_EMULATE_SMS=84
grep -a -h "early read\|EARLY_READ:\|EMULATE_SMS" /d/Projects/Strata-data/dflash-check/raw/ma_h84*_mtp.log | cut -c1-220 >> results.txt
for a in 86 89 75; do bash run_cw2.sh $a; done
echo "# $(date +%T) PRI-DONE" >> results.txt
