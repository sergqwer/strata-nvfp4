#!/bin/bash
cd /d/Projects/strata-multiarch/bench-ma
LK=/d/Projects/Strata-data/bench-tools/gpu_lock.sh
REL=D:/Projects/strata-rel0215/build-release/strata.exe
B=D:/Projects/strata-multiarch
until grep -q "CHECK-DONE" results.txt; do sleep 10; done
echo "# $(date +%T) e75 with STRATA_HEAD_PLANES=0 (the release, the planes probe kept off by hand)" >> results.txt
bash ma_test.sh all e75hp $B/build-ptx75/strata.exe STRATA_EMULATE_CC=75 STRATA_HEAD_PLANES=0
until [ $(grep -c "^rc" build_patched.log) -ge 2 ]; do sleep 10; done
bash $LK ma-build bash build_patched.sh 75 120 > build_patched2.log 2>&1
P86="STRATA_ATTN_STAGED=1 STRATA_PF_GROUP_SCRATCH=1 STRATA_PF_HC_REGION=2 STRATA_KV_PREFETCH=1 STRATA_PF_GRAPH=7 STRATA_GRAPH_PRIO=1 STRATA_ROUTE_DB=2 STRATA_HC_Q8=1 STRATA_PREFILL_EQUAL=2 STRATA_PF_HC_INJ=1 STRATA_GDN_TC=1 STRATA_GDN_CONVL2=2 STRATA_GDN_TC_NP=1 STRATA_QSA_I8=1 STRATA_DENSE_F16C=1"
echo "# $(date +%T) 5090 with the sm_86 portable set by hand (STRATA_NVFP4_SM120=0, FP4G=0): the on/off distance on sm_120" >> results.txt
DF_TIMEOUT=600 bash ma_test.sh pf rel_p86 $REL STRATA_NVFP4_SM120=0 STRATA_PF_FP4G=0 $P86
DF_TIMEOUT=600 bash ma_test.sh mtp rel_def $REL
echo "# $(date +%T) patched sm_120 build, defaults: must be the release's bits" >> results.txt
DF_TIMEOUT=600 bash ma_test.sh pf p120 $B/build-rel120/strata.exe
DF_TIMEOUT=600 bash ma_test.sh mtp p120 $B/build-rel120/strata.exe
for a in 86 89 75; do bash run_cw.sh $a; done
echo "# $(date +%T) NEXT-DONE" >> results.txt
