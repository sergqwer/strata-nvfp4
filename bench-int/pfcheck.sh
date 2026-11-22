#!/bin/bash
# GROUP_SCRATCH / HC_REGION on vs off (bitwise, reproducible mode), FP4G engaged or not, at 16K and 64K
X=/d/Projects/strata-dec-int/bench-int/exact.sh; I=D:/Projects/strata-dec-int/build-dev/strata.exe; F=D:/Projects/basalt-work/fixtures
for fx in prefill_16k prefill_64k; do
  bash $X pc_on_$fx N $I $F/$fx.txt 48 --max-context 70000
  STRATA_PF_GROUP_SCRATCH=0 STRATA_PF_HC_REGION=0 bash $X pc_off_$fx N $I $F/$fx.txt 48 --max-context 70000
  STRATA_PF_FP4G=0 bash $X pc_mmq_$fx N $I $F/$fx.txt 48 --max-context 70000
  STRATA_PF_FP4G=0 STRATA_PF_GROUP_SCRATCH=0 STRATA_PF_HC_REGION=0 bash $X pc_mmqoff_$fx N $I $F/$fx.txt 48 --max-context 70000
done
echo PC-DONE
