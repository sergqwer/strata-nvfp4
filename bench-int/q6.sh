#!/bin/bash
# Part B on the final defaults: PCIe share at wide / narrow windows, then stage profiles (Auto drafting)
H=/d/Projects/strata-dec-int/bench-int; cd $H
I=D:/Projects/strata-dec-int/build-dev/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
for r in 1 2; do for fx in struct prose conv; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh b6_A_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh b6_P0_r$r strata --pcie-frac 0
done; done > q6_pcie.txt 2>&1; echo PCIE-DONE
for fx in struct prose; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD STRATA_VERIFY_PROFILE=1" FIXTURES=$fx bash serve.sh b6_prof strata
  FIXTURES=$fx bash serve.sh b6_profB basalt --profile --wait-stats
done > q6_prof.txt 2>&1; echo PROF-DONE
echo Q6-DONE
