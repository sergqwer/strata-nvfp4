#!/bin/bash
# queue 16: QSA kv_post + unrot_gate fusions: identity (fixed costs), serve A/B prev5 / new / Basalt (T~3 and T~8)
H=/d/Projects/strata-dec-int/bench-int; cd $H
I=D:/Projects/strata-dec-int/build-dev/strata.exe; P=D:/Projects/strata-dec-int/bench-int/exe_prev5/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
bash exact8.sh > exact8.txt 2>&1; echo EXACT-DONE
for r in 1 2; do for fx in prose uk struct; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=$fx bash serve.sh k_p_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh k_i_r$r strata
  FIXTURES=$fx bash serve.sh k_B_r$r basalt
done; done > serve_kpost.txt 2>&1; echo SERVE-DONE
echo Q16-DONE
