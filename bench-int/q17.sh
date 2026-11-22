#!/bin/bash
# queue 17: attention round 1 (kv_rot_append, gate_q8_1, q proj halves) + drafter round 2: identity, nsys front, serve A/B
H=/d/Projects/strata-dec-int/bench-int; cd $H
I=D:/Projects/strata-dec-int/build-dev/strata.exe; P=D:/Projects/strata-dec-int/bench-int/exe_prev5/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
bash exact9.sh > exact9.txt 2>&1; echo EXACT-DONE
PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" bash nsys_dec.sh i4_prose strata D:/Projects/basalt-work/fixtures/prose.txt 600 > nsys/i4_prose.out 2>&1
python /d/Projects/strata-dec-host/bench-host/nsys_front_layer.py nsys/i4_prose.sqlite strata > front_i4.txt 2>&1; echo NSYS-DONE
for r in 1 2; do for fx in struct conv prose uk; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=$fx bash serve.sh a_p_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh a_i_r$r strata
  FIXTURES=$fx bash serve.sh a_B_r$r basalt
done; done > serve_att1.txt 2>&1; echo SERVE-DONE
echo Q17-DONE
