#!/bin/bash
# queue 21: round 7 staged attention: nsys front staged 0/1, serve A/B same exe (env) 4 fixtures x 2 rounds
H=/d/Projects/strata-dec-int/bench-int; cd $H
I=D:/Projects/strata-dec-int/build-dev/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"; F=D:/Projects/basalt-work/fixtures
for a in 0 1; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD STRATA_ATTN_STAGED=$a" bash nsys_dec.sh g_s$a strata $F/prose.txt 600 > nsys/g_s$a.out 2>&1
  echo "s$a: $(python /d/Projects/strata-dec-host/bench-host/nsys_front_layer.py nsys/g_s$a.sqlite strata | tail -2 | tr '\n' ' ')"
  python nsys_kern.py nsys/g_s$a.sqlite 2>/dev/null | grep -i "attn_chunk\|attn_merge" | head -4
  rm -f nsys/g_s$a.nsys-rep
done > front_r7.txt 2>&1; echo NSYS-DONE
for r in 1 2; do for fx in struct conv prose uk; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD STRATA_ATTN_STAGED=0" FIXTURES=$fx bash serve.sh d_s0_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD STRATA_ATTN_STAGED=1" FIXTURES=$fx bash serve.sh d_s1_r$r strata
done; done > serve_r7.txt 2>&1; echo SERVE-DONE
echo Q21-DONE
