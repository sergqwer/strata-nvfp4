#!/bin/bash
# queue 14: the final defaults build (dense_tc + x3p with the fixed 0.45 share): serve, full card and 16/24 GB, 2 rounds
H=/d/Projects/strata-dec-int/bench-int; cd $H
until grep -q Q13-DONE q13.log; do sleep 20; done
I=D:/Projects/strata-dec-int/build-dev/strata.exe; P=D:/Projects/strata-dec-int/bench-int/exe_prev3/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
for r in 1 2; do for fx in struct conv prose uk; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh z_i_r$r strata
done; done > serve_z.txt 2>&1; echo Z-DONE
for rv in 8144 16304; do for r in 1 2; do for fx in struct conv; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh z_v${rv}_i_r$r strata --vram-reserve-mib $rv
done; done; done > serve_zv.txt 2>&1; echo ZV-DONE
echo Q14-DONE
