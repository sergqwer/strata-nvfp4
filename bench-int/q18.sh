#!/bin/bash
# queue 18: gate rows after the K/V append + merge with the inverse rotation: quick identity, nsys front, serve A/B
H=/d/Projects/strata-dec-int/bench-int; cd $H
I=D:/Projects/strata-dec-int/build-dev/strata.exe; A=D:/Projects/strata-dec-int/bench-int/exe_prev5/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"; F=D:/Projects/basalt-work/fixtures
MTP=(--mtp D:/Projects/Strata-data/mtp/rt-tuned --mtp-head proxy --mtp-cum auto --spec 8)
SMP=(--temperature 1.0 --top-p 0.95 --top-k 20 --seed 42)
export STRATA_SPEC_FIXED=1
bash exact.sh n10_D_conv D $I $F/conv.txt 400 > exact10.txt
bash exact.sh n10_P8_conv M $I $F/conv.txt 400 "${MTP[@]}" >> exact10.txt
bash exact.sh n10_Ds_prose D $I $F/prose.txt 600 "${SMP[@]}" >> exact10.txt
STRATA_QPROJ_HALF=0 bash exact.sh n10_qh0_Ds_prose D $I $F/prose.txt 600 "${SMP[@]}" >> exact10.txt
unset STRATA_SPEC_FIXED
echo EXACT-DONE
PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" bash nsys_dec.sh i5_prose strata $F/prose.txt 600 > nsys/i5_prose.out 2>&1
python /d/Projects/strata-dec-host/bench-host/nsys_front_layer.py nsys/i5_prose.sqlite strata > front_i5.txt 2>&1
python nsys_front_list.py nsys/i5_prose.sqlite 15 0 > front_i5_list.txt 2>&1; echo NSYS-DONE
for r in 1 2; do for fx in prose uk struct; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$A PF_ENV="$TD" FIXTURES=$fx bash serve.sh b_p_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh b_i_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD STRATA_QPROJ_HALF=0" FIXTURES=$fx bash serve.sh b_q0_r$r strata
done; done > serve_att2.txt 2>&1; echo SERVE-DONE
echo Q18-DONE
