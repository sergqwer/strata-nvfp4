#!/bin/bash
# queue 20: round 6 final: build, tests, identity (fixed costs), nsys front, serve A/B prev5 / new / Basalt (4 fixtures, 2 rounds)
H=/d/Projects/strata-dec-int/bench-int; cd $H
L=/d/Projects/Strata-data/bench-tools/gpu_lock.sh
I=D:/Projects/strata-dec-int/build-dev/strata.exe; P=D:/Projects/strata-dec-int/bench-int/exe_prev5/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"; F=D:/Projects/basalt-work/fixtures
bash $L build-int cmd //c "D:\Projects\Strata-data\bench-tools\build_dev.cmd" "D:\Projects\strata-dec-int" strata qsa_post_parity nv_defaults_test gdn_rec_parity q80p_parity draft_proxy_parity gr_hc8_check fp4g_parity q81_div_parity > build_dev23.log 2>&1 || { echo BUILD-FAILED; exit 1; }
echo BUILD-OK
bash $L int-test bash -c "cd ../build-dev; ./qsa_post_parity.exe | tail -1; ./nv_defaults_test.exe | tail -1; ./q80p_parity.exe --selftest | tail -1; ./draft_proxy_parity.exe --selftest | tail -1; ./gr_hc8_check.exe --selftest | tail -1; ./gdn_rec_parity.exe | tail -1; ./fp4g_parity.exe | tail -1; ./q81_div_parity.exe" > tests20.txt 2>&1; echo TESTS-DONE
bash exact11.sh > exact11.txt 2>&1; echo EXACT-DONE
for e in i p; do X=$I; [ $e = p ] && X=$P
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$X PF_ENV="$TD" bash nsys_dec.sh f_$e strata $F/prose.txt 600 > nsys/f_$e.out 2>&1
  echo "$e: $(python /d/Projects/strata-dec-host/bench-host/nsys_front_layer.py nsys/f_$e.sqlite strata | tail -2 | tr '\n' ' ')"
  rm -f nsys/f_$e.nsys-rep
done > front_final.txt 2>&1; echo NSYS-DONE
for r in 1 2; do for fx in struct conv prose uk; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=$fx bash serve.sh c_p_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh c_i_r$r strata
  FIXTURES=$fx bash serve.sh c_B_r$r basalt
done; done > serve_r6.txt 2>&1; echo SERVE-DONE
echo Q20-DONE
