#!/bin/bash
# queue 15: host round 3's dense slab merged: build, tests, identity (fixed costs), slab trace, serve A/B prev4 / new / Basalt
H=/d/Projects/strata-dec-int/bench-int; cd $H
L=/d/Projects/Strata-data/bench-tools/gpu_lock.sh; PY=C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/python.exe
I=D:/Projects/strata-dec-int/build-dev/strata.exe; P=D:/Projects/strata-dec-int/bench-int/exe_prev4/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
bash $L build-int cmd //c "D:\Projects\Strata-data\bench-tools\build_dev.cmd" "D:\Projects\strata-dec-int" strata nv_defaults_test gdn_rec_parity q80p_parity draft_proxy_parity gr_hc8_check fp4g_parity > build_dev15.log 2>&1 || { echo BUILD-FAILED; exit 1; }
echo BUILD-OK
bash $L int-test bash -c "cd ../build-dev; ./nv_defaults_test.exe | tail -1; ./q80p_parity.exe --selftest | tail -1; ./draft_proxy_parity.exe --selftest | tail -1; ./gr_hc8_check.exe --selftest | tail -1; ./gdn_rec_parity.exe | tail -1; ./fp4g_parity.exe | tail -3" > tests15.txt 2>&1; echo TESTS-DONE
bash exact7.sh > exact7.txt 2>&1; echo EXACT-DONE
for e in p i; do X=$I; [ $e = p ] && X=$P
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$X PF_ENV="$TD STRATA_TRACE=1" FIXTURES=prose bash serve.sh s_tr_$e strata > /dev/null 2>&1
  echo "$e: $(grep -a -h 'native dense slab\|expert cache [0-9]* slots' serve/strata_s_tr_${e}_prose.log | cut -c1-200 | tr '\n' ' ')"
done > slab_trace.txt 2>&1; echo TRACE-DONE
for r in 1 2; do for fx in struct conv prose uk; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=$fx bash serve.sh s_p_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh s_i_r$r strata
  FIXTURES=$fx bash serve.sh s_B_r$r basalt
done; done > serve_slab.txt 2>&1; echo SERVE-DONE
echo Q15-DONE
