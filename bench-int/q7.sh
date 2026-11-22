#!/bin/bash
H=/d/Projects/strata-dec-int/bench-int; cd $H
L=/d/Projects/Strata-data/bench-tools/gpu_lock.sh
I=D:/Projects/strata-dec-int/build-dev/strata.exe; P=D:/Projects/strata-dec-int/bench-int/exe_prev/strata.exe; CI=$H/pf_client_int.py
bash $L build-int cmd //c "D:\Projects\Strata-data\bench-tools\build_dev.cmd" "D:\Projects\strata-dec-int" strata nv_defaults_test gdn_rec_parity q80p_parity draft_proxy_parity gr_hc8_check fp4g_parity b6_mmvq_bench > build_dev8.log 2>&1 || { echo BUILD-FAILED; exit 1; }
echo BUILD-OK
bash $L int-test bash -c "cd ../build-dev; ./nv_defaults_test.exe | tail -1; ./q80p_parity.exe --selftest | tail -1; ./draft_proxy_parity.exe --selftest | tail -1; ./gr_hc8_check.exe --selftest | tail -1; ./gdn_rec_parity.exe | tail -1; ./fp4g_parity.exe | tail -3" > tests8.txt 2>&1; echo TESTS-DONE
bash exact3.sh > exact3.txt 2>&1; echo EXACT3-DONE
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
for r in 1 2; do for fx in struct conv prose; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=$fx bash serve.sh d7_prev_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh d7_new_r$r strata
done; done > serve_d7.txt 2>&1; echo SERVE-DONE
echo Q7-DONE
