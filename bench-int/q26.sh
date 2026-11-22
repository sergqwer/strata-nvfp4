#!/bin/bash
# queue 26: the full set once for rounds 7-9 defaults (staged attention, layout c48+prio, gdn_mid, DF_PDL)
H=/d/Projects/strata-dec-int/bench-int; cd $H
L=/d/Projects/Strata-data/bench-tools/gpu_lock.sh
I=D:/Projects/strata-dec-int/build-dev/strata.exe; P=D:/Projects/strata-dec-int/bench-int/exe_r6/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
bash $L int-test bash -c "cd ../build-dev; ./nv_defaults_test.exe | tail -1; ./qsa_post_parity.exe | tail -1; ./qsa_merge_daily_parity.exe | tail -1; ./gdn_parity.exe | grep -E 'bitwise|NO' ; ./q80p_parity.exe --selftest | tail -1; ./draft_proxy_parity.exe --selftest | tail -1; ./gr_hc8_check.exe --selftest | tail -1; ./gdn_rec_parity.exe | tail -1; ./fp4g_parity.exe | tail -1; ./q81_div_parity.exe | tail -1" > tests26.txt 2>&1; echo TESTS-DONE
bash exact17.sh > exact17.txt 2>&1; echo EXACT-DONE
for r in 1 2; do for fx in struct conv prose uk; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=$fx bash serve.sh s9_p_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh s9_i_r$r strata
  FIXTURES=$fx bash serve.sh s9_B_r$r basalt
done
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=struct bash serve.sh s9v_p_r$r strata --vram-reserve-mib 16304
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=struct bash serve.sh s9v_i_r$r strata --vram-reserve-mib 16304
done > serve_r9.txt 2>&1; echo SERVE-DONE
echo Q26-DONE
