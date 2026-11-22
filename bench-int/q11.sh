#!/bin/bash
# queue 11: host round 2 merged + x3p defaults: build, tests, identity (fixed draft costs), swap determinism, TF arm,
# budgets, uncensored, final A/B (base / prev3 / int / Basalt), dense_tc at T=8
H=/d/Projects/strata-dec-int/bench-int; cd $H
L=/d/Projects/Strata-data/bench-tools/gpu_lock.sh; PY=C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/python.exe
B=D:/Projects/strata-dec/build-dev/strata.exe; I=D:/Projects/strata-dec-int/build-dev/strata.exe; P=D:/Projects/strata-dec-int/bench-int/exe_prev3/strata.exe
CI=$H/pf_client_int.py; W=/d/Projects/basalt-work
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
bash $L build-int cmd //c "D:\Projects\Strata-data\bench-tools\build_dev.cmd" "D:\Projects\strata-dec-int" strata nv_defaults_test gdn_rec_parity q80p_parity draft_proxy_parity gr_hc8_check fp4g_parity > build_dev12.log 2>&1 || { echo BUILD-FAILED; exit 1; }
echo BUILD-OK
bash $L int-test bash -c "cd ../build-dev; ./nv_defaults_test.exe | tail -1; ./q80p_parity.exe --selftest | tail -1; ./draft_proxy_parity.exe --selftest | tail -1; ./gr_hc8_check.exe --selftest | tail -1; ./gdn_rec_parity.exe | tail -1; ./fp4g_parity.exe | tail -3" > tests12.txt 2>&1; echo TESTS-DONE
bash exact5.sh > exact5.txt 2>&1; echo EXACT-DONE
FIX="STRATA_PREFILL_CPU_SHARE=0 STRATA_ADAPT_WAIT=1 STRATA_PCIE_BALANCE=0 STRATA_SPEC_FIXED=1"
for tag in x1 x2 x3; do
  PF_GEN=2000 PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$FIX STRATA_ADAPT_LOG=D:/Projects/strata-dec-int/bench-int/swaps/$tag.txt" PF_LOG=$H/swaps/$tag.log \
    bash $L int-serve $PY $CI strata $W/fixtures/struct.txt --pcie-frac 0 --expert-cache 8000 > swaps/$tag.client 2>&1
  echo "$tag $(wc -l < swaps/$tag.txt) rounds, $(md5sum < swaps/$tag.txt | cut -c1-10) | $(tail -1 swaps/$tag.client | cut -c1-60)"
done > swaps3.txt 2>&1; echo SWAPS-DONE
DF_EXE=$I bash $L int-acc $PY $W/score_strata.py $W/corpus/set.json $H/tf/acc_x3p.tsv > tf/acc_x3p.out 2>&1; echo ACC-DONE
for rv in 8144 16304; do for r in 1 2; do for fx in struct conv; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=$fx bash serve.sh x_v${rv}_p_r$r strata --vram-reserve-mib $rv
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh x_v${rv}_i_r$r strata --vram-reserve-mib $rv
done; done; done > serve_vram2.txt 2>&1; echo VRAM-DONE
for r in 1 2; do for fx in struct conv prose uk; do
  PF_CLIENT=$CI PF_CVEC=1 PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=$fx bash serve.sh x_unc_p_r$r strata
  PF_CLIENT=$CI PF_CVEC=1 PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh x_unc_i_r$r strata
done; done > serve_unc2.txt 2>&1; echo UNC-DONE
for r in 1 2; do for fx in struct conv prose uk; do
  PF_EXE=$B PF_ENV="$TD" FIXTURES=$fx bash serve.sh x_b_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=$fx bash serve.sh x_p_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh x_i_r$r strata
  FIXTURES=$fx bash serve.sh x_B_r$r basalt
done; done > serve_final.txt 2>&1; echo FINAL-DONE
for r in 1 2; do for fx in struct prose conv; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh x_d0_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD STRATA_DENSE_TC=1" FIXTURES=$fx bash serve.sh x_d1_r$r strata
done; done > serve_dtc2.txt 2>&1; echo DTC-DONE
echo Q11-DONE
