#!/bin/bash
H=/d/Projects/strata-dec-int/bench-int; cd $H
L=/d/Projects/Strata-data/bench-tools/gpu_lock.sh; PY=C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/python.exe
I=D:/Projects/strata-dec-int/build-dev/strata.exe; P=D:/Projects/strata-dec-int/bench-int/exe_prev2/strata.exe; CI=$H/pf_client_int.py; W=/d/Projects/basalt-work
bash $L build-int cmd //c "D:\Projects\Strata-data\bench-tools\build_dev.cmd" "D:\Projects\strata-dec-int" strata nv_defaults_test gdn_rec_parity q80p_parity draft_proxy_parity gr_hc8_check fp4g_parity > build_dev11.log 2>&1 || { echo BUILD-FAILED; exit 1; }
echo BUILD-OK
bash $L int-test bash -c "cd ../build-dev; ./nv_defaults_test.exe | tail -1; ./q80p_parity.exe --selftest | tail -1; ./draft_proxy_parity.exe --selftest | tail -1; ./gr_hc8_check.exe --selftest | tail -1; ./gdn_rec_parity.exe | tail -1; ./fp4g_parity.exe | tail -3" > tests11.txt 2>&1; echo TESTS-DONE
bash exact4.sh > exact4.txt 2>&1; echo EXACT-DONE
FIX="STRATA_PREFILL_CPU_SHARE=0 STRATA_ADAPT_WAIT=1 STRATA_PCIE_BALANCE=0"
for arm in "e1a|STRATA_ADAPT_EARLY=1" "e0a|STRATA_ADAPT_EARLY=0" "e1b|STRATA_ADAPT_EARLY=1" "e0b|STRATA_ADAPT_EARLY=0"; do
  IFS='|' read -r tag envs <<< "$arm"
  PF_GEN=2000 PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$FIX $envs STRATA_ADAPT_LOG=D:/Projects/strata-dec-int/bench-int/swaps/$tag.txt" PF_LOG=$H/swaps/$tag.log \
    bash $L int-serve $PY $CI strata $W/fixtures/struct.txt --pcie-frac 0 --expert-cache 8000 > swaps/$tag.client 2>&1
  echo "$tag $(wc -l < swaps/$tag.txt) rounds, $(md5sum < swaps/$tag.txt | cut -c1-10) | $(tail -1 swaps/$tag.client | cut -c1-60)"
done > swaps.txt 2>&1; echo SWAPS-DONE
for arm in "s1a|STRATA_ADAPT_EARLY=1" "s1b|STRATA_ADAPT_EARLY=1" "s0a|STRATA_ADAPT_EARLY=0"; do
  IFS='|' read -r tag envs <<< "$arm"
  ST_TAG=$tag ST_LOG=$H/swaps PF_EXE=$I PF_ENV="$FIX $envs" bash $L int-serve $PY $H/stop_test.py
done > stoptest.txt 2>&1; echo STOP-DONE
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
for r in 1 2; do for fx in struct prose; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh h9_dma_r$r strata --pcie-mode dma
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD STRATA_FLAGB_DIRECT=0 STRATA_DMA_FLUSH=0" FIXTURES=$fx bash serve.sh h9_dma0_r$r strata --pcie-mode dma
done; done > serve_dma.txt 2>&1; echo DMA-DONE
for r in 1 2; do for fx in struct conv prose uk; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$P PF_ENV="$TD" FIXTURES=$fx bash serve.sh h9_prev_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh h9_new_r$r strata
done; done > serve_h9.txt 2>&1; echo SERVE-DONE
echo Q9-DONE
