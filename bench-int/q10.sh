#!/bin/bash
H=/d/Projects/strata-dec-int/bench-int; cd $H
until grep -q Q9-DONE q9.log; do sleep 20; done
L=/d/Projects/Strata-data/bench-tools/gpu_lock.sh; PY=C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/python.exe
I=D:/Projects/strata-dec-int/build-dev/strata.exe; CI=$H/pf_client_int.py; W=/d/Projects/basalt-work
FIX="STRATA_PREFILL_CPU_SHARE=0 STRATA_ADAPT_WAIT=1 STRATA_PCIE_BALANCE=0"
for tag in e0c e1c e0d e1d e0e e1e; do
  e=${tag:1:1}
  PF_GEN=2000 PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$FIX STRATA_ADAPT_EARLY=$e STRATA_ADAPT_LOG=D:/Projects/strata-dec-int/bench-int/swaps/$tag.txt" PF_LOG=$H/swaps/$tag.log \
    bash $L int-serve $PY $CI strata $W/fixtures/struct.txt --pcie-frac 0 --expert-cache 8000 > swaps/$tag.client 2>&1
  echo "$tag $(wc -l < swaps/$tag.txt) rounds, $(md5sum < swaps/$tag.txt | cut -c1-10) | $(tail -1 swaps/$tag.client | cut -c1-60)"
done > swaps2.txt 2>&1; echo Q10-DONE >> swaps2.txt
