#!/bin/bash
# exact.sh TAG ARM EXE FIXTURE MAXNEW [flags...] - fixed-placement CLI run (no CPU prefill share, no PCIe share, no adaptive
# swaps, no suffix drafts, fixed slot count): greedy text independent of timing.  Log: dflash-check/raw/TAG.log
tag=$1; arm=$2; exe=$3; fx=$4; n=$5; shift 5
cd /d/Projects/Strata-data/dflash-check
DF_EXE=$exe STRATA_PREFILL_CPU_SHARE=0 STRATA_ADAPT_WAIT=1 STRATA_PCIE_BALANCE=0 STRATA_DRAFT_TIMING=1 \
  bash ../bench-tools/gpu_lock.sh int C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/python.exe dfl_run.py $tag $arm $fx $n \
  --stop-eos --max-context 66661 --expert-cache 8000 --pcie-frac 0 --adapt-every 100000 --suffix-draft 0 "$@" > /dev/null
L=raw/$tag.log
sp=$(grep -a "speculation" $L | tail -1 | sed -E 's/^ *[0-9.]+ +//' | cut -c1-120)
hd=$(grep -a "two-stage head:" $L | tail -1 | sed -E 's/.*two-stage head: //' | cut -c1-90)
dec=$(grep -a "^ *[0-9.]*  decode  " $L | grep -a -o -E "[0-9]+ tokens in [0-9.]+ ms  ->  [0-9.]+ tok/s")
md5=$(grep -a 'output  :' $L | sed 's/^ *[0-9.]* *//' | md5sum | cut -c1-10)
echo "$tag | $sp | $hd | $dec | out $md5"
