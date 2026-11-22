#!/bin/bash
# ma_test.sh SET TAGPFX EXE [ENV...] - the release check on one engine: a 16K prompt read (fixed placement, arm N, 48 tokens,
# first-token logits md5), a 200-token decode with Auto's new MTP (proxy head, cum auto, spec 8) and one with DFlash2.
# The env (e.g. STRATA_EMULATE_CC=86) is passed to the engine.  Results -> bench-ma/results.txt
source /d/Projects/Strata-data/rel04/common.sh
X=/d/Projects/strata-dec-int/bench-int/exact.sh; W=D:/Projects/basalt-work/fixtures
MA=/d/Projects/strata-multiarch/bench-ma; RAW=/d/Projects/Strata-data/dflash-check/raw
set_=$1; p=$2; exe=$3; shift 3; envs="$* CUDA_CACHE_MAXSIZE=4294967296 DF_TIMEOUT=${DF_TIMEOUT:-1500}"
MTP=(--mtp D:/Projects/Strata-data/mtp/rt-tuned --mtp-head proxy --mtp-cum auto --spec 8)
run_pf() {
  local b=D:/Projects/strata-multiarch/bench-ma/fl_$p.bin; rm -f $b
  gpu_free || return 1
  line=$(env $envs STRATA_SPEC_FIXED=1 STRATA_DUMP_FIRST_LOGITS=$b bash $X ma_${p}_pf16 N $exe $W/prefill_16k.txt 48 --max-context 140000)
  echo "${p}_pf16 | ${line##*|} | logits $(md5sum < $b 2>/dev/null | cut -c1-10) | $(grep -a -o 'prefill  *[0-9]* tokens in [0-9.]* ms' $RAW/ma_${p}_pf16.log | tail -1) | $(grep -a -c -i 'error\|trap\|illegal\|abort' $RAW/ma_${p}_pf16.log) err-lines" | tee -a $MA/results.txt
}
run_dec() {  # run_dec NAME ARM [flags]
  local n=$1 arm=$2; shift 2
  gpu_free || return 1
  line=$(env $envs bash $X ma_${p}_$n $arm $exe $W/conv.txt 200 "$@")
  echo "${p}_$n | $line | $(grep -a -c -i 'error\|trap\|illegal\|abort' $RAW/ma_${p}_$n.log) err-lines" | tee -a $MA/results.txt
}
case $set_ in
  pf) run_pf ;;
  mtp) run_dec mtp M "${MTP[@]}" ;;
  dfl) run_dec dfl D ;;
  all) run_pf; run_dec mtp M "${MTP[@]}"; run_dec dfl D ;;
esac
