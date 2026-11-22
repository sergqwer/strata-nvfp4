#!/bin/bash
H=/d/Projects/strata-dec-int/bench-int; cd $H
B=D:/Projects/strata-dec/build-dev/strata.exe; I=D:/Projects/strata-dec-int/build-dev/strata.exe
CI=$H/pf_client_int.py; W=/d/Projects/basalt-work
L=/d/Projects/Strata-data/bench-tools/gpu_lock.sh
bash $L build-int cmd //c "D:\Projects\Strata-data\bench-tools\build_dev.cmd" "D:\Projects\strata-dec-int" strata nv_defaults_test fp4g_parity > build_dev7.log 2>&1 || { echo BUILD-FAILED; exit 1; }
echo BUILD-OK
bash $L int-test bash -c "../build-dev/nv_defaults_test.exe | tail -1; ../build-dev/fp4g_parity.exe" > fp4g_parity2.txt 2>&1; echo TESTS-DONE
bash pfcheck.sh > pfcheck.txt 2>&1; echo PC-DONE
LK=/d/Projects/Strata-data/bench-tools/gpu_lock.sh; PY=C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/python.exe
for fx in $W/gdn/prefill_3k.txt $W/fixtures/prefill_16k.txt $W/fixtures/prefill_64k.txt $H/long/prefill_131k.txt; do for r in 1 2; do
  for arm in b i; do e=$B; [ $arm = i ] && e=$I
    log=$H/serve/pf_${arm}_$(basename $fx .txt)_r$r.log
    PF_CTX=140000 PF_EXE=$e PF_LOG=$log bash $LK int-serve $PY $CI strata $fx > /dev/null 2>&1
    echo "$arm $(basename $fx) r$r | $(grep -a -o 'prompt [0-9]* tokens = .* read in [0-9]* ms ([0-9.]* tok/s)' $log | tail -1)"
  done
done; done > prefill_n4.txt 2>&1; echo PF-DONE
mkdir -p long/tf
# precision: reference = base exe with fp16 experts (NOTES-pfx.md); base; int (the whole new prompt stack); int again (noise); int NP 3
for set in set:$W/corpus/long/set.json s128:$H/long/set_128k.json; do
  sn=${set%%:*}; sf=${set#*:}
  for arm in "ref|$B|STRATA_PREFILL_NVFP4=fp16" "base|$B|" "base2|$B|" "int|$I|" "int2|$I|" "intnp3|$I|STRATA_GDN_TC_NP=3"; do
    IFS='|' read -r tag exe envs <<< "$arm"
    t0=$(date +%s)
    DF_EXE=$exe SCORE_ENV="$envs" bash $LK int-long $PY $W/score_strata.py $sf $H/long/tf/${sn}_$tag.tsv > long/tf/${sn}_$tag.out 2>&1
    echo "$sn $tag $(( $(date +%s) - t0 )) s | $(grep -a -o 'read in [0-9]* ms ([0-9.]* tok/s)' long/tf/${sn}_$tag.tsv.engine.log 2>/dev/null | tr '\n' ' ')"
  done
done > long_n4.txt 2>&1; echo LONG-DONE
echo Q4-DONE
