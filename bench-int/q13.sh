#!/bin/bash
# queue 13: dense_tc on the sm_120 table: build, tests, identity (fixed draft costs), one teacher-forced arm with every default
H=/d/Projects/strata-dec-int/bench-int; cd $H
until [ -f q13.go ]; do sleep 10; done
L=/d/Projects/Strata-data/bench-tools/gpu_lock.sh; PY=C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/python.exe
I=D:/Projects/strata-dec-int/build-dev/strata.exe; W=/d/Projects/basalt-work
bash $L build-int cmd //c "D:\Projects\Strata-data\bench-tools\build_dev.cmd" "D:\Projects\strata-dec-int" strata nv_defaults_test gdn_rec_parity q80p_parity draft_proxy_parity gr_hc8_check fp4g_parity > build_dev13.log 2>&1 || { echo BUILD-FAILED; exit 1; }
echo BUILD-OK
bash $L int-test bash -c "cd ../build-dev; ./nv_defaults_test.exe | tail -1; ./q80p_parity.exe --selftest | tail -1; ./draft_proxy_parity.exe --selftest | tail -1; ./gr_hc8_check.exe --selftest | tail -1; ./gdn_rec_parity.exe | tail -1; ./fp4g_parity.exe | tail -3" > tests13.txt 2>&1; echo TESTS-DONE
bash exact6.sh > exact6.txt 2>&1; echo EXACT-DONE
DF_EXE=$I bash $L int-acc $PY $W/score_strata.py $W/corpus/set.json $H/tf/acc_def.tsv > tf/acc_def.out 2>&1; echo ACC-DONE
echo Q13-DONE
