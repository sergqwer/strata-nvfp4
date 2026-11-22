#!/bin/bash
# dense_tc (STRATA_DENSE_TC=1, build-b): serve A/B, then accuracy (corpus/set.json + long set)
H=/d/Projects/strata-dec-int/bench-int; cd $H
until grep -q Q7-DONE q7.log; do sleep 20; done
X=D:/Projects/strata-dec-int/build-b/strata.exe; CI=$H/pf_client_int.py; W=/d/Projects/basalt-work
L=/d/Projects/Strata-data/bench-tools/gpu_lock.sh; PY=C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/python.exe
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
for r in 1 2; do for fx in struct prose conv; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$X PF_ENV="$TD" FIXTURES=$fx bash serve.sh t8_d0_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$X PF_ENV="$TD STRATA_DENSE_TC=1" FIXTURES=$fx bash serve.sh t8_d1_r$r strata
done; done > serve_t8.txt 2>&1; echo SERVE-DONE
for arm in "d0|" "d0b|" "d1|STRATA_DENSE_TC=1" "d1b|STRATA_DENSE_TC=1"; do
  IFS='|' read -r tag envs <<< "$arm"
  DF_EXE=$X SCORE_ENV="$envs" bash $L int-acc $PY $W/score_strata.py $W/corpus/set.json $H/tf/acc_$tag.tsv > tf/acc_$tag.out 2>&1
  DF_EXE=$X SCORE_ENV="$envs" bash $L int-acc $PY $W/score_strata.py $W/corpus/long/set.json $H/tf/long_$tag.tsv > tf/long_$tag.out 2>&1
  echo "$tag done"
done > acc_t8.txt 2>&1; echo ACC-DONE
echo Q8-DONE
