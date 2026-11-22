#!/bin/bash
# e2_run.sh TAG EXE REQ... - one engine start through e2_client.py under the GPU lock (env PF_ADD, PF_ENV, PF_CONV pass through)
source /d/Projects/Strata-data/rel04/common.sh
source /d/Projects/strata-dec3/bench-d3/gpu_free.sh
tag=$1; exe=$2; shift 2
D=/d/Projects/strata-dec3/bench-d4
PF_EXE=$exe PF_ENV="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1 $PF_ENV" PF_LOG=D:/Projects/strata-dec3/bench-d4/log/e2_$tag.log \
  timeout 900 bash $LK e2-$tag bash -c 'gpu_free && exec "$@"' _ $PY -B $D/e2_client.py gptq D:/Projects/strata-dec3 "$@" 2>&1 | grep -a "^req" | tee -a $D/e2_runs.txt | sed "s/^/$tag /"
grep -a -o -E "expert cache [0-9]+ slots|elastic session tables.*|K/V grown.*|K/V trimmed.*|only [0-9]+ MiB of VRAM free|illegal.*|CUDA error.*|ERR .*" $D/log/e2_$tag.log | sort | uniq -c | head -20
