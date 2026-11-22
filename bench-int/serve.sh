#!/bin/bash
# serve.sh LABEL strata|basalt [engine flags...] - serve_ab.sh's greedy runs (pf_client.py: --serve, a fresh engine per
# fixture, ctx 66661) on conv / struct / prose and the Ukrainian prompt (uk.txt); env PF_EXE / PF_DRAFT / PF_ENV as
# serve_ab.sh.  FIXTURES="conv uk" limits them.  Logs: D:/Projects/strata-dec-int/bench-int/serve/<engine>_<LABEL>_<fixture>.log
W=/d/Projects/basalt-work; M=/d/Projects/Strata-data/mtp
PY=C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/python.exe
LOCK=/d/Projects/Strata-data/bench-tools/gpu_lock.sh
L=$1; e=$2; shift 2
declare -A N=([conv]=1200 [struct]=8192 [prose]=1200 [uk]=1200)
declare -A F=([conv]=$W/fixtures/conv.txt [struct]=$W/fixtures/struct.txt [prose]=$W/fixtures/prose.txt [uk]=$M/uk.txt)
for fx in ${FIXTURES:-conv struct prose uk}; do
  log=D:/Projects/strata-dec-int/bench-int/serve/${e}_${L}_$fx.log
  if [ $e = basalt ]; then x=(--decode-timing); else x=(); fi
  out=$(PF_GEN=${N[$fx]} PF_LOG=$log bash $LOCK int-serve $PY ${PF_CLIENT:-$W/pf_client.py} $e ${F[$fx]} "${x[@]}" "$@" 2>&1 | tail -1)
  req=$(grep -a -E 'serve: prompt [0-9]+ tokens' $log | tail -1 | sed -E 's/.*read in [0-9]+ ms \([0-9.]+ tok\/s\), //')
  dr=$(grep -a -o -E "decode timing: .*" $log | tail -1 | grep -a -o -E "verify [0-9.]+|draft [0-9.]+" | tr -s "[:space:]" " ")
  tw=$(grep -a -o -E "decode timing: [0-9]+ windows, avg T [0-9.]+, [0-9.]+ tokens/window, [0-9.]+ ms/window" $log | tail -1 | sed 's/decode timing: //')
  echo "$e $L $fx: $req | $tw | $dr" | cut -c1-260
  sleep 5
done
