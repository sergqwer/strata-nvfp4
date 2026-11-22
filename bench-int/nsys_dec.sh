#!/bin/bash
# nsys_dec.sh TAG strata|basalt FIXTURE NGEN [engine flags...] - one serve decode under nsys (CUDA trace, graph nodes),
# env PF_EXE / PF_DRAFT / PF_ENV as serve_ab.sh.  -> D:/Projects/strata-dec-int/bench-int/nsys/TAG.{nsys-rep,sqlite}
O=D:/Projects/strata-dec-int/bench-int/nsys; mkdir -p /d/Projects/strata-dec-int/bench-int/nsys
tag=$1; e=$2; fx=$3; n=$4; shift 4
NSYS=$(ls -d "/c/Program Files/NVIDIA Corporation/Nsight Systems"*/target-windows-x64/nsys.exe | tail -1)
PYW=C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/pythonw.exe
PY=C:/LLM_models/strata-nvfp4/.venv-serve/Scripts/python.exe
x=(); [ $e = basalt ] && x=(--decode-timing)
PF_GEN=$n PF_LOG=$O/$tag.engine.log bash /d/Projects/Strata-data/bench-tools/gpu_lock.sh nsys-int "$NSYS" profile --trace=cuda \
  --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true -o $O/$tag \
  $PYW D:/Projects/Strata-data/bench-tools/hidden_run.py --log $O/$tag.client.log $PY ${PF_CLIENT:-D:/Projects/basalt-work/pf_client.py} $e $fx "${x[@]}" "$@" > $O/$tag.nsys.out 2>&1
"$NSYS" export --type sqlite --force-overwrite true -o $O/$tag.sqlite $O/$tag.nsys-rep > /dev/null 2>&1
ls -la $O/$tag.*; tail -2 $O/$tag.client.log; grep -a "decode timing" $O/$tag.engine.log | tail -1 | cut -c1-300
