#!/bin/bash
# vc.sh TAG [env...] - venc_client.py on the dev exe under the GPU lock (gpu_free2 from run.sh)
source /d/Projects/strata-vision-gpu/bench-vision/run.sh --defs-only 2>/dev/null
tag=$1; shift
P=D:/Projects/strata-vision-gpu/tools/qwen35/pictures
pw
timeout 600 bash $LK vgpu-$tag bash -c 'gpu_free2 && exec "$@"' _ env "$@" $PY -B D:/Projects/strata-vision-gpu/bench-vision/venc_client.py \
  D:/Projects/strata-vision-gpu/build-dev/strata.exe D:/Projects/strata-vision-gpu/bench-vision/sve/$tag $P/shapes.png $P/big1024.png $P/code4217.png
pw
