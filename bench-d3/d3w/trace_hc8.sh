#!/bin/bash
# trace_hc8.sh TAG ARGS... - gr_hc8_check under the CUPTI tracer (gpu-probe host/cupti_trace.dll) -> d3w/TAG.csv, under the GPU lock
cd /d/Projects/strata-dec3
tag=$1; shift
export CUDA_INJECTION64_PATH="D:/Projects/gpu-probe/host/cupti_trace.dll" CUPTI_TRACE_OUT="D:/Projects/strata-dec3/bench-d3/d3w/$tag.csv"
export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/extras/CUPTI/lib64:$PATH"
timeout 900 bash /d/Projects/Strata-data/bench-tools/gpu_lock.sh d3-trace bash -c 'source bench-d3/gpu_free.sh; gpu_free && timeout 600 "$@"' _ "${EXE:-build-dev/gr_hc8_check.exe}" "$@" > bench-d3/d3w/$tag.txt 2>&1
echo "rc $?"
