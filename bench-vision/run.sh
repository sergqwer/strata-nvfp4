#!/bin/bash
# run.sh TAG TREE [vb.py args] - one vb.py serve run under the GPU lock.  gpu_free2: VRAM < 2500 MiB and no live
# strata.exe from C:\LLM_models (a process with one thread and no CPU time is an exiting orphan, not a loaded engine).
source /d/Projects/Strata-data/rel04/common.sh
gpu_free2() {
  local i used tray
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    tray=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"name='strata.exe'\" | Where-Object { \$_.ExecutablePath -like 'C:\LLM_models*' -and \$_.ThreadCount -gt 1 } | Measure-Object).Count" 2>/dev/null | tr -d '\r')
    [ "$used" -lt 2500 ] && [ "${tray:-0}" = 0 ] && return 0
    echo "waiting: VRAM used $used MiB, tray engines ${tray:-?}" >&2; sleep 10
  done
  echo "GPU busy, giving up" >&2; return 1
}
export -f gpu_free2
[ "$1" = "--defs-only" ] && return 0
tag=$1; shift
pw
timeout 1500 bash $LK vgpu-$tag bash -c 'gpu_free2 && exec "$@"' _ $PY -B D:/Projects/strata-vision-gpu/bench-vision/vb.py $tag "$@"
pw
