#!/bin/bash
# gpu_free.sh - rel04's gpu_free, minus tray-path PIDs listed in GPU_FREE_IGNORE (default 22984: the probe agent's CUPTI zombie,
# DECODE-PLAN.md section 5, holds no VRAM).  Waits max 10 min for VRAM < 2500 MiB and no live tray engine; never kills.
gpu_free() {
  local i used tray
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    tray=$(powershell.exe -NoProfile -Command "\$ig = '${GPU_FREE_IGNORE:-22984}' -split ' '; (Get-CimInstance Win32_Process -Filter \"name='strata.exe'\" | Where-Object { \$_.ExecutablePath -like 'C:\LLM_models*' -and (\$ig -notcontains [string]\$_.ProcessId) } | Measure-Object).Count" 2>/dev/null | tr -d '\r')
    [ "$used" -lt 2500 ] && [ "${tray:-0}" = 0 ] && return 0
    echo "waiting: VRAM used $used MiB, tray engines ${tray:-?}" >&2; sleep 10
  done
  echo "GPU busy, giving up" >&2; return 1
}
export -f gpu_free
