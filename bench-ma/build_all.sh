#!/bin/bash
# build the three PTX engines in sequence under the GPU lock
cd /d/Projects/strata-multiarch/bench-ma
for a in 75 86 89; do
  echo "== $a $(date +%T)"
  cmd.exe //c "D:\Projects\strata-multiarch\bench-ma\build_arch.cmd" ${a}-virtual build-ptx$a > build_ptx$a.log 2>&1
  echo "rc $? $(date +%T)"; tail -3 build_ptx$a.log
done
