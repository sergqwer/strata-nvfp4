#!/bin/bash
# rebuild the given arch dirs (e.g. "86 89" or "120") with the patch, under the GPU lock
cd /d/Projects/strata-multiarch/bench-ma
for a in "$@"; do
  if [ $a = 120 ]; then archs=120-real; d=build-rel120; else archs=${a}-virtual; d=build-ptx$a; fi
  echo "== $a $(date +%T)"
  cmd.exe //c "D:\Projects\strata-multiarch\bench-ma\build_arch.cmd" $archs $d strata nv_defaults_test > build_p$a.log 2>&1
  echo "rc $? $(date +%T)"; tail -2 build_p$a.log
done
