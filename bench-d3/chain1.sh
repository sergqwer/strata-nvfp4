#!/bin/bash
# chain1.sh - D1: base identity, then the D1 build (+ parity test), parity runs, release-target compile check
D=/d/Projects/strata-dec3/bench-d3; LK=/d/Projects/Strata-data/bench-tools/gpu_lock.sh
cd $D
bash ident.sh base D:/Projects/strata-dec3/bench-d3/base/strata.exe > ident_base.run 2>&1
echo "ident base done $(date +%T)"
bash $LK d3-build cmd //c 'D:\Projects\Strata-data\bench-tools\build_dev.cmd' 'D:\Projects\strata-dec3' strata native_multi_parity > build_d1.log 2>&1; echo "build rc=$? $(date +%T)"
mkdir -p d1; cp ../build-dev/strata.exe ../build-dev/strata_kq.dll base/cublas64_13.dll base/cublasLt64_13.dll d1/
for b in 1 4 8 16; do
  echo "== parity STRATA_ROUTE_DB=2 STRATA_ROUTE_DB_BLOCKS=$b"
  STRATA_ROUTE_DB=2 STRATA_ROUTE_DB_BLOCKS=$b bash $LK d3-parity ../build-dev/native_multi_parity.exe 2>&1 | grep -a -E "router|route|bitwise|NO|FAIL|differs" | head -20
done > parity_d1.txt
echo "parity done"
bash $LK d3-cc cmd //c 'D:\Projects\strata-dec3\bench-d3\cc_check.cmd' native_bf16.cu native_router.cu > cc_check.log 2>&1; echo "cc rc=$?"
grep -a -E "^(OK|FAIL)" cc_check.log
