#!/bin/bash
# micro2.sh - D1 micro: the bf16 multi GEMV by shape at T 7/8 (STRATA_MMVF_EXACT78=0/1), router + route_db at T 7.  Inside gpu_lock.
B=/d/Projects/strata-dec3/build-dev/router_chain_bench.exe
nvidia-smi --query-gpu=power.limit,clocks.mem,clocks.max.memory --format=csv,noheader
for e in 0 1; do echo "== EXACT78=$e"; STRATA_MMVF_EXACT78=$e STRATA_ROUTE_DB=2 STRATA_ROUTE_DB_BLOCKS=1 $B 7 50 | grep -a -E "SHAPE .* T [678] |T 7 (mmvf |mmvf\+route_db)"; done
for b in 1 8 16; do echo "== T=7 BLOCKS=$b"; STRATA_ROUTE_DB=2 STRATA_ROUTE_DB_BLOCKS=$b $B 7 50 | grep -a -E "T 7 route_db"; done
