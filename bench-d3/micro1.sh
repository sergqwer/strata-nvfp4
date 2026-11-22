#!/bin/bash
# micro1.sh - D1 micro: router_chain_bench (engine kernels, 48-layer graphs) for P4 (STRATA_MMVF_EXACT78=0/1) and P5
# (STRATA_ROUTE_DB=2, STRATA_ROUTE_DB_BLOCKS=1/4/8/16).  Run inside gpu_lock.
B=/d/Projects/strata-dec3/build-dev/router_chain_bench.exe
nvidia-smi --query-gpu=power.limit,clocks.mem,clocks.max.memory --format=csv,noheader
for e in 0 1; do echo "== EXACT78=$e"; STRATA_MMVF_EXACT78=$e STRATA_ROUTE_DB=2 STRATA_ROUTE_DB_BLOCKS=1 $B 8 50 | grep -a -E "PARITY n_in 2560 n_out 512|SHAPE n_in 2560 n_out 512|T 8 (mmvf |route_db|mmvf\+route_db)"; done
for t in 3 8; do for b in 1 4 8 16; do echo "== T=$t BLOCKS=$b"; STRATA_ROUTE_DB=2 STRATA_ROUTE_DB_BLOCKS=$b $B $t 50 | grep -a -E "route_db"; done; done
# all shapes' parity hashes, both arms (the dispatcher serves every bf16 multi GEMV)
for e in 0 1; do STRATA_MMVF_EXACT78=$e $B 2 3 | grep -a PARITY > parity_mmvf_$e.txt; done
cmp parity_mmvf_0.txt parity_mmvf_1.txt && echo "PARITY all shapes T 1..8: EXACT78=0 == 1 ($(wc -l < parity_mmvf_1.txt) hashes)"
