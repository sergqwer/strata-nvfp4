#!/bin/bash
# queue 12: x3p's PCIe share - the +0.10 rule (default) vs a fixed 0.45 (host round 2's measured x3p)
H=/d/Projects/strata-dec-int/bench-int; cd $H
until grep -q Q11-DONE q11.log; do sleep 20; done
I=D:/Projects/strata-dec-int/build-dev/strata.exe; CI=$H/pf_client_int.py
TD="STRATA_DECODE_TIMING=1 STRATA_DRAFT_TIMING=1"
for r in 1 2; do for fx in struct prose conv; do
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh y_a_r$r strata
  PF_CLIENT=$CI PF_DRAFT=auto PF_EXE=$I PF_ENV="$TD" FIXTURES=$fx bash serve.sh y_f_r$r strata --pcie-frac 0.45
done; done > serve_pf45.txt 2>&1; echo PF45-DONE
echo Q12-DONE
