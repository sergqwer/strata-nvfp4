#!/bin/bash
# queue 1: L2PF exactness, A/B of L2PF and adapt-min-gain, profiles of int and Basalt at matched windows
H=/d/Projects/strata-dec-int/bench-int
until grep -q SERVE-DONE $H/serve_1.txt; do sleep 10; done
BB=D:/Projects/strata-dec-int/build-b/strata.exe; F=D:/Projects/basalt-work/fixtures
MTP=(--mtp D:/Projects/Strata-data/mtp/rt-tuned --mtp-head proxy --mtp-cum 0.43:0.31 --spec 8)
STRATA_L2PF=64 bash $H/exact.sh ie_l2pf_P8_struct M $BB $F/struct.txt 2000 "${MTP[@]}" >> $H/exact_l2pf.txt
STRATA_L2PF=64 bash $H/exact.sh ie_l2pf_P8_conv M $BB $F/conv.txt 400 "${MTP[@]}" >> $H/exact_l2pf.txt
bash $H/ab.sh l2 "A|$BB|| L64|$BB|STRATA_L2PF=64| G0|$BB||--adapt-min-gain,0" "struct conv"
ROUNDS=1 bash $H/ab.sh prof "P|$BB|STRATA_VERIFY_PROFILE=1| BP|||--profile,--wait-stats" "conv struct"
echo Q1-DONE
