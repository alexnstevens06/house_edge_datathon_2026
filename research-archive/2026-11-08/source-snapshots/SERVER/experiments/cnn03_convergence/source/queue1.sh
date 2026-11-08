#!/bin/bash
cd ~/Documents/code/datathon_2026/house-edge; O=experiments/cnn03_convergence/runs
until grep -q "^exit" $O/canon_C_lr003_s0.log 2>/dev/null; do sleep 3; done
P=".venv/bin/python experiments/cnn03_convergence/source/conv_train.py --h 128 --layers 4 --W 64 --drop 0.3 --wd 0.00895 --bs 512 --epochs 30 --eval_every 2"
run(){ n=$1; shift; echo "$(date +%T) start $n" >> $O/QUEUE_STATUS.txt; $P "$@" --out $O/$n > $O/$n.log 2>&1; echo "$(date +%T) end $n exit $?" >> $O/QUEUE_STATUS.txt; }
run curve_control_s0 --lr 0.00197 --pct_start 0.3 --seed 0
run curve_A_pct010_s0 --lr 0.00197 --pct_start 0.1 --seed 0
run curve_C_lr003_s0 --lr 0.003 --pct_start 0.3 --seed 0
echo QUEUE1_DONE >> $O/QUEUE_STATUS.txt
