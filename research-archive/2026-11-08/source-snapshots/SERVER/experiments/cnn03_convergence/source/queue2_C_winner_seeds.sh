#!/bin/bash
cd ~/Documents/code/datathon_2026/house-edge; O=experiments/cnn03_convergence/runs
until grep -q QUEUE1_DONE $O/QUEUE_STATUS.txt; do sleep 3; done
P=".venv/bin/python experiments/cnn03_convergence/source/conv_train.py --h 128 --layers 4 --W 64 --drop 0.3 --wd 0.00895 --bs 512 --epochs 30 --eval_every 2"
run(){ n=$1; shift; echo "$(date +%T) start $n" >> $O/QUEUE_STATUS.txt; $P "$@" --out $O/$n > $O/$n.log 2>&1; echo "$(date +%T) end $n exit $?" >> $O/QUEUE_STATUS.txt; }
for s in 1 2; do run curve_C_lr003_s$s --lr 0.003 --pct_start 0.3 --seed $s; run curve_control_s$s --lr 0.00197 --pct_start 0.3 --seed $s; done
echo QUEUE2_DONE >> $O/QUEUE_STATUS.txt
