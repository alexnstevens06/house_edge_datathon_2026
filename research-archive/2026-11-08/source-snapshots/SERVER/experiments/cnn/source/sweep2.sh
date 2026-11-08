#!/bin/bash
cd ~/Documents/code/datathon_2026/house-edge
P=".venv/bin/python experiments/cnn/source/cnn.py"
run(){ n=$1; shift; $P "$@" --out experiments/cnn/runs/$n > experiments/cnn/runs/$n.log 2>&1; grep RESULT experiments/cnn/runs/$n.log >> experiments/cnn/runs/sweep_results.txt; }
run g_h128_L4_W64_d02_e30 --h 128 --layers 4 --W 64 --drop 0.2 --epochs 30 --lr 1e-3
run h_h128_L5_W64_d03_e30 --h 128 --layers 5 --W 64 --drop 0.3 --epochs 30 --lr 1e-3
echo SWEEP2_DONE >> experiments/cnn/runs/sweep_results.txt
