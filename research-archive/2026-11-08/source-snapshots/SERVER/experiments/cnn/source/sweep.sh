#!/bin/bash
cd ~/Documents/code/datathon_2026/house-edge
P=".venv/bin/python experiments/cnn/source/cnn.py"
run(){ n=$1; shift; $P "$@" --out experiments/cnn/runs/$n > experiments/cnn/runs/$n.log 2>&1; grep RESULT experiments/cnn/runs/$n.log >> experiments/cnn/runs/sweep_results.txt; }
: > experiments/cnn/runs/sweep_results.txt
run a_h64_L4_W64_e12 --h 64 --layers 4 --W 64 --epochs 12
run b_h128_L4_W128_e10 --h 128 --layers 4 --W 128 --epochs 10
run c_h64_L5_W128_e10 --h 64 --layers 5 --W 128 --epochs 10
run d_h96_L4_W96_d02_e10 --h 96 --layers 4 --W 96 --drop 0.2 --epochs 10
run e_h64_L3_W32_e10 --h 64 --layers 3 --W 32 --epochs 10
run f_h128_L4_W64_d02_e16 --h 128 --layers 4 --W 64 --drop 0.2 --epochs 16 --lr 1e-3
echo SWEEP_DONE >> experiments/cnn/runs/sweep_results.txt
