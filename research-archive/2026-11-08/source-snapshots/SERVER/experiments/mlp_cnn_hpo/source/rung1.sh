#!/bin/bash
# FROZEN before results. rung1: 8 epochs seed0 tune-only, all 20 configs.
cd ~/Documents/code/datathon_2026/house-edge; R=experiments/mlp_cnn_hpo/runs; P=".venv/bin/python experiments/mlp_cnn_hpo/source/train.py"
$P --model mlp --width 1024 --depth 3 --W 64 --drop 0.3 --lr 0.001 --wd 0.05 --epochs 30 --seed 0 --phase tune --out $R/mlpA_e30_s0 > $R/mlpA_e30_s0.log 2>&1
$P --model mlp --width 2048 --depth 2 --W 16 --drop 0.1 --lr 0.00222 --wd 0.21217 --epochs 8 --seed 0 --phase tune --out $R/r1_mlp00 > $R/r1_mlp00.log 2>&1
$P --model mlp --width 256 --depth 4 --W 64 --drop 0.1 --lr 0.00014 --wd 0.00756 --epochs 8 --seed 0 --phase tune --out $R/r1_mlp01 > $R/r1_mlp01.log 2>&1
$P --model mlp --width 256 --depth 2 --W 16 --drop 0.3 --lr 0.00091 --wd 0.01583 --epochs 8 --seed 0 --phase tune --out $R/r1_mlp02 > $R/r1_mlp02.log 2>&1
$P --model mlp --width 2048 --depth 2 --W 16 --drop 0.1 --lr 0.00113 --wd 0.12417 --epochs 8 --seed 0 --phase tune --out $R/r1_mlp03 > $R/r1_mlp03.log 2>&1
$P --model mlp --width 2048 --depth 2 --W 16 --drop 0.5 --lr 0.00063 --wd 0.01426 --epochs 8 --seed 0 --phase tune --out $R/r1_mlp04 > $R/r1_mlp04.log 2>&1
$P --model mlp --width 256 --depth 4 --W 64 --drop 0.1 --lr 0.00025 --wd 0.0239 --epochs 8 --seed 0 --phase tune --out $R/r1_mlp05 > $R/r1_mlp05.log 2>&1
$P --model mlp --width 1024 --depth 3 --W 32 --drop 0.3 --lr 0.00023 --wd 0.20532 --epochs 8 --seed 0 --phase tune --out $R/r1_mlp06 > $R/r1_mlp06.log 2>&1
$P --model mlp --width 2048 --depth 4 --W 64 --drop 0.1 --lr 0.00246 --wd 0.0015 --epochs 8 --seed 0 --phase tune --out $R/r1_mlp07 > $R/r1_mlp07.log 2>&1
$P --model mlp --width 1024 --depth 2 --W 64 --drop 0.3 --lr 0.00094 --wd 0.00259 --epochs 8 --seed 0 --phase tune --out $R/r1_mlp08 > $R/r1_mlp08.log 2>&1
$P --model mlp --width 256 --depth 4 --W 32 --drop 0.1 --lr 0.00013 --wd 0.00341 --epochs 8 --seed 0 --phase tune --out $R/r1_mlp09 > $R/r1_mlp09.log 2>&1
$P --model cnn --h 64 --layers 4 --W 32 --drop 0.3 --lr 0.00042 --wd 0.00408 --epochs 8 --seed 0 --phase tune --out $R/r1_cnn00 > $R/r1_cnn00.log 2>&1
$P --model cnn --h 64 --layers 4 --W 64 --drop 0.3 --lr 0.00192 --wd 0.00226 --epochs 8 --seed 0 --phase tune --out $R/r1_cnn01 > $R/r1_cnn01.log 2>&1
$P --model cnn --h 96 --layers 4 --W 32 --drop 0.2 --lr 0.00033 --wd 0.00592 --epochs 8 --seed 0 --phase tune --out $R/r1_cnn02 > $R/r1_cnn02.log 2>&1
$P --model cnn --h 128 --layers 4 --W 64 --drop 0.3 --lr 0.00197 --wd 0.00895 --epochs 8 --seed 0 --phase tune --out $R/r1_cnn03 > $R/r1_cnn03.log 2>&1
$P --model cnn --h 160 --layers 5 --W 32 --drop 0.2 --lr 0.00162 --wd 0.01093 --epochs 8 --seed 0 --phase tune --out $R/r1_cnn04 > $R/r1_cnn04.log 2>&1
$P --model cnn --h 160 --layers 4 --W 64 --drop 0.1 --lr 0.0018 --wd 0.0068 --epochs 8 --seed 0 --phase tune --out $R/r1_cnn05 > $R/r1_cnn05.log 2>&1
$P --model cnn --h 64 --layers 5 --W 32 --drop 0.2 --lr 0.00071 --wd 0.04301 --epochs 8 --seed 0 --phase tune --out $R/r1_cnn06 > $R/r1_cnn06.log 2>&1
$P --model cnn --h 128 --layers 5 --W 32 --drop 0.2 --lr 0.00038 --wd 0.02513 --epochs 8 --seed 0 --phase tune --out $R/r1_cnn07 > $R/r1_cnn07.log 2>&1
$P --model mlp --width 1024 --depth 3 --W 64 --drop 0.3 --lr 0.001 --wd 0.05 --epochs 8 --seed 0 --phase tune --out $R/r1_mlpA > $R/r1_mlpA.log 2>&1
$P --model cnn --h 128 --layers 4 --W 64 --drop 0.2 --lr 0.001 --wd 0.05 --epochs 8 --seed 0 --phase tune --out $R/r1_cnnA > $R/r1_cnnA.log 2>&1
echo RUNG1_DONE > $R/RUNG1_DONE
