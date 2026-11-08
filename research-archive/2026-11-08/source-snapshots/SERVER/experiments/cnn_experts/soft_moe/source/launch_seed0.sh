#!/bin/bash
# Local seed-0 entry point only. ADDENDUM t83: reject every other seed.
# No seeds 1/2, no promotion relaunch, no second batch-size run.
# This continuation does not execute this script.
SEED="${1:-0}"
if [ "$SEED" != "0" ]; then
  echo "REJECT seed=$SEED ADDENDUM t83: seed 0 only"
  exit 2
fi
set -u
cd /home/alexn/Documents/code/datathon_2026/house-edge
mkdir -p experiments/cnn_experts/soft_moe/runs
LOCK=experiments/cnn_experts/soft_moe/runs/train.lock
exec 9>"$LOCK"
flock -n 9 || { echo "LOCK_HELD soft_moe train.lock"; exit 1; }
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
echo "launch_free_mib $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)"
exec .venv/bin/python -u experiments/cnn_experts/soft_moe/source/soft_moe.py --seed 0
