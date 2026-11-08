2026-11-07 21:23:23 CST ADDENDUM (Alex t83) acknowledged: one seed-0 run only; no seeds 1/2; no confirm/test selection. Plan frozen sha256=36017f2f667ae60095c75babb137a9a56f36acca1c008619b5cf3aa10e56dafe SPLIT_IDS sha256=57e04dc8963efcadd33285cdd08e8c886302233948e35ae8c0f3ea5fb4a645e1
2026-11-07 21:23:23 CST tests pass: 13 passed in 3.10s (pytest, existing random_forest site-packages, no install)
2026-11-07 21:24:05 CST training started pid=3480208 out=experiments/cnn_experts/hard_signal/runs/s0 vram_frac=0.1293
2026-11-07 21:24:08 CST first epoch pretrain loss 1.225713 pid=3480208
2026-11-07 21:24:11 CST launch verified pid=3480208 log=experiments/cnn_experts/hard_signal/runs/s0.log command=env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python -u experiments/cnn_experts/hard_signal/source/run_seed.py --seed 0 --out experiments/cnn_experts/hard_signal/runs/s0
2026-11-07 21:25:11 CST final seed0 tune_R_routed=0.500991 tune_R_control=0.462112 incumbent=0.332064 verdict=negative seconds=66 peak_gpu_mb=818
