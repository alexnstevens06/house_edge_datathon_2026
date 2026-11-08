# CHANGES (mlp_cnn_hpo, session 2026-11-07)

Written ONLY inside experiments/mlp_cnn_hpo/. No GPU use, no installs, no git. CPU-only smoke checks.

## Files
- `source/base_cnn.py` — verbatim copy of `experiments/cnn/source/cnn.py` (hashes in `runs/SNAPSHOT_HASHES.txt`, identical).
- `source/train.py` — `import base_cnn as B; B.USE_SIGNAL=False` (signal never read). `MLP(width,depth,drop,W)`: (B,W,8) int8 payout idx -> one-hot payout(5) (missing=all zeros) + observed mask(1) -> (B,W*8*6), concat type one-hot (8*3) + t/256 -> depth hidden layers (Linear/GELU/LayerNorm/Dropout, width) -> Linear(width,40) -> (B,8,5); `forward(x,t,ty,sg)` ignores sg. Only Linear/LayerNorm/Dropout/GELU modules (Sequential container). `main()` reuses B.build_train/build_query/predict/to_preds/masked_loss/CNN and shared house_edge scoring; args: `--model {cnn,mlp} --width --depth --kernel (ignored) --phase {tune,final}` + base_cnn hparams, `--max_train_windows`, `--out`. Phase tune scores ONLY tune subset. Phase final scores tune/confirm/full, saves checkpoint, writes CPU fp32 2-thread validation/test CSVs. Always no signal (CNN `use_signal=False`); threads 2; cuda memory fraction 0.25; bf16 autocast on cuda only. Appends one JSON line per run to `runs/trials.jsonl` (args, param_count, n_train_windows, per-epoch losses, tune_R, confirm_R/full_R if final, seconds).
- `tests/test_mlp.py` — plain PASS/FAIL runner (CPU): module types restricted to Linear/LayerNorm/Dropout/GELU/Sequential/ModuleList; output (B,8,5) finite softmax summing to 1; causality on a real `short_history` validation episode (`house_edge.data.episode_conditions`) — for each machine, mutating all obs ticks >= t leaves `B.window` and MLP output bit-identical; missing vs payout-0 inputs give different outputs; flipping every signal in queries leaves `build_query` features and outputs bit-identical (SG all zeros with USE_SIGNAL=False).
- `runs/SNAPSHOT_HASHES.txt`, `runs/trials.jsonl`, `runs/smoke_mlp/config.json`.

## Results
- Tests: `CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= .venv/bin/python tests/test_mlp.py` → PASS module_types, output_shape_softmax, causality_short_history, missing_vs_payout0, signal_flip_invariant; ALL PASS.
- Smoke (CPU, <60s): `train.py --model mlp --epochs 1 --max_train_windows 2000` → 18,567 train windows built (1.5s), truncated to 2000; 5,318,696 params; epoch 0 loss 1.4691; tune_R 1.5374; total ~3.0s; one line appended to `runs/trials.jsonl`.

Stopped here per instructions (no GPU benchmark, no real training — HPO runs on the other server).
