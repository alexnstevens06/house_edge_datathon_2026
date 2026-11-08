# Low fixed-LR warm start (frozen before GPU jobs)

Protocol file `COORDINATOR_PLAN.md` SHA-256 `03d141087ac950e8026ebaeb6afcc4f0b83371508beee0961ecd993cc9f9e8f3`. This study does not edit that file. It does not run the ensemble or blend stream. Those stay with the other author under `ensemble_blend_20261108`.

## Recipe, declared before any new metric

Weights-only warm start from the original seed-2 parents. Fresh AdamW. Not an exact optimizer resume. The parents have no optimizer or scheduler state.

| Item | Value |
| --- | --- |
| Architecture | h 128, layers 4, kernel 3, W 64, drop 0.3, no signal |
| Train windows | 18567, the existing `build_train` on train only, min_t 16, stride 1 |
| Batch | 128. Historical curve training was bf16 autocast at batch 512. This study is FP32 at batch 128 on both GPUs so the 12 GB gfx1031 card can run the same source. |
| Steps | ceil(18567/128)=146 per epoch, 5 epochs, 730 steps per job |
| Optimizer | AdamW, weight decay 0.00895, grad clip 1, no scheduler, no OneCycle, no ramp |
| Rates | server 2e-5 for both arms; tower 5e-5 for both arms |
| Arms | primary `curve_C_lr003_s2` and control `curve_control_s2` |
| RNG | 20261108, set after the parent weights are loaded, same on both arms |
| Saves | completed epochs 1, 3, and 5, with optimizer, RNG, and provenance, before any tune score |
| Score | server CPU float64, 384 tune queries only, after the trainer has exited |

Platform is confounded with rate. A paired primary-versus-control comparison is valid inside one host. It is not a causal claim that one learning rate is better.

No extra epochs, rates, seeds, or automatic promotion. Current best tune R 0.30980333573689856 is the comparison point. Control seed 2 is 0.3334821793902223. The old canonical tune 0.33206398205344967 is context, not the bar. Packaging waits for an independent review of a comparative winner.

## What training reuses

`experiments/cnn03_convergence/source/base_cnn.py` supplies `build_train`, `window`, `masked_loss`, and the training `CNN`. `USE_SIGNAL` is set false before any window is built. The forward call is `(x, t, ty, sg)`; the label tensor is only the masked loss target. The float64 scorer loads validation with pinned `src/house_edge/data.py` `load_split` and no labels, the same loader as the published tune R. It forwards through `experiments/cnn_runtime/fresh_fp64_cnn03_1956/model.py`, which imports that folder's `data.py` for the missing-value constant, then `model.double()`. It does not use `base_cnn.predict`, which casts softmax. Production `PINNED.json` and `fp64_harness.py` are not imported.

GPU occupancy is checked with `nvidia-smi` on the server and `rocm-smi` on the tower. A foreign compute process blocks the launch. One sibling `train_lowrate.py` from this study may already be on the card, so the paired arm can run beside it. A third trainer is refused. TF32 is disabled with process-local backend flags. Host RSS above 2 GiB stops the job before any tune metric.
