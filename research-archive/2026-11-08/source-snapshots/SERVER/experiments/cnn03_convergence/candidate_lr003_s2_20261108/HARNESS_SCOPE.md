# Scoped harness difference

Production `experiments/cnn_runtime/fp64_harness.py` and release `experiments/cnn_runtime/fp64_release_harness2025.py` are not modified. `experiments/mlp_cnn_hpo/review/PINNED.json` is not modified, not bypassed, and not pointed at this candidate.

`fp64_candidate_harness.py` is a copy of the release harness with the path and parent-contract changes below. `HARNESS_DIFF.txt` is `diff -u` against the release harness. Assertions that remain: no `.float()` and no tensor-float assignment in `model.py` and `predict.py`; all 8 machines; a future tick does not change the cutoff window; tick t-1 does; MISSING is index 5 with mask 0 and payout 0 is index 0; slots are sorted; labels and conditions are not opened; the signal column is dropped; a missing split fails; 43 float64 parameters after `model.double()`; mask and tick inputs are float64; logits and softmax stay float64; no autocast; the tensor float method is not monkeypatched; cold subprocess repeat is bitwise; test coverage is 1536 ids; validation coverage is 768 ids; kit schema check; own-reference max abs after kit row norm fails above `1e-8`; a missing reference exits 1.

Changes, and why:

- `ROOT = Path(__file__).resolve().parents[3]`. The release file uses `parents[2]` because it lives in `experiments/cnn_runtime/`. This file is one level deeper, so `parents[2]` would be `experiments/` and every path would be wrong. The candidate folder is hard as the output parent: the harness refuses an `--out` path outside `candidate_lr003_s2_20261108`.
- Temp directories use this folder's `_work`, not `experiments/cnn_runtime`.
- Default entry is `unpacked/`, the ZIP extract, not the working tree and not `fp64_cnn03_entry`.
- References are this candidate's own test and validation CSVs. They are not the canonical test CSV `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d`.
- The parent config omits `kernel`. The harness does not call `cfg.get("kernel", 3)`. It requires `blocks.0` through `blocks.3` `conv.weight` shape `(128, 128, 3)` and then constructs `CNN(..., 3, ...)`. A config that grows a `kernel` field fails.
- Negative guards added, without removing the hash, config-hash, or signal guards: extra state key, conv kernel extent 5, and a float16 saved tensor. Those copies are temporary and are not the original checkpoint.
- Query-row order is reversed on an 8-machine synthetic episode. The sorted CSV must be bitwise identical. No metric is computed.
- `official_r` is still defined and is not called. `score_round` on the validation file includes confirm episodes. The harness prints `VALIDATION_R not_computed` instead. Tune R is computed only by `score_tune_only.py`.
- `check_submission.py --split validation` is not called. That kit command reports validation points and per-condition loss ratios. Validation coverage is checked by comparing submission ids to `validation/queries.csv` only. The test-split checker is still called; that split does not calculate an event score.
