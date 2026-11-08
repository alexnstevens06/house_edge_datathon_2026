# Scoped harness difference

Production `experiments/cnn_runtime/fp64_harness.py` and release `experiments/cnn_runtime/fp64_release_harness2025.py` are not modified. `experiments/mlp_cnn_hpo/review/PINNED.json` is not modified, not bypassed, and not pointed at this candidate.

`fp64_candidate_harness.py` is a separate harness for this ensemble. `HARNESS_DIFF.txt` is `diff -u` against the release harness. The numeric threshold stays `1e-8`. Official validation-round R is not computed. `check_submission.py --split validation` is not called.

What the harness still checks:

- no `.float()` call and no tensor-float assignment in `model.py` and `predict.py`
- future ticks do not change the cutoff window on any of the 8 machines; tick t-1 does
- MISSING is index 5 with mask 0, and payout 0 is index 0
- slots are sorted; labels and conditions are not opened; the signal column is dropped
- a missing split fails closed
- both parents: 43 float64 parameters after `model.double()`, kernel shape `(128, 128, 3)`, float64 mask and tick inputs, float64 logits and softmax, autocast off, tensor float method not monkeypatched
- cold subprocess repeat of the 1536-row test CSV is bitwise
- validation coverage is 768 ids and a finite simplex, with targets not opened
- kit format check on the test CSV only
- each parent refuses a wrong checkpoint hash, a wrong config hash, a signal tensor, an extra key, a wrong conv shape, a non-float32 saved tensor, and a non-finite weight
- an extra component directory and a changed weight manifest are refused
- both component infer seconds and the process RSS are printed; RSS above 2097152 KB fails
- the validation CSV matches the verified `ensemble_equal_s0_s2` map after kit row normalization within `1e-8`
- tune score is 384 rows only, through `score_tune_only.py`

Path notes:

- `ROOT = Path(__file__).resolve().parents[3]` because this file lives under `experiments/cnn03_convergence/candidate_t146_ensemble_s0_s2_20261108/`.
- The harness refuses an `--out` directory outside that folder.
- The default entry is `unpacked/`, the ZIP extract.
- Temp copies stay in this folder's `_work`.
