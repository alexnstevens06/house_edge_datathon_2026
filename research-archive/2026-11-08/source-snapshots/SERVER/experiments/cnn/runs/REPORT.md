# CNN workstream report (server, 2026-11-07)
Model: causal dilated temporal CNN (kernel 3, dilations 1..8) shared across the 8 machines. Each block mixes channels across machines using a mean over the other machines at the same tick. Inputs are payout embeddings, an explicit observed mask (missing = its own token plus mask 0, never treated as payout 0), the actual absolute tick, machine type, signal, and slot. Output: 8x5 softmax for tick t. bf16 autocast, 4 CPU threads, GPU memory capped at 35% of the card.
Training: sliding windows from TRAIN episodes only (pseudo-cutoffs t>=16, features from ticks < t on all machines, labels at t) plus the real train labels at the cutoff. No validation labels or targets are used in fitting.
Selection: lowest tune R across 8 runs. Selected g_h128_L4_W64_d02_e30.
Selected R: tune 0.3675, confirm 0.3874, full 0.3783. Per condition (full): stable 0.277, short_history 0.419, drift 0.390, shared_change 0.426. Baseline pooled_recent full R is 0.946.
Caveat: confirm has already been looked at, so it is not a pristine holdout. 8 configs were tried.
Tests (tests/test_cnn.py, plain runner because the venv has no pytest): 4/4 PASS. They cover window causality for all machines, temporal causal conv, missing vs 0, and checkpoint reload with CPU 2-thread test prediction (0.69s; finite values, 8x5 shape, sums to 1, query IDs aligned; max diff vs GPU bf16 CSV 8.3e-3).
Kit checker: all 16 CSVs pass (runs/checker.txt).
Issues: run b (h128, W128) hit OOM under the 35% cap and has no result. Single seed only. No packaging yet (reproduce.json, requirements, /data path handling). Training curves were still improving at 30 epochs.
