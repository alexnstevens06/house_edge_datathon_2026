# Frozen feature heads, seed 0

Device: AMD Radeon RX 6700 XT, torch 2.14.0+rocm10.1.0, pid 33141.
Trunk sha256 e482eeaa021d1b4b1a8e8fcf80b0cf0be820a755e2f3213b16c9a57fd42c8760, state hash unchanged (True).
Trainable parameters per head: 645. Frozen trunk parameters: 367109.
Cutoff labels: {-1: 250, 0: 267, 1: 251}. Steps: 10. Epoch-0 logits match the trunk: True.

Tune R only (lower is better). Confirm and test were not scored.

- Frozen baseline: 0.358981
- Shared head: 0.350971 (delta vs baseline -0.008009)
- Routed heads: 0.365661 (delta vs baseline +0.006680, vs shared +0.014690)

Final train loss: control 0.7869, signal -1 0.6687, signal 0 0.7289, signal 1 0.9216.
Peak GPU memory during the run: 122.2 MiB. Feature pass: 1.38s.

No further seed is trained. The tune split was already used for CNN03, each head sees about 250 labels, and the budget is 10 full-batch steps. Deltas of this size are not a reason to continue.
