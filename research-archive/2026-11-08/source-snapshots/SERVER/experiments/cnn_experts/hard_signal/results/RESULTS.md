# hard_signal seed 0

Verdict: **negative**. routed tune R did not beat both the matched control and CNN03 seed 0

Tune R only (lower is better). Confirm and test were not scored.

- Routed specialists tune R: 0.500991
- Matched control tune R: 0.462112
- Canonical CNN03 seed 0 tune R: 0.332064
- Eval: cpu_fp32_softmax_stored_float64. Train: cuda_bf16_weights_fp32_loss_fp32.

Train loss is mean cross-entropy over the labeled slots in each epoch.

- Pretrain final loss 0.7975 over 1110 steps and 18471 history windows.
- Control final loss 0.5791 over 10 steps, labeled slots 768.
- specialist -1: final train loss 0.3461, steps 10, labeled slots 250
- specialist 0: final train loss 0.4291, steps 10, labeled slots 267
- specialist 1: final train loss 0.5733, steps 10, labeled slots 251

Optimizer steps and windows match across control and each specialist. Labeled-slot exposure does not: control uses every cutoff label; each specialist uses only its signal. That difference is the intervention.

CNN03 trains 30 epochs on history windows plus the cutoff window in one loop, batch 512, memory fraction 0.25, and scores full/confirm/tune. This run holds cutoff rows out of pretrain, fine-tunes 10 epochs, microbatches 128 inside a 2GiB cap, and scores tune only.

Wall seconds 66.0. Peak GPU 818 MiB. PID 3480208.

No further seed will be launched.
