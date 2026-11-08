# frozen heads seed 0

Verdict: **negative**. routed tune R is not below both the frozen trunk and the shared head

Tune R only, CPU float64 softmax with row renormalization. Lower is better.

- Frozen trunk baseline: 0.358981
- Matched shared head: 0.355801
- Routed signal heads: 0.358978

Trunk unchanged: True. Trainable trunk params: 0. Each head: 30 params.
Schedule: {'epochs': 10, 'lr': 0.001, 'wd': 0.0, 'bs': 512}.
Control final loss 0.8021 over 10 steps.
Specialist -1 final loss 0.6937, steps 10, labeled slots 250.
Specialist 0 final loss 0.7516, steps 10, labeled slots 267.
Specialist 1 final loss 0.9573, steps 10, labeled slots 251.

Device NVIDIA GeForce RTX 5080. PID 102932. Peak GPU 189 MiB. Seconds 3.0.

No further seed will be launched.
