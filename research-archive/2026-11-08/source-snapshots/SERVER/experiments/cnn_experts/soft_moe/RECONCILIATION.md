# soft_moe reconciliation (prep only, no training)

Session `01a1b8e2-b683-7c51-98e1-bb9dbf732111`. Written 2026-11-07 20:20:21 CST.
ADDENDUM `experiments/cnn_experts/ADDENDUM.md` sha256
`e2fa1688d10ebe8dfa942dda8210bb4a57b96f5ad3c528b43851416aedfc314a`
is acknowledged in `PLAN.md` and `STATUS.md`. It overrides promotion. This file does not
authorize a launch. No GPU process was started. No seed-0 PID, no epoch, no tune R.

Inherited files kept: `pre_repair_context.md`, `repair_prompt_t83.txt`, `repair_session.log`,
`repair_context_check2.md`, `repair_author.pid`, `session.log`, the first `SPLIT_IDS.json`.

## Provenance (sha256)

| File | SHA256 |
| --- | --- |
| PLAN.md (current) | `f6a46db7aa83b6d74684bf03472288bf9ea011e541e3d0ffcbb877e79645d9c0` |
| PLAN.md draft before this repair | `995e273164cba0a912b2483ae59712f3c58d9a8e431cfc4d5c88966481f505ed` |
| SPLIT_IDS.json | `66081bf9abf6819c6bcf90142838d89ee75fce5f04461182e4fec6dc455d6ae4` |
| runs/splits.json | `8f266f53f5da262c3141e57ea780668b552dc5608b63c2eaeee667cad79891cb` |
| source/soft_moe.py | `c19e1982aba7ded493ebb70aa12a36ec4f4d7ac5d2e1d9630a3a66213f7584da` |
| source/launch_seed0.sh | `698f8629c0b8302edfdf4a2334bdb2f7ad1e0a9f4f06f3e9885b94832a85039c` |
| tests/test_soft_moe.py | `6276b381dd5e8f1bc01288cac08db24b3a979288c688fadf86290787a5f69753` |
| experiments/mlp_cnn_hpo/source/base_cnn.py | `b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5` |
| experiments/mlp_cnn_hpo/source/train.py | `916c6ee1b6e405300ddadbc5650583b250db8baa83a5227fc2437c1b1a55e066` |
| final_cnn03_s0/config.json | `6169196bc0827286d5bcbd931980727e0442b1ffdef664cb7f7b319cad4eb4f5` |

`SPLIT_IDS.json` tune and confirm lists equal `runs/splits.json`. Counts: train 96, validation 96,
test 192, tune 48, confirm 48. Tune and confirm are disjoint. Tune is a subset of validation.
Split ids were not rewritten in this repair.

## Canonical CNN03 seed 0

`final_cnn03_s0/config.json`: model cnn, h 128, layers 4, kernel 3, W 64, drop 0.3, epochs 30,
lr 0.00197, wd 0.00895, bs 512, seed 0, phase final, max_train_windows 0.
`train.py` forces `USE_SIGNAL=False`, builds windows with `base_cnn.build_train`, AdamW,
OneCycleLR on `epochs * ceil(n/bs)` steps, grad clip 1.0, bf16 autocast on CUDA, masked
cross-entropy in fp32, `torch.set_num_threads(2)`, memory fraction 0.25, no validation
early stop. Final scoring moves the model to CPU. Kernel in the CLI is ignored; the CNN
is constructed with kernel 3. Incumbent tune R for this seed is 0.33206398675779764.
Shared trunk parameter count in our tests is 367109.

## What matches

Full TRAIN `X`, `T`, `TY`, and `Y` from `build_train` are tensor-equal to
`base_cnn.build_train` with `USE_SIGNAL=False` (tested). Same `window` rule: ticks `< t` only,
min_t 16, stride 1, missing pad `-1`, cutoff labels from train `labels.csv`.
Shared trunk modules match the no-signal CNN (embedding 6, mask channel, type, slot, tick,
4 causal dilated blocks, head `Linear(2h,h)+GELU+Linear(h,5)`). Phase H uses that trunk for
30 epochs at the CNN03 lr, wd, batch, clip, and OneCycle schedule, with the signal channel
forced to ABSENT on every row, including cutoff rows. Loss for the trunk is masked fp32
cross-entropy. Planned tune eval is CPU fp32, then float64 numpy probabilities, same shape
as `base_cnn.predict`. Selection metric is tune R only (lower better).

## Mismatches fixed in this repair

The draft plan and `PIPELINE` used `ft_epochs=30` and told a win to launch seeds 1 and 2.
The draft also pretrained the mixture and the control as two separate phase-H runs, so the
trunks did not start from one shared state. `evaluate` scored confirm and wrote test CSVs.
`launch_seed0.sh` retried at batch 256 after OOM.

Now: 30 historical pretrain epochs plus 10 true-cutoff finetune epochs (`ft_lr=0.000197`,
lr/10, not in the CNN03 config and not swept). One phase-H state is copied into the mixture
trunk and into the control. Both finetunes use the same cutoff rows, batch, weight decay,
step count, and a reseed of seed 0. Mixture phase C trains only the three experts and the
gate. Control phase C trains only the single head. `reject_seed` and `launch_seed0.sh`
exit 2 for any seed other than 0. Confirm and test are not scored. The OOM path writes
`BLOCKER.md` and does not relaunch.

## Not equivalent to final_cnn03_s0

Do not treat a future soft_moe tune R as a reproduction of CNN03. Phase C does not exist
in `train.py`. The VRAM cap is 2 GiB (`set_per_process_memory_fraction`), not 0.25.
The mixture adds three residual CNN heads and a gate. The signal tensor is not the CNN03
zeros tensor: history stores ABSENT=3, and cutoff rows store legitimate signals
`{-1,0,+1} -> {0,1,2}` for phase C only. Phase H never feeds those stored signals in.
`ft_lr` and the fresh phase-C AdamW/OneCycle are extra. No training has been run, so
there is no measured loss or tune R.

## Signal counts

Test `test_train_windows_match_cnn03_and_signal_counts` asserts labelled TRAIN cutoff
slots are `(250, 267, 251)` for signals `-1`, `0`, `+1`. That is the expected 250/267/251.
`main` exits before training if the counts differ. Historical rows are all ABSENT.
All three groups are above the frozen sparse minimum of 32, so the real run keeps all
three experts. The all-disabled and one-disabled fallbacks are unit-tested.

## Tests

Command (existing pytest from `experiments/random_forest/.venv`, house-edge `.venv`
Python; no install, no network):

```
cd /home/alexn/Documents/code/datathon_2026/house-edge
PYTHONPATH=experiments/random_forest/.venv/lib/python3.12/site-packages \
  .venv/bin/python -m pytest experiments/cnn_experts/soft_moe/tests/test_soft_moe.py -q --tb=short
```

Result: `13 passed in 0.86s` at 2026-11-07 20:20 CST.
Covered: all-8-machine future mutation, all-8-machine label mutation, historical features
unchanged by query-signal edits, query order, missing versus 0, normalized soft blend
including a hand-computed mixture, sparse-group fallback to the shared head, shared
pipeline and 367109-param trunk, CNN03 window equality, signal counts, trunk-copy
equality, seed guard in `reject_seed` and `launch_seed0.sh` for seeds 1 and 2.
The shell guard was invoked only as `bash launch_seed0.sh 1` and `bash launch_seed0.sh 2`.
Seed 0 was not executed.

## GPU

At 20:20:21 CST, `nvidia-smi` reported 16303 MiB total, 2 MiB used, 15841 MiB free.
`pgrep` found no `soft_moe.py` process. Training was not started.

## Status

READY / PREP ONLY. A later coordinator continuation has to launch seed 0. This session
must not.
