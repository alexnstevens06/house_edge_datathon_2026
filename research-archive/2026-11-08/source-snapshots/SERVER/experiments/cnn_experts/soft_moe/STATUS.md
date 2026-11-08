# soft_moe status

2026-11-07 20:09:23 CST — prior session wrote SPLIT_IDS.json (train 96, tune 48, confirm 48) and a draft PLAN (sha256 995e273164cba0a912b2483ae59712f3c58d9a8e431cfc4d5c88966481f505ed). No metrics.

2026-11-07 before 20:13:51 CST — prior session: 9 CPU tests passed via the random_forest pytest install. `source/launch_seed0.sh` was written and not executed. No training PID, no epoch, no tune R.

2026-11-07 20:20:21 CST — ADDENDUM experiments/cnn_experts/ADDENDUM.md (Alex t83, sha256 e2fa1688d10ebe8dfa942dda8210bb4a57b96f5ad3c528b43851416aedfc314a) acknowledged. It overrides promotion, the CONTRACT promotion rule, and the draft plan. Seed 0 only. No seeds 1 or 2. No confirm/test selection.

2026-11-07 20:20:21 CST — repair applied in this session: shared phase-H pretrain copied into both models, finetune cut from 30 epochs to 10, seed!=0 rejected in `soft_moe.py` and `launch_seed0.sh`, promotion and batch-256 relaunch removed. PLAN sha256 f6a46db7aa83b6d74684bf03472288bf9ea011e541e3d0ffcbb877e79645d9c0. SPLIT_IDS sha256 66081bf9abf6819c6bcf90142838d89ee75fce5f04461182e4fec6dc455d6ae4. Details in RECONCILIATION.md.

2026-11-07 20:20:21 CST — CPU tests passed: `PYTHONPATH=experiments/random_forest/.venv/lib/python3.12/site-packages .venv/bin/python -m pytest experiments/cnn_experts/soft_moe/tests/test_soft_moe.py -q --tb=short` → `13 passed in 0.86s`. No network install.

2026-11-07 20:20:21 CST — READY / PREP ONLY. GPU free 15841 MiB of 16303. No soft_moe training process. No PID. No first epoch. No loss. No tune R. This session did not launch GPU training and must not launch seeds 1 or 2.
