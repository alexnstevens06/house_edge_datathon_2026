# experiments/cnn_experts shared contract (READ-ONLY for author sessions)
Owner: Alex Stevens, requested 2026-11-07 7:32 PM CST (t58/t59). Plan frozen before any metric.
Repo root (tower AND server, same path): /home/alexn/Documents/code/datathon_2026/house-edge
Write ONLY inside experiments/cnn_experts/<your subfolder>. Never modify kit/, other experiments/, git state. No commits/push/submits. No synthetic data. No random HPO sweep. No API keys, no web.
## Incumbent CNN03 (control)
h128 L4 W64 dropout .3 lr .00197 wd .00895, 367109 params. Source experiments/mlp_cnn_hpo/source/base_cnn.py sha256 b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5. Checkpoints on SERVER experiments/mlp_cnn_hpo/runs/final_cnn03_s{0,1,2}/checkpoint.pt. Tune R: s0 .3321, s1 .3465, s2 .3334 (full .3685/.3790/.3333). Lower R is better. CPU fp32 evaluation. Fixed-confirm split is SEEN, not pristine; report only after promotion, labeled "seen".
## Read first
kit/README.md DATA.md RULES.md EXPERIMENTS.md, kit/slot_forecasting, experiments/signal_networks (SPEC.md, source), experiments/cnn, experiments/mlp_cnn_hpo/source and runs/*.md. Reuse the exact CNN03 data pipeline, TRAIN/VAL split IDs, R metric. Write SPLIT_IDS.json + PLAN.md (frozen, hash recorded) BEFORE metrics.
## Leakage rules (critical)
- Historical pseudo-cutoff windows have NO legitimate signal. Never route/condition them on the final query signal. No signal backfill.
- Pretrain shared clean no-signal CNN on real TRAIN history only. Signal experts/router fine-tune ONLY on true actual TRAIN cutoff rows with legitimately provided signal+label (verify in kit). If such labels don't exist, STOP and write BLOCKER.md; do not invent.
- Held-out VAL never fits anything (no early stopping on VAL; use the CNN03 tune protocol exactly).
- Payout class order 0,1,2,5,20. Machine type mapping balanced=0 jackpot=1 steady=2 (verify against kit). Missing mask distinct from 0.
## Required tests (pytest, in your subfolder)
all-8-machine future mutation invariance; label mutation; signal mutation does not change historical features; query slot order invariance; missing vs 0 distinguished; router probabilities normalized (soft) / exact hard matching (hard); sparse-group fallback to shared CNN03-style model; shared control uses the same pipeline.
## Compute (SERVER-LOCAL venue, Alex t77)
You are running ON server. Train locally on this host GPU; never ssh/rsync to any other host. The coding CLI has CUDA hidden; when launching training set CUDA_VISIBLE_DEVICES=0 explicitly in the launch command. Use the python env the CNN03 runs used here. Max 2 jobs total on server GPU, each <=2GB VRAM, 2 CPU threads; check free memory first; physical 16GB; another synthetic utility job may use <=3GB. Do NOT use tower GPU. Run detached (nohup/tmux) with logs.
## Promotion rule (Alex t53/t54)
First FULL training run (>= comparable 30 epochs, or honestly defined true-cutoff fine-tune budget plus a matched shared control trained the same way). If tune R not better than matched incumbent seed -> DROP, no more seeds. If better -> 2 more full seeds, require 3/3 wins.
## Report
results/RESULTS.md + results.json: timings, steps, params, source hashes, train loss, tune R per condition and seed, verdict. Milestones in STATUS.md (append one line with CST timestamp at: plan frozen, tests pass, training started, first result, final).
Deadline: first full result 7:55 PM CST, hard cap 8:10 PM CST.
