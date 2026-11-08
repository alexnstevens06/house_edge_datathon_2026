# Low-rate handoff (source frozen before GPU jobs)

Written 2026-11-08 01:48 CST after contract tests and both preflights passed. The files listed in `SOURCE_SHA256.txt` are frozen. This note is not one of those files. Do not edit the frozen files to refresh this note.

Protocol `COORDINATOR_PLAN.md` SHA-256 `03d141087ac950e8026ebaeb6afcc4f0b83371508beee0961ecd993cc9f9e8f3`.

## Copy set

Copy this study folder's frozen files to the tower. Do not copy `runs/`. Do not copy `author.pid`, `author.log`, or `AUTHOR_PROMPT.txt`. Parents are not inside this folder. Stage the two immutable seed-2 checkpoints into the tower tree at `experiments/cnn03_convergence/ensemble_blend_20261108/inputs/` and pass those paths to `--parent`. This author does not write that folder and does not start tower processes.

| Role | SHA-256 | Original path on the server |
| --- | --- | --- |
| primary parent | `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` | `experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt` |
| control parent | `47694373a79521910aedce432a000a9f21ea57c4e3d921ce0d78ecfdc1b869f9` | `experiments/cnn03_convergence/runs/curve_control_s2/checkpoint.pt` |

`SOURCE_SHA256.txt` (SHA-256 of that file is recorded below the listing; recompute it after copy):

```
03d141087ac950e8026ebaeb6afcc4f0b83371508beee0961ecd993cc9f9e8f3  COORDINATOR_PLAN.md
463ea1dd6317a0fe64e009d8b32fd751ef756cc5fe0c7c102af1c4474f606b3e  PLAN.md
5d1963aab728ec0d0afdcfb6434c5614f0199453de0d70ffd84d0cbf6dd4e463  MANIFEST.json
21d0b34335349e8348d9a3a0edc578e063b08bc7e80e719fd2c93b4139dd566a  SPLIT_IDS.json
d908aefe6eab70cbedf70d0bf2b7fc2d2132cd5bafce7e3d844fc78cccaa3025  source/contract.py
206ee996bc6a48b34a5e4027e90542e5cb63a787fe36e7ab7aab1f90f0b04066  source/train_lowrate.py
138676a7fbca1be71b9348c85aac2eed444c667b5d899a2b017df01d20e463d4  source/score_fp64.py
adb42b4c65545f6d2ac89062a49f87a22ff32ff59f702b7312dad22840008325  source/preflight_fp64.py
ffdfd3d00dcccf216db8c3712efaadc13ccb77d844c7a17afb74b88b7893bd15  source/preflight_fp32.py
b131bcc4caf8ea1b7bce442ab93ff053a0d584eb19a011ed4a37743aadf62c0d  tests/test_lowrate.py
```

`sha256sum SOURCE_SHA256.txt` on the server at freeze time is not stored inside the file. After copying, `sha256sum -c SOURCE_SHA256.txt` from this study directory must pass before any tower launch.

## What the trainer is

Weights-only warm start, then a fresh AdamW. The parents have no optimizer or scheduler. This is not an exact resume. Continuation RNG `20261108` is set after `load_state_dict`, so the paired arms share one dropout and shuffle stream. Fixed LR comes from `--host`: server `2e-5`, tower `5e-5`. There is no `--lr` flag. Weight decay `0.00895`, grad clip `1`, no scheduler, no OneCycle, no ramp. FP32, no autocast. Batch `128` (historical curve runs were bf16 autocast at batch `512`). 18567 train windows, 146 steps per epoch, 5 epochs, 730 steps. Checkpoints at completed epochs 1, 3, and 5 include optimizer, RNG, and provenance, and `POST_RUN_SNAPSHOT.json` is written before any tune score. TF32 is turned off with process-local flags. Each job uses 2 CPU threads, a 0.40 device-memory fraction, and stops if host RSS exceeds 2 GiB. CUDA (the ROCm PyTorch `cuda` device on the tower) is required. A foreign GPU compute process blocks launch. One sibling `train_lowrate.py` from this study may already be on the card.

Platform is confounded with learning rate. Compare primary against control inside one host. Do not claim that one rate is better because of the tower-versus-server difference.

## Preflights already passed on the server

Tests: `.venv/bin/python` (CPython 3.12.13, torch 2.11.0+cu128) printed `LOW_RATE_TESTS_OK failures=0 windows=18567 steps=730`.

CPU float64 probe, `experiments/cnn_runtime/py313_verify_1936/env/bin/python` (CPython 3.13.13, torch 2.11.0+cpu, CUDA hidden): `FP64_PROBE_OK params=43 logit=torch.float64 softmax=torch.float64`.

FP32 probe, one forward and backward at batch 128, no optimizer step, no tune score: `FP32_PROBE_OK loss=1.614211 gpu_mb=1121.1 rss_kb=1451668 logit=torch.float32`. RSS is under the 2 GiB cap (`2097152` KB). Allocated device memory was about 1.1 GB, so the server pair can share the RTX 5080 at fraction 0.40.

## Tower training commands

Run from the repository root on the tower, with the existing interpreter that already trains on gfx1031. Do not install a new torch. Launch only after `sha256sum -c SOURCE_SHA256.txt` passes and the GPU has no foreign compute process. The two arms may run together if memory fits.

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=<existing gfx1031 python>
STUDY=experiments/cnn03_convergence/lowrate_20261108
INPUTS=experiments/cnn03_convergence/ensemble_blend_20261108/inputs

"$PY" "$STUDY/source/train_lowrate.py" \
  --host tower --arm primary \
  --parent "$INPUTS/<staged file whose sha256 is bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71>" \
  --out "$STUDY/runs/tower_primary_lr5e-5"

"$PY" "$STUDY/source/train_lowrate.py" \
  --host tower --arm control \
  --parent "$INPUTS/<staged file whose sha256 is 47694373a79521910aedce432a000a9f21ea57c4e3d921ce0d78ecfdc1b869f9>" \
  --out "$STUDY/runs/tower_control_lr5e-5"
```

The trainer refuses a parent whose bytes do not match the manifest, so the staged filename may differ. Keep the `--out` directories inside the copied study folder. Each run writes `launch.json`, `loss.jsonl`, `epoch_01/`, `epoch_03/`, `epoch_05/`, and `POST_RUN_SNAPSHOT.json` before any score.

## Authoritative scoring, on the server, after the tower runs are copied back

Score only on the server with the existing Py3.13 / torch 2.11 CPU interpreter. Hide the GPU. Two threads. Epochs 1, 3, and 5 only. The output path must stay inside the study folder. This scorer does not compute confirm, full, or test metrics.

```bash
export CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=experiments/cnn_runtime/py313_verify_1936/env/bin/python
STUDY=experiments/cnn03_convergence/lowrate_20261108
for run in tower_primary_lr5e-5 tower_control_lr5e-5; do
  for epoch in 1 3 5; do
    "$PY" "$STUDY/source/score_fp64.py" \
      --run "$STUDY/runs/$run" \
      --epoch "$epoch" \
      --out "$STUDY/runs/$run/scores/epoch_0${epoch}.json"
  done
done
```

The same command, with `server_primary_lr2e-5` and `server_control_lr2e-5`, is the server-pair scorer. Comparison bar is the current best tune R `0.30980333573689856`. Control seed-2 parent tune R is `0.3334821793902223`. The old canonical `0.33206398205344967` is not the bar. Tune-only selection has a multiplicity caveat. No packaging from this handoff.
