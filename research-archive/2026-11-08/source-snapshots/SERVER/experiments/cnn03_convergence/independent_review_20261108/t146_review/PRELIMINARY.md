# Low-rate study, preliminary tip review

**Status: PRELIMINARY. No final clearance.**

**Current tip: BLOCK for launching the four jobs.** The training step calls the network with the label tensor included. `CNN.forward` accepts four tensors. A CPU call with the trainer's five-tensor batch raises `TypeError: CNN.forward() takes 5 positional arguments but 6 were given`. The FP32 preflight drops the label (`model(*batch[:-1])`). The historical trainer also passes four tensors (`model(x, t, ty, sg)`). This tip has no saved low-rate checkpoints and no tune metrics.

Reviewed at `2026-11-08T07:44:59Z`. There is no `HANDOFF_READY`, no `tests/`, no `SPLIT_IDS.json`, and no `SOURCE_SHA256.txt`. The tip below is the tree at that time, not a frozen-source pass. `ensemble_blend_20261108` is not on this machine and was not fetched.

## Tip hashes

Protocol `COORDINATOR_PLAN.md` SHA-256 `03d141087ac950e8026ebaeb6afcc4f0b83371508beee0961ecd993cc9f9e8f3` (3646 bytes, mtime `2026-11-08T07:32:52Z`). The manifest records the same digest.

| File | SHA-256 | mtime UTC |
| --- | --- | --- |
| `PLAN.md` | `74a4c9bf3acb6cf33dd9f9511435b1a58e7c906fb7b2057a45dcad8e18274d35` | `2026-11-08T07:38:13Z` |
| `MANIFEST.json` | `5d1963aab728ec0d0afdcfb6434c5614f0199453de0d70ffd84d0cbf6dd4e463` | `2026-11-08T07:38:13Z` |
| `source/contract.py` | `d908aefe6eab70cbedf70d0bf2b7fc2d2132cd5bafce7e3d844fc78cccaa3025` | `2026-11-08T07:38:13Z` |
| `source/train_lowrate.py` | `fd0b50522d6c1dd53e8ae5c1cd129a3ab4815b94bccce98136b3629be6146f21` | `2026-11-08T07:39:08Z` |
| `source/score_fp64.py` | `0300e848a50029c5b3441bda8418cfbca38bd7c37b47096bab9adc29de978fef` | `2026-11-08T07:39:08Z` |
| `source/preflight_fp32.py` | `72249b98576d781a1f5d028805c95dbbf04c0997b1691557f6a67f6cddae2158` | `2026-11-08T07:39:08Z` |
| `source/preflight_fp64.py` | `adb42b4c65545f6d2ac89062a49f87a22ff32ff59f702b7312dad22840008325` | `2026-11-08T07:39:08Z` |

Protected bytes still match: current-best ZIP `dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29`, CSV `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6`, fallback ZIP `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599`, CSV `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d`, `PINNED.json` `cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb`, `fp64_harness.py` `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2`. This review does not reopen that package.

## What the tip gets right

The declared matrix matches the protocol: parents `curve_C_lr003_s2` and `curve_control_s2`, server `2e-5`, tower `5e-5`, five epochs, FP32, batch 128, AdamW weight decay `0.00895`, clip 1, drop 0.3, RNG `20261108`, saves at epochs 1, 3, and 5. `ceil(18567/128)=146` and `146*5=730`. Four jobs would be 2920 steps. The original seed-2 summary already records 18567 train windows; this tip still rechecks that count before training. The plan states that platform is confounded with rate.

`AdamW` is built with `lr=host_lr(host)` after `load_state_dict`. It does not read the parent `config.lr`. The loaded parents are weights-only: primary lr `0.003` seed 2, control lr `0.00197` seed 2, no kernel field. The saved training config stores `train_lr` separately from that base lr. `run_errors` rejects a `train_lr` of `0.003` or `0.00197`, a parent-hash swap, a bad key, conv extent 5, float16, a non-finite tensor, epoch 2, a short epoch, a missing tune id, and an id outside tune/confirm. A blob shaped like the trainer's epoch-1 server payload, and an epoch-5 tower control payload, pass. The training file has no OneCycle. Its only `autocast` text is the saved flag `autocast: False`.

RNG is seeded after the parent weights load. The trainer writes the checkpoint and the post-run snapshot and does not score. The scorer refuses a run outside the study folder, requires that snapshot, and returns only `tune_R` plus counts. It does not call the official validation checker.

A real primary-parent forward through `fresh_fp64_cnn03_1956/model.py` stayed on CPU: 43 float64 parameters, float64 logits, float64 softmax, autocast off, `Tensor.float` not reassigned. A four-argument `base_cnn` forward is float32. `masked_loss` calls `.float()` on the logit tensor. That is a cast inside the shared loss, not a monkeypatch, and the FP64 scorer does not use it.

## Gaps to fix before a frozen review

1. **Training call (BLOCK).** `train_lowrate.py` line 196 is `logits = model(*batch)` where `batch` is `(X, T, TY, SG, Y)`. The first step cannot run. The preflight is not a substitute for that line.

2. **Tower portability (CONCERN).** `gpu_processes()` runs `nvidia-smi` and exits if that command fails. There is no `rocm-smi` or HIP path. The tower `5e-5` pair cannot start on gfx1031 with this file. CPU training is refused when `torch.cuda.is_available()` is false, which is the right direction, but the NVIDIA-only probe makes the source not portable to the assigned tower device.

3. **Host flag selects the rate (CONCERN).** `--host tower` forces `5e-5` even if the process is on the server, and the reverse. The device name is recorded after the fact. Nothing checks that the flag matches the GPU that is actually present, so a mislabeled launch changes the rate.

4. **Replay (CONCERN).** Output directories use `exist_ok`. `loss.jsonl` is opened append. `torch.save` overwrites an existing epoch. There is no exclusive create, no job registry, and no refusal when the epoch file already exists. The CLI only offers the four host/arm pairs, but each pair can be pointed at a new directory or replayed over an old one.

5. **Scorer is not runnable on this tip (CONCERN).** `split_contract` reads `SPLIT_IDS.json`, which is absent. The scorer also does not require the forward set to be the full 768 validation queries. A prediction dict that contains only the 384 tune ids would pass `tune_id_errors`. There is no frozen test module exercising wrong hash, config, keys, shape, dtype, non-finite values, non-tune ids, or budget. The checks exist as functions; this review called them synthetically. That is not a substitute for tests frozen before the jobs.

6. **Not written yet.** No selector implements the protocol tie rule (improvement at least `1e-6`, tie keeps the current best). No low-rate metrics exist. The ensemble and blend stream is absent here.

`assert_input_pins` rechecks the protocol, the manifest source hashes, and the six parent files. It records the production pin and harness digests in the manifest but does not rehash those two files itself. This review did.

## Commands actually run

CPU only, 2 threads, `CUDA_VISIBLE_DEVICES` empty. CPython 3.13.13, torch 2.11.0+cpu. No author test suite exists, so none was run. The FP32 GPU preflight was not run. No training, no tune R, no confirm/full/test metric.

```sh
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
experiments/cnn_runtime/py313_verify_1936/env/bin/python \
  experiments/cnn03_convergence/independent_review_20261108/t146_review/probe_tip.py
```

Exit 0. The probe's exit means the reviewer script finished. It does not clear the study. Synthetic guard expectations after the trainer-shaped fixture: clean epoch-1 and tower epoch-5 empty; wrong hash, parent rates, shape, dtype, non-finite, budget, and non-tune cases caught. Five-argument forward: `TypeError`. FP64 forward: float64 logits and softmax.

## Still required later

A frozen-source review after `HANDOFF_READY`, frozen tests, and `SPLIT_IDS.json`. Then the four saved checkpoints, tune-only scores, and only then any package. Packaging still needs an independent fresh ZIP, offline 2 CPU at or under 2 GiB, network none, cold repeat, and cross-host normalized max abs `1e-8` before any improved delivery. The ensemble stream remains not yet reviewed.
