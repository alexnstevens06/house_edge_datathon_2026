# House Edge curve_C_lr003_s2 parent, float64 CPU inference

- Team ID: not supplied
- Round: main
- Forecast file: generated separately. This ZIP has no cached forecast.
- Code version: original parent checkpoint SHA-256 `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71`
- Config SHA-256: `a715bfe05eb00494065c7dfcc3ad6d71a3b2bc4ee6788006cf07ce86f3e43695`
- Environment: `python` in `reproduce.json`
- Status: separately staged candidate. Not a release. Not a production replacement. Not organizer acceptance. An independent review is still required. A study-only re-review pass does not clear this package.

## What is pinned

The weights and the checkpoint config are the original `experiments/cnn03_convergence/runs/curve_C_lr003_s2/checkpoint.pt` bytes. That file is weights-only (`config` and `state`). It is not a +10 continuation, a +20 continuation, or a blend. The saved config is seed 2, lr 0.003, wd 0.00895, bs 512, epochs 30, drop 0.3, h 128, layers 4, W 64, pct_start 0.3, eval_every 2, and the original `out` path. It has no `kernel`, `model`, `phase`, or `no_signal` field. Those fields were not added. Kernel 3 is the measured `blocks.*.conv.weight` shape `(128, 128, 3)` on all four layers, and the network is built with that constant only after the shape check. All 43 tensors are finite float32 on disk and float64 after `model.double()`.

Tune-only context, not a new selection: the study CPU float64 tune R of this parent is 0.30980333573689856 on 384 tune rows. That number was already measured before this package. This package does not compute confirm, full, or test R to decide anything. Seeds, epochs, and blend weights were not refit.

## Setup

- Python target: CPython 3.13 with the CPU stack named in `requirements.txt`. The lock text is the canonical 11-package lock copied byte-for-byte from the existing CNN03 seed 0 ZIP. This package does not download it.
- Dependencies: `numpy==2.5.4` and `torch==2.11.0+cpu`. Organizer kit text says the root pins are preinstalled and the rerun is then offline. The live guide also describes a standard-library image, an offered environment, and offline wheels. That conflict is unresolved here. No wheel is bundled, so the archive stays under 100 MB.
- Direct CPU wheel, not downloaded by this package: `torch-2.11.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl` SHA-256 `45025d7752dbc6b4c784c03afaee9c5f19730ce084b2e43fc9a2fe1677d9ff86`.
- Hardware: 2 CPU cores, 16 GiB RAM, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`. No GPU.
- Data: `/data` is the public round directory. The default command reads only `/data/test/history.csv` and `/data/test/queries.csv`.

`organizer_approval_claimed` is false. The command checks `config/config.json` and `checkpoint.pt` against the pinned SHA-256 values, checks 43 keys, shapes, kernel 3, finite float32 storage, casts with `model.double()`, and runs in float64. Softmax is not cast back to float32. Labels, conditions, targets, reference files, and the signal column are not used. The process does not train and does not write a checkpoint.

## Recreate the forecast

Run from this directory:

```sh
python predict.py --data /data --out /output/submission.csv
```

`reproduce.json` is that command with image `python`. `--split val` reads `validation/history.csv` and `validation/queries.csv` for a local structural check. The default split remains `test`.

- Required files: `checkpoint.pt`, `config/config.json`, `model.py`, `data.py`, `predict.py`
- Settings: `config/config.json` (equal to the checkpoint config)
- What it does: loads the original curve_C_lr003_s2 parent and predicts
- Training command: none

If the config hash, the checkpoint hash, the key set, a shape, the saved dtype, or finiteness does not match, the process exits 2 and writes no CSV.

## Local check

```sh
python check_submission.py path/to/submission.csv --data path/to/public
```

Local checks do not submit an entry. This ZIP was not uploaded.
