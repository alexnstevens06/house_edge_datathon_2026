# House Edge CNN03 seed 0, float64 CPU inference

- Team ID: not supplied
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: uploaded separately. This ZIP has no cached forecast.
- Code version: checkpoint SHA-256 `ce831b9b41653ae533ba05a64bb79ead340279fb93697caa5ab85c40bcc9203f`
- Environment: `python` in `reproduce.json`
- Status: not a release. An independent review is still required.

## Setup

- Python target: CPython 3.13 with the CPU wheel named in `requirements.txt`. This tower package was exercised with the existing project interpreter when that 3.13 environment could not import torch. Do not treat a tower 3.12 run as a 3.13 proof.
- Dependencies: `numpy==2.5.4` and `torch==2.11.0+cpu`. Organizer kit text says the root pins are preinstalled and the rerun is then offline. The live guide also describes a standard-library image, an offered environment, and offline wheels. That conflict is unresolved here. No wheel is bundled, so the archive stays under 100 MB.
- Hardware: 2 CPU cores, 16 GiB RAM, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`. No GPU.
- Data: `/data` is the public round directory. The default command reads only `/data/test/history.csv` and `/data/test/queries.csv`.

The command checks `config/config.json` and `checkpoint.pt` against the pinned SHA-256 values, casts the loaded weights with `model.double()`, and runs this seed's network in float64. The pinned config is dropout 0.3, kernel 3, h 128, layers 4, W 64, seed 0. It is not the anchor dropout. Mask, tick, and position features use the embedding dtype. Softmax is not cast back to float32. `Tensor.float` is not patched. Labels, conditions, and the signal column are not used. The process does not train and does not write a checkpoint. Seeds 1 and 2 are not in this package.

## Recreate the forecast

Run from this directory:

```sh
python predict.py --data /data --out /output/submission.csv
```

`reproduce.json` is that command with image `python`. `--split val` reads `validation/history.csv` and `validation/queries.csv` for a local check. The default split remains `test`.

- Required files: `checkpoint.pt`, `config/config.json`, `model.py`, `data.py`, `predict.py`
- Settings: `config/config.json`
- What it does: loads the saved CNN03 seed 0 checkpoint and predicts
- Training command: none

If the config hash or the checkpoint hash does not match, the process exits 2 and writes no CSV.

## Local check

```sh
python check_submission.py path/to/submission.csv --data path/to/public
```

Local checks do not submit an entry.
