# House Edge CNN03 seed 0, float64 CPU inference

- Team ID: not supplied
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: uploaded separately. This ZIP has no cached forecast.
- Code version: checkpoint SHA-256 `ce831b9b41653ae533ba05a64bb79ead340279fb93697caa5ab85c40bcc9203f`
- Environment: `python` in `reproduce.json`
- Status: CNN03 seed 0 float64 package using the verified 11-package lock. This file does not claim full organizer approval. A coordinator check of this fresh ZIP is still pending.

## Setup

- Python target: CPython 3.13.13. The server interpreter `experiments/cnn_runtime/py313_verify_1936/env/bin/python` imported `torch==2.11.0+cpu` and `numpy==2.5.4` with all 11 lock packages present.
- Dependencies: root `requirements.txt` SHA-256 `cf701c4d2f312e39a8883c114ed354106fbb7b7d0cd5968f99cc6bd8746a0cab` (filelock==4.2.9, fsspec==2026.9.0, jinja2==3.1.6, markupsafe==3.0.4, mpmath==1.3.0, networkx==3.7, numpy==2.5.4, setuptools==81.0.0, sympy==1.14.0, torch==2.11.0+cpu from the direct CPU wheel, typing-extensions==4.16.0). Direct CPU wheel `torch-2.11.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl` SHA-256 `45025d7752dbc6b4c784c03afaee9c5f19730ce084b2e43fc9a2fe1677d9ff86`. The wheel is not in this ZIP.
- CAVEAT: `kit/README.md` line 333, `kit/RULES.md` line 55, and `kit/SUBMISSION_README.md` line 23 say organizers prepare the pinned root `requirements.txt` before the offline run, with image `python`. The cached participant guide instead says the standard `python` image is Python 3.13 plus the standard library, and that extra libraries use an offered environment or compatible bundled wheels plus an offline setup script. Those statements conflict. This package does not choose a winner and does not treat either text as an organizer assurance. It is conditional on the kit pre-provision route. The CPython 3.13.13 CPU stack named above was verified on this server. No wheel is bundled, so the archive stays under 100 MB.
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
