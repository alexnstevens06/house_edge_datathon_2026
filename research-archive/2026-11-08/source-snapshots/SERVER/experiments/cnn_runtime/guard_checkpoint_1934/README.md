# House Edge CNN entry

- Team ID: not supplied to this packaging run
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: uploaded separately from this ZIP (local proof CSV is outside this folder)
- Forecast SHA-256: printed by `check_submission.py` on the separate CSV
- Code version: anchor checkpoint SHA-256 `43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80`
- Environment: `python` in `reproduce.json`

## Setup

- Python/runtime version: local proof used CPython 3.12.3. Organizer Python 3.13 was not executed.
- OS tested: Linux, AMD Ryzen 7 3700X, CPU only, 2 threads.
- Dependencies: `requirements.txt` (`numpy==2.5.4`, `torch==2.9.1`). The local proof imported `torch==2.9.1+rocm6.4` and did not install the CPU wheel. That CPU wheel's numeric match is untested.
- Required hardware: 2 CPU cores and 16 GiB RAM. No GPU.
- Put the released data at: `/data` (a public round directory that contains `test/history.csv` and `test/queries.csv`).

The reproduction command reads the mounted round at `--data` and writes one CSV. It loads `checkpoint.pt`, checks `config/config.json` and the checkpoint SHA-256, and runs the pinned network in fp32 on CPU. It does not train, does not read labels, conditions, or the signal column, and does not write a checkpoint.

## Recreate the forecast

Run from: this directory (the ZIP root).

```sh
python predict.py --data /data --out /output/submission.csv
```

- Output file: `/output/submission.csv`
- Required model/config files: `checkpoint.pt`, `config/config.json`, `model.py`, `data.py`, `predict.py`
- Random seeds and settings: `config/config.json` (seed 0, h 128, layers 4, W 64, drop 0.2, no_signal true). No new seed selection.
- What the command does: loads the saved anchor and predicts the test split.
- Measured runtime and hardware: see the packaging `REPORT.md` (outside this ZIP).
- Training command, if separate: none. This package does not train.
- Main-round data/model reused in adaptation, if any: none.

`reproduce.json` is the organizer command: image `python`, command `python predict.py --data /data --out /output/submission.csv`. The same script accepts any local paths in `--data` and `--out`.

If the packaged config hash or the checkpoint SHA-256 does not match the anchor, the process exits 2 and writes no CSV.

## Local check

```sh
python check_submission.py path/to/submission.csv --data path/to/public
```

- Checker result: recorded in the packaging report after the local run.
- Fresh-folder reproduction checked: recorded in the packaging report.

Local validation does not submit an entry. This folder is the code package only.
