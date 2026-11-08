# CNN03 seed 0 float64 — first submission build (2025-11-08)

Manual upload only. This workspace did not open the form, did not upload, and did not submit. The parent session already had the main Google Form signed in on the TAMU account. That form has no Team field. The manual upload is the record-email checkbox, the test CSV, and this ZIP.

This package does not claim full organizer approval. A coordinator fresh-ZIP sandbox and an independent review of this new archive are still pending.

## Files for the form

| Item | Path | SHA-256 | Size |
| --- | --- | --- | --- |
| Code ZIP | `experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip` | `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599` | 1,377,335 bytes |
| Test forecast CSV (not inside the ZIP) | `experiments/cnn_runtime/fp64_release_out2025/cnn03_test_a.csv` | `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d` | 197,886 bytes, 1536 ids |

`reproduce.json` is unchanged: image `python`, command `python predict.py --data /data --out /output/submission.csv`.

Older ZIPs were left in place: `anchor_1929.zip`, `anchor_py313_cpu211_fp32.zip`, `fp64_anchor_snapshot1955.zip`, `fp64_cnn03_snapshot1955.zip`.

## ZIP members (9)

`README.md`, `reproduce.json`, `requirements.txt`, `checkpoint.pt`, `config/config.json`, `model.py`, `data.py`, `predict.py`, `provenance/provenance.json`.

No cached CSV, public data, venv, wheel, or cache is inside the archive. Built with `kit/build_entry.py --source experiments/cnn_runtime/fp64_cnn03_entry --out experiments/cnn_runtime/cnn03_s0_fp64_first2025.zip` on `experiments/cnn_runtime/py313_verify_1936/env/bin/python`.

## Model bytes unchanged

| File | SHA-256 | Matches canonical |
| --- | --- | --- |
| `checkpoint.pt` | `ce831b9b41653ae533ba05a64bb79ead340279fb93697caa5ab85c40bcc9203f` | yes |
| `config/config.json` | `6169196bc0827286d5bcbd931980727e0442b1ffdef664cb7f7b319cad4eb4f5` | yes |
| `model.py` | `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a` | yes |
| `data.py` | `af23684b5348c53dafc0e3870706da7ed3dc64c852537371ac7c0ee97d916133` | yes |
| `predict.py` | `a2aa00a32b0af42445014fbe8980cfbbb2afc5644a474c10d48bdf43fc31e6c2` | yes |

Those five files were not edited. Explicit float64, no `Tensor.float` patch, signal off, h 128, layers 4, W 64, dropout 0.3, seed 0. Seed 0 is the tune selection, not a best-confirmation pick.

Docs that did change: `README.md` SHA-256 `3c25c923a8e1c063dff256b9f40621d3c879f2b271c34cbdfa6f28ac55e9fb73`, `provenance/provenance.json` SHA-256 `4975f2d78b6bc4c92d3640a5310d699ea0c082a8daa67c9f79a14eb27f3f6f3d`. `requirements.txt` was already the 11-package lock and was not rewritten: SHA-256 `cf701c4d2f312e39a8883c114ed354106fbb7b7d0cd5968f99cc6bd8746a0cab`. `reproduce.json` SHA-256 `cb91250a431dc92a45ceda7bf66df8006a4ecbce2b8e393f52f7284fd22246a1`.

## Runtime and lock

Server interpreter `experiments/cnn_runtime/py313_verify_1936/env/bin/python`: CPython 3.13.13, `torch==2.11.0+cpu`, `numpy==2.5.4`, 11 lock packages. CPU threads 2. No GPU, no training, no new install.

Direct CPU wheel (not bundled): `torch-2.11.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl` SHA-256 `45025d7752dbc6b4c784c03afaee9c5f19730ce084b2e43fc9a2fe1677d9ff86`. Recorded in `README.md` and `provenance/provenance.json`. `organizer_approval_claimed` is false.

## Harness

New harness only: `experiments/cnn_runtime/fp64_release_harness2025.py` (same depth as `fp64_harness.py`, so `parents[2]` is the repo root). It runs CNN03 only. References are `experiments/cnn_runtime/cnn03_release_refs2024/tower_cnn03_a.csv` and `tower_cnn03_val.csv`. Tests and the `1e-8` cutoff are unchanged. A missing reference still exits 1.

Log: `experiments/cnn_runtime/fp64_release_harness2025.log`. Exit file: `experiments/cnn_runtime/fp64_release_harness2025.exit`. **Exit code 0.** `HARNESS_OK`.

- `unit_ok`, `dtype_ok` (43 float64 params), `negative_ok`
- Cold repeat bitwise SHA-256 `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d`, 1536 test ids
- Checker: test 1536 valid, validation 768 valid
- `VALIDATION_R` mean loss ratio `0.368454536586`, technical points `43.8450809989` / 60
- Own-reference max abs after row norm: test `9.99200722163e-16`, val `4.99600361081e-16`, both under `1e-8`

The original `experiments/cnn_runtime/fp64_harness.py` was not edited. Its earlier run in `server_full_harness2023.log` still exited 1 only because the compare target `experiments/mlp_cnn_hpo/review/fp64_tower_s0_a/test_predictions_cpu.csv` is outside this scope. That original file is not claimed as a whole-harness pass.

## CAVEAT — dependency route

`kit/README.md` line 333, `kit/RULES.md` line 55, and `kit/SUBMISSION_README.md` line 23 say organizers prepare the pinned root `requirements.txt` before the offline prediction run. The cached participant guide instead says the standard `python` image is Python 3.13 plus the standard library, and that extra libraries use an offered environment or compatible bundled wheels plus an offline setup script. Those statements conflict. This package does not invent which text wins and does not treat either as an organizer assurance. The ZIP is conditional on the kit pre-provision route. The CPython 3.13.13 CPU stack above was verified on this server. No wheel is bundled.

## Still pending

Coordinator fresh-ZIP sandbox of `cnn03_s0_fp64_first2025.zip`, plus review. Not an organizer acceptance. Not a claim that the original two-package harness passed.
