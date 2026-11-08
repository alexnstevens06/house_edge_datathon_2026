# FP64 integration status

Not a release. An independent reviewer still has to assess this code. No upload was made. Existing archives and the FP32 `entry/` package were left in place.

## Packages

| Package | Directory | ZIP | ZIP bytes | ZIP SHA-256 |
| --- | --- | --- | --- | --- |
| Anchor | `experiments/cnn_runtime/fp64_anchor_entry/` | `experiments/cnn_runtime/fp64_anchor_entry.zip` | 1375476 | `f8441494d806581939d7be09bf29fe02c31d908c155aa1eb5443d8edaf555953` |
| CNN03 seed 0 | `experiments/cnn_runtime/fp64_cnn03_entry/` | `experiments/cnn_runtime/fp64_cnn03_entry.zip` | 1376720 | `0ea5b0d2fd3db1b6df861037bfb6a7825a2676e988ee3d199f8e37820c7cc319` |

`kit/build_entry.py` built both ZIPs. `inspect_archive` returned only `image=python` and `command=["python","predict.py","--data","/data","--out","/output/submission.csv"]`. Both archives are under 100 MB and contain no wheels and no forecast CSV.

Checkpoint bytes were copied, not rewritten.

- Anchor checkpoint `43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80`
- Anchor config `c9022a4305be1c4124e6a116cd0cf3132fb5624a206f9b5571090ca3c4769d2f` (drop 0.2, `no_signal` true)
- CNN03 checkpoint `ce831b9b41653ae533ba05a64bb79ead340279fb93697caa5ab85c40bcc9203f`, copied from `experiments/mlp_cnn_hpo/server_snapshot/runs/final_cnn03_s0/checkpoint.pt` (the path named by `review/PINNED.json`)
- CNN03 config `6169196bc0827286d5bcbd931980727e0442b1ffdef664cb7f7b319cad4eb4f5` (drop 0.3, kernel 3, seed 0, no anchor dropout)
- Canonical CNN source pin `b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5`
- Packaged `model.py` `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a`
- Packaged `data.py` `af23684b5348c53dafc0e3870706da7ed3dc64c852537371ac7c0ee97d916133`
- Anchor `predict.py` `8d97923a4bf3a2f3f2df562a07571518b99489d6723db4bd3f6a02b247316f33`
- CNN03 `predict.py` `a2aa00a32b0af42445014fbe8980cfbbb2afc5644a474c10d48bdf43fc31e6c2`

The forward keeps integer payout indexes. Mask, tick, and `arange` use the embedding dtype after `model.double()`. Softmax is not cast to float32. There is no `Tensor.float` assignment.

## Tower evidence

Interpreter: existing `.venv` CPython 3.12.3, `torch 2.9.1+rocm6.4`. `experiments/cnn_runtime/py313_verify_1936/env/bin/python` exists and does not import torch yet, so this tower did not run the CPython 3.13 proof. CPU was the Ryzen 7 3700X path already used here, with `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`. No GPU.

`experiments/cnn_runtime/fp64_harness.py` exited 0. It uses `sys.executable`. A compare above 1e-8 exits 1.

| Output | SHA-256 | Notes |
| --- | --- | --- |
| `fp64_out/anchor_test_a.csv` and `_b` | `02f08b8d44a2d57d798c5655f1519eae1c5c23aafd4e41bb1bd6be2b68b44710` | bitwise repeat, 1536 ids, 197493 bytes. Checker valid. Infer about 5.2–5.6 s, max RSS about 1.32e6 KB |
| `fp64_out/anchor_val.csv` | `d12c54b17fcb84269e1846bc3374cde89461a5e993feafb54f517ba55afa4121` | 768 queries. Official `mean_loss_ratio=0.403149380829`, points `42.7609496321/60` |
| `fp64_out/cnn03_test_a.csv` and `_b` | `0f6243cf989d08d4410f50fd163fa962298e7eaa44d4a0bce9d7b7bd9eb9b266` | bitwise repeat, 1536 ids, 197868 bytes. Checker valid. Infer about 5.5–5.6 s, max RSS about 1.32e6 KB |
| `fp64_out/cnn03_val.csv` | `c31f65c9bdc2de74f750e43cdda2954b649b09d4de4eee9407a0407367068d35` | Official `mean_loss_ratio=0.368454536586`, points `43.8450809989/60` |

CNN03 versus the reviewer float64 diagnostic, max abs after kit normalization:

- Tower `review/fp64_tower_s0_a` validation and test: `0` (bitwise)
- Server `review/fp64_server_s0_a` test: `9.99200722163e-16`
- Server `review/fp64_server_s0_a` validation: `4.99600361081e-16`

Both are under 1e-8. The harness also refused a missing split, a tampered checkpoint, a tampered config, and a checkpoint rewritten with `no_signal` false plus `sig.weight`, and it checked causality on all 8 machines, missing-versus-zero, slot order, no label/condition/signal reads, float64 parameters, float64 mask and tick inputs, float64 logits and softmax, and an unchanged `Tensor.float`.

## Commands

From the repo root:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python experiments/cnn_runtime/fp64_harness.py
.venv/bin/python kit/build_entry.py --source experiments/cnn_runtime/fp64_anchor_entry --out experiments/cnn_runtime/fp64_anchor_entry.zip
.venv/bin/python kit/build_entry.py --source experiments/cnn_runtime/fp64_cnn03_entry --out experiments/cnn_runtime/fp64_cnn03_entry.zip
```

Default contract command, from either entry directory:

```sh
python predict.py --data /data --out /output/submission.csv
```

Validation check, still without reading labels inside the package:

```sh
python predict.py --data /data --split val --out /output/validation.csv
```

## Pending

- Server strict sandbox of these two new ZIPs. The earlier server sandbox proved the unchanged FP32 entry only (`anchor_py313_cpu211_fp32.zip`, SHA-256 `9d9e20b231dd08a9cd4d728e1526a43c7ab4df59d336c8002fa7961c0d518c2f`, 1375494 bytes, reference SHA `9ea67f81…`, 1.471241 s, 610848 KiB). That ZIP is preserved and is not this FP64 package.
- Independent review. This note does not clear a release.
- Tower CPython 3.13 import. The local `py313_verify_1936` environment did not have torch at test time.
- The named server freeze `requirements_py313_cpu211_frozen.txt` is not in this workspace. Requirements name `numpy==2.5.4` and `torch==2.11.0+cpu` plus the CPU wheel URL and SHA-256 `45025d7752dbc6b4c784c03afaee9c5f19730ce084b2e43fc9a2fe1677d9ff86`. Transitive versions were not invented.
- Kit preinstall versus the live guide's standard library, offered environment, and bundled wheels is still unresolved.
