# Anchor runtime report

Not organizer-ready. The hash-locked entry was not changed. No upload was made.

The interrupted comparison command was a session restart, not a permission denial. Work continued. No permission rule was bypassed.

## Pending blockers

1. Tower FP32 versus the server reference CSV is above 1e-8. Max absolute difference after row normalization is `3.742622056712719e-07`. The tower repeat is bitwise. The server run of this same entry is bitwise with the reference. The entry default was not edited to chase that gap.
2. On this Ryzen 7 3700X, `baseline`, `ATEN_CPU_CAPABILITY=avx2`, and the oneDNN/MKL AVX2 caps are bitwise identical to each other and to `first.csv`. The CPU capability PyTorch reports is already AVX2, so those caps do not move the result. `mkldnn` off is a different CSV and farther from the reference (`4.59038415612234e-07`). No FP32 control run here meets 1e-8 against the reference.
3. This stability evidence is tower-only. The server still needs the same `diagnose.py` command before any control is selected. There is no cross-host proof that a pinned ISA reproduces one shared CSV.
4. FP64 is an explicit precision-change diagnostic under `stability/out_fp64/`. All four FP64 controls on the tower are bitwise identical, including `mkldnn` off. Max abs versus the server FP32 reference is `3.8694764875613075e-07`, still above 1e-8. It does not replace the entry.
5. Independent written review is still pending. `INDEPENDENT_REVIEW.md` was not touched.
6. CNN03 is not packaged. Recorded metadata only: tune-selected seed 0, tune R 0.3321 versus 0.3465 and 0.3334, full R 0.3685, official points 43.845081/60, checkpoint SHA-256 `ce831b9b41653ae533ba05a64bb79ead340279fb93697caa5ab85c40bcc9203f`, base CNN SHA-256 `b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5`. No promoted CNN03 release.
7. Contract conflict is unresolved, not decided: kit text says organizers preprovision pinned requirements and then run offline; the live guide describes Python 3.13, an offered environment, and offline wheels. Score split 80/20 versus 60/40 and archive limits 128 MiB versus 100 MB are both still open. `metadata.json` in this kit says `max_points` 60.
8. The cp313 CPU wheel `torch-2.9.1+cpu` (184380800 bytes, SHA-256 `39b3dff6d8fba240ae0d1bede4ca11c2531ae3b47329206512d99e17907ff74b`) was not downloaded, installed, or numerically tested. Local proof used CPython 3.12.3 and `torch==2.9.1+rocm6.4` on CPU.
9. This continuation did not run a local tower Docker sandbox. The coordinator's server Docker result is quoted below. This host is not an organizer Python 3.13 proof.
10. Local ZIP SHA-256 differs from the server ZIP SHA-256. Both are 1375077 bytes. The server ZIP was not on this host, so member CRCs were not compared.

## What matched

Server, quoted from the coordinator and `SERVER_PARITY.md`: the same entry, CPU fp32, 2 threads, produced 1536 rows bitwise with reference SHA-256 `9ea67f812b80de18f03b1f6d8f0aeb6d5b91d8ce5c935dc83ba588e04b6be4e9`. Direct server timing was infer 0.732446 s, total 1.584855 s, max RSS 906176 KB. Strict server Docker (no network, read-only code and data, 2 CPUs, 16 GiB, tmpfs `/tmp` 128 MiB, tmpfs `/output` 32 MiB) produced the same SHA, 1536 rows, 1.54 s total, max RSS 904236 KB. Official checker on that server CSV was valid, season `5fffdc4d0c700bbf61c3f911`.

Tower `first.csv`, `second.csv`, the fresh ZIP extract, and the FP32 baseline/AVX2/oneDNN runs are bitwise with each other:

- SHA-256 `36c6055725d470c1fc26a6bb18c80666bac2d8a80eaf51dcf7d331acbd2d7ca3`
- 1536 ids, header `query_id,p_0,p_1,p_2,p_5,p_20`, all finite, max simplex abs error about 1.1e-16
- 197506 bytes, versus the reference CSV at 197471 bytes
- `kit/check_submission.py` on `first.csv`: `VALID FORMAT: 1536 queries | main | test`
- Cells not bitwise versus the reference: 7477

Tower timings for the entry command: first run infer 2.084153 s, total 3.890370 s, max RSS 948556 KB. Second run infer 2.408046 s, total 4.688242 s, max RSS 949768 KB. CPU is AMD Ryzen 7 3700X. Threads 2. dtype fp32.

## Paths and hashes

| Item | Path | SHA-256 |
| --- | --- | --- |
| Entry checkpoint | `experiments/cnn_runtime/entry/checkpoint.pt` | `43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80` |
| Entry config | `experiments/cnn_runtime/entry/config/config.json` | `c9022a4305be1c4124e6a116cd0cf3132fb5624a206f9b5571090ca3c4769d2f` |
| Reference test CPU CSV | `experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/test_cpu_a.csv` | `9ea67f812b80de18f03b1f6d8f0aeb6d5b91d8ce5c935dc83ba588e04b6be4e9` |
| Reference validation CPU CSV (not regenerated) | `experiments/cnn_runtime/reference/.../validation_cpu_a.csv` | `50efa7cb18f6c6a925b0259ac3177136789893dc16f82a2075cc3b4afe903c1d` |
| Tower test CSV | `experiments/cnn_runtime/first.csv` | `36c6055725d470c1fc26a6bb18c80666bac2d8a80eaf51dcf7d331acbd2d7ca3` |
| Tower repeat | `experiments/cnn_runtime/second.csv` | same as `first.csv` |
| Fresh extract rerun | `experiments/cnn_runtime/extract.csv` | same as `first.csv` |
| Local ZIP | `experiments/cnn_runtime/anchor_entry.zip` | `05f1f40fcb2b0c5a2b5aa1814d71f145a313287e65b02fe8edf847c1123940b3` |
| Server ZIP (not stored here) | `experiments/cnn_runtime/anchor_1929.zip` | `474978a3076b5583e644f5527dcac02e6f0fd29e4dc313c4becc4b145f87c2f0` |
| FP32 mkldnn-off CSV | `experiments/cnn_runtime/stability/out/mkldnn_off/submission.csv` | `e9723ff0fec7cb3a257235d1a13e71aca18be9d874cbfa4eb4bd19417454f745` |
| FP64 diagnostic CSV | `experiments/cnn_runtime/stability/out_fp64/baseline/submission.csv` | `02f08b8d44a2d57d798c5655f1519eae1c5c23aafd4e41bb1bd6be2b68b44710` |

Entry file hashes at diagnostic time:

- `predict.py` `14627994eeadd6ff44d59f33754636ec90d4be84bd5715e8b373923db11717b2`
- `model.py` `18b783f7a5c28c7dbbbec95c6174418594765a406d11f09b9e29599f35c13f40`
- `data.py` `cbf99171c4dbddcacd5d168be629a8f04b1a6134421b0578e6067f21709c1b1a`
- `reproduce.json` `cb91250a431dc92a45ceda7bf66df8006a4ecbce2b8e393f52f7284fd22246a1`
- `requirements.txt` `8f61a4742845a4851622bd789343d069803bb12607d7391051b0d73e52c7f38b`

Reference source hashes recorded in `entry/provenance/provenance.json`: `cnn.py` `b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5`, `predict_cpu.py` `598bc7bad1c539d081b45f7bbb0ad532f133c073bd5a355c4c7d6761859b7175`, shared `data.py` `5cde790750b3b69187f6b92c2da949ca6e11de8d750e6b8a4959131da89c5c7f`, `scoring.py` `ee6cea92f103d8a987a4d8c7eb59e8e0df94c23b7829d0852185f9cbfe365f4a`.

Anchor config: `no_signal` true, h 128, layers 4, W 64, drop 0.2, seed 0. Reported validation R on the clean CPU reference is 0.4031494025371707. The contaminated original g result near 0.378 is not in the package.

`reproduce.json` keys are only `image` and `command`. `inspect_archive` returned `python` and `["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]`. Local ZIP is 1375077 bytes, under both 100 MB and 128 MiB. Checkpoint inside the entry is 1483637 bytes.

## Commands

From the repo root, with the existing `.venv` and no installs:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/entry/predict.py --data kit/public --out experiments/cnn_runtime/first.csv
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/entry/predict.py --data kit/public --out experiments/cnn_runtime/second.csv
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/test_entry.py
.venv/bin/python kit/check_submission.py experiments/cnn_runtime/first.csv --data kit/public
.venv/bin/python kit/build_entry.py --source experiments/cnn_runtime/entry --out experiments/cnn_runtime/anchor_entry.zip
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/test_entry.py --extract-only
```

Server rerun of the stability diagnostic, same CLI, process-local controls only:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/stability/diagnose.py --run-all --data kit/public --out-dir experiments/cnn_runtime/stability/out
```

Optional FP64 diagnostic, labeled precision change, separate output directory:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/stability/diagnose_fp64.py --run-all --data kit/public --out-dir experiments/cnn_runtime/stability/out_fp64
```

Machine-readable results:

- `experiments/cnn_runtime/stability/out/comparison.json`
- `experiments/cnn_runtime/stability/out/<control>/manifest.json`
- `experiments/cnn_runtime/stability/out_fp64/comparison.json`

Controls, each in its own process after clearing ISA variables: `baseline`; `ATEN_CPU_CAPABILITY=avx2`; `ONEDNN_MAX_CPU_ISA=AVX2` plus `DNNL_MAX_CPU_ISA=AVX2` plus `MKL_CBWR=AVX2`; `torch.backends.mkldnn.enabled = False`. Every child also sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 2 and blanks `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and `ROCR_VISIBLE_DEVICES`. The parent shell is not given those ISA assignments. The metric is max absolute difference of the row-normalized probabilities. Relative error is not the tolerance. The threshold stays 1e-8.

## Tests

`experiments/cnn_runtime/test_entry.py` passed: window causality (tick `t` does not enter the window; tick `t-1` does), MISSING maps to embedding index 5 with mask 0 while payout 0 stays index 0 with mask 1, sorted slot mapping, signal column not kept, no-signal meta stays zero. Negative runs fail closed with no CSV: missing `test/` history and queries, tampered checkpoint (`anchor hash mismatch`), checkpoint config `no_signal` false. Entry checkpoint hash was unchanged after the tests.

## Stability reading

The tower FP32 default already matches itself across a second process, a ZIP extract, and the AVX2 caps. It does not match the server FP32 reference within 1e-8. Turning oneDNN off changes tower FP32 and does not close the gap. FP64 removes that oneDNN difference on this tower, and it is still a different forecast from both the tower FP32 CSV and the server reference. Neither result is a reason to replace `entry/predict.py` or the packaged checkpoint.

## Unverified

- Server output of `stability/diagnose.py` and `stability/diagnose_fp64.py`
- Whether pinning AVX2 on the 9950X moves the server CSV
- Member-level equality of `anchor_entry.zip` and `anchor_1929.zip`
- Local tower Docker inference in this continuation
- Python 3.13 and the CPU wheel's numeric match to `2.9.1+rocm6.4`
- Regeneration of the validation CSV (the entry reads only `test/history.csv` and `test/queries.csv`)
- CNN03 review and any later promotion
