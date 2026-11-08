# Package review, ensemble_equal_s0_s2

Written 2026-11-08T03:08:43 CST. Same independent reviewer. `FROZEN_REVIEW.md` is not rewritten. The candidate tree, production pins, and both existing release pairs were not modified. This review did not call `fp64_candidate_harness.py` as `__main__` in the frozen SERVER candidate root. That main rewrites `candidate_t146_ensemble_s0_s2_20261108/tune_only.json`. The file was `e96358d9edd860942fbf55b7beb4a08e4483bfea5efefd22a841052073984932` before these checks and the same digest after them. The copied original under `evidence/crosshost/server_tune_only_original.json` is the same digest.

Interpreter for the host checks: `experiments/cnn_runtime/py313_verify_1936/env/bin/python`, CPython 3.13.13, torch 2.11.0+cpu. `PYTHONDONTWRITEBYTECODE=1`, two OpenMP/MKL/OpenBLAS threads, `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` empty. Pytest is not installed and was not installed.

## Verdict

| Component | Verdict |
| --- | --- |
| Exact ZIP members, both original parents, canonical `model.py` / `data.py`, new `predict.py` | PASS |
| CPU FP64 forward, softmax, guards, refusals, 1536/768 coverage, tune 384 | PASS |
| SERVER cold repeat and this review's fresh-ZIP repeat | PASS, byte-identical to `1f4f1a2c…` |
| Author Docker records | PASS for what the inspect files show; the first three containers are a 16 GiB cgroup |
| This review's three fresh containers at 2 GiB cgroup | PASS |
| Tower cross-host, kit-normalized 1536 test and 768 validation | PASS |
| Cross-runtime byte identity | Not claimed. Tower CSV bytes differ. Numeric gap is under `1e-8` |
| Tower 5e-5 low-rate jobs | BLOCK as training. Not in this ZIP |
| Shared-control blends | Not cleared and not in this ZIP |
| Closed low-rate replay, host binding, forward-count, and TF32 guards | CONCERN on that closed study only |
| Dependency contract | CONCERN, unresolved |
| Organizer acceptance, automatic production replacement, upload | Not claimed and not done |

**Overall: PASS for this exact ZIP as the technical offline package of the already cleared `ensemble_equal_s0_s2` candidate.** Nothing measured in the ZIP, the SERVER qualification, the strict 2 GiB Docker repeat, or the staged Tower CSVs blocks that technical release. This is not organizer acceptance, not a production-pin change, and not an upload.

## Package identity

| Artifact | SHA-256 | Bytes |
| --- | --- | --- |
| `ensemble_equal_s0_s2_fp64_candidate.zip` | `d04d96fd0e28c946f6704170326ee10b32d570e6ab63f5bc2855922cbfd5c991` | 2749917 |
| `refs/candidate_test_a.csv` and this review's fresh test CSV | `1f4f1a2c8176a8d2f70c099b5265f8432103fdaaf2fc8744bc60bfe376ef081c` | 198055, 1536 ids |
| `SOURCE_HASHES.txt` | `106e1deda225ca8902e1c69f75cb7f093910253bcfd927d2ffd83a23025806eb` | 48 listed rows, 0 mismatches |
| `predict.py` | `364bb0d69cfca6365c001a6ae7322bd8944056d6783855e22863fb59b6d45594` | |
| `model.py` | `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a` | matches canonical `fresh_fp64_cnn03_1956/model.py` |
| `data.py` | `af23684b5348c53dafc0e3870706da7ed3dc64c852537371ac7c0ee97d916133` | matches canonical `data.py` |
| seed 0 checkpoint | `d5d714649a93e1e552ccf5419cb01adee311c082af1063f80d8b42f6d8dfe3cb` | byte-identical to `runs/curve_C_lr003_s0/checkpoint.pt` |
| seed 0 config | `7ddfe3a4dd1439945c6145d1d5cc2a868653ba30f25523890bf95f5e23aa10ac` | original config, seed 0, lr 0.003 |
| seed 2 checkpoint | `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71` | byte-identical to `runs/curve_C_lr003_s2/checkpoint.pt` |
| seed 2 config | `a715bfe05eb00494065c7dfcc3ad6d71a3b2bc4ee6788006cf07ce86f3e43695` | original config, seed 2, lr 0.003 |
| `provenance/weights.json` | `30dad61dbcb82a33fb4a72c5296ae9ad700c2c0ff92fff629403703a40139e9f` | order `s0`, `s2`, weights `0.5`, `0.5` |

The fresh extract is `package_ensemble/candidate_t146_ensemble_s0_s2_20261108/fresh_zip`. The ZIP has 12 members, no absolute paths, no `..`, and no symlinks. All 12 member hashes match `SOURCE_HASHES.txt`. There is no third component and no low-rate or shared-control weight. Saved tensors stay the original float32 checkpoints. `predict.py` checks hash, config keys, 43 shapes, kernel `(128, 128, 3)`, signal keys, saved dtype, and finiteness before it reads a split, then calls `model.double()`.

Untouched, rehashed this review:

| File | SHA-256 |
| --- | --- |
| current-best ZIP | `dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29` |
| current-best CSV | `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6` |
| fallback ZIP | `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599` |
| fallback CSV | `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d` |
| production `PINNED.json` | `cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb` |
| production `fp64_harness.py` | `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2` |
| release harness | `a077e8771abdab58e3d107eeb1da3ecc3ee80ccc2ee08ce72e49b7ab36d89f2c` |

## Commands and outputs

Host proof, log `review_package_out.txt`, exit 0, tail `REVIEW_PACKAGE_HOST_OK`:

```
package_ensemble/review_package.py
```

The script extracted the ZIP under this review directory, checked pins, then loaded the unchanged harness module and called its test functions with `WORK` and `OUT` pointed at this review directory. It did not execute harness `main`. Tune scoring was the documented CLI:

```
score_tune_only.py --entry <fresh zip> --val-csv <review val csv> --out <this review sandbox>/tune_only.json
```

The output path contains the folder name the script requires, and it is not the frozen candidate root.

Results from that log:

- `unit_ok`, `query_order_ok`, `negative_ok`
- `dtype_ok s0 params=43 logit=float64` and the same for `s2`
- Refusals exercised inside the harness: wrong hash, wrong config hash, signal tensor, extra key, wrong conv shape, non-float32 saved tensor, non-finite weight, changed weights manifest, extra component directory, missing split
- Cold test repeat bitwise, SHA `1f4f1a2c…`, 198055 bytes, 1536 ids
- Format check on the test CSV only: `VALID FORMAT: 1536 queries`. The validation split of `check_submission.py` was not called. Confirm, full, and test R were not computed
- Validation coverage 768, finite simplex, targets not opened
- Package versus verified `ensemble_equal_s0_s2.npz`: kit max abs `5.55111512313e-16`, raw max abs the same, threshold `1e-8`
- `TUNE_ONLY R=0.28468175920919014 n=384 forward=768`
- Host RSS in that process: `336428` KiB. Combined infer on the first test pass `2.416175` s, total `3.102819` s. Second test pass combined `2.326740` s. All under the `2097152` KiB cap

`diff -u` of `fp64_release_harness2025.py` against `fp64_candidate_harness.py` is saved as `HARNESS_DIFF_RERUN.txt`. The two header lines differ by path and timestamp. The other 523 lines match `HARNESS_DIFF.txt` exactly. The published diff is the real base-to-candidate change, not a stale single-model package diff. Substance added for this ensemble includes both component directories, per-parent hash/config/shape/dtype/finite refusals, extra-component and weight-manifest refusals, the 2 GiB RSS cap, and the comparison to the verified ensemble npz. Production `PINNED.json` is not read or written. Tolerance stays `1e-8`.

## Docker

Author image, confirmed here with `docker image inspect` and no pull: `ubuntu:24.04` `sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55`.

Author inspect files, read and not rerun:

| Container | Memory | MemorySwap | NanoCpus | Network | OOMKilled | EXIT |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| `cand-t146-ens-a` | 17179869184 | 34359738368 | 2000000000 | none | false | 0 |
| `cand-t146-ens-b` | 17179869184 | 34359738368 | 2000000000 | none | false | 0 |
| `cand-t146-ens-val` | 17179869184 | 34359738368 | 2000000000 | none | false | 0 |
| `cand-t146-ens-cap2g` | 2147483648 | 2147483648 | 2000000000 | none | false | 0 |

Those four also record read-only root, `CapDrop ALL`, `no-new-privileges`, and user `1000:1000`. The first three are not a strict 2 GiB cgroup. `cand-t146-ens-cap2g` is. Its log says `NPROC=2`, max RSS `334036` KiB, and the test CSV hash in `QUALITY.md` is the SERVER test hash. This review did not treat the 16 GiB runs as the 2 GiB proof.

This review repeated three fresh containers from the same image and the same ZIP mount, in `package_ensemble/docker_review.sh`. Flags: `--pull=never`, `--network=none`, `--read-only`, `--cpus=2`, `--memory=2147483648`, `--memory-swap=2147483648`, `--cap-drop=ALL`, `--security-opt no-new-privileges`, `--user 1000:1000`, GPU variables empty. Each container extracted the ZIP in tmpfs and ran packaged `predict.py`. Inspect summary is `docker_review_out.txt`.

| Container | EXIT | OOM | max RSS KiB | CSV SHA-256 |
| --- | ---: | --- | ---: | --- |
| `rev-t146-ens-2g-a` test | 0 | false | 333944 | `1f4f1a2c…` |
| `rev-t146-ens-2g-b` test | 0 | false | 333960 | `1f4f1a2c…` |
| `rev-t146-ens-2g-val` validation | 0 | false | 287212 | `e330a3e4f6ff8155cbce01025346f04e95146fbb0ff6608c74848db5809a5f2d` |

Inside the container: CPython 3.13.13, torch 2.11.0+cpu, `NPROC=2`. Test A combined infer `2.390101` s, total `3.100887` s. The two test CSVs are byte-identical. The validation CSV matches the frozen reference validation hash. No validation-round score was computed.

## Cross-host

`evidence/crosshost/` was present at the single check after the local proof. `TRANSFER_SHA256.json` matches all 12 staged files, 0 mismatches. This review did not launch or copy the Tower job.

Tower log `crosshost.log` (`f860f853c0b9c98e01fc34f4353ccb757652fad302099a7aabbcb808fa3ea277`): Python 3.12.3, executable `.venv_gfx1031/bin/python`, fresh entry `/tmp/t146_ens_crosshost_e850f51eaad448039ef1d57d4d9a09b7`, `HARNESS_OK`. `runtime.json`: torch `2.14.0+rocm10.1.0`, HIP `7.16.26385`, `cuda_available` false, threads 2, `max_rss_kib` `1038168`. `status.json`: exit 0, finished `2026-11-08T03:05:51-06:00`, elapsed `268.2819355870015` s, affinity CPUs 0 and 1, GPU hidden, budget `2097152` KiB, `budget_stop` false. Peaks: max process RSS `996048` KiB, RSS sum `1882640` KiB, PSS sum `1265096` KiB. The harness print inside the log is `max_rss_kb=998168`. Those figures are different measurements of the same run. All of them are under 2 GiB. The wall time is the whole Tower harness, including refusals and both test passes, not a claim of a 3-second infer.

Tower cold repeat is bitwise with itself: both test CSVs SHA `a12fbe7a3e3a6c05e6fac9a2a6d7e877403f83862e2a08f630ba3777ba43d6e7`, 197976 bytes, 1536 ids. That is not the SERVER SHA `1f4f1a2c…` and not the SERVER size 198055. Validation SHA `7f0f6c8662359ee49cd1ca6957e10c3ecc06bbde0b08cc9a0996eb28050fffab`, 103344 bytes, is not the SERVER validation SHA `e330a3e4…`.

This review recomputed kit row normalization with `slot_forecasting.io.read_submission` (`schema.probabilities`, `math.fsum`, then divide by the sum). Output: `crosshost_compare.json`.

| Comparison | Ids | Kit max abs | Bytes equal |
| --- | ---: | ---: | --- |
| Tower test A vs Tower test B | 1536 | 0.0 | yes |
| Tower test vs this review's fresh SERVER test | 1536 | 8.881784197001252e-16 | no |
| Tower test vs frozen SERVER reference | 1536 | 8.881784197001252e-16 | no |
| Tower validation vs this review's fresh SERVER validation | 768 | 5.551115123125783e-16 | no |
| Tower validation vs frozen SERVER reference | 768 | 5.551115123125783e-16 | no |

Both cross-runtime gaps are under `1e-8`. Tower tune JSON says `tune_R` `0.28468175920919014`, `n_tune_queries` 384, `n_forward_queries` 768, `confirm_full_test_scored` false, map max abs versus the verified npz `0.0`, and CSV max abs `1.1102230246251565e-16`. The log's closing hashes for `model.py`, `data.py`, `predict.py`, and both checkpoints match this ZIP. SERVER 3.13 / torch 2.11 remains the qualification environment. Tower 3.12 / torch 2.14 is a different runtime. No cross-runtime byte identity is claimed.

`pin_integrity.json` records 34 candidate-local receipt entries and says the only generated metadata change on Tower was `tune_only.json` (`4be330ac2be64f4014bec95637c485fc909f30da19d3ec45918bdb007efcb1d8`). This review did not re-execute that 34-file check on Tower. It did confirm the SERVER `tune_only.json` still matches the preserved original.

## Standing qualifications

These stay on the record and do not change the ZIP hashes:

- The pair is tune-informed. Many predeclared tune comparisons were made. There is no significance claim and no hidden-test claim. Earlier packaging printed public validation diagnostics, so this is not a blind confirmation.
- `COMPARISON_STATUS_ADDENDUM.md` is `a4d7f36435415fcbd04264edd2e31436a11fcb97a098fb39c5055568b60204fd`. It supersedes the historical path-blocker wording. `TOWER_FINAL_RSS_GUARD.md` is `a42361ef2f618ab6a6b89a43246ebf58b4dd7935846c1f55cb10ae85dc193acc`. After staging, the native FP32 batch-128 forward/backward with no optimizer step reached RSS `2174996` KiB and exited 1 against cap `2097152`. No Tower 5e-5 epoch or checkpoint exists. Only 2 of 4 low-rate jobs completed. Six Tower rows have no score and are not ranked as worse. This package uses none of those weights.
- Blends that include the shared control stay not cleared. Historical FP32 reproduces `0.46211193532631023`. Canonical FP64 is `0.46211190598948326`. No probability correction was fit. This ZIP does not contain that control.
- Replay exclusivity, host-to-device binding, forward-count refusal, and TF32 abort-on-failure remain concerns of the closed low-rate study. They are not repaired by this package.
- The kit-versus-guide dependency contract remains unresolved. This review does not invent precedence.
- Technical PASS is not organizer acceptance and not an automatic replacement of the production pin or of either existing release. Nothing was uploaded.
