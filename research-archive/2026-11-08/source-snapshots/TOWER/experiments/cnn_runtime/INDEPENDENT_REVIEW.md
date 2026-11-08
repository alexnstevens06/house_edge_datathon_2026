# Independent review — CNN anchor runtime package

**Decision: BLOCKED RELEASE.** This package is not a clean release.

The packaged pin `torch==2.9.1` is a critical pin defect against the canonical server stack stated by the coordinator (Torch 2.11.0+cu128, Python 3.12.13). On this tower, the entry's own CPU fp32 CSV misses the contract tolerance against the canonical reference CSV. A same-machine repeat is bitwise and does not repair that gap. The anchor checkpoint was not edited. No FP64 release and no CNN03 promotion.

Labels used below:

- **VERIFIED** — measured in this review on the tower, from the current `experiments/cnn_runtime/entry` and the staged reference.
- **PROVIDED** — coordinator statement supplied with the resume instruction. Those logs and the named server ZIP were not re-opened or re-run for this write-up.
- **NOT RUN** — no test file, no execution, or a figure this write-up does not have.

Reviewer compute for the VERIFIED commands was one pinned CPU (`taskset -c 0`) with `PYTHONDONTWRITEBYTECODE=1`. No training, GPU, install, git, network fetch, credential use, upload, or edit of entry, reference, kit, checkpoint, or ZIP.

## Why release stays blocked

1. **Critical pin defect (requirements).** `experiments/cnn_runtime/entry/requirements.txt` lines 10–11 pin `numpy==2.5.4` and `torch==2.9.1` (VERIFIED). The comments on lines 4–9 say the local proof imported `torch==2.9.1+rocm6.4` and that the CPU wheel `torch==2.9.1` was not installed. **PROVIDED** canonical server stack for the bitwise original CSV: Torch **2.11.0+cu128**, Python **3.12.13**, AVX512, oneDNN 3.10.2. **PROVIDED** later server env that also matched that CSV: CPython **3.13.13**, Torch **2.11.0+cpu**, NumPy 2.5.4. Neither stated stack is `torch==2.9.1`. Shipping this requirements file asks the organizer preprovision path for a different Torch minor than the stack that produced the canonical CSV.
2. **Canonical numeric tolerance failed on this tower (VERIFIED).** Entry CPU fp32 versus reference `test_cpu_a.csv`: normalized max abs **3.742622056712719e-7** at `main-test-e00135-m002-next`. **7477** probability cells differ by any nonzero amount. The contract limit is **1e-8** after normalization (`kit/README.md` line 351, `kit/RULES.md` line 54). `within_1e-8` is false. Tower CSV SHA-256 `36c6055725d470c1fc26a6bb18c80666bac2d8a80eaf51dcf7d331acbd2d7ca3` (197506 bytes). Canonical reference SHA-256 `9ea67f812b80de18f03b1f6d8f0aeb6d5b91d8ce5c935dc83ba588e04b6be4e9` (197471 bytes). **PROVIDED** cell count above 1e-8 is 3441. This review did not compute that threshold count; it computed the max and the nonzero-cell count above.
3. **Same-backend repeat is not evidence of canonical match (VERIFIED).** Two cold CLI runs on this tower were byte-identical to each other and still the non-canonical SHA above. In-process entry `predict` versus the reference `cnn.py` implementation, both on this tower's Torch, had max abs probability difference **0.0**. The implementation matches itself here. It does not match the saved server CSV within 1e-8.
4. **Organizer mechanism is still ambiguous.** Kit README line 333 and RULES line 54 say Python 3.13 plus a root `requirements.txt` that organizers preprovision. Staged `LIVE_PARTICIPANT_GUIDE.txt` line 290 says Python 3.13 stdlib, an offered environment, or compatible Linux wheels and an offline setup script. **PROVIDED** server result (3.13.13 + Torch 2.11.0+cpu bitwise to the canonical CSV, 1.36 s, max RSS 610964 KB, checker PASS) shows one CPU 3.13 stack can run this entry. It does not choose between preprovision and offline wheels, and it does not change the bytes in `requirements.txt`. No precedence is invented here.
5. **FP64 / CNN03 are out of this release.** No second checkpoint and no CNN03 manifest are in `entry/`. Final FP64 work was described as upstream and separate. This anchor stays the fp32 checkpoint below. There is no FP64 release.

## Anchor identity (VERIFIED)

| Artifact | SHA-256 | Bytes |
|---|---|---|
| `entry/checkpoint.pt` | `43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80` | 1483637 |
| `reference/.../g_nosignal/checkpoint.pt` | same | 1483637 |
| `entry/config/config.json` and reference `config.json` | `c9022a4305be1c4124e6a116cd0cf3132fb5624a206f9b5571090ca3c4769d2f` | 180 |
| `reference/.../test_cpu_{a,b,c,d}.csv` | `9ea67f812b80de18f03b1f6d8f0aeb6d5b91d8ce5c935dc83ba588e04b6be4e9` each | 197471 |

Checkpoint load: top keys `config` and `state`; `no_signal` is boolean true; 43 tensors; only dtype `torch.float32`; no parameter name containing `sig`. Architecture in the saved config: h 128, layers 4, W 64, drop 0.2, seed 0. `predict.py` lines 18 and 81–83 pin that checkpoint SHA and the config SHA and exit 2 on a mismatch before test CSV write. The evaluated file is the anchor file, not a converted copy.

`entry/predict.py` line 92 sets `torch.set_num_threads(2)`. Line 144 calls `predict(..., "cpu")`. `model.py` lines 100–102 enable CUDA bfloat16 autocast only when `dev == "cuda"`. A forward hook on this CPU path recorded autocast off, parameter dtype float32, logit dtype float32. Export probabilities are float64 (`model.py` line 103), then `predict.py` lines 57–72 renorm with `sum` and `repr`, same pattern as reference `scoring.py` `write_submission_csv`.

## Source hashes examined (VERIFIED)

Entry:

| File | SHA-256 | Bytes |
|---|---|---|
| `entry/predict.py` | `14627994eeadd6ff44d59f33754636ec90d4be84bd5715e8b373923db11717b2` | 6263 |
| `entry/model.py` | `18b783f7a5c28c7dbbbec95c6174418594765a406d11f09b9e29599f35c13f40` | 3881 |
| `entry/data.py` | `cbf99171c4dbddcacd5d168be629a8f04b1a6134421b0578e6067f21709c1b1a` | 4082 |
| `entry/README.md` | `fb475ff537cd6a9d3e0ec02578f076226ecacf92ac92a05b4f6ced2d8ac730fe` | 2647 |
| `entry/reproduce.json` | `cb91250a431dc92a45ceda7bf66df8006a4ecbce2b8e393f52f7284fd22246a1` | 117 |
| `entry/requirements.txt` | `8f61a4742845a4851622bd789343d069803bb12607d7391051b0d73e52c7f38b` | 524 |
| `entry/provenance/provenance.json` | `aaf8442b0d5cf93cf1d566296632f2995af6bfdc0e25a3142ce3bd2ed64e9916` | 2049 |

Reference sources (these four match the hashes named inside `provenance.json`):

| File | SHA-256 | Bytes |
|---|---|---|
| `reference/experiments/cnn/source/cnn.py` | `b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5` | 9371 |
| `reference/experiments/cnn/source/predict_cpu.py` | `598bc7bad1c539d081b45f7bbb0ad532f133c073bd5a355c4c7d6761859b7175` | 3488 |
| `reference/src/house_edge/data.py` | `5cde790750b3b69187f6b92c2da949ca6e11de8d750e6b8a4959131da89c5c7f` | 5024 |
| `reference/src/house_edge/scoring.py` | `ee6cea92f103d8a987a4d8c7eb59e8e0df94c23b7829d0852185f9cbfe365f4a` | 3302 |

`src/house_edge/data.py` (tower tree) hashed to the same `5cde7907…` value. Pre-existing builder CSV `experiments/cnn_runtime/first.csv` is SHA `36c6055725d470c1fc26a6bb18c80666bac2d8a80eaf51dcf7d331acbd2d7ca3`, 197506 bytes: the same bytes as this review's tower run, not the canonical reference. It was not used as proof of a match.

Review ZIP built here (not the coordinator ZIP): `experiments/cnn_runtime/reviewlogs/entry_review.zip` SHA-256 `06510dc99d06359c4e062d7b245d44dbea54287856c70d2e81ce5acda065572a`, 1375077 bytes.

## Contract surface (VERIFIED unless marked)

`entry/reproduce.json` is exactly:

```json
{"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]}
```

Keys are only `image` and `command`. That matches `kit/README.md` lines 317–322 and `submission_contract.py` lines 57–64. Kit README lines 324–329 put the working directory at read-only `/submission`, data at read-only `/data`, scratch `/tmp` 128 MiB, and `/output/submission.csv` on a 32 MiB mount. The entry resolves model files from `Path(__file__).parent` (`predict.py` line 17), so the official argument list does not depend on the process cwd for the checkpoint. Local VERIFIED runs used cwd = `entry/` and absolute `--data` / `--out`.

Root files present: `README.md`, `reproduce.json`, `requirements.txt`, `checkpoint.pt`, plus `model.py`, `data.py`, `predict.py`, `config/config.json`, `provenance/provenance.json`. README line 8 names the anchor checkpoint SHA. README line 13 states CPython 3.12.3 was the local proof and organizer Python 3.13 was not executed in that write-up. README line 5 says team ID was not supplied. That is a submission-metadata gap, not the release blocker.

Official size code: ZIP ≤ 128 MiB and ≤ 5000 members (`submission_contract.py` lines 11–14, 28–34); README and `reproduce.json` at archive root (`submission_contract.py` lines 51–55). Kit prose also says 100 MB ZIP and 10 MB CSV (`kit/README.md` lines 291 and 349–351, `kit/RULES.md` line 51). Guide text lines 296–297 say 128 MiB ZIP and 16 MiB CSV. This package is far under every one of those ceilings. The prose conflict is unresolved and does not block on size.

## Equivalence of copy vs reference (VERIFIED)

Compared `entry/data.py` `load_test_split(kit/public)` with reference `data.py` `load_split(kit/public/test)` on the real test split.

- 192 episodes, 1536 queries, episode order equal, query-id order equal.
- Observation arrays, machine-id slot order, machine-type vectors, and cutoffs: 0 mismatches.
- Type map is `balanced=0`, `jackpot=1`, `steady=2`. Payout map is `0,1,2,5,20` → indices `0..4`. `MISSING` is `-1`.
- Example episode `main-test-e00000`, cutoff 221, slots `m000..m007`, types `[0,2,2,0,0,1,1,2]`. A missing cell at tick 0 slot 0 is `-1`. A payout-0 cell at tick 3 slot 0 is `0`. Missing is not stored as payout 0.
- Loaded observations: 61136 missing cells, 170743 payout-index-0 cells, 343800 cells total, 48 episodes with an interior missing tick, 0 episodes whose loaded `obs` extends to or past cutoff.
- Entry query dicts omit `signal` (`data.py` lines 70–76 keep cutoff and drop the signal field). Reference rows still contain `signal`. With `USE_SIGNAL` false, `model.py` lines 17–19 build an all-zero signal vector. Feature tensor sum of that channel was 0.
- Cutoff guard is strict for every machine: `data.py` lines 96–100 raise when `tick >= cutoff`.
- `window` (`model.py` lines 22–29) matched reference `cnn.window` on synthetic lengths t = 0, 1, 63, 64, 128, 256 at W = 64, and matched a prefix cut to ticks `< t`.
- On 40 real episodes, appending three all-zero rows at and after cutoff left `window(obs, cutoff, 64)` unchanged. On those 40, changing the last included tick changed the window.
- `build_query` tensors matched the reference implementation. Shapes `[192,64,8]`, `[192]`, `[192,8]`, `[192,8]`.
- Eight-episode logit max abs, entry CNN vs reference CNN, same state dict: **0.0**. Full in-process probability max abs: **0.0**. Both networks had `use_signal` false and `sig is None`.

`entry/` has no `test_*.py`. `experiments/cnn_runtime` package tests outside `reference/` and this review log: **none found. NOT RUN** as a package suite. The checks in the next section are this review's commands, not a builder-claimed suite. `STATUS.md` still listed tests and ZIP as unchecked; those checkboxes were not treated as evidence.

## Commands and results

Completed invocation (exit 0), after an earlier invocation died inside this reviewer's compare helper and did not reach ZIP or sandbox. The earlier `run1` predict itself had exited 0 (`infer_s=2.978602`, `total_s=5.632176`, `max_rss_kb=951228`). Figures below are from the completed invocation unless noted.

```text
taskset -c 0 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
  .venv/bin/python -u experiments/cnn_runtime/reviewlogs/review_check.py
```

Cwd: repository root. Host interpreter: Python 3.12.3, Torch 2.9.1+rocm6.4, NumPy 2.5.4. `torch.cuda.is_available()` was false. The script set Torch threads to 2 after the process was pinned to CPU 0. `os.cpu_count()` inside that process reported 16. Wall timing is one pinned CPU, not an organizer 2-CPU measurement.

CLI predict, cwd `experiments/cnn_runtime/entry`:

```text
strace -f -e trace=openat,open,connect,sendto,recvfrom,bind \
  -o experiments/cnn_runtime/reviewlogs/run1.strace \
  /usr/bin/time -v -o experiments/cnn_runtime/reviewlogs/run1.time \
  .venv/bin/python predict.py \
  --data /home/alexn/Documents/code/datathon_2026/house-edge/kit/public \
  --out .../reviewlogs/run1.csv
```

Exit 0. Stderr: `queries=1536 infer_s=3.242955 total_s=6.005719 max_rss_kb=950200 device=cpu dtype=fp32 threads=2`.

Repeat:

```text
/usr/bin/time -v -o .../reviewlogs/run2.time \
  .venv/bin/python predict.py --data <kit/public> --out .../reviewlogs/run2.csv
```

Exit 0. `infer_s=3.365663 total_s=5.227269 max_rss_kb=951304`. SHA identical to run1. Normalized max abs versus run1: **0**.

| Check | Result |
|---|---|
| Header `query_id,p_0,p_1,p_2,p_5,p_20` | PASS, reference and both runs |
| 1536 unique ids, set equals `kit/public/test/queries.csv` | PASS (1536 unique in the public query file too) |
| Finite, each value in `[0,1]`, `math.fsum` within 1e-6 of 1 | PASS. Max abs sum error about 1.11e-16 |
| Normalized max abs vs `test_cpu_a.csv` ≤ 1e-8 | **FAIL**. Max 3.742622056712719e-7. 7477 nonzero cells |
| run2 vs run1 bitwise | PASS |
| Official checker | PASS format only |

Checker, cwd `kit/`, `PYTHONPATH=kit`:

```text
.venv/bin/python check_submission.py .../reviewlogs/run1.csv --data <kit/public>
```

Exit 0:

```text
VALID FORMAT: 1536 queries | main | test
Season: 5fffdc4d0c700bbf61c3f911
SHA-256: 36c6055725d470c1fc26a6bb18c80666bac2d8a80eaf51dcf7d331acbd2d7ca3
Test answers are private; this check does not calculate your event score.
Local check only. Nothing was uploaded. Acceptance and code reproduction are separate.
```

Strace on run1 (1287 open lines, then the raw trace was deleted after the summary): opened `config/config.json`, `checkpoint.pt`, `kit/public/test/queries.csv`, `kit/public/test/history.csv`. No open of `train/`, `validation/`, `labels.csv`, `conditions.csv`, or `test_cpu_*.csv`. No `connect` / `sendto` lines. Signal-flip directory contained only `test/history.csv` and `test/queries.csv` (all 1536 signal values rewritten). `predict.main` on that directory exited 0 and wrote a CSV byte-identical to run1. That is a real signal-independence result, and it shows labels, train, and validation files are not required for this entrypoint.

History mutation through the entrypoint: first history row of `main-test-e00000-m000` tick 3, payout `0` → `1`. Exit 0, SHA still `36c60557…`, max abs vs run1 **0**. That episode's cutoff is 221 and W is 64, so tick 3 is outside the window. **This end-to-end mutate does not prove history sensitivity.** It is recorded as run, and as inconclusive for causality. The direct `window` checks above did show that the last in-window tick is read and that rows at or after cutoff are ignored.

Cutoff fixture (8 machines, one history tick equal to cutoff 2): `predict.main` printed `REFUSED: missing or unreadable test history/queries: history tick 2 >= cutoff 2` and wrote no CSV.

Tamper subprocesses, each a copy under `reviewlogs/fixtures/`, cwd = the copy, `predict.py --data kit/public --out <new csv>`:

| Case | Result |
|---|---|
| Weight tensor `+ 1` then `torch.save` | **PASS (refuse).** Exit 2. `REFUSED: anchor hash mismatch b6e8c3eb9c57486518c35731c4ffedb2b17542e064eee2ec12cbaf0775b5f7a5`. No CSV |
| Checkpoint config `no_signal` set false | **PASS (refuse).** Exit 2. `REFUSED: unclean config: no_signal is not True (signal mode refused)`. No CSV |
| One-byte config.json change | Subprocess was run. `reviewlogs/` has `tamper_config.csv.log` and no `tamper_config.csv`. The exact stderr line was in the truncated part of the summary and is not re-quoted here |

`predict_cpu.py` lines 18–19 also refuse a `*CONTAMINATED*` filename in a run directory. `entry/predict.py` has no such marker check. The hash pin and `no_signal` check fired on the copies above. Marker absence is a residual difference, not the release blocker.

ZIP, cwd `kit/`:

```text
.venv/bin/python kit/build_entry.py \
  --source .../experiments/cnn_runtime/entry \
  --out .../experiments/cnn_runtime/reviewlogs/entry_review.zip
```

Exit 0. `inspect_archive` accepted it. 9 members, unpacked 1503380 bytes, compressed 1375077 (under 100000000 and under 128 MiB). Names:

```text
README.md
checkpoint.pt
config/config.json
data.py
model.py
predict.py
provenance/provenance.json
reproduce.json
requirements.txt
```

`README.md` and `reproduce.json` are at archive root. No `.git`, venv, `__pycache__`, or `.csv`. `reproduce.json` inside the ZIP parsed to the same image and command. Fresh extract command exited 0 and wrote `reviewlogs/fresh.csv` (197506 bytes). Byte equality with run1 is stored in `reviewlogs/results.json` (`fresh_sha_equals_run1`) and was not in the summary text retained for this write-up.

### Tower sandbox (VERIFIED prediction, not organizer proof)

```text
docker run --rm --pull=never --network=none --read-only \
  --cpus=2 --memory=16g \
  --tmpfs /tmp:rw,size=128m --tmpfs /output:rw,size=32m \
  -v entry:/submission:ro -v kit/public:/data:ro \
  -v .venv:/opt/venv:ro -v /usr:/usr:ro -v /lib:/lib:ro \
  -v /lib64:/lib64:ro -v /opt/rocm:/opt/rocm:ro \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e CUDA_VISIBLE_DEVICES= -e HIP_VISIBLE_DEVICES= \
  -e ROCR_VISIBLE_DEVICES= -e HOME=/tmp \
  -w /submission --entrypoint /bin/bash ubuntu:24.04 -lc '<predict + probes>'
```

Exit 0. This is a real model run, not `docker … /bin/true`.

```text
SANDBOX_NPROC=1
SANDBOX_PYTHON=3.12.3
TORCH 2.9.1+rocm6.4
NUMPY 2.5.4
CUDA False
PREDICT_RC=0
SUBMISSION_WRITABLE=no
DATA_WRITABLE=no
NETWORK_CONNECT=no OSError
36c6055725d470c1fc26a6bb18c80666bac2d8a80eaf51dcf7d331acbd2d7ca3  /output/submission.csv
197506 /output/submission.csv
query_id,p_0,p_1,p_2,p_5,p_20
tmpfs 128M on /tmp
tmpfs 32M on /output
```

Stderr: `queries=1536 infer_s=2.427306 total_s=4.845350 max_rss_kb=949204 device=cpu dtype=fp32 threads=2`.

The container printed `nproc=1` even though the docker flag was `--cpus=2`. The bind mounts are this host's Python 3.12 and the ROCm Torch wheel. The CSV matches the tower run, which **fails** 1e-8 against the canonical reference. Memory headroom versus 16 GiB is large (about 930–950 MiB RSS on the CLI runs). Output CSV is about 198 KB versus 32 MiB. ZIP is about 1.38 MB.

## Coordinator facts (PROVIDED, not re-run here)

These are copied from the resume instruction so the decision record stays in one place. This review did not hash the server ZIP, did not read `URGENT_VERSION_ENV.md`, `REVIEW_EVIDENCE_READY.md`, or `requirements_py313_cpu211_frozen.txt`, and did not execute on the server.

- Canonical server versus this tower, as stated: server Torch 2.11.0+cu128, Python 3.12.13, AVX512, oneDNN 3.10.2; tower Torch 2.9.1+rocm6.4, Python 3.12.3, AVX2, oneDNN 3.7.1. This review verified only the tower Python and `torch.__version__` (`2.9.1+rocm6.4`). AVX and oneDNN strings were not read here.
- `provisional anchor_1929.zip` SHA-256 `474978a3076b5583e644f5527dcac02e6f0fd29e4dc313c4becc4b145f87c2f0`, size 1375077, official packager, 9 root files, inspect CRC pass. Size matches the ZIP this review built. The SHA does **not** match this review's ZIP (`06510dc9…`). They are different archives until hashed side by side.
- Server fresh-ZIP inference from the archive root, two runs, byte-exact to `9ea67f81…be4e9`, official checker PASS for all 1536.
- Server Docker with `--network=none`, read-only code and data, 2 CPU, 16 GiB, tmp 128 MiB, output 32 MiB, unprivileged, cap drop ALL: same canonical hash inside the live container, 1.54 s, max RSS 904236 KB. Host 3.12 libraries were bind-mounted read-only. That is not an organizer-image proof.
- Separate server env CPython 3.13.13 + Torch 2.11.0+cpu + NumPy 2.5.4 ran the existing entry and matched `9ea67f81…be4e9` bitwise, 1.36 s, max RSS 610964 KB, checker PASS. A uv freeze of 11 dependencies was written to `requirements_py313_cpu211_frozen.txt`. Tower reproduction of that env was described as download still pending (**NOT RUN** here).
- Stated normalized gap, tower versus server: max abs `3.742622056712719e-7` (same max this review measured) and 3441 cells above 1e-8 (**PROVIDED** count).

## What this does and does not clear

Still true, and not enough to release:

- The entrypoint is a CPU fp32 load of the exact anchor checkpoint, `no_signal` true, test history plus test queries only, with hash and config refusal tests that produced no CSV.
- Format, id set, finiteness, and row sums pass the official checker on the tower CSV.
- Local repeat, signal flip, and the tower sandbox agree with each other.
- Sizes fit the ZIP and CSV ceilings on both the code limit and the stricter prose limit.

Release remains blocked because:

- `requirements.txt` pins `torch==2.9.1`, and the canonical server stack named above is Torch 2.11.0.
- This tower's fp32 CSV is outside 1e-8 of the canonical reference CSV.
- Kit preprovision versus guide offline-wheels is unresolved. A provided server 3.13 CPU success does not rewrite either document.
- No FP64 artifact is in this package, and CNN03 was not reviewed here.

## Not run / not claimed

- Package `test_runtime.py` or any `test_*.py` under the runtime package: absent.
- Install of `torch==2.9.1` CPU wheel, and any tower run of CPython 3.13 or Torch 2.11.
- Organizer image, organizer preprovision, and upload.
- Server ZIP `474978a3…`, server Docker log, and server 3.13 bitwise run: provided only.
- Validation-split regeneration. The entry reads `test/` only (`data.py` line 65).
- End-to-end proof that an in-window history edit changes `predict.py` output. The tick-3 edit was outside W.
- Cell count strictly greater than 1e-8 (the 3441 figure).
- AVX / oneDNN version strings.
- Seed selection, full/confirm metric ranking, CNN03, and FP64. Not started. The anchor file `43dc7050…` was left in place.

## Next checks (not executed)

1. Hash `anchor_1929.zip` and diff member bytes against `entry/` without rewriting either.
2. Read `requirements_py313_cpu211_frozen.txt` and record its 11 pins next to `torch==2.9.1`. Any pin change is a new package edit, outside this review, and must keep checkpoint `43dc7050…`.
3. On the existing `reviewlogs/run1.csv` versus `test_cpu_a.csv`, count cells with normalized abs diff > 1e-8.
4. After a tower env of the frozen 3.13 / Torch 2.11 CPU lock exists, rerun the same `predict.py` command and compare to `9ea67f81…`. Do not treat that future run as done.
5. Keep CNN03 and FP64 in their own reviewed manifest. Do not overwrite this anchor.
