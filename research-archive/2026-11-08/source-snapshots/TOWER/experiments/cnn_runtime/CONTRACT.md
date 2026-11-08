# CONTRACT.md — CNN offline runtime fallback package (experiments/cnn_runtime)

Written FIRST per TASK.md, before any package code. Evidence-only: every claim below cites a
file/line in the organizer kit or the live/PDF participant guide. Contradictions are reported as
unresolved; no resolution is invented. All build work stays under `experiments/cnn_runtime/`.

## 1. Verified contract facts (organizer CODE — the enforced contract)

From `kit/slot_forecasting/submission_contract.py` (the actual archive validator, also used by
`kit/build_entry.py`):

| Requirement | Evidence |
|---|---|
| Code ZIP nonempty, ≤ 128 MiB compressed | `MAX_ZIP = 128*1024*1024`, inspect_archive (submission_contract.py:11,28-29) |
| ≤ 5000 entries, ≤ 512 MiB unpacked | `MAX_FILES`, `MAX_EXPANDED` (submission_contract.py:12-14,33-34) |
| `README.md` and `reproduce.json` at ZIP root, each ≤ 64 KiB; README nonempty UTF-8 | submission_contract.py:51-55 |
| `reproduce.json` has EXACTLY keys `{"image","command"}` | submission_contract.py:57-58 |
| `image` matches `[a-z][a-z0-9_-]{0,31}` → `"python"` | submission_contract.py:59-60 |
| `command` = JSON list, 1..64 nonempty string args ≤4096 chars | submission_contract.py:61-64 |
| No symlinks/encrypted/special files/duplicate (case-folded) paths/unsafe paths; CRC verified | safe_name + inspect_archive (submission_contract.py:17-49,66-74) |
| `package_code(source, out)`: zips a plain folder; ZIP must sit OUTSIDE the source folder | submission_contract.py:101-123; build_entry.py:22-23 |
| ZIP must open directly to root files, no enclosing folder | kit/README.md:349 |

From `kit/check_submission.py` + `kit/slot_forecasting/{io,schema}.py` (CSV validation):

| Requirement | Evidence |
|---|---|
| CSV columns exactly `query_id,p_0,p_1,p_2,p_5,p_20` | `SUBMISSION_COLUMNS` (schema.py:10, io.py read_submission) |
| Every test query exactly once, no extras/dupes | require_ids (io.py), check_submission.py:24-25 |
| Probabilities finite, in [0,1], `math.fsum` within 1e-6 of 1; then normalized `v/total` | probabilities() (schema.py:63-76) |
| Checker prints SHA-256 of the CSV bytes | check_submission.py:31 |
| `--split validation` scores via official contest code | check_submission.py:26,33-35 |

Runtime/execution contract (kit/README.md:324-351; RULES.md:49-54; guide §7):

| Requirement | Evidence |
|---|---|
| reproduce.json example: `{"image":"python","command":["python","predict.py","--data","/data","--out","/output/submission.csv"]}` | kit/README.md:317-322 |
| Command runs from `/submission` (code, read-only); `predict.py` must accept `--data` and `--out` | kit/README.md:324,333 |
| Mounts: `/data` current round public data RO; `/main_data` (adaptation only); `/tmp` writable ≤128 MiB; `/output` writable ≤32 MiB, final CSV at `/output/submission.csv` | kit/README.md:326-329 |
| No network access, no hidden answers, uploaded CSV not given to runner | kit/README.md:331 |
| 30 minutes, 2 CPU cores, 16 GiB RAM; max per-probability difference 1e-8 after CSV normalization | kit/README.md:351; RULES.md:54 |
| Extra packages: root `requirements.txt` with EXACT versions; organizers prepare them BEFORE the offline run; keep `"image":"python"` | kit/README.md:333; RULES.md:54 |

## 2. Unresolved contradictions (reported, NOT resolved — do not invent)

1. **Points split.** Kit README.md:192 + RULES.md:9 say Main **60** / Round two **40**.
   Participant guide (PDF §4, docs/house-edge-participant-guide.pdf; live Google-Doc text
   LIVE_PARTICIPANT_GUIDE.txt:170) says Main **80** / Round two **20**. Same R formula, different
   round maxima (scoring example tables differ accordingly). No organizer statement resolves
   which is frozen. Affects strategy only, not this package.
2. **CSV size limit.** Kit README.md:291,351 + RULES.md:51: CSV ≤ **10 MB** (MB = 1,000,000
   bytes). Guide (PDF §7; live txt:297) and kit CODE `MAX_CSV = 16*1024*1024`
   (submission_contract.py:14): **16 MiB**. Unresolved prose-vs-code.
3. **Code ZIP size limit.** Kit README.md:291,349 + RULES.md:51: **100 MB**. Guide (PDF §7; live
   txt:296) and kit CODE `MAX_ZIP = 128 MiB` (submission_contract.py:11): **128 MiB**. Unresolved.
4. **Extra dependencies.** Kit README.md:333 + RULES.md:54: pinned root `requirements.txt`;
   "Organizers prepare those dependencies in an isolated environment before the offline
   prediction run." Guide (PDF §7; live txt:290): "The standard python environment uses Python
   3.13 with the standard library... For extra libraries, use an offered environment or include
   compatible Linux wheels and an offline setup script." The two mechanisms differ materially
   for a torch+numpy entry. torch wheels cannot fit under either ZIP limit (a CPU torch wheel is
   ~200 MB before deps), so wheel-bundling is infeasible; this package therefore follows the kit
   README/RULES `requirements.txt` mechanism and documents the residual risk. UNRESOLVED.
5. **Submission channel.** Kit README.md:284-298: Google Forms + Google sign-in, latest response
   counts. Guide (PDF §7; live txt:420-457): upload page with team token, receipts, statuses
   (queued/running/passed/mismatch/failed), 20 entries/round, 12 attempts/min. Unresolved; does
   not affect archive contents.

Design consequence: the package stays safely under BOTH stated limits in each contradiction
(ZIP < 100 MB and ≪128 MiB; CSV ≈ 90 KiB ≪ 10 MB and ≪16 MiB), so either reading passes.

## 3. Model/artifact contract (clean-run-only, exact checkpoint)

- Clean model: `g_nosignal` — trained with `--h 128 --layers 4 --W 64 --drop 0.2 --epochs 30
  --lr 1e-3 --seed 0 --no_signal` (reference config.json; rerun.sh). Validation R = 0.403149
  (points_60 = 42.760949), matching TASK.md.
- The signal-using run `g_h128_L4_W64_d02_e30` is CONTAMINATED (signal leak, REVIEW_OPENCODE.md
  Q1; marker file `experiments/cnn/runs/g_h128_L4_W64_d02_e30.CONTAMINATED.marker`) and is NEVER
  packaged.
- Exact checkpoint required: sha256 starts `43dc7050a98dd23e`. The staged reference copy
  `experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/checkpoint.pt` hashes to
  `43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80` — EXACT match. No other
  checkpoint may be substituted (TASK.md: "Exact checkpoint only; no model changes").
- Reference CPU fp32 exports (server, 2 threads), all bitwise identical (the "4 bitwise CPU
  exports"): `test_cpu_{a,b,c,d}.csv` → sha256 `9ea67f812b80de18f03b1f6d8f0aeb6d5b91d8ce5c935dc83ba588e04b6be4e9`;
  `validation_cpu_{a,b,c,d}.csv` → sha256 `50efa7cb18f6c6a925b0259ac3177136789893dc16f82a2075cc3b4afe903c1d`.
  Both match TASK.md. These are the reproduction targets (bitwise ideal, normalized max diff
  ≤ 1e-8 required).
- Checkpoint internals (verified read-only, review-prelim.log): top keys `state`/`config`; config
  `no_signal: true`; 43 tensors, all float32; NO signal-embedding keys.
- Inference source = EXACT copy of the reference (server) CNN architecture/forward and data
  preprocessing: `reference/experiments/cnn/source/cnn.py` is byte-identical to tower's
  `experiments/cnn/source/cnn.py` (diff verified; only `__pycache__` and extra sweep files
  differ). `reference/src/house_edge/data.py` and `scoring.py` are byte-identical to tower's
  `src/house_edge/`. Only `train.py` differs (server AMP env toggle — training-only, unused by
  inference). Staging note (REFERENCE_READY.md): use the reference copy, which we do.

## 4. Package invariants (asserted by tests)

1. Predicts from `<data>/test` only: reads `queries.csv` + `history.csv`; never reads
   `labels.csv`, `conditions.csv`, `reference.csv`, `targets.csv`, `sample_submission.csv`, or
   any train data; no network/API; no training.
2. Signal/condition features: the `signal` column is never used as a feature
   (`no_signal=True`; SG is all-zero; flipping every signal value leaves predictions
   bit-identical — asserted). Condition labels are never read (not present in test split).
3. Causality: features use ONLY ticks strictly below the cutoff tick on all machines
   (`cnn.window`; mutating ticks ≥ cutoff leaves predictions bit-identical — asserted on real
   gappy episodes).
4. Missing-tick semantics: unobserved ticks are MISSING (-1), never zero payouts; the explicit
   mask channel encodes observed/missing exactly as the clean reference (asserted windows
   contain MISSING on short-history episodes).
5. Contamination guard: `predict.py` refuses (before loading any split data) a checkpoint whose
   config lacks `no_signal=True` or a run dir containing a `*CONTAMINATED*` marker (guard logic
   copied from `predict_cpu.py:14-25`).
6. Determinism: fixed batch order/size, `torch.set_num_threads(2)`, `model.eval()`,
   `torch.no_grad()`; repeated CPU 2-thread runs must produce a bit-identical CSV (asserted).
7. CSV writer: byte-exact copy of `house_edge.scoring.write_submission_csv` semantics (float64
   renormalization, `repr()` formatting, `csv.writer` with `\n` line terminator, sorted
   query_id order) so bitwise equality with the reference exports is meaningful.
8. Archive excludes `.git`, venv, training data, labels, unrelated outputs; stays < 100 MB
   (both limits) with `README.md` + `reproduce.json` at root and the exact pinned
   `requirements.txt`.

## 5. Environment facts (honest, this machine)

- Tower (this machine): Python 3.12.3 (.venv), torch 2.9.1+rocm6.4, numpy 2.5.4; AMD GPU
  present but NOT used (CPU-only, 2 threads, per TASK.md budget).
- Organizer runner (per kit/guide): `python` image = Python 3.13 + stdlib; torch/numpy are NOT
  stdlib → this entry relies on the kit-README pinned-`requirements.txt` provisioning mechanism
  (contradiction §2.4 unresolved). Wheels not bundled (size-infeasible). Py3.13 compatibility of
  pinned torch 2.9.1/numpy 2.5.4: cp313 wheels exist upstream, but organizer-side provisioning is
  UNVERIFIED (documented in REPORT.md).
- Cross-machine bitwise risk: CPU kernel/ISA differences (server vs tower vs organizer runner)
  can change float32 summation order. Server's own 4 exports are bitwise identical among
  themselves. Tower-vs-server bitwise equality is tested here; organizer-runner equality can
  only be verified by the organizers' rerun (1e-8 normalized tolerance is the binding limit).

## 6. Execution order (per TASK.md)

CONTRACT.md (this file) → STATUS.md → entry package → local CPU2 verification (bitwise vs
reference, repeat, official checker) → tests → official packaging + archive inspection →
fresh-extraction rerun + failure tests → sandbox attempt or documented fallback → REPORT.md.
No commits/pushes/submissions. No shared-source edits (everything copied INTO
experiments/cnn_runtime/). Checkpoint staging status: PRESENT and hash-verified (§3); remaining
staging needs, if any, are recorded in REPORT.md.
