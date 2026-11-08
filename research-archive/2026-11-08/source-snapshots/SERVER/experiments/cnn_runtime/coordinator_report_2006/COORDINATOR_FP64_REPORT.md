# FP64 runtime packaging — coordinator evidence
As of Nov 7, 2026, approximately 8:04 PM CST. NO SUBMISSION DONE. qualified_for_upload=false.

## Actual immutable tested artifacts
All relative paths below are under `/home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime` on BOTH server and tower. Box copies are in `/workspace/house-edge-research/cnn-runtime/fp64-evidence-1959/`. These are the coordinator snapshot ZIPs, not the separate `fp64_*_entry.zip` produced later by the coding agent. Core code/checkpoint/config hashes match that successfully tested source exactly.

### anchor
- ZIP `fp64_anchor_snapshot1955.zip`: 1375332 bytes; SHA256 `771ba29562c077cca8a64b2b11acd108cdfc0a765b51183e6e29aaaa16059c70`.
- Actual sandbox CSV `server_fp64_anchor_sandbox1956.csv`: 197488 bytes; SHA256 `af29d6402f5daad1c8b72d5f05bee52f335105e1a1c50fb1e2ebaf127e79c6f2`.
- Official validation R `0.40314938082908086`; kit points `42.760949632139464/60`.
- Actual sandbox log: `server_fp64_anchor_sandbox1956.log`: wrote /output/submission.csv split=test queries=1536 infer_s=1.954204 total_s=2.627659 max_rss_kb=978952 device=cpu dtype=fp64 threads=2; INFERENCE_EXIT=0
- ZIP member `checkpoint.pt` SHA256 `43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80`.
- ZIP member `config/config.json` SHA256 `c9022a4305be1c4124e6a116cd0cf3132fb5624a206f9b5571090ca3c4769d2f`.
- ZIP member `model.py` SHA256 `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a`.
- ZIP member `data.py` SHA256 `af23684b5348c53dafc0e3870706da7ed3dc64c852537371ac7c0ee97d916133`.
- ZIP member `predict.py` SHA256 `8d97923a4bf3a2f3f2df562a07571518b99489d6723db4bd3f6a02b247316f33`.
- ZIP member `requirements.txt` SHA256 `d836e33a0f3deff0abc6d98229a1b16ac92860bf7d462c3e9a738c6f2cc2afa4`.
- ZIP root has 9 files, no forecasts/data/wheels/venv/caches; exact reproduce.json: {   "image": "python",   "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"] }

### cnn03
- ZIP `fp64_cnn03_snapshot1955.zip`: 1376564 bytes; SHA256 `4a7c6edc595bb95930d5afbe4406b2799933d18b2b034ded132352cc47cd23b6`.
- Actual sandbox CSV `server_fp64_cnn03_sandbox1956.csv`: 197886 bytes; SHA256 `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d`.
- Official validation R `0.36845453658648775`; kit points `43.84508099893893/60`.
- Actual sandbox log: `server_fp64_cnn03_sandbox1956.log`: wrote /output/submission.csv split=test queries=1536 infer_s=1.966583 total_s=2.628271 max_rss_kb=978412 device=cpu dtype=fp64 threads=2; INFERENCE_EXIT=0
- ZIP member `checkpoint.pt` SHA256 `ce831b9b41653ae533ba05a64bb79ead340279fb93697caa5ab85c40bcc9203f`.
- ZIP member `config/config.json` SHA256 `6169196bc0827286d5bcbd931980727e0442b1ffdef664cb7f7b319cad4eb4f5`.
- ZIP member `model.py` SHA256 `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a`.
- ZIP member `data.py` SHA256 `af23684b5348c53dafc0e3870706da7ed3dc64c852537371ac7c0ee97d916133`.
- ZIP member `predict.py` SHA256 `a2aa00a32b0af42445014fbe8980cfbbb2afc5644a474c10d48bdf43fc31e6c2`.
- ZIP member `requirements.txt` SHA256 `d836e33a0f3deff0abc6d98229a1b16ac92860bf7d462c3e9a738c6f2cc2afa4`.
- ZIP root has 9 files, no forecasts/data/wheels/venv/caches; exact reproduce.json: {   "image": "python",   "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"] }

## Actual runtime proof
Server: CPython3.13.13, torch2.11.0+cpu, NumPy2.5.4, own project-scoped standalone environment. UV pip-check previously passed all11 packages. No inference-time setup or downloads. Tower: existing CPython3.12.3, torch2.9.1+rocm6.4, CPU only. Tower own3.13 install was stopped after prolonged stall; torch import there remains unavailable. No claim of tower3.13 proof.

Both immutable ZIPs were officially built, freshly extracted, then executed in existing stock ubuntu:24.04 containers with actual model inference, not just a sandbox preflight. Docker inspect verifies network none; read-only root and /submission,/data,/venv/interpreter; CPU2 quota; RAM16GiB and swap16GiB; tmp128MiB; output32MiB; uid:gid1000:1000; cap-dropALL; no-new-privileges. Both model exits0. Output was streamed from /output/submission.csv before tmpfs teardown, not a host output bind. No host OS-library bind was necessary. Exact inspect/log files are in box evidence directory.

Exact default command from ZIP root: `python predict.py --data /data --out /output/submission.csv`. Root reproduce.json has exactly image=python and command array for that command. Validation uses same entry plus `--split val`. No labels/conditions are read by inference. Final saved training checkpoint bytes remain FP32 and exact original SHA; live loaded parameters and all floating mask/tick/position features/logits/softmax are FP64. No global Tensor.float monkeypatch. Integer categorical indexes remain integer. No training or architecture change.

Both server sandbox cold repeats are byte-identical; both tower fresh-extract cold repeats are byte-identical. All4 cross-host comparisons of these exact ZIPs PASS normalized absolute tolerance1e-8: anchor test6.66133814775e-16, validation6.10622663544e-16; CNN03 test9.99200722163e-16, validation4.99600361081e-16. Tower test runtimes about9.15–9.40s at affinity1CPU, peak1,321,092KiB; server2CPU about2.628s, peak978,952KiB. Official checker passes all1536 testIDs and768 validationIDs; finite/simplex/columns checked. Test answers remain private and no test score is claimed.

Commands: official `kit/build_entry.py --source <entry> --out <NEWZIP>`; fresh extraction `python -m zipfile -e <ZIP> <NEWFOLDER>`; default command above; checker `python kit/check_submission.py <CSV> --data kit/public --split test` (or validation); exact official R via `slot_forecasting.rounds.score_round(validation_csv, "kit/public")`; crosshost via read-only `experiments/mlp_cnn_hpo/review/compare_csv.py <towerCSV> <serverCSV>`. Detailed coordinator batch output: box terminal560926.txt (full log, actual exit0), plus evidence Docker logs/inspect and official score JSONs.

## Independent review and test scope
Grok Build authored all Python changes on tower through existing grok.com subscription, API-key overrides unset, grok-4.7 medium; no OpenCodeGo retry after quota exhaustion, no metered fallback/new spend. Main new fp64_harness reports passed dtype/guard/missing/slot/no-signal/label independence tests; however independent reviewer correctly noted the harness future check was window-level, not full-network. A separate review session is adding whole-model cutoff checks in its own fp64_review folder. Do NOT claim written final independent clearance before its report exists. Tower report target: `experiments/cnn_runtime/fp64_review/INDEPENDENT_REVIEW_FP64.md`; tmux `he-cnn-runtime-fp64-review`, session01a1b8bf-3558-7972-8d3b-302133e8b4b8, bounded420s from about7:58PM CST. Old independent review is explicitly BLOCKED and applies to old FP32 entry, not these new variants.

## BLOCKERS — do not upload these provisional ZIPs
1. Existing FP64 requirements.txt declares `torch==2.11.0+cpu` and NumPy2.5.4 but puts the official CPU wheel URL only in a COMMENT. No CPU index/direct wheel requirement is active. Installation from that declaration was not proven; do not assume ordinary PyPI resolves +cpu. Actual verified server installation used the direct CP313 CPU wheel URL. The complete verified11-package lock exists, but the Grok follow-up Execute sha256sum+copy update was BLOCKED: "permission required ... was not run, as a headless run cannot ask for approval". That command was NOT retried, split, renamed, turned into a script, done through another tool, or manually applied by coordinator. Parent must obtain proper approval for the exact pending update before it can be applied; no permission waiver. See proposed full contents below.
2. Kit RULES/SUBMISSION_README explicitly say organizers preinstall pinned requirements then run offline; live guide instead describes Python3.13 stdlib/offered environments or compatible bundled wheels. No precedence or organizer promise is invented. Torch CPU wheel is not bundled. Numeric/runtime computational proof does not resolve organizer environment availability. Guide80/20 versus kit60/40 and differing size prose/code are recorded; artifacts are below stricter100MBZIP/10MBCSV.
3. Independent final review is pending at this report write. Do not confuse original diagnostic global Tensor.float patch with scoped final package. Current code hashes match our actual sandboxed source.

## Proposed exact dependency change — DRAFT ONLY, NOT APPLIED
Replace ONLY `fp64_anchor_entry/requirements.txt` and `fp64_cnn03_entry/requirements.txt` with this verified server freeze (SHA256cf701c4d2f312e39a8883c114ed354106fbb7b7d0cd5968f99cc6bd8746a0cab), then update provenance dependency hash/docs and build NEW ZIP names, preserve existing ZIPs, rerun fresh ZIP offline proof. User approval through the blocked Grok action is still required; this draft is not a workaround.
```text
filelock==4.2.9
fsspec==2026.9.0
jinja2==3.1.6
markupsafe==3.0.4
mpmath==1.3.0
networkx==3.7
numpy==2.5.4
setuptools==81.0.0
sympy==1.14.0
torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.11.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl
typing-extensions==4.16.0
```

## Preservation and exclusions
Clean anchor43dc7050... preserved, original CPU CSV9ea67... preserved, clean-reference.tar.gz preserved; contaminated originalg(.378) excluded. CNN03 is explicitly separate tune-selected seed0 ce831b... (not chosen from best full/confirmation seed). No ensemble, no future checkpoint replacement, no GPU/training/shared installs/network restriction changes/git commits/push/destruction/organizer messages. No submission. Deadline target7:30–7:45 qualified release was not achieved; actual scoped FP64 computational proof completed around7:56–8:01PM CST, with the above release blockers still open.

## Exact denied action — do not execute without proper approval
Working directory: /home/alexn/Documents/code/datathon_2026/house-edge on tower as alexn. Grok headless action was NOT RUN.
```sh
sha256sum experiments/cnn_runtime/requirements_py313_cpu211_frozen.txt && cp -a experiments/cnn_runtime/requirements_py313_cpu211_frozen.txt experiments/cnn_runtime/fp64_anchor_entry/requirements.txt && cp -a experiments/cnn_runtime/requirements_py313_cpu211_frozen.txt experiments/cnn_runtime/fp64_cnn03_entry/requirements.txt && sha256sum experiments/cnn_runtime/fp64_anchor_entry/requirements.txt experiments/cnn_runtime/fp64_cnn03_entry/requirements.txt experiments/cnn_runtime/fp64_anchor_entry/model.py experiments/cnn_runtime/fp64_anchor_entry/data.py experiments/cnn_runtime/fp64_anchor_entry/predict.py experiments/cnn_runtime/fp64_anchor_entry/checkpoint.pt experiments/cnn_runtime/fp64_anchor_entry/config/config.json experiments/cnn_runtime/fp64_cnn03_entry/model.py experiments/cnn_runtime/fp64_cnn03_entry/data.py experiments/cnn_runtime/fp64_cnn03_entry/predict.py experiments/cnn_runtime/fp64_cnn03_entry/checkpoint.pt experiments/cnn_runtime/fp64_cnn03_entry/config/config.json
```

Independent reviewer420s run ended without a written verdict. It was resumed at about8:06PM CST for120s, REPORT ONLY (no new tests, no denied action retries). New log grok-fp64-review-finish.log. Whole-model causality completion is not yet claimed.
