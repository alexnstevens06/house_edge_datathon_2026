# Runtime fallback lane — BLOCKED/partial as of 2026-11-07 19:09 CST

No submission, upload, commit, push, model/training change, shared source edit, dependency installation or provider switch performed. No ZIP exists yet.

## Immediate blocker: paid coding CLI stopped progressing

- Tower existing paid `opencode-go/glm-5.2` builder launched 18:50 CST. It read contract and source, wrote scoped CONTRACT.md and STATUS.md, then had no further log updates after 18:54:53 CST. The sanctioned SSH helper hit its 900-second timeout at 19:05 CST; its builder process was then confirmed gone.
- Independent readonly paid reviewer similarly produced no final review and no progress after 18:55:18 CST; canceled ONLY that own stalled process.
- Even sanctioned `opencode session list` produced no output for >3 minutes; canceled ONLY that diagnostic. No app database or credentials opened or modified.
- Same paid-model retry, persistent tmux `he-cnn-runtime-retry`, 19:05 CST, never got beyond model banner; canceled own process after >2 minutes, exit143 logged.
- Lower-plugin same-CLI retry `opencode run --pure -m opencode-go/glm-5.2` launched 19:07 CST in tmux `he-cnn-runtime-pure` with timeout1200. Last observed model banner only at19:08:35; final disposition to follow STATUS.txt.
- No provider error/quota response was logged, so root cause is UNPROVEN. Public `curl -I https://opencode.ai` from tower returned200. Other smallCNN/synthetic build logs also stopped around18:55 (read-only observation only; never interrupted other lanes).
- Tower host19:03 CST: load4.41 across16logicalCPU;16GiB RAM available, swap0used;119GB diskfree. NOT evidence of CPU/RAM/disk saturation.

## What is secured and freshly checked

Exact clean CNN anchor, no_signal=True, no synthetic influence. Server run root:
`/home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn/runs/g_nosignal`

Checkpoint SHA256:
`43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80`
Test CPU CSV SHA256:
`9ea67f812b80de18f03b1f6d8f0aeb6d5b91d8ce5c935dc83ba588e04b6be4e9`
Validation CPU CSV SHA256:
`50efa7cb18f6c6a925b0259ac3177136789893dc16f82a2075cc3b4afe903c1d`

Fresh server command:
`.venv/bin/python kit/check_submission.py experiments/cnn/runs/g_nosignal/test_cpu_a.csv --data kit/public`
Result: `VALID FORMAT: 1536 queries | main | test`, season5fffdc4d0c700bbf61c3f911, exact expected test SHA. 1537lines includingheader. Official checker explicitly says no upload and code reproduction separate.

Exact server source+checkpoint+reference CPU exports staged to TOWER under:
`experiments/cnn_runtime/reference/{experiments/cnn/{source,runs/g_nosignal},src}`
All staged ONLY in own runtime folder, no overwrite of existing CNN work. Box exact transferable archive:
`/workspace/house-edge-research/cnn-runtime/clean-reference.tar.gz`
Archive SHA256267abdcb90e3ddbb7ac54d953f376715d993f11440d87f2686a192d8c4e8b5b3.
Standalone box checkpoint/CSV/config in:
`/workspace/house-edge-research/cnn-runtime/anchor/experiments/cnn/runs/g_nosignal/`

## Contract evidence and unresolved issues

Kit RULES.md55 and SUBMISSION_README.md23 explicitly say root pinned requirements.txt dependencies are prepared by organizers before offline prediction. Exact config root reproduce.json has ONLY image='python' and argument-list command. Example `python predict.py --data /data --out /output/submission.csv`. CWD/submission and current-round/data readonly; /tmp128MiB, /output32MiB writable. 2CPU16GiB30min; normalized probability tolerance1e-8.

Live public guide verified this session through public challenge link:
https://tamudatathon.com/challenges/4
https://docs.google.com/document/d/1CqNLjCiojRxKkZopoxBeilkke6Y4yCuz7FYSuefvCMo/export?format=txt
Guide STILL says Python3.13 standard-library environment; extra libraries offered environment or compatible bundled offline wheels/setup. Also80/20 versus kit60/40 scoring, and128MiB ZIP/16MiB CSV versus kit prose100MB/10MB. No invented resolution. Below100MB/10MB would satisfy both size limits.

Public CPU torch2.9.1 cp313 manylinux2_28_x86_64 wheel exists, SHA39b3dff6d8fba240ae0d1bede4ca11c2531ae3b47329206512d99e17907ff74b; HEAD size184380800bytes before any archive recompression. Not downloaded/installed/tested. Tower existing env Python3.12.3, torch2.9.1+rocm6.4, numpy2.5.4. This is NOT organizer3.13 proof. Server CPU AMD9950X vs towerAMD3700X: portability tolerance must actually be measured, not inferred from local repeats.

## Sandbox preflight actually passed

Tower command:
`docker run --rm --pull=never --network=none --read-only --cpus=2 --memory=16g --tmpfs /tmp:rw,size=128m --tmpfs /output:rw,size=32m ubuntu:24.04 /bin/true`
Exit0. Existing image used, no downloads/network/global changes. This proves sandbox launch only, NOT model offline inference. Can potentially readonly-bind existing Python/venv/libs for a local offline proof, honestly distinct from organizer3.13 environment.

## Not completed

No entry/predict.py, reproduce.json, requirements.txt or upload ZIP exists. No fresh extraction/inference/repeat/portability diff, readonly model prediction sandbox, runtime/memory measurement, package tests or final independent review completed. No CNN03 variant attempted; anchor preserved. CNN03 requires reviewed source/checkpoint SHA and tune-only matched-seed promotion, not full/confirm seed selection.

## Next action for coordinator

Provider/CLI stall needs immediate attention. Explicit paid-tower-GLM-only instruction prevents silently switching code authorship to another route. User's newer Grok Build permission was explicitly described for synthetic, not automatically runtime. If that same existing-subscription route is authorized for runtime, hand the staged exact reference + RETRY_TASK.md to it; no new research is necessary. Otherwise retain bounded same-CLI retry and report no usable package yet. No organizer messages were sent. No submission done.
