## User

Authorized existing-subscription Grok Build runtime packaging. Own ONLY experiments/cnn_runtime in project root /home/alexn/Documents/code/datathon_2026/house-edge. No shared edits/git/install/newtraining/GPU/downloads/credentials/messages/upload/submission. CPU2. CREATE executable package NOW, no repeated contract research. Parent single uploader owns uploads. First usable anchor19:25CST; verified package19:30-45.
Exact clean anchor source+checkpoint+CPU references already here: experiments/cnn_runtime/reference/{experiments/cnn/{source,runs/g_nosignal},src}. USE exact reference source not stale tower sharedsource. Checkpoint SHA43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80. Test CPU refsha9ea67f812b80de18f03b1f6d8f0aeb6d5b91d8ce5c935dc83ba588e04b6be4e9. Valsha50efa7cb18f6c6a925b0259ac3177136789893dc16f82a2075cc3b4afe903c1d. Anchor no_signalTrueR.4031 contaminatedoriginalg.378EXCLUDED. No modelchange/ensemble/seedselection. HPOCNN03 separate later onlyafter reviewedsource/checkpointmanifest; do not waitHPO.
CREATE entry with minimal predict.py/model.py/data.py copied exactforward/preprocessing fromreference. Root README.md/reproduce.json/requirements.txt/checkpoint.pt/config/provenance. reproduce.json exactkeys image=python and command=[python,predict.py,--data,/data,--out,/output/submission.csv]. Testonly history/queries inputs; no train/val labels/conditions/signal/network/checkpointwrites. Exact fp32CPUeval and originalexportformat. Failclosed unclean config/anchorhash. Standalone accepts actualpaths. Run immediately existing .venv/bin/python entry/predict.py --data kit/public --out experiments/cnn_runtime/first.csv. Compare all1536IDs/header/finitesimplex/normalizeddiff<=1e-8 versusreference ideallybitwise. ServerCPU9950X vs tower3700X measureportability. If mismatch report, do not silentlychange precision. RepeatCPU2 exactCSV and freshZIPextraction; official kit/check_submission.py, kit/build_entry.py and inspect_archive. ZIPoutsideentry below100MB, rootREADME/reproducejson, no venv/git/trainingdata/logs. Pinnedrequirements exactversions/provenance. Existing envPython3.12.3 Torch2.9.1+rocm6.4 NumPy2.5.4; no installs. CPUtorch2.9.1requirements withofficialcpuindex may be viable but label distributionuntested.
Actual model inference offline sandbox proof required, not merely/bin/true. Dockerworks existingubuntu24.04 (--pull=never); readonlybindexistinghostPython/venv/libs/neededROCm libs WITHOUTGPUdevices forlocalproof. --networknone --readonly --cpus2 --memory16g readonlysubmission/data, tmpfs/tmp128MiB /output32MiB. Currentpublic readonly/data; entry readonly/submission; writes/output/submission.csv. Retaincontainer tocopyCSVout or limitedownoutputmountwithhonestlimitations. No pulls/installs/hostnetworkchanges. This host3.12libs proofNOTorganizer3.13proof. SANDBOX_AVAILABLE.md hasprovenpreflightcommand. Measureinference runtime/RSS andoutputsize. Negative tests tampercheckpoint/signalmode/missingdata; tests causality/missingvs0/slotmapping exactsource. Testscriptoutsideentry. Separate reviewer follows.
Contract alreadyresearched: CONTRACT.md LIVE_PARTICIPANT_GUIDE.txt. KitRULES55/SUBMISSION_README23 explicitly organizers preprovisionrootpinnedrequirements thenoffline. LiveguidePython3.13stdlib/offeredenv/offlinewheels contradicts; score80/20vs60/40,sizes128MiBvs100MB. Noinventedresolution/noorganizerreadyclaim. Rootrequirements viableunderkitrouteonly; cp313CPUtorch2.9.1wheel exists184380800bytes,nobundling/downloadsnow. Artifactsmallfitsbothsizes. Final REPORT.md paths/hashes/commands/tests/runtime/size/unverifieditems. No externalcontacts. Startentryfiles beforelengthyplanning.

## Assistant

I'll package only `experiments/cnn_runtime` from the clean anchor reference and start the entry files immediately.

## Tools

- ListDir: experiments/cnn_runtime
- ListDir: .
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/reference/experiments/cnn/source/cnn.py
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/reference/experiments/cnn/source/predict_cpu.py
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/config.json
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/SANDBOX_AVAILABLE.md
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/reference/src/house_edge/data.py (1-121)
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/kit/public/test/sample_submission.csv (1-5)
- Search: def write_submission_csv|def check|inspect_archive|SUBMISSION
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/test_cpu_a.csv (1-3)
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/kit/SUBMISSION_README.md (1-52)
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/reference/src/house_edge/scoring.py (1-69)
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/kit/slot_forecasting/submission_contract.py
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/kit/build_entry.py (1-66)
- Search: PROBABILITY_COLUMNS
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/kit/public/metadata.json (1-24)
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/kit/check_submission.py (1-46)
- Execute: sha256sum experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/checkpoint.pt experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/test_cpu_*.csv experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/test_predictions.csv experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/validation_cpu_*.csv experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/config.json; .venv/bin/python -c 'import sys,torch,numpy; print(sys.version); print(torch.__version__); print(numpy.__version__); print("threads", torch.get_num_threads())'; stat -c '%s' experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/checkpoint.pt; command -v time; ls -ld .venv /usr/bin/python3.12; readlink -f .venv/bin/python; python3 -c 'import pathlib; p=pathlib.Path(".venv/bin/python"); print(p.resolve())' (Hash anchor artifacts and print env versions)
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/PUBLIC_DEPENDENCY_EVIDENCE.md (1-2)
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/metrics.json (1-30)
- Search: "R"
- Execute: head -1 kit/public/test/queries.csv kit/public/test/history.csv; .venv/bin/python -c 'import torch,os; print(torch.__file__); print(os.cpu_count())'; ldd .venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so 2>/dev/null | head -40; echo '---'; ls .venv/lib/python3.12/site-packages/torch/lib | head -50 (Inspect data headers and torch library deps)
- Execute: env | grep -Ei 'OMP|MKL|THREAD|OPENBLAS|CUDA|HIP|ROCR' || true; .venv/bin/python - <<'PY'
import os, torch
print("omp", os.environ.get("OMP_NUM_THREADS"))
print("num", torch.get_num_threads(), "interop", torch.get_num_interop_threads())
print("mkldnn", torch.backends.mkldnn.enabled, "cpu_cap", torch.backends.cpu.get_cpu_capability() if hasattr(torch.backends,"cpu") else None)
PY
sha256sum experiments/cnn_runtime/reference/experiments/cnn/source/cnn.py experiments/cnn_runtime/reference/experiments/cnn/source/predict_cpu.py experiments/cnn_runtime/reference/src/house_edge/data.py experiments/cnn_runtime/reference/src/house_edge/scoring.py
ldd /usr/bin/python3.12
# numpy libs
find .venv/lib/python3.12/site-packages/numpy -name '*.so' | head -5 | while read f; do echo "== $f"; ldd "$f" | awk '/not found|{print}' ; done
uname -m; lscpu | awk -F: '/Model name|CPU\(s\)|Thread|Vendor/ {print}' (Check thread env, source hashes, loader deps)
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/REFERENCE_READY.md (1-1)
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/test_timing.json
- Read: /home/alexn/Documents/code/datathon_2026/house-edge/.venv/pyvenv.cfg
- Execute: ls -l .venv/bin/python .venv/bin/python3 .venv/bin/python3.12; # stdlib size hint
du -sh /usr/lib/python3.12 .venv/lib/python3.12/site-packages/torch .venv/lib/python3.12/site-packages/numpy 2>/dev/null | head (Check venv python links and library sizes)

## Assistant

The checkpoint hash matches the anchor. I'll add a self-contained CPU entry that uses that checkpoint and the reference forward pass.

## Tools

- Execute: mkdir -p experiments/cnn_runtime/entry/config experiments/cnn_runtime/entry/provenance && cp -a experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/checkpoint.pt experiments/cnn_runtime/entry/checkpoint.pt && cp -a experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/config.json experiments/cnn_runtime/entry/config/config.json && sha256sum experiments/cnn_runtime/entry/checkpoint.pt experiments/cnn_runtime/entry/config/config.json && python3 -c 'import json; print(json.load(open("experiments/cnn_runtime/entry/config/config.json")))' (Copy anchor checkpoint and config into entry)
- Edit: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/entry/data.py
- Edit: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/entry/model.py
- Edit: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/entry/predict.py
- Edit: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/entry/reproduce.json
- Edit: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/entry/requirements.txt
- Edit: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/entry/provenance/provenance.json
- Edit: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/entry/README.md
- Edit: /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_runtime/entry/predict.py
- Execute: chmod +x experiments/cnn_runtime/entry/predict.py && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/entry/predict.py --data kit/public --out experiments/cnn_runtime/first.csv (Run CPU anchor inference to first.csv)
- Execute: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/entry/predict.py --data kit/public --out experiments/cnn_runtime/second.csv && .venv/bin/python - <<'PY'
import csv, hashlib, math
from pathlib import Path
ref = Path("experiments/cnn_runtime/reference/experiments/cnn/runs/g_nosignal/test_cpu_a.csv")
ours = Path("experiments/cnn_runtime/first.csv")
second = Path("experiments/cnn_runtime/second.csv")
def load(p):
    text = p.read_bytes()
    rows = list(csv.reader(text.decode().splitlines()))
    return text, rows[0], rows[1:]
for label, path in (("first", ours), ("second", second), ("ref", ref)):
    b = path.read_bytes()
    print(label, len(b), hashlib.sha256(b).hexdigest())
bt, ht, rt = load(ours)
bs, hs, rs = load(second)
br, hr, rr = load(ref)
print("header", ht, "match_ref", ht==hr, "match_second", ht==hs)
print("nrows", len(rt), len(rr), "ids_equal_order", [r[0] for r in rt]==[r[0] for r in rr])
print("repeat_bitwise", bt==bs)
max_abs = 0.0
max_norm = 0.0
n_diff = 0
nonfinite = 0
simplex_bad = 0
for a,b in zip(rt, rr):
    if a[0]!=b[0] or len(a)!=6:
        print("row mismatch", a[0], b[0]); break
    av=[float(x) for x in a[1:]]; bv=[float(x) for x in b[1:]]
    if any(not math.isfinite(x) for x in av):
        nonfinite += 1
    s=sum(av)
    if abs(s-1.0)>1e-12:
        simplex_bad += 1
    for x,y in zip(av,bv):
        d=abs(x-y)
        if d:
            n_diff += 1
        if d>max_abs:
            max_abs=d
        denom=max(abs(y), 1e-12)
        max_norm=max(max_norm, d/denom)
print("cells_differ", n_diff, "max_abs", max_abs, "max_rel", max_norm, "nonfinite", nonfinite, "simplex_bad", simplex_bad)
print("bitwise_vs_ref", bt==br)
PY (Repeat inference and compare CSV to CPU reference)