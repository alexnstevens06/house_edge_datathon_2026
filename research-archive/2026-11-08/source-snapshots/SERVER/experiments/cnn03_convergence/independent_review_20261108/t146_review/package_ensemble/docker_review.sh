#!/bin/sh
# Independent strict 2 GiB fresh-ZIP proof. No pull and no install.
set -eu
ROOT=/home/alexn/Documents/code/datathon_2026/house-edge
REVIEW="$ROOT/experiments/cnn03_convergence/independent_review_20261108/t146_review/package_ensemble"
ZIP="$ROOT/experiments/cnn03_convergence/candidate_t146_ensemble_s0_s2_20261108/ensemble_equal_s0_s2_fp64_candidate.zip"
PY="$ROOT/experiments/cnn_runtime/py313_verify_1936"
DATA="$ROOT/kit/public"
OUT="$REVIEW/docker_out"
mkdir -p "$OUT"
docker image inspect ubuntu:24.04 --format '{{.Id}}' > "$OUT/image_id.txt"
run_one() {
  name="$1"
  split="$2"
  set +e
  docker run --name "$name" \
    --pull=never --network=none --read-only \
    --cpus=2 --memory=2147483648 --memory-swap=2147483648 \
    --tmpfs /tmp:rw,size=128m --tmpfs /output:rw,size=32m \
    --user 1000:1000 \
    --cap-drop=ALL --security-opt no-new-privileges \
    -v "$PY:$PY:ro" \
    -v "$ZIP:/pkg/ensemble_equal_s0_s2_fp64_candidate.zip:ro" \
    -v "$DATA:/data:ro" \
    -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 \
    -e CUDA_VISIBLE_DEVICES= -e HIP_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
    ubuntu:24.04 \
    /bin/bash -c "set -euo pipefail; echo NPROC=\$(nproc) >&2; mkdir -p /tmp/extract; \"$PY/env/bin/python\" -c 'import sys,torch,zipfile; print(sys.version, file=sys.stderr); print(torch.__version__, file=sys.stderr); zipfile.ZipFile(\"/pkg/ensemble_equal_s0_s2_fp64_candidate.zip\").extractall(\"/tmp/extract\")'; \"$PY/env/bin/python\" /tmp/extract/predict.py --data /data --out /output/submission.csv --split $split > /tmp/predict.err 2>&1; cat /tmp/predict.err >&2; cat /output/submission.csv" \
    > "$OUT/$name.csv" 2> "$OUT/$name.err"
  rc=$?
  set -e
  echo "$name EXIT:$rc" | tee "$OUT/$name.exit"
  docker inspect "$name" > "$OUT/$name.inspect.json"
  docker rm "$name" >/dev/null
  if [ "$rc" -ne 0 ]; then
    echo "container $name failed" >&2
    exit "$rc"
  fi
}
run_one rev-t146-ens-2g-a test
run_one rev-t146-ens-2g-b test
run_one rev-t146-ens-2g-val val
python3 - <<'PY'
import hashlib, json
from pathlib import Path
out = Path("/home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn03_convergence/independent_review_20261108/t146_review/package_ensemble/docker_out")
rows = []
for name in ("rev-t146-ens-2g-a", "rev-t146-ens-2g-b", "rev-t146-ens-2g-val"):
    blob = json.loads((out / f"{name}.inspect.json").read_text())[0]
    host = blob["HostConfig"]
    state = blob["State"]
    data = (out / f"{name}.csv").read_bytes()
    rows.append({
        "name": name,
        "exit": state.get("ExitCode"),
        "oom": state.get("OOMKilled"),
        "memory": host.get("Memory"),
        "memory_swap": host.get("MemorySwap"),
        "nano_cpus": host.get("NanoCpus"),
        "network": host.get("NetworkMode"),
        "readonly": host.get("ReadonlyRootfs"),
        "cap_drop": host.get("CapDrop"),
        "user": blob["Config"].get("User"),
        "security": host.get("SecurityOpt"),
        "csv_sha256": hashlib.sha256(data).hexdigest(),
        "csv_bytes": len(data),
    })
print(json.dumps(rows, indent=1))
PY
