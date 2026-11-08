#!/bin/sh
# Fresh-ZIP sandbox proof. Does not pull, install, or change the image.
set -eu
ROOT=/home/alexn/Documents/code/datathon_2026/house-edge
CAND="$ROOT/experiments/cnn03_convergence/candidate_t146_ensemble_s0_s2_20261108"
PY="$ROOT/experiments/cnn_runtime/py313_verify_1936"
ZIP="$CAND/ensemble_equal_s0_s2_fp64_candidate.zip"
DATA="$ROOT/kit/public"
OUT="$CAND/docker_out"
mkdir -p "$OUT"
run_one() {
  name="$1"
  split="$2"
  set +e
  docker run --name "$name" \
    --pull=never --network=none --read-only \
    --cpus=2 --memory=16g --memory-swap=32g \
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
run_one cand-t146-ens-a test
run_one cand-t146-ens-b test
run_one cand-t146-ens-val val
