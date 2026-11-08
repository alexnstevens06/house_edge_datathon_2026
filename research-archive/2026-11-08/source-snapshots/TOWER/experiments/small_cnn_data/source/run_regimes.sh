#!/bin/bash
# Launch the five regimes with at most 2 concurrent GPU jobs.
# Each regime is its own tmux session: scnn-<regime>.
set -u
REPO="/home/alexn/Documents/code/datathon_2026/house-edge"
ROOT="$REPO/experiments/small_cnn_data"
PY="$REPO/.venv/bin/python"
cd "$REPO"
mkdir -p "$ROOT/runs"
QLOG="$ROOT/runs/queue.log"
echo "QUEUE_START $(date --iso-8601=seconds)" >> "$QLOG"

start_one() {
  local name="$1"
  mkdir -p "$ROOT/runs/$name"
  echo "START $name $(date --iso-8601=seconds)" >> "$QLOG"
  tmux new-session -d -s "scnn-$name" "cd $REPO && env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES HSA_OVERRIDE_GFX_VERSION=10.3.0 OMP_NUM_THREADS=2 $PY $ROOT/source/scnn.py --regime $name --out $ROOT/runs/$name > $ROOT/runs/$name/train.log 2>&1; ec=\$?; echo EXIT=\$ec >> $ROOT/runs/$name/train.log; if [ \$ec -eq 0 ]; then python3 kit/check_submission.py $ROOT/runs/$name/validation_predictions.csv --split validation > $ROOT/runs/$name/check_validation.txt 2>&1; echo CHECK_VAL=\$? >> $ROOT/runs/$name/check_validation.txt; python3 kit/check_submission.py $ROOT/runs/$name/test_submission.csv --split test > $ROOT/runs/$name/check_test.txt 2>&1; echo CHECK_TEST=\$? >> $ROOT/runs/$name/check_test.txt; fi; echo DONE $name EXIT=\$ec >> $QLOG"
}

n_running() {
  local n=0 name
  for name in base episode_balanced window32 query_upweight min_obs; do
    if tmux has-session -t "scnn-$name" 2>/dev/null; then
      if ! grep -q '^EXIT=' "$ROOT/runs/$name/train.log" 2>/dev/null; then
        n=$((n + 1))
      fi
    fi
  done
  echo "$n"
}

for name in base episode_balanced window32 query_upweight min_obs; do
  while [ "$(n_running)" -ge 2 ]; do
    sleep 5
  done
  start_one "$name"
  sleep 1
done

while [ "$(n_running)" -gt 0 ]; do
  sleep 5
done
echo "ALL_DONE $(date --iso-8601=seconds)" >> "$QLOG"
