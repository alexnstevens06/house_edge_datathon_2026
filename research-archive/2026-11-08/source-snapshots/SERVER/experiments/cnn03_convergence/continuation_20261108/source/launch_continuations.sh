#!/bin/bash
# Twelve frozen warm starts. Two at a time on the single 5080. No extra seeds.
set -u
cd /home/alexn/Documents/code/datathon_2026/house-edge
PY=.venv/bin/python
STUDY=experiments/cnn03_convergence/continuation_20261108
mkdir -p "$STUDY/runs"
STATUS=$STUDY/STATUS.md
stamp() { TZ=America/Chicago date '+%Y-%m-%d %H:%M:%S %Z'; }

foreign_gpu() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF && $1+0>0 {print $1}'
}

ours=""
note_pid() { ours="$ours $1"; }

gpu_clear_of_others() {
  local pid
  for pid in $(foreign_gpu); do
    case " $ours " in
      *" $pid "*) ;;
      *) echo "REFUSE foreign GPU pid $pid"; return 1 ;;
    esac
  done
  return 0
}

LAST_PID=
run_one() {
  local arm=$1 extra=$2
  local name=${arm}_plus${extra}
  local out=$STUDY/runs/$name
  mkdir -p "$out"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PY" -u "$STUDY/source/continue_train.py" \
      --arm "$arm" --extra-epochs "$extra" --out "$out" \
      > "$out/train.log" 2>&1 &
  LAST_PID=$!
  echo "$LAST_PID" > "$out/pid"
  note_pid "$LAST_PID"
  echo "$(stamp) start $name pid=$LAST_PID" | tee -a "$STATUS"
}

wave() {
  gpu_clear_of_others || return 1
  run_one "$1" "$2"
  local first=$LAST_PID
  run_one "$3" "$4"
  local second=$LAST_PID
  local rc=0
  wait "$first" || rc=1
  wait "$second" || rc=1
  return "$rc"
}

{
  echo "# STATUS"
  echo
  echo "$(stamp) launch script pid=$$ gpu_before=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
} > "$STATUS"

# CPU FP64 of the six parents and the incumbent. Kept off the GPU.
CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" -u "$STUDY/source/score_fp64.py" > "$STUDY/runs/fp64_original_tune.log" 2>&1 &
FP64_PID=$!
echo "$(stamp) start fp64 originals pid=$FP64_PID" | tee -a "$STATUS"

wave curve_C_lr003_s0 20 curve_control_s0 20 || exit 1
wave curve_C_lr003_s1 20 curve_control_s1 20 || exit 1
wave curve_C_lr003_s2 20 curve_control_s2 20 || exit 1
wave curve_C_lr003_s0 10 curve_control_s0 10 || exit 1
wave curve_C_lr003_s1 10 curve_control_s1 10 || exit 1
wave curve_C_lr003_s2 10 curve_control_s2 10 || exit 1

wait "$FP64_PID" || exit 1
echo "$(stamp) fp64 originals done" | tee -a "$STATUS"

CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" -u "$STUDY/source/blend_tune.py" > "$STUDY/runs/blends_tune.log" 2>&1 || exit 1
echo "$(stamp) blends done" | tee -a "$STATUS"

CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" -u "$STUDY/source/score_saved.py" > "$STUDY/runs/fp64_continuation_tune.log" 2>&1 || exit 1
echo "$(stamp) continuation fp64 done" | tee -a "$STATUS"
echo "$(stamp) ALL_BOUNDED_JOBS_DONE" | tee -a "$STATUS"
