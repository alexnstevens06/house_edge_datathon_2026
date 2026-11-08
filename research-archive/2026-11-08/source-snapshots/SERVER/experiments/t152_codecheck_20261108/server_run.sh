#!/bin/bash
# t152 fresh check on SERVER as alexn. Writes only under experiments/t152_codecheck_20261108.
set -u
R=/home/alexn/Documents/code/datathon_2026/house-edge
SRC=$R/experiments/cnn03_convergence/candidate_t146_ensemble_s0_s2_20261108
T=$R/experiments/t152_codecheck_20261108
C=$T/candidate_t146_ensemble_s0_s2_20261108
PY=$R/experiments/cnn_runtime/py313_verify_1936/env/bin/python
ZIP=$SRC/ensemble_equal_s0_s2_fp64_candidate.zip
test -e $T && { echo "T exists, refusing"; exit 3; }
mkdir -p $C/refs $T/logs $T/docker_out
cp /tmp/t152_stage/*.py /tmp/t152_stage/*.sh $T/ 2>/dev/null
echo "ZIP $(sha256sum $ZIP)"
cp $SRC/fp64_candidate_harness.py $SRC/score_tune_only.py $C/
cp $SRC/refs/ensemble_equal_s0_s2.npz $C/refs/
sha256sum $C/*.py $C/refs/*.npz
cp $ZIP $C/code.zip; $PY -c "import zipfile;zipfile.ZipFile('$C/code.zip').extractall('$C/unpacked')"
ls -R $C/unpacked | head -20
cd $R
echo "=== HARNESS (host, $($PY -V), taskset 0,1) ==="
/usr/bin/time -v taskset -c 0,1 env OMP_NUM_THREADS=2 $PY $C/fp64_candidate_harness.py > $T/logs/harness.log 2> $T/logs/harness.time; echo "HARNESS_EXIT=$?"
tail -25 $T/logs/harness.log; grep -E 'Maximum resident|Elapsed|Exit status' $T/logs/harness.time
echo "=== EXTRA ==="
taskset -c 0,1 $PY $T/extra_checks.py $C/unpacked $R/kit/public $T/extra_work > $T/logs/extra.log 2>&1; echo "EXTRA_EXIT=$?"; tail -1 $T/logs/extra.log
echo "=== DOCKER 2GiB network none ==="
for spec in t152-test-a:test t152-test-b:test t152-val:val; do
 n=${spec%%:*}; s=${spec##*:}
 docker run --name $n --pull=never --network=none --read-only --cpus=2 --memory=2147483648 --memory-swap=2147483648 \
  --tmpfs /tmp:rw,size=128m --tmpfs /output:rw,size=32m --user 1000:1000 --cap-drop=ALL --security-opt no-new-privileges \
  -v "$R/experiments/cnn_runtime/py313_verify_1936:$R/experiments/cnn_runtime/py313_verify_1936:ro" \
  -v "$C/code.zip:/pkg/code.zip:ro" -v "$R/kit/public:/data:ro" \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 ubuntu:24.04 \
  /bin/bash -c "set -euo pipefail; echo NPROC=\$(nproc) >&2; $PY -c 'import sys,torch,zipfile;print(sys.version,torch.__version__,file=sys.stderr);zipfile.ZipFile(\"/pkg/code.zip\").extractall(\"/tmp/x\")'; cd /tmp/x; python_bin=$PY; \$python_bin predict.py --data /data --out /output/submission.csv --split $s >&2; cat /output/submission.csv" \
  > $T/docker_out/$n.csv 2> $T/docker_out/$n.err
 rc=$?; oom=$(docker inspect -f '{{.State.OOMKilled}} {{.State.ExitCode}} {{.HostConfig.Memory}} {{.HostConfig.NanoCpus}} {{.HostConfig.NetworkMode}}' $n); docker rm $n >/dev/null
 echo "$n rc=$rc inspect(oom exit mem nanocpus net)=$oom sha=$(sha256sum < $T/docker_out/$n.csv | cut -c1-64)"; tail -1 $T/docker_out/$n.err
done
$PY $T/csvcheck.py $T/docker_out/t152-test-a.csv $R/kit/public/test/queries.csv
cmp $T/docker_out/t152-test-a.csv $T/docker_out/t152-test-b.csv && echo DOCKER_TEST_REPEAT_BITWISE
$PY - <<P
import json;print("val_rows", sum(1 for _ in open("$T/docker_out/t152-val.csv"))-1)
P
echo "SRC_UNCHANGED $(sha256sum $ZIP | cut -c1-64)"
echo T152_SERVER_DONE
