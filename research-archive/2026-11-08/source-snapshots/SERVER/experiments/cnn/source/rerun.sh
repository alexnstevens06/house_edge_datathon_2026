#!/bin/bash
cd ~/Documents/code/datathon_2026/house-edge; R=experiments/cnn/runs; P=.venv/bin/python
$P experiments/cnn/source/cnn.py --h 128 --layers 4 --W 64 --drop 0.2 --epochs 30 --lr 1e-3 --seed 0 --no_signal --out $R/g_nosignal > $R/g_nosignal.log 2>&1
for s in a b; do $P experiments/cnn/source/predict_cpu.py $R/g_nosignal $s >> $R/g_nosignal_cpu.log 2>&1; done
$P experiments/cnn/source/predict_cpu.py $R/g_h128_L4_W64_d02_e30 a --audit >> $R/g_contaminated_cpu.log 2>&1
CNN_RUN=$R/g_nosignal $P experiments/cnn/tests/test_cnn.py > $R/g_nosignal_tests.log 2>&1
$P experiments/cnn/tests/test_signal.py >> $R/g_nosignal_tests.log 2>&1
for f in validation_cpu_a validation_predictions; do $P kit/check_submission.py --split validation $R/g_nosignal/$f.csv >> $R/g_nosignal_checker.log 2>&1; done
for f in test_cpu_a test_predictions; do $P kit/check_submission.py --split test $R/g_nosignal/$f.csv >> $R/g_nosignal_checker.log 2>&1; done
cmp $R/g_nosignal/validation_cpu_a.csv $R/g_nosignal/validation_cpu_b.csv && cmp $R/g_nosignal/test_cpu_a.csv $R/g_nosignal/test_cpu_b.csv && echo CPU_REPEAT_BITWISE_IDENTICAL >> $R/g_nosignal_cpu.log
echo RERUN_DONE >> $R/g_nosignal.log
