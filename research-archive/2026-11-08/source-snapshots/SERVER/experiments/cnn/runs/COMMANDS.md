cd ~/Documents/code/datathon_2026/house-edge
bash experiments/cnn/source/sweep.sh; bash experiments/cnn/source/sweep2.sh   # run in tmux cnn_sup
.venv/bin/python experiments/cnn/source/select.py
CNN_RUN=experiments/cnn/runs/best .venv/bin/python experiments/cnn/tests/test_cnn.py
.venv/bin/python kit/check_submission.py --split test experiments/cnn/runs/best/test_predictions.csv
