# from repo root; tmux sessions linear_lr, linear_lr2, linear_final
OMP_NUM_THREADS=4 .venv/bin/python experiments/linear/source/run.py . experiments/linear/runs/final 3e-3
OMP_NUM_THREADS=2 .venv/bin/python experiments/linear/tests/test_linear.py experiments/linear/runs/final .
(cd kit && ../.venv/bin/python check_submission.py --split test|validation <csv>)
