# BLOCKER: server GPU job was not started

TRAIN cutoff labels exist (96 episodes, 768/768 queries labeled in `SPLIT_IDS.json`). This is not a data blocker.

At 2026-11-07 19:49:57 CST the harness refused the launch command. The command rsyncs `experiments/cnn_experts/soft_moe/` to `server` (same repo path) and starts `launch_server.sh 0` under `setsid`/`nohup`. Refusal text: another-machine shells are hard-wait and were not executed. No retry, no tower GPU, no CPU substitute.

Nothing was fit. There is no tune R, no promotion decision, and no seed 1 or 2.

Ready command, unchanged, once SSH is allowed:

```bash
cd /home/alexn/Documents/code/datathon_2026/house-edge
rsync -az --exclude '__pycache__' --exclude 'runs' --exclude 'session.log' \
  experiments/cnn_experts/soft_moe/ \
  server:/home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_experts/soft_moe/
ssh -o BatchMode=yes server 'mkdir -p /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_experts/soft_moe/runs/seed0 && setsid nohup bash /home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_experts/soft_moe/launch_server.sh 0 >/home/alexn/Documents/code/datathon_2026/house-edge/experiments/cnn_experts/soft_moe/runs/seed0/nohup.out 2>&1 </dev/null & echo STARTED:$!'
```

`launch_server.sh` checks free GPU memory (abort under 2500 MiB), caps the process at 2 GiB, uses 2 CPU threads, and trains the matched control then the soft model for seed 0.
