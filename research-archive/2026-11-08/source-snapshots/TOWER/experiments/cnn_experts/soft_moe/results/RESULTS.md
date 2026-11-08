# soft_moe results

Verdict: **blocked before training**. No tune R was computed. Promotion was not applied. Seeds 1 and 2 were not launched.

## What finished

- Plan frozen before metrics. `PLAN.md` sha256 `5a5b3cc22ea50f1a71e990bbb01a3b05b89bcb99cde0c634af7023eb8dc075c6`. `SPLIT_IDS.json` sha256 `898951437a1af7bf19dc5972831af873dedb193355a6fb995bf56f714680668c`.
- TRAIN labels cover every train query (768/768). History windows are specified with signal ABSENT.
- Pytest: 9 passed in 3.34s (`experiments/cnn_experts/soft_moe/tests/test_soft_moe.py`), CPU, before any metric.

## What did not run

Server GPU seed 0 (control + soft, 30 pretrain epochs + 10 cutoff fine-tune epochs). The harness blocked `ssh server`. No timings, train losses, parameter counts from a real run, or tune R. Confirm (seen split) is not reported.

Incumbent CNN03 tune R stays the comparison bar: seed 0 `0.33206398675779764`. A win would have been a strictly lower soft tune R on a full seed, then seeds 1 and 2.

See `BLOCKER.md`.
