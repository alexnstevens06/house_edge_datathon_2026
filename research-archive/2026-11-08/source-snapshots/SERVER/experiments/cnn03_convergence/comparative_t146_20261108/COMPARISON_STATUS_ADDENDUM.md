# Comparison status addendum

2026-11-08 02:44 CST. Metadata only. This note supersedes the stale path-blocker reason in `COMPARISON_READY.md`, in `results/comparison.json` (rows stored as `not_run_preflight_blocked`), and in the historical path wording of `independent_review_20261108/t146_review/FROZEN_REVIEW.md`.

Those files were not rewritten. The comparison runner, `PINS.json`, `SOURCE_SHA256.txt`, `source/ranking.py`, `source/compare_runner.py`, the numerical table, and `results/comparison.json` stay as frozen at 2026-11-08 02:24 CST. `package_written` in that JSON remains false. This note does not itself claim a delivery pass.

## Current Tower reason

The 02:13 CST pin-path failure is historical. `lowrate_20261108/TOWER_PRIOR_PREFLIGHT_BLOCKER.md` (SHA-256 `0f79395894c605295c398ac121fe6bb33fa98a37ad58f3496850206813b94444`) records `FileNotFoundError` on the canonical FP64 `model.py` before any GPU step, because `assert_input_pins` hashes manifest paths that were not on Tower yet.

Parent 02:18 clarified that the immutable inputs could be staged at those exact paths. The coordinator copied the true pinned `model.py` and `data.py` and the six old checkpoint files, exclusive creation, no overwrites, no aliases. `lowrate_20261108/TOWER_INPUT_STAGING_VERIFIED.json` (SHA-256 `b90ee0c8a21d9d4d921dad0c76071a8085b2f136d900acdea2f82e3463147775`, 2026-11-08T02:19:46-06:00) records that all 14 inputs and 10 source hashes passed. Source and guards were unchanged. The copied checkpoints are the original parents, not Tower training runs. That path failure is resolved.

The same Tower context 91377 then ran the unit tests and the FP64 and FP32 probes. Units passed (18567 windows, 730 steps). The FP64 probe passed at float64. The native FP32 batch-128 forward and backward, with no optimizer step, ran and then failed the frozen RSS guard: `2174996` KiB versus cap `2097152` KiB, EXIT 1, gpu 1144.0 MB, loss 1.593103. The author stopped. There is no `runs/` directory, no training epoch, no training PID, and no Tower checkpoint. The definitive record is `lowrate_20261108/TOWER_FINAL_RSS_GUARD.md` (SHA-256 `a42361ef2f618ab6a6b89a43246ebf58b4dd7935846c1f55cb10ae85dc193acc`).

Parent 02:28 closed the 5e-5 arm. No retry, no cap change, and no recipe change. No new training is authorized under another name. The finished SERVER 2e-5 pair (730 steps, two jobs) is not rerun.

## Ranking left as completed

Six Tower low-rate rows are NOT RUN. They have no score and are not ranked as worse. Only 2 of 4 low-rate jobs completed: the SERVER 2e-5 primary and control. The completed-row ranking is unchanged. The selected row is `ensemble_equal_s0_s2`, tune R `0.28468175920919014`, current best `0.30980333573689856`, improvement `0.02512157652770841` (above 1e-6). This note does not claim that row beats an unrun checkpoint.

The selected ensemble is the original `curve_C_lr003_s0` and `curve_C_lr003_s2` parents, equal probability weights 0.5 and 0.5, in that order. It uses no new low-rate weights and no shared control. Blends stay not cleared. Low-rate source concerns (replay, host binding, forward count, TF32) stay concerns. Frozen studies were not edited.

The historical shared-control scalar was FP32. Exact old FP32 reproduces `0.46211193532631023`. Canonical FP64 is `0.46211190598948326`. No probability correction was fit.

## Frozen comparison hashes at this writing

| File | SHA-256 |
| --- | --- |
| `SOURCE_SHA256.txt` | `c083ca6c87472c675cf8d5366364d00d9775e9ed03452c8c8437bfbae798b5f9` |
| `PINS.json` | `aa1e515e122322caec69b0a13ae54324b8c5b8e04619ca14d388e6ac49ab0a94` |
| `COMPARISON_READY.md` | `a1fbf4b103876c50ce5a611dccbb510bcb913fb11d19eee54dfecf60bb9cd7d5` |
| `results/comparison.json` | `9341e9553020e55c8ed863d20b2c31b10be7e45c4d3ffeb4d7a7ceabf991a005` |
| `source/ranking.py` | `a396b4b3299decc4819948f91658fd8583070fc961dd992d96dee5c7ab8598d5` |
| `source/compare_runner.py` | `bde8e15adf7ba1691c652d1b2da59b0f910133fc3c85b47ffb81541d87db0202` |
