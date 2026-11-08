# Gap-closing changes after REVIEW_OPENCODE.md (2026-11-07)

Scope: only `experiments/cnn/{source,tests}` (+ this report). Model architecture, forward math and
training behaviour are unchanged — `masked_loss` is a verbatim factorisation of main's inline loss,
no parameters added/removed, existing checkpoints load and give identical predictions. No run
artifacts were deleted or modified.

## Changes

### 1. New gap tests — `experiments/cnn/tests/test_gaps.py` (plain `__main__` runner, no pytest)

- **1a End-to-end model causality on real gappy data** — `test_model_causality_on_real_gappy_episodes` (test_gaps.py:33).
  Uses `house_edge.data.episode_conditions` (data.py:107) to select 3 real `short_history`
  validation episodes (~68-75% MISSING ticks, asserted), a freshly-initialised
  `CNN(32, 2, 3, 0.0, 64, use_signal=False)` in eval mode under `torch.manual_seed(0)`
  (`_fresh_model`, test_gaps.py:19), and pseudo-cutoffs t ∈ {16, 64, cutoff//2, cutoff}. For each
  machine slot s (0..7) and each mutation kind (random payout fills; MISSING↔observed toggles) on
  all ticks >= t, the window rebuilt via `cnn.window` and the full model output at t are asserted
  bit-identical (`torch.equal`); windows are asserted to contain MISSING gaps; a sanity check
  confirms mutating the last in-window tick *is* detected (non-vacuous). 192 mutation forwards.
  Closes Q3 gaps 1 and 3.
- **1b Cutoff-label mutation** — `test_cutoff_label_mutation_never_enters_train_features` (test_gaps.py:69).
  On a 3-episode train subset (incl. one `short_history` episode), all `query_labels` values are
  changed to `(v+1)%5`; combined `build_train` X,T,TY,SG are bit-identical and Y changes only on
  rows with t ∈ {episode cutoffs}; per-episode builds assert Y is identical for all t < cutoff and
  changes exactly at the queried machine slots at t == cutoff. Closes Q3 gap 2.
- **1c Loss masking** — `test_loss_masking_matches_main_and_missing_contributes_zero` (test_gaps.py:97).
  `cnn.masked_loss` (cnn.py:91, used by main at cnn.py:129) is asserted equal (bit-exact) to the
  former inline `yl = y.long(); msk = yl >= 0; loss = F.cross_entropy(lg.float()[msk], yl[msk])`
  (old cnn.py:122-123, replicated in the test) and to two independent labelled-subset
  computations (nonzero indexing; compacted labelled-only tensors). Gradients: masked slots get
  exactly zero grad; adding +100 to masked-slot logits leaves the loss value and all gradients
  bit-identical. Also asserts main's source actually calls `masked_loss(lg, y)`.
  Closes Q3 gap 4.
- **1d No-signal metadata mutation** — `test_no_signal_build_query_ignores_query_signal`
  (test_gaps.py:121). With `USE_SIGNAL=False`, flipping every query signal in validation (all 96
  episodes) and a test-split subset leaves all `build_query` tensors bit-identical, SG all-zero,
  and the full no-signal model output bit-identical. Extends test_signal.py to the
  validation/test-style build path.

### 2. Clean-run selection

- New `experiments/cnn/source/clean_run.py:4` — `CLEAN_RUN = "g_nosignal"` single source of truth.
- `experiments/cnn/tests/test_cnn.py:4,7` — default `CNN_RUN` is now `runs/g_nosignal`
  (was the nonexistent `runs/best`); `CNN_RUN` env still overrides (rerun.sh:6 unchanged).
- Skip-not-error: `Skip` exception (test_cnn.py:9) raised by `load()` when the checkpoint is
  absent (test_cnn.py:14) or `test_predictions.csv` missing (test_cnn.py:49); the runner prints
  `SKIP <name> <reason>` and exits 0 unless a real failure occurs (test_cnn.py:61-67).

### 3. Contamination guard — `experiments/cnn/source/predict_cpu.py` (rewritten, same export math)

- `contamination_reason(run)` (predict_cpu.py:14): refuses (a) any run dir containing
  `CONTAMINATED.txt` / any `*CONTAMINATED*` marker (covers the `.CONTAMINATED.marker` naming),
  (b) a checkpoint config without `no_signal=True` (signal-leak mode, REVIEW_OPENCODE.md Q1);
  also missing run dir / checkpoint. Reads only the checkpoint config — never split data.
- `main(argv)` (predict_cpu.py:27): guard runs before any `load_split` (predict_cpu.py:33);
  refusal prints `REFUSED ...` to stderr and exits 2 (predict_cpu.py:35-36). With `--audit`
  (predict_cpu.py:29) the export proceeds and every output file is written with an
  `AUDIT_INVALID_` prefix (predict_cpu.py:37, 44-46) — clean runs keep normal names, so the
  positional usage `predict_cpu.py RUN_DIR SUFFIX` is unchanged for the clean run.
- `experiments/cnn/source/rerun.sh:5` — the contaminated re-export of `g_h128_L4_W64_d02_e30`
  now passes `--audit`; its outputs in that run dir will be `AUDIT_INVALID_{validation,test}_cpu_a.csv`
  and `AUDIT_INVALID_metrics_cpu_a.json`. Log destination and all other lines unchanged; no
  artifacts deleted.
- Guard tests (test_gaps.py:134-190): in-process refusals with `load_split`/
  `load_official_validation`/`score_predictions`/`write_submission_csv` monkeypatched to raise,
  proving refusal happens **before loading data** (test_gaps.py:152, 167); subprocess refusal
  (exit 2, `REFUSED`/`CONTAMINATED` in stderr); clean positional export produces
  `validation_cpu_<suf>.csv`/`test_cpu_<suf>.csv`/`metrics_cpu_<suf>.json` with no prefix;
  `--audit` on both a CONTAMINATED.txt dir and a signal-mode checkpoint writes only
  `AUDIT_INVALID_`-prefixed outputs (test_gaps.py:178).

## Test results (this machine, `.venv/bin/python`, CPU, torch 2.9.1)

| Command | Result |
|---|---|
| `python experiments/cnn/tests/test_gaps.py` | 7 PASS / 0 FAIL (22.5 s) |
| `python experiments/cnn/tests/test_cnn.py` | 4 SKIP (no `runs/g_nosignal` checkpoint here) / 0 FAIL, exit 0 (3.6 s) |
| `python experiments/cnn/tests/test_signal.py` | 3 PASS / 0 FAIL |
| `python experiments/cnn/source/predict_cpu.py <tmp-clean-run> cli` | exit 0, normal (unprefixed) output names — positional usage intact |
| `bash -n experiments/cnn/source/rerun.sh` | syntax OK |

Checkpoint-dependent test_cnn.py tests were not exercisable here (training machine only); they
SKIP cleanly and will run via rerun.sh:6-7 where `runs/g_nosignal` exists.

## Invariant notes

- `cnn.py`: only addition is `masked_loss` (cnn.py:91-96) + its call in main (cnn.py:129); the
  expression is byte-for-byte the previous inline loss, so training math, seeds, schedules,
  parameters (state-dict keys) and checkpoint compatibility are unchanged.
- `predict_cpu.py`: export/scoring math identical to the previous version; only the guard, the
  optional `AUDIT_INVALID_` prefix and the `main(argv)` restructuring (enables in-process guard
  testing) were added.
