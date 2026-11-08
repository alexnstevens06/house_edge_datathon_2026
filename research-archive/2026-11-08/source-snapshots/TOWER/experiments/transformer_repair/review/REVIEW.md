# Independent Read-Only Review — experiments/transformer_repair

Date: 2026-11-07. Reviewer: independent (did not author this code). No files outside
`experiments/transformer_repair/review/` were modified; no training, git, network, or installs.

## 0. Snapshot integrity

- `source/src/house_edge/models.py` sha256 =
  `726c421d88ed3dfd661b1287996a82f93cfcf5070269a433bb2f2592ea3a53f3` — **MATCHES** the required target hash.
- Discrepancy note: `source_hashes.txt` records `5258607e...` for `models.py`, i.e. the
  manifest is stale relative to the file (models.py was modified after the manifest was
  written; AUDIT.md documents a dead-code removal in models.py as the TASK B edit).
  Also present: `train.py.tower-orig` backup file. Does not affect the review target,
  which is the current file (hash verified above).

## 1. Causality — PASS

Every input token for every machine has tick < target tick, in both tokenizations,
including `HE_READOUT=own` extras.

- Dense window: `seg = episode.obs[lo:t]` (source/src/house_edge/windows.py:98) with
  `ticks = arange(lo, t)` (windows.py:101) → max input tick = t−1. Runtime assert
  windows.py:105 and model assert models.py:83 (`ticks < target_t`).
- Sparse window: tokens from `seg = obs[lo:t]` (windows.py:129), `tick = lo + pos`
  (windows.py:131) → tick < t. Asserts windows.py:149 and models.py:95.
- Query/cutoff path: history ticks are asserted `tick < cutoff` at load
  (data.py:81), and `obs` only ever contains ticks < cutoff, so a cutoff target
  (`Example(ep, cutoff, None)`, predict.py:32) can never see the target tick.
  `_materialize` reads labels at `obs[t]` only when `t < obs.shape[0]`
  (windows.py:74-77); since `obs.shape[0] <= cutoff`, cutoff examples get all-MISSING
  labels (never used as features).
- `HE_READOUT=own` extras (`_own_extras`, models.py:58-79): computed purely from the
  INPUT token tensors (symbols/ticks/slots/valid) passed to forward; it is invoked at
  models.py:88 / models.py:99, i.e. AFTER the causality asserts at models.py:83 /
  models.py:95 have validated those same tensors. `last_tick < target_t` ⟹ lag ≥ 1;
  never-observed machines are zeroed via `has` (models.py:79). Dense `valid` =
  `symbols > 0` (models.py:88) correctly excludes both MISSING cells and front-padding
  rows; sparse `valid` = `~pad_mask` (models.py:99).
- Test evidence: `test_dense_future_tokens_excluded` / `test_sparse_future_tokens_excluded`
  / `test_forward_asserts_on_leak` (tests/test_house_edge.py:72-99); `test_causality_asserts`
  (readout/test_readout.py:229); `test_own_causality_future_excluded` — future-tick
  perturbation leaves outputs bit-identical under own mode, both archs
  (readout/test_readout.py:424-445).
- Minor non-leak wart: when t < W, dense front-padding emits duplicate "MISSING at
  tick 0" tokens (symbols/ticks zero-initialized, windows.py:88-89) alongside the real
  tick-0 row. Deterministic, tick 0 < t, no information leak; untested cosmetic quirk.

## 2. Payout order 0,1,2,5,20 == class idx == p_0..p_20 — PASS

- `PAYOUTS = (0, 1, 2, 5, 20)` and `PAYOUT_TO_IDX` (data.py:15-17): class idx i ↔
  payout value PAYOUTS[i].
- Labels: history payouts → `PAYOUT_TO_IDX` (data.py:67); labels.csv outcomes →
  `PAYOUT_TO_IDX` (data.py:94). Head is 5-way over `N_PAYOUTS` (models.py:45);
  `predict_probs`/predict.py softmax in class-idx order (models.py:114, predict.py:39).
- Dense symbol = payout idx + 1, MISSING→0 (windows.py:100); sparse symbol = payout
  idx, PAD=5 (windows.py:132,138; models.py:19,30). Verified by
  `test_dense_missing_is_not_payout_zero` (tests/test_house_edge.py:122) and
  `test_layout_slots_types_queries_and_output` (readout/test_readout.py:185).
- Submission columns: kit `slot_forecasting/schema.py:8-9` — same `PAYOUTS=(0,1,2,5,20)`
  → `PROBABILITY_COLUMNS = (p_0,p_1,p_2,p_5,p_20)`, written verbatim by
  `write_submission_csv` (scoring.py:64). Kit `TARGET_COLUMNS` use the same order, so
  predicted column i, target column i, and class idx i all align end-to-end.

## 3. Masks — PASS

- `label_mask` excludes unobserved targets: mask = `labels >= 0` (windows.py:78);
  `masked_ce_loss` multiplies per-element CE by the mask (models.py:120-122), clamps
  labels only to keep CE finite for masked entries, and raises on an all-masked batch
  (models.py:124). Verified by `test_loss_ignores_unobserved_targets` (perturbing
  unobserved labels leaves loss unchanged) and `test_all_masked_raises`
  (tests/test_house_edge.py:166-190).
- Sparse pad masking: `pad_mask` True = pad (windows.py:142,148); passed as
  `src_key_padding_mask` extended with `new_zeros(B, N_MACHINES)` for the readout
  positions (models.py:101-102) — **readout tokens are never masked**; even a
  fully-padded input row retains 8 unmasked readout positions, so no NaN/collapse.
  Verified structurally by `test_sparse_queries_and_output`
  (readout/test_readout.py:222-227).
- Dense has no padding mask by design (all W×8 cells are tokens, MISSING is a symbol);
  output rows are read only from the last 8 (readout) positions: models.py:91,103.

## 4. Signal never used as a feature — PASS

`grep -rn signal` over `experiments/transformer_repair` (all .py):
- source/src/house_edge/scoring.py:26 — `"signal"` appears only as a column NAME in the
  tuple of columns read from the organizer's queries.csv; the dict built maps
  `query_id → episode_id` only. Never enters tokens, embeddings, or features.
- tests/test_house_edge.py:45, readout/test_readout.py:59,318 — `"signal": "0"` in
  synthetic query-row dicts (schema mimicry for fixtures); never read by the model.
- No occurrence anywhere in models.py/windows.py/data.py/predict.py/train.py feature
  paths. (The organizer kit's own baseline.py uses signal, but that is kit code, not
  this model.)

## 5. Output row i == slot i == predict.py query mapping — PASS

- Dense: token j = w·8+s carries slot s (`slots_tiled`, windows.py:107); readout
  queries are built from the first 8 tokens' slots = 0..7 (models.py:86, verified
  readout/test_readout.py:177); logits = `head(h[:, -N_MACHINES:])` (models.py:91) →
  row i ↔ readout query of slot i.
- Sparse: `slots_q = arange(N_MACHINES)` (windows.py:150), q at models.py:97, rows
  from last 8 positions (models.py:103) → row i ↔ slot i.
- predict.py: one example per episode at its cutoff (predict.py:28-33); for each query,
  `slot = episode.slot_of(machine_id)` and `predictions[query_id] = probs[i, slot]`
  (predict.py:43-44). `machine_ids` are the sorted machine ids (data.py:73) and `obs`
  columns / `slot_of` use the same enumeration (data.py:84-86, data.py:44), so slot
  indices are consistent between data, tokens, and readout. Chunk indexing `i` is
  aligned because `examples` and `groups` are appended together (predict.py:28-33).
- Test evidence: manual-reconstruction equality tests (readout/test_readout.py:187-195,
  218-227) and the train-free stub mapping test `TestPredictQueryAlignment`
  (readout/test_readout.py:295-333) which pins query_id → slot → one-hot payout for
  both archs through the real `predict_split`.

## 6. Fixture validity — PASS WITH CAVEATS (one loosened near-chance threshold)

Chance level for all toy rules is 0.20 (uniform 5 classes). Fresh-episode evaluation
(2 unseen episodes × 18 targets × 8 machines = 288 predictions) is the right design —
it blocks phase-table memorization (documented in AUDIT.md; initially shared-episode
fixture was fixed to fresh).

| Test (readout/test_readout.py) | Threshold | Verdict |
|---|---|---|
| test_layout_slots_types_queries_and_output (:164) | exact equalities; manual reconstruction == forward | Meaningful, strong |
| test_sparse_queries_and_output (:200) | exact token list; reconstruction == forward; no pad | Meaningful, strong |
| test_causality_asserts (:229) | AssertionError on tick==target | Meaningful |
| TestRowSlotBinding (:244) | acc>0.9 sanity; delta.argmax==i; delta[i]>0.5; ≥max−1e-9 | Meaningful (fresh episode; TV-distance 0.5 is a strong effect) |
| TestPredictQueryAlignment (:298) | exact one-hot per query, both archs, train-free stub | Meaningful; partial — see caveat (a) |
| test_fit_self_rule (:339) | loss<0.1; acc>0.9 (chance 0.2) | Meaningful, wide margin (measured 1.00 per AUDIT.md) |
| test_fit_cross_rule (:346) | loss<0.1; acc>0.4 dense, >0.22 sparse | **Loosened** — see caveat (b) |
| test_default_query_bit_for_bit (:368) | exact equality vs golden_query.npz (abs & lag, both archs) | Meaningful regression pin (change-detector, not independent correctness proof) |
| test_param_counts_equal_across_modes (:385) | query == own param count | Meaningful (own adds no parameters) |
| test_invalid_mode_rejected (:395) | AssertionError on bogus mode | Meaningful |
| test_own_extras_values_and_no_observation (:401) | hand-computed exact extras; zero for unobserved slot | Meaningful, strong unit test |
| test_own_causality_future_excluded (:424) | future perturbation ⇒ identical outputs, both archs | Meaningful |

No test is fully vacuous. Caveats:

- (a) TestPredictQueryAlignment: the comment "the pair (g1, g2) is injective" (:299)
  overstates coverage. The map m→(g1(m),g2(m)) is injective, but each query's expected
  value uses only its own episode's g, and within episode 0 the pairs {0,5},{1,6},{2,7}
  share constant payouts under g1 (within ep1: {1,5},{2,6},{3,7}). A row-permutation
  that swaps only payout-identical slots would pass undetected. The test still catches
  off-by-one row shifts, episode mix-ups, and most slot confusions for both archs —
  meaningful, just not exhaustive over all 8 slots.
- (b) test_fit_cross_rule: the >0.9 bar used for the self rule was dropped to floors of
  0.4 (dense) / 0.22 (sparse). I re-measured the fixture directly (HE_POS=lag,
  HE_READOUT=query, same seeds/steps as the suite):
  `cross dense: loss=0.0000 acc=0.7118`, `cross sparse: loss=0.0000 acc=0.3160`
  (n=288 fresh predictions). Dense floor 0.4 is ≈8.5 SE above chance (SE≈0.024) — a
  genuine above-chance-with-margin bar. **Sparse floor 0.22 is only ≈0.85 SE above the
  0.20 chance level: a chance-level sparse model would pass ~20% of the time.** The
  code comment (:347-351) transparently admits the loosening ("measured ~0.32 … the
  >0.9 bar needs larger capacity/budget"), but AUDIT.md's claim that thresholds were
  "set above chance with margin" is inaccurate for the sparse floor. Effectively the
  sparse cross-rule assertion is near-vacuous: it verifies the fit runs and train loss
  < 0.1 (train loss 0.0 reflects memorized train phases), not that sparse fresh-episode
  cross-machine binding is meaningfully learned at this budget. The cross-machine
  attention claim for sparse rests on the dense result plus the structural
  reconstruction tests, not on the sparse accuracy floor.
- Minor: tests/test_house_edge.py:200 `assertLess(|n_dense−n_sparse|/n_dense, 0.10)`
  is trivially implied by the `assertEqual` on the previous line — a vacuous second
  assertion (harmless redundancy, not a fixture problem).

## Required command run (recorded verbatim)

```
$ HE_POS=lag HE_READOUT=own PYTHONPATH=experiments/transformer_repair/source/src:kit \
    HSA_OVERRIDE_GFX_VERSION=10.3.0 timeout 95 .venv/bin/python -m unittest discover \
    -s experiments/transformer_repair/tests -v
test_dense_future_tokens_excluded (test_house_edge.TestCausality.test_dense_future_tokens_excluded) ... ok
test_forward_asserts_on_leak (test_house_edge.TestCausality.test_forward_asserts_on_leak) ... ok
test_sparse_future_tokens_excluded (test_house_edge.TestCausality.test_sparse_future_tokens_excluded) ... ok
test_all_masked_raises (test_house_edge.TestModel.test_all_masked_raises) ... ok
test_deterministic_eval (test_house_edge.TestModel.test_deterministic_eval) ... ok
test_loss_ignores_unobserved_targets (test_house_edge.TestModel.test_loss_ignores_unobserved_targets) ... ok
test_matched_parameter_counts (test_house_edge.TestModel.test_matched_parameter_counts) ... ok
test_output_shape_and_probabilities (test_house_edge.TestModel.test_output_shape_and_probabilities) ... ok
test_query_id_alignment (test_house_edge.TestPredictionAlignment.test_query_id_alignment) ... ok
test_dense_missing_is_not_payout_zero (test_house_edge.TestTokenization.test_dense_missing_is_not_payout_zero) ... ok
test_dense_ticks_are_absolute (test_house_edge.TestTokenization.test_dense_ticks_are_absolute) ... ok
test_sparse_preserves_original_ticks (test_house_edge.TestTokenization.test_sparse_preserves_original_ticks) ... ok

----------------------------------------------------------------------
Ran 12 tests in 0.354s

OK
```

Note: this discovery root covers only `tests/test_house_edge.py`. The readout fixtures
(readout/test_readout.py, 12 tests) are a separate suite; per readout/tests.txt both
the query and own mode runs of that suite passed (12 tests, ~63 s each), and I
independently re-measured the cross-rule fixture accuracies quoted in caveat (b).
The required run exercises the model under HE_READOUT=own end-to-end (all 12 pass),
which is the mode where causality risks were highest.

## Verdict

| # | Check | Result |
|---|---|---|
| 0 | models.py hash = 726c421d…a53f3 | PASS (manifest source_hashes.txt is stale — reported) |
| 1 | Causality (dense, sparse, own extras) | PASS |
| 2 | Payout order == class idx == p_0..p_20 | PASS |
| 3 | label_mask / sparse pad mask / readout never masked | PASS |
| 4 | signal never a feature | PASS |
| 5 | Output row i == slot i == predict.py mapping | PASS |
| 6 | Fixture validity | PASS WITH CAVEATS — no vacuous tests, but sparse cross-rule floor (0.22 vs 0.20 chance) was loosened to near-chance and AUDIT.md's "above chance with margin" claim overstates it; dense floor 0.4 is a genuine margin bar |

**OVERALL: PASS** (with the fixture caveat (b) and the stale source_hashes.txt manifest
flagged for the record).
