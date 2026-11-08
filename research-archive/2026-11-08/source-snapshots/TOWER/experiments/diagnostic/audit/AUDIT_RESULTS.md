# Independent read-only audit — house-edge
Run 2026-11-07 17:04:27 with `/home/alexn/Documents/code/datathon_2026/house-edge/.venv/bin/python` (numpy available; script itself stdlib+numpy). Scope: splits, scoring, payout ordering, baselines. No files outside experiments/diagnostic/audit/ were modified.


## 1. Splits

Episodes by split: train=96, validation=96, test=192 (from queries.csv).
| Split | episodes | queries | history rows |
|---|---|---|---|
| train | 96 | 768 | 139944 |
| validation | 96 | 768 | 141392 |
| test | 192 | 1536 | 282664 |


**Checks:**
- PASS :: train/validation episode IDs disjoint :: overlap=0
- PASS :: train/test episode IDs disjoint :: overlap=0
- PASS :: validation/test episode IDs disjoint :: overlap=0
- PASS :: train: every query maps to exactly one episode/machine :: rows=768, unique qids=768, unique (ep,machine)=768, episodes with 8 machines=96/96
- PASS :: train: no duplicate (episode,machine,tick) in history :: history rows=139944, unique keys=139944
- PASS :: train: history payout values subset of (0, 1, 2, 5, 20) :: observed values=[0, 1, 2, 5, 20]
- PASS :: train: cutoff > all history ticks of that episode :: violations=0; min margin (cutoff-1-max_tick)=0, max=0
- PASS :: train: one cutoff per episode :: episodes with multiple cutoffs=0
- PASS :: validation: every query maps to exactly one episode/machine :: rows=768, unique qids=768, unique (ep,machine)=768, episodes with 8 machines=96/96
- PASS :: validation: no duplicate (episode,machine,tick) in history :: history rows=141392, unique keys=141392
- PASS :: validation: history payout values subset of (0, 1, 2, 5, 20) :: observed values=[0, 1, 2, 5, 20]
- PASS :: validation: cutoff > all history ticks of that episode :: violations=0; min margin (cutoff-1-max_tick)=0, max=0
- PASS :: validation: one cutoff per episode :: episodes with multiple cutoffs=0
- PASS :: test: every query maps to exactly one episode/machine :: rows=1536, unique qids=1536, unique (ep,machine)=1536, episodes with 8 machines=192/192
- PASS :: test: no duplicate (episode,machine,tick) in history :: history rows=282664, unique keys=282664
- PASS :: test: history payout values subset of (0, 1, 2, 5, 20) :: observed values=[0, 1, 2, 5, 20]
- PASS :: test: cutoff > all history ticks of that episode :: violations=0; min margin (cutoff-1-max_tick)=0, max=0
- PASS :: test: one cutoff per episode :: episodes with multiple cutoffs=0
- PASS :: validation: one condition per episode, 4 conditions x 24 episodes :: episodes=96, per-condition={'drift': 24, 'shared_change': 24, 'short_history': 24, 'stable': 24}
- PASS :: splits.json tune/confirm disjoint :: |tune|=48, |confirm|=48, overlap=0
- PASS :: splits.json covers all 96 validation episodes :: union=96, missing=0, extra=0
- PASS :: splits.json: 12 tune + 12 confirm episodes per condition :: tune/cond={'drift': 12, 'shared_change': 12, 'short_history': 12, 'stable': 12}, confirm/cond={'drift': 12, 'shared_change': 12, 'short_history': 12, 'stable': 12}
- PASS :: recomputed split (src/house_edge/split.py, seed 0) equals runs/splits.json :: tune equal=True, confirm equal=True, seed=0 vs 0, by_condition equal=True
- PASS :: training windows (sliding+query examples) reference only train episodes :: sliding examples=18471 (episodes=96), query examples=96 (episodes=96), referenced episodes=96; leaks into val/test=0
- PASS :: train query examples predict at cutoff with labels from train labels.csv :: examples=96, cutoff match=True, label match=True
- PASS :: sliding examples start at tick >= T_START(16) :: min t=16
- PASS :: collator causality: every input token tick < target tick (dense+sparse sample) :: sampled 1596 examples (1500 random sliding + all 96 query), causality violations=0
- PASS :: window tokens decode to true history payouts (0/1/2/5/20 order end-to-end) :: decode mismatches=0, coverage misses=0 (dense symbols 1..5 = payout idx+1; sparse 0..4 = payout idx; ~768000 dense tokens + sparse tokens checked)

## 2. Scorer

Reimplemented from RULES.md only (no import of src/house_edge/scoring.py or kit slot_forecasting.scoring for these numbers). Probabilities are used as submitted; each vector is divided by its own sum (sums already within 1e-6 of 1), mirroring the RULES requirement that probabilities sum to 1.
| Fixture (predictions) | R | points = 60/(1+R) | per-condition E[c] |
|---|---|---|---|
| reference.csv itself | 1 | 30.000000 | {'stable': 0.1962508, 'short_history': 0.17734145, 'drift': 0.18983393, 'shared_change': 0.20556265} |
| targets.csv probabilities | 6.66781328009e-34 | 60.000000 | {'stable': 0.0, 'short_history': 0.0, 'drift': 0.0, 'shared_change': 0.0} |
| uniform 0.2 | 2.20358738903 | 18.729004 | {'stable': 0.42004907, 'short_history': 0.4300244, 'drift': 0.41781528, 'shared_change': 0.42103151} |

| condition | E[c] (mine) | B[c] ref (mine) | ratio mine | ratio src | ratio kit-check | episodes |
|---|---|---|---|---|---|---|
| stable | 0.188052833 | 0.196250803 | 0.958227075 | 0.958227075 | 0.958227 | 24 |
| short_history | 0.177440962 | 0.177341448 | 1.000561145 | 1.000561145 | 1.000561 | 24 |
| drift | 0.179659867 | 0.189833931 | 0.946405453 | 0.946405453 | 0.946405 | 24 |
| shared_change | 0.180993556 | 0.205562650 | 0.880478804 | 0.880478804 | 0.880479 | 24 |

| condition | E[c] (mine) | B[c] ref (mine) | ratio mine | ratio src | ratio kit-check | episodes |
|---|---|---|---|---|---|---|
| stable | 0.194134895 | 0.196250803 | 0.989218348 | 0.989218348 | 0.989218 | 24 |
| short_history | 0.181986897 | 0.177341448 | 1.026194943 | 1.026194943 | 1.026195 | 24 |
| drift | 0.183287157 | 0.189833931 | 0.965513152 | 0.965513152 | 0.965513 | 24 |
| shared_change | 0.180821772 | 0.205562650 | 0.879643129 | 0.879643129 | 0.879643 | 24 |

| condition | E[c] (mine) | B[c] ref (mine) | ratio mine | ratio src | ratio kit-check | episodes |
|---|---|---|---|---|---|---|
| stable | 0.194723051 | 0.196250803 | 0.992215309 | 0.992215309 | 0.992215 | 24 |
| short_history | 0.184264050 | 0.177341448 | 1.039035444 | 1.039035444 | 1.039035 | 24 |
| drift | 0.187433883 | 0.189833931 | 0.987357119 | 0.987357119 | 0.987357 | 24 |
| shared_change | 0.177231250 | 0.205562650 | 0.862176326 | 0.862176326 | 0.862176 | 24 |

Comparison columns: mine vs src = |dR|/|dpoints|/|max d ratio|; mine vs kit-check = |dpoints|/|max d ratio| (kit prints 6 decimals).
| file | ids | R mine | pts mine | R src | pts src | pts kit-check | mine-vs-src | mine-vs-kit | tune R (mine vs src) | confirm R (mine vs src) |
|---|---|---|---|---|---|---|---|---|---|---|
| baselines/pooled_recent_validation.csv | 768 | 0.946418119 | 30.825854 | 0.946418119 | 30.825854 | 30.825854 | 0.00e+00/0.00e+00/0.00e+00 | 3.94e-07/4.53e-07 | 0.945528340 vs src 0.945528340 (d=0.00e+00) | 0.947603619 vs src 0.947603619 (d=0.00e+00) |
| archive_dense_v1/validation_predictions.csv | 768 | 0.965142393 | 30.532139 | 0.965142393 | 30.532139 | 30.532139 | 0.00e+00/0.00e+00/0.00e+00 | 3.56e-07/3.48e-07 | 0.945556842 vs src 0.945556842 (d=0.00e+00) | 0.985309490 vs src 0.985309490 (d=0.00e+00) |
| v2_dense/validation_predictions.csv | 768 | 0.970196049 | 30.453822 | 0.970196049 | 30.453822 | 30.453822 | 0.00e+00/0.00e+00/0.00e+00 | 1.06e-07/4.44e-07 | 0.950317989 vs src 0.950317989 (d=0.00e+00) | 0.990775388 vs src 0.990775388 (d=0.00e+00) |

| condition (pooled_recent) | E_tune | E_confirm | E_full | B_tune | B_confirm | B_full | R_tune=E_t/B_t | R_confirm | R_full |
|---|---|---|---|---|---|---|---|---|---|
| stable | 0.192028 | 0.184077 | 0.188053 | 0.199219 | 0.193283 | 0.196251 | 0.963907 | 0.952373 | 0.958227 |
| short_history | 0.170808 | 0.184074 | 0.177441 | 0.169916 | 0.184767 | 0.177341 | 1.005249 | 0.996250 | 1.000561 |
| drift | 0.187799 | 0.171521 | 0.179660 | 0.199553 | 0.180115 | 0.189834 | 0.941097 | 0.952287 | 0.946405 |
| shared_change | 0.183363 | 0.178625 | 0.180994 | 0.210312 | 0.200814 | 0.205563 | 0.871861 | 0.889505 | 0.880479 |

**How tune/confirm R relate to full R.** Each subset score recomputes the reference denominator *on that subset*: R_tune[c] = E_tune[c]/B_tune[c] uses the reference error averaged over only the 12 tune episodes, and likewise for confirm. Because tune and confirm have equal episode counts (12/12 per condition), the numerator and denominator *errors* average exactly: E_full[c] = (E_tune[c]+E_confirm[c])/2 and B_full[c] = (B_tune[c]+B_confirm[c])/2 (verified to 1e-12 above). But the *ratios* do not average: E_full[c]/B_full[c] != (E_tune[c]/B_tune[c] + E_confirm[c]/B_confirm[c])/2 unless B_tune[c] = B_confirm[c] = B_full[c]. So R_full is NOT the average of R_tune and R_confirm, and neither subset R is a unbiased shard of the full R; to reconstruct R_full[c] from subsets you need the errors themselves: R_full[c] = (E_tune[c]+E_confirm[c])/(B_tune[c]+B_confirm[c]). Numerically (pooled_recent): R_tune=0.945528, R_confirm=0.947604, mean of the two=0.946566, but R_full=0.946418; points: tune=30.8400, confirm=30.8071, full=30.8259.

**Checks:**
- PASS :: fixture: reference scores R=1 (30 points) :: R=1, points=30.000000000
- PASS :: fixture: targets score R=0 (60 points) :: R=6.67e-34, points=60.000000000
- PASS :: fixture: uniform 0.2 computed :: R=2.203587389, points=18.729004
- PASS :: kit/check_submission.py --split validation runs on all 3 files :: pooled_recent_validation.csv: rc=0, points=30.825854; validation_predictions.csv: rc=0, points=30.532139; validation_predictions.csv: rc=0, points=30.453822
- PASS :: independent scorer matches src/house_edge/scoring.py (R, points, per-condition, tune/confirm) and kit/check_submission.py for all 3 files :: 3 files x (full, tune, confirm); worst |dR| vs src=0.00e+00
- PASS :: E/B decompose as subset means (12+12 episodes) while ratios do not :: E and B identities hold (dE,dB<1e-12); |mean(R_tune,R_confirm)-R_full|=0.000148 (nonzero => denominators differ per subset)

## 3. Payout column order (0/1/2/5/20)

| file | header | matches payout order |
|---|---|---|
| kit/public/validation/targets.csv | query_id,episode_id,p_0,p_1,p_2,p_5,p_20,outcome | True |
| kit/public/validation/reference.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| kit/public/test/sample_submission.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| kit/public/train/sample_submission.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| kit/public/validation/sample_submission.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/archive_dense_v1/validation_predictions.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/archive_sparse_v1/validation_predictions.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/pilot_dense/validation_predictions.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/pilot_sparse/validation_predictions.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/v2_dense/validation_predictions.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/v2_sparse/validation_predictions.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/archive_dense_v1/test_submission.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/archive_sparse_v1/test_submission.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/pilot_dense/test_submission.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/pilot_sparse/test_submission.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/v2_dense/test_submission.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/v2_sparse/test_submission.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/context_test.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/context_validation.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/empirical_test.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/empirical_validation.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/global_test.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/global_validation.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/pooled_recent_test.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/pooled_recent_validation.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/pooled_test.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/pooled_validation.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/recent_test.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/recent_validation.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/shrinkage64_test.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |
| runs/baselines/shrinkage64_validation.csv | query_id,p_0,p_1,p_2,p_5,p_20 | True |


**Checks:**
- PASS :: all 31 prediction/target/reference files use column order query_id,p_0,p_1,p_2,p_5,p_20 (targets add episode_id,outcome) :: files checked=31, mismatches=0
- PASS :: validation targets.csv outcome column in {0,1,2,5,20} :: outcomes seen=[0, 1, 2, 5, 20]
- PASS :: train labels.csv outcome in (0, 1, 2, 5, 20) :: seen=[0, 1, 2, 5, 20]
- PASS :: validation labels.csv outcome in (0, 1, 2, 5, 20) :: seen=[0, 1, 2, 5, 20]
- PASS :: targets.csv rows are probabilities (in [0,1], sum 1 within 1e-6) :: rows=768, bad sums=0
- PASS :: kit schema.PAYOUTS == src data.PAYOUTS == (0,1,2,5,20); p_* columns in same order; idx mapping round-trips :: kit=(0, 1, 2, 5, 20), src=(0, 1, 2, 5, 20), cols=('p_0', 'p_1', 'p_2', 'p_5', 'p_20'), roundtrip=True
- PASS :: history payout values map through PAYOUT_TO_IDX to the same columns (end-to-end token decode verified in section 1) :: history/labels/targets values in {0,1,2,5,20}; window symbols decode to the true history payouts (dense sym=idx+1, sparse sym=idx); see section 1 checks.

## 4. Baselines

| baseline validation CSV | rows | unique ids | = 768 validation ids |
|---|---|---|---|
| context_validation.csv | 768 | 768 | True |
| empirical_validation.csv | 768 | 768 | True |
| global_validation.csv | 768 | 768 | True |
| pooled_recent_validation.csv | 768 | 768 | True |
| pooled_validation.csv | 768 | 768 | True |
| recent_validation.csv | 768 | 768 | True |
| shrinkage64_validation.csv | 768 | 768 | True |

| baseline | ids | max |mine - file| (768x5 probs) | reproduced |
|---|---|---|---|
| pooled_recent | 768 | 0.000e+00 | True |
| shrinkage64 | 768 | 0.000e+00 | True |


**Checks:**
- PASS :: baselines.py loads labels only for train (no validation labels/targets into models) :: with_labels=True lines: ['train_data = load_split(os.path.join(public, "train"), with_labels=True)']; read_labels lines: ['labels = read_labels(os.path.join(public, "train", "labels.csv"), train_queries)']
- PASS :: official validation targets/reference used only inside score_predictions (reporting) :: uses of `official`: ['official = load_official_validation(os.path.join(public, "validation"))', 'tune = score_predictions(val_preds, official, episode_subset=tune_set)', 'confirm = score_predictions(val_preds, official, episode_subset=confirm_set)', 'full = score_predictions(val_preds, official)']
- PASS :: models fit on train queries/history/labels only; predict on val/test queries+history :: fit: ['def shrinkage_fit_predict(train_data, target_data, prior_strength=16.0, window=64):', 'val_preds = shrinkage_fit_predict(train_data, val_data)', 'test_preds = shrinkage_fit_predict(train_data, test_data)', 'model = FrequencyForecaster(name).fit(train_queries, train_history, labels)']
- PASS :: all 7 baseline *_validation.csv output exactly the 768 validation query ids :: files=7 (6 kit frequency methods + shrinkage64), all match=True; |validation ids|=768
- PASS :: baseline *_test.csv cover exactly the test query ids :: files=7, |test ids|=1536
- PASS :: pooled_recent and shrinkage64 validation CSVs reproduced exactly from train data + val queries/history alone (no val labels/targets/reference) :: max abs diffs: 0.000e+00 (pooled_recent), 0.000e+00 (shrinkage64)

## FINAL AUDIT SUMMARY

| Section | PASS | FAIL |
|---|---|---|
| 1. Splits | 28 | 0 |
| 2. Scorer | 6 | 0 |
| 3. Payout column order (0/1/2/5/20) | 7 | 0 |
| 4. Baselines | 6 | 0 |

**ALL CHECKS PASS** — 47 passed, 0 failed (47 checks total).

Audit runtime: 8.3s. Auditor script: experiments/diagnostic/audit/audit.py (stdlib + numpy; src/kit imports only for split recompute, window construction, constants, and post-hoc comparison).
