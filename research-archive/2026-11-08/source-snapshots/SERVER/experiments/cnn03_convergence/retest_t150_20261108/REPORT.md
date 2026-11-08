# t150 retest, 2026-11-08

Independent retest of the released equal pair of original curve-C seeds 0 and 2. This file is a new measurement report. It does not change the release, the production pin, or any frozen study.

Protocol SHA `beca226ba238e5c55f2a0087cd28142eccfa2035be71b8b2fdd37f620b326033` matched `PROTOCOL.sha256` before the freeze and again after the scoring run. Analysis source, pins, and tests were hashed in `FREEZE.json` before any score or bootstrap number was written. The first process start died in `outputs/` directory creation and wrote no forward and no score (`outputs/measure_attempt1.log`). `OUT.mkdir(..., exist_ok=True)` was applied and `FREEZE.json` was rewritten before the scoring run. That run printed `MEASURE_OK` and exited 0. Wall time was 10.65 seconds. Freeze hashes were rechecked afterward and had not drifted.

## Direct answers

There were three distinct seed training runs. Identity of those three finished processes is VERIFIED. Each run has its own launch line, its own config seed (0, 1, or 2), exit 0, and a final checkpoint whose 43 tensors all differ from the other two. They are not three copies of one checkpoint file. They trained on the same 18567-window training split, so they are not three independent datasets. The initial weights were never saved, so an empirical proof that the initializations differed is INCONCLUSIVE.

The tune gain reproduces. Reproduction is VERIFIED. A fresh CPU FP64 forward of the three parents, the equal probability mean of seeds 0 and 2, the equal mean of all three, and a fresh extract of the released ZIP all match the prior tune references with absolute delta 0 within 1e-12. That is deterministic reproduction of scores on the reused 384-query tune split. It is not evidence against selection luck.

The gain on this tune split is broad across episodes and is not a one-episode artifact of the ratio, and it is still concentrated. The pair has lower episode squared error than seed 2 on 30 of 48 episodes and higher error on 18, with wins and losses in every condition. Dropping any one episode leaves the canonical pair-minus-seed2 delta negative, from -0.027999885280731718 to -0.020617782281603547, with 0 sign reversals. The three largest episode error reductions are 32.1% of the gross error reduction and 50.4% of the net error-gap sum. Robustness and generalization stay INCONCLUSIVE.

## Verdicts

| Question | Verdict |
| --- | --- |
| Three distinct seed training runs | VERIFIED |
| Saved initial weights | NOTVERIFIED |
| Empirical initialization difference | INCONCLUSIVE |
| Independent datasets | NOTVERIFIED |
| Reproduction of the five tune scores and the released ZIP pair | VERIFIED |
| Robustness / generalization | INCONCLUSIVE |

Negative delta R means the left model has the better contest score.

## Identity

Original run folders are `experiments/cnn03_convergence/runs/curve_C_lr003_s0`, `s1`, and `s2`. Each folder contains only `checkpoint.pt`, `curve.jsonl`, and `summary.json`. There is no initial-weight file, warm-start file, or optimizer file. Checkpoint pins, rehashed before and after scoring:

- s0 `d5d714649a93e1e552ccf5419cb01adee311c082af1063f80d8b42f6d8dfe3cb`, config seed 0, lr 0.003, out path `.../curve_C_lr003_s0`
- s1 `8a528179ad626fd3dede0989d047991449808b0feca9032cbda6729ceb07ce74`, config seed 1, lr 0.003
- s2 `bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71`, config seed 2, lr 0.003

Each checkpoint is exactly the keys `config` and `state`, with 43 tensors. Pairwise float64 max abs, and identical-tensor counts: s0 vs s1 `5.543461322784424` and 0 of 43; s0 vs s2 `5.360453128814697` and 0 of 43; s1 vs s2 `4.746379733085632` and 0 of 43. The final weights are three different states.

`source/conv_train.py` (`bd269f926a83d9b736c50cf4f0131fcd979c75a7630d4221e11bdf49828b2e14`) sets `torch.manual_seed(a.seed)` and `np.random.seed(a.seed)`, then constructs `B.CNN(...)`. The only `torch.save` writes the final `state_dict` and `vars(a)`. The file contains no `load_state_dict`. `source/queue1.sh` launches `curve_C_lr003_s0 --lr 0.003 --pct_start 0.3 --seed 0`. `source/queue2_C_winner_seeds.sh` loops `s in 1 2` and launches `curve_C_lr003_s$s --lr 0.003 --pct_start 0.3 --seed $s`. `runs/QUEUE_STATUS.txt` (`af666b3c042f5bf44dee296967f5d04edb8bd981c78fce1ec5c348b86e858cea`) records separate processes: s0 start 02:00:45 end 02:01:59 exit 0; s1 start 02:02:00 end 02:03:13 exit 0; s2 start 02:04:26 end 02:05:40 exit 0.

Each summary records `n_train_windows` 18567, 1110 steps, lr 0.003, and its own seed. The in-training `final_tune_R` values are the training hook, not the FP64 bar: s0 `0.31022932089244687`, s1 `0.34546344629169745`, s2 `0.30950933696487987`.

Because the initial state was not written, distinct seeds and distinct final tensors do not empirically prove that the unsaved initial draws differed. A shared warm start is absent from this trainer source. The three runs are three seeds on one training split.

## Hierarchy

The tune contract is 48 episodes and 384 queries. Every tune episode has 8 queries and one condition. Condition counts are 12, 12, 12, and 12 for `stable`, `short_history`, `drift`, and `shared_change`. Query columns are `query_id`, `episode_id`, `machine_id`, `machine_type`, `signal`, `cutoff`. There is no id above `episode_id`.

The 384 tune `machine_id` strings each appear in exactly one episode, and each machine row has one query. The strings embed the episode id and a local suffix `m000` through `m007`. Those labels are local indices. Canonical R still pools episodes inside the four conditions, so the 48 episodes are not 48 independent copies of R. Episode-order SHA256 `bb499747e2f2a06a0cbaa1522efe154feeef199a738fddb27e74bd70839f5427`. Tune query-id SHA256 `c9f25039f9352c411dcf9540845e11f0b1b50f36cbb3327c44e5d0f6cc4ab4d8`.

## Reproduction

Environment: CPython 3.13.13, NumPy 2.5.4, torch 2.11.0+cpu, `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` empty, 2 threads. Parent max RSS 321976 KiB. Fresh ZIP validation predict max RSS 288384 KiB. Fresh ZIP test predict max RSS 335176 KiB. The 2 GiB line is 2097152 KiB. Scoring used `ensemble_blend_20261108/source/common.py` `forward_probabilities` and `score_tune_detail` on the 384 tune queries. Each forward covered 768 validation queries. Confirm, full, and test were not scored. The candidate harness main was not run. `candidate_t146_ensemble_s0_s2_20261108/tune_only.json` remains `e96358d9edd860942fbf55b7beb4a08e4483bfea5efefd22a841052073984932`.

Fresh tune R, absolute delta versus the prior reference, and absolute delta of the sufficient-statistic reconstruction versus the official 384-query score:

| Row | tune R | abs delta vs prior | reconstruction abs delta |
| --- | --- | --- | --- |
| s0 | 0.3101981645338761 | 0 | 0 |
| s1 | 0.3452074168357309 | 0 | 0 |
| s2 | 0.30980333573689856 | 0 | 0 |
| equal3 | 0.28975002607300254 | 0 | 0 |
| pair 0+2 | 0.28468175920919014 | 0 | 0 |

The reconstruction uses kit row normalization, the mean squared error inside each episode, the mean of those episode errors inside each resampled condition, the ratio to the frozen reference, and the unweighted mean of the four ratios. It does not average per-episode ratios. A synthetic counterexample in the frozen tests separates the two functionals: canonical R `0.777027027027027` versus the forbidden mean of episode ratios `0.979375`.

The fresh ZIP extract under `outputs/fresh_zip` scored tune R `0.28468175920919014`, absolute delta 0 versus the fresh parent pair. Kit-normalized max abs versus the fresh pair on tune queries is `1.1102230246251565e-16`. Fresh pair versus the stored npz is `4.996003610813204e-16`. Fresh equal-3 versus its stored npz is `4.440892098500626e-16`. The regenerated test CSV is byte-identical to `1f4f1a2c8176a8d2f70c099b5265f8432103fdaaf2fc8744bc60bfe376ef081c` and was not scored. Rehash after the run: released ZIP still `d04d96fd0e28c946f6704170326ee10b32d570e6ab63f5bc2855922cbfd5c991`, `SOURCE_HASHES.txt` still `106e1deda225ca8902e1c69f75cb7f093910253bcfd927d2ffd83a23025806eb`, production `PINNED.json` still `cdfcf1222835f5826796a95c20deff2a8c90435fda4df65a74338dc5410166cb`, production harness still `515c91bc176b3e10e65e314afadbb20d66a21b015f2c2c98c391759d549edae2`.

Point deltas on the 384-query tune score: pair minus s2 `-0.02512157652770841`, equal3 minus s2 `-0.020053309663896013`, pair minus equal3 `-0.0050682668638123984`.

## Probability mean, not logit mean

`common.equal_probability_mean` averages the five probabilities with `sum(item[key][i] for item in maps) / count` and then `canonical_renorm`. The released `predict.py` does the two-parent case as `(maps[0][key][i] + maps[1][key][i]) / 2.0` and the same renorm. Softmax is applied inside each parent forward, before that average.

A second float64 forward stored the raw logits and was not scored. On the sorted tune query ids, the max abs gap between the probability mean and `softmax(mean logits)` is `0.07021929501660051`. The frozen synthetic counterexample gap is `0.3817384593585713`. `logit_ensemble_tune_R` is `NOT_COMPUTED`. No logit model was selected.

## Resampling, conditional on this tune split

The bootstrap drew 10000 replicates with `numpy.random.default_rng(15020261108)`. NumPy 2.5.4 has no attribute `numpy.default_rng`. `numpy.random.default_rng` is the Generator constructor the protocol names. The seed, the 10000 by 48 draw, the shared draws, and `quantile(method="linear")` at 2.5% and 97.5% were fixed before the scores. The frozen test locks the draw stream. Saved draws have shape `(10000, 48)` int64. Replicate file SHA `a1c2b3b845a05e26dfc386efdc923df5f390b0fe1177213ee05c8028d2b4fb18`. Every replicate contained all four conditions, so no replicate was dropped and no alternate interval was invented. The same episode indices were used for every model. Query-independent bootstrap was not run.

Each replicate rebuilds R from the episode sufficient statistics. It does not average episode R.

Percentile intervals for model R:

| Model | 2.5% | 97.5% |
| --- | --- | --- |
| s0 | 0.25029216692141887 | 0.37939907461283773 |
| s1 | 0.27052182422045834 | 0.43637165423368723 |
| s2 | 0.2499225237370521 | 0.3834322932889998 |
| equal3 | 0.2296182086130417 | 0.3625772016060355 |
| pair 0+2 | 0.2284379009094809 | 0.35095519289411364 |

Paired differences:

| Comparison | 2.5% | 97.5% |
| --- | --- | --- |
| pair minus s2 | -0.04733496619889767 | -0.00645070743516942 |
| equal3 minus s2 | -0.047114033331565386 | 0.0037514268305680783 |
| pair minus equal3 | -0.016698567152097853 | 0.005392893029217964 |

On this reused tune sample, the central 95% of episode-resampled pair-minus-s2 values lies below zero. The equal3-minus-s2 interval and the pair-minus-equal3 interval both cross zero. These intervals describe resampling of the 48 episodes already in the selected tune split. The pair and the best seed were both already chosen on that split. The intervals are not a proof against selection luck, not a blind confirm result, and not a hidden-test or generalization result. Three training seeds do not estimate a training-seed population.

## How broad the gain is

Episode squared-error gaps are descriptive. Their sum is not canonical R. Tie tolerance is absolute gap at most 1e-12. There were no ties.

Pair minus seed 2: 30 episodes with lower pair error, 18 with higher. By condition, wins/losses are stable 8/4, short_history 9/3, drift 5/7, shared_change 8/4. Gross error reduction `0.35835878528999504`. Net error-gap sum, with a lower left error counted positive, `0.2283026063549457`. The largest episode, `main-validation-e00009` (`stable`), gap `-0.041902404025995765`, is 11.7% of the gross reduction and 18.4% of the net. The largest three, `e00009` (`stable`), `e00091` (`drift`), and `e00008` (`short_history`), are 32.1% of the gross reduction and 50.4% of the net.

Leave-one-episode-out on canonical R, pair minus seed 2: full delta `-0.02512157652770841`, held-out range `-0.027999885280731718` to `-0.020617782281603547`, 0 sign reversals. Removing any one episode leaves the pair ahead on this tune split. That is a statement about the ratio functional on these 48 episodes. The error-scale table still shows 18 episodes where seed 2 has the lower error, and half of the net error-gap sum sits in three episodes.

Equal-3 minus seed 2: 28 lower, 20 higher, 0 ties. LOO canonical range `-0.023619482175041284` to `-0.015033648888735929`, 0 sign reversals, full delta `-0.020053309663896013`. Its bootstrap interval crosses zero.

Pair minus equal-3: the pair has lower episode error on 20 episodes and higher error on 28. The net error-gap sum still favors the pair (`0.048211753838118265`) because the pair's winning gaps are larger. The largest three episodes contribute more than the net, share `1.1435742073398176`, which is what happens when the remaining episodes pull the other way. LOO canonical range stays negative and small: `-0.007098623745643573` to `-0.0030628709829137457`, 0 sign reversals. The bootstrap interval crosses zero.

Per-episode and per-machine tables are `outputs/episode_errors.json` (48 rows), `outputs/machine_errors.json` (384 rows), and `outputs/loo.json`.

## What this retest refuses to claim

The five reproduced numbers were already the prior tune references. Matching them shows the forward, the package, and the scorer are stable. It does not show that the choice of seed 2, or of the 0+2 pair, was free of tune-selection luck. No confirm R, full validation R, or test R was computed. No new seed, pair, weight, or logit ensemble was fit. The kit's own `bootstrap_mean_interval` was not used.
