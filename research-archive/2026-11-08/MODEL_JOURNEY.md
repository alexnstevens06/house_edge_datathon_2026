# How HouseEdge reached the final two-model ensemble

This is the research history for HouseEdge only. Several branches ran in parallel, so the sections follow the decisions rather than claim a strict execution order. The source snapshots are historical evidence, not current submission instructions. Some early reports contain blocks or claims that later work corrected. The current submission and dependency notes below take precedence.

The final choice is the equal probability average of curve-C seeds 0 and 2. Its reproduced CPU FP64 TUNE R is **0.28468175920919014**. Lower R is better. This is a tune-selected result, not a hidden-test score or a blind confirmation.

## What the score and split mean

The project uses payouts in the order `0, 1, 2, 5, 20`. Every prediction has five probabilities. The scorer normalizes each row, measures squared probability error within an episode, and takes the ratio to the reference error within each condition. R is the unweighted mean of those four condition ratios. It is not the mean of individual episode ratios.

There are 96 training episodes. Validation has 96 episodes, split into 48 TUNE and 48 CONFIRM episodes. TUNE has 384 queries, eight machines per episode, and 12 episodes in each condition. Test has 1,536 queries across 192 episodes. The training builders produced 18,567 history and cutoff windows for the final CNN recipe.

Earlier diagnostic scripts printed public full-validation and CONFIRM results. Later studies restricted selection and new scoring to TUNE, but that does not restore blindness. This archive preserves that limitation. No new held-out metric was computed to prepare this archive.

Evidence: `source-snapshots/TOWER/experiments/diagnostic/audit/AUDIT_RESULTS.md`, the split manifests, and `evidence/retest/PROTOCOL.md`.

## Frequency, linear, forest, and transformer baselines

The first comparisons included empirical, global, context, pooled, recent-window, and shrinkage frequency predictors. An independent audit checked the splits, causal windows, payout order, reference scoring, and exact baseline reproduction. It recorded 47 passing checks and no failures. The pooled-recent baseline's stored TUNE R was about 0.9455.

Linear feature fits remained near that range. The no-signal fit recorded TUNE R 0.9387000566637661. Four surviving `model.npz` files preserve the fitted linear models, with duplicate bytes stored once.

Random forests improved the stored TUNE score to 0.8177486544177324 for v1 and 0.7872998668082776 for v2 without signal. The signal-specialist forest was worse than its shared-signal comparison, about 0.9960 versus 0.9410. The two large base forests survive as roughly 173 MB pickles each. They must not be added as raw Git blobs. The archive provides lossless compressed parts and original hashes instead.

Dense and sparse transformers explored absolute and lag positions, learning rates, and a readout that adds the machine's own last observed payout. The audit found no behavioral readout or slot-binding bug. It removed unreachable code and documented a capacity-limited cross-machine toy fixture. Stored repaired-transformer TUNE scores remained well above the later CNN results, including 0.8481212565280617 for the base lag dense seed-1 run and 0.8588302706778108 for its own-readout counterpart. Improving a toy fixture was not treated as a competition-score win.

Evidence is under `source-snapshots/SERVER/experiments/linear`, `random_forest`, and `transformer_repair`, plus `source-snapshots/TOWER/experiments/transformer_repair/readout/AUDIT.md`. The original root `src`, `scripts`, `tests`, and surviving `runs` metadata are also archived.

## Small-data CNN tests did not solve the problem

The small CNN branch tested the base recipe, a second base seed, episode balancing, query upweighting, a minimum-observation rule, and a shorter window. The six completed saved runs recorded TUNE R between about 0.9446 and 0.9779. They did not approach the later large CNN. An incomplete upweighting seed is labeled incomplete in its original path and is not presented as a successful model.

These were experiments on training regimes and capacity. A changed sampler did not establish a new independent data source or validate hidden-test performance.

Evidence: `source-snapshots/TOWER/experiments/small_cnn_data/runs/*/metrics.json` and the corresponding configs and structured training curves.

## The larger temporal CNN and the MLP/CNN search

The larger CNN explored hidden sizes, depths, windows, dropout, training duration, and signal use. Stored examples include TUNE R 0.5417 for h64, four layers, W64, 12 epochs; 0.4500 for h128, four layers, W64, 16 epochs; and 0.3675 for the 30-epoch signal-using version. A no-signal anchor remained around 0.3952 before the later search.

The bounded MLP/CNN search retained trial configs, trial metrics, search spaces, refinement decisions, and selected checkpoints. The final CNN03 recipe used h128, four layers, kernel 3, W64, dropout 0.3, 30 epochs, learning rate 0.00197, weight decay 0.00895, and batch 512. The implementation fixed the actual kernel at 3 despite accepting a kernel argument. The archive does not describe that argument as an effective architecture search.

The CNN03 seeds had stored training-export TUNE R 0.33206398675779764, 0.34654552298986674, and 0.33342647148315785. Seed 0 was selected on TUNE. The final MLP02 seeds were much worse, about 1.2722, 1.3262, and 1.2754. The earlier CNN anchors were about 0.3959, 0.3954, and 0.3872.

A config file or trial record does not prove that a final checkpoint survived. The model index lists the checkpoint and fit files that exist now. Pruned trials may have configs and metrics but no surviving fitted weights.

Evidence: `source-snapshots/SERVER/experiments/cnn/runs`, `mlp_cnn_hpo/runs`, the search and refinement files, and `source-snapshots/TOWER/experiments/mlp_cnn_hpo/review/REVIEW_GROK_BUILD.md`.

## Synthetic data stayed research-only and did not earn promotion

The synthetic branch built toy families for stationarity, changing regimes, long lags, missing observations, cross-machine dependence, rare payouts, and other mechanisms. Generator tests checked prefix behavior, missing-versus-zero handling, mappings, and shared state. A passing generator test was not evidence that synthetic training improves the real task.

The independent review distinguished latent oracle probabilities from probabilities conditioned only on observable history. It also found type-order and mask hazards. The public embedding map had to use machine-type strings, not a generator's task-list order or a machine slot. Synthetic eligibility remained unresolved under the organizer-released-data rule. Those descendants are research-only, not submission candidates.

The full-budget paired seed-0 comparison was negative. The real CNN03 control recorded provisional CPU FP32 TUNE R 0.3291909269650826. The 25% abrupt mixture recorded 0.3968037462541104. Both completed 30 epochs and 1,110 updates. The tiny capacity arm recorded 0.8410400241114718. A lower mixed training loss did not offset the worse real TUNE score.

The frozen runner had a tuple-key versus string membership bug in its automatic promotion gate. The coordinator manually stopped the branch after the negative seed-0 result. Seeds 1 and 2 were not promoted. This comparison also used microbatch fallback, a different shuffle stream, and different dropout execution from the original resident-batch CNN. The review did not call it a bitwise recipe replay or a completed FP64 promotion.

Evidence: `source-snapshots/TOWER/experiments/synthetic_machines/grok_review/REVIEW.md` and `source-snapshots/SERVER/experiments/synthetic_machines/full_budget/run_v1/INDEPENDENT_REVIEW_FROZEN_V1.md`. The archive includes surviving research checkpoints with their eligibility warnings, but no raw synthetic or organizer datasets.

## Signal specialists and frozen heads were negative branches

The hard-signal CNN study pretrained on history windows, then compared cutoff fine-tuning through shared and signal-routed models. Its provisional CPU FP32 routed TUNE R was about 0.500991, worse than the matched shared control at 0.462112 and the earlier CNN03 comparison at 0.332064. This was a different training procedure, not a claim that routing alone caused the gap against CNN03. No further seed was promoted.

The soft mixture-of-experts branch had source and CPU tests but was blocked before its planned GPU training. Its record contains no real TUNE result. A blocked branch is not a scored negative model.

Frozen calibration heads on SERVER left the trunk unchanged and used 30 parameters per head. The shared head recorded TUNE R about 0.355801, the routed heads 0.358978, and the frozen baseline 0.358981. Routing did not beat the shared comparison.

TOWER's separate frozen final-layer-head study initially failed under a ROCm build without native gfx1031 support. A later isolated native runtime completed the same bounded study. It used 645 parameters per head and preserved the trunk. Its shared head recorded about 0.350971, while routing recorded 0.365661 against baseline 0.358981. The earlier ROCm block is historical, not the final outcome of that study. No second seed followed.

Evidence: `source-snapshots/SERVER/experiments/cnn_experts/hard_signal/results/RESULTS.md`, `frozen_heads_20261108/results/RESULTS.md`, and the corresponding TOWER `soft_moe` and `frozen_feature_heads_20261108` results and native-runtime record.

## Learning-rate seeds and warm starts

The curve-C branch changed the peak learning rate to 0.003 and retained separate seeds. Authoritative CPU FP64 TUNE R for seeds 0, 1, and 2 is 0.3101981645338761, 0.3452074168357309, and 0.30980333573689856. Seed 2 was the best of these singles. Matched control seeds and the alternate curve-A run remain in the index and snapshots.

The later identity retest found three separate successful training launches, config seeds 0, 1, and 2, 1,110 steps each, and all 43 final tensors different for every pair. The trainer seeds RNG before constructing a new CNN and has no warm-start load in that path. Initial tensors were not saved. Three distinct final states and documented seeded construction do not constitute a direct comparison of historical initial tensors, and the seeds share one training dataset.

The +10 and +20 continuation study made twelve weights-only warm starts with fresh AdamW and a new OneCycle schedule. It was not an exact optimizer resume. The frozen +20 high-LR seed-2 candidate reached R 0.3385772741455493, worse than its parent. Its +10 intermediate was 0.30925871088194445, but that checkpoint was not promoted or extended after inspection. The continuation history is preserved without retroactively changing its declared candidate.

The separate low-rate protocol declared four jobs. Only the SERVER 2e-5 primary and control jobs trained. Each completed 730 steps and exited successfully. Primary epochs 1, 3, and 5 scored 0.30951549616992924, 0.30706943777760692, and 0.30781275392021723. The matched control scored 0.33413621090622814, 0.33213683848995773, and 0.33244487608328904. Checkpoints were saved before FP64 scoring.

The TOWER 5e-5 pair did not train. An initial missing-input-path block was resolved and all 14 input pins passed. The required native FP32 forward/backward probe, with no optimizer step, then reached RSS 2,174,996 KiB against the frozen 2,097,152 KiB cap. The protocol stopped. No six epochs, new checkpoints, rate comparison, or completed four-job claim exists. Hardware and learning rate were confounded across the declared hosts.

Evidence: `source-snapshots/SERVER/experiments/cnn03_convergence/continuation_20261108/runs/RESULTS.md`, `lowrate_20261108/STATUS.md`, and `lowrate_20261108/TOWER_FINAL_RSS_GUARD.md`. A TOWER-local sentence about not starting SERVER work is not evidence that the separately completed SERVER jobs did not run.

## Probability ensembles won the final comparison

The frozen study evaluated equal probability means and a fixed set of blends, not a new unconstrained weight fit. The three-seed mean scored R 0.28975002607300254. The selected equal seed-0-plus-seed-2 mean scored 0.28468175920919014. That was better on the declared TUNE score than the singles, continuation candidates, and completed low-rate checkpoints.

The implementation applies softmax separately to each parent's logits, averages the probabilities, and renormalizes. It does not average logits. A real-forward diagnostic measured a maximum probability difference of 0.07021929501660051 between those two operations. No alternative logit-ensemble score was computed or selected.

The frozen shared-control blend weights produced TUNE values about 0.296455258, 0.300391185, 0.304412732, and 0.306936902 for 20%, 10%, 5%, and 2.5%. None beat the pair. Earlier shared-control scalar records used CPU FP32, while later comparison used canonical FP64. Those provenance and precision caveats were not fixed by altering predictions to fit a scalar. No blend replaced the selected package.

The global selection used the declared TUNE improvement threshold of 1e-6. This is still selection on reused TUNE. The archive does not turn that threshold into protection from search bias.

Evidence: `source-snapshots/SERVER/experiments/cnn03_convergence/ensemble_blend_20261108`, `comparative_t146_20261108/COMPARISON_READY.md`, its status addendum, and the final package review.

## Reproduction required a real FP64 inference path

Early CPU FP32 exports repeated on one host but differed across hosts by as much as roughly 8.47e-7. That exceeds the 1e-8 normalized-probability tolerance. An early review incorrectly called that number smaller than 1e-8. Its correction remains in the historical record.

A double-precision model could not use a forward that forcibly cast mask and tick inputs to float32. The final canonical path creates those inputs in the embedding dtype, keeps logits and softmax in float64, and does not monkeypatch `Tensor.float`.

The final package contains two models, not a one-model reference advertised as an ensemble. Fresh exact-ZIP tests used CPython 3.13.13 and Torch 2.11.0+cpu. Independent review recorded three fresh Docker runs with networking disabled, two CPUs, a 2 GiB memory and swap cap, successful exits, and no OOM. Native TOWER cross-host inference agreed with SERVER after canonical row normalization to 8.881784197001252e-16 on all 1,536 test rows and 5.551115123125783e-16 on 768 validation rows. Native TOWER cold repeats were identical to each other, but not byte-identical to SERVER. Normalized cross-host comparison, not matching only a scalar score, established the tolerance check.

The two-model TOWER test inference took about 80 seconds. The full native verification took about 268 seconds. Conservative summed RSS was 1,882,640 KiB. Independent review's exact package verdict was technical PASS. That is not a guarantee of organizer dependency provisioning or contest acceptance.

Evidence: `evidence/package/PACKAGE_REVIEW.md`, `evidence/package/FINAL_VERIFICATION_ADDENDUM.md`, and `source-snapshots/SERVER/experiments/cnn03_convergence/candidate_t146_ensemble_s0_s2_20261108/QUALITY.md`.

## The independent retest did not rule out tuning luck

The later frozen retest re-forwarded all three parents and the fresh exact ZIP. All five fixed TUNE scores reproduced. The test CSV reproduced byte-for-byte. Deterministic reproduction checks numerical stability. It is not statistical evidence against luck.

The retest resampled 48 whole episodes together with all their machine queries. It used 10,000 paired draws, seed 15020261108, and linear 2.5% and 97.5% quantiles. It recomputed the canonical ratio from sufficient statistics instead of averaging episode ratios. It did not treat 384 correlated machine queries as independent samples.

The pair beat seed 2 on 30 episodes and lost on 18. Removing any one episode left its score gain negative. The paired pair-minus-seed-2 interval was [-0.04733496619889767, -0.00645070743516942]. The all-three-minus-seed-2 interval was [-0.047114033331565386, 0.0037514268305680783]. The pair-minus-all-three interval was [-0.016698567152097853, 0.005392893029217964]. The last two cross zero.

The three largest episode contributions account for 50.13% of the pair's net canonical score gain. The gain is not a single-episode artifact, but it is concentrated. The pair and comparison seed were already chosen on this TUNE split. These are conditional, non-selection-adjusted resampling intervals. Generalization and absence of tuning luck remain INCONCLUSIVE. Three training seeds do not estimate a training-seed population.

A coordinator check found an inconsistent synthetic probability/logit fixture in the original retest source. The frozen source was not changed. A separate consistent softmax fixture gave a gap of 0.3324084855394026, while the actual-model diagnostic remained valid. The coordinator also regenerated every saved bootstrap vector and interval. Read `evidence/retest/COORDINATOR_VERIFICATION.md` before the original report. Bare historical queue clock strings lack a date and time zone and are not presented as CST timestamps.

## The dependency contract remains unresolved

The latest official kit README at line 333 and RULES at line 55 say organizers prepare the exact versions in root `requirements.txt` before the offline run. They do not say Torch is preinstalled. The live guide at line 290 describes Python 3.13 standard library support and extra libraries either offered in the environment or supplied as Linux wheels with offline setup. Neither source publishes a package allowlist.

The selected package follows the kit's root-requirements route. Its file pins 11 dependency entries, including a direct CPython-3.13 Linux CPU Torch wheel URL. It is not a pip `--hash` lock. The ZIP includes no wheelhouse. It performs no runtime download. Preparation under the kit route and acceptance under the live-guide wording have not been reconciled. This archive does not claim automatic Torch availability or guaranteed provisioning.

The stronger measured runtime tests do not resolve that documentation conflict. The immutable package has not been edited to add wheels or change requirements.

## The selected files and older fallbacks stay distinct

The exact files to submit manually are in `final-submission/`:

- `ensemble_equal_s0_s2_fp64_candidate.zip`, SHA256 `d04d96fd0e28c946f6704170326ee10b32d570e6ab63f5bc2855922cbfd5c991`.
- `candidate_test_a.csv`, SHA256 `1f4f1a2c8176a8d2f70c099b5265f8432103fdaaf2fc8744bc60bfe376ef081c`.

The previous single-model release stays identified separately, with ZIP `dcb8a564fed79eae008766a5213b967926a7d7671e80d8ac4b3be3b98d9ced29` and CSV `9bae92ea78b4f865caac0847cd9ff01415c18db96df9d8ac888cbf5de1db75e6`. The original fallback has ZIP `61bbac853e53fc03ecfa9ca9dead96d282e1b51da19442ea8f9425a4a50f5599` and CSV `08ee9e0e02dc3e6800be3083cc0b920e79f012da2e3c1776b05ac0ccd8042b1d`. Neither fallback was overwritten.

The model index covers surviving binary checkpoint and fit artifacts on SERVER and TOWER, their original project-relative locations, their exact hashes, and their archive storage. It distinguishes saved initializations and intentional corrupted fixtures from trained artifacts. It does not claim one independent training run per binary hash or recover weights that were never retained.

No official submission, Git push, new training, environment install, spending, or visibility change was part of archive preparation.
