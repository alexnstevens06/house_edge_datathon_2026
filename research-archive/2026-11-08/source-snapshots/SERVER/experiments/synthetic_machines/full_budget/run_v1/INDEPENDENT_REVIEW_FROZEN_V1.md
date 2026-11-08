# Independent review: frozen full-budget seed 0

Snapshot 2026-11-07T19:59:26-06:00. Read-only. No training and no forward rescore. This audit is the frozen server copy under `pinned_server_v1/`, source sha256 `294cde525e7fb4d20b61d0e00faec54d6b6ccdced47f3263cf02287fd449d656`. The live `full_budget/source/full_compare.py` is a later file and is not the executed runner. The 19:43 snapshot in this file is superseded.

## Verdict

Quarter-abrupt CNN03 loses the first FULL paired comparison. Final-epoch tune R is 0.3291909269650826 for real CNN03 and 0.3968037462541104 for the 25% abrupt mix. Both finished 30 epochs and 1110 updates. Higher R is worse. The coordinator's manual prune is the right call on these numbers. No seeds 1 or 2. Research-only. Eligibility stays unresolved. No flagship claim.

The automatic gate did not make that decision. In the pinned runner, `by` is keyed by `(case, seed)` tuples, and the test is `"mix" in by`. A string is never a member of that dict, so the branch that would launch seeds 1 and 2, or record `pruned`, does not run. The author fixes that separately. This review does not edit the runner.

CPU fp32 tune R is provisional. `FP64_PROMOTION_RULE.txt` and the run record both say a later FP64 check of the same checkpoints is still required. No FP64 evaluator result is in this tree. This review does not claim that check passed.

The 60-step screen (real tune R 1.0053866589636473, mix 0.25 R 1.0093277493540138, mix 0.50 R 1.0166853589603029) is a different recipe and is not this gate. Do not add a 300-step qualification.

## What the frozen metrics say

| Case | tune R | epoch-30 train loss | steps | exposures | real / synth windows | microbatch | init sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny real h32 L2 W32 | 0.8410400241114718 | 0.9967913499278955 | 1110 | 557010 | 557010 / 0 | 512 | `a5350c4b0ce12c005db16c9fbe98d3688ae50e1ffe625e63680e7cd8254661c2` |
| CNN03 real h128 L4 W64 | 0.3291909269650826 | 0.7921143844738799 | 1110 | 557010 | 557010 / 0 | 128 after OOM at 512 | `329c26e90d7864b0a0984b55917a67ed3469b9444e17329fb1be5994c39268db` |
| CNN03 + 25% abrupt | 0.3968037462541104 | 0.6941228278350375 | 1110 | 557010 | 417750 / 139260 | 128 after OOM at 512 | same as CNN03 |

Raw files: `cnn03_seed0/metrics.json` `f30998864dca351f57888297c8b4c63358526e4f30281ac40b06e03b4f361adc`, `mix_seed0/metrics.json` `51ff488840df1ee638da8380e4361580c88511b2540fefa271f3a7ee3b84ca3c`, `tiny_seed0/metrics.json` `cbf48b4b457c01eda8f9b7c08ab719b13aab696482d6f007e0172c49f4853432`. These match `RESULTS_AND_PRUNE.json` `6c70544a484beae4ac627d8e84484b28977e0e4f37ae552df884f254903629a5`. Delta mix minus control is 0.06761281928902779. CNN03 and mix share one logged init hash. Their trained checkpoints are different files.

CNN03 first-epoch loss is 1.225918606184852. The canonical final CNN03 record is epoch-0 loss 1.2260050686879749, epoch-29 loss 0.7904063574638712, tune R 0.33206398675779764, full R 0.36845454406496114. This run's epoch-30 loss is 0.7921143844738799. Same architecture and recipe, nearby trajectory, not a bitwise replay. This run did not compute full-validation R.

Mix train loss ends lower than the real CNN03 loss because the mixed objective includes synthetic tokens. The gate uses tune R. The mix is worse on tune R at every saved curve point from epoch 4 onward (epoch 1 is 1.03594 versus 1.03545).

Tiny at this canonical recipe reaches tune R 0.841 after 1110 steps. That is the capacity arm, not the control, and it is not the 60-step screen (dropout 0.1, constant lr 0.002, weight decay 0.05, batch 128). It shows the screen regime and this full recipe are different. It does not prove step count was the sole cause of screen R near 1.

## Checks that hold on the pinned runner and artifacts

Pinned sources, also recorded inside `diagnostics.json` `ce2934cccf9175f18aa7b889df14439f239a1e35455715d434358315c094391b`: `full_compare.py` `294cde52…d449d656`, `base_cnn.py` `f0e9469b579486de49f6ec43343d2159433d6be4786758095ebbfd0e052bfba4`, `abrupt_generate.py` `6f6e29c88e831c72084645d418bd51ba8173b20231e97d17bd70d27850bf2dde`. Canonical `base_cnn.py` / `train.py` remain `b4bbd9af456e5edfbd2f3123a0001e1da45f50f676ed98f336ee134e6cc778d5` and `916c6ee1b6e405300ddadbc5650583b250db8baa83a5227fc2437c1b1a55e066`. The copied CNN matches the canonical `CNN`, `Block`, `window`, `build_train`, `build_query`, `predict`, `to_preds`, and `masked_loss` ASTs. Byte difference is the repo path and the deleted CLI main that wrote confirm/test submissions. `USE_SIGNAL` is set false before the build. Diagnostics report AST equality and W64 tensor equality with the canonical builder.

Real TRAIN pack from that diagnostic, which this session rebuilt with the same `base_cnn.py` hash and matched: 96 episodes, 18567 windows, targets int8 in [-1, 4], ticks 16..256, signal channel all 0. Labelled slots 131050, missing slots 17486. Payout-index counts 76546, 25212, 19201, 8416, 1675. Y sha256 `e6467d9d5d96597566e5448f903882fac724c6cb0b6c68a8778abbf84e9759b3` for both W32 and W64. W32 X `8584de57b41d21a18a0a9f6d31d9fea414f10138f3bbe1e122eaef4cb5af5eec`. W64 X `36c896a670371395734446398ff6434beaf3b2e8390e2cce402887590e6e29f2`. `131050 * 30 = 3931500` labelled-slot exposures, matching the real arms. `18567 * 30 = 557010` window exposures. `ceil(18567/512) * 30 = 1110` steps.

Cutoff causality: `window()` reads ticks `< t`. The diagnostic alignment at tick 40 says a future edit is invisible and a past edit is visible. This session's rebuild on episode 0 also kept a sentinel written on the cutoff row out of the feature window, and the cutoff training target matched the query labels. Missing is -1. Observed payout 0 stays index 0. Public `TYPE_TO_IDX` is balanced 0, jackpot 1, steady 2. Query scoring uses canonical `to_preds` / `slot_of` on the tune split only. Confirm intersection is refused by the runner. Test is not loaded.

Abrupt ingest in the same diagnostic: generator sha `6f6e29c8…`, seed 0, 8×128×8, 8192 records, 6193 observed, 1999 unobserved numeric payouts ignored, 896 windows, 5432 labelled slots, 1736 missing. Truth rows and query rows discarded. The generator's `TYPE_TO_INDEX` (steady 0, balanced 1, jackpot 2) is task order and is not the embedding map. The runner maps `machine_type` strings through public `TYPE_TO_IDX`. Synth slots are raw `machine_id` 0..7. Real slots are sorted public machine-id strings. Those slot embeddings do not name the same machines.

Mix replacement is the tail of each shuffled real batch, with-replacement draws from the 896 synth windows (`numpy` Generator, seed+9176). 139260/557010 is the 25% synth share. Optimizer steps stay 1110. There is no BatchNorm. The step sums token cross-entropy, backprops each microbatch, then divides the accumulated gradient by the labelled-token count of the whole effective batch, clips at 1, and takes one AdamW and one OneCycle step. That is the mean-token gradient, not an average of microbatch means. A 4-row CPU toy of that identity matched `F.cross_entropy` mean to about 4e-9. It was not a replay of these checkpoints.

## Sampler and microbatch, explicitly

Canonical `train.py` draws `torch.randperm(..., device=dev)` on CUDA and runs one resident batch of 512. The pinned runner draws `torch.randperm(n)` on CPU and, for CNN03 and the mix, fell back from microbatch 512 to 128 after OOM (`oom_fallbacks` on both metrics; peak 805.7 MB; launch note says a 2.5 GiB allocator cap). Tiny stayed at microbatch 512. Dropout is 0.3 and runs per launch, so four microbatches are not the same masks as one 512-wide forward. bf16 autocast addition order also changes. The metrics sentence that a smaller microbatch changes only addition order is stronger than the code. Same init hash, same step count, same nominal batch 512, different shuffle stream and different dropout stream from the canonical server CNN03 run.

## Checkpoints, loaded on CPU, not fitted

| File | sha256 | tensors |
| --- | --- | --- |
| `cnn03_seed0/checkpoint.pt` | `5054d913a37da102b21b009912b42507c786bb6f7af01a9d5a6ae7febc064f5c` | 43× float32 CPU, `pay.weight` (6, 128), no `sig`, no BatchNorm buffers |
| `mix_seed0/checkpoint.pt` | `a3f592baf577835d124fccacf12f302af50511e8c40008267aad172f00cf37b0` | same layout, config mix 0.25 |
| `tiny_seed0/checkpoint.pt` | `324e0a5723d30579c8470da1f15f389b64087fbacb41a32e4277a83c3f3677cf` | 27× float32 CPU, `pay.weight` (6, 32) |

Each save has `eligibility=RESEARCH_ONLY`, `eligibility_status=unresolved`, `submission_export=false`. These hashes match `ARTIFACT_SHA256.json`. Weights are the epoch-30 export. Intermediate tune points were logged and were not used to pick a checkpoint.

## Memorization diagnostic

Pinned `diagnostics.json` fit 32 real TRAIN windows for 20 steps, dropout 0, no held-out score. Loss 1.686685562133789 to 1.002213954925537, token accuracy 0.57421875. That is not memorization of the fixture, and it is not a tune R. It does not explain the mix loss. The later live source has a different memorization routine; it is outside this frozen run.

## Not verified

- Tune R was not recomputed by a forward pass of the three checkpoints. Stored R is the runner's CPU fp32 `predict` plus the official mean loss ratio.
- No FP64 inference result exists here.
- Dropout masks were not dumped. The 4e-9 gradient check was a toy, not these models.
- The logged init sha was not recomputed from a fresh constructor. The trained checkpoints differ from each other, which is expected after different data.
- Full-validation and confirm R for this run were not produced. Canonical full R 0.36845 belongs to the earlier server final, not to these checkpoints.
- Undertraining is not established as the single cause of the old screen R near 1. The full mix loss stands on its own tune-R comparison.
