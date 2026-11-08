# hard_signal plan (FROZEN before any metric)

Experiment (A). Owner session writes only this folder. No HPO. No synthetic training data. No commits, pushes, or submits.

## Question

Does a hard router on the actual query signal (-1 / 0 / +1), sending each query to a CNN specialist fine-tuned only on true TRAIN cutoff rows of that signal, beat the published CNN03 incumbent on tune R? A matched shared control uses the same pipeline and budget and does not route.

## Architecture

CNN03, no signal parameters. Factory is `CNN(h=128, layers=4, k=3, drop=0.3, W=64, use_signal=False)` copied from `experiments/mlp_cnn_hpo/source/base_cnn.py`. That no-signal net is 367109 parameters (the with-signal net is 367493 and is not used). No `sig` embedding. The forward `sg` argument is ignored.

Payout class order is 0, 1, 2, 5, 20. Machine type indices are balanced=0, jackpot=1, steady=2 (`house_edge.data.TYPE_TO_IDX`). Missing is -1, distinct from payout 0: missing uses embedding row 5 plus a mask channel of 0; payout 0 uses embedding row 0 plus a mask channel of 1.

## Data (leakage)

Signal is read only from the query row that asks for that machine at that episode's cutoff. It is a routing key, never a feature, and it is never copied onto any earlier window.

- Pretrain uses real TRAIN history only. For each train episode, windows are `t = min_t, min_t+stride, ...` with `min_t=16`, `stride=1`, and `t < cutoff`. The label at `t` is the observed history payout index at tick `t` (missing slots stay missing). Query labels are not written into history. Query signal is not read.
- Fine-tune uses true TRAIN cutoff rows only: one window per train episode at `t = cutoff`, features from ticks `< cutoff` only, labels only from `train/labels.csv` via `query_labels`. These rows exist (96 episodes, 8 machines, signals in {-1, 0, 1}).
- Validation and test never fit anything. No early stopping. Confirm is not a selection metric.

`build_history` / the pretrain loop do not take a signal tensor. Cutoff features `X` do not contain signal. Routing sees signal only when choosing a specialist at fine-tune mask time and at prediction time.

## Training budget (CNN03 tune protocol, two phases)

Optimizer, schedule, and clips match CNN03 `train.py`: AdamW, OneCycleLR(`max_lr`, `total_steps`), grad clip 1.0, bf16 autocast on CUDA only, loss = masked mean cross-entropy in fp32 over slots with `y >= 0`, seed then CPU init then `.to(device)`.

| | value |
| --- | --- |
| lr | 0.00197 |
| wd | 0.00895 |
| effective batch | 512 |
| micro-batch | 32, gradients accumulated to 512 |
| pretrain epochs | 30 on history windows |
| fine-tune epochs | 30 on the 96 cutoff windows |
| threads | 2 |
| VRAM cap | 2 GiB via `set_per_process_memory_fraction` |

Micro-batch 32 is the only intentional deviation from a single CNN03 batch of 512: the 2 GiB job cap. Accumulation scales each micro-batch sum of cross-entropy by `1 / n_labels_in_the_effective_batch`, clips once, then takes one AdamW and one OneCycle step. Dropout masks are still per micro-batch. Last effective batch may be short, same rule as CNN03.

One pretrain. Snapshot those weights.

- Control: copy snapshot, reseed `seed+1` after the copy, fine-tune 30 epochs on all cutoff labels. No router. Prediction uses this one net for every slot.
- Specialist for signal s in (-1, 0, +1): copy the same snapshot (not the control), reseed `seed+2` / `seed+3` / `seed+4`, fine-tune 30 epochs. Loss mask keeps only slots whose own signal is `s` and whose label is present. All 96 cutoff windows stay in the loader so each fine-tune has the same step count (1 step/epoch, because 96 < 512). Context is still all 8 machines' histories.
- Pretrain RNG is not reseeded after init (CNN03 order).

## Sparse fallback

A specialist is fit only if its signal has at least `min_group_rows = 32` labeled TRAIN cutoff rows (query rows, not episodes). Otherwise that signal's router entry is the shared pretrained CNN, not the control and not an untrained specialist. Threshold is frozen here, before metrics. Published train counts are about 250 per signal, so a real run is expected to fit all three; the threshold is what the tests exercise.

## Prediction and scores

CPU fp32, model eval, 2 threads, softmax over the 5 payout classes. Hard router: for each query, the probability vector is exactly the selected model's vector for that machine's slot. No mixing. Selected model is `specialist_{signal}` or `shared`.

Official `house_edge.scoring.score_predictions` and `runs/splits.json` tune / confirm ids (copied into `SPLIT_IDS.json`). Promotion uses tune R only. Lower is better.

Incumbent CNN03 tune R (from `experiments/mlp_cnn_hpo/server_snapshot/runs/final_cnn03_s{0,1,2}/metrics.json`, read-only):

- seed 0: 0.33206398675779764
- seed 1: 0.34654552298986674
- seed 2: 0.33342647148315785

Rule: seed 0 routed tune R must be strictly lower than the seed-0 incumbent or the experiment is DROP and seeds 1 and 2 are not trained. If it wins, train seeds 1 and 2 the same way. Promote only if all three seeds beat their matched incumbent. Confirm and full validation are reported only after that promotion, and confirm is labeled seen. The matched control's tune R is reported with seed 0 as a diagnostic and is not the promotion bar.

## Outputs

`runs/seed{N}/` holds checkpoint, train log, `tune_only.json` (written before confirm is scored), predictions, and `metrics.json`. `results/RESULTS.md` and `results.json` after the decision. `STATUS.md` gets one CST line at plan frozen, tests pass, training started, first result, final.
