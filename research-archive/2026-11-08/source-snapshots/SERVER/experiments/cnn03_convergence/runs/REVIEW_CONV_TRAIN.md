# Review: conv_train vs canonical CNN trainer

**Verdict: CONCERN**

`train_canonical.py` is a byte-identical copy of `experiments/mlp_cnn_hpo/source/train.py` (`cmp` exit 0). The optimizer step matches that CNN path. The curve `tune_R` values are GPU bf16 autocast predictions, not a CPU FP64 forward.

Static read only. Curve logs were not re-trained.

## Checklist

| Item | Verdict | Where |
|---|---|---|
| Train windows | PASS | `conv_train.py:72-74` loads `kit/public/train` and calls `build_train` only on that split. `base_cnn.py:32-42` windows are `t in [min_t, cutoff]` on those episodes. Val/test are not windowed. Same builder as `train_canonical.py:61-63`. |
| Seed then init, same RNG use | PASS | `conv_train.py:65-67` is `set_num_threads(2)`, `torch.manual_seed`, `np.random.seed`, then `CNN` at `:85`. Canonical is the same pair of seeds at `train_canonical.py:55`, then `CNN` at `:67`. Intervening work is CSV/JSON/`torch.tensor` packing and the scoring import. No `torch` or NumPy draw before `CNN.__init__`. First torch RNG consumer is parameter init. Canonical also loads the test split before init (`train_canonical.py:62`); that load does not draw RNG. |
| AdamW | PASS | `conv_train.py:86` and `train_canonical.py:71`: `AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)`. No other kwargs. |
| OneCycleLR | PASS | `conv_train.py:19-20,76-78,87`: `total_steps = epochs * ceil(n/bs)`, and the only extra ctor arg vs `train_canonical.py:72` is `pct_start`. Control and C pass `pct_start=0.3` (`queue1.sh:6,8`, `queue2_C_winner_seeds.sh:6`), which is the library default (`pct_start: float = 0.3`). Curve A passes `0.1` on purpose. |
| Batch / randperm | PASS | `conv_train.py:101-108` matches `train_canonical.py:75-77`: `model.train()`, `torch.randperm(n, device=dev)`, `range(0, n, bs)`, partial last batch kept. |
| bf16 autocast | PASS | `conv_train.py:109-110` matches `train_canonical.py:78-79`: `autocast("cuda", bfloat16, enabled=dev=="cuda")` around the forward only. |
| Masked loss | PASS | `conv_train.py:111` calls `base_cnn.masked_loss` (`base_cnn.py:91-96`): CE on `lg.float()` where `y >= 0`. Same call as `train_canonical.py:80`. |
| Grad clip | PASS | `conv_train.py:112-116` matches `train_canonical.py:81`: `zero_grad`, `backward`, `clip_grad_norm_(..., 1.0)`, `opt.step()`, `sch.step()`. |
| Update count | PASS | `bs=512`, `epochs=30`, `n_train_windows=18567` → `ceil(18567/512)=37`, `30*37=1110`. Logged on every curve summary (`steps_per_epoch`, `total_steps`) and on `curve_control_s0/curve.jsonl:15` (`epoch 29`, `steps 1110`, `exposures 557010` = `18567*30`). Eval does not call `sch.step`. All 7 curve processes exited 0 (`QUEUE_STATUS.txt:1-16`). |
| Eval hook | PASS | `conv_train.py:30-40,48-59`: snapshot CPU + CUDA torch RNG, `model.eval()`, score, then `model.train()` and restore. `base_cnn.py:81-88` `predict` stays in `eval` under `no_grad`. `nn.Dropout` (`base_cnn.py:53,57,59`) does not sample when `training` is false, so the hook does not consume torch RNG and does not keep a dropout mask. Kit `contest_score` calls `score_predictions` with `bootstrap_repeats=0` (`kit/slot_forecasting/scoring.py:105`), so it does not draw `random`. Next epoch sets `model.train()` again before `randperm` (`conv_train.py:101-104`). |
| Tune-only kit score | PASS | `conv_train.py:53-56,80-84`: `runs/splits.json` key `tune` (episode ids) into `house_edge.scoring.score_predictions`, which subsets then calls kit `contest_score` (`src/house_edge/scoring.py:12,30-45`). No confirm/full score. |
| `use_signal=False` | PASS | `conv_train.py:10` sets `B.USE_SIGNAL=False` before `build_train`, so `episode_meta` does not read query signal (`base_cnn.py:17-18`). `CNN(..., use_signal=False)` at `conv_train.py:85` leaves `sig is None` and skips it in forward (`base_cnn.py:65,75`). Same as `train_canonical.py:10,67`. |

## Scoring numeric path (the concern)

Curve runs had `peak_gpu_mb ≈ 3165`, so `dev` was CUDA (`conv_train.py:68-69`). `eval_tune` calls `B.predict(..., dev)` (`conv_train.py:54`). That forward is CUDA autocast bf16 (`base_cnn.py:86-88`); softmax is fp32 of those logits, then `float64` numpy. It is not a CPU FP64 forward. Canonical **tune** scoring uses the same `predict(..., dev)` path (`train_canonical.py:89`). Canonical CPU scoring is only `phase=="final"`.

## Matched-seed tune_R (control − C)

Same seed, `wd=0.00895`, `bs=512`, `drop=0.3`, `h=128`, `layers=4`, `W=64`, `pct_start=0.3`, 30 epochs. Control `lr=0.00197`, C `lr=0.003`. Lower R is better. Margin is control `final_tune_R` minus C.

| Seed | Control | C | control − C |
|---|---:|---:|---:|
| 0 | 0.3319900847820123 | 0.31022932089244687 | +0.02176076388956544 |
| 1 | 0.34547671209003805 | 0.34546344629169745 | +0.00001326579834060 |
| 2 | 0.33332853645945915 | 0.30950933696487987 | +0.02381919949457928 |

C is lower on all three seeds. Seed 1 is a near-tie (`1.3e-5`). Seeds 0 and 2 are about `0.022` and `0.024`. These are the GPU bf16 tune scores above, not a CPU FP64 rescore.
