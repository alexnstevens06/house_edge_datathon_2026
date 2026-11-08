# Results (bounded jobs finished)

2026-11-08 00:17:20 CST. Launcher exit 0. Twelve CUDA warm starts, parent FP64 scores, continuation FP64 scores, and the blend grid. No confirm, test, or submission file.

Parents were weights-only. Each restart used a fresh AdamW and a new OneCycle covering only the extra epochs. Parent SHA-256 values still match `MANIFEST.json`. GPU is idle again (2 MiB, 0%).

Study tune R is CPU float64 through the pinned canonical FP64 module, then canonical sum-renorm, then kit `score_predictions` on `runs/splits.json` tune ids. Incumbent under that scorer: **0.33206398205344967**. The canonical FP64 CSV scores to the same value. Published CPU-fp32 tune R is 0.33206398675779764.

## Tune R

| Arm | Parent | +10 | +20 |
|---|---:|---:|---:|
| C s0 lr 0.003 | 0.3101981645338761 | 0.32175874405089355 | 0.342295142373659 |
| control s0 | 0.33239084535278074 | 0.3278502730949005 | 0.3523407569601887 |
| C s1 lr 0.003 | 0.3452074168357309 | 0.32251894300266665 | 0.39313234010633447 |
| control s1 | 0.34543239563228384 | 0.3604473040840377 | 0.3814229860445475 |
| C s2 lr 0.003 | 0.30980333573689856 | 0.30925871088194445 | 0.3385772741455493 |
| control s2 | 0.3334821793902223 | 0.3323948474741577 | 0.338873569818254 |

Frozen +20 candidate: **curve_C_lr003_s2_plus20**, tune R 0.3385772741455493. The matched control +20 is 0.338873569818254. This candidate is above the incumbent and above its own parent. The +20 warm start raised tune R for every high-LR seed.

The lowest high-LR number inside the predeclared table is **curve_C_lr003_s2_plus10** at 0.30925871088194445, about 0.000545 below its parent. It was not promoted and was not trained further.

## Blends

`w=0` reproduces the CNN03 CSV tune R 0.33206398205344967. Materialized hard-signal tune R matches the archived figures exactly: routed 0.5009909376779845, control 0.46211193532631023.

Lowest grid points, both at the top weight 0.2: control mix 0.32114647095721605, routed mix 0.32173621595454593. Every positive weight in the frozen set is below the CNN03 baseline on tune. The grid was scored on tune, so that minimum is biased. The weight was not increased after seeing the edge.

## Training evidence

Each pair shared the 5080 at 100% and about 7711–7715 MiB. Per-process peak allocated 3164.9 MiB. +20 wall time about 101 s per process, +10 about 51 s. First epochs and PIDs are in each `runs/*/train.log` and in `STATUS.md`.
