# Receiver ready

2026-11-08 01:50 CST. Ensemble and blend evaluation is finished. `lowrate_20261108/HANDOFF_READY` is not in the tree. No low-rate training was started. This file is not a poll.

## Tests

`tests/test_matrix.py`: 9 passed on `.venv_gfx1031` (Python 3.12, torch 2.14), CPU only, before the matrix. Guards covered wrong hash, config, keys, shape, signal tensor, saved dtype, non-finite weights, unknown and non-tune ids, split/source pins, no-future window, missing versus payout 0, query order, simplex, weight-0 exact identity, arithmetic means, and a real seed-2 float64 logit/softmax.

## Source hashes

| file | sha256 |
| --- | --- |
| PLAN.md | `7c48cdef610d3985b6f7cf07c23402cf0e624457aeffc47d7c16ab9af4fcc74f` |
| source/common.py | `7b53b7f7a77193bbc844de5fb28672d07081cfae29185d691790df1c89cd20ec` |
| source/evaluate.py | `27a8ca737ec247809f13a300560bfa8603493f7d703a19c2e167817ee704de75` |
| tests/test_matrix.py | `25e5e46dcd54da1dda992c8108b16bce667fb4d27c05a3826b4c4409b989fdd4` |
| COORDINATOR_PLAN.md | `03d141087ac950e8026ebaeb6afcc4f0b83371508beee0961ecd993cc9f9e8f3` |

Canonical model `7e63ccf7ad5b1177071d931aca13713390a5bdcc667f69652f6f97aa7e6c527a`. No GPU job, no `HSA_OVERRIDE`, no install.

## Tune R, 384 queries

Current best, unchanged: 0.30980333573689856.

| row | tune R | vs current best |
| --- | ---: | ---: |
| ensemble seeds 0+2 (tune-informed) | 0.28468175920919014 | -0.02512157652770842 |
| ensemble seeds 0+1+2 | 0.28975002607300254 | -0.02005330966389602 |
| blend w=0.2 | 0.29645525780787213 | -0.01334807792902643 |
| blend w=0.1 | 0.30039118477366389 | -0.00941215096323467 |
| blend w=0.05 | 0.30441273225560084 | -0.00539060348129772 |
| blend w=0.025 | 0.30693690199632961 | -0.00286643374056895 |
| blend w=0 and seed 2 | 0.30980333573689856 | 0 |

All positive blend weights and both ensembles clear 1e-6. Nothing was packaged. Confirm, full, and test were not scored. Prediction maps are in `results/predictions/` (768 rows each, float64). `results/matrix.json` has the per-file sha256.

The closed shared control's canonical float64 tune R is 0.46211190598948326, which is 2.934e-08 away from the historical scalar 0.46211193532631023. See `results/RESULTS.md`.

## When the handoff arrives

Read the frozen manifest and verify source, input, split, and data hashes, then launch only the two Tower commands it names for the 5e-5 pair. Do not edit the mirrored training source.
