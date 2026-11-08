# Final technical verification addendum

2026-11-08 03:15:56 CST. **FINAL TECHNICAL VERIFICATION PASS.** Ready to deliver as a separate conditional candidate. This addendum supersedes the earlier pending-cross-host/pending-review phase statements in frozen QUALITY.md, CANDIDATE_READY_FOR_CROSSHOST.md and the packaged README. Those files, ZIP, model, configs, weights, forecasts and source receipt have not been rewritten. Technical PASS is not organizer acceptance or an automatic canonical replacement.

## Exact delivery pair
- Forecast: `refs/candidate_test_a.csv`, 198055 bytes, 1536 test queries; SHA-256 `1f4f1a2c8176a8d2f70c099b5265f8432103fdaaf2fc8744bc60bfe376ef081c`.
- Code: `ensemble_equal_s0_s2_fp64_candidate.zip`, 2749917 bytes, 12 members; SHA-256 `d04d96fd0e28c946f6704170326ee10b32d570e6ab63f5bc2855922cbfd5c991`.
- Only original LR0.003 seeds0+2, equal probability weights0.5/0.5. No low-rate continuation or shared-control weight is inside this ZIP. Both original checkpoint/config pins and canonical model/data bytes were independently verified.
- Canonical CPUFP64 TUNE R **0.28468175920919014**, versus previous delivered candidate **0.30980333573689856**; improvement **0.02512157652770841**. Best verified completed row, not a claim about six unrun Tower checkpoints.

## Completed release gates
- Independent PACKAGE_REVIEW.md technical **PASS**, SHA-256 `ad281f714a9c9a16809c12443268d5e7c2c6238b19385ff3e0e1f6cb5bed68b9`; see `evidence/independent_review/`. No remaining package review gate.
- Fresh exact ZIP: both components float64 CPU forward/softmax; strict hash/config/keys/kernel/shape/signal/saved-dtype/finiteness/weight/component refusal tests, query-order/causality/missing-versus-zero, all1536test and768validation IDs, finite simplex, cold byte-repeat and verified-ensemble agreement. Only384TUNE rows scored, not confirm/full/test R.
- SERVER CPython3.13.13/Torch2.11.0+cpu: both models measured together. Host test about2.35s combined inference/3.03s total, peak336604KiB. Three NEW independent fresh containers at2CPU, memory AND swap2147483648bytes, networknone, read-only root, CapDropALL, no-new-privileges: all EXIT0/OOMfalse, test repeats match forecast bytes. Independent strict-container test peak333944/333960KiB. The author's first three containers had16GiB limits; its fourth and the three independent repeats provide strict2GiB proof.
- TOWER different-runtime CPU compatibility: existing Python3.12.3/Torch2.14.0+rocm10.1.0, GPUhidden, two-core affinity. Unchanged harness on fresh ZIP EXIT0/HARNESS_OK at03:05:51CST. Full harness268.28s, slowest two-model test80.36s total. Monitored RSSsum1882640KiB/PSSsum1265096KiB; budget_stopfalse. Qualification of pinned3.13/2.11 is SERVER-only.
- Independent canonical KIT row-normalized **actual SERVER-vs-TOWER CSV** comparison: all1536test maxabs8.881784197001252e-16; all768validation5.551115123125783e-16, both<=1e-8. Tower cold repeats match each other but not SERVER bytes. Deliver the SERVER CSV above, not the Tower derivative.
- Actual base-to-candidate HARNESS_DIFF regenerated and verified byte-for-byte, SHA b46e7891e2caf4374903f9f92ec83da522dc11eeea5c118b8739ee377172e7f7. All48 frozen receipt rows rehashed after review with0mismatches. Both previous ZIP/CSV pairs and production pins/harnesses unchanged. SERVER tune_only.json still e96358d9..., not overwritten by review. Only Tower's generated runtime metadata differs; its original SERVER metadata was preserved.

## Study outcomes and qualifications
- SERVER2e-5 pair completed730steps/job, EXIT0; primary best epoch3 R0.30706943777760692, then worsened at epoch5. Not selected over the ensemble. Warm starts were weights-only/freshAdamW/FP32batch128, not exact optimizer resumes.
- TOWER5e-5 arm CLOSED/BLOCKED: staged inputs/source pins all passed; nativeFP32 batch128 forward/backward with NO optimizer step reached2174996KiB above2097152KiB. No training job/epoch/checkpoint, no retry or cap/recipe/guard changes. Only2of4low-rate jobs completed; six Tower rows remain unscored, not ranked as worse.
- Blends remain not cleared: historical shared-control scalar was FP32; canonicalFP64 differs. No fitted probability correction. Low-rate replay/host-binding/forward-count/TF32 concerns remain on that closed study, not on this ensemble.
- The pair is tune-informed and many predeclared TUNE comparisons were made. No significance, hidden-test or blind-confirmation claim; earlier packaging already exposed public validation diagnostics.
- Dependency-contract ambiguity remains unresolved. The tested runtime had the dependencies preinstalled; no wheels are bundled. Organizer acceptance/runtime provisioning must be checked separately. Nothing uploaded/submitted/committed; no production pin changed.
