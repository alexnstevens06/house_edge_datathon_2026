# HouseEdge research archive

Prepared 2026-11-08 08:50:09 AM CST for the existing public repository https://github.com/alexnstevens06/house_edge_datathon_2026 . Preparation did not commit, push, submit a forecast, change visibility, install software, or train models.

Read [the model journey](MODEL_JOURNEY.md) for the decisions and the negative results. Read [the submission note](final-submission/README.md) for the exact selected files and unresolved dependency contract. Historical source snapshots may contain superseded blocks, old FP32 numbers, or earlier dependency interpretations. They are evidence of the path, not current instructions.

## Contents

- `final-submission/` contains the exact immutable selected CSV and ZIP. Do not replace them with another CSV from a source snapshot.
- `MODEL_INDEX.json` lists 206 observed binary checkpoint and fit paths, 117 distinct file hashes, and their original project-relative locations. Those counts are not independent training runs.
- 109 distinct surviving trained-checkpoint or fit artifacts and 2 saved research initializations are stored losslessly in `models/blobs/`. Six intentionally corrupted guard fixtures have hashes and locations but no archived bytes. Pruned trials with no surviving weights retain their available configs and metrics only.
- `source-snapshots/SERVER/` and `source-snapshots/TOWER/` preserve host-specific source, tests, plans, manifests, split IDs, configs, search records, and structured model metrics. Keep both versions where their hashes differ. Do not flatten or merge the snapshots.
- `evidence/package/` contains the actual independent technical PASS review and final verification record.
- `evidence/retest/` contains the frozen independent retest and its coordinator corrections. These are conditional TUNE diagnostics, not blind generalization evidence.
- `FILE_MANIFEST.json` and `SHA256SUMS.txt` cover this prepared archive. A later publisher must preserve the listed bytes and verify the exact final CSV and ZIP again.

## Restore model bytes without changing the submission

`MODEL_INDEX.json` records storage by original SHA256. A `raw` entry stores its original bytes directly. A `gzip_parts` entry lists ordered parts of one gzip stream. Concatenate those parts in listed order, check the stored gzip SHA256, decompress, and check the original model SHA256. Only load pickle or Torch artifacts you trust. Hash verification does not make arbitrary pickle execution safe.

The two original forest pickles are 172,984,008 and 172,854,424 bytes, above GitHub's regular-Git 100 MiB file limit. Their lossless gzip streams are 55,471,759 and 55,387,837 bytes. Each stream is split into three parts, none above 20,000,000 bytes. Do not commit the expanded raw forests. All staged model files fit below GitHub's warning and hard file limits. No Git LFS, release service, or additional storage account was installed or created.

The selected model storage represents 489,766,221 original bytes in 254,787,385 stored bytes. The whole project history is larger than one coding-agent attachment call. The separate publication attachment plan lists bounded bundles; the complete archive is not one under-46MB file.

## Scope and rights

Only HouseEdge was read. No files from other challenges were included. Organizer kit code and datasets, private targets, raw synthetic data, wheels, environments, caches, Git internals, credentials, SSH helpers, and raw author sessions are excluded. The archive applies no new license and does not claim a redistribution grant for the official kit. Obtain the official kit through the organizer's permitted route before running historical source that imports it.

Historical tests may need private local data, specific hardware, or old runtimes. This is a research archive, not a claim that every archived test passes in one environment. Do not run training or score new held-out data to publish it. See `EXCLUSIONS.md` and the per-host sanitized manifests.
