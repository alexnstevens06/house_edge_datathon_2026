# Manual upload instructions — main round

This is a technically verified, separate candidate, not an automatic replacement. Nothing has been uploaded.

1. Read FINAL_VERIFICATION_ADDENDUM.md and the unresolved environment qualification below.
2. On your existing House Edge **main-round** submission page, select the forecast/CSV field and choose exactly `refs/candidate_test_a.csv` from this delivery folder. It contains1536 test queries. Do not choose candidate_val.csv, the Tower derivative, or an older fallback CSV.
3. For the accompanying code/reproduction archive, choose exactly `ensemble_equal_s0_s2_fp64_candidate.zip`. Keep this ZIP intact: do not unzip/recompress/edit it for upload, and do not mix it with an older CSV. If the page does not request a code archive, do not put the ZIP into the forecast field.
4. Use the correct team/account identifier in portal fields. No Team ID was supplied for the packaged README. If organizer rules require a Team ID inside the archive, obtain a separately reverified metadata revision rather than editing this frozen ZIP.
5. Before manually submitting, confirm the organizer's accepted dependency/runtime provisioning route. The code was qualified with CPython3.13.13, torch2.11.0+cpu and numpy2.5.4 using the packaged lock. Kit-versus-guide provisioning rules remain unresolved; no wheel is bundled and no organizer approval is claimed.
6. Verify the exact pair if desired:

```sh
sha256sum refs/candidate_test_a.csv ensemble_equal_s0_s2_fp64_candidate.zip
```

Expected CSV SHA-256: `1f4f1a2c8176a8d2f70c099b5265f8432103fdaaf2fc8744bc60bfe376ef081c`

Expected ZIP SHA-256: `d04d96fd0e28c946f6704170326ee10b32d570e6ab63f5bc2855922cbfd5c991`

The frozen `reproduce.json` specifies image `python` and:

```sh
python predict.py --data /data --out /output/submission.csv
```

It predicts the test split, uses both original seed0/seed2 models with equal weights and no training. The separately delivered CSV is the verified SERVER output. Technical offline verification does not guarantee organizer acceptance. Previous candidate/fallback files remain available and unchanged. Submission is a separate manual decision, not performed here.
