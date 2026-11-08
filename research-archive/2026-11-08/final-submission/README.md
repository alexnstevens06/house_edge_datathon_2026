# Selected manual submission files

The forecast is `candidate_test_a.csv`: 1,536 test queries, 198,055 bytes, SHA256 `1f4f1a2c8176a8d2f70c099b5265f8432103fdaaf2fc8744bc60bfe376ef081c`.

The matching reproduction archive is `ensemble_equal_s0_s2_fp64_candidate.zip`: 2,749,917 bytes and 12 members, SHA256 `d04d96fd0e28c946f6704170326ee10b32d570e6ab63f5bc2855922cbfd5c991`.

Keep both files intact. Use the CSV for the existing HouseEdge main-round forecast field. Supply the exact ZIP only where the portal requests the code or reproduction archive. Do not upload a validation CSV, the research archive, model-storage parts, or this README as the forecast. No official submission was performed by archive preparation.

The ZIP predicts with two original CNNs, seeds 0 and 2, using the equal probability mean and CPU FP64. No training is required. Its `reproduce.json` declares `python predict.py --data /data --out /output/submission.csv`. The package is not a one-model reference.

## Technical verification and provisioning caveat

Independent exact-ZIP offline Docker checks passed with CPython 3.13.13, Torch 2.11.0+cpu, NumPy 2.5.4, two CPUs, a 2 GiB cap, and networking disabled. Cross-host normalized probability differences were below 1e-8. The fresh independent retest reproduced the CSV byte-for-byte. These tests establish technical reproduction in the prepared environments, not organizer acceptance or blind predictive performance.

The current official kit README:333 and RULES:55 say organizers prepare exact versions from root requirements before the offline run. They do not promise preinstalled Torch. The live guide:290 says Python 3.13 standard library, with extra libraries offered by the environment or supplied as Linux wheels with offline setup. Neither publishes a package allowlist.

This ZIP uses the kit root-requirements route, with 11 exact-version entries including a direct cp313 Linux CPU Torch wheel URL. `requirements-reference.txt` is a byte-identical reference copy, SHA256 `cf701c4d2f312e39a8883c114ed354106fbb7b7d0cd5968f99cc6bd8746a0cab`. It is not a pip --hash lock. No wheels are bundled and no runtime download occurs. The two descriptions of dependency provisioning remain unreconciled. Do not claim Torch is preinstalled or guaranteed.

If the portal requires a team identifier inside the ZIP, do not silently edit the frozen archive. A metadata revision would require its own complete verification. The old package upload note in historical evidence is not a replacement for this current dependency clarification.
