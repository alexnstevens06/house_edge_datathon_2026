# Clean CNN anchor — Python 3.13 CPU package (provisional FP32)

Main round, season `5fffdc4d0c700bbf61c3f911`. Team identifier is selected by the uploader and was not supplied to this packaging lane. No submission is performed by this package.

## Reproduce

Run from the ZIP root, mounted read-only at `/submission`:

```
python predict.py --data /data --out /output/submission.csv
```

The released public round is read-only at `/data`. Only test history and queries are read. No training, labels, conditions, signal features, internet, API, or uploaded forecast CSV is used. Saved checkpoint/config are included; no retraining occurs. Output columns are `query_id,p_0,p_1,p_2,p_5,p_20`.

Checkpoint SHA-256: `43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80`. Seed 0, h128, four layers, W64, dropout0.2, no_signal=true. This is the preserved clean anchor, not the contaminated original and not a tuned seed chosen on confirmation results. Inference remains FP32; no silent precision change.

## Tested dependencies and resources

CPython3.13.13, Linux x86_64, torch2.11.0+cpu, NumPy2.5.4. `requirements.txt` freezes all 11 installed packages, including an explicit official CP313 Linux CPU-wheel URL. Prepare dependencies before the offline prediction run; the prediction command never installs or downloads anything. CPU wheel SHA-256: `45025d7752dbc6b4c784c03afaee9c5f19730ce084b2e43fc9a2fe1677d9ff86`. Wheels are NOT bundled.

The same entry actually ran in an isolated, project-scoped Python3.13 CPU environment on the server: all1,536 queries, 1.360778 seconds total, maxRSS610964KiB, CPU2 threads. Its output was bitwise identical to the original clean CPU forecast:
`9ea67f812b80de18f03b1f6d8f0aeb6d5b91d8ce5c935dc83ba588e04b6be4e9`.

Official format checker passed. Two fresh-folder reruns in the original server environment were also bitwise identical. A strict no-network/read-only/2CPU/16GiB sandbox using the original Python3.12 server dependencies produced the same forecast; that test is not represented as an organizer-image test. Any further Python3.13 sandbox evidence is recorded separately by the coordinator.

## Unresolved gates — do not label this a qualified release

1. The original tower and server environments used DIFFERENT PyTorch versions/backends/CPU ISA. Their FP32 normalized probabilities differ by up to3.742622056712719e-7, exceeding the stated1e-8 tolerance. Same-machine repeatability does not resolve cross-machine reproducibility. FP64/backend-stable integration and independent clearance remain pending. Do not upload while this gate fails.
2. Kit RULES and SUBMISSION_README say organizers prepare pinned requirements before offline inference. The live participant guide instead describes Python3.13 stdlib plus offered environments or bundled compatible offline wheels. This packaging lane does not invent a resolution. The package follows the kit preprovisioning route conditionally; organizer runtime approval remains unresolved.

The corrected CPU2.11 pins supersede the earlier provisional package's incorrect torch2.9.1 pin. Original checkpoint, source forward pass, and preprocessing are unchanged. Canonical forecast is uploaded separately only by the parent's single-uploader process after all gates clear. No upload or organizer contact occurred here.
