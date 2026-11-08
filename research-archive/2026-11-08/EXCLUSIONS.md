# Exclusions and limits

This is a sanitized subset of the existing HouseEdge trees, not a filesystem backup.

- No organizer kit source or raw train, validation, confirm, test, reference, target, or synthetic dataset files are redistributed. The final submitted forecast CSV is included because it is the owner's output.
- No Python, Torch, ROCm, wheelhouse, standalone CPython distribution, virtual environment, site-packages, node_modules, or cache tree is included. An extra sanitization pass removed standalone-interpreter source files from the prepared source snapshot.
- No .git directory, authentication configuration, credential file, environment file, SSH helper, agent launcher, raw chat transcript, author session, or unrestricted execution log is included. Structured numeric model training curves and queue completion records are distinct from author sessions.
- Known key/private-key patterns were checked in candidate text files. This is a bounded archive-preparation check, not a claim that an automated scan proves the absence of every secret.
- Six unique intentional corruption/tamper fixtures are indexed but their binary bytes are not included. They are not trained-model successes.
- Some historical trials have configs or score records but no surviving checkpoint. No missing checkpoint was invented or retrained.
- Duplicate checkpoint bytes are stored once. Different SHA256 files can be serialization/config variants or epoch states; a file-hash count is not a count of statistically independent models.
- Two raw RF pickles exceed100MiB. Their exact bytes are represented losslessly by verified gzip parts below20,000,001bytes each. Do not add their expanded raw files to Git. No LFS or new service was used.
- The two host snapshots differ at some shared paths. Both are retained. Local unborn TOWER Git state is not the live GitHub history.

Current documented technical claims are in MODEL_JOURNEY.md and final-submission/README.md. Historical reports retain their original bytes, including superseded blocks and explicitly corrected mistakes. No new license is applied to project code or third-party material.

GitHub's documented policy warns above50MiB and blocks normal Git files above100MiB: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github . All staged files are below both thresholds. Browser/connector attachment limits are separate and are handled by the publication bundle plan.
