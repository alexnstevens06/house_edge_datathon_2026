# STATUS.md — cnn_runtime build progress (updated live)

- [x] TASK.md read; scope confirmed: ONLY experiments/cnn_runtime; no commit/push/submit; CPU 2 threads.
- [x] Kit inspection: README.md, RULES.md, SUBMISSION_README.md, submission_contract.py, build_entry.py, check_submission.py, schema.py, io.py, DATA.md listing.
- [x] Participant guide: PDF unreadable by model reader (no PDF support) → extracted via `pdftotext -layout` (docs/house-edge-participant-guide.pdf → /tmp/opencode/cnn_runtime/guide.txt, 668 lines). Live Google-Doc guide text also staged by user as LIVE_PARTICIPANT_GUIDE.txt.
- [x] Contract evidence + contradictions (60/40 vs 80/20; 10MB vs 16MiB CSV; 100MB vs 128MiB ZIP; requirements-provisioning vs Py3.13-stdlib/wheels; Google Forms vs upload page) recorded in CONTRACT.md. UNRESOLVED, per instruction.
- [x] Checkpoint staged & verified: reference/experiments/cnn/runs/g_nosignal/checkpoint.pt sha256 43dc7050a98dd23e...bff80 (TASK.md prefix match). 4 bitwise CPU exports (a–d) hash-match TASK.md (test 9ea67f8..., validation 50efa7cb...). Reference source byte-identical to tower cnn source (diff); data.py/scoring.py identical; only server train.py differs (training-only).
- [x] CONTRACT.md written (first, before package code).
- [ ] entry/ package: predict.py + model.py + data.py + checkpoint + requirements.txt + README.md + reproduce.json
- [ ] Local CPU 2-thread inference test+validation; bitwise compare vs reference; repeat determinism
- [ ] Official checker (test + validation splits)
- [ ] tests/test_runtime.py written & run
- [ ] ZIP built via kit build_entry.py; official inspect_archive; limits/content checks
- [ ] Fresh-extraction rerun; mismatch-failure test
- [ ] Docker sandbox attempt (or documented fallback + local CPU2 CLI repeat)
- [ ] REPORT.md final

Timestamps: work session 2026-11-07 ~18:5x CST onward. Fallback aim 19:30 CST.
