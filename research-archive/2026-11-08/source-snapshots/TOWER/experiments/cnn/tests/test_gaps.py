"""Gap-closing tests for REVIEW_OPENCODE.md Q3/Q4 (run: python experiments/cnn/tests/test_gaps.py).

1a end-to-end model causality on real gappy short_history validation episodes;
1b cutoff-label mutation never enters build_train features (only Y at t==cutoff changes);
1c loss masking: cnn.masked_loss (cnn.py:91, used by main at cnn.py:129, factored verbatim from
   the former inline `yl = y.long(); msk = yl >= 0; loss = F.cross_entropy(lg.float()[msk], yl[msk])`
   at old cnn.py:122-123) equals labelled-subset cross-entropy; masked slots give zero loss/grad;
1d USE_SIGNAL=False: flipping query signals leaves build_query tensors and model outputs identical;
3  export contamination guard in predict_cpu.py (refusal before any split data is loaded; --audit
   writes AUDIT_INVALID_-prefixed outputs; clean positional usage unchanged)."""
import os, sys, copy, time, inspect, tempfile, subprocess, numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "source"))
import cnn, predict_cpu
from house_edge.data import load_split, episode_conditions, MISSING, N_MACHINES
torch.set_num_threads(2)
PUB_TRAIN = os.path.join(cnn.PUB, "train"); PUB_VAL = os.path.join(cnn.PUB, "validation"); PUB_TEST = os.path.join(cnn.PUB, "test")

def _fresh_model():
    torch.manual_seed(0)
    return cnn.CNN(32, 2, 3, 0.0, 64, use_signal=False).eval()

def _subset(data, episode_ids):
    d = copy.copy(data); keep = set(episode_ids)
    d.episodes = [e for e in data.episodes if e.episode_id in keep]
    d.episode_index = {e.episode_id: i for i, e in enumerate(d.episodes)}
    d.queries = [r for r in data.queries if r["episode_id"] in keep]
    return d

def _flip_signal(r):
    s = int(r["signal"]); return str(1 if s == 0 else -s)

def test_model_causality_on_real_gappy_episodes():
    """Q3 gap 1+3: full-forward causality (ticks >= t on each machine, payout fills and
    MISSING<->observed toggles) on real short_history episodes with MISSING gaps, not synthetic data."""
    va = load_split(PUB_VAL); conds = episode_conditions(PUB_VAL)
    eps = [e for e in va.episodes if conds[e.episode_id] == "short_history"][:3]
    assert len(eps) == 3
    m = _fresh_model(); W = 64; rng = np.random.default_rng(7)
    checked = 0
    for ep in eps:
        assert (ep.obs < 0).any(), "episode must contain MISSING ticks (real gappy history)"
        full = np.full((ep.cutoff + 1, N_MACHINES), MISSING, np.int8); full[:ep.obs.shape[0]] = ep.obs
        ty = torch.tensor(ep.machine_types)[None]; sg = torch.zeros(1, N_MACHINES, dtype=torch.long)
        for t in sorted({16, 64, ep.cutoff // 2, ep.cutoff}):
            base = torch.tensor(cnn.window(full, t, W))[None]
            assert (base < 0).any(), "window at t must contain MISSING gaps"
            with torch.no_grad(): p0 = m(base, torch.tensor([t]), ty, sg)
            for s in range(N_MACHINES):  # mutate ticks >= t on EACH machine, one at a time
                for kind in ("set_payouts", "toggle_missing"):
                    f2 = full.copy()
                    if kind == "set_payouts":
                        f2[t:, s] = rng.integers(0, 5, f2[t:, s].shape)
                    else:
                        col = f2[t:, s].copy(); obsd = col >= 0
                        col[obsd] = MISSING; col[~obsd] = rng.integers(0, 5, int((~obsd).sum()))
                        f2[t:, s] = col
                    w2 = torch.tensor(cnn.window(f2, t, W))[None]
                    assert torch.equal(w2, base), f"window leaked ticks >= t ({ep.episode_id} t={t} m={s} {kind})"
                    with torch.no_grad(): p = m(w2, torch.tensor([t]), ty, sg)
                    assert torch.equal(p, p0), f"model output changed from ticks >= t ({ep.episode_id} t={t} m={s} {kind})"
                    checked += 1
        # sanity: the check is not vacuous — mutating the last in-window tick IS detected
        t = ep.cutoff; f3 = full.copy(); f3[t - 1, 0] = 0 if full[t - 1, 0] != 0 else 1
        w3 = torch.tensor(cnn.window(f3, t, W))[None]
        with torch.no_grad(): assert not torch.equal(m(w3, torch.tensor([t]), ty, sg), m(torch.tensor(cnn.window(full, t, W))[None], torch.tensor([t]), ty, sg))
    assert checked == 3 * 4 * N_MACHINES * 2

def test_cutoff_label_mutation_never_enters_train_features():
    """Q3 gap 2: flipping train query_labels leaves build_train X,T,TY,SG bit-identical;
    only Y at t==cutoff changes, exactly at the queried machine slots."""
    W = 64; TR = load_split(PUB_TRAIN, with_labels=True); conds = episode_conditions(PUB_TRAIN)
    ids = [e.episode_id for e in TR.episodes[:2]]
    short = next(e.episode_id for e in TR.episodes if conds[e.episode_id] == "short_history")
    if short not in ids: ids.append(short)
    SUB = _subset(TR, ids)
    MUT = copy.copy(SUB); MUT.query_labels = {q: (v + 1) % 5 for q, v in SUB.query_labels.items()}
    cnn.USE_SIGNAL = False
    X, T, TY, SG, Y = cnn.build_train(SUB, W); X2, T2, TY2, SG2, Y2 = cnn.build_train(MUT, W)
    assert torch.equal(X, X2) and torch.equal(T, T2) and torch.equal(TY, TY2) and torch.equal(SG, SG2)
    changed = (Y != Y2).any(-1); cuts = {ep.cutoff for ep in SUB.episodes}
    assert changed.any() and all(int(T[i]) in cuts for i in changed.nonzero().flatten()), "Y may only change at t==cutoff"
    for ep in SUB.episodes:  # exact per-episode semantics
        one = _subset(SUB, [ep.episode_id]); one2 = _subset(MUT, [ep.episode_id])
        a = cnn.build_train(one, W); b = cnn.build_train(one2, W)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
        cut = torch.tensor([ep.cutoff]); below = a[1] < cut; at = a[1] == cut
        assert at.any() and torch.equal(a[4][below], b[4][below]), "labels below cutoff must be history, not query labels"
        slots = sorted(ep.slot_of(r["machine_id"]) for r in one.queries)
        for i in at.nonzero().flatten():
            for s in range(N_MACHINES):
                if s in slots:
                    assert int(a[4][i, s]) != int(b[4][i, s]) and int(b[4][i, s]) == (int(a[4][i, s]) + 1) % 5
                else:
                    assert int(a[4][i, s]) == MISSING == int(b[4][i, s])

def test_loss_masking_matches_main_and_missing_contributes_zero():
    """Q3 gap 4: cnn.masked_loss (cnn.py:91, used by main at cnn.py:129) equals main's former
    inline cross-entropy (old cnn.py:122-123) and the labelled-subset loss; masked-slot logits
    get zero gradient and changing them leaves loss and labelled gradients bit-identical."""
    assert "masked_loss(lg, y)" in inspect.getsource(cnn.main), "main must use the shared masked_loss helper"
    rng = np.random.default_rng(11)
    lg = torch.randn(6, 8, 5, requires_grad=True)
    y = torch.tensor(rng.integers(-1, 5, (6, 8))).long(); y[0, :5] = MISSING; y[1, 6:] = MISSING
    yl = y.long(); msk = yl >= 0
    assert msk.any() and (~msk).any()
    inline = F.cross_entropy(lg.float()[msk], yl[msk])  # exact replication of old cnn.py:122-123
    assert torch.equal(inline, cnn.masked_loss(lg, y))
    bi, bj = msk.nonzero(as_tuple=True)  # independent labelled-subset computation
    assert torch.equal(inline, F.cross_entropy(lg.detach()[bi, bj], yl[bi, bj]))
    lg_sub, y_sub = lg.detach()[msk].clone(), yl[msk].clone()  # drop missing slots entirely
    assert torch.equal(inline, F.cross_entropy(lg_sub, y_sub)), "missing targets contribute zero: compacting to the labelled subset must not change the loss"
    lg1 = lg.detach().clone().requires_grad_(True); l1 = cnn.masked_loss(lg1, y); l1.backward()
    assert (lg1.grad[~msk] == 0).all(), "masked slots must get zero gradient"
    lg2 = lg.detach().clone()
    with torch.no_grad(): lg2[~msk] += 100.0  # wildly change masked-slot logits
    lg2 = lg2.requires_grad_(True); l2 = cnn.masked_loss(lg2, y); l2.backward()
    assert torch.equal(l2.detach(), l1.detach()), "masked-slot logits must not change the loss"
    assert torch.equal(lg2.grad, lg1.grad), "gradients must be unaffected by masked-slot logits"

def test_no_signal_build_query_ignores_query_signal():
    """Q1b (validation/test style): with USE_SIGNAL=False, flipping every query signal leaves
    all build_query tensors and the full model output bit-identical; SG stays all-zero."""
    cnn.USE_SIGNAL = False; m = _fresh_model()
    va = load_split(PUB_VAL); te = load_split(PUB_TEST)
    for data in (va, _subset(te, [e.episode_id for e in te.episodes[:8]])):
        d2 = copy.copy(data); d2.queries = [dict(r, signal=_flip_signal(r)) for r in data.queries]
        assert any(a["signal"] != b["signal"] for a, b in zip(data.queries, d2.queries)), "flip must change something"
        Q, _ = cnn.build_query(data, 64); Q2, _ = cnn.build_query(d2, 64)
        for u, v in zip(Q, Q2): assert torch.equal(u, v)
        assert int(Q[3].abs().sum()) == 0
        with torch.no_grad(): assert torch.equal(m(*Q), m(*Q2))

def _fake_run_dir(path, contaminated=False, no_signal=True):
    os.makedirs(path, exist_ok=True)
    if contaminated: open(os.path.join(path, "CONTAMINATED.txt"), "w").write("fake contamination marker (guard test)\n")
    torch.manual_seed(0)
    m = cnn.CNN(8, 1, 3, 0.0, 16, use_signal=not no_signal)  # predict_cpu builds the model with use_signal=not no_signal
    torch.save({"state": m.state_dict(), "config": {"h": 8, "layers": 1, "W": 16, "drop": 0.0, "no_signal": no_signal}}, os.path.join(path, "checkpoint.pt"))
    return path

def _patch_data_loaders():
    def boom(*a, **k): raise AssertionError("split data was loaded before the guard refused")
    names = ("load_split", "load_official_validation", "score_predictions", "write_submission_csv")
    saved = {n: getattr(predict_cpu, n) for n in names}
    for n in names: setattr(predict_cpu, n, boom)
    return saved

def _restore_data_loaders(saved):
    for n, v in saved.items(): setattr(predict_cpu, n, v)

def test_export_guard_refuses_contaminated_run_before_loading_data():
    """Q4: run dir containing CONTAMINATED.txt is refused (exit 2) even with a clean checkpoint,
    BEFORE any split data is loaded; same refusal end-to-end via subprocess."""
    with tempfile.TemporaryDirectory(prefix="cnn_gap_guard_") as tmp:
        _fake_run_dir(tmp, contaminated=True, no_signal=True)
        saved = _patch_data_loaders()
        try:
            try:
                predict_cpu.main([tmp, "x"]); raise AssertionError("expected refusal SystemExit")
            except SystemExit as e: assert e.code == 2
        finally: _restore_data_loaders(saved)
        r = subprocess.run([sys.executable, os.path.join(HERE, "..", "source", "predict_cpu.py"), tmp, "x"], capture_output=True, text=True)
        assert r.returncode == 2 and "REFUSED" in r.stderr and "CONTAMINATED" in r.stderr, r.stderr
        assert sorted(os.listdir(tmp)) == ["CONTAMINATED.txt", "checkpoint.pt"], "refusal must not write outputs"

def test_export_guard_refuses_signal_mode_checkpoint():
    """Q4: checkpoint config without no_signal=True is refused before loading data, no outputs written."""
    with tempfile.TemporaryDirectory(prefix="cnn_gap_guard_") as tmp:
        _fake_run_dir(tmp, contaminated=False, no_signal=False)
        saved = _patch_data_loaders()
        try:
            try:
                predict_cpu.main([tmp, "x"]); raise AssertionError("expected refusal SystemExit")
            except SystemExit as e: assert e.code == 2
        finally: _restore_data_loaders(saved)
        assert os.listdir(tmp) == ["checkpoint.pt"]
def test_export_guard_audit_prefix_and_clean_positional_usage():
    """Q4: clean run exports via plain `predict_cpu.py RUN SUFFIX` (no prefix); --audit exports of a
    CONTAMINATED.txt dir and of a signal-mode checkpoint write AUDIT_INVALID_-prefixed outputs only."""
    with tempfile.TemporaryDirectory(prefix="cnn_gap_clean_") as tmp:
        _fake_run_dir(tmp); predict_cpu.main([tmp, "gap"])
        for f in ("validation_cpu_gap.csv", "test_cpu_gap.csv", "metrics_cpu_gap.json"): assert os.path.exists(os.path.join(tmp, f)), f
        assert not any(n.startswith("AUDIT_INVALID_") for n in os.listdir(tmp))
    for fake in (lambda p: _fake_run_dir(p, contaminated=True), lambda p: _fake_run_dir(p, no_signal=False)):
        with tempfile.TemporaryDirectory(prefix="cnn_gap_audit_") as tmp:
            fake(tmp); predict_cpu.main([tmp, "gap", "--audit"])
            for f in ("AUDIT_INVALID_validation_cpu_gap.csv", "AUDIT_INVALID_test_cpu_gap.csv", "AUDIT_INVALID_metrics_cpu_gap.json"):
                assert os.path.exists(os.path.join(tmp, f)), f
            assert not os.path.exists(os.path.join(tmp, "validation_cpu_gap.csv")), "audited non-compliant export must not write clean names"

if __name__ == "__main__":
    fails = 0; t0 = time.time()
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS", n)
            except Exception as e: fails += 1; print("FAIL", n, repr(e))
    print("FAILURES", fails, f"{time.time() - t0:.1f}s"); sys.exit(1 if fails else 0)
