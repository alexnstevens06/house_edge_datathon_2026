import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'source')))
import spec
import rf

TYPES = rf.TYPES


def make_episode(eid, seed=0, M=8, cutoff=48, signals=None, types=None):
    rng = np.random.default_rng(seed)
    if signals is None:
        signals = [(-1, 0, 1)[i % 3] for i in range(M)]
    if types is None:
        types = [TYPES[i % 3] for i in range(M)]
    mids = [f'{eid}-m{i:03d}' for i in range(M)]
    qe = pd.DataFrame(dict(query_id=[f'{m}-next' for m in mids], episode_id=eid, machine_id=mids,
                           machine_type=types, signal=signals, cutoff=cutoff))
    P = np.full((M, cutoff), np.nan)
    for i in range(M):
        n = int(rng.integers(16, cutoff))
        ticks = np.sort(rng.choice(cutoff, size=n, replace=False))
        P[i, ticks] = rng.integers(0, 5, size=n)
    return qe, P


def make_world(seed=0, n_episodes=3):
    rng = np.random.default_rng(seed)
    eps, lab = {}, []
    for k in range(n_episodes):
        eid = f'e{k:02d}'
        qe, P = make_episode(eid, seed=100 * seed + k, cutoff=40 + 4 * k)
        eps[eid] = (qe, P)
        for qid in qe.query_id:
            lab.append((qid, int(rng.choice(rf.PAY))))
    return eps, pd.DataFrame(lab, columns=['query_id', 'outcome'])


def flip_signals(eps):
    return {e: (qe.assign(signal=[-int(s) for s in qe.signal]), P) for e, (qe, P) in eps.items()}


def small_specialists(X, y, sig):
    return spec.train_specialists(X, y, sig, n_jobs=1, n_estimators=10, min_samples_leaf=1, max_features=1.0)


def test_routing_uses_the_forest_of_each_machines_own_signal():
    sig_class = {-1: 0, 0: 1, 1: 2}
    eps, lab = {}, []
    for k, s in enumerate((-1, 0, 1)):
        eid = f'tr-s{s}'
        qe, P = make_episode(eid, seed=200 + k, signals=[s] * 8)
        eps[eid] = (qe, P)
        for qid, sq in zip(qe.query_id, qe.signal):
            lab.append((qid, int(rf.PAY[sig_class[int(sq)]])))
    labels = pd.DataFrame(lab, columns=['query_id', 'outcome'])
    X, y, sig, typ, qids = spec.build_train_rows(eps, labels, use_signal=False)
    forests, priors = small_specialists(X, y, sig)
    assert all(forests[s] is not None for s in (-1, 0, 1))
    qe, P = make_episode('mix', seed=7, signals=[-1, 0, 1, -1, 0, 1, -1, 0])
    veps = {'mix': (qe, P)}
    F = spec.episode_matrices(veps, use_signal=False)
    out = spec.predict_specialists(forests, priors, veps, F, 0.0)
    for i, s in enumerate(qe.signal.values.astype(int)):
        p = out[qe.query_id.iloc[i]]
        direct = np.zeros(5)
        direct[forests[s].classes_] = forests[s].predict_proba(F['mix'][i:i + 1])[0]
        assert np.allclose(p, direct)
        assert int(np.argmax(p)) == sig_class[s]
        assert p[sig_class[s]] > 0.99


def test_signal_flip_changes_no_feature_row_and_rows_equal_queries():
    eps, labels = make_world(seed=1)
    X, y, sig, typ, qids = spec.build_train_rows(eps, labels, use_signal=False)
    n_queries = sum(len(qe) for qe, _ in eps.values())
    assert X.shape[0] == len(y) == n_queries == len(labels)
    assert sorted(qids) == sorted(labels.query_id.tolist())
    eps2 = flip_signals(eps)
    X2, y2, sig2, _, _ = spec.build_train_rows(eps2, labels, use_signal=False)
    assert np.array_equal(X, X2)
    for e, (qe, P) in eps.items():
        cut = P.shape[1]
        for t in (8, cut // 2, cut - 1, cut):
            assert np.array_equal(spec.feats_with(P, qe, t, False), spec.feats_with(P, eps2[e][0], t, False))
        assert not np.array_equal(spec.feats_with(P, qe, cut, True), spec.feats_with(P, eps2[e][0], cut, True))


def test_features_are_only_ever_built_at_the_cutoff_tick(monkeypatch):
    eps, labels = make_world(seed=2)
    seen = []
    real = spec.feats_with

    def spy(P, qe, t, use_signal):
        seen.append((int(t), int(P.shape[1]), bool(use_signal)))
        return real(P, qe, t, use_signal)

    monkeypatch.setattr(spec, 'feats_with', spy)
    spec.build_train_rows(eps, labels, use_signal=False)
    spec.build_train_rows(eps, labels, use_signal=True)
    spec.episode_matrices(eps, use_signal=False)
    spec.episode_matrices(eps, use_signal=True)
    assert seen and all(t == c for t, c, _ in seen)
    assert any(u for _, _, u in seen) and any(not u for _, _, u in seen)


def test_probabilities_finite_five_way_sum1_and_query_order(tmp_path):
    eps, labels = make_world(seed=3)
    X, y, sig, typ, qids = spec.build_train_rows(eps, labels, use_signal=False)
    forests, priors = small_specialists(X, y, sig)
    prior_all = np.bincount(y, minlength=5) / len(y)
    veps = {f'v{k}': make_episode(f'v{k}', seed=300 + k) for k in range(2)}
    Fv = spec.episode_matrices(veps, use_signal=False)
    ids = [q for e in veps for q in veps[e][0].query_id]
    for alpha in (0.0, 0.2, 0.5):
        out = spec.predict_specialists(forests, priors, veps, Fv, alpha, prior_all)
        assert list(out.keys()) == ids
        for qid in ids:
            p = out[qid]
            assert p.shape == (5,)
            assert np.all(np.isfinite(p)) and p.min() >= 0
            assert abs(p.sum() - 1.0) < 1e-9
    Xb, yb, _, _, _ = spec.build_train_rows(eps, labels, use_signal=True)
    model, prior_b = spec.train_shared(Xb, yb, n_jobs=1, n_estimators=10, min_samples_leaf=1, max_features=1.0)
    Fvb = spec.episode_matrices(veps, use_signal=True)
    outb = spec.predict_shared(model, veps, Fvb, prior_b, 0.2)
    assert list(outb.keys()) == ids
    for qid in ids:
        p = outb[qid]
        assert p.shape == (5,)
        assert np.all(np.isfinite(p)) and p.min() >= 0
        assert abs(p.sum() - 1.0) < 1e-9
    path = tmp_path / 'validation_predictions.csv'
    spec.write_predictions(outb, path, pd.DataFrame({'query_id': ids}))
    df = pd.read_csv(path)
    assert df.columns.tolist() == ['query_id', 'p_0', 'p_1', 'p_2', 'p_5', 'p_20']
    assert df.query_id.tolist() == ids
    assert np.allclose(df[['p_0', 'p_1', 'p_2', 'p_5', 'p_20']].values,
                       np.stack([outb[i] for i in ids]), atol=1e-9)
