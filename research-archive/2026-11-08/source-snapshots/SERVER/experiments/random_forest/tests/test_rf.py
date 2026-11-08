import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../source')); import rf
def ep(M=8, T=60, seed=0):
    r = np.random.default_rng(seed); P = r.integers(0, 5, (M, T)).astype(float)
    qe = pd.DataFrame(dict(query_id=[f'q{i}' for i in range(M)], machine_type=['steady','balanced','jackpot','steady']*2, signal=[0]*M, machine_id=range(M)))
    return P, qe
def test_causal_future_invariant():
    P, qe = ep(); a = rf.feats(P, qe, 40); P2 = P.copy(); P2[:, 40:] = 4 - P2[:, 40:]
    assert np.array_equal(a, rf.feats(P2, qe, 40))  # incl. same-tick other machines
def test_cutoff_uses_last_tick():
    P, qe = ep(); P2 = P.copy(); P2[:, 39] = (P2[:, 39] + 1) % 5
    assert not np.array_equal(rf.feats(P, qe, 40), rf.feats(P2, qe, 40))
def test_missing_not_zero():
    P, qe = ep(); P[0, 39] = 0; a = rf.feats(P, qe, 40); P[0, 39] = np.nan; b = rf.feats(P, qe, 40)
    assert not np.array_equal(a[0], b[0]) and b[0][3 + int(rf.USE_SIGNAL)] == -1  # lag1 code -1 not class 0
def test_probabilities_and_alignment():
    P, qe = ep(); from sklearn.ensemble import RandomForestClassifier
    X = rf.feats(P, qe, 40); m = RandomForestClassifier(5, random_state=0).fit(X, [0,1,2,0,1,2,0,1])
    out = rf.predict(m, {'e': (qe, P)}, np.full(5, .2), 0.1, 0.8)
    assert list(out) == list(qe.query_id)
    for p in out.values(): assert p.shape == (5,) and np.isfinite(p).all() and (p >= 0).all() and abs(p.sum()-1) < 1e-9
def test_signal_broadcast_documented_and_removed():
    P, qe = ep(); q2 = qe.copy(); q2['signal'] = [1]*8
    rf.USE_SIGNAL = True
    assert not np.array_equal(rf.feats(P, qe, 30), rf.feats(P, q2, 30))  # original: final-query signal leaks into historical t=30
    rf.USE_SIGNAL = False
    try:
        a, b = rf.feats(P, qe, 30), rf.feats(P, q2, 30)
        assert np.array_equal(a, b) and a.shape[1] == 143 - 8
    finally: rf.USE_SIGNAL = True
