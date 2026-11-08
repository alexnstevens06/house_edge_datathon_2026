"""Signal-specialist random forests for House Edge.

Specialists: one forest per query signal (-1/0/+1), trained only on the 768
labelled train queries with signal-free causal features (rf.USE_SIGNAL=False).
The query signal exists only at the cutoff tick, so it never enters historical
feature rows; it is used solely to route each machine to its own signal's
forest. Shared_signal: one forest on all 768 rows with rf.USE_SIGNAL=True
(query-time signal at the cutoff tick only, which is legal for query rows).
"""
import argparse, json, os, pickle, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', 'source')))
import rf

SIGNALS = (-1, 0, 1)
GRID = dict(n_estimators=300, min_samples_leaf=(5, 20), max_features=(0.3, 0.6), alpha=(0.0, 0.2, 0.5))


def feats_with(P, qe, t, use_signal):
    old = rf.USE_SIGNAL
    rf.USE_SIGNAL = bool(use_signal)
    try:
        return rf.feats(P, qe, t)
    finally:
        rf.USE_SIGNAL = old


def build_train_rows(eps, labels, use_signal):
    """Rows for train queries only: X = feats(P, qe, cutoff), y = label class of the cutoff outcome."""
    lab = dict(zip(labels.query_id, labels.outcome))
    X, y, sig, typ, qid = [], [], [], [], []
    for e, (qe, P) in eps.items():
        F = feats_with(P, qe, P.shape[1], use_signal)
        for i in range(len(qe)):
            q = qe.query_id.iloc[i]
            X.append(F[i]); y.append(rf.CLS[lab[q]])
            sig.append(int(qe.signal.iloc[i])); typ.append(str(qe.machine_type.iloc[i])); qid.append(q)
    assert len(X) == len(labels) and set(qid) == set(labels.query_id)
    return np.array(X), np.array(y), np.array(sig), np.array(typ), qid


def episode_matrices(eps, use_signal):
    return {e: feats_with(P, qe, P.shape[1], use_signal) for e, (qe, P) in eps.items()}


def train_specialists(X, y, sig, n_jobs, n_estimators, min_samples_leaf, max_features, random_state=0):
    forests, priors = {}, {}
    for s in SIGNALS:
        m = sig == s
        if not m.any():
            forests[s], priors[s] = None, None
            continue
        forests[s] = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                            max_features=max_features, n_jobs=n_jobs,
                                            random_state=random_state).fit(X[m], y[m])
        priors[s] = np.bincount(y[m], minlength=5) / int(m.sum())
    return forests, priors


def route_predict(forests, priors, qe, F, alpha, prior_all):
    """Per-machine routing: each machine is scored by the forest of its own query signal."""
    M = len(qe); pr = np.zeros((M, 5)); sig = qe.signal.values.astype(int)
    for s in np.unique(sig):
        idx = np.flatnonzero(sig == s)
        f, prior = forests.get(int(s)), priors.get(int(s))
        if f is None:
            pr[idx] = prior_all if prior_all is not None else np.full(5, 0.2)
            continue
        row = np.zeros((len(idx), 5)); row[:, f.classes_] = f.predict_proba(F[idx])
        pr[idx] = (1 - alpha) * row + alpha * prior
    return pr / pr.sum(1, keepdims=True)


def predict_specialists(forests, priors, eps, F_by_ep, alpha, prior_all=None):
    out = {}
    for e, (qe, P) in eps.items():
        pr = route_predict(forests, priors, qe, F_by_ep[e], alpha, prior_all)
        for qid, p in zip(qe.query_id, pr):
            out[qid] = p
    return out


def train_shared(X, y, n_jobs, n_estimators, min_samples_leaf, max_features, random_state=0):
    model = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                   max_features=max_features, n_jobs=n_jobs,
                                   random_state=random_state).fit(X, y)
    return model, np.bincount(y, minlength=5) / len(y)


def predict_shared(model, eps, F_by_ep, prior, alpha):
    out = {}
    for e, (qe, P) in eps.items():
        pr = np.zeros((len(qe), 5)); pr[:, model.classes_] = model.predict_proba(F_by_ep[e])
        pr = (1 - alpha) * pr + alpha * prior
        pr /= pr.sum(1, keepdims=True)
        for qid, p in zip(qe.query_id, pr):
            out[qid] = p
    return out


def write_predictions(pred, path, order):
    ids = order['query_id'].tolist() if hasattr(order, 'columns') else list(order)
    rows = [[i] + [float(v) for v in pred[i]] for i in ids]
    pd.DataFrame(rows, columns=['query_id', 'p_0', 'p_1', 'p_2', 'p_5', 'p_20']).to_csv(path, index=False, float_format='%.12f')


def row_counts(sig, typ):
    c = {str(s): {t: 0 for t in rf.TYPES} for s in SIGNALS}
    for s, t in zip(sig, typ):
        c[str(int(s))][t] += 1
    return c


def evaluate(pred, tune, conf):
    allv = tune | conf
    Rt, pt = rf.official_R(pred, tune)
    Rc, pc = rf.official_R(pred, conf)
    Rf, pf = rf.official_R(pred, allv)
    return dict(R_tune=Rt, R_confirm=Rc, R_full=Rf, per_condition=dict(tune=pt, confirm=pc, full=pf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.abspath(os.path.join(HERE, '..', 'runs', 'spec_v1')))
    ap.add_argument('--n_jobs', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'specialists'), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'shared_signal'), exist_ok=True)
    t0 = time.time(); np.random.seed(0)
    qtr, tr = rf.load('train'); qva, va = rf.load('validation'); qte, te = rf.load('test')
    labels = pd.read_csv(os.path.join(rf.PUB, 'train', 'labels.csv'))
    assert len(qtr) == len(labels)
    Xa, ya, sig, typ, _ = build_train_rows(tr, labels, use_signal=False)
    Xb, yb, _, _, _ = build_train_rows(tr, labels, use_signal=True)
    prior_all = np.bincount(ya, minlength=5) / len(ya)
    print('train rows', Xa.shape, Xb.shape, 'by signal', {s: int((sig == s).sum()) for s in SIGNALS}, flush=True)
    sp = json.load(open(os.path.join(rf.ROOT, 'runs', 'splits.json')))
    tune, conf = set(sp['tune']), set(sp['confirm'])
    Fva_a = episode_matrices(va, False); Fte_a = episode_matrices(te, False)
    Fva_b = episode_matrices(va, True); Fte_b = episode_matrices(te, True)
    ne = GRID['n_estimators']; results = {'specialists': [], 'shared_signal': []}; best = {}
    for msl in GRID['min_samples_leaf']:
        for mf in GRID['max_features']:
            forests, priors = train_specialists(Xa, ya, sig, a.n_jobs, ne, msl, mf)
            model, prior_b = train_shared(Xb, yb, a.n_jobs, ne, msl, mf)
            for alpha in GRID['alpha']:
                pa = predict_specialists(forests, priors, va, Fva_a, alpha, prior_all)
                Ra = rf.official_R(pa, tune)[0]
                results['specialists'].append(dict(min_samples_leaf=msl, max_features=mf, alpha=alpha, R_tune=Ra))
                if 'specialists' not in best or Ra < best['specialists'][0]:
                    best['specialists'] = (Ra, dict(min_samples_leaf=msl, max_features=mf, alpha=alpha), forests, priors, pa)
                pb = predict_shared(model, va, Fva_b, prior_b, alpha)
                Rb = rf.official_R(pb, tune)[0]
                results['shared_signal'].append(dict(min_samples_leaf=msl, max_features=mf, alpha=alpha, R_tune=Rb))
                if 'shared_signal' not in best or Rb < best['shared_signal'][0]:
                    best['shared_signal'] = (Rb, dict(min_samples_leaf=msl, max_features=mf, alpha=alpha), model, prior_b, pb)
            json.dump(results, open(os.path.join(a.out, 'search.json'), 'w'), indent=1, default=float)
            print('grid done', msl, mf, round(time.time() - t0), flush=True)
    metrics = dict(n_jobs=a.n_jobs, n_estimators=ne, random_state=0, train_rows=int(len(ya)),
                   train_rows_by_signal={str(s): int((sig == s).sum()) for s in SIGNALS},
                   row_counts_signal_x_type=row_counts(sig, typ),
                   feature_cols=dict(specialists=int(Xa.shape[1]), shared_signal=int(Xb.shape[1])), elapsed_s=0.0)
    Ra, cfga, forests, priors, pva = best['specialists']
    metrics['specialists'] = dict(best=cfga, **evaluate(pva, tune, conf), prior_all=prior_all.tolist(),
                                  priors={str(s): (priors[s].tolist() if priors[s] is not None else None) for s in SIGNALS})
    pickle.dump(dict(forests=forests, priors=priors, prior_all=prior_all, cfg=cfga),
                open(os.path.join(a.out, 'specialists', 'model.pkl'), 'wb'))
    write_predictions(pva, os.path.join(a.out, 'specialists', 'validation_predictions.csv'), qva)
    t1 = time.time()
    pka = pickle.load(open(os.path.join(a.out, 'specialists', 'model.pkl'), 'rb'))
    pta = predict_specialists(pka['forests'], pka['priors'], te, Fte_a, pka['cfg']['alpha'], pka['prior_all'])
    write_predictions(pta, os.path.join(a.out, 'specialists', 'test_predictions.csv'), qte)
    metrics['specialists']['test_reload_predict_s'] = time.time() - t1
    Rb, cfgb, model, prior_b, pvb = best['shared_signal']
    metrics['shared_signal'] = dict(best=cfgb, **evaluate(pvb, tune, conf), prior=prior_b.tolist())
    pickle.dump(dict(model=model, prior=prior_b, cfg=cfgb), open(os.path.join(a.out, 'shared_signal', 'model.pkl'), 'wb'))
    write_predictions(pvb, os.path.join(a.out, 'shared_signal', 'validation_predictions.csv'), qva)
    t2 = time.time()
    pkb = pickle.load(open(os.path.join(a.out, 'shared_signal', 'model.pkl'), 'rb'))
    ptb = predict_shared(pkb['model'], te, Fte_b, pkb['prior'], pkb['cfg']['alpha'])
    write_predictions(ptb, os.path.join(a.out, 'shared_signal', 'test_predictions.csv'), qte)
    metrics['shared_signal']['test_reload_predict_s'] = time.time() - t2
    metrics['elapsed_s'] = time.time() - t0
    json.dump(metrics, open(os.path.join(a.out, 'metrics.json'), 'w'), indent=1, default=float)
    print('DONE', json.dumps({m: {k: metrics[m][k] for k in ('best', 'R_tune', 'R_confirm', 'R_full')}
                              for m in ('specialists', 'shared_signal')}), flush=True)


if __name__ == '__main__':
    main()
