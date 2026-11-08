"""Probabilistic random forest for House Edge. Causal features only (ticks < t)."""
import argparse, json, os, sys, time, pickle, hashlib
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
KIT = os.path.join(ROOT, 'kit'); PUB = os.path.join(KIT, 'public')
PAY = np.array([0, 1, 2, 5, 20]); CLS = {v: i for i, v in enumerate(PAY)}
TYPES = ['steady', 'balanced', 'jackpot']; L = 8

def load(split):
    q = pd.read_csv(f'{PUB}/{split}/queries.csv'); h = pd.read_csv(f'{PUB}/{split}/history.csv')
    eps = {}
    for e, qe in q.groupby('episode_id', sort=True):
        qe = qe.sort_values('machine_id').reset_index(drop=True)
        cut = int(qe.cutoff.iloc[0]); M = len(qe)
        P = np.full((M, cut), np.nan)
        he = h[h.episode_id == e]
        idx = {m: i for i, m in enumerate(qe.machine_id)}
        for m, t, p in zip(he.machine_id, he.tick, he.payout):
            if t >= cut: raise ValueError(f'history at/after cutoff {e}')
            P[idx[m], t] = CLS[p]
        eps[e] = (qe, P)
    return q, eps

def feats(P, qe, t):
    """Features for predicting tick t for every machine; uses only P[:, :t]."""
    M = P.shape[0]; H = P[:, :t]
    typ = np.array([[x == k for k in TYPES] for x in qe.machine_type], float)
    sig = qe.signal.values.astype(float)
    lag = np.full((M, L), -1.0); obs = np.zeros((M, L))
    for k in range(1, L + 1):
        if t - k >= 0:
            v = H[:, t - k]; o = ~np.isnan(v); lag[o, k - 1] = v[o]; obs[:, k - 1] = o
    stats = []
    for w in (8, 32, 10**6):
        W = H[:, max(0, t - w):]; n = (~np.isnan(W)).sum(1)
        cnt = np.stack([(W == c).sum(1) for c in range(5)], 1)
        stats.append(np.column_stack([n, (cnt + 1) / (n[:, None] + 5)]))
    stats = np.concatenate(stats, 1)
    last = np.array([t - (np.flatnonzero(~np.isnan(r))[-1]) if (~np.isnan(r)).any() else 999 for r in H], float)
    nobs = (~np.isnan(H)).sum(1)
    own = np.column_stack([typ, sig, lag, obs, stats, last, nobs, np.full(M, t)])
    # episode-level: other machines' lags (ordered relative to self) + episode mean recent stats
    rows = []
    epi = stats.mean(0)
    for i in range(M):
        order = [(i + j) % M for j in range(1, M)]
        oth = np.concatenate([np.concatenate([typ[j], [sig[j]], lag[j]]) for j in order])
        rows.append(np.concatenate([own[i], oth, epi]))
    return np.array(rows)

def train_rows(eps, min_t=24):
    X, y, g = [], [], []
    for e, (qe, P) in eps.items():
        cut = P.shape[1]
        for t in range(min_t, cut):
            lab = P[:, t]; ok = ~np.isnan(lab)
            if not ok.any(): continue
            F = feats(P, qe, t); X.append(F[ok]); y.append(lab[ok].astype(int)); g += [e] * ok.sum()
    return np.vstack(X), np.concatenate(y), np.array(g)

def predict(model, eps, prior, alpha, temp):
    out = {}
    for e, (qe, P) in eps.items():
        F = feats(P, qe, P.shape[1]); pr = np.zeros((len(qe), 5))
        pr[:, model.classes_] = model.predict_proba(F)
        if temp != 1: pr = pr ** temp; pr /= pr.sum(1, keepdims=True)
        pr = (1 - alpha) * pr + alpha * prior
        pr /= pr.sum(1, keepdims=True)
        for qid, p in zip(qe.query_id, pr): out[qid] = p
    return out

def official_R(pred, ids_eps):
    tg = pd.read_csv(f'{PUB}/validation/targets.csv'); ref = pd.read_csv(f'{PUB}/validation/reference.csv')
    cond = pd.read_csv(f'{PUB}/validation/conditions.csv')
    df = tg.merge(cond, on='query_id').merge(ref, on='query_id', suffixes=('', '_r'))
    df = df[df.episode_id.isin(ids_eps)]
    cols = ['p_0', 'p_1', 'p_2', 'p_5', 'p_20']
    Q = df[cols].values; B = df[[c + '_r' for c in cols]].values
    Pm = np.array([pred[q] for q in df.query_id])
    df = df.assign(e=((Pm - Q) ** 2).sum(1), b=((B - Q) ** 2).sum(1))
    per = {}
    for c, d in df.groupby('condition'):
        E = d.groupby('episode_id').e.mean().mean(); Bc = d.groupby('episode_id').b.mean().mean()
        per[c] = {'E': E, 'B': Bc, 'ratio': E / max(Bc, 1e-9)}
    return float(np.mean([v['ratio'] for v in per.values()])), per

def write_csv(pred, path, split):
    q = pd.read_csv(f'{PUB}/{split}/queries.csv')
    rows = [[qid] + list(pred[qid]) for qid in q.query_id]
    pd.DataFrame(rows, columns=['query_id', 'p_0', 'p_1', 'p_2', 'p_5', 'p_20']).to_csv(path, index=False, float_format='%.12f')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '../runs/v1'))
    ap.add_argument('--n_jobs', type=int, default=8); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    np.random.seed(0); t0 = time.time()
    _, tr = load('train'); _, va = load('validation'); _, te = load('test')
    X, y, g = train_rows(tr); print('train rows', X.shape, time.time() - t0, flush=True)
    prior = np.bincount(y, minlength=5) / len(y)
    sp = json.load(open(os.path.join(ROOT, 'runs/splits.json')))
    tune, conf = set(sp['tune']), set(sp['confirm']); allv = tune | conf
    results = []; best = None
    for msl in (20, 50, 150):
        for mf in (0.2, 0.5):
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=msl, max_features=mf, n_jobs=a.n_jobs, random_state=0).fit(X, y)
            for alpha in (0.0, 0.1, 0.25):
                for temp in (1.0, 0.8):
                    p = predict(m, va, prior, alpha, temp); R, _ = official_R(p, tune)
                    r = dict(min_samples_leaf=msl, max_features=mf, alpha=alpha, temp=temp, R_tune=R); results.append(r)
                    print(r, round(time.time() - t0), flush=True)
                    if best is None or R < best[0]: best = (R, r, m)
            json.dump(results, open(f'{a.out}/search.json', 'w'), indent=1)
    R, cfg, m = best
    pv = predict(m, va, prior, cfg['alpha'], cfg['temp'])
    Rt, pt = official_R(pv, tune); Rc, pc = official_R(pv, conf); Rf, pf = official_R(pv, allv)
    metrics = dict(best=cfg, R_tune=Rt, R_confirm=Rc, R_full=Rf, per_condition=dict(tune=pt, confirm=pc, full=pf), train_rows=int(len(y)), prior=prior.tolist(), elapsed_s=time.time() - t0)
    json.dump(metrics, open(f'{a.out}/metrics.json', 'w'), indent=1, default=float)
    pickle.dump(dict(model=m, prior=prior, cfg=cfg), open(f'{a.out}/model.pkl', 'wb'))
    write_csv(pv, f'{a.out}/validation_predictions.csv', 'validation')
    t1 = time.time(); m2 = pickle.load(open(f'{a.out}/model.pkl', 'rb'))
    ptst = predict(m2['model'], te, m2['prior'], cfg['alpha'], cfg['temp'])
    write_csv(ptst, f'{a.out}/test_predictions.csv', 'test')
    metrics['test_reload_predict_s'] = time.time() - t1
    json.dump(metrics, open(f'{a.out}/metrics.json', 'w'), indent=1, default=float)
    print('DONE', json.dumps({k: metrics[k] for k in ('best', 'R_tune', 'R_confirm', 'R_full', 'test_reload_predict_s')}), flush=True)
