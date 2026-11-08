"""Causal features + L2 multinomial logistic regression (torch CPU, LBFGS).
Features for predicting obs[t, m] use only obs[:t] (all machines). Missing != 0."""
from __future__ import annotations
import os, numpy as np, torch
NP, NM, LAGS, ALPHA, REC = 5, 8, 8, 4.0, 32
USE_SIGNAL = os.environ.get("LINEAR_NO_SIGNAL") != "1"

def onehot(obs):  # (L,8) int, -1 missing -> (L,8,6) channel 5 = missing
    L = obs.shape[0]; O = np.zeros((L, NM, 6), np.float32)
    idx = np.where(obs < 0, 5, obs)
    O[np.arange(L)[:, None], np.arange(NM)[None, :], idx] = 1
    return O

def episode_features(obs, types, signals, ticks, prior):
    """obs (L,8) (ticks >= L treated missing); ticks: array of target ticks t.
    Returns X (len(ticks),8,F) features for every machine at each t, built from obs[:t] only."""
    T = int(max(ticks)) + 1
    pad = np.full((max(T, obs.shape[0]) + LAGS, NM), -1, np.int8)
    pad[LAGS:LAGS + obs.shape[0]] = obs
    O = onehot(pad)  # row r <-> tick r-LAGS ; rows < LAGS are pre-history (missing)
    obsd = O[:, :, :5]
    cum = np.concatenate([np.zeros((1, NM, 5), np.float32), np.cumsum(obsd, 0)], 0)  # cum[r]=sum rows<r
    seen = O[:, :, 5] == 0
    last = np.full((O.shape[0] + 1, NM), -1e9); cur = np.full(NM, -1e9)
    for r in range(O.shape[0]):
        last[r] = cur; cur = np.where(seen[r], r, cur)
    last[O.shape[0]] = cur
    same = (types[:, None] == types[None, :]) & ~np.eye(NM, dtype=bool)
    oth = ~np.eye(NM, dtype=bool)
    out = []
    for t in ticks:
        r = t + LAGS; f = []
        for k in range(1, LAGS + 1):
            o = O[r - k]                      # (8,6) tick t-k, strictly before t
            f.append(o)
            f.append(oth.astype(np.float32) @ o / 7.0)
            ns = same.sum(1, keepdims=True)
            f.append(np.where(ns > 0, same.astype(np.float32) @ o / np.maximum(ns, 1), 0))
        c_all = cum[r]; c_rec = cum[r] - cum[max(r - REC, 0)]
        pr = prior[types]
        for c in (c_all, c_rec, oth.astype(np.float32) @ c_all / 7.0):
            n = c.sum(1, keepdims=True); f.append(np.log((c + ALPHA * pr) / (n + ALPHA)))
        n_own = c_all.sum(1, keepdims=True)
        gap = np.minimum(r - last[r], 300).astype(np.float32)[:, None]
        sig = [np.eye(3, dtype=np.float32)[signals + 1]] if USE_SIGNAL else []
        f += [np.log1p(n_own) / 6, np.log1p(gap) / 6, O[max(r - 40, 0):r, :, 5].mean(0)[:, None] if r > 0 else np.ones((NM, 1)),
              *sig, np.eye(3, dtype=np.float32)[types], np.ones((NM, 1), np.float32)]
        base = np.concatenate(f, 1).astype(np.float32)
        ty = np.eye(3, dtype=np.float32)[types]
        out.append(np.concatenate([base] + [base * ty[:, j:j + 1] for j in range(3)], 1))
    return np.stack(out)

def fit(X, y, lam, iters=200, threads=4, W0=None):
    torch.set_num_threads(threads)
    X = torch.from_numpy(X); y = torch.from_numpy(y).long()
    mu = X.mean(0); sd = X.std(0).clamp_min(1e-3); mu[-1] = 0; sd[-1] = 1
    Xs = (X - mu) / sd
    W = torch.zeros(X.shape[1], NP, requires_grad=True) if W0 is None else torch.tensor(W0, requires_grad=True)
    opt = torch.optim.LBFGS([W], max_iter=iters, line_search_fn="strong_wolfe", history_size=20)
    def closure():
        opt.zero_grad(); l = torch.nn.functional.cross_entropy(Xs @ W, y) + lam * (W[:-1] ** 2).sum()
        l.backward(); return l
    opt.step(closure)
    return dict(W=W.detach().numpy(), mu=mu.numpy(), sd=sd.numpy())

def predict(m, X):
    X = np.asarray(X, np.float64); z = ((X - m["mu"].astype(np.float64)) / m["sd"].astype(np.float64)) @ m["W"].astype(np.float64); z -= z.max(-1, keepdims=True)
    p = np.exp(z); return p / p.sum(-1, keepdims=True)
