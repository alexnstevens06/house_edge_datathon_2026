import sys, json, time, os, numpy as np
from pathlib import Path
REPO = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(Path(__file__).parent))
from house_edge.data import load_split
from house_edge.scoring import load_official_validation, score_predictions, write_submission_csv
import linfeat as LF
K = REPO / "kit/public"; T0 = 16
def sig(d, e): q = {r["machine_id"]: int(r["signal"]) for r in d.queries if r["episode_id"] == e.episode_id}; return np.array([q[m] for m in e.machine_ids])
tr = load_split(K / "train", with_labels=True)
prior = np.ones((3, 5)) * 1e-3
for e in tr.episodes:
    for s in range(8):
        v = e.obs[:, s]; v = v[v >= 0]; prior[e.machine_types[s]] += np.bincount(v, minlength=5)
prior /= prior.sum(1, keepdims=True)
t0 = time.time(); Xs, ys = [], []
for e in tr.episodes:
    s = sig(tr, e); ticks = np.arange(T0, e.cutoff + 1)
    X = LF.episode_features(e.obs, e.machine_types, s, ticks, prior)
    lab = np.full((len(ticks), 8), -1)
    lab[:e.obs.shape[0] - T0] = e.obs[T0:]
    for r in tr.queries:
        if r["episode_id"] == e.episode_id: lab[-1, e.slot_of(r["machine_id"])] = tr.query_labels[r["query_id"]]
    keep = lab >= 0; Xs.append(X[keep]); ys.append(lab[keep])
X = np.concatenate(Xs); y = np.concatenate(ys); print("train", X.shape, time.time() - t0, flush=True)
def qpred(d, m):
    P = {}
    for e in d.episodes:
        Xq = LF.episode_features(e.obs, e.machine_types, sig(d, e), np.array([e.cutoff]), prior)[0]
        p = LF.predict(m, Xq)
        for i, mid in enumerate(e.machine_ids): P[mid + "-next"] = p[i].tolist()
    return {r["query_id"]: P[r["query_id"]] for r in d.queries}
va = load_split(K / "validation"); off = load_official_validation(K / "validation")
sp = json.load(open(REPO / "runs/splits.json")); tune, conf = set(sp["tune"]), set(sp["confirm"])
res = {}; best = None
for lam in [float(a) for a in sys.argv[3].split(",")]:
    t1 = time.time(); m = LF.fit(X, y, lam); P = qpred(va, m)
    st = score_predictions(P, off, tune); res[lam] = dict(tune_R=st["R"], fit_s=time.time() - t1)
    print(lam, res[lam], flush=True)
    if best is None or st["R"] < best[1]: best = (lam, st["R"], m, P)
lam, _, m, P = best
out = dict(lams=res, best_lam=lam, tune=score_predictions(P, off, tune), confirm=score_predictions(P, off, conf),
           full=score_predictions(P, off), n_train=int(len(y)), n_feat=int(X.shape[1]))
np.savez(OUT / "model.npz", prior=prior, **m)
write_submission_csv(OUT / "validation_predictions.csv", P)
te = load_split(K / "test"); t2 = time.time(); write_submission_csv(OUT / "test_submission.csv", qpred(te, m)); out["test_pred_s"] = time.time() - t2
json.dump(out, open(OUT / "metrics.json", "w"), indent=1); print(json.dumps({k: out[k] for k in ("best_lam","n_train")}), "tune", out["tune"]["R"], "conf", out["confirm"]["R"], "full", out["full"]["R"], out["full"]["loss_ratio_by_condition"])
