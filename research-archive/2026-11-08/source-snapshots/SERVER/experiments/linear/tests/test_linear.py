import sys, os, importlib, numpy as np, csv, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source")); import linfeat as LF
rng = np.random.default_rng(0); pr = np.full((3, 5), .2)
obs = rng.integers(0, 5, (100, 8)).astype(np.int8); obs[rng.random((100, 8)) < .3] = -1
ty = np.array([0, 1, 2, 0, 1, 2, 0, 1]); sg = np.array([-1, 0, 1, 0, 0, 1, -1, 0])
for t in (20, 57, 99):
    a = LF.episode_features(obs, ty, sg, np.array([t]), pr)
    o2 = obs.copy(); o2[t:] = rng.integers(0, 5, o2[t:].shape)  # future + same tick, ALL machines
    assert np.array_equal(a, LF.episode_features(o2, ty, sg, np.array([t]), pr)), "causality"
    o3 = obs.copy(); o3[t-1, 3] = -1; o4 = obs.copy(); o4[t-1, 3] = 0
    assert not np.array_equal(LF.episode_features(o3, ty, sg, np.array([t]), pr), LF.episode_features(o4, ty, sg, np.array([t]), pr)), "missing!=0"
X = LF.episode_features(obs, ty, sg, np.arange(16, 100), pr); assert np.isfinite(X).all()
os.environ.pop("LINEAR_NO_SIGNAL", None); importlib.reload(LF); assert LF.USE_SIGNAL
Xs = LF.episode_features(obs, ty, sg, np.arange(16, 100), pr)
os.environ["LINEAR_NO_SIGNAL"] = "1"; importlib.reload(LF); assert not LF.USE_SIGNAL
Xn = LF.episode_features(obs, ty, sg, np.arange(16, 100), pr)
assert Xn.shape == Xs.shape[:2] + (Xs.shape[-1] - 12,), "no-signal width 3*4=12 smaller"
for s2 in (np.array([2, 2, 2, 2, 2, 2, 2, 2]), None):  # arbitrary mutation / garbage
    assert np.array_equal(Xn, LF.episode_features(obs, ty, s2, np.arange(16, 100), pr)), "signal ignored"
os.environ.pop("LINEAR_NO_SIGNAL"); importlib.reload(LF); assert LF.USE_SIGNAL
m = dict(W=rng.normal(size=(X.shape[-1], 5)), mu=np.zeros(X.shape[-1]), sd=np.ones(X.shape[-1]))
p = LF.predict(m, X); assert p.shape == X.shape[:2] + (5,) and np.allclose(p.sum(-1), 1) and np.isfinite(p).all()
print("unit OK")
if len(sys.argv) > 1:  # run dir + repo: checkpoint reload, ID alignment, timing
    run, repo = Path(sys.argv[1]), Path(sys.argv[2]); z = dict(np.load(run / "model.npz")); LF.USE_SIGNAL = z["W"].shape[0] == 676
    ids = [r["query_id"] for r in csv.DictReader(open(repo / "kit/public/test/queries.csv"))]
    sub = {r["query_id"]: [float(r[k]) for k in ("p_0","p_1","p_2","p_5","p_20")] for r in csv.DictReader(open(run / "test_submission.csv"))}
    assert sorted(ids) == sorted(sub) and len(ids) == len(set(ids))
    sys.path.insert(0, str(repo / "src")); from house_edge.data import load_split
    import torch; torch.set_num_threads(2)
    te = load_split(repo / "kit/public/test"); t0 = time.time(); mx = 0
    for e in te.episodes:
        s = {r["machine_id"]: int(r["signal"]) for r in te.queries if r["episode_id"] == e.episode_id}
        q = LF.predict(z, LF.episode_features(e.obs, e.machine_types, np.array([s[i] for i in e.machine_ids]), np.array([e.cutoff]), z["prior"])[0])
        for i, mid in enumerate(e.machine_ids): mx = max(mx, np.abs(q[i] / q[i].sum() - np.array(sub[mid + "-next"])).max())
    print("reload maxdiff", mx, "2-thread test inference s", round(time.time() - t0, 2), "queries", len(ids)); assert mx < 1e-8
