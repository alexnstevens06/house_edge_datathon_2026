import os, sys, time, json, csv, subprocess, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "source"))
import cnn
from clean_run import CLEAN_RUN
from house_edge.data import load_split, MISSING
torch.set_num_threads(2)
RUN = os.environ.get("CNN_RUN", os.path.join(HERE, "..", "runs", CLEAN_RUN))

class Skip(Exception):
    """Checkpoint/run artifacts unavailable on this machine (e.g. no runs/g_nosignal here)."""

def load():
    if not os.path.exists(os.path.join(RUN, "checkpoint.pt")):
        raise Skip(f"checkpoint not found: {RUN} (set CNN_RUN or run rerun.sh on the training machine)")
    ck = torch.load(os.path.join(RUN, "checkpoint.pt"), map_location="cpu"); c = ck["config"]
    ns = c.get("no_signal", False); cnn.USE_SIGNAL = not ns; m = cnn.CNN(c["h"], c["layers"], 3, c["drop"], c["W"], use_signal=not ns); m.load_state_dict(ck["state"]); m.eval(); return m, c

def test_causality_all_machines():
    m, c = load(); W = c["W"]; rng = np.random.default_rng(0)
    obs = rng.integers(0, 5, (200, 8)).astype(np.int8); t = 150
    base = torch.tensor(cnn.window(obs, t, W))[None]
    args = lambda x: (x, torch.tensor([t]), torch.zeros(1, 8, dtype=torch.long), torch.ones(1, 8, dtype=torch.long))
    p0 = m(*args(base))
    for s in range(8):  # perturb tick >= t on each machine: window must ignore it
        o2 = obs.copy(); o2[t:, s] = (o2[t:, s] + 1) % 5
        assert torch.equal(torch.tensor(cnn.window(o2, t, W))[None], base)
    # model-level: altering the last in-window tick changes output (sanity, inputs used) but internal causal conv: earlier positions unaffected by later
    x2 = base.clone(); x2[0, -1, 3] = (x2[0, -1, 3] + 1) % 5
    with torch.no_grad():
        z = lambda x: m.pay(torch.where(x.long().transpose(1,2) >= 0, x.long().transpose(1,2), 5))
    assert not torch.allclose(m(*args(x2)), p0)

def test_conv_is_causal_in_time():
    m, c = load(); W = c["W"]; z = torch.randn(1, 8, W, m.pay.embedding_dim)
    for b in m.blocks: b.eval()
    def run(z):
        for b in m.blocks: z = b(z)
        return z
    z2 = z.clone(); z2[:, :, W // 2:] += 5
    with torch.no_grad(): assert torch.allclose(run(z)[:, :, :W // 2], run(z2)[:, :, :W // 2], atol=1e-5)

def test_missing_differs_from_zero():
    m, c = load(); W = c["W"]; obs = np.full((100, 8), 1, np.int8); obs[60:90, 2] = MISSING
    o0 = obs.copy(); o0[60:90, 2] = 0; f = lambda o: m(torch.tensor(cnn.window(o, 100, W))[None], torch.tensor([100]), torch.zeros(1, 8, dtype=torch.long), torch.ones(1, 8, dtype=torch.long))
    with torch.no_grad(): assert not torch.allclose(f(obs), f(o0))

def test_reload_and_test_predictions_shape_alignment_timing():
    m, c = load(); te = load_split(os.path.join(cnn.PUB, "test")); t0 = time.time()
    if not os.path.exists(os.path.join(RUN, "test_predictions.csv")): raise Skip(f"no test_predictions.csv in {RUN}")
    Q, eps = cnn.build_query(te, c["W"]); P = cnn.predict(m, Q, "cpu"); dt = time.time() - t0
    assert P.shape == (len(te.episodes), 8, 5) and np.isfinite(P).all() and np.allclose(P.sum(-1), 1, atol=1e-5)
    pred = cnn.to_preds(P, eps, te); assert set(pred) == {r["query_id"] for r in te.queries}
    saved = {r["query_id"]: [float(r[k]) for k in ("p_0","p_1","p_2","p_5","p_20")] for r in csv.DictReader(open(os.path.join(RUN, "test_predictions.csv")))}
    assert set(saved) == set(pred)
    diff = max(abs(a - b) for q in pred for a, b in zip(pred[q], saved[q]))
    print(f"CPU2thread test predict {dt:.2f}s, max diff vs GPU bf16 csv {diff:.2e}")
    assert diff < 2e-2
    json.dump({"cpu2_test_predict_s": dt, "max_abs_diff_cpu_fp32_vs_gpu_bf16": diff}, open(os.path.join(RUN, "test_timing.json"), "w"))

if __name__ == "__main__":
    fails = skips = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS", n)
            except Skip as e: skips += 1; print("SKIP", n, e)
            except Exception as e: fails += 1; print("FAIL", n, repr(e))
    print("FAILURES", fails, "SKIPS", skips); sys.exit(1 if fails else 0)
