"""Plain-runner tests for experiments/mlp_cnn_hpo MLP + base_cnn reuse. CPU only.
Usage: CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ../../.venv/bin/python tests/test_mlp.py"""
import os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.abspath(os.path.join(HERE, "..", "source"))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, SRC); sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, os.path.join(REPO, "kit"))
import base_cnn as B
B.USE_SIGNAL = False
import train as TR
from house_edge.data import load_split, episode_conditions, MISSING, N_MACHINES
from train import MLP

PUB = os.path.join(REPO, "kit", "public")
ALLOWED = {"Linear", "LayerNorm", "Dropout", "GELU", "Sequential", "ModuleList", "MLP"}
FAILS = []


def check(name, fn):
    try:
        fn(); print(f"PASS {name}", flush=True)
    except Exception as e:
        FAILS.append(name); print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)


def mkmodel():
    torch.manual_seed(0); m = MLP(width=64, depth=2, drop=0.1, W=64); m.eval(); return m


def rand_x(Bs, W, seed=1):
    g = torch.Generator().manual_seed(seed); x = torch.randint(-1, 5, (Bs, W, 8), generator=g, dtype=torch.int8)
    return x


def t_module_types():
    m = MLP(width=512, depth=4, drop=0.3, W=32)
    bad = [type(mod).__name__ for mod in m.modules() if mod is not m and type(mod).__name__ not in ALLOWED]
    assert not bad, f"disallowed modules: {bad}"
    assert not any("Conv" in type(mod).__name__ or "Embedding" in type(mod).__name__ for mod in m.modules())


def t_output_shape_softmax():
    m = mkmodel(); x = rand_x(5, 64)
    t = torch.tensor([100, 150, 200, 250, 256]); ty = torch.randint(0, 3, (5, 8)); sg = torch.randint(0, 3, (5, 8))
    out = m(x, t, ty, sg)
    assert out.shape == (5, 8, 5), out.shape
    p = torch.softmax(out.float(), -1)
    assert torch.isfinite(out).all() and (p.sum(-1) - 1).abs().max() < 1e-5


def short_ep():
    va = load_split(os.path.join(PUB, "validation"))
    cond = episode_conditions(os.path.join(PUB, "validation"))
    for ep in va.episodes:
        if cond[ep.episode_id] == "short_history": return ep
    raise AssertionError("no short_history episode in validation")


def t_causality():
    ep = short_ep(); m = mkmodel(); W, t = 64, ep.cutoff
    full = np.full((ep.cutoff + 1, N_MACHINES), MISSING, np.int8); full[:ep.obs.shape[0]] = ep.obs
    x0 = B.window(full, t, W)
    i0 = torch.tensor(x0[None]); o0 = m(i0, torch.tensor([t]), torch.tensor(ep.machine_types[None]), torch.zeros(1, 8))
    rng = np.random.default_rng(7)
    for slot in range(N_MACHINES):  # mutate ALL ticks >= t on this machine; features must be identical
        mut = full.copy(); mut[t:, slot] = rng.integers(0, 5, size=mut[t:, slot].shape).astype(np.int8)
        x1 = B.window(mut, t, W)
        assert np.array_equal(x0, x1), f"window changed by future ticks (slot {slot})"
        o1 = m(torch.tensor(x1[None]), torch.tensor([t]), torch.tensor(ep.machine_types[None]), torch.zeros(1, 8))
        assert torch.equal(o0, o1), f"output changed by ticks >= t (slot {slot})"


def t_missing_vs_payout0():
    ep = short_ep(); m = mkmodel(); W, t = 64, ep.cutoff
    full = np.full((ep.cutoff + 1, N_MACHINES), MISSING, np.int8); full[:ep.obs.shape[0]] = ep.obs
    x0 = B.window(full, t, W).copy(); args = (torch.tensor([t]), torch.tensor(ep.machine_types[None]), torch.zeros(1, 8))
    xa = x0.copy(); xa[-1, 0] = MISSING   # most recent tick unobserved
    xb = x0.copy(); xb[-1, 0] = 0         # most recent tick = payout 0
    oa = m(torch.tensor(xa[None]), *args); ob = m(torch.tensor(xb[None]), *args)
    assert not torch.equal(oa, ob), "missing encoded same as payout 0"


def t_signal_flip_invariant():
    va = load_split(os.path.join(PUB, "validation"))
    assert not B.USE_SIGNAL
    m = mkmodel(); W = 64
    Q0, ids0 = B.build_query(va, W)
    for r in va.queries:  # flip every signal value
        r["signal"] = "1" if r["signal"] != "1" else "-1"
    Q1, ids1 = B.build_query(va, W)
    for q0, q1 in zip(Q0, Q1): assert torch.equal(q0, q1), "build_query features changed by signal flip"
    with torch.no_grad():
        p0 = torch.softmax(m(Q0[0][:8], Q0[1][:8], Q0[2][:8], Q0[3][:8]).float(), -1)
        p1 = torch.softmax(m(Q1[0][:8], Q1[1][:8], Q1[2][:8], Q1[3][:8]).float(), -1)
    assert torch.equal(p0, p1), "outputs changed by signal flip"
    assert (Q0[3] == 0).all(), "signal tensor nonzero with USE_SIGNAL=False"


if __name__ == "__main__":
    for nm, fn in [("module_types", t_module_types), ("output_shape_softmax", t_output_shape_softmax),
                   ("causality_short_history", t_causality), ("missing_vs_payout0", t_missing_vs_payout0),
                   ("signal_flip_invariant", t_signal_flip_invariant)]:
        check(nm, fn)
    print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
