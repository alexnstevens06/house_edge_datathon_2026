"""Signal provenance / leak mutation tests."""
import os, sys, copy, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "source"))
import cnn
from house_edge.data import load_split
TR = load_split(os.path.join(cnn.PUB, "train"), with_labels=True)
SUB = copy.copy(TR); SUB.episodes = TR.episodes[:3]; SUB.episode_index = {e.episode_id: i for i, e in enumerate(SUB.episodes)}
SUB.queries = [r for r in TR.queries if r["episode_id"] in SUB.episode_index]
def flip(d):
    d2 = copy.copy(d); d2.queries = [dict(r, signal=str(-int(r["signal"]) if r["signal"] != "0" else 1)) for r in d.queries]; return d2

def test_original_signal_mode_leaks_query_signal_into_history():
    """Documents the ORIGINAL leak: cutoff-query signal reaches windows with t<cutoff."""
    cnn.USE_SIGNAL = True
    a = cnn.build_train(SUB, 64); b = cnn.build_train(flip(SUB), 64)
    hist = a[1] < torch.tensor([SUB.episodes[0].cutoff])  # rough: any t below first cutoff
    assert not torch.equal(a[3][hist], b[3][hist]), "expected leak in original mode"

def test_no_signal_mode_history_invariant_to_query_signal():
    cnn.USE_SIGNAL = False
    a = cnn.build_train(SUB, 64); b = cnn.build_train(flip(SUB), 64)
    for x, y in zip(a, b): assert torch.equal(x, y)
    assert int(a[3].abs().sum()) == 0

def test_no_signal_model_has_no_signal_params_and_ignores_sg():
    m = cnn.CNN(32, 2, 3, 0.0, 64, use_signal=False).eval()
    assert not any("sig" in n for n, _ in m.named_parameters())
    x = torch.randint(-1, 5, (2, 64, 8)).to(torch.int8); t = torch.tensor([100, 200]); ty = torch.zeros(2, 8, dtype=torch.long)
    with torch.no_grad(): assert torch.equal(m(x, t, ty, torch.zeros(2, 8, dtype=torch.long)), m(x, t, ty, torch.full((2, 8), 2)))

if __name__ == "__main__":
    f = 0
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            try: fn(); print("PASS", n)
            except Exception as e: f += 1; print("FAIL", n, repr(e))
    print("FAILURES", f); sys.exit(1 if f else 0)
