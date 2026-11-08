"""CPU checks for the convergence trainer. Uses a short slice of real train windows."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "source"))
import conv_train as C
import base_cnn as B
from house_edge.data import subset_split_data

torch.set_num_threads(2)


def _args(**kw):
    base = dict(lr=0.00197, wd=0.00895, bs=64, epochs=2, drop=0.3, h=32, layers=2, W=16,
                seed=0, pct_start=0.3, eval_every=1, out=None)
    base.update(kw)
    return argparse_namespace(base)


class argparse_namespace(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _windows():
    if getattr(_windows, "cache", None) is None:
        tr = B.load_split(os.path.join(B.PUB, "train"), with_labels=True)
        ids = [tr.episodes[0].episode_id, tr.episodes[1].episode_id]
        sub = subset_split_data(tr, ids)
        _windows.cache = C.subset_windows(*B.build_train(sub, 16), 128)
        va = B.load_split(os.path.join(B.PUB, "validation"))
        from house_edge.scoring import load_official_validation
        off = load_official_validation(os.path.join(B.PUB, "validation"))
        import json
        tune = set(json.load(open(os.path.join(B.REPO, "runs", "splits.json")))["tune"])
        _windows.va = va
        _windows.off = off
        _windows.tune = tune
    return _windows.cache, _windows.va, _windows.off, _windows.tune


def _weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def test_eval_hook_weights_bit_identical_and_train_mode():
    windows, va, off, tune = _windows()
    a = _args()
    m_eval, _, _ = C.run_cnn(a, max_train_windows=128, do_eval=True, dev="cpu",
                             windows=windows, va=va, off=off, tune_ids=tune)
    m_plain, _, _ = C.run_cnn(a, max_train_windows=128, do_eval=False, dev="cpu",
                              windows=windows, va=va, off=off, tune_ids=tune)
    w_eval, w_plain = _weights(m_eval), _weights(m_plain)
    assert w_eval.keys() == w_plain.keys()
    for k in w_eval:
        assert torch.equal(w_eval[k], w_plain[k]), k
    assert m_eval.training is True


def test_scheduler_total_steps_and_pct_start():
    windows, _, _, _ = _windows()
    n = len(windows[0])
    epochs, bs, lr, pct = 2, 64, 0.002, 0.5
    total = epochs * C.steps_per_epoch(n, bs)
    assert total == epochs * int(np.ceil(n / bs))
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=lr)
    sch = C.make_scheduler(opt, lr, total, pct)
    assert sch.total_steps == total
    assert sch._schedule_phases[0]["end_step"] == pct * total - 1
    assert abs(opt.param_groups[0]["lr"] - lr / 25) < 1e-12
    lrs = []
    for _ in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        p.grad = torch.zeros_like(p)
        opt.step()
        sch.step()
    peak = max(range(len(lrs)), key=lambda i: lrs[i])
    assert abs(peak - pct * total) <= 1


def test_masked_loss_ignores_missing_targets():
    torch.manual_seed(0)
    lg = torch.randn(2, 8, 5)
    y = torch.randint(0, 5, (2, 8))
    y[0, 0] = -1
    y[1, 3] = -1
    y_other = y.clone()
    y_other[0, 0] = -5
    y_other[1, 3] = -2
    assert torch.allclose(B.masked_loss(lg, y), B.masked_loss(lg, y_other))
    valid = y >= 0
    ref = F.cross_entropy(lg.float()[valid], y.long()[valid])
    assert torch.allclose(B.masked_loss(lg, y), ref)
    lg2 = lg.detach().clone().requires_grad_(True)
    B.masked_loss(lg2, y).backward()
    assert torch.count_nonzero(lg2.grad[y < 0]) == 0
    assert torch.count_nonzero(lg2.grad[y >= 0]) > 0
