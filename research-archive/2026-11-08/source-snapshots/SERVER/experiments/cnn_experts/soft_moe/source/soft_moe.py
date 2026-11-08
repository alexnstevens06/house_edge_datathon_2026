"""Soft MoE on a CNN03 trunk. Historical windows never see query signal."""
import argparse, json, os, sys, time, hashlib
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "kit"))

from house_edge.data import load_split, MISSING, N_MACHINES, PAYOUTS, TYPE_TO_IDX

PUB = os.path.join(REPO, "kit", "public")
SPLITS_PATH = os.path.join(REPO, "runs", "splits.json")
BASE_CNN = os.path.join(REPO, "experiments", "mlp_cnn_hpo", "source", "base_cnn.py")

# Machine type and payout order are part of the frozen contract.
assert TYPE_TO_IDX == {"balanced": 0, "jackpot": 1, "steady": 2}
assert PAYOUTS == (0, 1, 2, 5, 20)
assert MISSING == -1

ABSENT = 3  # signal channel. Distinct from {-1,0,1} -> {0,1,2}. Never a payout.
SIGS = {-1: 0, 0: 1, 1: 2}
# Reference bar only. ADDENDUM t83: do not launch the other incumbent seeds.
INCUMBENT_TUNE_S0 = 0.33206398675779764
# Labelled TRAIN cutoff slots for signals -1, 0, +1 (index order SIGS).
EXPECTED_SIGNAL_COUNTS = (250, 267, 251)

PIPELINE = dict(
    h=128, layers=4, k=3, W=64, drop=0.3,
    lr=0.00197, wd=0.00895, bs=512,
    epochs=30, ft_epochs=10, ft_lr=0.000197,
    sparse_min=32, n_experts=3, expert_e=32, min_t=16, stride=1,
)


def reject_seed(seed):
    """ADDENDUM t83: one full run, seed 0 only. No seeds 1/2 and no promotion relaunch."""
    if int(seed) != 0:
        raise SystemExit(2)


def seed_all(seed):
    reject_seed(seed)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def window(obs, t, W):
    """Features use ONLY ticks < t on all machines. (W, 8) payout idx, MISSING pad."""
    x = np.full((W, N_MACHINES), MISSING, np.int8)
    lo = max(0, t - W)
    seg = obs[lo:min(t, obs.shape[0])]
    if len(seg):
        x[W - (t - lo):W - (t - lo) + len(seg)] = seg
    return x


def _queries_by_episode(data):
    grouped = defaultdict(list)
    for row in data.queries:
        grouped[row["episode_id"]].append(row)
    return grouped


def _cutoff_signal(ep, rows):
    """Legitimate per-machine signal at the true cutoff. Index 3 if a machine has no query."""
    sg = np.full(N_MACHINES, ABSENT, np.int64)
    for row in rows:
        sg[ep.slot_of(row["machine_id"])] = SIGS[int(row["signal"])]
    return sg


def build_train(data, W, min_t=16, stride=1):
    """CNN03 windows. Signal tensor is ABSENT whenever t < cutoff. No backfill."""
    if data.query_labels is None:
        raise RuntimeError("train query labels missing")
    grouped = _queries_by_episode(data)
    X, T, TY, SG, Y, C = [], [], [], [], [], []
    counts = np.zeros(3, np.int64)
    for ep in data.episodes:
        rows = grouped.get(ep.episode_id, [])
        full = np.full((ep.cutoff + 1, N_MACHINES), MISSING, np.int8)
        full[:ep.obs.shape[0]] = ep.obs
        cutoff_sg = _cutoff_signal(ep, rows)
        for row in rows:
            full[ep.cutoff, ep.slot_of(row["machine_id"])] = data.query_labels[row["query_id"]]
        for t in range(min_t, ep.cutoff + 1, stride):
            y = full[t]
            if (y >= 0).sum() == 0:
                continue
            X.append(window(full, t, W))
            T.append(t)
            TY.append(ep.machine_types)
            Y.append(y.copy())
            if t == ep.cutoff:
                SG.append(cutoff_sg.copy())
                C.append(True)
                for s in range(N_MACHINES):
                    if y[s] >= 0 and cutoff_sg[s] < 3:
                        counts[cutoff_sg[s]] += 1
            else:
                # Historical pseudo-cutoffs: signal channel stays ABSENT.
                SG.append(np.full(N_MACHINES, ABSENT, np.int64))
                C.append(False)
    pack = {
        "X": torch.tensor(np.array(X)),
        "T": torch.tensor(np.array(T)),
        "TY": torch.tensor(np.array(TY)),
        "SG": torch.tensor(np.array(SG)),
        "Y": torch.tensor(np.array(Y)),
        "C": torch.tensor(np.array(C), dtype=torch.bool),
        "signal_counts": counts,
    }
    return pack


def build_query(data, W):
    """True cutoff queries. Signal is the legitimate query signal for that row's machine."""
    grouped = _queries_by_episode(data)
    X, T, TY, SG, ids = [], [], [], [], []
    for ep in data.episodes:
        X.append(window(ep.obs, ep.cutoff, W))
        T.append(ep.cutoff)
        TY.append(ep.machine_types)
        SG.append(_cutoff_signal(ep, grouped.get(ep.episode_id, [])))
        ids.append(ep)
    pack = [torch.tensor(np.array(a)) for a in (X, T, TY, SG)]
    return pack, ids


def signal_for_phase(sg, is_cutoff, mode):
    """absent: every row is ABSENT. cutoff: keep stored signal only on true cutoff rows."""
    if mode == "absent":
        return torch.full_like(sg, ABSENT)
    if mode != "cutoff":
        raise ValueError(mode)
    blank = torch.full_like(sg, ABSENT)
    return torch.where(is_cutoff.unsqueeze(-1), sg, blank)


class Block(nn.Module):
    def __init__(self, h, d, k, drop):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(h, h, k, dilation=d)
        self.mix = nn.Linear(2 * h, h)
        self.n1 = nn.LayerNorm(h)
        self.n2 = nn.LayerNorm(h)
        self.dr = nn.Dropout(drop)

    def forward(self, z):  # (B, 8, W, H)
        B, M, W, H = z.shape
        c = self.conv(F.pad(z.reshape(B * M, W, H).transpose(1, 2), (self.pad, 0))).transpose(1, 2).reshape(B, M, W, H)
        z = self.n1(z + self.dr(F.gelu(c)))
        others = (z.sum(1, keepdim=True) - z) / (M - 1)
        return self.n2(z + self.dr(F.gelu(self.mix(torch.cat([z, others], -1)))))


class SharedCNN(nn.Module):
    """CNN03 trunk + single head. No signal parameters. sg is accepted and ignored."""

    def __init__(self, h=128, layers=4, k=3, drop=0.3, W=64):
        super().__init__()
        self.W = W
        self.h = h
        self.pay = nn.Embedding(6, h)
        self.mask = nn.Linear(1, h)
        self.typ = nn.Embedding(3, h)
        self.slot = nn.Embedding(8, h)
        self.tick = nn.Linear(2, h)
        self.blocks = nn.ModuleList([Block(h, 2 ** i, k, drop) for i in range(layers)])
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 5))

    def encode(self, x, t, ty):
        B, W, M = x.shape
        x = x.long().transpose(1, 2)  # (B, 8, W)
        m = (x >= 0).float().unsqueeze(-1)
        idx = torch.where(x >= 0, x, torch.full_like(x, 5))
        ticks = (t.view(B, 1, 1).float() - W + torch.arange(W, device=x.device).view(1, 1, W)).expand(B, M, W)
        tf = torch.stack([ticks / 256.0, (t.view(B, 1, 1).float() - ticks) / W], -1)
        meta = self.typ(ty) + self.slot.weight[None]
        z = self.pay(idx) + self.mask(m) + self.tick(tf) + meta.unsqueeze(2)
        for block in self.blocks:
            z = block(z)
        last = z[:, :, -1]
        pooled = last.mean(1, keepdim=True).expand_as(last)
        return z, torch.cat([last, pooled], -1)

    def forward(self, x, t, ty, sg=None):
        _ = sg  # matched pipeline passes the tensor; the control has no signal weights
        _, feat = self.encode(x, t, ty)
        return self.head(feat)

    def trunk_modules(self):
        return [self.pay, self.mask, self.typ, self.slot, self.tick, self.blocks]


class ExpertHead(nn.Module):
    """Small causal CNN residual. out.* is zero so the residual starts at 0."""

    def __init__(self, h, e=32, k=3, drop=0.3):
        super().__init__()
        self.pad = k - 1
        self.proj = nn.Linear(h, e)
        self.conv = nn.Conv1d(e, e, k)
        self.norm = nn.LayerNorm(e)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(e, 5)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        B, M, W, H = z.shape
        u = self.proj(z)
        flat = u.reshape(B * M, W, -1).transpose(1, 2)
        c = self.conv(F.pad(flat, (self.pad, 0))).transpose(1, 2).reshape(B, M, W, -1)
        last = self.norm(u[:, :, -1] + self.drop(F.gelu(c[:, :, -1])))
        return self.out(last)


class Gate(nn.Module):
    def __init__(self, h, n_exp=3):
        super().__init__()
        self.typ = nn.Embedding(3, 16)
        self.sig = nn.Embedding(4, 16)
        nn.init.zeros_(self.sig.weight)  # absent and real signals start identical
        self.mlp = nn.Sequential(nn.Linear(2 * h + 32, 64), nn.GELU(), nn.Linear(64, n_exp))

    def forward(self, feat, ty, sg):
        inp = torch.cat([feat, self.typ(ty), self.sig(sg)], -1)
        return torch.softmax(self.mlp(inp).float(), dim=-1)


class SoftMoE(nn.Module):
    def __init__(self, h=128, layers=4, k=3, drop=0.3, W=64, expert_e=32, n_experts=3):
        super().__init__()
        self.trunk = SharedCNN(h, layers, k, drop, W)
        self.experts = nn.ModuleList([ExpertHead(h, expert_e, k, drop) for _ in range(n_experts)])
        self.gate = Gate(h, n_experts)
        self.register_buffer("expert_active", torch.ones(n_experts, dtype=torch.bool))

    def set_active_from_counts(self, counts, sparse_min):
        counts = np.asarray(counts)
        active = torch.tensor([int(c) >= sparse_min for c in counts], dtype=torch.bool)
        if active.numel() != self.expert_active.numel():
            raise RuntimeError("signal group count does not match experts")
        self.expert_active.copy_(active)

    def mask_gate(self, g):
        active = self.expert_active
        if not bool(active.any()):
            return None
        g = g * active.to(dtype=g.dtype).view(*([1] * (g.dim() - 1)), -1)
        return g / g.sum(-1, keepdim=True).clamp_min(1e-8)

    def trunk_logits(self, x, t, ty):
        _, feat = self.trunk.encode(x, t, ty)
        return self.trunk.head(feat), feat

    def forward(self, x, t, ty, sg):
        z, feat = self.trunk.encode(x, t, ty)
        shared = self.trunk.head(feat)
        residuals = torch.stack([expert(z) for expert in self.experts], 0)  # (E,B,M,5)
        return blend_probs(shared, residuals, self.mask_gate(self.gate(feat, ty, sg)))


def blend_probs(shared, residuals, gate):
    """Convex blend of softmax(shared + residual_k). gate None => shared softmax only."""
    if gate is None:
        return torch.softmax(shared.float(), dim=-1)
    expert_probs = torch.softmax((shared.unsqueeze(0) + residuals).float(), dim=-1)
    return (gate.permute(2, 0, 1).unsqueeze(-1) * expert_probs).sum(0)


def masked_nll_logits(logits, y):
    yl = y.long()
    msk = yl >= 0
    return F.cross_entropy(logits.float()[msk], yl[msk])


def masked_nll_probs(probs, y):
    yl = y.long()
    msk = yl >= 0
    return F.nll_loss(probs.float()[msk].clamp_min(1e-8).log(), yl[msk])


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def set_phase(model, phase):
    for p in model.parameters():
        p.requires_grad_(phase == "H")
    if isinstance(model, SoftMoE):
        if phase == "C":
            for p in list(model.experts.parameters()) + list(model.gate.parameters()):
                p.requires_grad_(True)
            model.trunk.eval()
            model.experts.train()
            model.gate.train()
        else:
            model.train()
    else:
        if phase == "C":
            for p in model.head.parameters():
                p.requires_grad_(True)
            for mod in model.trunk_modules():
                mod.eval()
            model.head.train()
        else:
            model.train()


def run_phase(model, pack, idx, epochs, lr, wd, bs, dev, signal_mode, tag, t0):
    params = trainable(model)
    if not params:
        raise RuntimeError(f"{tag}: no trainable parameters")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    steps = epochs * ((len(idx) + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    X, T, TY, SG, Y, C = pack
    losses = []
    kind = "moe" if isinstance(model, SoftMoE) else "control"
    for ep in range(epochs):
        if kind == "moe" and signal_mode == "cutoff":
            model.trunk.eval(); model.experts.train(); model.gate.train()
        elif kind == "control" and signal_mode == "cutoff":
            for mod in model.trunk_modules():
                mod.eval()
            model.head.train()
        else:
            model.train()
        perm = idx[torch.randperm(len(idx), device=dev)]
        tot = n = 0.0
        for i in range(0, len(idx), bs):
            j = perm[i:i + bs]
            sg = signal_for_phase(SG[j], C[j], signal_mode)
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                out = model(X[j], T[j], TY[j], sg)
            loss = masked_nll_probs(out, Y[j]) if kind == "moe" else masked_nll_logits(out, Y[j])
            if not torch.isfinite(loss):
                raise RuntimeError(f"{tag} non-finite loss at epoch {ep}")
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sch.step()
            tot += float(loss.item()) * len(j); n += len(j)
        losses.append(tot / n)
        print(f"{tag} epoch {ep} loss {tot / n:.4f} t {time.time() - t0:.0f}s", flush=True)
    return losses


def predict(model, Q, dev, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Q[0]), bs):
            batch = [a[i:i + bs].to(dev) for a in Q]
            pred = model(*batch)
            if not isinstance(model, SoftMoE):
                pred = torch.softmax(pred.float(), dim=-1)
            out.append(pred.float().cpu())
    return torch.cat(out).double().numpy()


def to_preds(P, eps, data):
    pred = {}
    ids = [e.episode_id for e in eps]
    for row in data.queries:
        i = ids.index(row["episode_id"])
        ep = eps[i]
        pred[row["query_id"]] = P[i, ep.slot_of(row["machine_id"])].tolist()
    return pred


def _load_ids():
    ids = json.load(open(os.path.join(ROOT, "SPLIT_IDS.json")))
    sp = json.load(open(SPLITS_PATH))
    if list(ids["tune"]) != list(sp["tune"]) or list(ids["confirm"]) != list(sp["confirm"]):
        raise RuntimeError("SPLIT_IDS.json does not match runs/splits.json")
    return ids


def _vram_cap():
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    # Contract: this job <= 2 GiB.
    frac = min(0.95, (2 * 1024 ** 3) / total)
    torch.cuda.set_per_process_memory_fraction(frac)


def _save_model(model, kind, seed, signal_counts, out_dir, extra):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state": model.state_dict(), "kind": kind, "config": dict(PIPELINE), "seed": seed,
                "signal_counts": list(signal_counts), **extra}, os.path.join(out_dir, "checkpoint.pt"))


def pretrain_shared(pack, dev, seed, t0):
    """One CNN03-schedule trunk. MoE and control both start from this state."""
    p = PIPELINE
    seed_all(seed)
    model = SharedCNN(p["h"], p["layers"], p["k"], p["drop"], p["W"]).to(dev)
    n_params = sum(x.numel() for x in model.parameters())
    print(f"shared pretrain params {n_params}", flush=True)
    gpu_pack = {k: pack[k].to(dev) if torch.is_tensor(pack[k]) else pack[k] for k in ("X", "T", "TY", "SG", "Y", "C")}
    idx_all = torch.arange(len(pack["X"]), device=dev)
    set_phase(model, "H")
    losses = run_phase(model, gpu_pack, idx_all, p["epochs"], p["lr"], p["wd"], p["bs"], dev, "absent", "shared/H", t0)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model.cpu(), state, n_params, losses, gpu_pack, idx_all


def finetune_from_shared(kind, state, pack, gpu_pack, idx_all, dev, seed, t0):
    """Same cutoff rows, step count, lr, wd, and seed for the MoE and the control."""
    p = PIPELINE
    seed_all(seed)
    if kind == "moe":
        model = SoftMoE(p["h"], p["layers"], p["k"], p["drop"], p["W"], p["expert_e"], p["n_experts"]).to(dev)
        model.trunk.load_state_dict(state)
        model.set_active_from_counts(pack["signal_counts"], p["sparse_min"])
    elif kind == "control":
        model = SharedCNN(p["h"], p["layers"], p["k"], p["drop"], p["W"]).to(dev)
        model.load_state_dict(state)
    else:
        raise RuntimeError(kind)
    n_params = sum(x.numel() for x in model.parameters())
    idx_cut = idx_all[gpu_pack["C"]]
    steps = p["ft_epochs"] * ((len(idx_cut) + p["bs"] - 1) // p["bs"])
    print(f"{kind} params {n_params} cutoff rows {len(idx_cut)} ft_steps {steps}", flush=True)
    set_phase(model, "C")
    losses = run_phase(model, gpu_pack, idx_cut, p["ft_epochs"], p["ft_lr"], p["wd"], p["bs"], dev, "cutoff", f"{kind}/C", t0)
    model_cpu = model.to("cpu").eval()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return model_cpu, n_params, losses, int(steps)


def evaluate_tune(model, va, out_dir):
    """Tune-split R only. Confirm and test are not scored and not used for selection."""
    from house_edge.scoring import load_official_validation, score_predictions, write_submission_csv
    off = load_official_validation(os.path.join(PUB, "validation"))
    ids = _load_ids()
    Qv, ev = build_query(va, PIPELINE["W"])
    pv = to_preds(predict(model, Qv, "cpu"), ev, va)
    tune = score_predictions(pv, off, set(ids["tune"]))
    write_submission_csv(os.path.join(out_dir, "validation_predictions_cpu.csv"), pv)
    return tune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    reject_seed(a.seed)  # before CUDA, files, or any second-seed path
    torch.set_num_threads(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", dev, "cuda_visible", os.environ.get("CUDA_VISIBLE_DEVICES"), "bs", PIPELINE["bs"], flush=True)
    if dev != "cuda":
        open(os.path.join(ROOT, "BLOCKER.md"), "w").write(
            "CUDA is not visible in the training process. Refusing to train seed 0 on CPU.\n")
        raise SystemExit("CUDA unavailable")
    _vram_cap()
    t0 = time.time()
    tr = load_split(os.path.join(PUB, "train"), with_labels=True)
    if not tr.query_labels:
        open(os.path.join(ROOT, "BLOCKER.md"), "w").write("Train query labels are missing. Stop.\n")
        raise SystemExit("no train labels")
    va = load_split(os.path.join(PUB, "validation"))
    pack = build_train(tr, PIPELINE["W"], PIPELINE["min_t"], PIPELINE["stride"])
    counts = tuple(int(c) for c in pack["signal_counts"].tolist())
    if counts != EXPECTED_SIGNAL_COUNTS:
        open(os.path.join(ROOT, "BLOCKER.md"), "w").write(
            f"TRAIN signal counts {counts} != {EXPECTED_SIGNAL_COUNTS}. Stop.\n")
        raise SystemExit("signal count mismatch")
    hist = ~pack["C"]
    if int((pack["SG"][hist] != ABSENT).sum()) != 0:
        raise RuntimeError("historical windows contain a signal")
    print("train windows", len(pack["X"]), "cutoff", int(pack["C"].sum()),
          "signal_counts", counts, "build", round(time.time() - t0, 3), flush=True)
    run = os.path.join(ROOT, "runs", "seed0")
    shared, state, n_shared, losses_h, gpu_pack, idx_all = pretrain_shared(pack, dev, a.seed, t0)
    _save_model(shared, "shared_pretrain", a.seed, counts, os.path.join(run, "shared_pretrain"),
                {"losses_H": losses_h})
    results = {"seed": 0, "pipeline": dict(PIPELINE), "signal_counts": counts,
               "base_cnn_sha256": sha256_file(BASE_CNN), "plan_sha256": sha256_file(os.path.join(ROOT, "PLAN.md")),
               "source_sha256": sha256_file(os.path.abspath(__file__)),
               "incumbent_tune_R_s0": INCUMBENT_TUNE_S0,
               "selection_metric": "tune_R_only",
               "shared_pretrain": {"params": n_shared, "losses_H": losses_h}}
    # One finetune each. No further seeds. Control uses the same pretrain state and step budget.
    for kind in ("moe", "control"):
        model, n_params, losses_c, steps = finetune_from_shared(kind, state, pack, gpu_pack, idx_all, dev, a.seed, t0)
        out_dir = os.path.join(run, kind)
        _save_model(model, kind, a.seed, counts, out_dir, {"losses_C": losses_c, "ft_steps": steps})
        tune = evaluate_tune(model, va, out_dir)
        rec = {"params": n_params, "losses_H_shared": losses_h, "losses_C": losses_c, "ft_steps": steps,
               "tune_R": tune["R"], "tune": tune, "seconds": time.time() - t0}
        results[kind] = rec
        json.dump(rec, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
        print(f"RESULT {kind} tune_R {rec['tune_R']:.6f}", flush=True)
        del model
    results["seconds"] = time.time() - t0
    results["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    results["verdict_note"] = "Stop after seed 0. Do not launch seeds 1 or 2."
    os.makedirs(run, exist_ok=True)
    json.dump(results, open(os.path.join(run, "summary.json"), "w"), indent=1)
    print("DONE", run, "peakMB", round(results["peak_gpu_mb"], 1), f"{results['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        open(os.path.join(ROOT, "BLOCKER.md"), "w").write(
            "CUDA OOM under the 2 GiB cap. ADDENDUM t83: do not relaunch at another batch size.\n")
        raise
