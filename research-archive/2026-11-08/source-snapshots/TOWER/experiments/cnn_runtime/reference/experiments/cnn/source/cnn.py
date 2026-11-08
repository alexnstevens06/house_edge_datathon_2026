"""Causal temporal CNN for House Edge (experiments/cnn). Own code path; imports shared
read-only modules src/house_edge/{data,scoring} and kit scoring (hashes recorded)."""
import argparse, json, os, sys, time, hashlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src")); sys.path.insert(0, os.path.join(REPO, "kit"))
from house_edge.data import load_split, MISSING, N_MACHINES, PAYOUTS
PUB = os.path.join(REPO, "kit", "public")
SIGS = {-1: 0, 0: 1, 1: 2}

USE_SIGNAL = True  # set False by --no_signal; then signal is never read

def episode_meta(data):
    """LEAK NOTE: signal comes from queries.csv (the cutoff query row) and was broadcast to
    every pseudo-cutoff window t<cutoff in the original runs. With USE_SIGNAL=False it is not read."""
    meta = {}
    if not USE_SIGNAL:
        return {r["episode_id"]: np.zeros(N_MACHINES, np.int64) for r in data.queries}
    for r in data.queries:
        ep = data.episodes[data.episode_index[r["episode_id"]]]
        s = ep.slot_of(r["machine_id"])
        meta.setdefault(r["episode_id"], np.zeros(N_MACHINES, np.int64))[s] = SIGS[int(r["signal"])]
    return meta

def window(obs, t, W):
    """Features use ONLY ticks < t on all machines. Returns (W,8) payout idx with MISSING pad."""
    x = np.full((W, N_MACHINES), MISSING, np.int8)
    lo = max(0, t - W); seg = obs[lo:min(t, obs.shape[0])]
    if len(seg): x[W - (t - lo):W - (t - lo) + len(seg)] = seg
    return x

def build_train(data, W, min_t=16, stride=1):
    meta = episode_meta(data); X, T, TY, SG, Y = [], [], [], [], []
    for ep in data.episodes:
        full = np.full((ep.cutoff + 1, N_MACHINES), MISSING, np.int8); full[:ep.obs.shape[0]] = ep.obs
        for r in data.queries:
            if r["episode_id"] == ep.episode_id: full[ep.cutoff, ep.slot_of(r["machine_id"])] = data.query_labels[r["query_id"]]
        for t in range(min_t, ep.cutoff + 1, stride):
            y = full[t]
            if (y >= 0).sum() == 0: continue
            X.append(window(full, t, W)); T.append(t); TY.append(ep.machine_types); SG.append(meta[ep.episode_id]); Y.append(y)
    return [torch.tensor(np.array(a)) for a in (X, T, TY, SG, Y)]

def build_query(data, W):
    meta = episode_meta(data); X, T, TY, SG, ids = [], [], [], [], []
    for ep in data.episodes:
        X.append(window(ep.obs, ep.cutoff, W)); T.append(ep.cutoff); TY.append(ep.machine_types); SG.append(meta[ep.episode_id]); ids.append(ep)
    return [torch.tensor(np.array(a)) for a in (X, T, TY, SG)], ids

class Block(nn.Module):
    def __init__(s, h, d, k, drop):
        super().__init__(); s.pad = (k - 1) * d
        s.conv = nn.Conv1d(h, h, k, dilation=d); s.mix = nn.Linear(2 * h, h); s.n1 = nn.LayerNorm(h); s.n2 = nn.LayerNorm(h); s.dr = nn.Dropout(drop)
    def forward(s, z):  # z: (B,8,W,H)
        B, M, W, H = z.shape
        c = s.conv(F.pad(z.reshape(B * M, W, H).transpose(1, 2), (s.pad, 0))).transpose(1, 2).reshape(B, M, W, H)  # causal over time, shared per machine
        z = s.n1(z + s.dr(F.gelu(c)))
        others = (z.sum(1, keepdim=True) - z) / (M - 1)  # cross-machine channel interaction, same time step only (already < t)
        return s.n2(z + s.dr(F.gelu(s.mix(torch.cat([z, others], -1)))))

class CNN(nn.Module):
    def __init__(s, h=64, layers=4, k=3, drop=0.1, W=64, use_signal=True):
        super().__init__(); s.W = W; s.use_signal = use_signal
        s.pay = nn.Embedding(6, h)  # 0..4 payouts, 5 = missing (plus explicit mask channel)
        s.mask = nn.Linear(1, h); s.typ = nn.Embedding(3, h); s.sig = nn.Embedding(3, h) if use_signal else None; s.slot = nn.Embedding(8, h)
        s.tick = nn.Linear(2, h)  # actual absolute tick (scaled) and recency
        s.blocks = nn.ModuleList([Block(h, 2 ** i, k, drop) for i in range(layers)])
        s.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 5))
    def forward(s, x, t, ty, sg):
        B, W, M = x.shape; x = x.long().transpose(1, 2)  # (B,8,W)
        m = (x >= 0).float().unsqueeze(-1); idx = torch.where(x >= 0, x, torch.full_like(x, 5))
        ticks = (t.view(B, 1, 1).float() - W + torch.arange(W, device=x.device).view(1, 1, W)).expand(B, M, W)
        tf = torch.stack([ticks / 256.0, (t.view(B, 1, 1).float() - ticks) / W], -1)
        meta = s.typ(ty) + s.slot.weight[None]
        if s.use_signal: meta = meta + s.sig(sg)
        z = s.pay(idx) + s.mask(m) + s.tick(tf) + meta.unsqueeze(2)
        for b in s.blocks: z = b(z)
        last = z[:, :, -1]; pooled = last.mean(1, keepdim=True).expand_as(last)
        return s.head(torch.cat([last, pooled], -1))  # (B,8,5) logits for tick t

def predict(model, Q, dev, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(Q[0]), bs):
            b = [a[i:i + bs].to(dev) for a in Q]
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                lg = model(*b)
            out.append(torch.softmax(lg.float(), -1).cpu())
    return torch.cat(out).double().numpy()

def masked_loss(lg, y):
    """Cross-entropy over labelled slots only (y >= 0); MISSING (-1) targets are masked out and
    contribute zero loss and zero gradient. Factored verbatim from main's inline loss
    (was: yl = y.long(); msk = yl >= 0; loss = F.cross_entropy(lg.float()[msk], yl[msk]))."""
    yl = y.long(); msk = yl >= 0
    return F.cross_entropy(lg.float()[msk], yl[msk])

def to_preds(P, eps, data):
    pred = {}
    for r in data.queries:
        i = [e.episode_id for e in eps].index(r["episode_id"]); ep = eps[i]
        pred[r["query_id"]] = P[i, ep.slot_of(r["machine_id"])].tolist()
    return pred

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--h", type=int, default=64); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--W", type=int, default=64); ap.add_argument("--drop", type=float, default=0.1); ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no_signal", action="store_true"); ap.add_argument("--out", required=True); a = ap.parse_args()
    global USE_SIGNAL; USE_SIGNAL = not a.no_signal
    torch.set_num_threads(4); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda": torch.cuda.set_per_process_memory_fraction(0.25)
    os.makedirs(a.out, exist_ok=True); json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)
    from house_edge.scoring import load_official_validation, score_predictions, write_submission_csv
    t0 = time.time(); tr = load_split(os.path.join(PUB, "train"), with_labels=True)
    va = load_split(os.path.join(PUB, "validation")); te = load_split(os.path.join(PUB, "test"))
    X, T, TY, SG, Y = build_train(tr, a.W); print("train windows", len(X), "build", time.time() - t0, flush=True)
    model = CNN(a.h, a.layers, 3, a.drop, a.W, use_signal=not a.no_signal).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((len(X) + a.bs - 1) // a.bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    D = [d.to(dev) for d in (X, T, TY, SG, Y)]
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(X), device=dev); tot = n = 0
        for i in range(0, len(X), a.bs):
            j = perm[i:i + a.bs]; x, t, ty, sg, y = (d[j] for d in D)
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                lg = model(x, t, ty, sg)
            loss = masked_loss(lg, y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            tot += loss.item() * len(j); n += len(j)
        print(f"epoch {ep} loss {tot / n:.4f} t {time.time() - t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "config": vars(a)}, os.path.join(a.out, "checkpoint.pt"))
    off = load_official_validation(os.path.join(PUB, "validation")); sp = json.load(open(os.path.join(REPO, "runs", "splits.json")))
    Qv, ev = build_query(va, a.W); pv = to_preds(predict(model, Qv, dev), ev, va)
    Qt, et = build_query(te, a.W); pt = to_preds(predict(model, Qt, dev), et, te)
    met = {k: score_predictions(pv, off, s) for k, s in (("full", None), ("tune", set(sp["tune"])), ("confirm", set(sp["confirm"])))}
    met["train_seconds"] = time.time() - t0; met["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0
    met["observed_target_masks"] = {"train_windows": int(len(X)), "labelled_slots": int((Y >= 0).sum()), "missing_slots_masked": int((Y < 0).sum())}
    json.dump(met, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    write_submission_csv(os.path.join(a.out, "validation_predictions.csv"), pv); write_submission_csv(os.path.join(a.out, "test_predictions.csv"), pt)
    print("RESULT", a.out, {k: round(met[k]["R"], 4) for k in ("full", "tune", "confirm")}, "gpuMB", round(met["peak_gpu_mb"]), flush=True)

if __name__ == "__main__": main()
