"""Small causal CNN for House Edge (experiments/small_cnn_data).

Clean reimplementation of the experiments/cnn idea (reference R=0.4031, no signal):
- causal dilated Conv1d over time per machine, shared weights across machines
- cross-machine mixing of OTHER machines' features at the same (earlier) timestep
- payout embedding with explicit MISSING symbol + mask channel
- machine type + slot embeddings, absolute-tick + recency features
- readout from each machine's own last position + pooled mean
- (8,5) softmax; cross-entropy on observed labels only

NO SIGNAL anywhere: the queries.csv signal column is never read for features.
Inputs use ONLY ticks < target tick for ALL machines; missing ticks stay MISSING
(never 0). Train inputs use only train-split observations + train query labels as
targets; validation/test labels are never read.

Regimes (--regime), identical seed 0 / optimizer / total optimizer steps
(fixed by the base window count; only sampling/weighting/masking differs):
  base            uniform sliding windows over all train ticks (t>=16) + query cutoff
  episode_balanced sample windows so each train episode contributes equally
  window32        W=32 (shorter history coverage)
  query_upweight  query-cutoff examples weighted x8 in the loss (others 1)
  min_obs         label-mask targets of machines with <8 observed ticks in their
                  window; drop windows left with no valid target (label masks only,
                  never fabricate)
Selection: best epoch by TUNE R only (runs/splits.json); confirm reported, never tuned.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "kit"))
from house_edge.data import load_split, MISSING, N_MACHINES, PAYOUTS  # noqa: E402

PUB = os.path.join(REPO, "kit", "public")
MIN_OBS = 8      # regime D: minimum observed ticks for a machine's target to count
QUERY_UPWEIGHT = 8.0
MIN_T = 16


def window(obs, t, W):
    """(W,8) payout indices using ONLY ticks < t on ALL machines; MISSING padding."""
    x = np.full((W, N_MACHINES), MISSING, np.int8)
    lo = max(0, t - W)
    seg = obs[lo:min(t, obs.shape[0])]
    if len(seg):
        x[W - (t - lo):W - (t - lo) + len(seg)] = seg
    return x


def build_train(data, W, min_t=MIN_T):
    """Sliding windows over train ticks t in [min_t, cutoff]; y = payouts at tick t
    (row `cutoff` = train query labels, targets only). Labels never enter X."""
    per_ep = {}
    if data.query_labels is not None:
        for r in data.queries:
            per_ep.setdefault(r["episode_id"], []).append(r)
    X, T, TY, Y, EP, ISQ, OBC = [], [], [], [], [], [], []
    for ei, ep in enumerate(data.episodes):
        full = np.full((ep.cutoff + 1, N_MACHINES), MISSING, np.int8)
        full[:ep.obs.shape[0]] = ep.obs
        for r in per_ep.get(ep.episode_id, []):
            full[ep.cutoff, ep.slot_of(r["machine_id"])] = data.query_labels[r["query_id"]]
        cum = np.zeros((full.shape[0] + 1, N_MACHINES), np.int64)
        np.cumsum(full >= 0, axis=0, out=cum[1:])
        for t in range(min_t, ep.cutoff + 1):
            y = full[t]
            if not (y >= 0).any():
                continue
            lo = max(0, t - W)
            X.append(window(full, t, W))
            T.append(t)
            TY.append(ep.machine_types)
            Y.append(y)
            EP.append(ei)
            ISQ.append(t == ep.cutoff)
            OBC.append(cum[t] - cum[lo])
    return {"X": np.array(X, np.int8), "T": np.array(T, np.int64),
            "TY": np.array(TY, np.int64), "Y": np.array(Y, np.int8),
            "EP": np.array(EP, np.int64), "ISQ": np.array(ISQ, bool),
            "OBC": np.array(OBC, np.int64)}


def build_query(data, W):
    """Inference features per episode: window over ticks < cutoff from history only.
    Never reads labels.csv outcomes nor the queries.csv signal column."""
    X, T, TY, ids = [], [], [], []
    for ep in data.episodes:
        X.append(window(ep.obs, ep.cutoff, W))
        T.append(ep.cutoff)
        TY.append(ep.machine_types)
        ids.append(ep)
    return {"X": np.array(X, np.int8), "T": np.array(T, np.int64),
            "TY": np.array(TY, np.int64)}, ids


class Block(nn.Module):
    def __init__(s, h, d, k, drop):
        super().__init__()
        s.pad = (k - 1) * d
        s.conv = nn.Conv1d(h, h, k, dilation=d)
        s.mix = nn.Linear(2 * h, h)
        s.n1 = nn.LayerNorm(h)
        s.n2 = nn.LayerNorm(h)
        s.dr = nn.Dropout(drop)

    def forward(s, z):  # z: (B,M,W,H)
        B, M, W, H = z.shape
        c = s.conv(F.pad(z.reshape(B * M, W, H).transpose(1, 2), (s.pad, 0))).transpose(1, 2).reshape(B, M, W, H)
        z = s.n1(z + s.dr(F.gelu(c)))  # causal over time, weights shared across machines
        others = (z.sum(1, keepdim=True) - z) / (M - 1)  # other machines, same (earlier) tick
        return s.n2(z + s.dr(F.gelu(s.mix(torch.cat([z, others], -1)))))


class CNN(nn.Module):
    def __init__(s, h=32, layers=3, k=3, drop=0.1, W=64):
        super().__init__()
        s.W = W
        s.pay = nn.Embedding(6, h)   # payout idx 0..4 (PAYOUTS order), 5 = MISSING symbol
        s.mask = nn.Linear(1, h)     # explicit observed/missing channel
        s.typ = nn.Embedding(3, h)
        s.slot = nn.Embedding(8, h)
        s.tick = nn.Linear(2, h)     # absolute tick (scaled) + recency within window
        s.blocks = nn.ModuleList([Block(h, 2 ** i, k, drop) for i in range(layers)])
        s.head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 5))

    def forward(s, x, t, ty):
        B, W, M = x.shape
        x = x.long().transpose(1, 2)  # (B,8,W)
        m = (x >= 0).float().unsqueeze(-1)
        idx = torch.where(x >= 0, x, torch.full_like(x, 5))
        ticks = (t.view(B, 1, 1).float() - W + torch.arange(W, device=x.device).view(1, 1, W)).expand(B, M, W)
        tf = torch.stack([ticks / 256.0, (t.view(B, 1, 1).float() - ticks) / W], -1)
        meta = s.typ(ty) + s.slot.weight[None]
        z = s.pay(idx) + s.mask(m) + s.tick(tf) + meta.unsqueeze(2)
        for b in s.blocks:
            z = b(z)
        last = z[:, :, -1]
        pooled = last.mean(1, keepdim=True).expand_as(last)
        return s.head(torch.cat([last, pooled], -1))  # (B,8,5) logits at tick t


def masked_loss(lg, y, w=None, obs_mask=None):
    """Cross-entropy over observed labels only. MISSING (-1) targets are masked out:
    zero loss, zero gradient, and (regime C) their per-example weight is not counted.
    obs_mask (regime D) additionally masks gappy machine targets; never fabricates."""
    yl = y.long()
    msk = yl >= 0
    if obs_mask is not None:
        msk = msk & obs_mask
    if msk.any():
        ce = F.cross_entropy(lg.float()[msk], yl[msk], reduction="none")
        ww = torch.ones_like(ce)
        if w is not None:
            ww = w.view(-1, 1).expand_as(yl).float()[msk]
        return (ce * ww).sum() / ww.sum().clamp_min(1.0)
    return lg.sum() * 0.0


def _batch(a, dev):
    if not torch.is_tensor(a):
        a = torch.from_numpy(np.ascontiguousarray(a))
    return a.to(dev)


def predict(model, Q, dev, bs=256):
    model.eval()
    out = []
    X = Q["X"]
    with torch.no_grad():
        for i in range(0, len(X), bs):
            b = (_batch(X[i:i + bs], dev), _batch(Q["T"][i:i + bs], dev), _batch(Q["TY"][i:i + bs], dev))
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                lg = model(*b)
            out.append(torch.softmax(lg.float(), -1).cpu())
    return torch.cat(out).double().numpy()


def to_preds(P, eps, data):
    """query_id -> (5,) probs for that query's machine slot, aligned with queries.csv."""
    byid = {e.episode_id: i for i, e in enumerate(eps)}
    pred = {}
    for r in data.queries:
        i = byid[r["episode_id"]]
        pred[r["query_id"]] = P[i, eps[i].slot_of(r["machine_id"])].tolist()
    return pred


def epoch_indices(regime, D, n_samples, rng):
    """Deterministic index stream of exactly n_samples for one eval epoch."""
    N = len(D["X"])
    if regime == "episode_balanced":
        cnt = np.bincount(D["EP"])
        p = 1.0 / cnt[D["EP"]]
        p = p / p.sum()
        return rng.choice(N, size=n_samples, replace=True, p=p)
    out, order = [], rng.permutation(N)
    while len(out) < n_samples:
        out.extend(order[:n_samples - len(out)].tolist())
        if len(out) < n_samples:
            order = rng.permutation(N)
    return np.array(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["base", "episode_balanced", "window32", "query_upweight", "min_obs"], default="base")
    ap.add_argument("--h", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.regime == "window32":
        a.W = 32
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.15)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)

    from house_edge.scoring import load_official_validation, score_predictions, write_submission_csv
    t0 = time.time()
    tr = load_split(os.path.join(PUB, "train"), with_labels=True)
    va = load_split(os.path.join(PUB, "validation"))
    te = load_split(os.path.join(PUB, "test"))
    off = load_official_validation(os.path.join(PUB, "validation"))
    sp = json.load(open(os.path.join(REPO, "runs", "splits.json")))
    tune_ep, confirm_ep = set(sp["tune"]), set(sp["confirm"])

    DS = build_train(tr, a.W)
    N_full = len(DS["X"])  # base window count: fixes optimizer steps for ALL regimes
    keep = np.ones(N_full, bool)
    om_arr = None
    if a.regime == "min_obs":
        om_arr = DS["OBC"] >= MIN_OBS          # per-machine target validity
        keep = (DS["Y"] >= 0) & om_arr
        keep = keep.any(axis=1)                # drop windows with no valid target
    D = {k: (v[keep] if k != "OBC" else v) for k, v in DS.items()}

    XD = torch.from_numpy(D["X"]).to(dev)
    TD = torch.from_numpy(D["T"]).to(dev)
    TYD = torch.from_numpy(D["TY"]).to(dev)
    YD = torch.from_numpy(D["Y"]).long().to(dev)
    WD = None
    if a.regime == "query_upweight":
        WD = torch.where(torch.from_numpy(D["ISQ"]).to(dev), torch.full((len(D["X"]),), QUERY_UPWEIGHT, device=dev), torch.ones(len(D["X"]), device=dev))
    OMD = torch.from_numpy(om_arr[keep]).to(dev) if om_arr is not None else None

    Qv, eps_v = build_query(va, a.W)
    Qt, eps_t = build_query(te, a.W)

    model = CNN(a.h, a.layers, 3, a.drop, a.W).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    K = (N_full + a.bs - 1) // a.bs             # steps per eval epoch (base cadence)
    total_steps = a.epochs * K
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total_steps)

    logf = open(os.path.join(a.out, "log.jsonl"), "w")
    best = None
    step = 0
    print(f"device {dev} windows {N_full} used {len(D['X'])} steps {total_steps} params {n_params}", flush=True)
    for ev in range(a.epochs):
        idxs = epoch_indices(a.regime, D, K * a.bs, rng)
        model.train()
        tot = n = 0
        for i in range(0, K * a.bs, a.bs):
            j = torch.from_numpy(idxs[i:i + a.bs]).to(dev)
            x, t, ty, y = XD[j], TD[j], TYD[j], YD[j]
            w = WD[j] if WD is not None else None
            om = OMD[j] if OMD is not None else None
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                lg = model(x, t, ty)
            loss = masked_loss(lg, y, w, om)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            tot += loss.item() * len(j)
            n += len(j)
        P = predict(model, Qv, dev)
        pv = to_preds(P, eps_v, va)
        rs = {k: score_predictions(pv, off, s)["R"] for k, s in (("tune", tune_ep), ("confirm", confirm_ep), ("full", None))}
        rec = {"eval": ev, "step": step, "train_loss": tot / n, **{k: round(v, 6) for k, v in rs.items()}, "elapsed": round(time.time() - t0, 1)}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(f"epoch {ev} loss {tot / n:.4f} tuneR {rs['tune']:.4f} fullR {rs['full']:.4f} t {rec['elapsed']}s", flush=True)
        if best is None or rs["tune"] < best["tune_R"]:
            best = {"eval": ev, "tune_R": rs["tune"], "confirm_R": rs["confirm"], "full_R": rs["full"],
                    "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
    logf.close()

    model.load_state_dict(best["state"])
    model.to(dev)
    torch.save({"state": model.state_dict(), "config": vars(a), "eval": best["eval"], "tune_R": best["tune_R"]},
               os.path.join(a.out, "best.pt"))
    pv = to_preds(predict(model, Qv, dev), eps_v, va)
    pt = to_preds(predict(model, Qt, dev), eps_t, te)
    full_met = score_predictions(pv, off, None)
    met = {
        "regime": a.regime, "params": n_params, "steps": total_steps, "epochs": a.epochs,
        "best_eval": best["eval"], "tune_R": best["tune_R"], "confirm_R": best["confirm_R"], "full_R": full_met["R"],
        "loss_ratio_by_condition": full_met["loss_ratio_by_condition"],
        "probability_error_by_condition": full_met["probability_error"],
        "points_60": full_met["points_60"],
        "final_train_loss": tot / n,
        "train_seconds": round(time.time() - t0, 1),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if dev == "cuda" else 0,
        "windows_total": int(N_full), "windows_used": int(len(D["X"])),
        "windows_dropped_min_obs": int(N_full - len(D["X"])),
        "labelled_slots": int(((YD >= 0) if OMD is None else ((YD >= 0) & OMD)).sum()),
        "missing_slots_masked": int(((YD < 0) if OMD is None else ((YD < 0) | ~OMD)).sum()),
        "query_cutoff_windows": int(D["ISQ"].sum()), "seed": a.seed, "device": dev,
    }
    json.dump(met, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    write_submission_csv(os.path.join(a.out, "validation_predictions.csv"), pv)
    write_submission_csv(os.path.join(a.out, "test_submission.csv"), pt)
    print("RESULT", a.regime, "tune", round(met["tune_R"], 4), "confirm", round(met["confirm_R"], 4),
          "full", round(met["full_R"], 4), "params", n_params, "best_eval", best["eval"], flush=True)


if __name__ == "__main__":
    main()
