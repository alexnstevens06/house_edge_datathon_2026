"""MLP vs CNN trainer for experiments/mlp_cnn_hpo. Reuses base_cnn (verbatim copy of
experiments/cnn/source/cnn.py) for data build / loss / predict / scoring. Signal NEVER read:
B.USE_SIGNAL=False at import; CNN always constructed with use_signal=False."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import base_cnn as B

B.USE_SIGNAL = False  # signal is never read anywhere below

RUNS = os.path.abspath(os.path.join(HERE, "..", "runs"))
TRIALS = os.path.join(RUNS, "trials.jsonl")


class MLP(nn.Module):
    """Fully-connected net over the flattened (W,8) window. Per cell: one-hot payout(5)
    (missing = all zeros) + observed mask(1) -> W*8*6; concat machine-type one-hot (8*3)
    and t/256 -> depth hidden layers of width -> Linear(width,40) -> (B,8,5).
    forward ignores sg (no signal parameters). Only Linear/LayerNorm/Dropout/GELU modules."""

    def __init__(s, width=1024, depth=3, drop=0.3, W=64):
        super().__init__(); s.W = W
        d = W * 8 * 6 + 8 * 3 + 1
        mods = []
        for i in range(depth):
            mods += [nn.Linear(d if i == 0 else width, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(drop)]
        s.net = nn.Sequential(*mods)
        s.head = nn.Linear(width, 40)

    def forward(s, x, t, ty, sg):
        Bs, W, M = x.shape
        onehot = (x.unsqueeze(-1) == torch.arange(5, device=x.device)).float()      # (B,W,8,5) missing = zeros
        mask = (x >= 0).float()                                                      # (B,W,8) observed mask
        cells = torch.cat([onehot, mask.unsqueeze(-1)], -1).reshape(Bs, W * M * 6)   # (B,W*8*6)
        typ = F.one_hot(ty.long(), 3).float().reshape(Bs, -1)                        # (B,8*3)
        tt = (t.float() / 256.0).unsqueeze(-1)                                       # (B,1)
        z = s.net(torch.cat([cells, typ, tt], -1))
        return s.head(z).view(Bs, 8, 5)                                              # (B,8,5) logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mlp"], default="mlp")
    ap.add_argument("--h", type=int, default=64); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=3)  # ignored (CLI compat)
    ap.add_argument("--width", type=int, default=1024); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--W", type=int, default=64); ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--phase", choices=["tune", "final"], default="tune")
    ap.add_argument("--max_train_windows", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(2); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda": torch.cuda.set_per_process_memory_fraction(0.25)
    out = a.out or os.path.join(RUNS, f"{a.model}_{a.phase}_seed{a.seed}")
    os.makedirs(out, exist_ok=True); json.dump(vars(a), open(os.path.join(out, "config.json"), "w"), indent=1)
    from house_edge.scoring import load_official_validation, score_predictions, write_submission_csv
    t0 = time.time(); tr = B.load_split(os.path.join(B.PUB, "train"), with_labels=True)
    va = B.load_split(os.path.join(B.PUB, "validation")); te = B.load_split(os.path.join(B.PUB, "test"))
    X, T, TY, SG, Y = B.build_train(tr, a.W); print("train windows", len(X), "build", time.time() - t0, flush=True)
    if a.max_train_windows and a.max_train_windows < len(X):
        n = a.max_train_windows; X, T, TY, SG, Y = (d[:n] for d in (X, T, TY, SG, Y))
    if a.model == "cnn":
        model = B.CNN(a.h, a.layers, 3, a.drop, a.W, use_signal=False)  # kernel arg ignored, no signal ever
    else:
        model = MLP(a.width, a.depth, a.drop, a.W)
    model = model.to(dev); param_count = sum(p.numel() for p in model.parameters()); print("params", param_count, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((len(X) + a.bs - 1) // a.bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps)
    D = [d.to(dev) for d in (X, T, TY, SG, Y)]; losses = []
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(X), device=dev); tot = n_ = 0
        for i in range(0, len(X), a.bs):
            j = perm[i:i + a.bs]; x, t, ty, sg, y = (d[j] for d in D)
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                lg = model(x, t, ty, sg)
            loss = B.masked_loss(lg, y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            tot += loss.item() * len(j); n_ += len(j)
        losses.append(tot / n_); print(f"epoch {ep} loss {tot / n_:.4f} t {time.time() - t0:.0f}s", flush=True)
    if a.phase == "final":
        torch.save({"state": model.state_dict(), "config": vars(a)}, os.path.join(out, "checkpoint.pt"))
    off = load_official_validation(os.path.join(B.PUB, "validation")); sp = json.load(open(os.path.join(B.REPO, "runs", "splits.json")))
    if a.phase == "final":
        model = model.to("cpu").eval()  # CPU fp32 2-thread export for all final scoring/CSVs
    Qv, ev = B.build_query(va, a.W); pv = B.to_preds(B.predict(model, Qv, "cpu" if a.phase == "final" else dev), ev, va)
    rec = {"args": vars(a), "param_count": param_count, "n_train_windows": int(len(X)), "losses": losses,
           "tune_R": score_predictions(pv, off, set(sp["tune"]))["R"], "seconds": time.time() - t0}
    if a.phase == "final":
        met = {k: score_predictions(pv, off, s) for k, s in (("full", None), ("tune", set(sp["tune"])), ("confirm", set(sp["confirm"])))}
        rec["tune_R"], rec["confirm_R"], rec["full_R"] = met["tune"]["R"], met["confirm"]["R"], met["full"]["R"]
        Qt, et = B.build_query(te, a.W); pt = B.to_preds(B.predict(model, Qt, "cpu"), et, te)
        write_submission_csv(os.path.join(out, "validation_predictions_cpu.csv"), pv)
        write_submission_csv(os.path.join(out, "test_predictions_cpu.csv"), pt)
        json.dump(met, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    with open(TRIALS, "a") as f: f.write(json.dumps(rec) + "\n")
    print("RESULT", out, {k: round(v, 4) for k, v in rec.items() if k.endswith("_R")}, f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__": main()
