"""CNN03 training with a tune-R curve. Eval saves and restores torch RNG so the
optimizer trajectory matches a run that never evaluates."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import base_cnn as B

B.USE_SIGNAL = False  # signal is never read

REACH = 0.345


def steps_per_epoch(n, bs):
    return (int(n) + int(bs) - 1) // int(bs)


def make_scheduler(opt, lr, total_steps, pct_start):
    return torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total_steps, pct_start=pct_start)


def subset_windows(X, T, TY, SG, Y, max_train_windows):
    if max_train_windows and max_train_windows < len(X):
        n = int(max_train_windows)
        return tuple(d[:n] for d in (X, T, TY, SG, Y))
    return X, T, TY, SG, Y


def capture_rng():
    state = {"cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        state["cuda"] = [t.clone() for t in torch.cuda.get_rng_state_all()]
    return state


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def cuda_sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def eval_tune(model, va, dev, W, off, tune_ids):
    """Tune R only. Restores train mode and the torch RNG captured on entry."""
    rng = capture_rng()
    model.eval()
    try:
        Qv, ev = B.build_query(va, W)
        pv = B.to_preds(B.predict(model, Qv, dev), ev, va)
        from house_edge.scoring import score_predictions
        tune_R = float(score_predictions(pv, off, tune_ids)["R"])
    finally:
        model.train()
        restore_rng(rng)
    return tune_R


def run_cnn(a, max_train_windows=0, do_eval=True, dev=None, windows=None, va=None, off=None, tune_ids=None):
    """Train a no-signal CNN. `do_eval=False` skips the eval hook (tests)."""
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.25)
    if windows is None:
        tr = B.load_split(os.path.join(B.PUB, "train"), with_labels=True)
        windows = B.build_train(tr, a.W)
    X, T, TY, SG, Y = subset_windows(*windows, max_train_windows)
    n = int(len(X))
    spe = steps_per_epoch(n, a.bs)
    total_steps = a.epochs * spe
    if do_eval and va is None:
        from house_edge.scoring import load_official_validation
        va = B.load_split(os.path.join(B.PUB, "validation"))
        off = load_official_validation(os.path.join(B.PUB, "validation"))
        sp = json.load(open(os.path.join(B.REPO, "runs", "splits.json")))
        tune_ids = set(sp["tune"])
    model = B.CNN(a.h, a.layers, 3, a.drop, a.W, use_signal=False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sch = make_scheduler(opt, a.lr, total_steps=total_steps, pct_start=a.pct_start)
    D = [d.to(dev) for d in (X, T, TY, SG, Y)]
    train_seconds = 0.0
    eval_seconds = 0.0
    steps = 0
    exposures = 0
    curve = []
    first = None
    out = getattr(a, "out", None)
    curve_f = None
    if out and do_eval:
        os.makedirs(out, exist_ok=True)
        curve_f = open(os.path.join(out, "curve.jsonl"), "w")
    for ep in range(a.epochs):
        model.train()
        cuda_sync(dev)
        t_tr = time.perf_counter()
        perm = torch.randperm(n, device=dev)
        tot = n_ = 0
        for i in range(0, n, a.bs):
            j = perm[i:i + a.bs]
            x, t, ty, sg, y = (d[j] for d in D)
            with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
                lg = model(x, t, ty, sg)
            loss = B.masked_loss(lg, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += loss.item() * len(j)
            n_ += len(j)
            steps += 1
            exposures += int(len(j))
        cuda_sync(dev)
        train_seconds += time.perf_counter() - t_tr
        train_loss = tot / n_
        print(f"epoch {ep} loss {train_loss:.4f} t_train {train_seconds:.0f}s", flush=True)
        need = do_eval and (((ep + 1) % a.eval_every == 0) or ep == a.epochs - 1)
        tune_R = None
        if need:
            cuda_sync(dev)
            t_ev = time.perf_counter()
            tune_R = eval_tune(model, va, dev, a.W, off, tune_ids)
            cuda_sync(dev)
            eval_seconds += time.perf_counter() - t_ev
            rec = {
                "epoch": ep,
                "steps": steps,
                "exposures": exposures,
                "train_seconds_cum": train_seconds,
                "eval_seconds_cum": eval_seconds,
                "train_loss": train_loss,
                "tune_R": tune_R,
            }
            curve.append(rec)
            if curve_f:
                curve_f.write(json.dumps(rec) + "\n")
                curve_f.flush()
            if first is None and tune_R <= REACH:
                first = {"epoch": ep, "steps": steps, "train_seconds": train_seconds}
            print(f"eval epoch {ep} tune_R {tune_R:.4f} t_eval {eval_seconds:.0f}s", flush=True)
    if curve_f:
        curve_f.close()
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else 0.0
    summary = {
        "args": vars(a),
        "n_train_windows": n,
        "steps_per_epoch": spe,
        "total_steps": total_steps,
        "final_tune_R": None if not curve else curve[-1]["tune_R"],
        "first_tune_R_le_0.345": first,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "peak_gpu_mb": peak,
    }
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
        torch.save({"state": model.state_dict(), "config": vars(a)}, os.path.join(out, "checkpoint.pt"))
    return model, summary, sch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--wd", type=float, required=True)
    ap.add_argument("--bs", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--drop", type=float, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pct_start", type=float, default=0.3)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, summary, _ = run_cnn(a)
    print("RESULT", a.out, "tune_R", summary["final_tune_R"], "train_s", round(summary["train_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
