"""One full hard_signal seed on the server GPU. CPU fp32 scoring. No early stopping."""
import hashlib
import json
import os
import socket
import sys
import time

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import torch

torch.set_num_threads(2)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hard_signal as H

REPO = H.REPO
CAP_BYTES = 2 * 1024 ** 3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refuse_host_and_memory():
    host = socket.gethostname()
    if host.lower().startswith("tower"):
        sys.exit("REFUSED: tower training disabled")
    if not torch.cuda.is_available():
        sys.exit("REFUSED: no CUDA/ROCm GPU")
    free, total = torch.cuda.mem_get_info()
    print(f"gpu {torch.cuda.get_device_name(0)} free_mib {free / 2**20:.0f} total_mib {total / 2**20:.0f}", flush=True)
    if free < 2200 * 2 ** 20:
        sys.exit(f"REFUSED: free GPU memory {free / 2**20:.0f} MiB")
    frac = min(0.95, CAP_BYTES / total)
    torch.cuda.set_per_process_memory_fraction(frac)
    print(f"vram cap {CAP_BYTES / 2**20:.0f} MiB fraction {frac:.4f}", flush=True)


def preds_from(P, data, episode_ids):
    by_id = {ep.episode_id: ep for ep in data.episodes}
    return H.to_preds(P, episode_ids, by_id, data)


def route_rows(data, episode_ids, used, signals):
    by_id = {ep.episode_id: ep for ep in data.episodes}
    sig_np = signals.numpy()
    rows = []
    pos = {eid: i for i, eid in enumerate(episode_ids)}
    for r in data.queries:
        i = pos[r["episode_id"]]
        slot = by_id[r["episode_id"]].slot_of(r["machine_id"])
        rows.append({
            "query_id": r["query_id"],
            "episode_id": r["episode_id"],
            "signal": int(sig_np[i, slot]),
            "model_used": str(used[i, slot]),
        })
    return rows


def score_split(preds, official, episode_ids):
    from house_edge.scoring import score_predictions
    return score_predictions(preds, official, set(episode_ids) if episode_ids is not None else None)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seed not in H.INCUMBENT_TUNE_R:
        sys.exit("REFUSED: seed must be 0, 1, or 2")
    refuse_host_and_memory()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    src = {
        "hard_signal.py": sha256_file(os.path.join(HERE, "hard_signal.py")),
        "train.py": sha256_file(os.path.join(HERE, "train.py")),
        "PLAN.md": sha256_file(os.path.join(HERE, "..", "PLAN.md")),
        "base_cnn.py": sha256_file(os.path.join(REPO, "experiments", "mlp_cnn_hpo", "source", "base_cnn.py")),
        "splits.json": sha256_file(os.path.join(REPO, "runs", "splits.json")),
    }
    json.dump({"cfg": H.CFG, "seed": a.seed, "source_sha256": src, "torch": torch.__version__},
              open(os.path.join(a.out, "started.json"), "w"), indent=1)
    print("load train", flush=True)
    tr = H.load_train()
    if not tr.query_labels:
        sys.exit("REFUSED: train query labels missing")
    fitted = H.fit(tr, a.seed, device="cuda")
    print("fit_seconds", round(fitted["seconds_fit"], 1), "steps", fitted["steps"], flush=True)

    torch.set_num_threads(2)
    va = H.load_split(os.path.join(H.PUB, "validation"), with_labels=False)
    te = H.load_split(os.path.join(H.PUB, "test"), with_labels=False)
    ev = H.build_eval(va)
    et = H.build_eval(te)
    t_pred = time.perf_counter()
    Pr, used_v = H.predict_routed(fitted["specialists"], fitted["shared"], fitted["active"], ev.X, ev.T, ev.TY, ev.signals)
    Pc = H.predict_all(fitted["control"], ev.X, ev.T, ev.TY)
    Ptr, _ = H.predict_routed(fitted["specialists"], fitted["shared"], fitted["active"], et.X, et.T, et.TY, et.signals)
    Ptc = H.predict_all(fitted["control"], et.X, et.T, et.TY)
    pred_r = preds_from(Pr, va, ev.episode_ids)
    pred_c = preds_from(Pc, va, ev.episode_ids)
    pred_rt = preds_from(Ptr, te, et.episode_ids)
    pred_ct = preds_from(Ptc, te, et.episode_ids)
    print(f"cpu_predict_seconds {time.perf_counter() - t_pred:.1f}", flush=True)

    from house_edge.scoring import load_official_validation, write_submission_csv
    official = load_official_validation(os.path.join(H.PUB, "validation"))
    splits = json.load(open(os.path.join(REPO, "runs", "splits.json")))
    tune_ids = splits["tune"]
    routed_tune = score_split(pred_r, official, tune_ids)
    control_tune = score_split(pred_c, official, tune_ids)
    incumbent = H.INCUMBENT_TUNE_R[a.seed]
    tune_doc = {
        "seed": a.seed,
        "routed_tune_R": routed_tune["R"],
        "control_tune_R": control_tune["R"],
        "incumbent_tune_R": incumbent,
        "routed_beats_incumbent": bool(routed_tune["R"] < incumbent),
        "routed_tune": routed_tune,
        "control_tune": control_tune,
    }
    json.dump(tune_doc, open(os.path.join(a.out, "tune_only.json"), "w"), indent=1)
    print(f"TUNE_R routed {routed_tune['R']:.10f} control {control_tune['R']:.10f} incumbent {incumbent:.10f}", flush=True)

    # Confirm/full are stored after the tune file. Promotion does not read them.
    routed = {
        "tune": routed_tune,
        "confirm": score_split(pred_r, official, splits["confirm"]),
        "full": score_split(pred_r, official, None),
    }
    control = {
        "tune": control_tune,
        "confirm": score_split(pred_c, official, splits["confirm"]),
        "full": score_split(pred_c, official, None),
    }
    write_submission_csv(os.path.join(a.out, "validation_predictions_routed.csv"), pred_r)
    write_submission_csv(os.path.join(a.out, "validation_predictions_control.csv"), pred_c)
    write_submission_csv(os.path.join(a.out, "test_predictions_routed.csv"), pred_rt)
    write_submission_csv(os.path.join(a.out, "test_predictions_control.csv"), pred_ct)
    routes = route_rows(va, ev.episode_ids, used_v, ev.signals)
    with open(os.path.join(a.out, "route_log.csv"), "w", encoding="utf-8") as handle:
        handle.write("query_id,signal,model_used\n")
        for row in routes:
            handle.write(f"{row['query_id']},{row['signal']},{row['model_used']}\n")
    ckpt = {
        "cfg": H.CFG,
        "seed": a.seed,
        "active": {str(k): v for k, v in fitted["active"].items()},
        "counts": {str(k): v for k, v in fitted["counts"].items()},
        "shared": fitted["shared"].state_dict(),
        "control": fitted["control"].state_dict(),
        "specialists": {str(k): (None if m is None else m.state_dict()) for k, m in fitted["specialists"].items()},
    }
    torch.save(ckpt, os.path.join(a.out, "checkpoint.pt"))
    metrics = {
        "seed": a.seed,
        "cfg": H.CFG,
        "param_count": 367109,
        "counts": {str(k): v for k, v in fitted["counts"].items()},
        "active": {str(k): v for k, v in fitted["active"].items()},
        "steps": {
            "pretrain": fitted["steps"]["pretrain"],
            "control": fitted["steps"]["control"],
            "specialist": {str(k): v for k, v in fitted["steps"]["specialist"].items()},
        },
        "losses": {
            "pretrain": fitted["losses"]["pretrain"],
            "control": fitted["losses"]["control"],
            "specialist": {str(k): v for k, v in fitted["losses"]["specialist"].items()},
        },
        "n_history": fitted["n_history"],
        "n_cutoff": fitted["n_cutoff"],
        "routed": routed,
        "control": control,
        "incumbent_tune_R": incumbent,
        "routed_beats_incumbent": bool(routed_tune["R"] < incumbent),
        "seconds_fit": fitted["seconds_fit"],
        "seconds_total": time.perf_counter() - t0,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
        "source_sha256": src,
        "torch": torch.__version__,
        "host": socket.gethostname(),
    }
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(f"EXIT_OK seconds {metrics['seconds_total']:.1f} peak_gpu_mb {metrics['peak_gpu_mb']:.1f}", flush=True)


if __name__ == "__main__":
    main()
