"""Seed-0 GPU training of 645-parameter heads on the frozen trunk. Tune R only."""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import torch

torch.set_num_threads(2)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frozen_heads as F

STUDY = F.STUDY
STATUS = os.path.join(STUDY, "STATUS.md")
BLOCKER = os.path.join(STUDY, "BLOCKER.md")


def cst_now():
    return datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S %Z")


def status(line):
    with open(STATUS, "a", encoding="utf-8") as handle:
        handle.write(f"{cst_now()} {line}\n")


def fail_blocker(message):
    text = f"{cst_now()} BLOCKER\n\n{message}\n"
    with open(BLOCKER, "w", encoding="utf-8") as handle:
        handle.write(text)
    status("BLOCKER written: " + message.splitlines()[0])
    print(text, flush=True)
    sys.exit(2)


def gpu_users_text():
    proc = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=30)
    return proc.stdout + proc.stderr


def sample_smi():
    proc = subprocess.run(
        ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showpids"],
        capture_output=True, text=True, timeout=30,
    )
    return proc.stdout + proc.stderr


def fp64_probs(trunk, head, X, T, TY, bs=16):
    model = F.new_trunk()
    model.load_state_dict(trunk.state_dict())
    model = model.double().cpu().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layer = torch.nn.Linear(128, 5).double()
    layer.load_state_dict({k: v.detach().cpu().double() for k, v in head.state_dict().items()})
    layer.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            feats = model.features(X[i:i + bs], T[i:i + bs], TY[i:i + bs])
            chunks.append(torch.softmax(layer(feats), dim=-1))
    probs = torch.cat(chunks, 0).numpy()
    return F.row_normalize(probs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(STUDY, "runs", "seed0"))
    a = ap.parse_args()
    try:
        F.require_seed(a.seed)
    except RuntimeError as exc:
        fail_blocker(str(exc))
    os.makedirs(a.out, exist_ok=True)
    pid = os.getpid()
    open(os.path.join(a.out, "pid"), "w").write(str(pid) + "\n")
    users = gpu_users_text()
    open(os.path.join(a.out, "gpu_users_before.txt"), "w").write(users)
    if "No KFD PIDs currently running" not in users:
        fail_blocker("GPU already has a KFD process; refusing to start.\n" + users)
    if not torch.cuda.is_available():
        fail_blocker("torch.cuda.is_available() is false. No CPU training fallback.")
    torch.cuda.set_per_process_memory_fraction(0.25)
    try:
        device_name = F.gpu_op_check("cuda")
    except Exception as exc:
        fail_blocker(f"ROCm tensor backward failed: {exc!r}")
    status(f"training started pid {pid} device {device_name} torch {torch.__version__}")
    print(f"DEVICE {device_name} PID {pid}", flush=True)

    t0 = time.perf_counter()
    trunk, ckpt = F.load_trunk()
    trunk_hash = F.state_hash(trunk)
    tr = F.load_split(os.path.join(F.PUB, "train"), with_labels=True)
    X, T, TY, Y, signals, eids = F.build_cutoff(tr)
    counts = F.count_labeled(signals, Y)
    if counts != F.EXPECTED_COUNTS:
        fail_blocker(f"cutoff counts {counts} != {F.EXPECTED_COUNTS}")
    active = F.active_map(counts)
    if not all(active.values()):
        fail_blocker(f"expected every signal to clear {F.MIN_GROUP_ROWS} rows, got {counts}")

    trunk = trunk.to("cuda").eval()
    held = {}
    smi_box = {}

    def grab():
        smi_box["during"] = sample_smi()

    sampler = threading.Thread(target=grab)
    sampler.start()
    torch.cuda.synchronize()
    t_feat = time.perf_counter()
    feats = F.featurize(trunk, X, T, TY, "cuda")
    torch.cuda.synchronize()
    feat_seconds = time.perf_counter() - t_feat
    sampler.join()
    open(os.path.join(a.out, "gpu_during_featurize.txt"), "w").write(smi_box.get("during", ""))

    with torch.no_grad():
        trunk_logits = trunk.head[2](feats)
        probe = F.make_head(trunk).to("cuda").eval()
        probe_logits = probe(feats)
    if not torch.equal(trunk_logits, probe_logits):
        err = (trunk_logits - probe_logits).abs().max().item()
        fail_blocker(f"epoch-0 head does not match trunk head[2], max abs {err}")
    print("epoch0_logit_match true", flush=True)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    control = F.make_head(trunk)
    specialists = {s: F.make_head(trunk) for s in F.SIGNALS}
    # Identical init: every head equals the trunk output layer before any step.
    for s in F.SIGNALS:
        if not torch.equal(specialists[s].weight, control.weight) or not torch.equal(specialists[s].bias, control.bias):
            fail_blocker("head initializations diverged before training")

    control, control_losses = F.train_head(
        control, feats, Y, signals, None, F.SCHEDULE["epochs"], F.SCHEDULE["lr"], F.SCHEDULE["wd"], "cuda",
    )
    print(f"first_epoch control_loss {control_losses[0]:.6f} device {device_name}", flush=True)
    status(f"first epoch control loss {control_losses[0]:.6f} pid {pid}")
    open(os.path.join(a.out, "gpu_after_first_head.txt"), "w").write(sample_smi())
    spec_losses = {}
    for s in F.SIGNALS:
        specialists[s], spec_losses[s] = F.train_head(
            specialists[s], feats, Y, signals, s, F.SCHEDULE["epochs"], F.SCHEDULE["lr"], F.SCHEDULE["wd"], "cuda",
        )
        print(f"signal {s} last_loss {spec_losses[s][-1]:.6f}", flush=True)

    trunk = trunk.cpu().eval()
    F.freeze_check(trunk, trunk_hash)
    after_hash = F.state_hash(trunk)
    if after_hash != trunk_hash:
        fail_blocker("trunk hash changed during head training")
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20

    splits = json.load(open(os.path.join(STUDY, "SPLIT_IDS.json")))
    tune_ids = set(splits["tune"])
    from house_edge.data import subset_split_data
    va = F.load_split(os.path.join(F.PUB, "validation"), with_labels=False)
    va = subset_split_data(va, tune_ids)
    ev = F.build_eval(va)
    base_head = F.make_head(trunk)
    t_score = time.perf_counter()
    p_base = fp64_probs(trunk, base_head, ev[0], ev[1], ev[2])
    p_shared = fp64_probs(trunk, control.cpu(), ev[0], ev[1], ev[2])
    per_signal = {s: fp64_probs(trunk, specialists[s].cpu(), ev[0], ev[1], ev[2]) for s in F.SIGNALS}
    routed_logits_as_probs = per_signal  # already probabilities; router copies rows
    # route_logits expects arrays shaped (N,8,5). Use them directly, then renorm each row.
    routed, used = F.route_logits(per_signal, p_base, ev[4], active)
    routed = F.row_normalize(routed)
    score_seconds = time.perf_counter() - t_score

    from house_edge.scoring import load_official_validation, score_predictions
    official = load_official_validation(os.path.join(F.PUB, "validation"))
    pred_base = F.to_preds(p_base, ev[5], va)
    pred_shared = F.to_preds(p_shared, ev[5], va)
    pred_routed = F.to_preds(routed, ev[5], va)
    tune_base = score_predictions(pred_base, official, tune_ids)["R"]
    tune_shared = score_predictions(pred_shared, official, tune_ids)["R"]
    tune_routed = score_predictions(pred_routed, official, tune_ids)["R"]

    # Route audit: every tune query used its own signal's head.
    by_id = {ep.episode_id: ep for ep in va.episodes}
    pos = {eid: i for i, eid in enumerate(ev[5])}
    for row in va.queries:
        i = pos[row["episode_id"]]
        slot = by_id[row["episode_id"]].slot_of(row["machine_id"])
        sig = int(row["signal"])
        if used[i, slot] != F.specialist_name(sig):
            fail_blocker(f"route mismatch {row['query_id']} signal {sig} used {used[i, slot]}")

    result = {
        "seed": 0,
        "device": device_name,
        "torch": torch.__version__,
        "pid": pid,
        "trunk_sha256": F.TRUNK_SHA,
        "trunk_steps": ckpt["steps"],
        "trunk_state_hash_before": trunk_hash,
        "trunk_state_hash_after": after_hash,
        "frozen_unchanged": trunk_hash == after_hash,
        "trainable_per_head": F.HEAD_PARAMS,
        "frozen_params": F.TRUNK_PARAMS,
        "counts": counts,
        "active": {str(k): v for k, v in active.items()},
        "schedule": F.SCHEDULE,
        "steps": F.SCHEDULE["epochs"],
        "epoch0_logit_match": True,
        "train_loss": {
            "control": control_losses,
            "specialist": {str(k): v for k, v in spec_losses.items()},
        },
        "final_train_loss": {
            "control": control_losses[-1],
            "specialist": {str(k): v[-1] for k, v in spec_losses.items()},
        },
        "tune_R": {
            "baseline": tune_base,
            "shared": tune_shared,
            "routed": tune_routed,
            "shared_minus_baseline": tune_shared - tune_base,
            "routed_minus_baseline": tune_routed - tune_base,
            "routed_minus_shared": tune_routed - tune_shared,
        },
        "feat_seconds": feat_seconds,
        "score_seconds": score_seconds,
        "seconds_total": time.perf_counter() - t0,
        "peak_gpu_mb": peak_mb,
        "n_cutoff": int(len(X)),
        "n_tune_episodes": len(tune_ids),
        "buffers": [(n, tuple(b.shape)) for n, b in trunk.named_buffers()],
    }
    json.dump(result, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    json.dump(result, open(os.path.join(STUDY, "results", "results.json"), "w"), indent=1)
    write_report(result)
    status(
        "final seed 0 tune R baseline {baseline:.6f} shared {shared:.6f} routed {routed:.6f} "
        "peak_gpu_mb {peak:.1f} pid {pid}".format(pid=pid, peak=peak_mb, **result["tune_R"])
    )
    print("TUNE_R", {k: round(v, 6) for k, v in result["tune_R"].items()}, flush=True)
    print("EXIT_OK", flush=True)
    del held


def write_report(result):
    os.makedirs(os.path.join(STUDY, "results"), exist_ok=True)
    t = result["tune_R"]
    lines = [
        "# Frozen feature heads, seed 0",
        "",
        f"Device: {result['device']}, torch {result['torch']}, pid {result['pid']}.",
        f"Trunk sha256 {result['trunk_sha256']}, state hash unchanged ({result['frozen_unchanged']}).",
        f"Trainable parameters per head: {result['trainable_per_head']}. Frozen trunk parameters: {result['frozen_params']}.",
        f"Cutoff labels: {result['counts']}. Steps: {result['steps']}. Epoch-0 logits match the trunk: {result['epoch0_logit_match']}.",
        "",
        "Tune R only (lower is better). Confirm and test were not scored.",
        "",
        f"- Frozen baseline: {t['baseline']:.6f}",
        f"- Shared head: {t['shared']:.6f} (delta vs baseline {t['shared_minus_baseline']:+.6f})",
        f"- Routed heads: {t['routed']:.6f} (delta vs baseline {t['routed_minus_baseline']:+.6f}, vs shared {t['routed_minus_shared']:+.6f})",
        "",
        f"Final train loss: control {result['final_train_loss']['control']:.4f}, "
        + ", ".join(f"signal {k} {v:.4f}" for k, v in result["final_train_loss"]["specialist"].items())
        + ".",
        f"Peak GPU memory during the run: {result['peak_gpu_mb']:.1f} MiB. Feature pass: {result['feat_seconds']:.2f}s.",
        "",
        "No further seed is trained. The tune split was already used for CNN03, each head sees about 250 labels, and the budget is 10 full-batch steps. Deltas of this size are not a reason to continue.",
        "",
    ]
    open(os.path.join(STUDY, "results", "RESULTS.md"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
