"""Warm-start extra epochs from a weights-only curve checkpoint. CUDA only."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from study_common import (  # noqa: E402
    BS,
    CONV_SOURCE,
    DROP,
    EXPECTED_WINDOWS,
    H,
    LAYERS,
    REPO,
    STEPS_PER_EPOCH,
    W,
    WD,
    assert_inside_study,
    check_state,
    continuation_seed,
    load_curve_blob,
    manifest,
    parent_is_weights_only,
    sha256_file,
)

sys.path.insert(0, str(CONV_SOURCE))
import base_cnn as B  # noqa: E402

B.USE_SIGNAL = False


def gpu_line() -> str:
    try:
        text = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,name", "--format=csv,noheader"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"nvidia-smi-failed {exc}"
    return text.strip().replace("\n", " | ")


def train(arm: str, extra_epochs: int, out: Path) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required; refusing CPU training")
    pin = manifest()["checkpoints"][arm]
    parent = REPO / pin["path"]
    digest = sha256_file(parent)
    if digest != pin["sha256"]:
        raise SystemExit(f"parent sha mismatch for {arm}")
    blob = load_curve_blob(parent)
    if not parent_is_weights_only(blob):
        raise SystemExit(f"{arm} unexpectedly contains optimizer or extra keys {sorted(blob)}")
    cfg = blob["config"]
    if int(cfg["seed"]) != int(pin["seed"]) or abs(float(cfg["lr"]) - float(pin["lr"])) > 1e-12:
        raise SystemExit("checkpoint lr/seed does not match the frozen arm")
    check_state(blob["state"])
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.40)
    dev = "cuda"
    out = assert_inside_study(out)
    out.mkdir(parents=True, exist_ok=True)
    tr = B.load_split(os.path.join(B.PUB, "train"), with_labels=True)
    windows = B.build_train(tr, W)
    X, T, TY, SG, Y = windows
    n = int(len(X))
    spe = (n + BS - 1) // BS
    if n != EXPECTED_WINDOWS or spe != STEPS_PER_EPOCH:
        raise SystemExit(f"window count {n} steps/epoch {spe}")
    seed = continuation_seed(int(cfg["seed"]), extra_epochs)
    model = B.CNN(H, LAYERS, 3, DROP, W, use_signal=False).to(dev)
    model.load_state_dict(blob["state"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=WD)
    total_steps = extra_epochs * spe
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, float(cfg["lr"]), total_steps=total_steps, pct_start=0.3)
    tensors = [d.to(dev) for d in (X, T, TY, SG, Y)]
    launch = {
        "pid": os.getpid(),
        "device": torch.cuda.get_device_name(0),
        "arm": arm,
        "extra_epochs": extra_epochs,
        "continuation_seed": seed,
        "warm_start": "weights_only_parent",
        "parent_sha256": digest,
        "parent_keys": sorted(blob),
        "total_steps": total_steps,
        "lr": float(cfg["lr"]),
        "gpu": gpu_line(),
    }
    (out / "launch.json").write_text(json.dumps(launch, indent=1) + "\n", encoding="utf-8")
    print(
        f"LAUNCH pid={os.getpid()} device={launch['device']} arm={arm} extra={extra_epochs} "
        f"warm_start=weights_only seed={seed} steps={total_steps} gpu={launch['gpu']}",
        flush=True,
    )
    losses = []
    t0 = time.perf_counter()
    for ep in range(extra_epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = n_ = 0
        for i in range(0, n, BS):
            j = perm[i:i + BS]
            x, t, ty, sg, y = (d[j] for d in tensors)
            with torch.autocast("cuda", torch.bfloat16, enabled=True):
                logits = model(x, t, ty, sg)
            loss = B.masked_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            tot += loss.item() * len(j)
            n_ += len(j)
        train_loss = tot / n_
        losses.append(train_loss)
        if ep == 0:
            print(
                f"FIRST_EPOCH pid={os.getpid()} arm={arm} extra={extra_epochs} "
                f"loss={train_loss:.6f} gpu={gpu_line()}",
                flush=True,
            )
        print(f"epoch {ep} loss {train_loss:.4f} t {time.perf_counter() - t0:.1f}s", flush=True)
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    check_state(cpu_state)
    saved = {
        "state": cpu_state,
        "optimizer": opt.state_dict(),
        "scheduler": sch.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "completed_extra_epochs": extra_epochs,
        "warm_start_from": "weights_only",
        "parent_checkpoint_sha256": digest,
        "parent_had_optimizer": False,
        "continuation_seed": seed,
        "losses": losses,
        "config": {
            "arm": arm,
            "lr": float(cfg["lr"]),
            "seed": int(cfg["seed"]),
            "extra_epochs": extra_epochs,
            "total_steps": total_steps,
            "pct_start": 0.3,
            "wd": WD,
            "bs": BS,
        },
    }
    torch.save(saved, out / "checkpoint.pt")
    summary = {
        "arm": arm,
        "extra_epochs": extra_epochs,
        "final_train_loss": losses[-1],
        "losses": losses,
        "pid": os.getpid(),
        "device": launch["device"],
        "train_seconds": time.perf_counter() - t0,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20,
        "warm_start": "weights_only_parent",
        "total_steps": total_steps,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"DONE arm={arm} extra={extra_epochs} loss={losses[-1]:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(manifest()["checkpoints"]))
    parser.add_argument("--extra-epochs", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train(args.arm, args.extra_epochs, Path(args.out))


if __name__ == "__main__":
    main()
