"""FP32 fixed-LR warm start. CUDA only. Does not score."""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import (  # noqa: E402
    ARMS,
    BS,
    CONTINUATION_SEED,
    CONV_SOURCE,
    DROP,
    EPOCHS,
    EXPECTED_WINDOWS,
    GRAD_CLIP,
    H,
    KERNEL,
    LAYERS,
    PARENT_BATCH,
    REPO,
    SAVE_EPOCHS,
    STEPS_PER_EPOCH,
    STUDY,
    TOTAL_STEPS,
    W,
    WD,
    assert_input_pins,
    host_lr,
    parent_errors,
    sha256_file,
    state_errors,
)

sys.path.insert(0, str(CONV_SOURCE))
import base_cnn as B  # noqa: E402

B.USE_SIGNAL = False


def disable_tf32() -> dict:
    """Process-local flags only. This does not change another process."""
    flags = {}
    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    if hasattr(matmul, "allow_tf32"):
        matmul.allow_tf32 = False
        flags["matmul_allow_tf32"] = bool(matmul.allow_tf32)
    if hasattr(cudnn, "allow_tf32"):
        cudnn.allow_tf32 = False
        flags["cudnn_allow_tf32"] = bool(cudnn.allow_tf32)
    if hasattr(matmul, "fp32_precision"):
        matmul.fp32_precision = "ieee"
        flags["matmul_fp32_precision"] = str(matmul.fp32_precision)
    if hasattr(cudnn, "fp32_precision"):
        cudnn.fp32_precision = "ieee"
        flags["cudnn_fp32_precision"] = str(cudnn.fp32_precision)
    return flags


HOST_RSS_LIMIT_KB = 2 * 1024 * 1024


def rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _cmdline(pid: str) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace")


def gpu_compute_rows() -> list[tuple[str, str]]:
    """PIDs using this machine's GPU. nvidia-smi on the server, rocm-smi on the tower."""
    if shutil.which("nvidia-smi"):
        try:
            text = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"nvidia-smi failed: {exc}") from exc
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            pid = line.split(",", 1)[0].strip()
            rows.append((pid, line))
        return rows
    if shutil.which("rocm-smi"):
        try:
            text = subprocess.check_output(["rocm-smi", "--showpids"], text=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"rocm-smi failed: {exc}") from exc
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if parts and parts[0].isdigit():
                rows.append((parts[0], line.strip()))
        return rows
    raise SystemExit("no nvidia-smi or rocm-smi; refusing to launch")


def assert_gpu_free_for_pair() -> None:
    """Refuse another user's GPU process. Allow one sibling low-rate trainer, so the pair can share the card."""
    foreign = []
    own = 0
    for pid, desc in gpu_compute_rows():
        command = _cmdline(pid)
        if "train_lowrate.py" in command and "lowrate_20261108" in command:
            own += 1
        else:
            foreign.append(desc if not command else f"{desc} cmd={command}")
    if foreign:
        raise SystemExit("GPU has another user's compute process; not launching: " + " | ".join(foreign))
    if own >= 2:
        raise SystemExit(f"already {own} lowrate trainers on this GPU; not launching a third")


def gpu_line() -> str:
    if shutil.which("nvidia-smi"):
        try:
            return subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,name", "--format=csv,noheader"],
                text=True,
            ).strip().replace("\n", " | ")
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"nvidia-smi-failed {exc}"
    if shutil.which("rocm-smi"):
        try:
            text = subprocess.check_output(["rocm-smi", "--showuse"], text=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"rocm-smi-failed {exc}"
        compact = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return compact[:400]
    return "gpu-query-unavailable"


def save_epoch(out: Path, payload: dict, completed: int, steps: int) -> str:
    dest = out / f"epoch_{completed:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "checkpoint.pt"
    torch.save(payload, path)
    digest = sha256_file(path)
    provenance = {
        "not_a_pretraining_pin": True,
        "written_before_metrics": True,
        "sha256": digest,
        "completed_epochs": completed,
        "completed_steps": steps,
        "path": str(path.relative_to(REPO)),
    }
    (dest / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")
    snapshot_path = out / "POST_RUN_SNAPSHOT.json"
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    else:
        snapshot = {"not_a_pretraining_pin": True, "written_before_metrics": True, "epochs": {}}
    snapshot["epochs"][str(completed)] = provenance
    snapshot_path.write_text(json.dumps(snapshot, indent=1) + "\n", encoding="utf-8")
    return digest


def train(host: str, arm: str, parent: Path, out: Path) -> None:
    from contract import refuse_if_no_cuda

    refuse_if_no_cuda(torch.cuda.is_available())
    assert_input_pins()
    if host not in ("server", "tower") or arm not in ARMS:
        raise SystemExit("host/arm is outside the frozen four-job set")
    assert_gpu_free_for_pair()
    lr = host_lr(host)
    parent = parent.resolve()
    digest = sha256_file(parent)
    blob = torch.load(parent, map_location="cpu", weights_only=False)
    errors = parent_errors(blob, arm, digest)
    if errors:
        raise SystemExit("parent refused: " + "; ".join(errors))
    out = out.resolve()
    if STUDY.resolve() not in out.parents:
        raise SystemExit(f"refusing to write outside the study folder: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    flags = disable_tf32()
    torch.cuda.set_per_process_memory_fraction(0.40)
    dev = "cuda"
    tr = B.load_split(os.path.join(B.PUB, "train"), with_labels=True)
    windows = B.build_train(tr, W)
    X, T, TY, SG, Y = windows
    n = int(len(X))
    spe = (n + BS - 1) // BS
    if n != EXPECTED_WINDOWS or spe != STEPS_PER_EPOCH or int(SG.sum()) != 0:
        raise SystemExit(f"train construction refused n={n} spe={spe}")
    if rss_kb() > HOST_RSS_LIMIT_KB:
        raise SystemExit(f"host RSS {rss_kb()} KB exceeds 2 GiB before training")
    model = B.CNN(H, LAYERS, KERNEL, DROP, W, use_signal=False).to(dev)
    model.load_state_dict(blob["state"])
    if model.use_signal or model.sig is not None:
        raise SystemExit("signal mode refused")
    torch.manual_seed(CONTINUATION_SEED)
    np.random.seed(CONTINUATION_SEED)
    torch.cuda.manual_seed_all(CONTINUATION_SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    tensors = [item.to(dev) for item in (X, T, TY, SG, Y)]
    launch = {
        "pid": os.getpid(),
        "host": host,
        "arm": arm,
        "parent": ARMS[arm],
        "parent_sha256": digest,
        "parent_had_optimizer": False,
        "warm_start": "weights_only_fresh_adamw",
        "train_lr": lr,
        "continuation_seed": CONTINUATION_SEED,
        "device": torch.cuda.get_device_name(0),
        "precision": "fp32",
        "historical_parent_precision": "bf16_autocast",
        "historical_parent_batch": PARENT_BATCH,
        "batch": BS,
        "epochs": EPOCHS,
        "steps_per_epoch": spe,
        "total_steps": TOTAL_STEPS,
        "tf32": flags,
        "gpu": gpu_line(),
    }
    (out / "launch.json").write_text(json.dumps(launch, indent=1) + "\n", encoding="utf-8")
    print(
        f"LAUNCH pid={os.getpid()} device={launch['device']} host={host} arm={arm} "
        f"lr={lr:.8g} seed={CONTINUATION_SEED} steps={TOTAL_STEPS} gpu={launch['gpu']}",
        flush=True,
    )
    losses = []
    steps = 0
    t0 = time.perf_counter()
    loss_log = out / "loss.jsonl"
    for epoch_index in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = seen = epoch_steps = 0
        for start in range(0, n, BS):
            rows = perm[start:start + BS]
            batch = [item[rows] for item in tensors]
            logits = model(*batch[:-1])
            loss = B.masked_loss(logits, batch[-1])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            tot += loss.item() * len(rows)
            seen += len(rows)
            steps += 1
            epoch_steps += 1
        if epoch_steps != STEPS_PER_EPOCH or seen != n:
            raise SystemExit(f"epoch step count {epoch_steps} seen {seen}")
        if rss_kb() > HOST_RSS_LIMIT_KB:
            raise SystemExit(f"host RSS {rss_kb()} KB exceeds 2 GiB at epoch {epoch_index + 1}")
        train_loss = tot / seen
        losses.append(train_loss)
        completed = epoch_index + 1
        record = {
            "epoch": completed,
            "steps": steps,
            "train_loss": train_loss,
            "seconds": time.perf_counter() - t0,
            "gpu": gpu_line(),
        }
        with loss_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if completed == 1:
            print(
                f"FIRST_EPOCH pid={os.getpid()} host={host} arm={arm} loss={train_loss:.6f} gpu={record['gpu']}",
                flush=True,
            )
        print(f"epoch {completed} loss {train_loss:.6f} steps {steps}", flush=True)
        if completed in SAVE_EPOCHS:
            cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            problems = state_errors(cpu_state)
            if problems:
                raise SystemExit("save refused: " + "; ".join(problems))
            payload = {
                "state": cpu_state,
                "optimizer": opt.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(),
                "completed_epochs": completed,
                "completed_steps": steps,
                "losses": list(losses),
                "parent_checkpoint_sha256": digest,
                "parent_had_optimizer": False,
                "warm_start_from": "weights_only",
                "continuation_seed": CONTINUATION_SEED,
                "config": {
                    "host": host,
                    "arm": arm,
                    "parent": ARMS[arm],
                    "train_lr": lr,
                    "wd": WD,
                    "bs": BS,
                    "drop": DROP,
                    "h": H,
                    "layers": LAYERS,
                    "kernel": KERNEL,
                    "W": W,
                    "parent_seed": 2,
                    "continuation_seed": CONTINUATION_SEED,
                    "completed_epochs": completed,
                    "completed_steps": steps,
                    "epochs_budget": EPOCHS,
                    "steps_per_epoch": STEPS_PER_EPOCH,
                    "total_steps_budget": TOTAL_STEPS,
                    "precision": "fp32",
                    "scheduler": "none",
                    "autocast": False,
                    "grad_clip": GRAD_CLIP,
                    "warm_start_from": "weights_only",
                    "parent_train_bs": PARENT_BATCH,
                    "parent_precision": "bf16_autocast",
                },
            }
            saved = save_epoch(out, payload, completed, steps)
            print(f"SAVED epoch={completed} sha256={saved}", flush=True)
    summary = {
        "host": host,
        "arm": arm,
        "losses": losses,
        "pid": os.getpid(),
        "device": launch["device"],
        "train_seconds": time.perf_counter() - t0,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20,
        "total_steps": steps,
        "exit_ready": steps == TOTAL_STEPS,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    if steps != TOTAL_STEPS:
        raise SystemExit(f"finished at {steps} steps")
    print(f"DONE host={host} arm={arm} loss={losses[-1]:.6f} steps={steps}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-LR FP32 warm start")
    parser.add_argument("--host", required=True, choices=("server", "tower"))
    parser.add_argument("--arm", required=True, choices=("primary", "control"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    train(args.host, args.arm, args.parent, args.out)


if __name__ == "__main__":
    main()
