"""EXPLICIT PRECISION CHANGE. FP64 diagnostic only. Not the entry default.

Same checkpoint values and the same architecture as the anchor. Tick features
use float64 instead of the entry's .float() casts, and softmax is not cast
back to float32. Nothing here replaces entry/, first.csv, or the FP32 path.

From the repo root:

  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/cnn_runtime/stability/diagnose_fp64.py --run-all --data kit/public --out-dir experiments/cnn_runtime/stability/out_fp64
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import diagnose

HERE = Path(__file__).resolve().parent
ENTRY = diagnose.ENTRY
CONTROLS = diagnose.CONTROLS


def forward_fp64(module, x, t, ty, sg):
    import torch
    B, W, M = x.shape
    x = x.long().transpose(1, 2)
    mask = (x >= 0).to(dtype=torch.float64).unsqueeze(-1)
    idx = torch.where(x >= 0, x, torch.full_like(x, 5))
    ticks = (
        t.view(B, 1, 1).to(dtype=torch.float64)
        - W
        + torch.arange(W, device=x.device, dtype=torch.float64).view(1, 1, W)
    ).expand(B, M, W)
    features = torch.stack([ticks / 256.0, (t.view(B, 1, 1).to(dtype=torch.float64) - ticks) / W], -1)
    meta = module.typ(ty) + module.slot.weight[None]
    if module.use_signal:
        raise RuntimeError("signal mode refused")
    hidden = module.pay(idx) + module.mask(mask) + module.tick(features) + meta.unsqueeze(2)
    for block in module.blocks:
        hidden = block(hidden)
    last = hidden[:, :, -1]
    pooled = last.mean(1, keepdim=True).expand_as(last)
    return module.head(torch.cat([last, pooled], -1))


def predict_fp64(module, features, bs=256):
    import torch
    module.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(features[0]), bs):
            batch = [item[start:start + bs] for item in features]
            logits = forward_fp64(module, *batch)
            pieces.append(torch.softmax(logits, -1).cpu())
    return torch.cat(pieces).numpy()


def run_control(name: str, data: str, out: Path) -> None:
    set_keys = diagnose.apply_control(name)
    import torch
    torch.set_num_threads(2)
    if name == "mkldnn_off":
        torch.backends.mkldnn.enabled = False
    sys.path.insert(0, str(ENTRY))
    import predict
    from model import CNN, build_query, to_preds
    torch_mod, blob, cfg = predict.load_anchor()
    del torch_mod
    net = CNN(cfg["h"], cfg["layers"], 3, cfg["drop"], cfg["W"], use_signal=False)
    net.load_state_dict(blob["state"])
    net.double()
    net.eval()
    loaded = load_test(data)
    features, episodes = build_query(loaded, cfg["W"])
    if any(int(row.sum()) != 0 for row in features[3]):
        raise SystemExit("signal features were not zero")
    probabilities = predict_fp64(net, features)
    predictions = to_preds(probabilities, episodes, loaded)
    out.parent.mkdir(parents=True, exist_ok=True)
    predict.write_submission_csv(out, predictions)
    capability = None
    getter = getattr(torch.backends.cpu, "get_cpu_capability", None)
    if getter is not None:
        capability = getter()
    manifest = {
        "label": "PRECISION_CHANGE",
        "precision": "fp64",
        "precision_change": True,
        "replaces_entry_default": False,
        "control": name,
        "process_env_set": set_keys,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cpu_model": diagnose.cpu_model(),
        "cpu_capability": capability,
        "mkldnn_enabled_after_control": bool(torch.backends.mkldnn.enabled),
        "num_threads": int(torch.get_num_threads()),
        "checkpoint_sha256": diagnose.sha256_file(ENTRY / "checkpoint.pt"),
        "entry_sha256": diagnose.entry_hashes(),
        "csv_sha256": diagnose.sha256_file(out),
        "csv_bytes": out.stat().st_size,
        "n_rows": 1536,
        "note": "float64 forward is a separate diagnostic. The entry remains fp32.",
        "command": [
            sys.executable,
            str(Path(__file__).resolve()),
            "--control",
            name,
            "--data",
            data,
            "--out",
            str(out),
        ],
    }
    if manifest["checkpoint_sha256"] != diagnose.ANCHOR:
        raise SystemExit("anchor checkpoint hash changed")
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"precision_change": True, "control": name, "csv_sha256": manifest["csv_sha256"], "cpu_capability": capability}))


def load_test(data: str):
    sys.path.insert(0, str(ENTRY))
    from data import load_test_split
    return load_test_split(data)


def run_all(data: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in CONTROLS:
        out = out_dir / name / "submission.csv"
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--control", name, "--data", data, "--out", str(out)],
            cwd=Path.cwd(),
            env=diagnose.child_env(name),
            text=True,
        )
        if proc.returncode != 0 or not out.is_file():
            failures.append({"control": name, "returncode": proc.returncode})
    produced = [name for name in CONTROLS if (out_dir / name / "submission.csv").is_file()]
    targets = [("fp32_reference", diagnose.REF), ("fp32_tower_first", diagnose.RUNTIME / "first.csv")]
    targets += [(f"fp64_{name}", out_dir / name / "submission.csv") for name in produced]
    targets = [(name, path) for name, path in targets if path.is_file()]
    pairs = []
    for i, (left_name, left_path) in enumerate(targets):
        for right_name, right_path in targets[i + 1:]:
            metrics = diagnose.pair_metrics(left_path, right_path)
            metrics["left"] = left_name
            metrics["right"] = right_name
            pairs.append(metrics)
    fp64_pairs = [item for item in pairs if item["left"].startswith("fp64_") and item["right"].startswith("fp64_")]
    summary = {
        "label": "PRECISION_CHANGE",
        "precision": "fp64",
        "precision_change": True,
        "replaces_entry_default": False,
        "host_role": "tower_only",
        "not_cross_host_proof": True,
        "tolerance_max_abs_after_row_normalization": diagnose.TOL,
        "relative_error_used_as_tolerance": False,
        "failures": failures,
        "fp64_controls_all_within_1e-8": all(item["within_1e-8"] for item in fp64_pairs) if fp64_pairs else False,
        "pairs": pairs,
        "command": [sys.executable, str(Path(__file__).resolve()), "--run-all", "--data", data, "--out-dir", str(out_dir)],
    }
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "precision_change": True,
        "wrote": str(out_dir / "comparison.json"),
        "failures": failures,
        "fp64_controls_all_within_1e-8": summary["fp64_controls_all_within_1e-8"],
        "pairs": [
            {
                "left": item["left"],
                "right": item["right"],
                "bitwise": item["bitwise"],
                "max_abs_after_row_normalization": item["max_abs_after_row_normalization"],
                "within_1e-8": item["within_1e-8"],
            }
            for item in pairs
        ],
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="PRECISION CHANGE fp64 diagnostic, not the entry")
    parser.add_argument("--control", choices=CONTROLS)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--data", default="kit/public")
    parser.add_argument("--out")
    parser.add_argument("--out-dir", default=str(HERE / "out_fp64"))
    args = parser.parse_args()
    if args.run_all:
        run_all(args.data, Path(args.out_dir))
        return
    if not args.control or not args.out:
        raise SystemExit("pass --run-all or both --control and --out")
    run_control(args.control, args.data, Path(args.out))


if __name__ == "__main__":
    main()
