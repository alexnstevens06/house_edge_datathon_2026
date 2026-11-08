"""CPU float64 export of the original curve_C_lr003_s2 parent.

This is a separately staged candidate. It is not the CNN03 seed 0 production
package and it is not a production replacement. Checkpoint bytes stay the
saved parent values and are cast with model.double(). The parent config has
no kernel field. Kernel 3 is the measured conv weight shape (128, 128, 3),
checked before the network is built. Tensor float is not reassigned.
Refuses a bad config, checkpoint hash, key set, shape, saved dtype, or
non-finite weight before any history or queries are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
CHECKPOINT_SHA256 = "bcef4104371484eedabce1ab1aca5fc340dcf0e4e2c4e6b2c946f0a1789e1f71"
CONFIG_SHA256 = "a715bfe05eb00494065c7dfcc3ad6d71a3b2bc4ee6788006cf07ce86f3e43695"
PROBABILITY_COLUMNS = ("p_0", "p_1", "p_2", "p_5", "p_20")
KERNEL = 3
H = 128
LAYERS = 4
WIDTH = 64
DROP = 0.3
EXPECTED_CONFIG = {
    "lr": 0.003,
    "wd": 0.00895,
    "bs": 512,
    "epochs": 30,
    "drop": 0.3,
    "h": 128,
    "layers": 4,
    "W": 64,
    "seed": 2,
    "pct_start": 0.3,
    "eval_every": 2,
    "out": "experiments/cnn03_convergence/runs/curve_C_lr003_s2",
}


def fail(message: str) -> None:
    print(f"REFUSED: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        "pay.weight": (6, H),
        "mask.weight": (H, 1),
        "mask.bias": (H,),
        "typ.weight": (3, H),
        "slot.weight": (8, H),
        "tick.weight": (H, 2),
        "tick.bias": (H,),
        "head.0.weight": (H, 2 * H),
        "head.0.bias": (H,),
        "head.2.weight": (5, H),
        "head.2.bias": (5,),
    }
    for i in range(LAYERS):
        prefix = f"blocks.{i}"
        shapes[f"{prefix}.conv.weight"] = (H, H, KERNEL)
        shapes[f"{prefix}.conv.bias"] = (H,)
        shapes[f"{prefix}.mix.weight"] = (H, 2 * H)
        shapes[f"{prefix}.mix.bias"] = (H,)
        shapes[f"{prefix}.n1.weight"] = (H,)
        shapes[f"{prefix}.n1.bias"] = (H,)
        shapes[f"{prefix}.n2.weight"] = (H,)
        shapes[f"{prefix}.n2.bias"] = (H,)
    return shapes


def unclean_config(cfg) -> str | None:
    if not isinstance(cfg, dict):
        return "checkpoint config is not an object"
    if cfg.get("no_signal") is False:
        return "unclean config: signal mode refused"
    if set(cfg) != set(EXPECTED_CONFIG):
        return "unclean config: keys do not match the curve_C_lr003_s2 parent"
    for key, value in EXPECTED_CONFIG.items():
        if cfg[key] != value:
            return f"unclean config: {key} does not match the curve_C_lr003_s2 parent"
    return None


def configs_equal(checkpoint_cfg: dict, packaged_cfg: dict) -> bool:
    if set(checkpoint_cfg) != set(packaged_cfg):
        return False
    for key, value in packaged_cfg.items():
        if checkpoint_cfg[key] != value:
            return False
    return True


def signal_keys(state) -> list[str]:
    return [str(key) for key in state if key == "sig.weight" or str(key).startswith("sig.")]


def state_reason(state) -> str | None:
    if not isinstance(state, dict):
        return "checkpoint state is not an object"
    leaked = signal_keys(state)
    if leaked:
        return "signal mode refused: " + ", ".join(leaked)
    want = expected_shapes()
    if len(want) != 43:
        return "internal shape table is not 43 tensors"
    if set(state) != set(want):
        missing = sorted(set(want) - set(state))
        extra = sorted(set(state) - set(want))
        return f"architecture keys mismatch missing={missing} extra={extra}"
    import torch
    for key, shape in want.items():
        tensor = state[key]
        got = tuple(tensor.shape)
        if got != shape:
            return f"{key} shape {got} != {shape}"
        if tensor.dtype != torch.float32:
            return f"{key} saved dtype {tensor.dtype} is not float32"
        if not bool(torch.isfinite(tensor).all()):
            return f"{key} is not finite"
    return None


def write_submission_csv(path: Path, predictions: dict[str, list[float]]) -> None:
    rows = []
    for query_id in sorted(predictions):
        values = [float(v) for v in predictions[query_id]]
        if any(v != v or v in (float("inf"), float("-inf")) for v in values):
            fail(f"non-finite probabilities for {query_id}")
        total = sum(values)
        if not total > 0.0:
            fail(f"probability mass is not positive for {query_id}")
        values = [v / total for v in values]
        rows.append([query_id, *[repr(v) for v in values]])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("query_id", *PROBABILITY_COLUMNS))
        writer.writerows(rows)


def load_pinned():
    config_path = HERE / "config" / "config.json"
    checkpoint_path = HERE / "checkpoint.pt"
    if not config_path.is_file():
        fail(f"missing {config_path}")
    config_hash = sha256_file(config_path)
    if config_hash != CONFIG_SHA256:
        fail(f"unclean config hash {config_hash}")
    packaged_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    packaged_reason = unclean_config(packaged_cfg)
    if packaged_reason:
        fail(packaged_reason)
    if not checkpoint_path.is_file():
        fail(f"missing {checkpoint_path}")
    checkpoint_hash = sha256_file(checkpoint_path)
    import torch
    torch.set_num_threads(2)
    try:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        if checkpoint_hash != CHECKPOINT_SHA256:
            fail(f"checkpoint hash mismatch {checkpoint_hash}")
        fail(f"checkpoint unreadable: {exc}")
    if not isinstance(blob, dict) or set(blob) != {"config", "state"}:
        fail("checkpoint is not weights-only")
    reason = state_reason(blob["state"])
    if reason:
        fail(reason)
    cfg = blob.get("config")
    config_reason = unclean_config(cfg)
    if config_reason:
        fail(config_reason)
    if not configs_equal(cfg, packaged_cfg):
        fail("unclean config: checkpoint config does not match config/config.json")
    if checkpoint_hash != CHECKPOINT_SHA256:
        fail(f"checkpoint hash mismatch {checkpoint_hash}")
    return torch, blob, cfg


def main(argv: list[str] | None = None) -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="CPU float64 curve_C_lr003_s2 parent inference")
    parser.add_argument("--data", required=True, help="Round public directory")
    parser.add_argument("--out", required=True, help="Destination submission CSV path")
    parser.add_argument("--split", default="test", choices=("test", "val", "validation"))
    args = parser.parse_args(argv)
    split = "validation" if args.split == "val" else args.split
    torch, blob, cfg = load_pinned()
    import model as model_mod
    from data import load_public_split
    from model import CNN, build_query, predict, to_preds

    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        fail("autocast is enabled")
    model_mod.USE_SIGNAL = False
    net = CNN(cfg["h"], cfg["layers"], KERNEL, cfg["drop"], cfg["W"], use_signal=False)
    for index, block in enumerate(net.blocks):
        if tuple(block.conv.weight.shape) != (H, H, KERNEL):
            fail(f"blocks.{index}.conv.weight kernel contract failed")
    net.load_state_dict(blob["state"])
    net.double()
    net.eval()
    if net.use_signal or net.sig is not None:
        fail("signal mode refused")
    floating = [p for p in net.parameters() if p.is_floating_point()]
    if len(floating) != 43 or any(p.dtype != torch.float64 for p in floating):
        fail("floating parameters are not 43 float64 tensors")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        fail("autocast is enabled")
    try:
        data = load_public_split(args.data, split)
    except (OSError, ValueError, KeyError) as exc:
        fail(f"missing or unreadable {split} history/queries: {exc}")
    if not data.queries:
        fail(f"{split} queries are empty")
    infer_started = time.perf_counter()
    features, episodes = build_query(data, cfg["W"])
    if any(int(row.sum()) != 0 for row in features[3]):
        fail("signal features were not all zero")
    probabilities = predict(net, features, "cpu")
    if str(probabilities.dtype) != "float64":
        fail(f"predict dtype {probabilities.dtype}")
    predictions = to_preds(probabilities, episodes, data)
    if len(predictions) != len(data.queries):
        fail("query coverage mismatch")
    write_submission_csv(Path(args.out), predictions)
    elapsed = time.perf_counter() - infer_started
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"wrote {args.out} split={split} queries={len(predictions)} "
        f"infer_s={elapsed:.6f} total_s={time.perf_counter() - started:.6f} "
        f"max_rss_kb={rss_kb} device=cpu dtype=fp64 threads=2",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
