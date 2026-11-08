"""CPU fp32 export of the pinned no_signal anchor. Writes one submission CSV.

Refuses an unclean packaged config or a checkpoint whose SHA-256 is not the anchor,
before any history or queries are read. Does not train and does not write checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANCHOR_SHA256 = "43dc7050a98dd23ef5b622350e956af3ea53573dcc43fb74945e154cab6bff80"
CONFIG_SHA256 = "c9022a4305be1c4124e6a116cd0cf3132fb5624a206f9b5571090ca3c4769d2f"
PROBABILITY_COLUMNS = ("p_0", "p_1", "p_2", "p_5", "p_20")


def fail(message: str) -> None:
    print(f"REFUSED: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unclean_config(cfg) -> str | None:
    if not isinstance(cfg, dict):
        return "checkpoint config is not an object"
    if cfg.get("no_signal") is not True:
        return "unclean config: no_signal is not True (signal mode refused)"
    required = ("h", "layers", "W", "drop", "epochs", "lr", "wd", "bs", "seed", "no_signal", "out")
    missing = [key for key in required if key not in cfg]
    if missing:
        return f"unclean config: missing {missing}"
    return None


def configs_equal(checkpoint_cfg: dict, packaged_cfg: dict) -> bool:
    if set(checkpoint_cfg) != set(packaged_cfg):
        return False
    for key, value in packaged_cfg.items():
        if checkpoint_cfg[key] != value:
            return False
    return True


def write_submission_csv(path: Path, predictions: dict[str, list[float]]) -> None:
    """Same export as the anchor write_submission_csv: float64 renorm, repr, sorted ids."""
    rows = []
    for query_id in sorted(predictions):
        values = [float(v) for v in predictions[query_id]]
        if any(v != v or v == float("inf") or v == float("-inf") for v in values):
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


def load_anchor():
    checkpoint_path = HERE / "checkpoint.pt"
    config_path = HERE / "config" / "config.json"
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
        if checkpoint_hash != ANCHOR_SHA256:
            fail(f"anchor hash mismatch {checkpoint_hash}")
        fail(f"checkpoint unreadable: {exc}")
    cfg = blob.get("config") if isinstance(blob, dict) else None
    reason = unclean_config(cfg)
    if reason:
        fail(reason)
    if not configs_equal(cfg, packaged_cfg):
        fail("unclean config: checkpoint config does not match config/config.json")
    if checkpoint_hash != ANCHOR_SHA256:
        fail(f"anchor hash mismatch {checkpoint_hash}")
    if not isinstance(blob, dict) or "state" not in blob:
        fail("checkpoint has no state")
    return torch, blob, cfg


def main(argv: list[str] | None = None) -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="CPU fp32 anchor inference")
    parser.add_argument("--data", required=True, help="Round public directory containing test/")
    parser.add_argument("--out", required=True, help="Destination submission CSV path")
    args = parser.parse_args(argv)
    out_path = Path(args.out)
    torch, blob, cfg = load_anchor()
    if cfg["h"] != 128 or cfg["layers"] != 4 or cfg["W"] != 64 or cfg["drop"] != 0.2:
        fail("unclean config: architecture does not match the anchor")
    import model as model_mod
    from data import load_test_split
    from model import CNN, build_query, predict, to_preds

    model_mod.USE_SIGNAL = False
    net = CNN(cfg["h"], cfg["layers"], 3, cfg["drop"], cfg["W"], use_signal=False)
    net.load_state_dict(blob["state"])
    net.eval()
    if net.use_signal or net.sig is not None:
        fail("signal mode refused")
    try:
        data = load_test_split(args.data)
    except (OSError, ValueError, KeyError) as exc:
        fail(f"missing or unreadable test history/queries: {exc}")
    if not data.queries:
        fail("test queries are empty")
    infer_started = time.perf_counter()
    features, episodes = build_query(data, cfg["W"])
    if any(int(row.sum()) != 0 for row in features[3]):
        fail("signal features were not all zero")
    probabilities = predict(net, features, "cpu")
    predictions = to_preds(probabilities, episodes, data)
    if len(predictions) != len(data.queries):
        fail("query coverage mismatch")
    write_submission_csv(out_path, predictions)
    elapsed = time.perf_counter() - infer_started
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"wrote {out_path} queries={len(predictions)} "
        f"infer_s={elapsed:.6f} total_s={time.perf_counter() - started:.6f} "
        f"max_rss_kb={rss_kb} device=cpu dtype=fp32 threads=2",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
