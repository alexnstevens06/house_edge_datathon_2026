"""CPU fp32 export for a pinned cnn03 run. Refuses before any dataset read.

Usage: predict_guarded.py RUN_DIR OUT_DIR
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def die(msg):
    print("REFUSED: " + msg, file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guards(run_dir):
    """Exit 2 before any public-data load. Returns (config, state_dict)."""
    review = Path(__file__).resolve().parent
    snap_src = review.parent / "server_snapshot" / "source"
    pin_path = review / "PINNED.json"
    if not pin_path.is_file():
        die("pin file missing: %s" % pin_path)
    pin = json.loads(pin_path.read_text())
    cfg_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoint.pt"
    if not cfg_path.is_file():
        die("config.json missing")
    try:
        cfg = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as exc:
        die("config.json is not json (%s)" % exc)
    if cfg.get("model") != "cnn":
        die("config model is %r, want 'cnn'" % cfg.get("model"))
    contaminated = sorted(p.name for p in run_dir.glob("CONTAMINATED*"))
    if contaminated:
        die("run dir has %s" % ", ".join(contaminated))
    if not ckpt_path.is_file():
        die("checkpoint.pt missing")
    for name, want in pin.get("sources", {}).items():
        src = snap_src / name
        got = sha256_file(src)
        if got != want:
            die("source %s sha256 %s != pin %s" % (name, got, want))
    import torch
    try:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(ckpt_path, map_location="cpu")
    state = blob["state"] if isinstance(blob, dict) and "state" in blob else blob
    sig = [k for k in state if "sig" in k.split(".")]
    if sig:
        die("checkpoint contains sig params: %s" % ", ".join(sig))
    digest = sha256_file(ckpt_path)
    expected = pin.get("checkpoints", {}).get(run_dir.name)
    if expected is None:
        die("no pin for run dir %s" % run_dir.name)
    if digest != expected:
        die("checkpoint sha256 %s != pin %s" % (digest, expected))
    return cfg, state


def main(argv):
    if len(argv) != 3:
        die("usage: predict_guarded.py RUN_DIR OUT_DIR")
    run_dir = Path(argv[1]).resolve()
    out_dir = Path(argv[2]).resolve()
    if not run_dir.is_dir():
        die("RUN_DIR is not a directory")
    cfg, state = guards(run_dir)

    import torch
    review = Path(__file__).resolve().parent
    repo = review.parents[2]
    snap_src = review.parent / "server_snapshot" / "source"
    sys.path.insert(0, str(snap_src))
    sys.path.insert(0, str(repo / "src"))
    import base_cnn as B
    B.USE_SIGNAL = False
    from house_edge.data import load_split
    from house_edge.scoring import write_submission_csv

    torch.set_num_threads(2)
    model = B.CNN(cfg["h"], cfg["layers"], 3, cfg["drop"], cfg["W"], use_signal=False)
    model.load_state_dict(state)
    model.eval()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        die("parameters are not fp32")
    pub = repo / "kit" / "public"
    va = load_split(pub / "validation")
    te = load_split(pub / "test")
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, data, name in (
        ("validation", va, "validation_predictions_cpu.csv"),
        ("test", te, "test_predictions_cpu.csv"),
    ):
        Q, eps = B.build_query(data, cfg["W"])
        preds = B.to_preds(B.predict(model, Q, "cpu"), eps, data)
        write_submission_csv(out_dir / name, preds)
        print("wrote", split, out_dir / name, "queries", len(preds))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except SystemExit:
        raise
    except Exception as exc:
        die("export failed: %s" % exc)
