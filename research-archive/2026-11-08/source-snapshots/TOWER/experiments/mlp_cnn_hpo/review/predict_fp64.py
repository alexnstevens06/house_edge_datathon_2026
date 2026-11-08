"""CPU float64 export. Same guards as predict_guarded.py, dtype change only.

Usage: predict_fp64.py RUN_DIR OUT_DIR

base_cnn.CNN.forward (server_snapshot/source/base_cnn.py) builds the mask and
tick features with Tensor.float(), which is float32. After model.double() the
Linear weights are float64, and this torch build rejects a float32 input to
those linears. Addition of a float32 tensor into a float64 embedding would
promote, but the forward never gets that far unless the .float() casts are
float64. This script therefore redirects Tensor.float to float64 only for the
inference calls, then restores it. The original CNN.forward, blocks, and head
still run. No autocast. Softmax is float64. Query features and order come from
base_cnn.build_query / to_preds.
"""
import json
import os
import platform
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["ROCR_VISIBLE_DEVICES"] = ""

from predict_guarded import die, guards, sha256_file


def model_name():
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def bind_float64_casts():
    """Return the previous Tensor.float. Redirect is dtype-only."""
    original = torch.Tensor.float

    def float64(self, memory_format=torch.preserve_format):
        return self.to(dtype=torch.float64, memory_format=memory_format)

    torch.Tensor.float = float64
    return original


def predict_fp64(model, Q, bs=256):
    """Original module forward, no autocast, softmax in float64."""
    model.eval()
    chunks = []
    logit_dtypes = []
    with torch.no_grad():
        for i in range(0, len(Q[0]), bs):
            batch = [a[i:i + bs] for a in Q]
            logits = model(*batch)
            logit_dtypes.append(str(logits.dtype).replace("torch.", ""))
            if logits.dtype != torch.float64:
                die("logits dtype is %s, want float64" % logits.dtype)
            chunks.append(torch.softmax(logits, dim=-1).cpu())
    probs = torch.cat(chunks)
    if probs.dtype != torch.float64:
        die("softmax dtype is %s, want float64" % probs.dtype)
    return probs.numpy(), logit_dtypes


def main(argv):
    if len(argv) != 3:
        die("usage: predict_fp64.py RUN_DIR OUT_DIR")
    run_dir = Path(argv[1]).resolve()
    out_dir = Path(argv[2]).resolve()
    if not run_dir.is_dir():
        die("RUN_DIR is not a directory")
    cfg, state = guards(run_dir)

    global torch
    import torch
    torch.set_num_threads(2)
    torch.backends.mkldnn.enabled = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        die("use_deterministic_algorithms failed: %s" % exc)

    review = Path(__file__).resolve().parent
    repo = review.parents[2]
    snap_src = review.parent / "server_snapshot" / "source"
    sys.path.insert(0, str(snap_src))
    sys.path.insert(0, str(repo / "src"))
    import base_cnn as B
    B.USE_SIGNAL = False
    from house_edge.data import load_split
    from house_edge.scoring import write_submission_csv

    model = B.CNN(cfg["h"], cfg["layers"], 3, cfg["drop"], cfg["W"], use_signal=False)
    model.load_state_dict(state)
    model.double()
    model.eval()
    if model.use_signal or model.sig is not None:
        die("signal mode refused")
    if any(p.dtype != torch.float64 for p in model.parameters()):
        die("parameters are not float64 after double()")

    pub = repo / "kit" / "public"
    va = load_split(pub / "validation")
    te = load_split(pub / "test")
    out_dir.mkdir(parents=True, exist_ok=True)

    observed = {}

    def hook(name):
        def fn(module, inputs, output):
            observed[name] = {
                "input_dtype": str(inputs[0].dtype).replace("torch.", ""),
                "output_dtype": str(output.dtype).replace("torch.", ""),
            }
        return fn

    model.mask.register_forward_hook(hook("mask"))
    model.tick.register_forward_hook(hook("tick"))
    model.pay.register_forward_hook(hook("pay"))

    Qv, ev = B.build_query(va, cfg["W"])
    unpatched = None
    try:
        with torch.no_grad():
            sample = model(*[t[:1] for t in Qv])
        unpatched = "dtype=%s" % str(sample.dtype).replace("torch.", "")
    except Exception as exc:
        unpatched = "error=%s" % exc
    unpatched_io = dict(observed)
    observed.clear()

    original_float = bind_float64_casts()
    try:
        with torch.no_grad():
            patched_sample = model(*[t[:1] for t in Qv])
        if patched_sample.dtype != torch.float64:
            die("patched forward dtype is %s" % patched_sample.dtype)
        pv, dtypes_v = predict_fp64(model, Qv)
        Qt, et = B.build_query(te, cfg["W"])
        pt, dtypes_t = predict_fp64(model, Qt)
        write_submission_csv(out_dir / "validation_predictions_cpu.csv", B.to_preds(pv, ev, va))
        write_submission_csv(out_dir / "test_predictions_cpu.csv", B.to_preds(pt, et, te))
    finally:
        torch.Tensor.float = original_float

    val_path = out_dir / "validation_predictions_cpu.csv"
    te_path = out_dir / "test_predictions_cpu.csv"
    env = {
        "precision": "fp64",
        "precision_change": True,
        "torch": torch.__version__,
        "torch_config": torch.__config__.show(),
        "platform": platform.platform(),
        "cpu_model": model_name(),
        "num_threads": int(torch.get_num_threads()),
        "deterministic_algorithms": True,
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "checkpoint_sha256": sha256_file(run_dir / "checkpoint.pt"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "validation_csv_sha256": sha256_file(val_path),
        "test_csv_sha256": sha256_file(te_path),
        "parameter_dtype": "float64",
        "unpatched_forward": unpatched,
        "unpatched_module_io_dtype": unpatched_io,
        "dtype_patch": (
            "Tensor.float redirected to float64 during inference only. "
            "base_cnn.CNN.forward uses .float() for mask and tick features. "
            "Those float32 tensors do not match float64 Linear weights. "
            "Original forward body, blocks, embeddings, and head are unchanged."
        ),
        "module_io_dtype": observed,
        "logits_dtype_validation_batches": dtypes_v,
        "logits_dtype_test_batches": dtypes_t,
        "softmax": "float64",
        "autocast": False,
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print("wrote", val_path, "queries", len(va.queries))
    print("wrote", te_path, "queries", len(te.queries))
    print("unpatched_forward", unpatched)
    print("module_io_dtype", json.dumps(observed))
    print("env", out_dir / "env.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except SystemExit:
        raise
    except Exception as exc:
        die("export failed: %s" % exc)
