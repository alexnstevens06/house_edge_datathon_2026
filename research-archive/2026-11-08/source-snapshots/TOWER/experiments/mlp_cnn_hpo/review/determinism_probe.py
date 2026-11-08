"""Cross-machine CSV diffs and a bounded CPU determinism sweep. No training.

Parent (no args): env info, tower-recompute vs server max abs, 6-decimal rounding vs R.
Child (--infer NAME): one s0 validation+test export under a named CPU setting.
"""
import argparse
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[2]
SNAP = REVIEW.parent / "server_snapshot"
RUNS = SNAP / "runs"
SRC = SNAP / "source"
PUB = REPO / "kit" / "public"
TMP = REVIEW / "tmp"
PY = REPO / ".venv" / "bin" / "python"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "kit"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_rows(path):
    import csv
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    out = {}
    for row in rows[1:]:
        out[row[0]] = [float(x) for x in row[1:]]
    return out


def kit_rows(path):
    from slot_forecasting.io import read_submission
    return read_submission(path)


def max_abs(a, b):
    worst = 0.0
    n = 0
    where = ""
    for key, pa in a.items():
        pb = b[key]
        for i, (x, y) in enumerate(zip(pa, pb)):
            n += 1
            d = abs(float(x) - float(y))
            if d > worst:
                worst = d
                where = "%s[%d]" % (key, i)
    return worst, where, n


def round6_renorm(rows):
    out = {}
    for key, probs in rows.items():
        rounded = [round(float(p), 6) for p in probs]
        total = sum(rounded)
        out[key] = [v / total for v in rounded]
    return out


def score_full(preds):
    from house_edge.scoring import load_official_validation, score_predictions
    official = load_official_validation(PUB / "validation")
    return score_predictions(preds, official, None)


def report_pair(label, left, right):
    worst, where, n = max_abs(left, right)
    print("%s max_abs=%.12g at %s n=%d" % (label, worst, where, n), flush=True)
    return worst


def env_block():
    import torch
    print("torch", torch.__version__, flush=True)
    print("python", platform.python_version(), flush=True)
    cpu = subprocess.check_output(["lscpu"], text=True, errors="replace")
    for line in cpu.splitlines():
        if line.startswith("Model name:") or line.startswith("Vendor ID:") or line.startswith("CPU(s):"):
            print(line, flush=True)
    print("TORCH_CONFIG_BEGIN", flush=True)
    print(torch.__config__.show(), flush=True)
    print("TORCH_CONFIG_END", flush=True)
    print("splits_sha256", sha256_file(REPO / "runs" / "splits.json"), flush=True)
    anchors = []
    for seed in (0, 1, 2):
        import json
        met = json.loads((RUNS / ("anchor_cnn_s%d" % seed) / "metrics.json").read_text())
        fin = json.loads((RUNS / ("final_cnn03_s%d" % seed) / "metrics.json").read_text())
        anchors.append(met["full"]["R"])
        print(
            "seed %d anchor_full %.12g final_tune %.12g final_confirm %.12g final_full %.12g"
            % (seed, met["full"]["R"], fin["tune"]["R"], fin["confirm"]["R"], fin["full"]["R"]),
            flush=True,
        )
    print("anchor_full_mean %.12g" % (sum(anchors) / len(anchors)), flush=True)


def compare_existing():
    print("COMPARE_EXISTING", flush=True)
    for seed in (0, 1, 2):
        for split, fname in (
            ("validation", "validation_predictions_cpu.csv"),
            ("test", "test_predictions_cpu.csv"),
        ):
            server = RUNS / ("final_cnn03_s%d" % seed) / fname
            tower = TMP / ("re_s%d_%s" % (seed, fname))
            raw_s, raw_t = raw_rows(server), raw_rows(tower)
            kit_s, kit_t = kit_rows(server), kit_rows(tower)
            report_pair("raw s%d %s tower_vs_server" % (seed, split), raw_t, raw_s)
            report_pair("kit s%d %s tower_vs_server" % (seed, split), kit_t, kit_s)
            if split == "validation":
                r_server = score_full({k: list(map(float, v)) for k, v in kit_s.items()})["R"]
                r_tower = score_full({k: list(map(float, v)) for k, v in kit_t.items()})["R"]
                rounded = round6_renorm(kit_t)
                r_round = score_full(rounded)["R"]
                report_pair("kit s%d validation round6(tower)_vs_server" % seed, rounded, kit_s)
                report_pair("kit s%d validation round6(tower)_vs_tower" % seed, rounded, kit_t)
                report_pair("kit s%d validation round6(server)_vs_server" % seed, round6_renorm(kit_s), kit_s)
                print(
                    "R s%d server %.12g tower %.12g round6_tower %.12g d_tower %.3g d_round %.3g"
                    % (seed, r_server, r_tower, r_round, r_tower - r_server, r_round - r_server),
                    flush=True,
                )
    r1 = REVIEW / "rc_s0_run1" / "validation_predictions_cpu.csv"
    r2 = REVIEW / "rc_s0_run2" / "validation_predictions_cpu.csv"
    t1 = REVIEW / "rc_s0_run1" / "test_predictions_cpu.csv"
    t2 = REVIEW / "rc_s0_run2" / "test_predictions_cpu.csv"
    if r1.is_file() and r2.is_file():
        print("rc_s0_run1_vs_run2 validation bytes", r1.read_bytes() == r2.read_bytes(), flush=True)
        print("rc_s0_run1_vs_run2 test bytes", t1.read_bytes() == t2.read_bytes(), flush=True)


def infer(name):
    import torch
    import base_cnn as B
    from house_edge.data import load_split
    from house_edge.scoring import write_submission_csv

    B.USE_SIGNAL = False
    spec = {
        "det_t1": dict(threads=1, deterministic=True, mkldnn=False, dtype="fp32"),
        "det_t2": dict(threads=2, deterministic=True, mkldnn=False, dtype="fp32"),
        "fp64_t2": dict(threads=2, deterministic=False, mkldnn=True, dtype="fp64"),
    }[name]
    torch.set_num_threads(spec["threads"])
    torch.backends.mkldnn.enabled = spec["mkldnn"]
    try:
        torch.use_deterministic_algorithms(spec["deterministic"])
        det_status = "ok"
    except Exception as exc:
        det_status = "ERR %s" % exc
    print("INFER", name, spec, "det", det_status, "threads", torch.get_num_threads(), flush=True)
    if det_status != "ok":
        return 2
    run = RUNS / "final_cnn03_s0"
    import json
    cfg = json.loads((run / "config.json").read_text())
    blob = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    model = B.CNN(cfg["h"], cfg["layers"], 3, cfg["drop"], cfg["W"], use_signal=False)
    model.load_state_dict(blob["state"])
    model.eval()
    if spec["dtype"] == "fp64":
        # base_cnn.CNN.forward calls Tensor.float() for the mask and tick
        # features (base_cnn.py lines 71-73), so model.double() cannot run
        # the stock forward. This local mirror keeps every activation double.
        model = model.double()

        def logits(batch):
            x, t, ty, sg = batch
            bsz, width, machines = x.shape
            x = x.long().transpose(1, 2)
            observed = (x >= 0).double().unsqueeze(-1)
            idx = torch.where(x >= 0, x, torch.full_like(x, 5))
            ticks = (t.view(bsz, 1, 1).double() - width + torch.arange(width).view(1, 1, width)).expand(bsz, machines, width)
            tf = torch.stack([ticks / 256.0, (t.view(bsz, 1, 1).double() - ticks) / float(width)], -1)
            meta = model.typ(ty) + model.slot.weight[None]
            if model.use_signal:
                meta = meta + model.sig(sg)
            z = model.pay(idx) + model.mask(observed) + model.tick(tf) + meta.unsqueeze(2)
            for block in model.blocks:
                z = block(z)
            last = z[:, :, -1]
            pooled = last.mean(1, keepdim=True).expand_as(last)
            return model.head(torch.cat([last, pooled], -1))
    else:
        def logits(batch):
            return model(*batch)

    def predict(Q):
        out = []
        with torch.no_grad():
            for i in range(0, len(Q[0]), 256):
                batch = [a[i:i + 256] for a in Q]
                lg = logits(batch)
                prob = torch.softmax(lg.double() if spec["dtype"] == "fp64" else lg.float(), -1)
                out.append(prob.cpu())
        return torch.cat(out).double().numpy()

    for split in ("validation", "test"):
        data = load_split(PUB / split)
        Q, eps = B.build_query(data, cfg["W"])
        preds = B.to_preds(predict(Q), eps, data)
        out = TMP / ("det_%s_s0_%s_predictions_cpu.csv" % (name, split))
        write_submission_csv(out, preds)
        server = run / ("%s_predictions_cpu.csv" % split)
        tower = TMP / ("re_s0_%s_predictions_cpu.csv" % split)
        report_pair("infer %s %s vs_server_kit" % (name, split), kit_rows(out), kit_rows(server))
        report_pair("infer %s %s vs_tower_kit" % (name, split), kit_rows(out), kit_rows(tower))
        print("infer %s %s vs_tower_bytes" % (name, split), out.read_bytes() == tower.read_bytes(), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", choices=("det_t1", "det_t2", "fp64_t2"), default=None)
    args = parser.parse_args()
    if args.infer:
        return infer(args.infer)
    env_block()
    compare_existing()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    for name in ("det_t1", "det_t2", "fp64_t2"):
        print("SPAWN", name, flush=True)
        proc = subprocess.run(
            [str(PY), str(Path(__file__).resolve()), "--infer", name],
            cwd=str(REPO),
            env=env,
        )
        print("SPAWN_EXIT", name, proc.returncode, flush=True)
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
