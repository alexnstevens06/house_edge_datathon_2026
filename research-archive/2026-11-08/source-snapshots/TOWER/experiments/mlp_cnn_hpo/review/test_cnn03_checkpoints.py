"""Bit-identical checks for the three final_cnn03 CPU checkpoints.

Plain runner. No tolerance. Prints PASS or FAIL per check.
"""
import copy
import csv
import hashlib
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[2]
SNAP = REVIEW.parent / "server_snapshot"
SRC = SNAP / "source"
RUNS = SNAP / "runs"
PUB = REPO / "kit" / "public"
TMP = REVIEW / "tmp"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "kit"))

import numpy as np
import torch
import torch.nn.functional as F

import base_cnn as B
from house_edge.data import SplitData, episode_conditions, load_split, subset_split_data
from house_edge.scoring import write_submission_csv
import slot_forecasting.contest as kit_contest
import house_edge.scoring as scoring

B.USE_SIGNAL = False
torch.set_num_threads(2)
torch.manual_seed(0)

FAILS = []


def check(name, ok, detail=""):
    print(("PASS" if ok else "FAIL") + " " + name + ((" | " + detail) if detail else ""))
    if not ok:
        FAILS.append(name + ((" | " + detail) if detail else ""))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(run_dir):
    cfg = json.loads((run_dir / "config.json").read_text())
    blob = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    state = blob["state"]
    model = B.CNN(cfg["h"], cfg["layers"], 3, cfg["drop"], cfg["W"], use_signal=False)
    model.load_state_dict(state)
    model.eval()
    return cfg, blob, state, model


def has_missing_gap(obs):
    for slot in range(obs.shape[1]):
        col = obs[:, slot]
        seen = np.flatnonzero(col >= 0)
        if len(seen) >= 2 and (col[seen[0]:seen[-1] + 1] < 0).any():
            return True
    return False


def mutate_obs(obs, t, slot, mode):
    out = np.array(obs, copy=True)
    if out.shape[0] <= t:
        extra = np.full((t + 1 - out.shape[0], out.shape[1]), np.int8(-1), np.int8)
        out = np.concatenate([out, extra], axis=0)
    sl = out[t:, slot]
    new = np.empty_like(sl)
    for i, v in enumerate(sl.tolist()):
        if mode == "payout":
            new[i] = np.int8(1 if v < 0 else (int(v) + 1) % 5)
        elif mode == "missing":
            new[i] = np.int8(0 if v < 0 else -1)
        else:
            raise AssertionError(mode)
    out[t:, slot] = new
    changed = out.shape != obs.shape or not np.array_equal(out, obs)
    if not changed:
        raise AssertionError("mutation was a no-op")
    return out


def logits_at(model, ep, queries, W):
    # Queries stay so episode_meta has a key. USE_SIGNAL is False, so values are unread.
    data = SplitData([ep], {ep.episode_id: 0}, queries, None)
    (X, T, TY, SG), _ids = B.build_query(data, W)
    with torch.no_grad():
        return model(X, T, TY, SG)


def causality(model, ep, queries, W, label):
    for kind, t in (("pseudo", max(1, min(ep.cutoff // 2, ep.cutoff - 1))), ("cutoff", ep.cutoff)):
        base_obs = np.array(ep.obs, copy=True)
        base_ep = copy.copy(ep)
        base_ep.obs = base_obs
        base_ep.cutoff = t
        base = logits_at(model, base_ep, queries, W)
        ok = True
        detail = ""
        for slot in range(8):
            for mode in ("payout", "missing"):
                mut = copy.copy(ep)
                mut.obs = mutate_obs(base_obs, t, slot, mode)
                mut.cutoff = t
                out = logits_at(model, mut, queries, W)
                if not torch.equal(base, out):
                    ok = False
                    detail = "%s t=%s slot=%s mode=%s" % (kind, t, slot, mode)
                    break
            if not ok:
                break
        check("causality %s %s" % (label, kind), ok, detail if not ok else "t=%s" % t)


def flip_signals(data):
    queries = []
    for row in data.queries:
        flipped = dict(row)
        sig = int(flipped["signal"])
        flipped["signal"] = "1" if sig <= 0 else "-1"
        if flipped["signal"] == row["signal"]:
            raise AssertionError("signal did not change")
        queries.append(flipped)
    return SplitData(data.episodes, data.episode_index, queries, data.query_labels)


def tensors_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def query_ids(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    return [row[0] for row in rows[1:]]


def main():
    check("threads", torch.get_num_threads() == 2, "threads=%s" % torch.get_num_threads())
    check(
        "scorer_is_kit_contest_score",
        scoring.contest_score is kit_contest.contest_score,
        "scoring.py imports kit slot_forecasting.contest.contest_score",
    )
    splits_path = REPO / "runs" / "splits.json"
    check(
        "splits_sha256",
        sha256_file(splits_path) == "8f266f53f5da262c3141e57ea780668b552dc5608b63c2eaeee667cad79891cb",
        sha256_file(splits_path),
    )
    cnn_py = REPO / "experiments" / "cnn" / "source" / "cnn.py"
    check(
        "base_cnn_sha_equals_cnn_py",
        sha256_file(SRC / "base_cnn.py") == sha256_file(cnn_py),
        sha256_file(SRC / "base_cnn.py"),
    )
    check(
        "snapshot_REPO_is_not_house_edge",
        Path(B.REPO).resolve() != REPO.resolve(),
        "base_cnn.REPO=%s PUB_exists=%s" % (B.REPO, Path(B.PUB).exists()),
    )

    print("loading validation", flush=True)
    va = load_split(PUB / "validation")
    print("loading test", flush=True)
    te = load_split(PUB / "test")
    cond = episode_conditions(PUB / "validation")
    short = next(
        ep for ep in va.episodes
        if cond[ep.episode_id] == "short_history" and has_missing_gap(ep.obs) and ep.cutoff > 2
    )
    stable = next(ep for ep in va.episodes if cond[ep.episode_id] == "stable" and ep.cutoff > 2)
    print("episodes", short.episode_id, "short_history cutoff", short.cutoff,
          "stable", stable.episode_id, "cutoff", stable.cutoff, flush=True)

    print("loading train for label mutation", flush=True)
    tr = load_split(PUB / "train", with_labels=True)
    label_ids = []
    for ep in tr.episodes:
        qids = [r["query_id"] for r in tr.queries if r["episode_id"] == ep.episode_id]
        if len(qids) == 8 and all(tr.query_labels[q] >= 0 for q in qids):
            label_ids.append(ep.episode_id)
        if len(label_ids) == 4:
            break
    sub = subset_split_data(tr, label_ids)
    labs = dict(sub.query_labels)
    touched = [r["query_id"] for r in sub.queries]
    for qid in touched:
        labs[qid] = (labs[qid] + 1) % 5
    mutated = SplitData(sub.episodes, sub.episode_index, sub.queries, labs)
    W_label = 64
    built = B.build_train(sub, W_label)
    built_m = B.build_train(mutated, W_label)
    names = ("X", "T", "TY", "SG")
    label_ok = True
    label_detail = "episodes=%s windows=%s" % (len(label_ids), len(built[0]))
    for name, a, b in zip(names, built[:4], built_m[:4]):
        if not torch.equal(a, b):
            label_ok = False
            label_detail = name + " changed"
            break
    if label_ok and torch.equal(built[4], built_m[4]):
        label_ok = False
        label_detail = "Y did not change; mutation missed the cutoff targets"
    check("cutoff_label_mutation", label_ok, label_detail)

    pin = json.loads((REVIEW / "PINNED.json").read_text())
    kit_va = {r["query_id"] for r in va.queries}
    kit_te = {r["query_id"] for r in te.queries}

    for seed in (0, 1, 2):
        run = RUNS / ("final_cnn03_s%d" % seed)
        cfg, blob, state, model = load_model(run)
        W = cfg["W"]
        tag = "s%d" % seed
        fp32 = all(t.dtype == torch.float32 for t in state.values() if t.is_floating_point())
        check("load %s" % tag, cfg["model"] == "cnn" and cfg["h"] == 128 and cfg["layers"] == 4 and fp32,
              "h=%s layers=%s kernel_cfg=%s W=%s drop=%s" % (cfg["h"], cfg["layers"], cfg.get("kernel"), W, cfg["drop"]))
        sig_keys = [k for k in state if "sig" in k.split(".")]
        check("no_sig_keys %s" % tag, sig_keys == [] and model.sig is None, "sig_keys=%s" % sig_keys)
        check(
            "pin %s" % tag,
            sha256_file(run / "checkpoint.pt") == pin["checkpoints"]["final_cnn03_s%d" % seed],
            sha256_file(run / "checkpoint.pt"),
        )
        short_q = [r for r in va.queries if r["episode_id"] == short.episode_id]
        stable_q = [r for r in va.queries if r["episode_id"] == stable.episode_id]
        causality(model, short, short_q, W, tag + " short_history")
        causality(model, stable, stable_q, W, tag + " stable")

        x0 = torch.tensor(B.window(stable.obs, stable.cutoff, W)).unsqueeze(0).clone()
        x1 = x0.clone()
        x0[0, -1, 0] = 0
        x1[0, -1, 0] = -1
        t = torch.tensor([stable.cutoff])
        ty = torch.tensor(stable.machine_types).unsqueeze(0)
        sg = torch.zeros(1, 8, dtype=torch.long)
        with torch.no_grad():
            o0 = model(x0, t, ty, sg)
            o1 = model(x1, t, ty, sg)
        check("missing_vs_payout0 %s" % tag, not torch.equal(o0, o1))

        lg = torch.randn(4, 8, 5)
        y = torch.randint(0, 5, (4, 8))
        y[0, 1] = -1
        y[1, :] = -1
        y[2, 3] = -1
        loss1 = B.masked_loss(lg, y)
        lg2 = lg.clone()
        lg2[y < 0] += 3.141592653589793
        loss2 = B.masked_loss(lg2, y)
        yl = y.long()
        msk = yl >= 0
        ce = F.cross_entropy(lg.float()[msk], yl[msk])
        check("masked_loss %s" % tag, torch.equal(loss1, loss2) and torch.equal(loss1, ce))

        for split, data, kit_ids, fname in (
            ("validation", va, kit_va, "validation_predictions_cpu.csv"),
            ("test", te, kit_te, "test_predictions_cpu.csv"),
        ):
            flipped = flip_signals(data)
            Q0, _e0 = B.build_query(data, W)
            Q1, _e1 = B.build_query(flipped, W)
            same_q = tensors_equal(Q0, Q1)
            with torch.no_grad():
                p0 = B.predict(model, Q0, "cpu")
                p1 = B.predict(model, Q1, "cpu")
            check(
                "signal_mutation %s %s" % (tag, split),
                same_q and np.array_equal(p0, p1),
                "tensors=%s outputs=%s" % (same_q, np.array_equal(p0, p1)),
            )
            preds = B.to_preds(p0, _e0, data)
            out = TMP / ("re_s%d_%s" % (seed, fname))
            write_submission_csv(out, preds)
            saved = (run / fname).read_bytes()
            got = out.read_bytes()
            detail = "bytes=%s" % len(got)
            if got != saved:
                gl, sl = got.splitlines(), saved.splitlines()
                detail = "len %s vs %s lines %s vs %s" % (len(got), len(saved), len(gl), len(sl))
                for i, (a, b) in enumerate(zip(gl, sl)):
                    if a != b:
                        detail += " first_diff_line=%s" % (i + 1)
                        break
            check("csv_bytes %s %s" % (tag, split), got == saved, detail)
            got_ids = set(query_ids(out))
            check(
                "query_ids %s %s" % (tag, split),
                got_ids == kit_ids == set(preds),
                "n=%s" % len(got_ids),
            )

    print("FAILS", len(FAILS))
    for item in FAILS:
        print(" -", item)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
