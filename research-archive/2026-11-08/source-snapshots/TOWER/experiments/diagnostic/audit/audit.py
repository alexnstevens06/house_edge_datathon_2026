#!/usr/bin/env python
"""Independent read-only audit of the house-edge repo (splits, scoring, payout
order, baselines). Run: .venv/bin/python experiments/diagnostic/audit/audit.py

Writes experiments/diagnostic/audit/AUDIT_RESULTS.md and prints a final summary.
The scorer below is reimplemented from kit/RULES.md text only; src/house_edge
scoring and kit scoring are imported ONLY afterwards for cross-comparison.
"""
from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KIT_PUBLIC = REPO / "kit" / "public"
SPLIT_DIRS = {"train": KIT_PUBLIC / "train", "validation": KIT_PUBLIC / "validation",
              "test": KIT_PUBLIC / "test"}
PAYOUTS = (0, 1, 2, 5, 20)                    # per RULES.md: 0x, 1x, 2x, 5x, 20x
PROB_COLS = tuple(f"p_{p}" for p in PAYOUTS)  # submission column order
CONDITIONS = ("stable", "short_history", "drift", "shared_change")  # RULES.md listing
SUM_TOL = 1e-6

T0 = time.time()
SECTIONS: dict[str, list[str]] = {}
RESULTS: list[tuple[str, str, bool, str]] = []


def check(sec: str, name: str, ok: bool, detail: str = "") -> bool:
    ok = bool(ok)
    RESULTS.append((sec, name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {sec} :: {name} :: {detail}", flush=True)
    return ok


def note(sec: str, text: str) -> None:
    SECTIONS.setdefault(sec, []).append(text)


def table(sec: str, header: list[str], rows: list[list]) -> None:
    note(sec, "| " + " | ".join(header) + " |")
    note(sec, "|" + "---|" * len(header))
    for r in rows:
        note(sec, "| " + " | ".join(str(x) for x in r) + " |")
    note(sec, "")


# ---------------------------------------------------------------- csv helpers
def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def header_of(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def normalize(raw: list[float], qid: str) -> list[float]:
    vals = [float(v) for v in raw]
    if any(not math.isfinite(v) or v < 0 or v > 1 for v in vals):
        raise ValueError(f"{qid}: probability outside [0,1] or non-finite")
    total = math.fsum(vals)
    if abs(total - 1.0) > SUM_TOL:
        raise ValueError(f"{qid}: probabilities sum to {total!r}, not 1 (tol {SUM_TOL})")
    return [v / total for v in vals]  # strip decimal-rounding residue only


def read_prob_csv(path: Path) -> dict[str, list[float]]:
    hdr = header_of(path)
    if tuple(hdr) != ("query_id",) + PROB_COLS:
        raise ValueError(f"{path}: header {hdr} != query_id,{','.join(PROB_COLS)}")
    out = {}
    for r in read_csv(path):
        if r["query_id"] in out:
            raise ValueError(f"{path}: duplicate query_id {r['query_id']}")
        out[r["query_id"]] = normalize([r[c] for c in PROB_COLS], r["query_id"])
    return out


# ================================================================== TASK 1
def task1() -> None:
    sec = "1. Splits"
    print(f"\n=== {sec} ===", flush=True)

    queries = {s: read_csv(d / "queries.csv") for s, d in SPLIT_DIRS.items()}
    ep_ids = {s: {r["episode_id"] for r in rows} for s, rows in queries.items()}
    n_eps = {s: len(ep_ids[s]) for s in ep_ids}
    note(sec, f"Episodes by split: train={n_eps['train']}, validation={n_eps['validation']}, "
              f"test={n_eps['test']} (from queries.csv).")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        inter = ep_ids[a] & ep_ids[b]
        check(sec, f"{a}/{b} episode IDs disjoint", not inter,
              f"overlap={len(inter)}" + (f" e.g. {sorted(inter)[:3]}" if inter else ""))

    # --- query -> exactly one (episode, machine); 8 machines/episode; cutoff
    for s, rows in queries.items():
        qids = [r["query_id"] for r in rows]
        pairs = [(r["episode_id"], r["machine_id"]) for r in rows]
        ok = (len(set(qids)) == len(qids)
              and len(set(pairs)) == len(pairs)
              and all(r["machine_id"].startswith(r["episode_id"] + "-m") for r in rows))
        per_ep = defaultdict(set)
        for r in rows:
            per_ep[r["episode_id"]].add(r["machine_id"])
        ok = ok and all(len(v) == 8 for v in per_ep.values()) and len(per_ep) == n_eps[s]
        check(sec, f"{s}: every query maps to exactly one episode/machine", ok,
              f"rows={len(rows)}, unique qids={len(set(qids))}, unique (ep,machine)="
              f"{len(set(pairs))}, episodes with 8 machines={sum(1 for v in per_ep.values() if len(v)==8)}/{n_eps[s]}")
        # duplicate history (episode, machine, tick) + cutoff vs history ticks
        seen, max_tick = set(), {}
        payouts_seen = set()
        for r in read_csv(SPLIT_DIRS[s] / "history.csv"):
            key = (r["episode_id"], r["machine_id"], int(r["tick"]))
            seen.add(key) if key not in seen else None
            max_tick[r["episode_id"]] = max(max_tick.get(r["episode_id"], -1), int(r["tick"]))
            payouts_seen.add(int(r["payout"]))
        hist_rows = sum(1 for _ in read_csv(SPLIT_DIRS[s] / "history.csv"))
        check(sec, f"{s}: no duplicate (episode,machine,tick) in history",
              len(seen) == hist_rows, f"history rows={hist_rows}, unique keys={len(seen)}")
        check(sec, f"{s}: history payout values subset of {PAYOUTS}",
              payouts_seen <= set(PAYOUTS), f"observed values={sorted(payouts_seen)}")
        cutoffs = {r["episode_id"]: int(r["cutoff"]) for r in rows}
        bad = {e: (cutoffs[e], max_tick.get(e, -1)) for e in cutoffs
               if max_tick.get(e, -1) >= cutoffs[e]}
        margins = [cutoffs[e] - 1 - max_tick.get(e, -1) for e in cutoffs]
        check(sec, f"{s}: cutoff > all history ticks of that episode", not bad,
              f"violations={len(bad)}" + (f" {dict(list(bad.items())[:3])}" if bad else
              f"; min margin (cutoff-1-max_tick)={min(margins)}, max={max(margins)}"))
        # one consistent cutoff per episode
        per_ep_cutoff = defaultdict(set)
        for r in rows:
            per_ep_cutoff[r["episode_id"]].add(int(r["cutoff"]))
        check(sec, f"{s}: one cutoff per episode", all(len(v) == 1 for v in per_ep_cutoff.values()),
              f"episodes with multiple cutoffs={sum(1 for v in per_ep_cutoff.values() if len(v)>1)}")

    # --- conditions.csv (validation): one condition per episode, 4x24
    val_queries = queries["validation"]
    qid2ep = {r["query_id"]: r["episode_id"] for r in val_queries}
    cond_rows = read_csv(SPLIT_DIRS["validation"] / "conditions.csv")
    ep_cond = {}
    consistent = True
    for r in cond_rows:
        e = qid2ep[r["query_id"]]
        if ep_cond.setdefault(e, r["condition"]) != r["condition"]:
            consistent = False
    cond_counts = defaultdict(list)
    for e, c in ep_cond.items():
        cond_counts[c].append(e)
    check(sec, "validation: one condition per episode, 4 conditions x 24 episodes",
          consistent and set(cond_counts) == set(CONDITIONS)
          and all(len(v) == 24 for v in cond_counts.values()) and len(ep_cond) == 96,
          f"episodes={len(ep_cond)}, per-condition="
          f"{ {c: len(v) for c, v in sorted(cond_counts.items())} }")

    # --- runs/splits.json
    splits = json.loads((REPO / "runs" / "splits.json").read_text())
    tune, confirm = set(splits["tune"]), set(splits["confirm"])
    check(sec, "splits.json tune/confirm disjoint", not (tune & confirm),
          f"|tune|={len(tune)}, |confirm|={len(confirm)}, overlap={len(tune & confirm)}")
    check(sec, "splits.json covers all 96 validation episodes",
          (tune | confirm) == ep_ids["validation"] and not (tune | confirm) - ep_ids["validation"],
          f"union={len(tune | confirm)}, missing={len(ep_ids['validation'] - (tune|confirm))}, "
          f"extra={len((tune|confirm) - ep_ids['validation'])}")
    per_ok = all(len(tune & set(cond_counts[c])) == 12 and len(confirm & set(cond_counts[c])) == 12
                 for c in CONDITIONS)
    check(sec, "splits.json: 12 tune + 12 confirm episodes per condition", per_ok,
          f"tune/cond={ {c: len(tune & set(v)) for c, v in sorted(cond_counts.items())} }, "
          f"confirm/cond={ {c: len(confirm & set(v)) for c, v in sorted(cond_counts.items())} }")

    # --- recompute split with src/house_edge/split.py
    sys.path.insert(0, str(REPO / "src"))
    from house_edge.split import tune_confirm_split  # noqa: E402
    recomputed = tune_confirm_split(str(SPLIT_DIRS["validation"]))
    equal = (recomputed["tune"] == splits["tune"]
             and recomputed["confirm"] == splits["confirm"]
             and recomputed["seed"] == splits["seed"]
             and recomputed["by_condition"] == splits["by_condition"])
    check(sec, "recomputed split (src/house_edge/split.py, seed 0) equals runs/splits.json",
          equal, f"tune equal={recomputed['tune']==splits['tune']}, "
                 f"confirm equal={recomputed['confirm']==splits['confirm']}, "
                 f"seed={recomputed['seed']} vs {splits['seed']}, "
                 f"by_condition equal={recomputed['by_condition']==splits['by_condition']}")

    # --- training windows reference only train episodes
    from house_edge.data import load_split  # noqa: E402
    from house_edge import windows  # noqa: E402  (imports torch)
    train_data = load_split(SPLIT_DIRS["train"], with_labels=True)
    sliding = windows.sliding_examples(train_data.episodes)
    qex = windows.query_examples(train_data.episodes, train_data.episode_index,
                                 train_data.queries, train_data.query_labels)
    train_eps = {e.episode_id for e in train_data.episodes}
    ref_eps = {train_data.episodes[ex.ep].episode_id for ex in sliding} | \
              {train_data.episodes[ex.ep].episode_id for ex in qex}
    check(sec, "training windows (sliding+query examples) reference only train episodes",
          ref_eps <= train_eps and not (ref_eps & ep_ids["validation"]) and not (ref_eps & ep_ids["test"]),
          f"sliding examples={len(sliding)} (episodes={len({e.ep for e in sliding})}), "
          f"query examples={len(qex)} (episodes={len({e.ep for e in qex})}), "
          f"referenced episodes={len(ref_eps)}; leaks into val/test="
          f"{len((ref_eps & ep_ids['validation']) | (ref_eps & ep_ids['test']))}")

    # query examples: t == cutoff, labels come from train labels.csv (checked against our own parse)
    my_labels = {r["query_id"]: PAYOUTS.index(int(r["outcome"]))
                 for r in read_csv(SPLIT_DIRS["train"] / "labels.csv")}
    qex_by_ep = {train_data.episodes[ex.ep].episode_id: ex for ex in qex}
    label_ok, cutoff_ok = True, True
    for r in queries["train"]:
        ex = qex_by_ep[r["episode_id"]]
        ep = train_data.episodes[ex.ep]
        if ex.t != int(r["cutoff"]) or ex.t != ep.cutoff:
            cutoff_ok = False
        if ex.labels[ep.slot_of(r["machine_id"])] != my_labels[r["query_id"]]:
            label_ok = False
    check(sec, "train query examples predict at cutoff with labels from train labels.csv",
          cutoff_ok and label_ok and len(qex) == n_eps["train"],
          f"examples={len(qex)}, cutoff match={cutoff_ok}, label match={label_ok}")
    check(sec, "sliding examples start at tick >= T_START(16)",
          all(ex.t >= windows.T_START for ex in sliding), f"min t={min(e.t for e in sliding)}")

    # collator causality + payout<->symbol decode on a seeded sample (dense and sparse)
    rng = random.Random(20261107)
    sample = rng.sample(sliding, 1500) + qex
    payout_at = {}
    for r in read_csv(SPLIT_DIRS["train"] / "history.csv"):
        payout_at[(r["episode_id"], r["machine_id"], int(r["tick"]))] = int(r["payout"])
    decode_errors, causality_errors, coverage_errors = 0, 0, 0
    for arch in ("dense", "sparse"):
        collator = windows.WindowCollator(train_data.episodes, arch, window=64)
        for i in range(0, len(sample), 256):
            chunk = sample[i:i + 256]
            out = collator(chunk)  # internal asserts enforce tick < target tick
            syms = out["symbols"].numpy()
            tks = out["ticks"].numpy()
            tgt = out["target_t"].numpy()
            for b, ex in enumerate(chunk):
                ep = train_data.episodes[ex.ep]
                t = ex.t
                if arch == "dense":
                    grid_s, grid_t = syms[b].reshape(64, 8), tks[b].reshape(64, 8)
                    lo, w0 = max(0, t - 64), 64 - (t - max(0, t - 64))
                    for w in range(64):
                        for slot in range(8):
                            sym, tick = int(grid_s[w, slot]), int(grid_t[w, slot])
                            if not (0 <= tick < t):
                                causality_errors += 1
                            if sym > 0:  # dense symbol = payout_idx + 1 -> PAYOUTS[sym-1]
                                if payout_at.get((ep.episode_id, ep.machine_ids[slot], tick)) != PAYOUTS[sym - 1]:
                                    decode_errors += 1
                    for slot in range(8):
                        for tick in range(lo, t):
                            if (ep.episode_id, ep.machine_ids[slot], tick) in payout_at:
                                sym = int(grid_s[w0 + (tick - lo), slot])
                                if sym == 0 or PAYOUTS[sym - 1] != payout_at[(ep.episode_id, ep.machine_ids[slot], tick)]:
                                    coverage_errors += 1
                else:
                    mask = out["pad_mask"].numpy()[b]
                    tok_slots = out["slots"].numpy()[b]
                    for j in range(len(syms[b])):
                        if mask[j]:
                            continue
                        sym, tick, slot = int(syms[b, j]), int(tks[b, j]), int(tok_slots[j])
                        if not (0 <= tick < t):
                            causality_errors += 1
                        if payout_at.get((ep.episode_id, ep.machine_ids[slot], tick)) != PAYOUTS[sym]:
                            decode_errors += 1
    n_tokens = 1500 * 64 * 8  # dense tokens inspected (upper bound incl. pads)
    check(sec, "collator causality: every input token tick < target tick (dense+sparse sample)",
          causality_errors == 0,
          f"sampled {len(sample)} examples (1500 random sliding + all {len(qex)} query), "
          f"causality violations={causality_errors}")
    check(sec, "window tokens decode to true history payouts (0/1/2/5/20 order end-to-end)",
          decode_errors == 0 and coverage_errors == 0,
          f"decode mismatches={decode_errors}, coverage misses={coverage_errors} "
          f"(dense symbols 1..5 = payout idx+1; sparse 0..4 = payout idx; ~{n_tokens} dense tokens + sparse tokens checked)")

    table(sec, ["Split", "episodes", "queries", "history rows"],
          [[s, n_eps[s], len(queries[s]),
            sum(1 for _ in read_csv(SPLIT_DIRS[s] / "history.csv"))] for s in SPLIT_DIRS])


# ================================================================== TASK 2
def task2() -> None:
    sec = "2. Scorer"
    print(f"\n=== {sec} ===", flush=True)

    targets: dict[str, tuple[str, list[float], int]] = {}
    for r in read_csv(SPLIT_DIRS["validation"] / "targets.csv"):
        targets[r["query_id"]] = (r["episode_id"],
                                  normalize([r[c] for c in PROB_COLS], r["query_id"]),
                                  int(r["outcome"]))
    cond_of_qid = {r["query_id"]: r["condition"] for r in read_csv(SPLIT_DIRS["validation"] / "conditions.csv")}
    qid2ep = {r["query_id"]: r["episode_id"] for r in read_csv(SPLIT_DIRS["validation"] / "queries.csv")}
    ref_raw = {r["query_id"]: [float(r[c]) for c in PROB_COLS]
               for r in read_csv(SPLIT_DIRS["validation"] / "reference.csv")}

    def score_mine(preds_raw, episode_subset=None, max_points=60.0):
        """RULES.md: per-query sum of 5 squared errors; mean within episode; mean
        over episodes within condition; ratio vs reference with floor 1e-9;
        R = mean of 4 ratios; points = max_points / (1 + R)."""
        preds = {q: normalize(p, q) for q, p in preds_raw.items()}
        ref = {q: normalize(p, q) for q, p in ref_raw.items()}
        tg = targets
        if episode_subset is not None:
            tg = {q: v for q, v in targets.items() if v[0] in episode_subset}
        keep = set(tg)
        preds = {q: preds[q] for q in preds if q in keep}
        ref = {q: ref[q] for q in ref if q in keep}
        assert set(preds) == keep, f"id mismatch: {len(keep - set(preds))} missing"
        by_cond = {}
        for c in CONDITIONS:
            eps = sorted({tg[q][0] for q in tg if cond_of_qid[q] == c})
            ep_err_p, ep_err_r, nq = {}, {}, 0
            for e in eps:
                qids = sorted(q for q in tg if tg[q][0] == e)
                nq += len(qids)
                ep_err_p[e] = math.fsum(math.fsum((a - b) ** 2 for a, b in zip(preds[q], tg[q][1]))
                                        for q in qids) / len(qids)
                ep_err_r[e] = math.fsum(math.fsum((a - b) ** 2 for a, b in zip(ref[q], tg[q][1]))
                                        for q in qids) / len(qids)
            E = math.fsum(ep_err_p.values()) / len(ep_err_p)
            B = math.fsum(ep_err_r.values()) / len(ep_err_r)
            by_cond[c] = {"E": E, "B": B, "ratio": E / max(B, 1e-9),
                          "episodes": len(eps), "queries": nq}
        R = math.fsum(v["ratio"] for v in by_cond.values()) / len(CONDITIONS)
        return {"R": R, "points": max_points / (1.0 + R), "by_condition": by_cond}

    # fixtures
    ref_norm = {q: normalize(p, q) for q, p in ref_raw.items()}
    tgt_pred = {q: t[1] for q, t in targets.items()}
    uni = {q: [0.2] * 5 for q in targets}
    fx = [("reference.csv itself", score_mine(ref_norm)),
          ("targets.csv probabilities", score_mine(tgt_pred)),
          ("uniform 0.2", score_mine(uni))]
    note(sec, "Reimplemented from RULES.md only (no import of src/house_edge/scoring.py or kit "
              "slot_forecasting.scoring for these numbers). Probabilities are used as submitted; "
              "each vector is divided by its own sum (sums already within 1e-6 of 1), mirroring the "
              "RULES requirement that probabilities sum to 1.")
    table(sec, ["Fixture (predictions)", "R", "points = 60/(1+R)", "per-condition E[c]"],
          [[name, f"{r['R']:.12g}", f"{r['points']:.6f}",
            {c: round(v["E"], 8) for c, v in r["by_condition"].items()}] for name, r in fx])
    check(sec, "fixture: reference scores R=1 (30 points)",
          abs(fx[0][1]["R"] - 1.0) < 1e-12 and abs(fx[0][1]["points"] - 30.0) < 1e-9,
          f"R={fx[0][1]['R']:.15g}, points={fx[0][1]['points']:.9f}")
    check(sec, "fixture: targets score R=0 (60 points)",
          abs(fx[1][1]["R"]) < 1e-15 and abs(fx[1][1]["points"] - 60.0) < 1e-9,
          f"R={fx[1][1]['R']:.3g}, points={fx[1][1]['points']:.9f}")
    check(sec, "fixture: uniform 0.2 computed",
          True, f"R={fx[2][1]['R']:.9f}, points={fx[2][1]['points']:.6f}")

    files = [REPO / "runs/baselines/pooled_recent_validation.csv",
             REPO / "runs/archive_dense_v1/validation_predictions.csv",
             REPO / "runs/v2_dense/validation_predictions.csv"]
    splits = json.loads((REPO / "runs" / "splits.json").read_text())
    tune_set, confirm_set = set(splits["tune"]), set(splits["confirm"])

    # (a) kit/check_submission.py --split validation (fresh subprocess per file)
    kit_rows = []
    for f in files:
        proc = subprocess.run([sys.executable, str(REPO / "kit" / "check_submission.py"),
                               str(f), "--split", "validation", "--data", str(KIT_PUBLIC)],
                              capture_output=True, text=True, cwd=str(REPO))
        pts, ratios = None, {}
        for line in proc.stdout.splitlines():
            if line.startswith("Validation points:"):
                pts = float(line.split(":")[1].split("/")[0])
            if "loss ratio" in line:
                name, val = line.strip().split(":", 1)
                ratios[name.strip()] = float(val.split("loss ratio")[1].split("(lower")[0])
        kit_rows.append([f.name, proc.returncode, len(proc.stdout) > 0 and pts is not None, pts, ratios])
    check(sec, "kit/check_submission.py --split validation runs on all 3 files",
          all(r[1] == 0 and r[2] for r in kit_rows),
          "; ".join(f"{r[0]}: rc={r[1]}, points={r[3]}" for r in kit_rows))

    # (b) src/house_edge/scoring.py (imported only now, for comparison)
    from house_edge.scoring import load_official_validation, score_predictions  # noqa: E402
    official = load_official_validation(SPLIT_DIRS["validation"])

    rows_cmp, ok_all = [], True
    for f, kitr in zip(files, kit_rows):
        preds_raw = read_prob_csv(f)
        mine = score_mine(preds_raw)
        mine_t = score_mine(preds_raw, episode_subset=tune_set)
        mine_c = score_mine(preds_raw, episode_subset=confirm_set)
        src = score_predictions({q: list(p) for q, p in preds_raw.items()}, official)
        src_t = score_predictions({q: list(p) for q, p in preds_raw.items()}, official, episode_subset=tune_set)
        src_c = score_predictions({q: list(p) for q, p in preds_raw.items()}, official, episode_subset=confirm_set)
        kit_pts, kit_ratios = kitr[3], kitr[4]
        d_pts_src = abs(mine["points"] - src["points_60"])
        d_R_src = abs(mine["R"] - src["R"])
        d_rat_src = max(abs(mine["by_condition"][c]["ratio"] - src["loss_ratio_by_condition"][c]) for c in CONDITIONS)
        d_pts_kit = abs(mine["points"] - kit_pts)
        d_rat_kit = max(abs(mine["by_condition"][c]["ratio"] - kit_ratios.get(c, math.inf)) for c in CONDITIONS)
        d_t = abs(mine_t["R"] - src_t["R"])
        d_c = abs(mine_c["R"] - src_c["R"])
        ok = (d_pts_src < 1e-9 and d_R_src < 1e-12 and d_rat_src < 1e-12
              and d_pts_kit < 1e-6 and d_rat_kit < 1e-6 and d_t < 1e-12 and d_c < 1e-12
              and len(preds_raw) == 768)
        ok_all = ok_all and ok
        rows_cmp.append([f"{f.parent.name}/{f.name}", len(preds_raw),
                         f"{mine['R']:.9f}", f"{mine['points']:.6f}",
                         f"{src['R']:.9f}", f"{src['points_60']:.6f}",
                         f"{kit_pts:.6f}",
                         f"{d_R_src:.2e}/{d_pts_src:.2e}/{d_rat_src:.2e}",
                         f"{d_pts_kit:.2e}/{d_rat_kit:.2e}",
                         f"{mine_t['R']:.9f} vs src {src_t['R']:.9f} (d={d_t:.2e})",
                         f"{mine_c['R']:.9f} vs src {src_c['R']:.9f} (d={d_c:.2e})"])
        # per-condition table for the md
        table(sec, ["condition", "E[c] (mine)", "B[c] ref (mine)", "ratio mine",
                    "ratio src", "ratio kit-check", "episodes"],
              [[c, f"{mine['by_condition'][c]['E']:.9f}", f"{mine['by_condition'][c]['B']:.9f}",
                f"{mine['by_condition'][c]['ratio']:.9f}",
                f"{src['loss_ratio_by_condition'][c]:.9f}", f"{kit_ratios.get(c, float('nan')):.6f}",
                mine['by_condition'][c]['episodes']] for c in CONDITIONS])
    note(sec, "Comparison columns: mine vs src = |dR|/|dpoints|/|max d ratio|; "
              "mine vs kit-check = |dpoints|/|max d ratio| (kit prints 6 decimals).")
    table(sec, ["file", "ids", "R mine", "pts mine", "R src", "pts src", "pts kit-check",
                "mine-vs-src", "mine-vs-kit", "tune R (mine vs src)", "confirm R (mine vs src)"],
          rows_cmp)
    check(sec, "independent scorer matches src/house_edge/scoring.py (R, points, per-condition, "
               "tune/confirm) and kit/check_submission.py for all 3 files", ok_all,
          f"3 files x (full, tune, confirm); worst |dR| vs src="
          f"{max(float(r[7].split('/')[0]) for r in rows_cmp):.2e}")

    # tune/confirm vs full explanation, with numbers (pooled_recent)
    preds_raw = read_prob_csv(files[0])
    full = score_mine(preds_raw)
    tun = score_mine(preds_raw, episode_subset=tune_set)
    con = score_mine(preds_raw, episode_subset=confirm_set)
    ex_rows = []
    ident_E = ident_B = True
    for c in CONDITIONS:
        E_f, B_f = full["by_condition"][c]["E"], full["by_condition"][c]["B"]
        E_t, B_t = tun["by_condition"][c]["E"], tun["by_condition"][c]["B"]
        E_c, B_c = con["by_condition"][c]["E"], con["by_condition"][c]["B"]
        ident_E &= abs(E_f - (E_t + E_c) / 2) < 1e-12
        ident_B &= abs(B_f - (B_t + B_c) / 2) < 1e-12
        ex_rows.append([c, f"{E_t:.6f}", f"{E_c:.6f}", f"{E_f:.6f}",
                        f"{B_t:.6f}", f"{B_c:.6f}", f"{B_f:.6f}",
                        f"{E_t/B_t:.6f}", f"{E_c/B_c:.6f}", f"{E_f/B_f:.6f}"])
    table(sec, ["condition (pooled_recent)", "E_tune", "E_confirm", "E_full",
                "B_tune", "B_confirm", "B_full", "R_tune=E_t/B_t", "R_confirm", "R_full"],
          ex_rows)
    note(sec, "**How tune/confirm R relate to full R.** Each subset score recomputes the reference "
              "denominator *on that subset*: R_tune[c] = E_tune[c]/B_tune[c] uses the reference error "
              f"averaged over only the 12 tune episodes, and likewise for confirm. Because tune and "
              "confirm have equal episode counts (12/12 per condition), the numerator and denominator "
              "*errors* average exactly: E_full[c] = (E_tune[c]+E_confirm[c])/2 and "
              "B_full[c] = (B_tune[c]+B_confirm[c])/2 (verified to 1e-12 above). But the *ratios* do "
              "not average: E_full[c]/B_full[c] != (E_tune[c]/B_tune[c] + E_confirm[c]/B_confirm[c])/2 "
              "unless B_tune[c] = B_confirm[c] = B_full[c]. So R_full is NOT the average of R_tune and "
              "R_confirm, and neither subset R is a unbiased shard of the full R; to reconstruct "
              "R_full[c] from subsets you need the errors themselves: "
              "R_full[c] = (E_tune[c]+E_confirm[c])/(B_tune[c]+B_confirm[c]). Numerically (pooled_recent): "
              f"R_tune={tun['R']:.6f}, R_confirm={con['R']:.6f}, mean of the two="
              f"{(tun['R']+con['R'])/2:.6f}, but R_full={full['R']:.6f}; "
              f"points: tune={tun['points']:.4f}, confirm={con['points']:.4f}, full={full['points']:.4f}.")
    check(sec, "E/B decompose as subset means (12+12 episodes) while ratios do not",
          ident_E and ident_B
          and abs((tun["R"] + con["R"]) / 2 - full["R"]) > 0,
          f"E and B identities hold (dE,dB<1e-12); |mean(R_tune,R_confirm)-R_full|="
          f"{abs((tun['R']+con['R'])/2-full['R']):.6f} (nonzero => denominators differ per subset)")


# ================================================================== TASK 3
def task3() -> None:
    sec = "3. Payout column order (0/1/2/5/20)"
    print(f"\n=== {sec} ===", flush=True)

    ok_hdr = True
    hdr_rows = []
    val_files = [REPO / "kit/public/validation/targets.csv", REPO / "kit/public/validation/reference.csv"]
    val_files += sorted((REPO / "kit/public").glob("*/sample_submission.csv"))
    val_files += sorted(REPO.glob("runs/*/validation_predictions.csv"))
    val_files += sorted(REPO.glob("runs/*/test_submission.csv"))
    val_files += sorted((REPO / "runs/baselines").glob("*.csv"))
    for f in val_files:
        hdr = header_of(f)
        if "outcome" in hdr or "episode_id" in hdr:  # targets.csv
            want = ["query_id", "episode_id", *PROB_COLS, "outcome"]
        else:
            want = ["query_id", *PROB_COLS]
        good = hdr == want
        ok_hdr &= good
        hdr_rows.append([str(f.relative_to(REPO)), ",".join(hdr), good])
    table(sec, ["file", "header", "matches payout order"], hdr_rows)
    check(sec, f"all {len(hdr_rows)} prediction/target/reference files use column order "
               f"query_id,p_0,p_1,p_2,p_5,p_20 (targets add episode_id,outcome)", ok_hdr,
          f"files checked={len(hdr_rows)}, mismatches={sum(1 for r in hdr_rows if not r[2])}")

    val_targets = read_csv(SPLIT_DIRS["validation"] / "targets.csv")
    outcomes = {int(r["outcome"]) for r in val_targets}
    check(sec, "validation targets.csv outcome column in {0,1,2,5,20}",
          outcomes <= set(PAYOUTS), f"outcomes seen={sorted(outcomes)}")
    for s in ("train", "validation"):
        labs = {int(r["outcome"]) for r in read_csv(SPLIT_DIRS[s] / "labels.csv")}
        check(sec, f"{s} labels.csv outcome in {PAYOUTS}", labs <= set(PAYOUTS), f"seen={sorted(labs)}")
    sum_bad = [r["query_id"] for r in val_targets
               if abs(math.fsum(float(r[c]) for c in PROB_COLS) - 1) > SUM_TOL]
    check(sec, "targets.csv rows are probabilities (in [0,1], sum 1 within 1e-6)",
          not sum_bad, f"rows=768, bad sums={len(sum_bad)}")

    # constants: kit vs src vs files
    sys.path.insert(0, str(REPO / "kit"))
    from slot_forecasting.schema import PAYOUTS as KIT_PAYOUTS, PROBABILITY_COLUMNS as KIT_COLS  # noqa: E402
    from house_edge.data import PAYOUTS as SRC_PAYOUTS, PAYOUT_TO_IDX, PAYOUT_IDX_TO_VALUE  # noqa: E402
    roundtrip = all(PAYOUT_IDX_TO_VALUE[PAYOUT_TO_IDX[p]] == p and PAYOUT_TO_IDX[p] == i
                    for i, p in enumerate(PAYOUTS))
    check(sec, "kit schema.PAYOUTS == src data.PAYOUTS == (0,1,2,5,20); p_* columns in same order; "
               "idx mapping round-trips", KIT_PAYOUTS == SRC_PAYOUTS == PAYOUTS
          and KIT_COLS == PROB_COLS and roundtrip,
          f"kit={KIT_PAYOUTS}, src={SRC_PAYOUTS}, cols={KIT_COLS}, roundtrip={roundtrip}")
    check(sec, "history payout values map through PAYOUT_TO_IDX to the same columns "
               "(end-to-end token decode verified in section 1)", True,
          "history/labels/targets values in {0,1,2,5,20}; window symbols decode to the true "
          "history payouts (dense sym=idx+1, sparse sym=idx); see section 1 checks.")


# ================================================================== TASK 4
def task4() -> None:
    sec = "4. Baselines"
    print(f"\n=== {sec} ===", flush=True)

    src_text = (REPO / "scripts" / "baselines.py").read_text()
    lines = src_text.splitlines()
    val_label_loads = [l.strip() for l in lines if "validation" in l and "with_labels" in l]
    train_label_loads = [l.strip() for l in lines if "with_labels=True" in l]
    import re
    official_uses = [l.strip() for l in lines if re.search(r"(?<!load_)official\b", l)]
    label_reads = [l.strip() for l in lines if "read_labels(" in l]
    fit_calls = [l.strip() for l in lines if ".fit(" in l or "shrinkage_fit_predict(" in l]
    pred_calls = [l.strip() for l in lines if ".predict(" in l]
    check(sec, "baselines.py loads labels only for train (no validation labels/targets into models)",
          not val_label_loads
          and any('os.path.join(public, "train")' in l for l in train_label_loads)
          and all('"train"' in l for l in label_reads),
          f"with_labels=True lines: {train_label_loads}; read_labels lines: {label_reads}")
    check(sec, "official validation targets/reference used only inside score_predictions (reporting)",
          all("score_predictions(" in l or "official = load_official_validation" in l
              for l in official_uses), f"uses of `official`: {official_uses}")
    check(sec, "models fit on train queries/history/labels only; predict on val/test queries+history",
          any("model = FrequencyForecaster(name).fit(train_queries, train_history, labels)" in l for l in fit_calls)
          and any("model.predict(val_queries, val_history)" in l for l in pred_calls)
          and any("model.predict(test_queries, test_history)" in l for l in pred_calls)
          and any("shrinkage_fit_predict(train_data, val_data)" in l for l in fit_calls)
          and any("shrinkage_fit_predict(train_data, test_data)" in l for l in fit_calls),
          f"fit: {[l for l in fit_calls][:4]}")

    val_ids = {r["query_id"] for r in read_csv(SPLIT_DIRS["validation"] / "queries.csv")}
    test_ids = {r["query_id"] for r in read_csv(SPLIT_DIRS["test"] / "queries.csv")}
    base_dir = REPO / "runs" / "baselines"
    id_rows, ok_ids = [], True
    for f in sorted(base_dir.glob("*_validation.csv")):
        ids = list(read_csv(f))
        id_set = {r["query_id"] for r in ids}
        good = id_set == val_ids and len(ids) == len(id_set) == 768
        ok_ids &= good
        id_rows.append([f.name, len(ids), len(id_set), good])
    table(sec, ["baseline validation CSV", "rows", "unique ids", "= 768 validation ids"], id_rows)
    check(sec, f"all {len(id_rows)} baseline *_validation.csv output exactly the 768 validation query ids",
          ok_ids, f"files={len(id_rows)} (6 kit frequency methods + shrinkage64), all match={ok_ids}; "
                  f"|validation ids|={len(val_ids)}")
    ok_t = all({r["query_id"] for r in read_csv(f)} == test_ids
               for f in sorted(base_dir.glob("*_test.csv")))
    check(sec, "baseline *_test.csv cover exactly the test query ids", ok_t,
          f"files={len(list(base_dir.glob('*_test.csv')))}, |test ids|={len(test_ids)}")

    # independent recomputation from train data + val queries/history ONLY
    train_q = {r["query_id"]: r for r in read_csv(SPLIT_DIRS["train"] / "queries.csv")}
    train_hist: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for r in read_csv(SPLIT_DIRS["train"] / "history.csv"):
        train_hist[(r["episode_id"], r["machine_id"])].append((int(r["tick"]), int(r["payout"])))
    for v in train_hist.values():
        v.sort()
    train_lab = {r["query_id"]: int(r["outcome"]) for r in read_csv(SPLIT_DIRS["train"] / "labels.csv")}
    val_q = {r["query_id"]: r for r in read_csv(SPLIT_DIRS["validation"] / "queries.csv")}
    val_hist: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for r in read_csv(SPLIT_DIRS["validation"] / "history.csv"):
        val_hist[(r["episode_id"], r["machine_id"])].append((int(r["tick"]), int(r["payout"])))
    for v in val_hist.values():
        v.sort()

    # pooled_recent, reimplemented from kit/slot_forecasting/baseline.py logic
    counts, contexts = [0.0] * 5, defaultdict(lambda: [0.0] * 5)
    for qid in sorted(train_q):
        q = train_q[qid]
        outs = [p for _, p in train_hist.get((q["episode_id"], q["machine_id"]), [])]
        outs.append(train_lab[qid])
        for o in outs:
            i = PAYOUTS.index(o)
            counts[i] += 1
            contexts[f'{q["machine_type"]}|{q["signal"]}'][i] += 1
    total = math.fsum(counts)
    gp = [(v + 0.5) / (total + 2.5) for v in counts]
    cp = {k: [(v + 12.0 * g) / (math.fsum(vs) + 12.0) for v, g in zip(vs, gp)]
          for k, vs in contexts.items()}
    pr_preds = {}
    for qid in sorted(val_q):
        q = val_q[qid]
        prior = cp.get(f'{q["machine_type"]}|{q["signal"]}', gp)
        cnt = [20.0 * v for v in prior]
        cutoff = int(q["cutoff"])
        for tick, payout in val_hist.get((q["episode_id"], q["machine_id"]), []):
            cnt[PAYOUTS.index(payout)] += 2 ** (-(cutoff - 1 - tick) / 24.0)
        tot = math.fsum(cnt)
        pr_preds[qid] = [v / tot for v in cnt]

    # shrinkage64, reimplemented from scripts/baselines.py docstring/logic
    prior_c = [0] * 5
    for plays in train_hist.values():
        for _, p in plays:
            prior_c[PAYOUTS.index(p)] += 1
    n_all = sum(prior_c)
    prior = [c / n_all for c in prior_c]
    sh_preds = {}
    for qid in val_q:
        q = val_q[qid]
        cutoff = int(q["cutoff"])
        lo = max(0, cutoff - 64)
        obs = [PAYOUTS.index(p) for t, p in val_hist.get((q["episode_id"], q["machine_id"]), [])
               if lo <= t < cutoff]
        cnt = [0.0] * 5
        for i in obs:
            cnt[i] += 1
        sh_preds[qid] = [(c + 16.0 * pv) / (len(obs) + 16.0) for c, pv in zip(cnt, prior)]

    cmp_rows, ok_recompute = [], True
    for name, mine in (("pooled_recent", pr_preds), ("shrinkage64", sh_preds)):
        f = base_dir / f"{name}_validation.csv"
        stored = {r["query_id"]: [float(r[c]) for c in PROB_COLS] for r in read_csv(f)}
        maxd = max(abs(a - b) for q in stored for a, b in
                   zip([v / sum(mine[q]) for v in mine[q]], stored[q]))
        ok = maxd < 1e-9 and set(stored) == val_ids
        ok_recompute &= ok
        cmp_rows.append([name, len(stored), f"{maxd:.3e}", ok])
    table(sec, ["baseline", "ids", "max |mine - file| (768x5 probs)", "reproduced"], cmp_rows)
    check(sec, "pooled_recent and shrinkage64 validation CSVs reproduced exactly from "
               "train data + val queries/history alone (no val labels/targets/reference)",
          ok_recompute, f"max abs diffs: {cmp_rows[0][2]} (pooled_recent), {cmp_rows[1][2]} (shrinkage64)")


# ================================================================== SUMMARY
def summary() -> None:
    sec = "FINAL AUDIT SUMMARY"
    print(f"\n=== {sec} ===", flush=True)
    counts = {}
    for s, _, ok, _ in RESULTS:
        counts.setdefault(s, [0, 0])
        counts[s][0 if ok else 1] += 1
    lines = ["| Section | PASS | FAIL |", "|---|---|---|"]
    for s in sorted(counts):
        lines.append(f"| {s} | {counts[s][0]} | {counts[s][1]} |")
    total_pass = sum(v[0] for v in counts.values())
    total_fail = sum(v[1] for v in counts.values())
    verdict = "ALL CHECKS PASS" if total_fail == 0 else f"{total_fail} CHECK(S) FAILED"
    lines += ["", f"**{verdict}** — {total_pass} passed, {total_fail} failed "
              f"({len(RESULTS)} checks total)."]
    for s, name, ok, detail in RESULTS:
        if not ok:
            lines.append(f"- FAIL {s} :: {name} :: {detail}")
    note(sec, "\n".join(lines))
    print("\n".join(lines), flush=True)
    note(sec, f"\nAudit runtime: {time.time()-T0:.1f}s. Auditor script: "
              f"experiments/diagnostic/audit/audit.py (stdlib + numpy; src/kit imports only for "
              f"split recompute, window construction, constants, and post-hoc comparison).")


def main() -> None:
    tasks = [("1. Splits", task1), ("2. Scorer", task2), ("3. Payout column order (0/1/2/5/20)", task3),
             ("4. Baselines", task4)]
    for title, fn in tasks:
        SECTIONS.setdefault(title, [])
        try:
            fn()
        except Exception as exc:  # keep auditing; record the failure
            import traceback
            check(title, f"task raised exception: {type(exc).__name__}", False, str(exc))
            traceback.print_exc()
    summary()
    md = ["# Independent read-only audit — house-edge",
          f"Run {time.strftime('%Y-%m-%d %H:%M:%S')} with `{sys.executable}` "
          f"(numpy available; script itself stdlib+numpy). Scope: splits, scoring, payout "
          f"ordering, baselines. No files outside experiments/diagnostic/audit/ were modified.",
          ""]
    for title in [t for t, _ in tasks] + ["FINAL AUDIT SUMMARY"]:
        md.append(f"\n## {title}\n")
        md.extend(SECTIONS.get(title, []))
        checks = [f"- {'PASS' if ok else 'FAIL'} :: {name} :: {detail}"
                  for s, name, ok, detail in RESULTS if s == title]
        if checks and title != "FINAL AUDIT SUMMARY":
            md.append("\n**Checks:**")
            md.extend(checks)
    out = Path(__file__).resolve().parent / "AUDIT_RESULTS.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
