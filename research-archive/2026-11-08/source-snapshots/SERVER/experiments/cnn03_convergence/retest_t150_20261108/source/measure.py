"""Produce t150 scores only after FREEZE.json matches. Writes under this retest folder."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

RETEST = Path(__file__).resolve().parents[1]
REPO = RETEST.parents[2]
OUT = RETEST / "outputs"
FRESH = OUT / "fresh_zip"
PY = REPO / "experiments/cnn_runtime/py313_verify_1936/env/bin/python"
RUNS = REPO / "experiments/cnn03_convergence/runs"
STUDY_SOURCE = REPO / "experiments/cnn03_convergence/source"
COMMON_PATH = REPO / "experiments/cnn03_convergence/ensemble_blend_20261108/source/common.py"
ZIP_PATH = REPO / "experiments/cnn03_convergence/candidate_t146_ensemble_s0_s2_20261108/ensemble_equal_s0_s2_fp64_candidate.zip"
TEST_CSV = REPO / "experiments/cnn03_convergence/candidate_t146_ensemble_s0_s2_20261108/refs/candidate_test_a.csv"
TUNE_JSON = REPO / "experiments/cnn03_convergence/candidate_t146_ensemble_s0_s2_20261108/tune_only.json"
NPZ = REPO / "experiments/cnn03_convergence/ensemble_blend_20261108/results/predictions/ensemble_equal_s0_s2.npz"
EQ3_NPZ = REPO / "experiments/cnn03_convergence/ensemble_blend_20261108/results/predictions/ensemble_equal_s0_s1_s2.npz"

sys.path.insert(0, str(RETEST / "source"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "kit"))
import analysis as A  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"MEASURE_REFUSED {message}")


def load_common():
    spec = importlib.util.spec_from_file_location("frozen_ensemble_common", COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pins() -> dict:
    return json.loads((RETEST / "pins/PINS.json").read_text(encoding="utf-8"))


def check_freeze(pins: dict) -> None:
    freeze = json.loads((RETEST / "FREEZE.json").read_text(encoding="utf-8"))
    for rel, digest in freeze["files"].items():
        require(sha256_file(RETEST / rel) == digest, f"freeze drift {rel}")
    require(freeze["protocol_sha256"] == pins["protocol_sha256"], "protocol pin drift")
    require(not (OUT / "RESULTS.json").exists(), "RESULTS.json already exists; not a first measurement")


def pin_paths(pins: dict) -> None:
    paths = {
        pins["protocol_sha256"]: RETEST / "PROTOCOL.md",
        pins["checkpoints"]["curve_C_lr003_s0"]: RUNS / "curve_C_lr003_s0/checkpoint.pt",
        pins["checkpoints"]["curve_C_lr003_s1"]: RUNS / "curve_C_lr003_s1/checkpoint.pt",
        pins["checkpoints"]["curve_C_lr003_s2"]: RUNS / "curve_C_lr003_s2/checkpoint.pt",
        pins["released_zip_sha256"]: ZIP_PATH,
        pins["released_test_csv_sha256"]: TEST_CSV,
        pins["source_hashes_receipt_sha256"]: REPO / "experiments/cnn03_convergence/candidate_t146_ensemble_s0_s2_20261108/SOURCE_HASHES.txt",
        pins["production_PINNED_sha256"]: REPO / "experiments/mlp_cnn_hpo/review/PINNED.json",
        pins["production_fp64_harness_sha256"]: REPO / "experiments/cnn_runtime/fp64_harness.py",
        pins["common_py_sha256"]: COMMON_PATH,
        pins["split_ids_sha256"]: REPO / "experiments/cnn03_convergence/ensemble_blend_20261108/SPLIT_IDS.json",
        pins["kit_scoring_sha256"]: REPO / "kit/slot_forecasting/scoring.py",
        pins["kit_contest_sha256"]: REPO / "kit/slot_forecasting/contest.py",
        pins["house_edge_scoring_sha256"]: REPO / "src/house_edge/scoring.py",
        pins["base_cnn_sha256"]: STUDY_SOURCE / "base_cnn.py",
        pins["conv_train_sha256"]: STUDY_SOURCE / "conv_train.py",
        pins["queue1_sha256"]: STUDY_SOURCE / "queue1.sh",
        pins["queue2_sha256"]: STUDY_SOURCE / "queue2_C_winner_seeds.sh",
        pins["verified_pair_npz_sha256"]: NPZ,
    }
    for digest, path in paths.items():
        require(sha256_file(path) == digest, f"pin mismatch {path}")


def identity(pins: dict) -> dict:
    import torch
    rows = {}
    states = {}
    for name, seed in (("curve_C_lr003_s0", 0), ("curve_C_lr003_s1", 1), ("curve_C_lr003_s2", 2)):
        folder = RUNS / name
        names = sorted(path.name for path in folder.iterdir())
        blob = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
        require(set(blob) == {"config", "state"}, f"{name} checkpoint keys {set(blob)}")
        require(int(blob["config"]["seed"]) == seed, f"{name} config seed")
        require(float(blob["config"]["lr"]) == 0.003, f"{name} lr")
        init_names = [item for item in names if "init" in item or "warm" in item or item.startswith("optimizer")]
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        rows[name] = {
            "files": names,
            "initial_weight_files": init_names,
            "config_seed": int(blob["config"]["seed"]),
            "config_out": blob["config"]["out"],
            "summary_final_tune_R_training_log": summary["final_tune_R"],
            "summary_steps": summary["total_steps"],
            "log_sha256": sha256_file(RUNS / f"{name}.log"),
            "summary_sha256": sha256_file(folder / "summary.json"),
        }
        states[name] = {key: value.detach().float().numpy() for key, value in blob["state"].items()}
    pairs = {}
    keys = list(states["curve_C_lr003_s0"])
    for left, right in (("s0", "s1"), ("s0", "s2"), ("s1", "s2")):
        a = states[f"curve_C_lr003_{left}"]
        b = states[f"curve_C_lr003_{right}"]
        require(set(a) == set(b) == set(keys), "state keys differ")
        worst = 0.0
        identical = 0
        for key in keys:
            gap = float(np_max_abs(a[key], b[key]))
            worst = max(worst, gap)
            if gap == 0.0:
                identical += 1
        pairs[f"{left}_vs_{right}"] = {"max_abs": worst, "identical_tensors": identical, "tensors": len(keys)}
        require(worst > 0.0, f"{left} and {right} tensors are identical")
    text = (STUDY_SOURCE / "conv_train.py").read_text(encoding="utf-8")
    require("torch.manual_seed(a.seed)" in text and "CNN(" in text, "seed-then-construct text missing")
    require("load_state_dict" not in text and "warm" not in text, "conv_train contains a warm start")
    seed_at = text.index("torch.manual_seed(a.seed)")
    construct_at = text.index("model = B.CNN(")
    save_at = text.index('torch.save({"state"')
    queue = (RUNS / "QUEUE_STATUS.txt").read_text(encoding="utf-8")
    return {
        "runs": rows,
        "tensor_pairs": pairs,
        "code_order": {
            "manual_seed_before_cnn": seed_at < construct_at < save_at,
            "checkpoint_is_final_state_only": True,
            "initial_weights_saved": False,
        },
        "queue_status_sha256": sha256_file(RUNS / "QUEUE_STATUS.txt"),
        "queue_has_three_curve_c_exits": all(
            line in queue for line in (
                "end curve_C_lr003_s0 exit 0",
                "end curve_C_lr003_s1 exit 0",
                "end curve_C_lr003_s2 exit 0",
            )
        ),
        "launch": {
            "s0": "queue1.sh --lr 0.003 --pct_start 0.3 --seed 0",
            "s1": "queue2_C_winner_seeds.sh --lr 0.003 --pct_start 0.3 --seed 1",
            "s2": "queue2_C_winner_seeds.sh --lr 0.003 --pct_start 0.3 --seed 2",
        },
    }


def np_max_abs(left, right) -> float:
    import numpy as np
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def hierarchy(contract: dict) -> dict:
    queries = REPO / "kit/public/validation/queries.csv"
    conditions = REPO / "kit/public/validation/conditions.csv"
    with queries.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with conditions.open(newline="", encoding="utf-8") as handle:
        cond = {row["query_id"]: row["condition"] for row in csv.DictReader(handle)}
    tune = []
    for row in rows:
        if row["episode_id"] in contract["tune_episodes"]:
            tune.append(row)
    by_episode = defaultdict(list)
    for row in tune:
        by_episode[row["episode_id"]].append(row)
    condition_of = {}
    machine_labels = defaultdict(set)
    per_episode_machines = {}
    counts = []
    for episode, group in by_episode.items():
        conditions_here = {cond[row["query_id"]] for row in group}
        require(len(conditions_here) == 1, f"{episode} spans conditions")
        condition_of[episode] = next(iter(conditions_here))
        labels = [row["machine_id"] for row in group]
        per_episode_machines[episode] = sorted(set(labels))
        for label in labels:
            machine_labels[label].add(episode)
        counts.append(len(group))
    global_labels = sorted(machine_labels)
    queries_per_episode = {episode: len(group) for episode, group in by_episode.items()}
    return {
        "tune_episodes": len(by_episode),
        "tune_queries": len(tune),
        "queries_per_episode_min": min(counts),
        "queries_per_episode_max": max(counts),
        "queries_per_episode": queries_per_episode,
        "condition_counts": {name: sum(value == name for value in condition_of.values()) for name in A.CONDITIONS},
        "episode_to_condition": condition_of,
        "machines_per_episode": {episode: len(labels) for episode, labels in per_episode_machines.items()},
        "machine_labels": global_labels,
        "episodes_sharing_each_machine_label": {label: len(machine_labels[label]) for label in global_labels},
        "query_columns": list(rows[0].keys()) if rows else [],
        "query_to_episode": {row["query_id"]: row["episode_id"] for row in tune},
        "query_to_machine": {row["query_id"]: row["machine_id"] for row in tune},
    }


def query_errors(preds: dict, truth: dict, query_ids) -> dict[str, float]:
    return {qid: A.squared_error(preds[qid], truth[qid]) for qid in query_ids}


def map_max_abs(left: dict, right: dict, query_ids) -> float:
    worst = 0.0
    for qid in query_ids:
        a = A.kit_probabilities(left[qid])
        b = A.kit_probabilities(right[qid])
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    return worst


def read_csv_map(path: Path) -> dict:
    out = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["query_id"]] = [float(row[name]) for name in ("p_0", "p_1", "p_2", "p_5", "p_20")]
    return out


def load_npz(path: Path) -> dict:
    import numpy as np
    blob = np.load(path)
    ids = [str(item) for item in blob["query_id"].tolist()]
    return {key: [float(v) for v in row] for key, row in zip(ids, blob["probability"])}


def extract_zip() -> None:
    require(not FRESH.exists(), "fresh zip extract already exists")
    FRESH.mkdir(parents=True)
    with zipfile.ZipFile(ZIP_PATH) as blob:
        for info in blob.infolist():
            if info.filename.startswith("/") or ".." in Path(info.filename).parts:
                raise SystemExit(f"MEASURE_REFUSED unsafe zip member {info.filename}")
        blob.extractall(FRESH)


def run_zip_predict(split: str, dest: Path) -> None:
    cmd = [str(PY), str(FRESH / "predict.py"), "--data", str(REPO / "kit/public"), "--out", str(dest), "--split", split]
    if Path("/usr/bin/time").is_file():
        cmd = ["/usr/bin/time", "-v", *cmd]
    proc = subprocess.run(cmd, cwd=REPO, env=os.environ.copy(), text=True, capture_output=True)
    (OUT / f"zip_{split}.err").write_text(proc.stderr, encoding="utf-8")
    require(proc.returncode == 0 and dest.is_file(), f"zip predict {split} failed {proc.stderr[-2000:]}")


def logit_maps(common, states: dict) -> dict[str, dict]:
    import torch
    model_mod = common.load_canonical_model()
    data_mod = common.load_canonical_data()
    validation = data_mod.load_public_split(REPO / "kit/public", "validation")
    query, episodes = model_mod.build_query(validation, common.W)
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
        raise SystemExit("MEASURE_REFUSED autocast is enabled")
    maps = {}
    for name, state in states.items():
        model_mod.USE_SIGNAL = False
        net = model_mod.CNN(common.H, common.LAYERS, common.K, common.DROP, common.W, use_signal=False)
        net.load_state_dict(state)
        net.double()
        net.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(query[0]), 4):
                batch = [item[start:start + 4] for item in query]
                logits = net(*batch)
                if logits.dtype != torch.float64:
                    raise SystemExit("MEASURE_REFUSED logits are not float64")
                chunks.append(logits.cpu())
        stacked = torch.cat(chunks).numpy()
        maps[name] = model_mod.to_preds(stacked, episodes, validation)
    return maps


def machine_rows(query_error_maps, info) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    for name, errors in query_error_maps.items():
        for qid, error in errors.items():
            episode = info["query_to_episode"][qid]
            machine = info["query_to_machine"][qid]
            grouped[(episode, machine)][name].append(error)
    rows = []
    for (episode, machine), columns in sorted(grouped.items()):
        row = {
            "episode_id": episode,
            "condition": info["episode_to_condition"][episode],
            "machine_id": machine,
            "n_queries": len(next(iter(columns.values()))),
        }
        for name, values in columns.items():
            row[name] = sum(values) / len(values)
        rows.append(row)
    return rows


def main() -> None:
    import numpy as np
    import torch
    torch.set_num_threads(2)
    pins = load_pins()
    check_freeze(pins)
    tune_before = sha256_file(TUNE_JSON)
    pin_paths(pins)
    OUT.mkdir(parents=True, exist_ok=True)
    ident = identity(pins)
    common = load_common()
    contract = common.tune_contract()
    info = hierarchy(contract)
    require(info["tune_episodes"] == 48 and info["tune_queries"] == 384, "tune hierarchy is not 48/384")
    states = {}
    prob_maps = {}
    for name in ("curve_C_lr003_s0", "curve_C_lr003_s1", "curve_C_lr003_s2"):
        blob = torch.load(RUNS / name / "checkpoint.pt", map_location="cpu", weights_only=False)
        states[name] = blob["state"]
        prob_maps[name] = common.forward_probabilities(blob["state"])
        print(f"FORWARD {name} n={len(prob_maps[name])}", flush=True)
    fresh_pair = common.equal_probability_mean([prob_maps["curve_C_lr003_s0"], prob_maps["curve_C_lr003_s2"]])
    fresh_equal3 = common.equal_probability_mean([
        prob_maps["curve_C_lr003_s0"], prob_maps["curve_C_lr003_s1"], prob_maps["curve_C_lr003_s2"],
    ])
    scored_maps = {
        "s0": prob_maps["curve_C_lr003_s0"],
        "s1": prob_maps["curve_C_lr003_s1"],
        "s2": prob_maps["curve_C_lr003_s2"],
        "equal3": fresh_equal3,
        "pair02": fresh_pair,
    }
    scores = {}
    for name, preds in scored_maps.items():
        detail = common.score_tune_detail(preds)
        require(detail["n_tune_queries"] == 384 and detail["n_tune_episodes"] == 48, f"{name} counts")
        require(detail["n_forward_queries"] == 768, f"{name} forward {detail['n_forward_queries']}")
        scores[name] = detail
        print(f"SCORE {name} R={detail['tune_R']:.17f}", flush=True)
    extract_zip()
    run_zip_predict("validation", OUT / "zip_val.csv")
    run_zip_predict("test", OUT / "zip_test.csv")
    zip_val = read_csv_map(OUT / "zip_val.csv")
    zip_detail = common.score_tune_detail(zip_val)
    require(zip_detail["n_tune_queries"] == 384 and zip_detail["n_forward_queries"] == 768, "zip tune counts")
    test_same = (OUT / "zip_test.csv").read_bytes() == TEST_CSV.read_bytes()
    from house_edge.scoring import load_official_validation
    official = load_official_validation(REPO / "kit/public/validation")
    targets, _groups, reference, query_eps = official
    tune_ids = contract["tune_query_ids"]
    truth = {row.query_id: list(row.probabilities) for row in targets if row.query_id in tune_ids}
    ref_rows = {qid: list(reference[qid]) for qid in tune_ids}
    query_to_episode = {qid: query_eps[qid] for qid in tune_ids}
    require(query_to_episode == info["query_to_episode"], "tune query episode map drifted")
    model_ep = {}
    query_error_maps = {}
    for name, preds in scored_maps.items():
        qerr = query_errors(preds, truth, tune_ids)
        query_error_maps[name] = qerr
        model_ep[name] = A.episode_mean_errors(qerr, query_to_episode)
    ref_ep = A.episode_mean_errors(query_errors(ref_rows, truth, tune_ids), query_to_episode)
    draws = A.ordered_episode_ids(info["episode_to_condition"])
    rebuilt = {name: A.canonical_r(errors, ref_ep, info["episode_to_condition"], draws) for name, errors in model_ep.items()}
    for name, detail in scores.items():
        require(abs(rebuilt[name] - detail["tune_R"]) <= 1e-12, f"reconstruction {name}")
    logits = logit_maps(common, states)
    import numpy as np
    tune_query_list = sorted(tune_ids)
    def stack(rows):
        return np.array([[float(v) for v in rows[qid]] for qid in tune_query_list], dtype=np.float64)
    real_gap = A.probability_versus_logit_gap(
        stack(prob_maps["curve_C_lr003_s0"]), stack(prob_maps["curve_C_lr003_s2"]),
        stack(logits["curve_C_lr003_s0"]), stack(logits["curve_C_lr003_s2"]),
    )
    stored_pair = load_npz(NPZ)
    stored_eq3 = load_npz(EQ3_NPZ)
    comparisons = {
        "pair02_minus_s2": ("pair02", "s2"),
        "equal3_minus_s2": ("equal3", "s2"),
        "pair02_minus_equal3": ("pair02", "equal3"),
    }
    boot = A.paired_bootstrap(model_ep, ref_ep, info["episode_to_condition"], draws)
    intervals = {}
    for name, values in boot["replicate_R"].items():
        intervals[name] = A.percentile_interval(values)
    delta_intervals = {}
    loo = {}
    gaps = {}
    for label, (left, right) in comparisons.items():
        delta = boot["replicate_R"][left] - boot["replicate_R"][right]
        delta_intervals[label] = A.percentile_interval(delta)
        loo[label] = A.leave_one_out(model_ep[left], model_ep[right], ref_ep, info["episode_to_condition"], draws)
        gaps[label] = A.error_gap_summary(model_ep[left], model_ep[right], draws)
    replicate_path = OUT / "bootstrap_replicates.npz"
    np.savez_compressed(
        replicate_path,
        draws=boot["draws"],
        episode_order=np.array(boot["episode_order"]),
        **{f"R_{name}": values for name, values in boot["replicate_R"].items()},
        **{f"delta_{label}": boot["replicate_R"][left] - boot["replicate_R"][right] for label, (left, right) in comparisons.items()},
    )
    prior = pins["prior_tune_R"]
    reproduction = {
        name: {
            "tune_R": scores[name]["tune_R"],
            "prior": prior[name],
            "abs_delta": abs(scores[name]["tune_R"] - prior[name]),
            "within_1e-12": abs(scores[name]["tune_R"] - prior[name]) <= 1e-12,
            "rebuilt_R": rebuilt[name],
            "rebuilt_abs_delta": abs(rebuilt[name] - scores[name]["tune_R"]),
        }
        for name in ("s0", "s1", "s2", "equal3", "pair02")
    }
    reproduction["zip_pair"] = {
        "tune_R": zip_detail["tune_R"],
        "abs_delta_vs_fresh_pair": abs(zip_detail["tune_R"] - scores["pair02"]["tune_R"]),
        "within_1e-12": abs(zip_detail["tune_R"] - scores["pair02"]["tune_R"]) <= 1e-12,
        "test_csv_byte_identical": test_same,
        "test_csv_sha256": sha256_file(OUT / "zip_test.csv"),
        "val_kit_max_abs_vs_fresh_pair": map_max_abs(zip_val, fresh_pair, tune_ids),
    }
    maps = {
        "fresh_pair_vs_stored_npz": map_max_abs(fresh_pair, stored_pair, tune_ids),
        "fresh_equal3_vs_stored_npz": map_max_abs(fresh_equal3, stored_eq3, tune_ids),
        "real_probability_vs_logit_mean_max_abs": real_gap,
        "logit_ensemble_tune_R": "NOT_COMPUTED",
        "synthetic_logit_gap": A.synthetic_logit_gap(),
        "synthetic_ratio": A.synthetic_ratio_counterexample(),
    }
    episode_table = []
    for episode in draws:
        episode_table.append({
            "episode_id": episode,
            "condition": info["episode_to_condition"][episode],
            "n_queries": info["queries_per_episode"][episode],
            "n_machine_labels": len(set(info["query_to_machine"][qid] for qid in tune_ids if info["query_to_episode"][qid] == episode)),
            **{name: model_ep[name][episode] for name in model_ep},
            "reference": ref_ep[episode],
        })
    loo_summary = {}
    for label, rows in loo.items():
        deltas = [row["delta_R"] for row in rows]
        loo_summary[label] = {
            "min_delta_R": min(deltas),
            "max_delta_R": max(deltas),
            "sign_reversals": sum(bool(row["sign_reversal"]) for row in rows),
            "full_delta_R": rows[0]["full_delta_R"],
        }
    id_hash = hashlib.sha256("\n".join(draws).encode("utf-8")).hexdigest()
    q_hash = hashlib.sha256("\n".join(sorted(tune_ids)).encode("utf-8")).hexdigest()
    all_reproduced = all(item["within_1e-12"] for item in reproduction.values() if isinstance(item, dict) and "within_1e-12" in item)
    tensors_differ = all(item["max_abs"] > 0.0 for item in ident["tensor_pairs"].values())
    three_runs = tensors_differ and ident["queue_has_three_curve_c_exits"] and ident["code_order"]["manual_seed_before_cnn"]
    verdicts = {
        "identity_three_distinct_seed_training_runs": "VERIFIED" if three_runs else "NOTVERIFIED",
        "identity_saved_initial_weights": "NOTVERIFIED",
        "identity_empirical_initialization_difference": "INCONCLUSIVE",
        "identity_independent_datasets": "NOTVERIFIED",
        "reproduction": "VERIFIED" if all_reproduced and maps["fresh_pair_vs_stored_npz"] <= 1e-8 else "NOTVERIFIED",
        "robustness_generalization": "INCONCLUSIVE",
    }
    machines = machine_rows(query_error_maps, info)
    payload = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "verdicts": verdicts,
        "identity": ident,
        "hierarchy": {
            "tune_episodes": info["tune_episodes"],
            "tune_queries": info["tune_queries"],
            "queries_per_episode_min": info["queries_per_episode_min"],
            "queries_per_episode_max": info["queries_per_episode_max"],
            "condition_counts": info["condition_counts"],
            "machine_labels": info["machine_labels"],
            "episodes_sharing_each_machine_label": info["episodes_sharing_each_machine_label"],
            "query_columns": info["query_columns"],
            "episode_id_sha256": id_hash,
            "tune_query_id_sha256": q_hash,
            "dependence_note": "Query columns are query_id, episode_id, machine_id, machine_type, signal, cutoff. There is no id above episode_id. machine_id strings in this file embed the episode id, so a count of episodes per label is required and a unique label is not a globally independent machine. Canonical R pools the 48 episodes inside four conditions, so the episodes are not 48 independent copies of R.",
        },
        "reproduction": reproduction,
        "maps": maps,
        "bootstrap": {
            "replicates": 10000,
            "seed": 15020261108,
            "generator": "numpy.random.default_rng",
            "protocol_generator_name": "np.default_rng",
            "numpy_has_top_level_default_rng": False,
            "unit": "whole episode",
            "quantile": "linear 2.5/97.5",
            "conditional_on": "reused selected 384-query tune split",
            "not_a_proof_of": ["absence of selection luck", "blind confirmation", "hidden-test generalization"],
            "model_intervals": intervals,
            "paired_delta_intervals": delta_intervals,
            "replicates_sha256": sha256_file(replicate_path),
        },
        "loo": loo_summary,
        "episode_error_gaps": gaps,
        "point_deltas": {label: scores[left]["tune_R"] - scores[right]["tune_R"] for label, (left, right) in comparisons.items()},
    }
    (OUT / "episode_errors.json").write_text(json.dumps(episode_table, indent=1) + "\n", encoding="utf-8")
    (OUT / "loo.json").write_text(json.dumps(loo, indent=1) + "\n", encoding="utf-8")
    (OUT / "machine_errors.json").write_text(json.dumps(machines, indent=1) + "\n", encoding="utf-8")
    (OUT / "RESULTS.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tune_after = sha256_file(TUNE_JSON)
    require(tune_after == tune_before, "candidate tune_only.json changed")
    pin_paths(pins)
    print("MEASURE_OK", flush=True)


if __name__ == "__main__":
    main()
