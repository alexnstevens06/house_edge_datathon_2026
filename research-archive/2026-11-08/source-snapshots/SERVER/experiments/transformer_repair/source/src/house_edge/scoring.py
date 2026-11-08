"""Official contest scoring, reused from the organizer kit (never reimplemented)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_DIR = REPO_ROOT / "kit"
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))

from slot_forecasting.contest import contest_score  # noqa: E402
from slot_forecasting.io import read_rows, read_submission, read_targets  # noqa: E402
from slot_forecasting.schema import PROBABILITY_COLUMNS  # noqa: E402

MAX_POINTS_MAIN = 60


def load_official_validation(validation_dir: str | Path):
    validation_dir = Path(validation_dir)
    targets = read_targets(validation_dir / "targets.csv")
    groups = {r["query_id"]: r["condition"]
              for r in read_rows(validation_dir / "conditions.csv", ("query_id", "condition"))}
    reference = read_submission(validation_dir / "reference.csv")
    queries = {r["query_id"]: r["episode_id"] for r in read_rows(validation_dir / "queries.csv", (
        "query_id", "episode_id", "machine_id", "machine_type", "signal", "cutoff"))}
    return targets, groups, reference, queries


def score_predictions(predictions: dict[str, list[float]], official, episode_subset=None,
                      max_points: float = MAX_POINTS_MAIN) -> dict:
    """Official per-condition episode-averaged squared error, ratio vs frozen
    reference, R = mean ratio, points = max_points / (1 + R).

    episode_subset: optional set of episode_ids to restrict to (TUNE/CONFIRM).
    """
    targets, groups, reference, queries = official
    if episode_subset is not None:
        targets = tuple(t for t in targets if queries[t.query_id] in episode_subset)
        keep = {t.query_id for t in targets}
        groups = {q: c for q, c in groups.items() if q in keep}
        reference = {q: p for q, p in reference.items() if q in keep}
        predictions = {q: p for q, p in predictions.items() if q in keep}
    preds = {q: [float(v) for v in p] for q, p in predictions.items()}
    result = contest_score(preds, targets, groups, reference, max_points=max_points)
    ratio = result["mean_loss_ratio"]
    return {
        "R": ratio,
        "points_60": 60.0 / (1.0 + ratio),
        "points_80": 80.0 / (1.0 + ratio),
        "probability_error": {c: v["error"] for c, v in result["by_condition"].items()},
        "reference_error": {c: v["reference_error"] for c, v in result["by_condition"].items()},
        "loss_ratio_by_condition": {c: v["loss_ratio"] for c, v in result["by_condition"].items()},
        "episodes": {c: v["episodes"] for c, v in result["by_condition"].items()},
    }


def write_submission_csv(path: str | Path, predictions: dict[str, list[float]]) -> None:
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("query_id", *PROBABILITY_COLUMNS))
        for query_id in sorted(predictions):
            values = [float(v) for v in predictions[query_id]]
            total = sum(values)
            values = [v / total for v in values]  # exact renormalization in float64
            writer.writerow([query_id, *[repr(v) for v in values]])
