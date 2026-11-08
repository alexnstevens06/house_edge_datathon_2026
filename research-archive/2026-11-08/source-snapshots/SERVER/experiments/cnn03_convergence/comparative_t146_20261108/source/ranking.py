"""Predeclared 1e-6 ranking. No I/O and no new combinations."""
from __future__ import annotations

CURRENT_BEST_R = 0.30980333573689856
THRESHOLD = 1e-6

SCORED_NAMES = (
    "current_best",
    "ensemble_equal_s0_s1_s2",
    "ensemble_equal_s0_s2",
    "blend_s2_control_w0.0",
    "blend_s2_control_w0.025",
    "blend_s2_control_w0.05",
    "blend_s2_control_w0.1",
    "blend_s2_control_w0.2",
    "server_primary_lr2e-5_epoch1",
    "server_primary_lr2e-5_epoch3",
    "server_primary_lr2e-5_epoch5",
    "server_control_lr2e-5_epoch1",
    "server_control_lr2e-5_epoch3",
    "server_control_lr2e-5_epoch5",
)
BLOCKED_NAMES = (
    "tower_primary_lr5e-5_epoch1",
    "tower_primary_lr5e-5_epoch3",
    "tower_primary_lr5e-5_epoch5",
    "tower_control_lr5e-5_epoch1",
    "tower_control_lr5e-5_epoch3",
    "tower_control_lr5e-5_epoch5",
)


def _secondary(row: dict) -> tuple:
    kind = 0 if row["kind"] == "single" else 1
    lr = row["lr"] if row["lr"] is not None else 1e9
    epoch = row["epoch"] if row["epoch"] is not None else 10**6
    weight = row["blend_weight"] if row["blend_weight"] is not None else -1.0
    return (kind, float(lr), int(epoch), int(row["n_components"]), float(weight), row["name"])


def is_r_tie(left: float, right: float) -> bool:
    """A gap below 1e-6 is a tie. A gap of exactly 1e-6 is a lower-R win."""
    return abs(float(left) - float(right)) < THRESHOLD


def prefer(left: dict, right: dict) -> dict:
    if is_r_tie(left["tune_R"], right["tune_R"]):
        if left.get("is_current_best") and not right.get("is_current_best"):
            return left
        if right.get("is_current_best") and not left.get("is_current_best"):
            return right
        left_key = _secondary(left)
        right_key = _secondary(right)
        if left_key < right_key:
            return left
        if right_key < left_key:
            return right
        return left if left["name"] <= right["name"] else right
    return left if float(left["tune_R"]) < float(right["tune_R"]) else right


def _require_shape(row: dict, scored: bool) -> None:
    for key in ("name", "status", "kind", "n_components"):
        if key not in row:
            raise RuntimeError(f"row missing {key}")
    if scored:
        if row["status"] != "scored":
            raise RuntimeError(f"{row['name']} is not scored")
        if row.get("tune_R") is None:
            raise RuntimeError(f"{row['name']} has no tune R")
        value = float(row["tune_R"])
        if value != value or value in (float("inf"), float("-inf")):
            raise RuntimeError(f"{row['name']} tune R is not finite")
    else:
        if row["status"] != "not_run_preflight_blocked":
            raise RuntimeError(f"{row['name']} blocked status is wrong")
        if row.get("tune_R") is not None:
            raise RuntimeError(f"{row['name']} blocked row must not carry a score")


def validate(rows: list[dict]) -> None:
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate candidate name")
    allowed = set(SCORED_NAMES) | set(BLOCKED_NAMES)
    extra = sorted(set(names) - allowed)
    if extra:
        raise RuntimeError("candidate outside the predeclared set: " + ", ".join(extra))
    missing = sorted(allowed - set(names))
    if missing:
        raise RuntimeError("predeclared candidate missing: " + ", ".join(missing))
    by_name = {row["name"]: row for row in rows}
    for name in SCORED_NAMES:
        _require_shape(by_name[name], scored=True)
    for name in BLOCKED_NAMES:
        _require_shape(by_name[name], scored=False)
    if not by_name["current_best"].get("is_current_best"):
        raise RuntimeError("current best flag missing")
    if abs(float(by_name["current_best"]["tune_R"]) - CURRENT_BEST_R) > 0.0:
        raise RuntimeError("current best tune R was replaced")


def ranking(rows: list[dict]) -> list[dict]:
    validate(rows)
    scored = [row for row in rows if row["status"] == "scored"]

    def compare(left: dict, right: dict) -> int:
        winner = prefer(left, right)
        if winner["name"] == left["name"] and winner["name"] == right["name"]:
            return 0
        return -1 if winner["name"] == left["name"] else 1

    from functools import cmp_to_key

    ordered = sorted(scored, key=cmp_to_key(compare))
    return ordered


def select(rows: list[dict]) -> dict:
    ordered = ranking(rows)
    winner = ordered[0]
    improvement = CURRENT_BEST_R - float(winner["tune_R"])
    clears = (not winner.get("is_current_best")) and improvement >= THRESHOLD
    blocked = [row["name"] for row in rows if row["status"] == "not_run_preflight_blocked"]
    return {
        "selected_name": winner["name"],
        "selected_tune_R": float(winner["tune_R"]),
        "improvement_vs_current_best": improvement,
        "clears_1e-6": clears,
        "package_written": False,
        "would_request_package": clears,
        "ranking": [row["name"] for row in ordered],
        "blocked_not_scored": blocked,
    }
