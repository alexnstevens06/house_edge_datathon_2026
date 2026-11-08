"""Negative tests for the frozen selector. They do not score a model."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))

import ranking as S  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def row(name: str, tune_r: float, **kwargs) -> dict:
    base = {
        "name": name,
        "status": "scored",
        "kind": "single",
        "tune_R": tune_r,
        "lr": 0.003,
        "epoch": None,
        "n_components": 1,
        "blend_weight": None,
        "is_current_best": name == "current_best",
    }
    base.update(kwargs)
    return base


def blocked(name: str) -> dict:
    return {
        "name": name,
        "status": "not_run_preflight_blocked",
        "kind": "single",
        "tune_R": None,
        "lr": 5e-5,
        "epoch": int(name[-1]),
        "n_components": 1,
        "blend_weight": None,
        "is_current_best": False,
    }


def full_set(overrides: dict[str, dict] | None = None) -> list[dict]:
    overrides = overrides or {}
    rows = []
    for name in S.SCORED_NAMES:
        if name in overrides:
            rows.append(overrides[name])
            continue
        if name == "current_best":
            rows.append(row(name, S.CURRENT_BEST_R))
        elif name.startswith("ensemble_equal_s0_s1"):
            rows.append(row(name, 0.5, kind="ensemble", n_components=3, lr=None))
        elif name.startswith("ensemble"):
            rows.append(row(name, 0.5, kind="ensemble", n_components=2, lr=None))
        elif name.startswith("blend"):
            weight = float(name.rsplit("w", 1)[1])
            rows.append(row(name, 0.5, kind="blend", n_components=2, blend_weight=weight, lr=None))
        elif "primary" in name:
            epoch = int(name[-1])
            rows.append(row(name, 0.5, lr=2e-5, epoch=epoch))
        else:
            epoch = int(name[-1])
            rows.append(row(name, 0.5, lr=2e-5, epoch=epoch))
    for name in S.BLOCKED_NAMES:
        rows.append(overrides.get(name, blocked(name)))
    return rows


def test_threshold_and_current_best_tie() -> None:
    below = S.CURRENT_BEST_R - 5e-7
    above = S.CURRENT_BEST_R - 5e-6
    exact = S.CURRENT_BEST_R - S.THRESHOLD
    kept = S.select(full_set({
        "server_primary_lr2e-5_epoch1": row("server_primary_lr2e-5_epoch1", below, lr=2e-5, epoch=1),
    }))
    check(kept["selected_name"] == "current_best", f"sub-threshold selected {kept['selected_name']}")
    check(kept["would_request_package"] is False, "sub-threshold requested a package")
    cleared = S.select(full_set({
        "server_primary_lr2e-5_epoch1": row("server_primary_lr2e-5_epoch1", above, lr=2e-5, epoch=1),
    }))
    check(cleared["selected_name"] == "server_primary_lr2e-5_epoch1", "clear improvement lost")
    boundary = S.select(full_set({
        "server_primary_lr2e-5_epoch1": row("server_primary_lr2e-5_epoch1", exact, lr=2e-5, epoch=1),
    }))
    improvement = S.CURRENT_BEST_R - exact
    if improvement >= S.THRESHOLD:
        check(boundary["selected_name"] == "server_primary_lr2e-5_epoch1", "exact 1e-6 did not clear")
    else:
        check(boundary["selected_name"] == "current_best", "inexact 1e-6 gap displaced current best")


def test_tie_breaks() -> None:
    best = S.CURRENT_BEST_R
    ensemble_r = best - 1.5e-6
    single_r = best - 1.2e-6
    tied = S.select(full_set({
        "ensemble_equal_s0_s2": row(
            "ensemble_equal_s0_s2", ensemble_r, kind="ensemble", n_components=2, lr=None
        ),
        "server_primary_lr2e-5_epoch1": row(
            "server_primary_lr2e-5_epoch1", single_r, lr=2e-5, epoch=1
        ),
    }))
    check(tied["selected_name"] == "server_primary_lr2e-5_epoch1", "single lost a within-1e-6 tie to an ensemble")
    wide = S.select(full_set({
        "ensemble_equal_s0_s2": row(
            "ensemble_equal_s0_s2", best - 3e-6, kind="ensemble", n_components=2, lr=None
        ),
        "server_primary_lr2e-5_epoch1": row(
            "server_primary_lr2e-5_epoch1", best - 1e-6, lr=2e-5, epoch=1
        ),
    }))
    check(wide["selected_name"] == "ensemble_equal_s0_s2", "lower R lost a non-tie")
    shared = best - 0.01
    lower_lr = S.select(full_set({
        "server_primary_lr2e-5_epoch5": row("server_primary_lr2e-5_epoch5", shared, lr=2e-5, epoch=5),
        "server_control_lr2e-5_epoch1": row("server_control_lr2e-5_epoch1", shared, lr=5e-5, epoch=1),
    }))
    check(lower_lr["selected_name"] == "server_primary_lr2e-5_epoch5", "higher LR won a tie")
    earlier = S.select(full_set({
        "server_primary_lr2e-5_epoch1": row("server_primary_lr2e-5_epoch1", shared, lr=2e-5, epoch=1),
        "server_primary_lr2e-5_epoch5": row("server_primary_lr2e-5_epoch5", shared, lr=2e-5, epoch=5),
    }))
    check(earlier["selected_name"] == "server_primary_lr2e-5_epoch1", "later epoch won a tie")
    fewer = S.select(full_set({
        "ensemble_equal_s0_s2": row("ensemble_equal_s0_s2", shared, kind="ensemble", n_components=2, lr=None),
        "ensemble_equal_s0_s1_s2": row(
            "ensemble_equal_s0_s1_s2", shared, kind="ensemble", n_components=3, lr=None
        ),
    }))
    check(fewer["selected_name"] == "ensemble_equal_s0_s2", "more components won a tie")
    smaller = S.select(full_set({
        "blend_s2_control_w0.025": row(
            "blend_s2_control_w0.025", shared, kind="blend", n_components=2, blend_weight=0.025, lr=None
        ),
        "blend_s2_control_w0.2": row(
            "blend_s2_control_w0.2", shared, kind="blend", n_components=2, blend_weight=0.2, lr=None
        ),
    }))
    check(smaller["selected_name"] == "blend_s2_control_w0.025", "larger blend weight won a tie")


def test_blocked_and_extra_are_refused() -> None:
    rows = full_set({
        "ensemble_equal_s0_s2": row(
            "ensemble_equal_s0_s2", 0.285, kind="ensemble", n_components=2, lr=None
        ),
    })
    chosen = S.select(rows)
    check(chosen["selected_name"] == "ensemble_equal_s0_s2", f"predeclared lowest was {chosen['selected_name']}")
    check("tower_primary_lr5e-5_epoch1" not in chosen["ranking"], "blocked row entered the ranking")
    check(len(chosen["blocked_not_scored"]) == 6, "blocked count")
    poisoned = full_set()
    poisoned[-1]["tune_R"] = 0.0
    try:
        S.select(poisoned)
        check(False, "a zero score on a blocked row was accepted")
    except RuntimeError as exc:
        check("must not carry a score" in str(exc), str(exc))
    extra = full_set()
    extra.append(row("blend_s2_control_w0.3", 0.01, kind="blend", n_components=2, blend_weight=0.3, lr=None))
    try:
        S.select(extra)
        check(False, "an unlisted blend weight was accepted")
    except RuntimeError as exc:
        check("outside the predeclared set" in str(exc), str(exc))
    missing = [item for item in full_set() if item["name"] != "current_best"]
    try:
        S.select(missing)
        check(False, "a missing current best was accepted")
    except RuntimeError as exc:
        check("missing" in str(exc), str(exc))


def main() -> None:
    test_threshold_and_current_best_tie()
    test_tie_breaks()
    test_blocked_and_extra_are_refused()
    if FAILURES:
        print("SELECT_TESTS_FAIL")
        for item in FAILURES:
            print(item)
        raise SystemExit(1)
    print("SELECT_TESTS_OK")


if __name__ == "__main__":
    main()
