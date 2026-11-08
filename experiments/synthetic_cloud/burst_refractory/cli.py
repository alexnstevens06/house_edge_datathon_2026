"""Bounded command line for the internal synthetic generators.

The command writes an event JSONL file and a separate latent-truth JSON file.
It does not read an input dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from burst_refractory.generators import ParameterError, generate_burst, generate_refractory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="burst-refractory",
        description=(
            "Write a small synthetic 8-machine episode. "
            "Internal research schema only. Not an organizer schema."
        ),
    )
    subparsers = parser.add_subparsers(dest="process", required=True)
    for name, help_text in (
        ("burst", "Bursty self-exciting categorical rewards"),
        ("refractory", "Refractory cooldown renewal rewards with machine-type variation"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--seed", type=int, required=True)
        sub.add_argument("--episodes", type=int, required=True)
        sub.add_argument("--ticks", type=int, required=True)
        sub.add_argument("--observe-probability", type=float, default=0.75)
        sub.add_argument("--out-dir", type=str, required=True)
    return parser


def _bounded_out_dir(raw: str) -> Path:
    if not isinstance(raw, str) or raw == "" or len(raw) > 180:
        raise ParameterError("out-dir must be a path of at most 180 characters")
    if "\x00" in raw:
        raise ParameterError("out-dir must not contain a NUL")
    path = Path(raw)
    if any(part == ".." for part in path.parts):
        raise ParameterError("out-dir must not contain '..'")
    return path


def _write_outputs(out_dir: Path, events: list[dict[str, object]], truth: dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    truth_path = out_dir / "truth.json"
    with open(events_path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump(truth, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Parse bounded arguments, generate one process, and write two files."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        out_dir = _bounded_out_dir(args.out_dir)
        config = {
            "seed": args.seed,
            "n_episodes": args.episodes,
            "n_ticks": args.ticks,
            "observe_probability": args.observe_probability,
        }
        if args.process == "burst":
            events, truth = generate_burst(config)
        elif args.process == "refractory":
            events, truth = generate_refractory(config)
        else:
            raise ParameterError("unknown process")
        _write_outputs(out_dir, events, truth)
    except ParameterError as exc:
        print(f"parameter error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    return 0
