"""Seed-0 entry point. ADDENDUM: one full run, seed 0 only. No confirm scoring."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hard_signal as H  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report-seen-confirm", action="store_true")
    args = ap.parse_args(argv)
    H.require_seed0(args.seed)
    if args.report_seen_confirm:
        raise SystemExit("rejected --report-seen-confirm: ADDENDUM forbids confirm/test scoring")
    H.train_seed0(args.out)


if __name__ == "__main__":
    main()
