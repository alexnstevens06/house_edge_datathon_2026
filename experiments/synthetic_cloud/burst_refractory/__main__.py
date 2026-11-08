"""CLI entry: python -m burst_refractory burst|refractory ..."""

from __future__ import annotations

import sys

from burst_refractory.cli import main

if __name__ == "__main__":
    sys.exit(main())
