#!/usr/bin/env python3
"""Score a directory of estimate-evaluation records without live calls.

Each ``record-*.json`` file holds one frozen prediction paired with its
independent observation (see ``llm_launchpad/core/evaluation_store.py``).
This script loads them, scores them, persists ``scorecard.json`` beside them,
and prints the human-readable summary.

Nothing here deploys, benchmarks, or reads the live calibration and
certificate caches: scoring stays isolated from the evidence the predictor
learns from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_launchpad.core.evaluation_store import (
    format_scorecard,
    score_run_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        help="Directory holding record-*.json files.",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"No such directory: {run_dir}")
        return 2
    records, scorecard, report_path = score_run_dir(run_dir)
    print(f"records: {len(records)} from {run_dir}")
    for line in format_scorecard(scorecard):
        print(f"  {line}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
