#!/usr/bin/env python3
"""Sync the canonical strategy code into the self-contained LEAN projects.

LEAN runs each project in an isolated container that does **not** have the repo
on its path, so each project must physically contain its strategy modules. That
makes the copies a LEAN execution requirement — but the **source of truth is
`model_library`**. This script regenerates the copies so they never drift
(resolves the duplication flagged in docs/memory/0003-import-dont-copy.md).

Usage:
    python scripts/sync_lean_strategies.py            # copy canonical -> LEAN projects
    python scripts/sync_lean_strategies.py --check    # fail if any copy is stale (CI)
"""

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAN_PROJECTS = [
    ROOT / "strategy_testing/lean_engine/logistic_regression_project",
    ROOT / "strategy_testing/lean_engine/sma_crossover_project",
]
# filename in each LEAN project  ->  canonical source in model_library
CANONICAL = {
    "logistic_regression.py": ROOT / "model_library/ml_zoo/logistic_regression.py",
    "sma_crossover_signal.py": ROOT / "model_library/technical/signals/sma_crossover_signal.py",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if any copy is out of sync")
    args = ap.parse_args()

    stale = []
    for project in LEAN_PROJECTS:
        for name, src in CANONICAL.items():
            dest = project / name
            in_sync = dest.exists() and filecmp.cmp(src, dest, shallow=False)
            if args.check:
                if not in_sync:
                    stale.append(str(dest.relative_to(ROOT)))
            elif not in_sync:
                shutil.copyfile(src, dest)
                print(f"synced {dest.relative_to(ROOT)}")

    if args.check and stale:
        print("Out of sync (run `make sync-lean`):\n  " + "\n  ".join(stale), file=sys.stderr)
        return 1
    if not args.check:
        print("LEAN strategy projects are in sync with model_library.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
