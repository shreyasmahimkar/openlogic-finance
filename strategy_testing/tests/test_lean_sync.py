"""Guards that the LEAN project strategy copies stay in sync with model_library.

LEAN projects are self-contained (run in an isolated container), so they vendor
the strategy modules — but model_library is the source of truth. Regenerate with
`make sync-lean` if this fails.
"""

import filecmp
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "sync_lean_strategies", ROOT / "scripts/sync_lean_strategies.py"
)
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


def test_lean_copies_match_canonical():
    stale = []
    for project in sync.LEAN_PROJECTS:
        for name, src in sync.CANONICAL.items():
            dest = project / name
            if not (dest.exists() and filecmp.cmp(src, dest, shallow=False)):
                stale.append(str(dest.relative_to(ROOT)))
    assert not stale, f"LEAN copies out of sync (run `make sync-lean`): {stale}"
