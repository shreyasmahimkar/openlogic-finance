"""Guards the ADK evalset files from rot: they must stay schema-valid.

Runs without API keys (validation only, no agent invocation). The scored
`adk eval` run is a separate, keyed step (see the eval/ README).
"""

import glob
import json

import pytest
from google.adk.evaluation.eval_set import EvalSet

# All evalsets across the repo (flagship agent + per-box connector agents).
EVALSETS = sorted(
    p for p in glob.glob("**/eval/*.evalset.json", recursive=True) if ".openlogic-env" not in p
)


def test_evalsets_exist():
    assert EVALSETS, "no evalset files found — eval coverage is missing"


@pytest.mark.parametrize("path", EVALSETS)
def test_evalset_is_schema_valid(path):
    with open(path) as f:
        data = json.load(f)
    es = EvalSet.model_validate(data)
    assert es.eval_set_id
    assert es.eval_cases, f"{path} has no eval cases"
    for case in es.eval_cases:
        assert case.conversation, f"{case.eval_id} has no conversation turns"
