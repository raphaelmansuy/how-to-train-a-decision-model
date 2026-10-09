from pathlib import Path

import pytest

from bank_routing.contract import Contract, load


@pytest.fixture()
def contract() -> Contract:
    loaded = load()
    return Contract(
        version=loaded.version,
        labels=loaded.labels,
        handoff_route=loaded.handoff_route,
        queue_scope=loaded.queue_scope,
        costs=loaded.costs,
        owners=loaded.owners,
        intent_mapping=loaded.intent_mapping,
        temperature=1.24,
        threshold=0.94,
        raw=loaded.raw,
    )


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
