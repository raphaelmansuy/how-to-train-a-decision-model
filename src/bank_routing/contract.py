"""Decision contract. Label order is read from the contract file, never hard-coded."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bank_routing.paths import config_dir

REQUIRED_LABELS = ("cards", "transfers", "cash", "fees", "account", "fraud")


@dataclass(frozen=True)
class Contract:
    version: str
    labels: tuple[str, ...]
    handoff_route: str
    queue_scope: dict[str, str]
    costs: dict[str, float]
    owners: dict[str, Any]
    intent_mapping: dict[str, str]
    temperature: float | None
    threshold: float | None
    raw: dict[str, Any]

    def label_to_id(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}

    def id_to_label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))

    def require_fitted(self) -> tuple[float, float]:
        if self.temperature is None or self.threshold is None:
            raise ValueError("temperature and threshold are not fitted on this contract")
        return float(self.temperature), float(self.threshold)


def load(path: Path | None = None) -> Contract:
    path = path or (config_dir() / "contract.yaml")
    raw = yaml.safe_load(path.read_text())
    labels = tuple(raw["labels"])
    if labels != REQUIRED_LABELS:
        raise ValueError(f"contract 1.x label order must be {REQUIRED_LABELS}, got {labels}")
    mapping = dict(raw["intent_mapping"])
    if len(mapping) != 77:
        raise ValueError(f"expected 77 intents, found {len(mapping)}")
    unknown = sorted({q for q in mapping.values() if q not in labels})
    if unknown:
        raise ValueError(f"intents map to unknown queues: {unknown}")
    if len(set(mapping)) != 77:
        raise ValueError("duplicate intent names in the contract")
    costs = raw["costs"]
    for key in ("human", "wrong_queue", "missed_fraud"):
        if key not in costs:
            raise ValueError(f"missing cost {key}")
    owners = raw["owners"]
    for key in (
        "version",
        "labels",
        "label_order",
        "handoff_route",
        "queue_scope",
        "intent_mapping",
        "costs",
        "temperature",
        "threshold",
    ):
        if key not in owners or "role" not in owners[key]:
            raise ValueError(f"contract value {key} has no owner role")
    temperature = raw.get("temperature", {}).get("value")
    threshold = raw.get("threshold", {}).get("value")
    return Contract(
        version=str(raw["version"]),
        labels=labels,
        handoff_route=str(raw["handoff_route"]),
        queue_scope={label: str(raw["queue_scope"][label]) for label in labels},
        costs={k: float(costs[k]) for k in ("human", "wrong_queue", "missed_fraud")},
        owners=owners,
        intent_mapping=mapping,
        temperature=None if temperature is None else float(temperature),
        threshold=None if threshold is None else float(threshold),
        raw=raw,
    )


def queue_for_intent(contract: Contract, intent: str) -> str:
    if intent not in contract.intent_mapping:
        raise ValueError(f"unmapped intent: {intent}")
    return contract.intent_mapping[intent]
