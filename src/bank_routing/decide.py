"""decide(): one route, or needs_human. needs_human is never a trained class."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from bank_routing.contract import Contract
from bank_routing.metrics import action_expected_cost, softmax
from bank_routing.pii import redact

LogitsFn = Callable[[str], np.ndarray]


def _round_probabilities(labels: tuple[str, ...] | list[str], proba: np.ndarray) -> dict[str, float]:
    rounded = {label: round(float(proba[i]), 4) for i, label in enumerate(labels)}
    drift = round(1.0 - sum(rounded.values()), 4)
    top = labels[int(np.argmax(proba))]
    adjusted = round(rounded[top] + drift, 4)
    if 0.0 <= adjusted <= 1.0:
        rounded[top] = adjusted
    return rounded


def _top3(labels: tuple[str, ...] | list[str], proba: np.ndarray) -> list[dict[str, float | str]]:
    order = np.argsort(-proba)[:3]
    return [{"queue": labels[int(i)], "probability": round(float(proba[int(i)]), 4)} for i in order]


def fail_safe(
    *,
    reason: str,
    message_id: str | None,
    threshold: float,
    contract_version: str,
    model_version: str,
) -> dict[str, Any]:
    return {
        "route": "needs_human",
        "label": None,
        "confidence": None,
        "probabilities": None,
        "threshold": threshold,
        "reason": reason,
        "message_id": message_id,
        "contract_version": contract_version,
        "model_version": model_version,
        "top3": [],
    }


def decide(
    text: str | None,
    contract: Contract,
    logits_fn: LogitsFn,
    *,
    message_id: str | None = None,
    model_version: str = "unreleased",
    kill_switch: bool = False,
    timeout_ms: int | None = None,
) -> dict[str, Any]:
    temperature, threshold = contract.require_fitted()
    base = {
        "message_id": message_id,
        "threshold": threshold,
        "contract_version": contract.version,
        "model_version": model_version,
    }
    if text is None or not str(text).strip():
        return fail_safe(reason="invalid_input", **base)
    if kill_switch:
        return fail_safe(reason="kill_switch", **base)
    redacted = redact(str(text))
    try:
        if timeout_ms is not None:
            from concurrent.futures import ThreadPoolExecutor
            from concurrent.futures import TimeoutError as FuturesTimeout

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(logits_fn, redacted)
                try:
                    logits = future.result(timeout=timeout_ms / 1000)
                except FuturesTimeout:
                    return fail_safe(reason="timeout", **base)
        else:
            logits = logits_fn(redacted)
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        if logits.shape != (len(contract.labels),):
            raise ValueError(f"expected {len(contract.labels)} logits, got {logits.shape}")
        proba = softmax(logits.reshape(1, -1), temperature)[0]
    except Exception:
        return fail_safe(reason="model_error", **base)
    k = int(proba.argmax())
    confidence = float(proba[k])
    label = contract.labels[k]
    expected = float(action_expected_cost(proba.reshape(1, -1), list(contract.labels), contract.costs)[0, k])
    route = label if expected < float(contract.costs["human"]) else "needs_human"
    if route != "needs_human" and route != label:
        return fail_safe(reason="model_error", **base)
    probabilities = _round_probabilities(contract.labels, proba)
    if abs(sum(probabilities.values()) - 1.0) > 0.001:
        return fail_safe(reason="model_error", **base)
    return {
        "route": route,
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "threshold": threshold,
        "decision_rule": "expected_cost",
        "expected_cost": round(expected, 4),
        "reason": "auto_routed" if route == label else "expected_cost",
        "message_id": message_id,
        "contract_version": contract.version,
        "model_version": model_version,
        "top3": _top3(contract.labels, proba),
    }


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_bundle_contract(bundle_dir: Path, active: Contract) -> Contract:
    """Refuse a bundle whose label order differs from the active contract."""
    payload = json.loads((Path(bundle_dir) / "contract.json").read_text())
    if tuple(payload["labels"]) != active.labels:
        raise ValueError(
            "refusing to load bundle: label order "
            f"{payload['labels']} differs from the active contract {list(active.labels)}"
        )
    if str(payload["version"]).split(".")[0] != active.version.split(".")[0]:
        raise ValueError("refusing to load bundle: major contract version differs")
    return Contract(
        version=active.version,
        labels=active.labels,
        handoff_route=active.handoff_route,
        queue_scope=active.queue_scope,
        costs=active.costs,
        owners=active.owners,
        intent_mapping=active.intent_mapping,
        temperature=float(payload["temperature"]),
        threshold=float(payload["threshold"]),
        raw=active.raw,
    )
