"""Scores, calibration, cost, and the hand-off threshold. Test labels are not used here to choose T or the threshold."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def nll(proba: np.ndarray, y_idx: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-15, 1.0)
    return float(-np.log(p).mean())


def expected_calibration_error(proba: np.ndarray, y_idx: np.ndarray, n_bins: int = 15) -> float:
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_idx
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_idx)
    if total == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        acc = float(correct[mask].mean())
        mean_conf = float(conf[mask].mean())
        ece += (count / total) * abs(acc - mean_conf)
    return float(ece)


def reliability_table(proba: np.ndarray, y_idx: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_idx
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == 0:
            mask = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        rows.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "n": count,
                "mean_confidence": None if count == 0 else float(conf[mask].mean()),
                "accuracy": None if count == 0 else float(correct[mask].mean()),
            }
        )
    return rows


def classification_summary(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray, labels: list[str]) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in labels
    }
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "labels": list(labels),
    }


def _is_handoff(confidence: float, threshold: float) -> bool:
    return confidence < threshold


def action_expected_cost(
    proba: np.ndarray,
    labels: list[str],
    costs: dict[str, float],
    fraud_label: str = "fraud",
) -> np.ndarray:
    """Expected cost of sending each row to each queue.

    A correct send costs 0. Any other send costs ``wrong_queue``, except a real
    fraud message sent to some other queue, which costs ``missed_fraud`` instead.
    """
    proba = np.asarray(proba, dtype=np.float64)
    fraud_i = list(labels).index(fraud_label)
    wrong = float(costs["wrong_queue"])
    extra = float(costs["missed_fraud"]) - wrong
    expected = wrong * (1.0 - proba) + extra * proba[:, fraud_i : fraud_i + 1]
    expected[:, fraud_i] = wrong * (1.0 - proba[:, fraud_i])
    return expected


def expected_cost_auto(
    proba: np.ndarray,
    labels: list[str],
    costs: dict[str, float],
    fraud_label: str = "fraud",
) -> tuple[np.ndarray, np.ndarray]:
    """Send to the top queue only when that action is strictly cheaper than a person.

    Under these costs the minimum-cost queue is the top queue whenever it beats
    a person: a false send costs the same in every queue, and a missed fraud
    costs more, so it never pays to act on a queue that is not first.
    """
    expected = action_expected_cost(proba, labels, costs, fraud_label)
    top = np.asarray(proba, dtype=np.float64).argmax(axis=1)
    chosen = expected[np.arange(len(top)), top]
    return chosen < float(costs["human"]), chosen


def routing_counts(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    threshold: float,
    fraud_label: str = "fraud",
    auto: np.ndarray | None = None,
) -> dict[str, int]:
    pred_idx = proba.argmax(axis=1)
    if auto is None:
        conf = proba.max(axis=1)
        acted = np.array([not _is_handoff(float(conf[i]), threshold) for i in range(len(y_true))])
    else:
        acted = np.asarray(auto, dtype=bool)
    handoff = wrong = missed = correct = 0
    for i, truth in enumerate(y_true):
        pred = labels[int(pred_idx[i])]
        if not acted[i]:
            handoff += 1
        elif pred != truth:
            if truth == fraud_label:
                missed += 1
            else:
                wrong += 1
        else:
            correct += 1
    return {"handoff": handoff, "wrong": wrong, "missed_fraud": missed, "correct": correct, "n": int(len(y_true))}


def cost_from_counts(counts: dict[str, int], costs: dict[str, float]) -> float:
    total = (
        costs["human"] * counts["handoff"]
        + costs["wrong_queue"] * counts["wrong"]
        + costs["missed_fraud"] * counts["missed_fraud"]
    )
    return float(total / counts["n"]) if counts["n"] else 0.0


def cost_per_ticket(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    threshold: float,
    costs: dict[str, float],
    fraud_label: str = "fraud",
) -> float:
    return cost_from_counts(routing_counts(y_true, proba, labels, threshold, fraud_label), costs)


def abstain_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    threshold: float,
    costs: dict[str, float],
    fraud_label: str = "fraud",
    auto: np.ndarray | None = None,
) -> dict[str, Any]:
    pred_idx = proba.argmax(axis=1)
    acted = proba.max(axis=1) >= threshold if auto is None else np.asarray(auto, dtype=bool)
    n = len(y_true)
    n_auto = int(acted.sum())
    wrong_auto = 0
    missed = 0
    fraud_total = 0
    fraud_caught = 0
    for i, truth in enumerate(y_true):
        pred = labels[int(pred_idx[i])]
        is_fraud = truth == fraud_label
        if is_fraud:
            fraud_total += 1
            if (not acted[i]) or pred == fraud_label:
                fraud_caught += 1
        if acted[i] and pred != truth:
            wrong_auto += 1
            if is_fraud:
                missed += 1
    counts = routing_counts(y_true, proba, labels, threshold, fraud_label, auto=acted)
    return {
        "n": int(n),
        "rule": "confidence_threshold" if auto is None else "expected_cost",
        "threshold": None if auto is not None else float(threshold),
        "auto_route_rate": float(n_auto / n) if n else 0.0,
        "accepted_error": None if n_auto == 0 else float(wrong_auto / n_auto),
        "missed_fraud": int(missed),
        "fraud_total": int(fraud_total),
        "missed_fraud_rate": None if fraud_total == 0 else float(missed / fraud_total),
        "fraud_catch_rate": None if fraud_total == 0 else float(fraud_caught / fraud_total),
        "cost_per_ticket": cost_from_counts(counts, costs),
        "counts": counts,
    }


def expected_cost_report(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    costs: dict[str, float],
    fraud_label: str = "fraud",
) -> dict[str, Any]:
    acted, chosen = expected_cost_auto(proba, labels, costs, fraud_label)
    expected = action_expected_cost(proba, labels, costs, fraud_label)
    top = np.asarray(proba).argmax(axis=1)
    best = expected.argmin(axis=1)
    human = float(costs["human"])
    report = abstain_report(y_true, proba, labels, 0.0, costs, fraud_label, auto=acted)
    report["human_cost"] = human
    report["mean_expected_cost_of_top"] = float(chosen.mean()) if len(chosen) else 0.0
    report["unconstrained_non_top_sends"] = int(((best != top) & (expected.min(axis=1) < human)).sum())
    return report


def fit_temperature(logits: np.ndarray, y_idx: np.ndarray, grid: dict[str, float], refine: dict[str, float]) -> float:
    coarse = np.linspace(float(grid["start"]), float(grid["stop"]), int(grid["num"]))

    def loss(t: float) -> float:
        return nll(softmax(logits, t), y_idx)

    best = float(min(coarse, key=loss))
    lo = max(float(refine["floor"]), best - float(refine["half_width"]))
    hi = best + float(refine["half_width"])
    fine = np.linspace(lo, hi, int(refine["num"]))
    return float(min(fine, key=loss))


def pick_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    costs: dict[str, float],
    start: float = 0.0,
    stop: float = 1.0,
    step: float = 0.01,
    fraud_label: str = "fraud",
) -> dict[str, Any]:
    grid = np.round(np.arange(start, stop + step / 2, step), 2)
    best: dict[str, Any] | None = None
    for threshold in grid:
        report = abstain_report(y_true, proba, labels, float(threshold), costs, fraud_label)
        if best is None:
            best = report
            continue
        cheaper = report["cost_per_ticket"] < best["cost_per_ticket"] - 1e-12
        tied = abs(report["cost_per_ticket"] - best["cost_per_ticket"]) <= 1e-12
        if cheaper or (tied and report["auto_route_rate"] > best["auto_route_rate"]):
            best = report
    assert best is not None
    return best


def top1_labels(proba: np.ndarray, labels: list[str]) -> list[str]:
    return [labels[int(i)] for i in proba.argmax(axis=1)]
