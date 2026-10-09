"""Population stability index on the six-queue mix."""

from __future__ import annotations

import numpy as np


def population_stability_index(reference: np.ndarray, current: np.ndarray, eps: float = 1e-4) -> float:
    ref = np.asarray(reference, dtype=np.float64) + eps
    cur = np.asarray(current, dtype=np.float64) + eps
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def label_shares(labels: list[str], order: list[str]) -> np.ndarray:
    counts = np.array([(np.array(labels) == name).sum() for name in order], dtype=np.float64)
    if counts.sum() == 0:
        return np.ones(len(order)) / len(order)
    return counts / counts.sum()
