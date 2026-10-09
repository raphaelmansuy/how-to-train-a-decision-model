"""Fit temperature and the hand-off threshold on validation, then score the frozen test set once."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from bank_routing.contract import load as load_contract
from bank_routing.data import load_training_config, read_splits
from bank_routing.metrics import (
    abstain_report,
    classification_summary,
    cost_per_ticket,
    expected_calibration_error,
    fit_temperature,
    nll,
    pick_threshold,
    reliability_table,
    softmax,
    top1_labels,
)
from bank_routing.paths import models_dir, results_dir
from bank_routing.runlog import run_record


def candidate_id(cfg: dict[str, Any], checksums: dict[str, str]) -> str:
    payload = json.dumps({"metric_version": 1, "route_a": cfg["route_a"], "checksums": checksums}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _ids(labels: list[str], label_to_id: dict[str, int]) -> np.ndarray:
    return np.array([label_to_id[label] for label in labels], dtype=np.int64)


def _score_split(
    name: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    labels: list[str],
    costs: dict[str, float],
    *,
    logits: np.ndarray | None,
    temperature: float | None,
    y_idx: np.ndarray,
    threshold: float | None,
) -> dict[str, Any]:
    pred = top1_labels(proba, labels)
    summary = classification_summary(y_true, pred, labels)
    summary["nll"] = nll(proba, y_idx)
    summary["ece"] = expected_calibration_error(proba, y_idx, 15)
    summary["reliability"] = reliability_table(proba, y_idx, 10)
    summary["temperature"] = temperature
    if threshold is not None:
        summary["abstain"] = abstain_report(y_true, proba, labels, threshold, costs)
    summary["split"] = name
    return summary


def _length_slices(texts: list[str], y_true: np.ndarray, proba: np.ndarray, labels: list[str], costs: dict[str, float], threshold: float) -> list[dict[str, Any]]:
    buckets = {"short": [], "medium": [], "long": []}
    for i, text in enumerate(texts):
        n_words = len(str(text).split())
        key = "short" if n_words <= 8 else "medium" if n_words <= 20 else "long"
        buckets[key].append(i)
    overall_pred = top1_labels(proba, labels)
    overall = classification_summary(y_true, overall_pred, labels)
    rows = []
    for name, idxs in buckets.items():
        if not idxs:
            continue
        idx = np.array(idxs)
        pred = top1_labels(proba[idx], labels)
        summary = classification_summary(y_true[idx], pred, labels)
        abstain = abstain_report(y_true[idx], proba[idx], labels, threshold, costs)
        fraud_recall = summary["per_class"]["fraud"]["recall"] if summary["per_class"]["fraud"]["support"] else None
        rows.append(
            {
                "slice": name,
                "n": summary["n"],
                "macro_f1": summary["macro_f1"],
                "fraud_recall": fraud_recall,
                "handoff_rate": 1.0 - abstain["auto_route_rate"],
                "cost_per_ticket": abstain["cost_per_ticket"],
                "macro_f1_gap": overall["macro_f1"] - summary["macro_f1"],
                "eligible_for_gate": summary["n"] >= 200,
            }
        )
    return rows


def _gate(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"gate": name, "pass": bool(passed), "detail": detail}


def _gates(candidate: dict[str, Any], tfidf_cost: float, slices: list[dict[str, Any]], data_report: dict[str, Any]) -> list[dict[str, Any]]:
    abstain = candidate["test"]["abstain"]
    per_class = candidate["test"]["per_class"]
    missed_rate = abstain["missed_fraud_rate"]
    catch = abstain["fraud_catch_rate"]
    accepted = abstain["accepted_error"]
    other = [label for label in per_class if label != "fraud"]
    other_ok = all(per_class[label]["recall"] >= 0.90 for label in other)
    slice_ok = True
    slice_notes = []
    for row in slices:
        if not row["eligible_for_gate"]:
            slice_notes.append(f"{row['slice']} n={row['n']} below 200, not gated")
            continue
        if row["macro_f1_gap"] > 0.05:
            slice_ok = False
            slice_notes.append(f"{row['slice']} macro-F1 is more than 5 points below overall")
    leaks = data_report["val_test_exact_overlaps"] == 0 and data_report["val_test_near_duplicates"] == 0
    cost = abstain["cost_per_ticket"]
    beats_tfidf = cost <= tfidf_cost + 1e-12
    under_cap = cost <= 0.07
    g4 = missed_rate is not None and missed_rate <= 0.015 and catch is not None and catch >= 0.99 and per_class["fraud"]["recall"] >= 0.88
    g8 = abstain["auto_route_rate"] >= 0.70 and accepted is not None and accepted <= 0.02
    return [
        _gate("G1", False, "77-intent mapping loads. Business sign-off of sections 1 to 3 is still open."),
        _gate("G2", leaks, "exact and near-duplicate overlaps between validation and test are zero" if leaks else "leakage remains"),
        _gate(
            "G3",
            under_cap and beats_tfidf,
            f"cost {cost:.4f}; cap 0.07 {'met' if under_cap else 'missed'}; TF-IDF cost {tfidf_cost:.4f}. No business justification is invented when the candidate is more expensive.",
        ),
        _gate(
            "G4",
            g4,
            f"missed fraud rate {missed_rate}, catch {catch}, top-1 fraud recall {per_class['fraud']['recall']:.3f}",
        ),
        _gate("G5", other_ok, "top-1 recall by queue: " + ", ".join(f"{label} {per_class[label]['recall']:.3f}" for label in other)),
        _gate("G6", candidate["test"]["macro_f1"] >= 0.92, f"macro-F1 {candidate['test']['macro_f1']:.4f} on n={candidate['test']['n']}"),
        _gate("G7", candidate["test"]["ece"] <= 0.02, f"ECE {candidate['test']['ece']:.4f} after temperature"),
        _gate(
            "G8",
            g8,
            f"auto-route {abstain['auto_route_rate']:.3f}, accepted error {accepted}",
        ),
        _gate("G9", False, "ONNX parity and latency are filled by the ship step"),
        _gate("G10", slice_ok, "; ".join(slice_notes) or "message-length slices within 5 points"),
        _gate("G11", False, "data card and model card are written; ML lead and risk owner approval is still open"),
    ]


def _model_block(
    name: str,
    y_val: np.ndarray,
    y_test: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    labels: list[str],
    costs: dict[str, float],
    cfg: dict[str, Any],
    *,
    val_logits: np.ndarray | None = None,
    test_logits: np.ndarray | None = None,
    val_proba: np.ndarray | None = None,
    test_proba: np.ndarray | None = None,
    subset_note: str | None = None,
) -> dict[str, Any]:
    label_to_id = {label: i for i, label in enumerate(labels)}
    if val_logits is not None:
        temperature = fit_temperature(val_logits, val_idx, cfg["calibration"]["grid"], cfg["calibration"]["refine"])
        val_p = softmax(val_logits, temperature)
        test_p = softmax(test_logits, temperature)
    else:
        temperature = None
        val_p = val_proba
        test_p = test_proba
    assert val_p is not None and test_p is not None
    chosen = pick_threshold(
        y_val,
        val_p,
        labels,
        costs,
        start=float(cfg["threshold"]["start"]),
        stop=float(cfg["threshold"]["stop"]),
        step=float(cfg["threshold"]["step"]),
    )
    threshold = float(chosen["threshold"])
    block = {
        "name": name,
        "temperature": temperature,
        "threshold": threshold,
        "validation_choice": chosen,
        "validation": _score_split("validation", y_val, val_p, labels, costs, logits=val_logits, temperature=temperature, y_idx=val_idx, threshold=threshold),
        "test": _score_split("test", y_test, test_p, labels, costs, logits=test_logits, temperature=temperature, y_idx=test_idx, threshold=threshold),
        "subset_note": subset_note,
        "sensitivity": {},
    }
    for missed in (2.0, 20.0):
        alt = dict(costs)
        alt["missed_fraud"] = missed
        alt_choice = pick_threshold(y_val, val_p, labels, alt, start=0.0, stop=1.0, step=0.01)
        block["sensitivity"][f"missed_fraud_{missed}"] = {
            "threshold": alt_choice["threshold"],
            "validation_cost": alt_choice["cost_per_ticket"],
            "test": abstain_report(y_test, test_p, labels, float(alt_choice["threshold"]), alt),
        }
    block["test_proba"] = test_p
    block["val_proba"] = val_p
    return block


def calibrate(git_commit: str | None = None, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_training_config()
    contract = load_contract()
    splits = read_splits()
    labels = list(contract.labels)
    costs = contract.costs
    data_report = json.loads((results_dir() / "01_data_report.json").read_text())
    cid = candidate_id(cfg, data_report["checksums"])
    stamp_path = results_dir() / "test_scored.json"
    report_path = results_dir() / "06_calibration_abstain.json"
    if stamp_path.exists() and os.environ.get("ALLOW_RESCORE") != "1":
        stamp = json.loads(stamp_path.read_text())
        if stamp.get("candidate_id") == cid and report_path.exists():
            return json.loads(report_path.read_text())
    if stamp_path.exists() and os.environ.get("ALLOW_RESCORE") == "1":
        rescore_warning = "ALLOW_RESCORE=1 overwrote a previous test scoring. Treat this as a leak if the threshold or model was changed after seeing test results."
    else:
        rescore_warning = None

    y_val = splits["val"]["label"].to_numpy()
    y_test = splits["test"]["label"].to_numpy()
    y_val_idx = _ids(list(y_val), contract.label_to_id())
    y_test_idx = _ids(list(y_test), contract.label_to_id())
    results = results_dir()

    tfidf = _model_block(
        "tfidf",
        y_val,
        y_test,
        y_val_idx,
        y_test_idx,
        labels,
        costs,
        cfg,
        val_proba=np.load(results / "tfidf_val_proba.npy"),
        test_proba=np.load(results / "tfidf_test_proba.npy"),
    )
    encoder = _model_block(
        "modernbert",
        y_val,
        y_test,
        y_val_idx,
        y_test_idx,
        labels,
        costs,
        cfg,
        val_logits=np.load(results / "encoder_val_logits.npy"),
        test_logits=np.load(results / "encoder_test_logits.npy"),
    )
    zero = _model_block(
        "zero_shot",
        y_val,
        y_test,
        y_val_idx,
        y_test_idx,
        labels,
        costs,
        cfg,
        val_logits=np.load(results / "zeroshot_val_logits.npy"),
        test_logits=np.load(results / "zeroshot_test_logits.npy"),
    )
    val_idx = np.load(results / "lora_val_idx.npy")
    test_idx = np.load(results / "lora_test_idx.npy")
    lora = _model_block(
        "lora",
        y_val[val_idx],
        y_test[test_idx],
        y_val_idx[val_idx],
        y_test_idx[test_idx],
        labels,
        costs,
        cfg,
        val_logits=np.load(results / "lora_val_logits.npy"),
        test_logits=np.load(results / "lora_test_logits.npy"),
        subset_note=f"validation subset n={len(val_idx)}, test subset n={len(test_idx)}",
    )
    setfit_proba = np.load(results / "setfit_test_proba.npy")
    setfit_pred = top1_labels(setfit_proba, labels)
    setfit = classification_summary(y_test, setfit_pred, labels)
    setfit["nll"] = nll(setfit_proba, y_test_idx)
    setfit["ece"] = expected_calibration_error(setfit_proba, y_test_idx, 15)
    setfit["note"] = "not part of the threshold comparison"

    people_proba = np.zeros((len(y_test), len(labels)))
    people = {
        "name": "everything_to_people",
        "test": {
            "n": int(len(y_test)),
            "abstain": {
                "auto_route_rate": 0.0,
                "accepted_error": None,
                "missed_fraud": 0,
                "cost_per_ticket": cost_per_ticket(y_test, np.full_like(people_proba, 1.0 / len(labels)), labels, 1.01, costs),
            },
        },
    }
    slices = _length_slices(splits["test"]["text"].tolist(), y_test, encoder["test_proba"], labels, costs, float(encoder["threshold"]))
    gates = _gates(encoder, tfidf["test"]["abstain"]["cost_per_ticket"], slices, data_report)
    scored_at = datetime.now(timezone.utc).astimezone().isoformat()
    model_version = f"router-modernbert-{scored_at[:10]}-{cid}"
    bundle = models_dir() / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    bundle_contract = {
        "version": contract.version,
        "labels": list(contract.labels),
        "temperature": encoder["temperature"],
        "threshold": encoder["threshold"],
        "costs": costs,
        "owners": contract.owners,
        "model_version": model_version,
        "candidate_id": cid,
    }
    (bundle / "contract.json").write_text(json.dumps(bundle_contract, indent=2))
    manifest = {
        "model_version": model_version,
        "contract_version": contract.version,
        "candidate_id": cid,
        "temperature": encoder["temperature"],
        "threshold": encoder["threshold"],
        "labels": list(contract.labels),
        "data_checksums": data_report["checksums"],
        "test_scored_at": scored_at,
        "release_candidate": "modernbert",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
    stamp_path.write_text(json.dumps({"candidate_id": cid, "scored_at": scored_at, "model_version": model_version, "n_test": int(len(y_test))}, indent=2))

    def public(block: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in block.items() if k not in ("test_proba", "val_proba")}

    report = {
        "candidate_id": cid,
        "model_version": model_version,
        "test_scored_at": scored_at,
        "n_test": int(len(y_test)),
        "n_validation": int(len(y_val)),
        "rescore_warning": rescore_warning,
        "comparison": {
            "modernbert": public(encoder),
            "tfidf": public(tfidf),
            "everything_to_people": people,
            "zero_shot": public(zero),
            "lora": public(lora),
        },
        "setfit": setfit,
        "length_slices": slices,
        "gates": gates,
        "run": run_record("calibrate", git=git_commit, hardware=hardware, extra={"contract_version": contract.version}),
    }
    report_path.write_text(json.dumps(report, indent=2))
    np.save(results / "encoder_test_proba.npy", encoder["test_proba"])
    np.save(results / "encoder_val_proba.npy", encoder["val_proba"])
    from bank_routing.cards import write_decision_log, write_model_card

    write_model_card(report)
    write_decision_log(report)
    return report
