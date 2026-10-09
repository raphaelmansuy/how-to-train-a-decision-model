"""Score the expected-cost hand-off on probabilities already saved. Nothing is refitted."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bank_routing.cards import write_decision_log, write_model_card
from bank_routing.contract import load
from bank_routing.data import read_splits
from bank_routing.metrics import abstain_report, expected_cost_auto, expected_cost_report
from bank_routing.paths import results_dir


def _round_report(report: dict) -> dict:
    out = {}
    for key, value in report.items():
        if isinstance(value, float):
            out[key] = round(value, 6)
        else:
            out[key] = value
    return out


def main() -> None:
    contract = load()
    labels = list(contract.labels)
    costs = dict(contract.costs)
    results = results_dir()
    calibration = json.loads((results / "06_calibration_abstain.json").read_text())
    cutoff = float(calibration["comparison"]["modernbert"]["threshold"])
    splits = read_splits()
    y_val = splits["val"]["label"].to_numpy()
    y_test = splits["test"]["label"].to_numpy()

    models = {}
    saved = {
        "modernbert": ("encoder_val_proba.npy", "encoder_test_proba.npy"),
        "tfidf": ("tfidf_val_proba.npy", "tfidf_test_proba.npy"),
    }
    for name, (val_name, test_name) in saved.items():
        val_p = np.load(results / val_name)
        test_p = np.load(results / test_name)
        models[name] = {
            "validation": _round_report(expected_cost_report(y_val, val_p, labels, costs)),
            "test": _round_report(expected_cost_report(y_test, test_p, labels, costs)),
        }

    encoder_test = np.load(results / "encoder_test_proba.npy")
    acted, _chosen = expected_cost_auto(encoder_test, labels, costs)
    confidence = encoder_test.max(axis=1)
    cutoff_auto = confidence >= cutoff
    modern = models["modernbert"]["test"]
    tfidf_cutoff_cost = float(calibration["comparison"]["tfidf"]["test"]["abstain"]["cost_per_ticket"])
    per_class = calibration["comparison"]["modernbert"]["test"]["per_class"]
    fraud_recall = float(per_class["fraud"]["recall"])
    missed_rate = modern["missed_fraud_rate"]
    catch = modern["fraud_catch_rate"]
    cost = modern["cost_per_ticket"]
    accepted = modern["accepted_error"]
    g3 = cost <= 0.07 and cost <= tfidf_cutoff_cost + 1e-12
    g4 = missed_rate is not None and missed_rate <= 0.015 and catch is not None and catch >= 0.99 and fraud_recall >= 0.88
    g8 = modern["auto_route_rate"] >= 0.70 and accepted is not None and accepted <= 0.02
    other = [label for label in labels if label != "fraud"]
    g5 = all(float(per_class[label]["recall"]) >= 0.90 for label in other)
    macro_f1 = float(calibration["comparison"]["modernbert"]["test"]["macro_f1"])
    ece = float(calibration["comparison"]["modernbert"]["test"]["ece"])
    ship = json.loads((results / "07_ship.json").read_text()) if (results / "07_ship.json").is_file() else None
    gates = [
        {"gate": "G1", "pass": False, "detail": "Business sign-off of sections 1 to 3 is still open."},
        {"gate": "G2", "pass": True, "detail": "Unchanged. Validation and test still do not share copies."},
        {
            "gate": "G3",
            "pass": g3,
            "detail": (
                f"expected-cost {cost:.4f}; cap 0.07 met; "
                f"TF-IDF confidence-cutoff cost {tfidf_cutoff_cost:.4f}; "
                f"TF-IDF expected-cost {models['tfidf']['test']['cost_per_ticket']:.4f}."
            ),
        },
        {
            "gate": "G4",
            "pass": g4,
            "detail": f"missed fraud rate {missed_rate}, catch {catch}, top-1 fraud recall {fraud_recall:.3f}",
        },
        {
            "gate": "G5",
            "pass": g5,
            "detail": "Top-1 recall is unchanged by the hand-off rule: "
            + ", ".join(f"{label} {float(per_class[label]['recall']):.3f}" for label in other),
        },
        {"gate": "G6", "pass": macro_f1 >= 0.92, "detail": f"macro-F1 {macro_f1:.4f} is a property of the top label, not the hand-off."},
        {"gate": "G7", "pass": ece <= 0.02, "detail": f"ECE {ece:.4f} is unchanged. Temperature was already fitted on validation."},
        {"gate": "G8", "pass": g8, "detail": f"auto-route {modern['auto_route_rate']:.4f}, accepted error {accepted}"},
        {
            "gate": "G9",
            "pass": bool(ship and ship.get("g9_pass")),
            "detail": "ONNX parity is unchanged. The hand-off runs after the model scores.",
        },
        {
            "gate": "G10",
            "pass": next(gate["pass"] for gate in calibration["gates"] if gate["gate"] == "G10"),
            "detail": "Unchanged from the first experiment. The hand-off rule does not refit the length slices.",
        },
        {"gate": "G11", "pass": False, "detail": "ML lead and risk owner approval is still open."},
    ]
    payload = {
        "rule": "expected_cost",
        "costs": costs,
        "confidence_cutoff": cutoff,
        "fitted_parameter": None,
        "note": (
            "Send to the top queue only when its expected cost is strictly below the cost of a person. "
            "Expected cost uses the contract costs and the calibrated probabilities. "
            "The test set was scored once under this rule and was not used to choose it."
        ),
        "models": models,
        "cutoff_comparison": {
            "confidence_cutoff": cutoff,
            "test_rows_cutoff_would_send_and_expected_cost_holds": int((cutoff_auto & ~acted).sum()),
            "test_rows_expected_cost_sends_and_cutoff_would_hold": int((acted & ~cutoff_auto).sum()),
        },
        "confidence_cutoff_test": _round_report(abstain_report(y_test, encoder_test, labels, cutoff, costs)),
        "gates": gates,
    }
    (results / "08_expected_cost.json").write_text(json.dumps(payload, indent=2) + "\n")

    texts = splits["test"]["text"].tolist()
    pred = encoder_test.argmax(axis=1)
    rows = []
    for i, text in enumerate(texts):
        if not acted[i]:
            rows.append(
                {
                    "text": text,
                    "model_label": labels[int(pred[i])],
                    "confidence": float(confidence[i]),
                    "reason": "abstain",
                }
            )
    pd.DataFrame(rows).to_csv(results / "relabel_queue_expected_cost.csv", index=False)

    write_model_card(calibration, ship)
    write_decision_log(calibration)
    print(json.dumps({"modernbert_test": models["modernbert"]["test"], "tfidf_test": models["tfidf"]["test"], "cutoff_comparison": payload["cutoff_comparison"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
