"""ONNX export, parity, latency, PSI, and the relabel queue for the Route A bundle."""

from __future__ import annotations

import json
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd

from bank_routing.contract import load as load_contract
from bank_routing.data import load_training_config, read_splits
from bank_routing.metrics import expected_cost_auto, top1_labels
from bank_routing.monitoring import label_shares, population_stability_index
from bank_routing.paths import models_dir, results_dir
from bank_routing.runlog import run_record


def onnx_logits(bundle_dir):
    import onnxruntime as ort
    from transformers import AutoTokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(bundle_dir / "router.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(bundle_dir / "tokenizer")

    def logits(text: str) -> np.ndarray:
        encoded = tokenizer([text], return_tensors="np", truncation=True, max_length=64)
        return session.run(["logits"], {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]})[0][0]

    return logits


def ship(git_commit: str | None = None, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = load_training_config()
    contract = load_contract()
    bundle = models_dir() / "bundle"
    source = models_dir() / "encoder" / "best"
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)
    model.eval()
    model.to("cpu")
    example = tokenizer("I lost my card on the bus", return_tensors="pt", truncation=True, max_length=64)
    onnx_path = bundle / "router.onnx"
    torch.onnx.export(
        model,
        (example["input_ids"], example["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
        dynamo=True,
    )
    tokenizer.save_pretrained(bundle / "tokenizer")
    splits = read_splits()
    test = splits["test"].head(int(cfg["serving"]["parity"]["messages"]))
    pytorch_rows = []
    onnx_fn = onnx_logits(bundle)
    onnx_rows = []
    with torch.no_grad():
        for text in test["text"]:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            pytorch_rows.append(model(**encoded).logits[0].numpy())
            onnx_rows.append(np.asarray(onnx_fn(text), dtype=np.float64))
    pytorch = np.stack(pytorch_rows)
    exported = np.stack(onnx_rows)
    max_abs = float(np.max(np.abs(pytorch - exported)))
    agreement = float(np.mean(pytorch.argmax(1) == exported.argmax(1)))
    parity_ok = max_abs <= float(cfg["serving"]["parity"]["max_abs_logit_diff"]) and agreement >= float(cfg["serving"]["parity"]["argmax_agreement"])

    times_ms = []
    for text in test["text"]:
        started = time.perf_counter()
        onnx_fn(text)
        times_ms.append((time.perf_counter() - started) * 1000)
    arr = np.array(times_ms)
    latency = {"n": int(len(arr)), "p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)), "threads": 4}

    report = json.loads((results_dir() / "06_calibration_abstain.json").read_text())
    val_pred = top1_labels(np.load(results_dir() / "encoder_val_proba.npy"), list(contract.labels))
    test_pred = top1_labels(np.load(results_dir() / "encoder_test_proba.npy"), list(contract.labels))
    psi = population_stability_index(label_shares(val_pred, list(contract.labels)), label_shares(test_pred, list(contract.labels)))
    fraud_or_cards = splits["test"]["label"].isin(["fraud", "cards"]).to_numpy()
    shifted = [label for label, keep in zip(test_pred, fraud_or_cards) if keep]
    psi_shift = population_stability_index(label_shares(val_pred, list(contract.labels)), label_shares(shifted, list(contract.labels)))

    proba = np.load(results_dir() / "encoder_test_proba.npy")
    acted, chosen = expected_cost_auto(proba, list(contract.labels), contract.costs)
    conf = proba.max(1)
    pred = top1_labels(proba, list(contract.labels))
    rng = np.random.default_rng(0)
    rows = []
    for i, text in enumerate(splits["test"]["text"]):
        if not acted[i]:
            rows.append(
                {
                    "text": text,
                    "model_label": pred[i],
                    "confidence": float(conf[i]),
                    "expected_cost": float(chosen[i]),
                    "reason": "abstain",
                }
            )
        elif rng.random() < 0.02:
            rows.append(
                {
                    "text": text,
                    "model_label": pred[i],
                    "confidence": float(conf[i]),
                    "expected_cost": float(chosen[i]),
                    "reason": "random_audit",
                }
            )
    pd.DataFrame(rows).to_csv(results_dir() / "relabel_queue.csv", index=False)

    g9_pass = parity_ok and latency["p50_ms"] <= 30 and latency["p95_ms"] <= 50
    for gate in report["gates"]:
        if gate["gate"] == "G9":
            gate["pass"] = g9_pass
            gate["detail"] = f"parity max abs {max_abs:.3e}, agreement {agreement:.3f}; p50 {latency['p50_ms']:.1f} ms, p95 {latency['p95_ms']:.1f} ms"
    (results_dir() / "06_calibration_abstain.json").write_text(json.dumps(report, indent=2))
    ship_report = {
        "parity": {"n": int(len(test)), "max_abs_logit_diff": max_abs, "argmax_agreement": agreement, "pass": parity_ok},
        "latency": latency,
        "psi_validation_vs_test": psi,
        "psi_simulated_fraud_cards_shift": psi_shift,
        "relabel_rows": int(len(rows)),
        "g9_pass": g9_pass,
        "run": run_record("ship", git=git_commit, hardware=hardware, extra={"contract_version": contract.version}),
    }
    (results_dir() / "07_ship.json").write_text(json.dumps(ship_report, indent=2))
    from bank_routing.cards import write_model_card

    write_model_card(report, ship_report)
    if bundle.joinpath("tokenizer").exists():
        pass
    shutil.copy(source / "config.json", bundle / "encoder_config.json")
    return ship_report
