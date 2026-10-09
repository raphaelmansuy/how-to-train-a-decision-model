"""SetFit few-shot. Test probabilities only; not part of the threshold comparison."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from bank_routing.contract import load as load_contract
from bank_routing.data import load_training_config, read_splits
from bank_routing.paths import models_dir, results_dir
from bank_routing.runlog import run_record


def train(git_commit: str | None = None, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

    cfg = load_training_config()["setfit"]
    contract = load_contract()
    splits = read_splits()
    label2id = contract.label_to_id()
    full = Dataset.from_dict(
        {
            "text": splits["train"]["text"].tolist(),
            "label": [label2id[label] for label in splits["train"]["label"]],
        }
    )
    sampled = sample_dataset(full, label_column="label", num_samples=int(cfg["num_samples_per_label"]), seed=int(cfg["training_arguments"]["seed"]))
    model = SetFitModel.from_pretrained(cfg["model"], revision=cfg["revision"], labels=list(range(len(contract.labels))))
    args_cfg = cfg["training_arguments"]
    args = TrainingArguments(
        batch_size=int(args_cfg["batch_size"]),
        num_epochs=int(args_cfg["num_epochs"]),
        num_iterations=int(args_cfg["num_iterations"]),
        seed=int(args_cfg["seed"]),
        output_dir=str(models_dir() / "setfit"),
    )
    trainer = Trainer(model=model, args=args, train_dataset=sampled)
    trainer.train()
    model.save_pretrained(str(models_dir() / "setfit" / "best"))
    proba = np.asarray(model.predict_proba(splits["test"]["text"].tolist()), dtype=np.float64)
    np.save(results_dir() / "setfit_test_proba.npy", proba)
    report = {
        "model": cfg["model"],
        "revision": cfg["revision"],
        "train_rows": int(len(sampled)),
        "test_rows_inferred": int(len(splits["test"])),
        "note": "Test probabilities only. SetFit is not in the threshold comparison.",
        "run": run_record("setfit", git=git_commit, hardware=hardware, extra={"contract_version": contract.version}),
    }
    (results_dir() / "04b_setfit.json").write_text(json.dumps(report, indent=2))
    return report
