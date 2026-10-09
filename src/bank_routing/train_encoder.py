"""Route A: ModernBERT-base with a six-way classification head."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from bank_routing.contract import load as load_contract
from bank_routing.data import load_training_config, read_splits
from bank_routing.paths import models_dir, results_dir
from bank_routing.runlog import run_record


def train(git_commit: str | None = None, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("route A is a GPU job in this harness")
    cfg = load_training_config()["route_a"]
    args_cfg = cfg["training_arguments"]
    contract = load_contract()
    torch.manual_seed(int(args_cfg["seed"]))
    splits = read_splits()
    label2id = contract.label_to_id()
    id2label = {i: label for label, i in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"],
        revision=cfg["revision"],
        num_labels=len(contract.labels),
        id2label=id2label,
        label2id=label2id,
    )

    def encode(frame, with_labels: bool):
        data = {"text": frame["text"].tolist()}
        if with_labels:
            data["labels"] = [label2id[label] for label in frame["label"]]
        ds = Dataset.from_dict(data)

        def tok(batch):
            return tokenizer(batch["text"], truncation=True, max_length=int(cfg["max_length"]))

        return ds.map(tok, batched=True, remove_columns=["text"])

    train_ds = encode(splits["train"], True)
    val_ds = encode(splits["val"], True)
    test_ds = encode(splits["test"], True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return {"macro_f1": float(f1_score(labels, np.argmax(logits, axis=-1), average="macro"))}

    out = models_dir() / "encoder"
    training_args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=float(args_cfg["num_train_epochs"]),
        learning_rate=float(args_cfg["learning_rate"]),
        per_device_train_batch_size=int(args_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(args_cfg["per_device_eval_batch_size"]),
        weight_decay=float(args_cfg["weight_decay"]),
        warmup_steps=float(args_cfg["warmup_steps"]),
        eval_strategy=args_cfg["eval_strategy"],
        save_strategy=args_cfg["save_strategy"],
        load_best_model_at_end=bool(args_cfg["load_best_model_at_end"]),
        metric_for_best_model=args_cfg["metric_for_best_model"],
        save_total_limit=int(args_cfg["save_total_limit"]),
        logging_steps=int(args_cfg["logging_steps"]),
        report_to=args_cfg["report_to"],
        seed=int(args_cfg["seed"]),
        use_cpu=bool(args_cfg["use_cpu"]),
        bf16=bool(args_cfg["bf16"]),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    best = out / "best"
    trainer.save_model(str(best))
    tokenizer.save_pretrained(str(best))
    val_pred = trainer.predict(val_ds)
    test_pred = trainer.predict(test_ds)
    np.save(results_dir() / "encoder_val_logits.npy", val_pred.predictions)
    np.save(results_dir() / "encoder_test_logits.npy", test_pred.predictions)
    report = {
        "model": cfg["model"],
        "revision": cfg["revision"],
        "train_metrics": train_result.metrics,
        "validation_macro_f1": float(val_pred.metrics.get("test_macro_f1", val_pred.metrics.get("eval_macro_f1", float("nan")))),
        "saved_to": str(best),
        "gpu_override": "use_cpu false, bf16 true, compared with the CPU demo",
        "run": run_record("route_a", git=git_commit, hardware=hardware, extra={"contract_version": contract.version, "seed": args_cfg["seed"]}),
    }
    (results_dir() / "04_encoder.json").write_text(json.dumps(report, indent=2, default=str))
    os.environ["BANK_ROUTING_ENCODER"] = str(best)
    return report
