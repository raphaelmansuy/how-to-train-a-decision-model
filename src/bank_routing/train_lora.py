"""Route B: LoRA on Qwen3-0.6B. Inference is a logit read, never free generation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from bank_routing.contract import load as load_contract
from bank_routing.data import load_training_config, read_splits
from bank_routing.llm import logit_read, routing_prompt
from bank_routing.paths import models_dir, results_dir
from bank_routing.runlog import run_record


def _take(frame, n: int, seed: int):
    indices = np.arange(len(frame))
    chosen, _ = train_test_split(indices, train_size=n, stratify=frame["label"], random_state=seed)
    return np.sort(np.asarray(chosen, dtype=np.int64))


def train(git_commit: str | None = None, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("route B is a GPU job in this harness")
    cfg = load_training_config()["route_b"]
    if cfg.get("cpu_loss_override"):
        raise RuntimeError("CPU loss override is not used on the GPU harness")
    contract = load_contract()
    torch.manual_seed(int(cfg["sft_config"]["seed"]))
    splits = read_splits()
    train_idx = _take(splits["train"], int(cfg["train_examples"]), int(cfg["train_random_state"]))
    subset = splits["train"].iloc[train_idx]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=torch.bfloat16)
    dataset = Dataset.from_dict(
        {
            "prompt": [routing_prompt(text, contract) for text in subset["text"]],
            "completion": [" " + label for label in subset["label"]],
        }
    )
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
    )
    sft = cfg["sft_config"]
    out = models_dir() / "lora"
    sft_args = SFTConfig(
        output_dir=str(out),
        max_steps=int(sft["max_steps"]),
        per_device_train_batch_size=int(sft["per_device_train_batch_size"]),
        learning_rate=float(sft["learning_rate"]),
        lr_scheduler_type=sft["lr_scheduler_type"],
        warmup_steps=float(sft["warmup_steps"]),
        logging_steps=int(sft["logging_steps"]),
        save_strategy=sft["save_strategy"],
        bf16=bool(sft["bf16"]),
        gradient_checkpointing=bool(sft["gradient_checkpointing"]),
        max_length=int(sft["max_length"]),
        report_to=sft["report_to"],
        use_cpu=bool(sft["use_cpu"]),
        seed=int(sft["seed"]),
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    adapter = out / "adapter"
    trainer.model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    subset_cfg = cfg["eval_subset"]
    val_idx = _take(splits["val"], int(subset_cfg["val"]), int(subset_cfg["random_state"]))
    test_idx = _take(splits["test"], int(subset_cfg["test"]), int(subset_cfg["random_state"]))
    np.save(results_dir() / "lora_val_idx.npy", val_idx)
    np.save(results_dir() / "lora_test_idx.npy", test_idx)
    trained = trainer.model
    if torch.cuda.is_available():
        trained = trained.to("cuda")
    batch = int(cfg["logit_read"]["batch_size"])
    val_logits = logit_read(trained, tokenizer, splits["val"].iloc[val_idx]["text"].tolist(), contract, batch_size=batch)
    test_logits = logit_read(trained, tokenizer, splits["test"].iloc[test_idx]["text"].tolist(), contract, batch_size=batch)
    np.save(results_dir() / "lora_val_logits.npy", val_logits)
    np.save(results_dir() / "lora_test_logits.npy", test_logits)
    counts = {label: int((subset["label"] == label).sum()) for label in contract.labels}
    report = {
        "model": cfg["model"],
        "revision": cfg["revision"],
        "train_rows": int(len(subset)),
        "train_rows_by_queue": counts,
        "val_subset": int(len(val_idx)),
        "test_subset": int(len(test_idx)),
        "train_metrics": train_result.metrics,
        "gpu_override": "bf16 true, default TRL loss, no CPUSFTTrainer",
        "run": run_record("route_b", git=git_commit, hardware=hardware, extra={"contract_version": contract.version}),
    }
    (results_dir() / "05_lora.json").write_text(json.dumps(report, indent=2, default=str))
    return report
