"""TF-IDF logistic regression and the zero-shot logit-read baseline."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

from bank_routing.contract import load as load_contract
from bank_routing.data import load_training_config, read_splits
from bank_routing.llm import free_generation_sample, logit_read
from bank_routing.paths import models_dir, results_dir
from bank_routing.runlog import run_record


def _align(proba: np.ndarray, classes: list[str], labels: tuple[str, ...]) -> np.ndarray:
    index = {name: i for i, name in enumerate(classes)}
    return np.column_stack([proba[:, index[label]] for label in labels])


def train_tfidf(git_commit: str | None = None, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_training_config()["baseline_tfidf"]
    contract = load_contract()
    splits = read_splits()
    word = cfg["word"]
    char = cfg["char"]
    logistic = cfg["logistic_regression"]
    pipeline = make_pipeline(
        make_union(
            TfidfVectorizer(ngram_range=tuple(word["ngram_range"]), sublinear_tf=bool(word["sublinear_tf"]), min_df=int(word["min_df"])),
            TfidfVectorizer(
                analyzer=char["analyzer"],
                ngram_range=tuple(char["ngram_range"]),
                sublinear_tf=bool(char["sublinear_tf"]),
                min_df=int(char["min_df"]),
            ),
        ),
        LogisticRegression(
            C=float(logistic["C"]),
            max_iter=int(logistic["max_iter"]),
            class_weight=logistic["class_weight"],
        ),
    )
    started = time.perf_counter()
    pipeline.fit(splits["train"]["text"], splits["train"]["label"])
    fit_s = time.perf_counter() - started
    labels = list(contract.labels)
    val_proba = _align(pipeline.predict_proba(splits["val"]["text"]), list(pipeline.classes_), contract.labels)
    test_proba = _align(pipeline.predict_proba(splits["test"]["text"]), list(pipeline.classes_), contract.labels)
    np.save(results_dir() / "tfidf_val_proba.npy", val_proba)
    np.save(results_dir() / "tfidf_test_proba.npy", test_proba)
    import joblib

    joblib.dump(pipeline, models_dir() / "tfidf.joblib")
    report = {
        "model": "tfidf_logistic_regression",
        "fit_seconds": fit_s,
        "train_rows": int(len(splits["train"])),
        "note": "Test probabilities are saved for the single calibration pass. Test metrics are not computed here.",
        "run": run_record("tfidf", git=git_commit, hardware=hardware, extra={"contract_version": contract.version}),
    }
    (results_dir() / "02_tfidf.json").write_text(json.dumps(report, indent=2))
    return report


def _load_causal(cfg: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if cfg.get("dtype") == "float32" else torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=dtype)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


def train_zero_shot(git_commit: str | None = None, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch

    cfg = load_training_config()["baseline_zero_shot"]
    if not torch.cuda.is_available():
        raise RuntimeError("zero-shot logit read is a GPU job in this harness")
    torch.manual_seed(int(cfg["seed"]))
    contract = load_contract()
    splits = read_splits()
    model, tokenizer = _load_causal(cfg)
    batch = int(cfg["logit_read"]["batch_size"])
    val_logits = logit_read(model, tokenizer, splits["val"]["text"].tolist(), contract, batch_size=batch)
    test_logits = logit_read(model, tokenizer, splits["test"]["text"].tolist(), contract, batch_size=batch)
    np.save(results_dir() / "zeroshot_val_logits.npy", val_logits)
    np.save(results_dir() / "zeroshot_test_logits.npy", test_logits)
    sample_cfg = cfg["free_generation_sample"]
    try:
        diagnostic = free_generation_sample(
            model,
            tokenizer,
            splits["test"]["text"].tolist(),
            splits["test"]["label"].tolist(),
            contract,
            n=int(sample_cfg["n"]),
            batch=int(sample_cfg["batch"]),
            max_new_tokens=int(sample_cfg["max_new_tokens"]),
            seed=int(cfg["seed"]),
        )
    except Exception as exc:
        diagnostic = {"error": str(exc), "note": "diagnostic free generation failed; logit read logits were still saved"}
    report = {
        "model": cfg["model"],
        "revision": cfg["revision"],
        "val_rows": int(len(splits["val"])),
        "test_rows_inferred": int(len(splits["test"])),
        "free_generation_diagnostic": diagnostic,
        "note": "Logit read is the baseline. Free generation is diagnostic and is not a serving path.",
        "run": run_record("zero_shot", git=git_commit, hardware=hardware, extra={"contract_version": contract.version}),
    }
    (results_dir() / "03_zero_shot.json").write_text(json.dumps(report, indent=2))
    return report
