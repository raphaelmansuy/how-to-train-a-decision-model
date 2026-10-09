"""Banking77 download, intent mapping, dedup, leakage removal, and the stratified split."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from bank_routing.contract import Contract, load as load_contract, queue_for_intent
from bank_routing.paths import config_dir, data_dir, results_dir
from bank_routing.runlog import run_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def normalise(text: str, pattern: str = r"[^a-z0-9 ]") -> str:
    return re.sub(pattern, "", str(text).lower()).strip()


def load_training_config() -> dict[str, Any]:
    return yaml.safe_load((config_dir() / "training.yaml").read_text())


def download_sources(cfg: dict[str, Any] | None = None) -> tuple[Path, Path]:
    cfg = cfg or load_training_config()
    data = data_dir()
    pairs = (
        ("banking77_train.csv", cfg["data"]["train_url"]),
        ("banking77_test.csv", cfg["data"]["test_url"]),
    )
    paths = []
    for name, url in pairs:
        dest = data / name
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
        expected = cfg["data"]["raw_sha256"][name]
        actual = sha256_file(dest)
        if actual != expected:
            raise ValueError(f"{name} sha256 {actual} does not match the spec value {expected}")
        paths.append(dest)
    return paths[0], paths[1]


def map_frame(frame: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    if "text" not in frame.columns or "category" not in frame.columns:
        raise ValueError("Banking77 CSV must have text and category columns")
    rows = []
    for text, intent in zip(frame["text"], frame["category"]):
        rows.append({"text": str(text), "intent": str(intent), "label": queue_for_intent(contract, str(intent))})
    return pd.DataFrame(rows)


def drop_exact_duplicates(frame: pd.DataFrame, pattern: str) -> tuple[pd.DataFrame, int]:
    norms = frame["text"].map(lambda s: normalise(s, pattern))
    keep = ~norms.duplicated(keep="first")
    removed = int((~keep).sum())
    return frame.loc[keep].reset_index(drop=True), removed


def drop_exact_overlap(test: pd.DataFrame, reference_norms: set[str], pattern: str) -> tuple[pd.DataFrame, int]:
    norms = test["text"].map(lambda s: normalise(s, pattern))
    keep = ~norms.isin(reference_norms)
    return test.loc[keep].reset_index(drop=True), int((~keep).sum())


def char_tfidf(texts: list[str], *, ngram_range: tuple[int, int], min_df: int):
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=min_df)
    return vectorizer, vectorizer.fit_transform(texts)


def max_cosine(vectorizer: TfidfVectorizer, reference, queries: list[str]) -> tuple[np.ndarray, list[int]]:
    if not queries:
        return np.zeros(0, dtype=np.float64), []
    query = vectorizer.transform(queries)
    max_sim = np.zeros(query.shape[0], dtype=np.float64)
    argmax = np.zeros(query.shape[0], dtype=np.int64)
    batch = 256
    for start in range(0, query.shape[0], batch):
        block = cosine_similarity(query[start : start + batch], reference)
        max_sim[start : start + block.shape[0]] = block.max(axis=1)
        argmax[start : start + block.shape[0]] = block.argmax(axis=1)
    return max_sim, argmax.tolist()


def near_duplicate_mask(
    reference_texts: list[str],
    query_texts: list[str],
    *,
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 1,
    threshold: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return a boolean mask of query rows at or above the cosine threshold, the max similarity, and the argmax index."""
    vectorizer, reference = char_tfidf(reference_texts, ngram_range=ngram_range, min_df=min_df)
    max_sim, argmax = max_cosine(vectorizer, reference, query_texts)
    return max_sim >= threshold, max_sim, argmax


def prepare_frames(train_raw: pd.DataFrame, test_raw: pd.DataFrame, contract: Contract, cfg: dict[str, Any]) -> dict[str, Any]:
    pattern = cfg["data"]["normalise_regex"]
    near = cfg["data"]["near_duplicate"]
    split = cfg["data"]["split"]
    mapped_train = map_frame(train_raw, contract)
    mapped_test = map_frame(test_raw, contract)
    deduped, exact_removed = drop_exact_duplicates(mapped_train, pattern)
    train, rest = train_test_split(
        deduped,
        train_size=int(split["train"]),
        stratify=deduped["label"],
        random_state=int(split["random_state"]),
    )
    val, _unused = train_test_split(
        rest,
        train_size=int(split["val"]),
        stratify=rest["label"],
        random_state=int(split["random_state"]),
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    reference = deduped
    ref_norms = set(reference["text"].map(lambda s: normalise(s, pattern)))
    test_after_exact, exact_overlap = drop_exact_overlap(mapped_test, ref_norms, pattern)
    ref_norm_texts = [normalise(s, pattern) for s in reference["text"]]
    query_norm_texts = [normalise(s, pattern) for s in test_after_exact["text"]]
    vectorizer, ref_matrix = char_tfidf(
        ref_norm_texts,
        ngram_range=tuple(near["ngram_range"]),
        min_df=int(near["min_df"]),
    )
    sims, nearest = max_cosine(vectorizer, ref_matrix, query_norm_texts)
    mask = sims >= float(near["cosine_threshold"])
    examples = []
    order = np.argsort(-sims)
    for idx in order:
        if not mask[idx]:
            break
        if len(examples) >= 5:
            break
        src = int(nearest[int(idx)])
        examples.append(
            {
                "test_text": str(test_after_exact.iloc[int(idx)]["text"]),
                "train_text": str(reference.iloc[src]["text"]),
                "similarity": float(sims[int(idx)]),
            }
        )
    test = test_after_exact.loc[~mask].reset_index(drop=True)
    ref_index = {norm: i for i, norm in enumerate(ref_norm_texts)}
    val_positions = [ref_index[normalise(text, pattern)] for text in val["text"]]
    train_positions = [ref_index[normalise(text, pattern)] for text in train["text"]]
    kept_queries = [normalise(text, pattern) for text in test["text"]]
    val_sims, _ = max_cosine(vectorizer, ref_matrix[val_positions], kept_queries)
    train_sims, _ = max_cosine(vectorizer, ref_matrix[train_positions], kept_queries)
    val_vs_test_exact = len(
        set(val["text"].map(lambda s: normalise(s, pattern))) & set(test["text"].map(lambda s: normalise(s, pattern)))
    )
    threshold = float(near["cosine_threshold"])
    return {
        "train": train,
        "val": val,
        "test": test,
        "report": {
            "raw_train_rows": int(len(train_raw)),
            "raw_test_rows": int(len(test_raw)),
            "exact_duplicates_removed_from_train": exact_removed,
            "deduplicated_train_rows": int(len(deduped)),
            "unused_train_rows": int(len(deduped) - len(train) - len(val)),
            "exact_train_test_overlaps_removed": exact_overlap,
            "near_duplicates_removed": int(mask.sum()),
            "near_duplicate_examples": examples,
            "val_test_exact_overlaps": int(val_vs_test_exact),
            "val_test_near_duplicates": int((val_sims >= threshold).sum()) if len(val_sims) else 0,
            "max_val_test_similarity": float(val_sims.max()) if len(val_sims) else 0.0,
            "train_test_near_duplicates": int((train_sims >= threshold).sum()) if len(train_sims) else 0,
            "max_train_test_similarity": float(train_sims.max()) if len(train_sims) else 0.0,
            "split_sizes": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
            "class_balance": {
                split_name: {str(k): int(v) for k, v in frame["label"].value_counts().to_dict().items()}
                for split_name, frame in (("train", train), ("val", val), ("test", test))
            },
            "seed": int(split["random_state"]),
            "near_duplicate_threshold": float(near["cosine_threshold"]),
        },
    }


def read_splits() -> dict[str, pd.DataFrame]:
    data = data_dir()
    return {name: pd.read_csv(data / f"{name}.csv") for name in ("train", "val", "test")}


def prepare(git_commit: str | None = None) -> dict[str, Any]:
    cfg = load_training_config()
    contract = load_contract()
    train_path, test_path = download_sources(cfg)
    prepared = prepare_frames(pd.read_csv(train_path), pd.read_csv(test_path), contract, cfg)
    data = data_dir()
    for name in ("train", "val", "test"):
        prepared[name].to_csv(data / f"{name}.csv", index=False)
    checksums = {
        "banking77_train.csv": sha256_file(train_path),
        "banking77_test.csv": sha256_file(test_path),
        "train.csv": sha256_file(data / "train.csv"),
        "val.csv": sha256_file(data / "val.csv"),
        "test.csv": sha256_file(data / "test.csv"),
    }
    report = prepared["report"]
    report["checksums"] = checksums
    report["contract_version"] = contract.version
    report["run"] = run_record("prepare_data", git=git_commit, extra={"config_data": cfg["data"]})
    if report["val_test_exact_overlaps"] or report["val_test_near_duplicates"] or report["train_test_near_duplicates"]:
        raise ValueError(
            "validation and test still overlap after leakage removal: "
            f"exact={report['val_test_exact_overlaps']} "
            f"val_near={report['val_test_near_duplicates']} max_val={report['max_val_test_similarity']} "
            f"train_near={report['train_test_near_duplicates']} max_train={report['max_train_test_similarity']}"
        )
    out = results_dir() / "01_data_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    from bank_routing.cards import write_data_card

    write_data_card(report)
    return report
