"""Run records for every training job (TRN-03)."""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import subprocess
from typing import Any

from bank_routing.paths import root


TRACKED = (
    "torch",
    "transformers",
    "datasets",
    "scikit-learn",
    "pandas",
    "numpy",
    "setfit",
    "sentence-transformers",
    "trl",
    "peft",
    "accelerate",
    "huggingface_hub",
    "tokenizers",
    "onnx",
    "onnxruntime",
    "onnxscript",
)


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root(), stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def library_versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in TRACKED:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def run_record(job: str, *, git: str | None = None, hardware: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "job": job,
        "git_commit": git or git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": library_versions(),
        "hardware": hardware or {},
    }
    if extra:
        record.update(extra)
    return record
