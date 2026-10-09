"""Paths. The repo root is local; Modal sets BANK_ROUTING_ROOT and BANK_ROUTING_ARTIFACTS."""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    env = os.environ.get("BANK_ROUTING_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def artifacts() -> Path:
    env = os.environ.get("BANK_ROUTING_ARTIFACTS")
    if env:
        return Path(env)
    return root()


def config_dir() -> Path:
    return root() / "config"


def schema_dir() -> Path:
    return root() / "schemas"


def data_dir() -> Path:
    path = artifacts() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir() -> Path:
    path = artifacts() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = artifacts() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path
