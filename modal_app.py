"""Modal harness. Training runs remotely. A missing login stops the job."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import modal

app = modal.App("bank-routing-decision-model")
cache_vol = modal.Volume.from_name("bank-routing-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("bank-routing-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.1")
    .pip_install(
        "transformers==5.19.0",
        "datasets==5.1.0",
        "scikit-learn==1.9.1",
        "pandas==3.0.6",
        "numpy==2.5.3",
        "setfit==1.2.0",
        "sentence-transformers==6.1.0",
        "trl==1.15.0",
        "peft==0.21.2",
        "accelerate==1.15.0",
        "huggingface_hub==1.33.0",
        "tokenizers==0.23.2",
        "onnx==1.23.2",
        "onnxruntime==1.31.0",
        "onnxscript==0.7.2",
        "pyyaml==6.0.2",
        "jsonschema==4.25.1",
    )
    .env({"BANK_ROUTING_ROOT": "/root", "BANK_ROUTING_ARTIFACTS": "/artifacts", "HF_HOME": "/cache"})
    .add_local_python_source("bank_routing")
    .add_local_dir(str(ROOT / "config"), remote_path="/root/config")
    .add_local_dir(str(ROOT / "schemas"), remote_path="/root/schemas")
)


def _hardware(gpu: str | None) -> dict:
    info = {"requested_gpu": gpu or "cpu"}
    try:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - remote only
        info["torch_error"] = str(exc)
    return info


@app.function(image=image, volumes={"/cache": cache_vol}, timeout=60 * 45, cpu=2.0, memory=8192)
def warmup(git_commit: str) -> dict:
    from huggingface_hub import snapshot_download

    pairs = (
        ("answerdotai/ModernBERT-base", "8949b909ec900327062f0ebf497f51aef5e6f0c8"),
        ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
        ("sentence-transformers/paraphrase-MiniLM-L3-v2", "4ca70771034acceecb2e72475f72050fcdde4ddc"),
    )
    for name, revision in pairs:
        snapshot_download(name, revision=revision)
    cache_vol.commit()
    return {"git_commit": git_commit, "models": [name for name, _ in pairs]}


@app.function(image=image, volumes={"/artifacts": out_vol}, timeout=60 * 30, cpu=2.0, memory=8192)
def prepare(git_commit: str) -> dict:
    from bank_routing.data import prepare as run_prepare

    report = run_prepare(git_commit=git_commit)
    out_vol.commit()
    return report["split_sizes"]


@app.function(image=image, volumes={"/artifacts": out_vol}, timeout=60 * 30, cpu=2.0, memory=8192)
def fit_tfidf(git_commit: str) -> dict:
    out_vol.reload()
    from bank_routing.baselines import train_tfidf

    report = train_tfidf(git_commit=git_commit, hardware=_hardware(None))
    out_vol.commit()
    return {"job": "tfidf", "fit_seconds": report["fit_seconds"]}


@app.function(image=image, gpu="L4", volumes={"/artifacts": out_vol, "/cache": cache_vol}, timeout=60 * 60)
def fit_zero_shot(git_commit: str) -> dict:
    out_vol.reload()
    cache_vol.reload()
    from bank_routing.baselines import train_zero_shot

    train_zero_shot(git_commit=git_commit, hardware=_hardware("L4"))
    out_vol.commit()
    return {"job": "zero_shot"}


@app.function(image=image, gpu="L4", volumes={"/artifacts": out_vol, "/cache": cache_vol}, timeout=60 * 60)
def fit_route_a(git_commit: str) -> dict:
    out_vol.reload()
    cache_vol.reload()
    from bank_routing.train_encoder import train

    report = train(git_commit=git_commit, hardware=_hardware("L4"))
    out_vol.commit()
    return {"job": "route_a", "validation_macro_f1": report["validation_macro_f1"]}


@app.function(image=image, gpu="L4", volumes={"/artifacts": out_vol, "/cache": cache_vol}, timeout=60 * 60)
def fit_setfit(git_commit: str) -> dict:
    out_vol.reload()
    cache_vol.reload()
    from bank_routing.train_setfit import train

    report = train(git_commit=git_commit, hardware=_hardware("L4"))
    out_vol.commit()
    return {"job": "setfit", "train_rows": report["train_rows"]}


@app.function(image=image, gpu="L4", volumes={"/artifacts": out_vol, "/cache": cache_vol}, timeout=60 * 60)
def fit_route_b(git_commit: str) -> dict:
    out_vol.reload()
    cache_vol.reload()
    from bank_routing.train_lora import train

    report = train(git_commit=git_commit, hardware=_hardware("L4"))
    out_vol.commit()
    return {"job": "route_b", "train_rows": report["train_rows"], "test_subset": report["test_subset"]}


@app.function(image=image, volumes={"/artifacts": out_vol}, timeout=60 * 30, cpu=2.0, memory=8192)
def calibrate_job(git_commit: str) -> dict:
    out_vol.reload()
    from bank_routing.calibrate import calibrate

    report = calibrate(git_commit=git_commit, hardware=_hardware(None))
    out_vol.commit()
    return {"model_version": report["model_version"], "gates": report["gates"]}


@app.function(image=image, volumes={"/artifacts": out_vol, "/cache": cache_vol}, timeout=60 * 45, cpu=4.0, memory=16384)
def ship_job(git_commit: str) -> dict:
    out_vol.reload()
    cache_vol.reload()
    from bank_routing.ship import ship

    report = ship(git_commit=git_commit, hardware=_hardware(None))
    out_vol.commit()
    return report["parity"] | {"g9_pass": report["g9_pass"], "latency": report["latency"]}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _download() -> None:
    import shutil

    staging = ROOT / ".modal-download"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    fetches = ("results", "models/bundle", "models/encoder", "data")
    for remote in fetches:
        subprocess.check_call(["modal", "volume", "get", "--force", "bank-routing-artifacts", remote, str(staging)])
    for remote in fetches:
        source = staging / remote
        dest = ROOT / remote
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
    shutil.rmtree(staging, ignore_errors=True)
    docs = ROOT / "docs"
    for name in ("data-card.md", "model-card.md", "decision-log.md"):
        source = ROOT / "results" / name
        if source.exists():
            (docs / name).write_text(source.read_text())


@app.local_entrypoint()
def main() -> None:
    auth = subprocess.run(["modal", "profile", "current"], capture_output=True, text=True)
    if auth.returncode != 0:
        raise SystemExit("Modal is not authenticated. Run `modal setup`. This harness will not train on the local machine.")
    commit = _git_commit()
    print(warmup.remote(commit))
    print(prepare.remote(commit))
    handles = [
        fit_tfidf.spawn(commit),
        fit_zero_shot.spawn(commit),
        fit_route_a.spawn(commit),
        fit_setfit.spawn(commit),
        fit_route_b.spawn(commit),
    ]
    for handle in handles:
        print(handle.get())
    print(calibrate_job.remote(commit))
    print(ship_job.remote(commit))
    _download()
