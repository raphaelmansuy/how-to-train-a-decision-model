"""HTTP service. Failures become needs_human, except malformed JSON and oversized bodies."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import jsonschema
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from bank_routing.contract import Contract, load as load_contract
from bank_routing.decide import decide, load_bundle_contract, text_hash
from bank_routing.paths import schema_dir
from bank_routing.pii import redact

SERVICE_VERSION = "1.0.0"


def _checker() -> jsonschema.FormatChecker:
    return jsonschema.FormatChecker()


class RouterService:
    def __init__(
        self,
        logits_fn: Callable[[str], Any],
        *,
        contract: Contract | None = None,
        bundle_dir: Path | None = None,
        audit_path: Path | None = None,
        kill_switch_file: Path | None = None,
        timeout_ms: int = 150,
        rate_limit_per_minute: int = 600,
        model_version: str = "unreleased",
    ) -> None:
        self.active = contract or load_contract()
        self.contract = load_bundle_contract(bundle_dir, self.active) if bundle_dir else self.active
        self.logits_fn = logits_fn
        self.audit_path = audit_path
        self.kill_switch_file = kill_switch_file
        self.timeout_ms = timeout_ms
        self.rate_limit_per_minute = rate_limit_per_minute
        self.model_version = model_version
        self._idempotency: dict[str, tuple[datetime, dict[str, Any]]] = {}
        self._hits: list[datetime] = []
        self.ready = False
        request_schema = json.loads((schema_dir() / "route-request.json").read_text())
        decision_schema = json.loads((schema_dir() / "route-decision.json").read_text())
        self._request_validator = jsonschema.Draft202012Validator(request_schema, format_checker=_checker())
        self._decision_validator = jsonschema.Draft202012Validator(decision_schema, format_checker=_checker())

    def kill_switch_on(self) -> bool:
        if os.environ.get("BANK_ROUTING_KILL_SWITCH") == "1":
            return True
        return bool(self.kill_switch_file and self.kill_switch_file.exists())

    def _rate_limited(self) -> bool:
        now = datetime.now(timezone.utc)
        self._hits = [hit for hit in self._hits if now - hit < timedelta(minutes=1)]
        if len(self._hits) >= self.rate_limit_per_minute:
            return True
        self._hits.append(now)
        return False

    def _audit(self, raw_text: str, decision: dict[str, Any], latency_ms: float) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "request_id": decision.get("request_id"),
            "message_id": decision.get("message_id"),
            "text_sha256": text_hash(raw_text),
            "redacted_text": redact(raw_text),
            "route": decision.get("route"),
            "label": decision.get("label"),
            "confidence": decision.get("confidence"),
            "probabilities": decision.get("probabilities"),
            "threshold": decision.get("threshold"),
            "decision_rule": decision.get("decision_rule"),
            "expected_cost": decision.get("expected_cost"),
            "temperature": self.contract.temperature,
            "reason": decision.get("reason"),
            "model_version": decision.get("model_version"),
            "contract_version": decision.get("contract_version"),
            "service_version": SERVICE_VERSION,
            "latency_ms": latency_ms,
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")

    def route(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        message_id = payload.get("message_id") if isinstance(payload, dict) else None
        text = payload.get("text") if isinstance(payload, dict) else None
        if isinstance(message_id, str) and message_id in self._idempotency:
            cached_at, cached = self._idempotency[message_id]
            if datetime.now(timezone.utc) - cached_at < timedelta(hours=24):
                return dict(cached)
        errors = list(self._request_validator.iter_errors(payload))
        text_too_long = isinstance(text, str) and len(text) > 2000
        missing = not isinstance(payload, dict) or "message_id" not in payload or "text" not in payload
        if missing or text_too_long:
            raise SchemaError("required fields are missing" if missing else "text longer than 2000 characters")
        whitespace = not isinstance(text, str) or not text.strip()
        other_errors = [err for err in errors if not (list(err.path) == ["text"] and whitespace)]
        if whitespace or other_errors:
            decision = decide(
                None,
                self.contract,
                self.logits_fn,
                message_id=str(message_id) if message_id is not None else None,
                model_version=self.model_version,
            )
        else:
            decision = decide(
                text,
                self.contract,
                self.logits_fn,
                message_id=str(message_id),
                model_version=self.model_version,
                kill_switch=self.kill_switch_on(),
                timeout_ms=self.timeout_ms,
            )
        decision["request_id"] = request_id
        decision["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if isinstance(message_id, str):
            self._idempotency[message_id] = (datetime.now(timezone.utc), decision)
        self._decision_validator.validate(decision)
        self._audit(text if isinstance(text, str) else "", decision, decision["latency_ms"])
        return decision


class SchemaError(Exception):
    pass


def create_app(service: RouterService) -> FastAPI:
    app = FastAPI(title="bank routing decision model", version=SERVICE_VERSION)
    app.state.service = service

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if request.url.path == "/v1/route":
            length = request.headers.get("content-length")
            if length and int(length) > 16 * 1024:
                return JSONResponse({"error": "body larger than 16 KB", "request_id": str(uuid.uuid4())}, status_code=413)
        return await call_next(request)

    @app.get("/healthz")
    def healthz():
        if not service.ready:
            return JSONResponse({"error": "model not ready", "request_id": str(uuid.uuid4())}, status_code=503)
        return {"status": "ok"}

    @app.get("/v1/info")
    def info():
        return {
            "model_version": service.model_version,
            "contract_version": service.contract.version,
            "temperature": service.contract.temperature,
            "threshold": service.contract.threshold,
            "decision_rule": "expected_cost",
            "human_cost": service.contract.costs["human"],
            "service_version": SERVICE_VERSION,
        }

    @app.post("/v1/route")
    async def route(request: Request):
        if service._rate_limited():
            return JSONResponse(
                {"error": "rate limit exceeded", "request_id": str(uuid.uuid4())},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        raw = await request.body()
        if len(raw) > 16 * 1024:
            return JSONResponse({"error": "body larger than 16 KB", "request_id": str(uuid.uuid4())}, status_code=413)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return JSONResponse({"error": "malformed JSON", "request_id": str(uuid.uuid4())}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "malformed JSON", "request_id": str(uuid.uuid4())}, status_code=400)
        try:
            decision = service.route(payload)
        except SchemaError as exc:
            return JSONResponse({"error": str(exc), "request_id": str(uuid.uuid4())}, status_code=400)
        except jsonschema.ValidationError:
            return JSONResponse({"error": "decision failed validation", "request_id": str(uuid.uuid4())}, status_code=500)
        return decision

    @app.post("/v1/override")
    async def override(request: Request):
        payload = await request.json()
        message_id = payload.get("message_id")
        new_queue = payload.get("new_queue")
        allowed = list(service.contract.labels) + ["out_of_scope"]
        if not message_id or new_queue not in allowed:
            return JSONResponse({"error": "message_id and a queue are required", "request_id": str(uuid.uuid4())}, status_code=400)
        line = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
            "request_id": str(uuid.uuid4()),
            "message_id": message_id,
            "event": "override",
            "new_queue": new_queue,
            "reason": payload.get("reason"),
            "model_version": service.model_version,
            "contract_version": service.contract.version,
            "service_version": SERVICE_VERSION,
        }
        if service.audit_path:
            service.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with service.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
        return {"status": "recorded", "message_id": message_id, "new_queue": new_queue}

    try:
        warmup = service.route({"message_id": "warmup", "text": "Why was I charged extra when I paid in euros?"})
        service.ready = warmup["route"] in list(service.contract.labels) + ["needs_human"]
    except Exception:
        service.ready = False
    return app
