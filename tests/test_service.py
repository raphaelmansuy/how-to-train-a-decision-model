import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bank_routing.service import RouterService, create_app


class CountingLogits:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str) -> np.ndarray:
        self.calls += 1
        if "slow" in text:
            import time

            time.sleep(0.2)
        return np.array([-3.0, -3.0, -3.0, 4.0, -3.0, -3.0])


@pytest.fixture()
def client(contract, tmp_path: Path):
    logits = CountingLogits()
    service = RouterService(
        logits,
        contract=contract,
        audit_path=tmp_path / "audit.log",
        kill_switch_file=tmp_path / "KILL",
        timeout_ms=50,
        rate_limit_per_minute=100,
        model_version="test-bundle",
    )
    app = create_app(service)
    return TestClient(app), service, logits


def test_http_decisions(client):
    http, service, logits = client
    assert http.get("/healthz").status_code == 200
    info = http.get("/v1/info").json()
    assert info["contract_version"] == "1.0.0"
    assert info["temperature"] == pytest.approx(1.24)
    ok = http.post("/v1/route", json={"message_id": "a", "text": "Why was I charged extra when I paid in euros?", "channel": "chat", "locale": "en-GB"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["route"] == body["label"] == "fees"
    assert body["reason"] == "auto_routed"
    assert len(body["top3"]) == 3
    calls_after_first = logits.calls
    again = http.post("/v1/route", json={"message_id": "a", "text": "a different message should not be scored"})
    assert again.status_code == 200
    assert again.json()["route"] == "fees"
    assert logits.calls == calls_after_first
    blank = http.post("/v1/route", json={"message_id": "b", "text": "   "})
    assert blank.status_code == 200 and blank.json()["reason"] == "invalid_input" and blank.json()["label"] is None
    missing = http.post("/v1/route", json={"text": "hello"})
    assert missing.status_code == 400
    bad = http.post("/v1/route", content=b"not-json", headers={"content-type": "application/json"})
    assert bad.status_code == 400
    huge = http.post("/v1/route", content=b'{"text":"' + b"x" * 17000 + b'"}', headers={"content-type": "application/json"})
    assert huge.status_code == 413
    bad_channel = http.post("/v1/route", json={"message_id": "c", "text": "hello there", "channel": "sms"})
    assert bad_channel.status_code == 200 and bad_channel.json()["reason"] == "invalid_input"
    service.kill_switch_file.write_text("on")
    stopped = http.post("/v1/route", json={"message_id": "d", "text": "hello there"})
    assert stopped.status_code == 200 and stopped.json()["reason"] == "kill_switch"
    service.kill_switch_file.unlink()
    lines = service.audit_path.read_text().strip().splitlines()
    assert lines
    logged = json.loads(lines[0])
    assert "text_sha256" in logged and "redacted_text" in logged and logged["contract_version"] == "1.0.0"
    override = http.post("/v1/override", json={"message_id": "a", "new_queue": "fraud", "reason": "lost card"})
    assert override.status_code == 200
    assert any(json.loads(line).get("event") == "override" for line in service.audit_path.read_text().splitlines())


def test_rate_limit(contract, tmp_path: Path):
    service = RouterService(
        CountingLogits(),
        contract=contract,
        audit_path=tmp_path / "audit.log",
        timeout_ms=50,
        rate_limit_per_minute=2,
        model_version="test-bundle",
    )
    http = TestClient(create_app(service))
    codes = [http.post("/v1/route", json={"message_id": f"r{i}", "text": "hello there"}).status_code for i in range(3)]
    assert codes[-1] == 429


def test_ship_report_when_present(repo_root: Path):
    report = repo_root / "results" / "07_ship.json"
    if not report.exists():
        pytest.skip("ONNX ship report is written by the Modal run")
    payload = json.loads(report.read_text())
    assert payload["parity"]["argmax_agreement"] == pytest.approx(1.0)
    assert payload["parity"]["max_abs_logit_diff"] <= 1e-3
