import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bank_routing.contract import load, queue_for_intent
from bank_routing.data import drop_exact_duplicates, drop_exact_overlap, map_frame, near_duplicate_mask, normalise
from bank_routing.decide import decide, load_bundle_contract
from bank_routing.metrics import (
    action_expected_cost,
    cost_per_ticket,
    expected_cost_report,
    fit_temperature,
    pick_threshold,
    softmax,
)
from bank_routing.pii import luhn_ok, redact


def test_contract_has_77_intents_in_order():
    contract = load()
    assert contract.labels == ("cards", "transfers", "cash", "fees", "account", "fraud")
    assert len(contract.intent_mapping) == 77
    assert contract.intent_mapping["Refund_not_showing_up"] == "transfers"
    assert contract.intent_mapping["reverted_card_payment?"] == "transfers"
    assert contract.intent_mapping["transaction_charged_twice"] == "fraud"
    assert "needs_human" not in contract.labels
    with pytest.raises(ValueError, match="unmapped"):
        queue_for_intent(contract, "not_a_real_intent")


def test_cost_counts_missed_fraud_once():
    labels = ["cards", "transfers", "cash", "fees", "account", "fraud"]
    y = np.array(["fraud", "cards", "fees", "transfers"])
    proba = np.array(
        [
            [0.9, 0.02, 0.02, 0.02, 0.02, 0.02],  # missed fraud, auto
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],  # hand-off
            [0.05, 0.05, 0.05, 0.8, 0.03, 0.02],  # correct fees
            [0.1, 0.1, 0.7, 0.05, 0.03, 0.02],  # wrong queue
        ]
    )
    costs = {"human": 0.2, "wrong_queue": 1.0, "missed_fraud": 5.0}
    assert cost_per_ticket(y, proba, labels, 0.5, costs) == pytest.approx((5.0 + 0.2 + 0.0 + 1.0) / 4)
    assert cost_per_ticket(y, np.full((4, 6), 1 / 6), labels, 1.01, costs) == pytest.approx(0.2)


def test_temperature_does_not_change_the_top_label():
    logits = np.array([[0.2, 3.0, 0.1, 0.0, -1.0, 0.4], [2.0, 2.1, 0.0, 0.0, 0.0, 0.0]])
    raw = softmax(logits, 1.0).argmax(1)
    scaled = softmax(logits, 1.8).argmax(1)
    assert np.array_equal(raw, scaled)


def test_threshold_tie_prefers_higher_coverage():
    labels = ["cards", "fraud"]
    y = np.array(["cards", "cards"])
    proba = np.array([[0.8, 0.2], [0.55, 0.45]])
    costs = {"human": 0.2, "wrong_queue": 1.0, "missed_fraud": 5.0}
    # Both thresholds that sit under 0.55 auto-route both rows and cost 0.
    chosen = pick_threshold(y, proba, labels, costs, start=0.0, stop=0.5, step=0.1)
    assert chosen["cost_per_ticket"] == pytest.approx(0.0)
    assert chosen["auto_route_rate"] == pytest.approx(1.0)


def test_fit_temperature_returns_a_positive_scalar():
    logits = np.array([[5.0, 0.0, 0.0], [0.2, 0.1, 4.0], [1.0, 1.2, 0.0]])
    y = np.array([0, 2, 1])
    temperature = fit_temperature(logits, y, {"start": 0.25, "stop": 5.0, "num": 20}, {"half_width": 0.1, "num": 11, "floor": 0.05})
    assert temperature > 0


def test_redaction_covers_the_named_patterns():
    assert luhn_ok("4242424242424242")
    text = "card 4242424242424242 email a@b.co IBAN GB82WEST12345698765432 call +1 202 555 0147 ssn 123-45-6789"
    cleaned = redact(text)
    assert "4242424242424242" not in cleaned
    assert "[CARD]" in cleaned
    assert "[EMAIL]" in cleaned
    assert "[IBAN]" in cleaned
    assert "[PHONE]" in cleaned
    assert "[NATIONAL_ID]" in cleaned


def test_exact_and_near_duplicate_removal():
    contract = load()
    train = pd.DataFrame(
        {
            "text": ["I am waiting on my card", "I am waiting on my card", "A totally different request about fees"],
            "category": ["card_arrival", "card_arrival", "exchange_rate"],
        }
    )
    mapped = map_frame(train, contract)
    deduped, removed = drop_exact_duplicates(mapped, r"[^a-z0-9 ]")
    assert removed == 1
    test = pd.DataFrame({"text": ["I am waiting on my card", "I am waiting on my new card"], "category": ["card_arrival", "card_arrival"]})
    mapped_test = map_frame(test, contract)
    norms = set(deduped["text"].map(lambda s: normalise(s)))
    kept, overlaps = drop_exact_overlap(mapped_test, norms, r"[^a-z0-9 ]")
    assert overlaps == 1
    mask, sims, _ = near_duplicate_mask(
        [normalise(s) for s in deduped["text"]],
        [normalise(s) for s in kept["text"]],
        threshold=0.90,
    )
    assert sims[0] >= 0.90
    assert bool(mask[0]) is True


def test_expected_cost_hands_off_a_sure_looking_call_when_fraud_is_still_plausible(contract):
    labels = ["cards", "transfers", "cash", "fees", "account", "fraud"]
    costs = {"human": 0.2, "wrong_queue": 1.0, "missed_fraud": 5.0}
    proba = np.array([[0.95, 0.0025, 0.0025, 0.0025, 0.0025, 0.04]])
    expected = action_expected_cost(proba, labels, costs)
    assert expected[0, 0] == pytest.approx(0.21)
    assert expected[0, 5] == pytest.approx(0.96)
    report = expected_cost_report(np.array(["fraud"]), proba, labels, costs)
    assert report["auto_route_rate"] == 0.0
    assert report["missed_fraud"] == 0
    assert report["unconstrained_non_top_sends"] == 0
    assert report["cost_per_ticket"] == pytest.approx(0.2)

    def logits(_text: str) -> np.ndarray:
        return contract.temperature * np.log(proba[0])

    decision = decide("I lost my card but the model likes cards", contract, logits, message_id="m-cost", model_version="test")
    assert decision["label"] == "cards"
    assert decision["confidence"] == pytest.approx(0.95, abs=1e-3)
    assert decision["route"] == "needs_human"
    assert decision["reason"] == "expected_cost"
    assert decision["expected_cost"] == pytest.approx(0.21, abs=1e-3)
    assert decision["decision_rule"] == "expected_cost"


def test_decide_routes_or_hands_off(contract):
    def logits(text: str) -> np.ndarray:
        if "boom" in text:
            raise RuntimeError("boom")
        if "unsure" in text:
            return np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
        return np.array([-3.0, -3.0, -3.0, 4.0, -3.0, -3.0])

    sure = decide("Why was I charged extra when I paid in euros?", contract, logits, message_id="m1", model_version="test")
    assert sure["route"] == "fees"
    assert sure["label"] == sure["route"]
    assert abs(sum(sure["probabilities"].values()) - 1) <= 0.001
    assert set(sure["probabilities"]) == set(contract.labels)
    unsure = decide("unsure about this", contract, logits, message_id="m2", model_version="test")
    assert unsure["route"] == "needs_human"
    assert unsure["reason"] == "expected_cost"
    assert unsure["label"] == "cards"
    broken = decide("boom please", contract, logits, message_id="m3", model_version="test")
    assert broken["reason"] == "model_error" and broken["label"] is None
    blank = decide("   ", contract, logits, message_id="m4", model_version="test")
    assert blank["reason"] == "invalid_input"
    stopped = decide("hello there", contract, logits, kill_switch=True, message_id="m5", model_version="test")
    assert stopped["reason"] == "kill_switch"


def test_decide_timeout(contract):
    import time

    def slow(_text: str) -> np.ndarray:
        time.sleep(0.2)
        return np.zeros(6)

    decision = decide("hello there", contract, slow, timeout_ms=20, message_id="m", model_version="test")
    assert decision["reason"] == "timeout"


def test_bundle_label_order_is_refused(tmp_path: Path, contract):
    payload = {"version": "1.0.0", "labels": ["fraud", "cards", "transfers", "cash", "fees", "account"], "temperature": 1.0, "threshold": 0.5}
    (tmp_path / "contract.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="label order"):
        load_bundle_contract(tmp_path, contract)
