"""Data card, model card, and the decision log."""

from __future__ import annotations

import json
import os
from typing import Any

from bank_routing.paths import results_dir, root

PREAMBLE = """# Decision log

Standing decisions for this build. Business sign-offs in the specification stay open.

## 2026-10-09 — Illustrative costs

The contract stores human 0.2, wrong queue 1.0, and missed fraud 5.0. These are the specification's illustrative values. The business owner has not approved them. The threshold fitted on validation uses this table and is not a production threshold.

## 2026-10-09 — GPU training on Modal

Route A, Route B, SetFit, and the zero-shot logit read run on a Modal L4 GPU. The specification's demo used CPU (`use_cpu` true, `bf16` false, and a CPU-only TRL loss subclass). On the GPU harness Route A sets `use_cpu` false and `bf16` true. Route B uses TRL's default loss, `bf16` true, and does not use `CPUSFTTrainer`. Epochs, learning rate, batch size, seeds, LoRA rank, and the 240-example / 60-step Route B budget stay as specified. Validation macro-F1 is not required to match the CPU demo, because the hardware is different.

## 2026-10-09 — One seed

`TRN-04` asks for three seeds. This run uses the specification's single seeds. The spread is not estimated.

## 2026-10-09 — Expected-cost hand-off

The first experiment picked one confidence cutoff on validation. That cutoff treats every wrong queue as the same kind of mistake. The contract does not: a missed fraud costs 5.0 and a wrong queue costs 1.0. The service now sends the top queue only when the expected cost of that send is strictly below the cost of a person (0.2). There is no new fitted number. Temperature stays the validation fit. The 0.86 cutoff remains the recorded result of the first experiment. Costs are still illustrative and unapproved.

## 2026-10-09 — Open items

Real cost values, the three fraud edge cases, a separate cost for a false send to fraud, production message lengths, agreement and PII recall on bank data, and regulatory sign-off remain open. This log does not claim a production release.
"""


def _publish(name: str, text: str) -> None:
    (results_dir() / name).write_text(text)
    if not os.environ.get("BANK_ROUTING_ARTIFACTS"):
        docs = root() / "docs"
        if docs.is_dir():
            (docs / name).write_text(text)


def write_data_card(report: dict[str, Any]) -> None:
    balance = report["class_balance"]
    lines = [
        "# Data card: Banking77 queue split",
        "",
        "- Name and version: banking77-queues, contract " + str(report.get("contract_version")),
        "- Source: Banking77, PolyAI, Casanueva et al. 2020, arXiv 2003.04807.",
        "- Licence: CC-BY-4.0. Attribution: PolyAI Ltd. and the Banking77 authors.",
        "- Date range: the public Banking77 release. No bank production messages are included.",
        "- How labels were made: the public 77 intents, mapped by the contract. No model pre-labels were used.",
        "- Agreement: not measured. Banking77 arrived already labelled.",
        "- PII: the public set is not production traffic. Serving redacts card numbers, IBANs, phone numbers, emails, and national-id patterns before scoring.",
        "- Known gaps: English only. No attachments. The unused remainder of the deduplicated train file is not in the 3,000-row training split.",
        "- Owner: ML lead for the dataset build. Business owner has not signed the mapping.",
        "",
        "## Counts",
        "",
        f"- Raw train rows: {report['raw_train_rows']}",
        f"- Raw test rows: {report['raw_test_rows']}",
        f"- Exact duplicates removed inside train: {report['exact_duplicates_removed_from_train']}",
        f"- Exact train/test overlaps removed: {report['exact_train_test_overlaps_removed']}",
        f"- Near-duplicates removed at cosine 0.90: {report['near_duplicates_removed']}",
        f"- Split sizes: {report['split_sizes']}",
        f"- Validation/test exact overlaps remaining: {report['val_test_exact_overlaps']}",
        f"- Validation/test near-duplicates remaining: {report['val_test_near_duplicates']}",
        f"- Seed: {report['seed']}",
        "",
        "## Class balance",
        "",
        f"- Train: {balance['train']}",
        f"- Validation: {balance['val']}",
        f"- Test: {balance['test']}",
        "",
        "## Checksums",
        "",
    ]
    for name, digest in report["checksums"].items():
        lines.append(f"- `{name}`: `{digest}`")
    lines.append("")
    _publish("data-card.md", "\n".join(lines))


def write_model_card(report: dict[str, Any], ship: dict[str, Any] | None = None) -> None:
    candidate = report["comparison"]["modernbert"]
    tfidf = report["comparison"]["tfidf"]
    people = report["comparison"]["everything_to_people"]
    test = candidate["test"]
    lines = [
        "# Model card: bank message router",
        "",
        f"- Name: {report['model_version']}",
        f"- Date: {report['test_scored_at']}",
        "- Owner roles: ML lead trains and reports. Business owner owns costs, labels, and the threshold. Fraud operations lead owns the missed-fraud gate. Neither has signed this card.",
        "",
        "## Intended use",
        "",
        "Read one English banking message and return one of six queues, or `needs_human` when the expected cost of the top queue is not below the cost of a person, or the service fails.",
        "",
        "## Out of scope uses",
        "",
        "- The model does not write replies to customers.",
        "- The model does not decide whether a payment is fraud. It only sends the message to the fraud queue.",
        "- The model does not block cards, move money, or change accounts.",
        "- Version 1 does not handle languages other than English.",
        "- Version 1 does not handle attachments, images, or voice.",
        "- Version 1 has no out-of-scope class. Those messages go to a person through the hand-off rule.",
        "- This card does not claim compliance with any law or regulation.",
        "",
        "## Training",
        "",
        "- Contract version: 1.0.0. Labels, in index order: cards, transfers, cash, fees, account, fraud.",
        "- Base model: answerdotai/ModernBERT-base, revision 8949b909ec900327062f0ebf497f51aef5e6f0c8.",
        "- Data: the Banking77 split in the data card. Seed 42. One training seed.",
        "- Settings: 2 epochs, learning rate 5e-5, batch size 32, max length 64, best checkpoint by validation macro-F1.",
        "- Hardware: Modal L4, bf16, not the CPU demo.",
        f"- Temperature fitted on validation: {candidate['temperature']}",
        f"- Confidence cutoff from the first experiment, chosen on validation by minimum cost: {candidate['threshold']}. Serving uses the expected-cost rule instead of this cutoff.",
        "",
        "## Evaluation",
        "",
        f"- Frozen test set, n={test['n']}, scored at {report['test_scored_at']}.",
        f"- Accuracy {test['accuracy']:.4f}, macro-F1 {test['macro_f1']:.4f}, NLL {test['nll']:.4f}, ECE {test['ece']:.4f}.",
        f"- Auto-route rate {test['abstain']['auto_route_rate']:.4f}, accepted error {test['abstain']['accepted_error']}, missed fraud {test['abstain']['missed_fraud']}, cost per ticket {test['abstain']['cost_per_ticket']:.4f}.",
        f"- TF-IDF cost per ticket on the same test set: {tfidf['test']['abstain']['cost_per_ticket']:.4f}.",
        f"- Everything to people cost per ticket: {people['test']['abstain']['cost_per_ticket']:.4f}.",
        "",
        "Per-class scores are in `results/06_calibration_abstain.json`.",
        "",
        "## Known limits",
        "",
        "- English only, and trained on public Banking77 phrasing rather than the bank's own messages.",
        "- One seed. No interval is reported.",
        "- A lost card can score `cards` highly. The hand-off rule is what sends low-confidence cases to a person. Example from the specification: \"I lost my card on the bus this morning\" is a fraud intent, and a model can still put `cards` first.",
        "- Costs are illustrative and unapproved.",
        "",
        "## Monitoring and rollback",
        "",
        "Watch queue-mix PSI, the hand-off rate, and override events. The kill switch returns `needs_human` for every message without a new deploy. Roll back by pointing the service at the previous bundle.",
        "",
        "## Licences",
        "",
        "Banking77 is CC-BY-4.0. ModernBERT-base is used under the licence on its model card.",
        "",
        "## Gates",
        "",
    ]
    for gate in report["gates"]:
        lines.append(f"- {gate['gate']}: {'pass' if gate['pass'] else 'fail'}. {gate['detail']}")
    if ship:
        lines.extend(
            [
                "",
                "## Serving",
                "",
                f"- ONNX parity max absolute logit difference {ship['parity']['max_abs_logit_diff']:.3e}, agreement {ship['parity']['argmax_agreement']:.3f}.",
                f"- Latency p50 {ship['latency']['p50_ms']:.1f} ms, p95 {ship['latency']['p95_ms']:.1f} ms, n={ship['latency']['n']}.",
            ]
        )
    lines.extend(_expected_cost_lines())
    lines.append("")
    _publish("model-card.md", "\n".join(lines))


def write_decision_log(report: dict[str, Any] | None = None) -> None:
    parts = [PREAMBLE]
    if report:
        parts.append("## Run results\n")
        parts.append(f"Candidate `{report['model_version']}` scored the test set once at {report['test_scored_at']} on n={report['n_test']}.\n")
        if report.get("rescore_warning"):
            parts.append(report["rescore_warning"] + "\n")
        for gate in report["gates"]:
            parts.append(f"- {gate['gate']}: {'pass' if gate['pass'] else 'fail'}. {gate['detail']}")
        parts.append("")
    note = _expected_cost_lines()
    if note:
        parts.append("\n".join(note))
        parts.append("")
    _publish("decision-log.md", "\n".join(parts))


def _expected_cost_lines() -> list[str]:
    path = results_dir() / "08_expected_cost.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    modern = payload["models"]["modernbert"]["test"]
    tfidf = payload["models"]["tfidf"]["test"]
    lines = [
        "",
        "## Expected-cost hand-off",
        "",
        "The run results above use the confidence cutoff from the first experiment. "
        f"That cutoff is {payload['confidence_cutoff']}. "
        "Serving now compares the expected cost of the top queue with the cost of a person. "
        "No new number was fitted, and the test labels were not used to choose the rule.",
        "",
        f"- ModernBERT on the frozen test set: auto-route {modern['auto_route_rate']:.4f}, "
        f"accepted error {modern['accepted_error']}, missed fraud {modern['missed_fraud']} of {modern['fraud_total']}, "
        f"catch {modern['fraud_catch_rate']}, cost per ticket {modern['cost_per_ticket']:.4f}.",
        f"- TF-IDF under the same rule, without temperature scaling: cost per ticket {tfidf['cost_per_ticket']:.4f}.",
        f"- Sends to a queue that was not the model's first choice: {modern['unconstrained_non_top_sends']}.",
        "",
        "Numeric gates read again under this rule. Business sign-off stays open.",
        "",
    ]
    for gate in payload["gates"]:
        lines.append(f"- {gate['gate']}: {'pass' if gate['pass'] else 'fail'}. {gate['detail']}")
    return lines
