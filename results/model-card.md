# Model card: bank message router

- Name: router-modernbert-2026-10-09-e3dc058a50f7f54e
- Date: 2026-10-09T14:55:58.542966+00:00
- Owner roles: ML lead trains and reports. Business owner owns costs, labels, and the threshold. Fraud operations lead owns the missed-fraud gate. Neither has signed this card.

## Intended use

Read one English banking message and return one of six queues, or `needs_human` when the expected cost of the top queue is not below the cost of a person, or the service fails.

## Out of scope uses

- The model does not write replies to customers.
- The model does not decide whether a payment is fraud. It only sends the message to the fraud queue.
- The model does not block cards, move money, or change accounts.
- Version 1 does not handle languages other than English.
- Version 1 does not handle attachments, images, or voice.
- Version 1 has no out-of-scope class. Those messages go to a person through the hand-off rule.
- This card does not claim compliance with any law or regulation.

## Training

- Contract version: 1.0.0. Labels, in index order: cards, transfers, cash, fees, account, fraud.
- Base model: answerdotai/ModernBERT-base, revision 8949b909ec900327062f0ebf497f51aef5e6f0c8.
- Data: the Banking77 split in the data card. Seed 42. One training seed.
- Settings: 2 epochs, learning rate 5e-5, batch size 32, max length 64, best checkpoint by validation macro-F1.
- Hardware: Modal L4, bf16, not the CPU demo.
- Temperature fitted on validation: 1.455
- Confidence cutoff from the first experiment, chosen on validation by minimum cost: 0.86. Serving uses the expected-cost rule instead of this cutoff.

## Evaluation

- Frozen test set, n=2379, scored at 2026-10-09T14:55:58.542966+00:00.
- Accuracy 0.9416, macro-F1 0.9411, NLL 0.1960, ECE 0.0124.
- Auto-route rate 0.8441, accepted error 0.01942231075697211, missed fraud 5, cost per ticket 0.0560.
- TF-IDF cost per ticket on the same test set: 0.0525.
- Everything to people cost per ticket: 0.2000.

Per-class scores are in `results/06_calibration_abstain.json`.

## Known limits

- English only, and trained on public Banking77 phrasing rather than the bank's own messages.
- One seed. No interval is reported.
- A lost card can score `cards` highly. The hand-off rule is what sends low-confidence cases to a person. Example from the specification: "I lost my card on the bus this morning" is a fraud intent, and a model can still put `cards` first.
- Costs are illustrative and unapproved.

## Monitoring and rollback

Watch queue-mix PSI, the hand-off rate, and override events. The kill switch returns `needs_human` for every message without a new deploy. Roll back by pointing the service at the previous bundle.

## Licences

Banking77 is CC-BY-4.0. ModernBERT-base is used under the licence on its model card.

## Gates

- G1: fail. 77-intent mapping loads. Business sign-off of sections 1 to 3 is still open.
- G2: pass. exact and near-duplicate overlaps between validation and test are zero
- G3: fail. cost 0.0560; cap 0.07 met; TF-IDF cost 0.0525. No business justification is invented when the candidate is more expensive.
- G4: fail. missed fraud rate 0.02127659574468085, catch 0.9787234042553191, top-1 fraud recall 0.928
- G5: pass. top-1 recall by queue: cards 0.936, transfers 0.951, cash 0.932, fees 0.943, account 0.967
- G6: pass. macro-F1 0.9411 on n=2379
- G7: pass. ECE 0.0124 after temperature
- G8: pass. auto-route 0.844, accepted error 0.01942231075697211
- G9: pass. parity max abs 1.078e-04, agreement 1.000; p50 19.0 ms, p95 30.2 ms
- G10: pass. message-length slices within 5 points
- G11: fail. data card and model card are written; ML lead and risk owner approval is still open

## Serving

- ONNX parity max absolute logit difference 1.078e-04, agreement 1.000.
- Latency p50 19.0 ms, p95 30.2 ms, n=300.

## Expected-cost hand-off

The run results above use the confidence cutoff from the first experiment. That cutoff is 0.86. Serving now compares the expected cost of the top queue with the cost of a person. No new number was fitted, and the test labels were not used to choose the rule.

- ModernBERT on the frozen test set: auto-route 0.8247, accepted error 0.018349, missed fraud 1 of 235, catch 0.995745, cost per ticket 0.0519.
- TF-IDF under the same rule, without temperature scaling: cost per ticket 0.0642.
- Sends to a queue that was not the model's first choice: 0.

Numeric gates read again under this rule. Business sign-off stays open.

- G1: fail. Business sign-off of sections 1 to 3 is still open.
- G2: pass. Unchanged. Validation and test still do not share copies.
- G3: pass. expected-cost 0.0519; cap 0.07 met; TF-IDF confidence-cutoff cost 0.0525; TF-IDF expected-cost 0.0642.
- G4: pass. missed fraud rate 0.004255, catch 0.995745, top-1 fraud recall 0.928
- G5: pass. Top-1 recall is unchanged by the hand-off rule: cards 0.936, transfers 0.951, cash 0.932, fees 0.943, account 0.967
- G6: pass. macro-F1 0.9411 is a property of the top label, not the hand-off.
- G7: pass. ECE 0.0124 is unchanged. Temperature was already fitted on validation.
- G8: pass. auto-route 0.8247, accepted error 0.018349
- G9: pass. ONNX parity is unchanged. The hand-off runs after the model scores.
- G10: pass. Unchanged from the first experiment. The hand-off rule does not refit the length slices.
- G11: fail. ML lead and risk owner approval is still open.
