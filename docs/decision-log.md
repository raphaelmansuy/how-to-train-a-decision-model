# Decision log

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

## Run results

Candidate `router-modernbert-2026-10-09-e3dc058a50f7f54e` scored the test set once at 2026-10-09T14:55:58.542966+00:00 on n=2379.

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
