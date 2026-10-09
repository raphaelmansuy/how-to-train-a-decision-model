"""Queue prompt and the six-label logit read. Free generation is not a serving path."""

from __future__ import annotations

from typing import Any

import numpy as np

from bank_routing.contract import Contract


def routing_prompt(text: str, contract: Contract) -> str:
    lines = ["Route the customer message to exactly one support queue.", "Queues:"]
    for label in contract.labels:
        lines.append(f"{label}: {contract.queue_scope[label]}")
    lines.append(f"Message: {text}")
    lines.append("Queue:")
    return "\n".join(lines)


def label_token_ids(tokenizer: Any, labels: list[str] | tuple[str, ...]) -> list[int]:
    ids = []
    for label in labels:
        pieces = tokenizer.encode(" " + label, add_special_tokens=False)
        if not pieces:
            raise ValueError(f"label {label!r} encoded to no tokens")
        ids.append(int(pieces[0]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"label first-tokens are not unique: {list(zip(labels, ids))}")
    return ids


def logit_read(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    contract: Contract,
    *,
    batch_size: int = 16,
) -> np.ndarray:
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids = label_token_ids(tokenizer, contract.labels)
    model.eval()
    rows = []
    for start in range(0, len(texts), batch_size):
        batch = [routing_prompt(text, contract) for text in texts[start : start + batch_size]]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, token_ids]
        rows.append(logits.float().cpu().numpy())
    return np.concatenate(rows, axis=0)


def free_generation_sample(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    true_labels: list[str],
    contract: Contract,
    *,
    n: int = 200,
    batch: int = 20,
    max_new_tokens: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Diagnostic only. Production and the release gates use the logit read, never free generation."""
    import torch

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    valid = 0
    exact = 0
    answers = []
    model.eval()
    label_set = {label.lower(): label for label in contract.labels}
    for start in range(0, len(chosen), batch):
        chunk = chosen[start : start + batch]
        prompts = []
        for idx in chunk:
            body = routing_prompt(texts[int(idx)], contract) + "\nAnswer with the queue name only."
            messages = [{"role": "user", "content": body}]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=False,
                )
            )
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_width = encoded["input_ids"].shape[1]
        for row, idx in zip(generated, chunk):
            text = tokenizer.decode(row[prompt_width:], skip_special_tokens=True).strip().lower()
            first = text.split()[0] if text.split() else ""
            answers.append(first)
            if first in label_set:
                valid += 1
                if label_set[first] == true_labels[int(idx)]:
                    exact += 1
        del generated
    sample_n = int(len(chosen))
    return {
        "n": sample_n,
        "indices": chosen.astype(int).tolist(),
        "answers": answers,
        "valid_rate": float(valid / sample_n) if sample_n else 0.0,
        "exact_match_accuracy": float(exact / sample_n) if sample_n else 0.0,
        "note": "diagnostic free generation; not used for routing or gates",
    }
