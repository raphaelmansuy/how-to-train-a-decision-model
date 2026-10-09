"""Redact card numbers, IBANs, phone numbers, emails, and national-id-like strings before the model sees the text."""

from __future__ import annotations

import re

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.IGNORECASE)
PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}(?!\w)")
NATIONAL_ID = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
DIGITS = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _card_sub(match: re.Match[str]) -> str:
    raw = match.group(0)
    if luhn_ok(raw):
        return "[CARD]"
    return raw


def redact(text: str) -> str:
    cleaned = EMAIL.sub("[EMAIL]", text)
    cleaned = IBAN.sub("[IBAN]", cleaned)
    cleaned = NATIONAL_ID.sub("[NATIONAL_ID]", cleaned)
    cleaned = DIGITS.sub(_card_sub, cleaned)
    cleaned = PHONE.sub("[PHONE]", cleaned)
    return cleaned
