"""Deterministic acceptance checks for LLM Romanization Batch responses."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from .validation import contains_bengali


_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_URL = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
_OPTION_LABEL = re.compile(r"(?m)^\s*([A-Za-z])\s*[).:]")


def _normalized_numbers(text: str) -> list[str]:
    return _NUMBER.findall(text.translate(_BENGALI_DIGITS))


def _response_json(batch_row: dict[str, Any]) -> dict[str, Any]:
    """Extract the structured-output JSON from a standard Batch output row."""
    response = batch_row.get("response") or {}
    if response.get("status_code") != 200:
        raise ValueError("non_200_response")
    body = response.get("body") or {}
    text = body.get("output_text")
    if text is None:
        for output in body.get("output", []):
            for content in output.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    text = content.get("text")
                    break
            if text is not None:
                break
    if not isinstance(text, str):
        raise ValueError("missing_output_text")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("invalid_output_json") from error
    if not isinstance(parsed, dict):
        raise ValueError("output_not_object")
    return parsed


def _has_introduced_diacritic(text: str) -> bool:
    return any(unicodedata.combining(char) for char in unicodedata.normalize("NFD", text))


def validate_item_response(
    item: dict[str, Any], batch_row: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve valid variants and quarantine only the variants that fail."""
    try:
        payload = _response_json(batch_row)
    except ValueError as error:
        return [], [
            {"item_id": item["item_id"], "variant_id": variant_id, "reasons": [str(error)]}
            for variant_id in (1, 2, 3)
        ]

    variants = payload.get("variants")
    if not isinstance(variants, list) or len(variants) != 3:
        return [], [
            {"item_id": item["item_id"], "variant_id": variant_id, "reasons": ["wrong_variant_count"]}
            for variant_id in (1, 2, 3)
        ]
    ids = [variant.get("variant_id") if isinstance(variant, dict) else None for variant in variants]
    if sorted(ids) != [1, 2, 3]:
        return [], [
            {"item_id": item["item_id"], "variant_id": variant_id, "reasons": ["invalid_variant_ids"]}
            for variant_id in (1, 2, 3)
        ]

    source = item["native_text"]
    source_numbers = _normalized_numbers(source)
    source_urls = _URL.findall(source)
    source_labels = _OPTION_LABEL.findall(source)
    source_line_count = source.count("\n")
    cap = (batch_row.get("response") or {}).get("body", {}).get("usage", {}).get("output_tokens")
    request_cap = batch_row.get("request_max_output_tokens")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    normalized_variants: set[str] = set()

    for variant in variants:
        text = variant.get("text") if isinstance(variant, dict) else None
        if not isinstance(text, str) or not text.strip():
            rejected.append({"item_id": item["item_id"], "variant_id": variant["variant_id"], "reasons": ["empty_variant"]})
            continue
        normalized = " ".join(text.lower().split())
        reasons: list[str] = []
        if normalized in normalized_variants:
            reasons.append("duplicate_variant")
        normalized_variants.add(normalized)
        if contains_bengali(text):
            reasons.append("bengali_script")
        if _has_introduced_diacritic(text):
            reasons.append("latin_diacritic")
        if _normalized_numbers(text) != source_numbers:
            reasons.append("numbers_changed")
        if _URL.findall(text) != source_urls:
            reasons.append("urls_changed")
        if _OPTION_LABEL.findall(text) != source_labels:
            reasons.append("option_labels_or_order_changed")
        if text.count("\n") != source_line_count:
            reasons.append("line_breaks_changed")
        if reasons:
            rejected.append({"item_id": item["item_id"], "variant_id": variant["variant_id"], "reasons": sorted(set(reasons))})
        else:
            accepted.append({"item_id": item["item_id"], "variant_id": variant["variant_id"], "text": text})

    if isinstance(cap, int) and isinstance(request_cap, int) and cap >= request_cap:
        return [], [
            {"item_id": item["item_id"], "variant_id": variant["variant_id"], "reasons": ["possible_truncation"]}
            for variant in variants
        ]
    return accepted, rejected


def validate_batch_rows(
    items: Iterable[dict[str, Any]], batch_rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join Batch outputs to frozen items and quarantine all incomplete items."""
    by_id = {item["item_id"]: item for item in items}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch_row in batch_rows:
        custom_id = str(batch_row.get("custom_id", ""))
        item_id = custom_id.split(":", 1)[0]
        item = by_id.get(item_id)
        if not item:
            rejected.append({"item_id": item_id, "reasons": ["unknown_item_id"]})
            continue
        seen.add(item_id)
        variants, failures = validate_item_response(item, batch_row)
        accepted.extend(variants)
        rejected.extend(failures)
    for item_id in sorted(set(by_id) - seen):
        rejected.extend(
            {"item_id": item_id, "variant_id": variant_id, "reasons": ["missing_batch_response"]}
            for variant_id in (1, 2, 3)
        )
    return accepted, rejected
