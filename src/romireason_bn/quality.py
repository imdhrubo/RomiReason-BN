"""Non-subjective structural quality gates for native source candidates."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


def _has_bengali(text: str) -> bool:
    return any("\u0980" <= character <= "\u09ff" for character in text)


def structural_quality_filter(
    rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply only objective integrity checks; no subjective quality judgments."""
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text", ""))
        answer = str(row.get("answer", ""))
        reasons: list[str] = []
        if not text.strip() or not answer.strip():
            reasons.append("empty_text_or_answer")
        if "\ufffd" in text or "\ufffd" in answer:
            reasons.append("unicode_replacement_character")
        if not _has_bengali(text):
            reasons.append("no_bengali_script_in_text")
        if len(text) > 8_000 or len(answer) > 1_000:
            reasons.append("field_exceeds_structural_length_limit")
        if answer.strip() in {"A", "B", "C", "D"}:
            missing_labels = [
                label
                for label in ("A", "B", "C", "D")
                if not re.search(rf"(?:^|\n){label}\)\s+", text)
            ]
            if missing_labels:
                reasons.append("mcq_answer_without_all_rendered_options")
        if reasons:
            quarantined.append(
                {
                    **row,
                    "dedup_action": "quarantine_structural_quality",
                    "structural_quality_reasons": reasons,
                }
            )
        else:
            accepted.append({**row, "structural_quality_status": "pass"})
    return accepted, quarantined
