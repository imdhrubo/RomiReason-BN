"""Versioned deterministic canonical Romanization for frozen Bengali seeds."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import unicodedata


CANONICAL_ENGINE = "aksharamukha-2.3:Bengali->ISO:nativize=false:bn-compat-v1"
_BENGALI_COMPATIBILITY = str.maketrans({"ৰ": "র", "ৱ": "ব", "৷": ".", "৳": "Tk.", "ৗ": "ৌ"})


def canonicalize_bengali(text: str) -> str:
    """Convert Bengali script to the ISO Latin target used throughout the study."""
    from aksharamukha import transliterate

    normalized = (
        unicodedata.normalize("NFC", text)
        .replace("োৗ", "ৌ")
        .translate(_BENGALI_COMPATIBILITY)
    )
    return transliterate.process("Bengali", "ISO", normalized, nativize=False)


def build_canonical_forms(seeds: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render canonical forms without changing seed IDs, answers, or source fields."""
    forms: list[dict[str, Any]] = []
    for seed in seeds:
        forms.append(
            {
                "item_id": seed["item_id"],
                "task_type": seed["task_type"],
                "canonical_text": canonicalize_bengali(seed["native_text"]),
                "answer": seed["answer"],
                "canonical_engine": CANONICAL_ENGINE,
            }
        )
    return forms


def build_canonical_records(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create schema-valid canonical paired records from frozen native items."""
    records: list[dict[str, Any]] = []
    for item in items:
        source = item["source_provenance"]
        records.append(
            {
                "row_id": f"{item['item_id']}:canonical",
                "item_id": item["item_id"],
                "collection": "core_paired",
                "task_type": item["task_type"],
                "condition": "canonical",
                "text": canonicalize_bengali(item["native_text"]),
                "answer": item["answer"],
                "source_dataset": source["source_dataset"],
                "source_item_id": source["source_item_id"],
                "provenance": "deterministic",
                "meaning_preserved": True,
                "answer_preserved": True,
                "dialectal": False,
                "code_mixed": False,
                "canonical_engine": CANONICAL_ENGINE,
                "variant_rule_ids": [],
                "generator_model": None,
                "generator_prompt_hash": None,
                "generator_seed": None,
            }
        )
    return records
