"""Deterministic rule-synthetic Romanization for the complete LLM cohort."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .validation import contains_bengali


@dataclass(frozen=True)
class SpellingRule:
    rule_id: str
    find: str
    replace: str


@dataclass(frozen=True)
class RuleCatalog:
    version: str
    seed: int
    rules: tuple[SpellingRule, ...]
    identity_rule_id: str


def load_rule_catalog(path: Path) -> RuleCatalog:
    """Load and validate the versioned, ordered spelling-rule catalog."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    rules = tuple(
        SpellingRule(str(rule["id"]), str(rule["find"]), str(rule["replace"]))
        for rule in raw["rules"]
    )
    if not raw.get("catalog_version") or not rules or not raw.get("identity_rule_id"):
        raise ValueError("rule catalog requires version, rules, and identity_rule_id")
    if len({rule.rule_id for rule in rules}) != len(rules):
        raise ValueError("rule catalog contains duplicate rule IDs")
    if any(not rule.find or not rule.replace for rule in rules):
        raise ValueError("rule catalog rules require non-empty find and replace values")
    return RuleCatalog(
        version=str(raw["catalog_version"]),
        seed=int(raw["seed"]),
        rules=rules,
        identity_rule_id=str(raw["identity_rule_id"]),
    )


def render_rule_synthetic(text: str, catalog: RuleCatalog) -> tuple[str, tuple[str, ...]]:
    """Apply ordered ASCII spelling rules while retaining formulas and notation."""
    rendered = text
    applied: list[str] = []
    for rule in catalog.rules:
        if rule.find in rendered:
            rendered = rendered.replace(rule.find, rule.replace)
            applied.append(rule.rule_id)
    if not applied:
        applied.append(catalog.identity_rule_id)
    if contains_bengali(rendered):
        raise ValueError("rendered synthetic text contains Bengali script")
    return rendered, tuple(applied)


def build_rule_synthetic_records(
    canonical_rows: Iterable[dict[str, Any]], complete_triplet_item_ids: set[str], catalog: RuleCatalog
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one schema-valid synthetic record per complete LLM-triplet item."""
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in canonical_rows:
        item_id = row["item_id"]
        if item_id not in complete_triplet_item_ids:
            continue
        if item_id in seen:
            rejected.append({"item_id": item_id, "reasons": ["duplicate_canonical_item"]})
            continue
        seen.add(item_id)
        try:
            text, applied_rules = render_rule_synthetic(row["text"], catalog)
        except ValueError as error:
            rejected.append({"item_id": item_id, "reasons": [str(error)]})
            continue
        records.append(
            {
                "row_id": f"{item_id}:synthetic:{catalog.version}",
                "item_id": item_id,
                "collection": "core_paired",
                "task_type": row["task_type"],
                "condition": "synthetic_variant",
                "text": text,
                "answer": row["answer"],
                "source_dataset": row["source_dataset"],
                "source_item_id": row["source_item_id"],
                "provenance": "synthetic",
                "meaning_preserved": True,
                "answer_preserved": True,
                "dialectal": False,
                "code_mixed": row["code_mixed"],
                "canonical_engine": row["canonical_engine"],
                "variant_rule_ids": list(applied_rules),
                "generator_model": None,
                "generator_prompt_hash": None,
                "generator_seed": None,
            }
        )
    missing = sorted(complete_triplet_item_ids - seen)
    rejected.extend({"item_id": item_id, "reasons": ["missing_canonical_item"]} for item_id in missing)
    records.sort(key=lambda row: row["item_id"])
    return records, rejected
