"""Deterministic source-to-task mapping for curated native candidates."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any


VALID_TASKS = {"math_reasoning", "basic_understanding", "factual_qa"}


def load_task_mapping(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_matching_task(
    value: str, rules: list[dict[str, str]]
) -> str | None:
    for rule in rules:
        if re.search(rule["pattern"], value, flags=re.IGNORECASE):
            return rule["task_type"]
    return None


def map_task(row: dict[str, Any], mapping: dict[str, Any]) -> tuple[str, str] | None:
    """Return a task type and an auditable deterministic mapping rule."""
    source = row["source_dataset"]
    if source == "BEnQA":
        task = _first_matching_task(
            str(row.get("source_file", "")), mapping["benqa_file_patterns"]
        )
        return (task, "benqa_file_pattern") if task else None
    if source == "BnMMLU":
        task = _first_matching_task(
            str(row.get("source_subject", "")), mapping["bnmmlu_subject_patterns"]
        )
        return (task, "bnmmlu_subject_pattern") if task else None
    task = mapping["default_task_by_source"].get(source)
    return (task, "source_default") if task else None


def map_candidate_tasks(
    rows: Iterable[dict[str, Any]], mapping: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map all supported sources and quarantine rows with no deterministic rule."""
    mapped: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for row in rows:
        result = map_task(row, mapping)
        if result is None or result[0] not in VALID_TASKS:
            quarantined.append(
                {
                    **row,
                    "dedup_action": "quarantine_unmapped_task",
                    "task_mapping_reason": "no deterministic task mapping rule",
                }
            )
            continue
        task_type, method = result
        mapped.append(
            {
                **row,
                "source_task_type": row["task_type"],
                "task_type": task_type,
                "task_mapping_method": method,
            }
        )
    return mapped, quarantined
