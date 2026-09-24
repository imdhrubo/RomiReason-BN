"""File and experiment-level validation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .schema import Collection, Condition, Record


REQUIRED_LAYOUT = (
    "configs",
    "data/raw",
    "data/interim",
    "data/final",
    "data/examples",
    "docs",
    "reports",
    "artifacts",
    "src/romireason_bn",
    "tests",
)


def bengali_character_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    bengali = sum("\u0980" <= char <= "\u09ff" for char in letters)
    return bengali / len(letters)


def contains_bengali(text: str) -> bool:
    return any("\u0980" <= char <= "\u09ff" for char in text)


def load_jsonl(path: Path) -> list[Record]:
    records: list[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                records.append(Record.from_dict(raw))
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def validate_records(records: Iterable[Record]) -> list[str]:
    errors: list[str] = []
    rows = list(records)
    row_ids: set[str] = set()
    groups: dict[str, list[Record]] = defaultdict(list)

    for record in rows:
        if record.row_id in row_ids:
            errors.append(f"duplicate row_id: {record.row_id}")
        row_ids.add(record.row_id)
        groups[record.item_id].append(record)

        if record.condition is Condition.NATIVE:
            if bengali_character_ratio(record.text) < 0.60:
                errors.append(f"{record.row_id}: native text has low Bengali ratio")
        elif contains_bengali(record.text):
            errors.append(f"{record.row_id}: Romanized condition contains Bengali script")

    for item_id, group in groups.items():
        collections = {row.collection for row in group}
        tasks = {row.task_type for row in group}
        if len(collections) != 1:
            errors.append(f"{item_id}: mixed collections")
            continue
        if len(tasks) != 1:
            errors.append(f"{item_id}: mixed task types")
        if Collection.CORE_PAIRED in collections:
            conditions = [row.condition for row in group]
            for required in (
                Condition.NATIVE,
                Condition.CANONICAL,
                Condition.LLM_GENERATED_VARIANT,
                Condition.SYNTHETIC_VARIANT,
            ):
                if required not in conditions:
                    errors.append(f"{item_id}: missing {required.value}")
            for singleton in (
                Condition.NATIVE,
                Condition.CANONICAL,
                Condition.LLM_GENERATED_VARIANT,
            ):
                if conditions.count(singleton) != 1:
                    errors.append(f"{item_id}: expected one {singleton.value}")
            if len({row.answer for row in group}) != 1:
                errors.append(f"{item_id}: answers differ across paired forms")
    return errors


def check_layout(root: Path) -> list[str]:
    return [entry for entry in REQUIRED_LAYOUT if not (root / entry).exists()]


def audit_sources(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    names: set[str] = set()
    for index, source in enumerate(raw.get("sources", [])):
        label = source.get("name") or f"source[{index}]"
        if label in names:
            errors.append(f"duplicate source name: {label}")
        names.add(label)
        if source.get("enabled"):
            for field in ("url", "license", "revision", "sha256"):
                if not source.get(field):
                    errors.append(f"{label}: enabled source missing {field}")
            permitted_statuses = {
                "approved",
                "authorized_by_authors_research_use",
                "project_owner_confirmed",
            }
            if source.get("license_status") not in permitted_statuses:
                errors.append(f"{label}: enabled source license is not approved or author-authorized")
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
