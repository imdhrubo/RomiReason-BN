"""Freeze reviewed source candidates into stable native pilot seed IDs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any


def freeze_native_seeds(rows: Iterable[dict[str, Any]], prefix: str = "rrbn-pilot") -> list[dict[str, Any]]:
    """Assign stable IDs while retaining the reviewed source content unchanged."""
    frozen: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        required = {"source_dataset", "source_item_id", "task_type", "text", "answer"}
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"seed candidate missing fields: {', '.join(missing)}")
        frozen.append(
            {
                "item_id": f"{prefix}-{index:04d}",
                "native_text": row["text"],
                "answer": row["answer"],
                "task_type": row["task_type"],
                "source_dataset": row["source_dataset"],
                "source_item_id": row["source_item_id"],
                **({"source_subject": row["source_subject"]} if "source_subject" in row else {}),
                **({"source_config": row["source_config"]} if "source_config" in row else {}),
            }
        )
    return frozen


def freeze_curated_native(
    rows: Iterable[dict[str, Any]], prefix: str = "rrbn-native"
) -> list[dict[str, Any]]:
    """Freeze the full curated corpus while retaining source and curation facts."""
    frozen: list[dict[str, Any]] = []
    for index, row in enumerate(
        sorted(rows, key=lambda candidate: candidate["dedup_record_id"]), start=1
    ):
        required = {
            "dedup_record_id",
            "source_dataset",
            "source_item_id",
            "source_revision",
            "task_type",
            "text",
            "answer",
        }
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"curated candidate missing fields: {', '.join(missing)}")
        source_provenance = {
            key: value for key, value in row.items() if key.startswith("source_")
        }
        curation_provenance = {
            key: value
            for key, value in row.items()
            if key.startswith("dedup_")
            or key.startswith("duplicate_")
            or key.startswith("strict_near_")
            or key.startswith("lexical_near_")
            or key in {"task_mapping_method", "structural_quality_status"}
        }
        frozen.append(
            {
                "item_id": f"{prefix}-{index:05d}",
                "native_text": row["text"],
                "answer": row["answer"],
                "task_type": row["task_type"],
                "source_provenance": source_provenance,
                "curation_provenance": curation_provenance,
            }
        )
    return frozen


def native_dataset_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    frozen = list(rows)
    return {
        "row_count": len(frozen),
        "task_counts": dict(Counter(row["task_type"] for row in frozen)),
        "source_counts": dict(
            Counter(
                row["source_provenance"]["source_dataset"]
                for row in frozen
            )
        ),
    }
