"""Assemble review-ready paired data and portable Hugging Face Parquet output."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill


REVIEW_COLUMNS = (
    "item_id", "form_id", "condition", "variant_id", "task_type", "answer",
    "native_text", "candidate_text", "source_dataset", "source_item_id",
    "meaning_preserved", "answer_preserved", "naturalness_1_to_5", "dialectal",
    "code_mixed", "spelling_processes", "review_notes", "adjudication",
)

HF_TASK_COLUMNS = (
    "item_id", "task_type", "answer", "native_text", "canonical_text",
    "llm_generated_variants", "rule_synthetic_text",
)

_EXCEL_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _excel_text(value: Any) -> Any:
    """Remove XML-illegal controls for XLSX only; Parquet retains raw text."""
    return _EXCEL_ILLEGAL.sub("", value) if isinstance(value, str) else value


def assemble_paired_candidate(
    native_rows: Iterable[dict[str, Any]],
    canonical_rows: Iterable[dict[str, Any]],
    llm_rows: Iterable[dict[str, Any]],
    synthetic_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join the complete LLM-triplet cohort to its native, canonical, and rule forms."""
    native = {row["item_id"]: row for row in native_rows}
    canonical = {row["item_id"]: row for row in canonical_rows}
    synthetic = {row["item_id"]: row for row in synthetic_rows}
    llm: dict[str, list[dict[str, Any]]] = {}
    for row in llm_rows:
        llm.setdefault(row["item_id"], []).append(row)

    item_ids = set(llm)
    if set(synthetic) != item_ids:
        raise ValueError("synthetic and complete LLM cohorts differ")
    missing = item_ids - set(native) - set()
    if missing:
        raise ValueError(f"native rows missing for {len(missing)} complete-triplet items")
    if item_ids - set(canonical):
        raise ValueError("canonical rows missing for complete-triplet items")

    assembled: list[dict[str, Any]] = []
    for item_id in sorted(item_ids):
        variants = sorted(llm[item_id], key=lambda row: row["variant_id"])
        if [row["variant_id"] for row in variants] != [1, 2, 3]:
            raise ValueError(f"{item_id}: LLM rows are not a complete ordered triplet")
        item, canon, rule = native[item_id], canonical[item_id], synthetic[item_id]
        if len({item["answer"], canon["answer"], rule["answer"], *(row["answer"] for row in variants)}) != 1:
            raise ValueError(f"{item_id}: answers differ across conditions")
        if len({item["task_type"], canon["task_type"], rule["task_type"], *(row["task_type"] for row in variants)}) != 1:
            raise ValueError(f"{item_id}: task types differ across conditions")
        source = item["source_provenance"]
        assembled.append({
            "item_id": item_id,
            "task_type": item["task_type"],
            "answer": str(item["answer"]),
            "native_text": item["native_text"],
            "canonical_text": canon["text"],
            "llm_generated_variants": [
                {"variant_id": row["variant_id"], "text": row["romanized_text"]}
                for row in variants
            ],
            "rule_synthetic_text": rule["text"],
            "rule_synthetic_rule_ids": rule["variant_rule_ids"],
            "source_dataset": source["source_dataset"],
            "source_item_id": str(source["source_item_id"]),
            "source_provenance_json": json.dumps(source, ensure_ascii=False, sort_keys=True),
            "review_status": "pending_expert_review",
        })
    return assembled


def export_review_candidate(rows: list[dict[str, Any]], parquet_path: Path, workbook_path: Path) -> dict[str, int]:
    """Write one item-level Parquet and one form-level, blank-label reviewer workbook."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), parquet_path, compression="zstd")

    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook(write_only=False)
    guide = workbook.active
    guide.title = "README"
    guide.append(["RomiReason-BN reviewer workbook — candidate dataset"])
    guide.append(["Status", "Pending two-expert review; do not treat as a frozen benchmark release."])
    guide.append(["Review rows", len(rows) * 4])
    guide.append(["How to use", "Each reviewer should work in an independent copy and return their completed file to the coordinator."])
    guide.append(["Labels", "meaning_preserved and answer_preserved: yes/no; naturalness: 1–5; dialectal and code_mixed: yes/no."])
    guide.append(["Do not edit", "item_id, form_id, source fields, native text, candidate text, or answer."])
    guide.column_dimensions["A"].width = 22
    guide.column_dimensions["B"].width = 110

    sheet = workbook.create_sheet("review_items")
    sheet.append(list(REVIEW_COLUMNS))
    for row in rows:
        common = [row["item_id"], row["task_type"], row["answer"], row["native_text"], row["source_dataset"], row["source_item_id"]]
        forms = [
            (f"{row['item_id']}:llm:1", "llm_generated_variant", 1, row["llm_generated_variants"][0]["text"]),
            (f"{row['item_id']}:llm:2", "llm_generated_variant", 2, row["llm_generated_variants"][1]["text"]),
            (f"{row['item_id']}:llm:3", "llm_generated_variant", 3, row["llm_generated_variants"][2]["text"]),
            (f"{row['item_id']}:rule:1", "synthetic_variant", 1, row["rule_synthetic_text"]),
        ]
        for form_id, condition, variant_id, candidate in forms:
            item_id, task_type, answer, native_text, source_dataset, source_item_id = common
            sheet.append([_excel_text(value) for value in [
                item_id, form_id, condition, variant_id, task_type, answer, native_text, candidate,
                source_dataset, source_item_id, "", "", "", "", "", "", "", "",
            ]])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(wrap_text=True)
    for column, width in {
        "A": 21, "B": 29, "C": 24, "D": 11, "E": 20, "F": 16,
        "G": 60, "H": 60, "I": 18, "J": 20, "K": 18, "L": 18,
        "M": 20, "N": 14, "O": 14, "P": 28, "Q": 45, "R": 24,
    }.items():
        sheet.column_dimensions[column].width = width
    for row in sheet.iter_rows(min_row=2):
        for index in (6, 7, 16):
            row[index].alignment = Alignment(wrap_text=True, vertical="top")
    workbook.save(workbook_path)
    return {"items": len(rows), "review_rows": len(rows) * 4}


def freeze_reviewed_release(candidate_path: Path, frozen_path: Path) -> dict[str, int]:
    """Promote the reviewed candidate into an immutable evaluation release."""
    rows = pq.read_table(candidate_path).to_pylist()
    if not rows:
        raise ValueError("review candidate is empty")
    if any(row.get("review_status") != "pending_expert_review" for row in rows):
        raise ValueError("candidate does not have the expected pending-review status")
    frozen_rows = [
        {**row, "review_status": "frozen", "review_outcome": "approved_all_forms"}
        for row in rows
    ]
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(frozen_rows), frozen_path, compression="zstd")
    return {"items": len(frozen_rows), "forms": len(frozen_rows) * 6}


def export_hf_task_release(internal_path: Path, hf_path: Path) -> dict[str, int]:
    """Export the minimal task-facing schema from the frozen internal archive."""
    rows = pq.read_table(internal_path).to_pylist()
    if not rows or any(row.get("review_status") != "frozen" for row in rows):
        raise ValueError("internal source must be a non-empty frozen release")
    hf_rows = [{column: row[column] for column in HF_TASK_COLUMNS} for row in rows]
    hf_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(hf_rows), hf_path, compression="zstd")
    return {"items": len(hf_rows), "columns": len(HF_TASK_COLUMNS), "forms": len(hf_rows) * 6}
