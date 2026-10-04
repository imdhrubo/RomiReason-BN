"""Leakage-controlled preparation for in-domain supervised adaptation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .curation import sha256_file, write_jsonl_rows


ANSWER_ONLY_USER_PROMPT = """Solve the following assessment item. Give only the final answer enclosed in <answer></answer> tags.

Inside the answer tag:
- If the item provides answer choices, write only the selected choice label exactly as shown.
- If it presents a premise and hypothesis, write exactly one of: Entailment, Contradiction, Neutral.
- If the answer is numeric, write only the value in standard notation, with no unit.
- Otherwise, write only the concise answer.

Question: {item_text}"""


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def split_rows(rows: list[dict[str, Any]], seed: int, development_fraction: float, test_fraction: float) -> dict[str, str]:
    """Deterministically stratify frozen items by task, never splitting variants."""
    if not 0 < development_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("development and test fractions must be between zero and one")
    if development_fraction + test_fraction >= 1:
        raise ValueError("development and test fractions leave no training data")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("review_status") != "frozen":
            raise ValueError("in-domain SFT requires frozen reviewed rows")
        by_task[row["task_type"]].append(row)
    split_by_item: dict[str, str] = {}
    for task, task_rows in by_task.items():
        ordered = sorted(task_rows, key=lambda row: _rank(seed, row["item_id"]))
        size = len(ordered)
        development = round(size * development_fraction)
        test = round(size * test_fraction)
        for row in ordered[:development]:
            split_by_item[row["item_id"]] = "development"
        for row in ordered[development:development + test]:
            split_by_item[row["item_id"]] = "test"
        for row in ordered[development + test:]:
            split_by_item[row["item_id"]] = "train"
    return split_by_item


def selected_llm_romanization(row: dict[str, Any], seed: int) -> tuple[int, str]:
    """Select one of the pre-existing LLM Romanizations reproducibly per item."""
    variants = sorted(row["llm_generated_variants"], key=lambda variant: variant["variant_id"])
    if len(variants) != 3:
        raise ValueError(f"{row['item_id']}: expected exactly three LLM Romanizations")
    selected = variants[int(_rank(seed, row["item_id"]), 16) % len(variants)]
    return int(selected["variant_id"]), str(selected["text"])


def sft_record(row: dict[str, Any], split: str, input_condition: str, item_text: str, variant_id: int | None) -> dict[str, Any]:
    return {
        "record_id": f"{row['item_id']}:{input_condition}",
        "item_id": row["item_id"],
        "split": split,
        "task_type": row["task_type"],
        "source_dataset": row["source_dataset"],
        "input_condition": input_condition,
        "romanization_variant_id": variant_id,
        "messages": [
            {"role": "user", "content": ANSWER_ONLY_USER_PROMPT.format(item_text=item_text)},
            {"role": "assistant", "content": f"<answer>{row['answer']}</answer>"},
        ],
    }


def build_in_domain_sft(
    rows: list[dict[str, Any]], config: dict[str, Any], output_dir: Path, manifest_path: Path
) -> dict[str, Any]:
    """Write matched native, Romanized, and mixed SFT training records.

    The mixed condition contains separate native and Romanized records; it never
    gives both scripts for one inference input.
    """
    if config.get("status") != "frozen":
        raise ValueError("in-domain SFT configuration is not frozen")
    split_by_item = split_rows(
        rows, config["split_seed"], config["development_fraction"], config["test_fraction"]
    )
    records: dict[str, list[dict[str, Any]]] = {
        "native_only_train": [], "romanized_only_train": [], "mixed_train": [],
        "native_only_development": [], "romanized_only_development": [], "mixed_development": [],
        "split_manifest": [],
    }
    for row in sorted(rows, key=lambda entry: entry["item_id"]):
        split = split_by_item[row["item_id"]]
        variant_id, romanized = selected_llm_romanization(row, config["romanization_selection_seed"])
        records["split_manifest"].append({
            "item_id": row["item_id"], "split": split, "task_type": row["task_type"],
            "source_dataset": row["source_dataset"], "romanization_variant_id": variant_id,
        })
        if split == "test":
            continue
        native = sft_record(row, split, "native", row["native_text"], None)
        romanized_record = sft_record(row, split, "llm_romanized", romanized, variant_id)
        prefix = "train" if split == "train" else "development"
        records[f"native_only_{prefix}"].append(native)
        records[f"romanized_only_{prefix}"].append(romanized_record)
        records[f"mixed_{prefix}"].extend((native, romanized_record))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    for name, entries in records.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl_rows(path, entries)
        output_paths[name] = str(path)
        output_hashes[name] = sha256_file(path)
    split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in records["split_manifest"]:
        split_counts[row["split"]][row["task_type"]] += 1
    manifest = {
        "status": "frozen_in_domain_sft_inputs",
        "config": config,
        "items": len(rows),
        "train_items": len(records["native_only_train"]),
        "mixed_train_records": len(records["mixed_train"]),
        "split_counts": {split: dict(counts) for split, counts in sorted(split_counts.items())},
        "output_paths": output_paths,
        "output_sha256": output_hashes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
