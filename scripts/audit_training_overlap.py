#!/usr/bin/env python3
"""Audit direct text overlap between a frozen evaluation set and training files.

This detects documented direct contamination only. It cannot establish absence
from opaque or web-scale pretraining corpora.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value)).casefold()).strip()


def compact_text(value: Any) -> str:
    return re.sub(r"[^\w]+", "", normalized_text(value), flags=re.UNICODE)


def training_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.parquet")))
            paths.extend(sorted(candidate.glob("*.csv")))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            paths.extend(sorted(Path().glob(value)))
    if not paths:
        raise ValueError("no training Parquet or CSV files found")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--training", nargs="+", required=True,
                        help="Parquet/CSV paths, directories, or glob patterns")
    parser.add_argument("--training-columns", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    evaluation = pq.read_table(args.evaluation, columns=["item_id", "native_text"])
    exact_to_ids: dict[str, list[str]] = defaultdict(list)
    compact_to_ids: dict[str, list[str]] = defaultdict(list)
    for item_id, text in zip(evaluation["item_id"].to_pylist(), evaluation["native_text"].to_pylist(), strict=True):
        exact_to_ids[normalized_text(text)].append(str(item_id))
        compact_to_ids[compact_text(text)].append(str(item_id))

    matches: dict[str, dict[str, set[str]]] = {
        column: {"exact": set(), "compact": set()} for column in args.training_columns
    }
    examples: dict[str, dict[str, list[dict[str, str]]]] = {
        column: {"exact": [], "compact": []} for column in args.training_columns
    }
    scanned_rows = 0
    paths = training_paths(args.training)
    for path in paths:
        if path.suffix == ".parquet":
            schema_names = set(pq.ParquetFile(path).schema.names)
            missing = set(args.training_columns) - schema_names
            if missing:
                raise ValueError(f"{path} lacks requested columns: {sorted(missing)}")
            rows = pq.read_table(path, columns=args.training_columns).to_pylist()
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = set(reader.fieldnames or [])
                missing = set(args.training_columns) - fieldnames
                if missing:
                    raise ValueError(f"{path} lacks requested columns: {sorted(missing)}")
                rows = ({column: row[column] for column in args.training_columns} for row in reader)
                for row in rows:
                    scanned_rows += 1
                    for column in args.training_columns:
                        value = row[column]
                        for kind, index in (("exact", exact_to_ids), ("compact", compact_to_ids)):
                            normalized = normalized_text(value) if kind == "exact" else compact_text(value)
                            item_ids = index.get(normalized, [])
                            matches[column][kind].update(item_ids)
                            if item_ids and len(examples[column][kind]) < args.max_examples:
                                examples[column][kind].append({
                                    "item_id": item_ids[0],
                                    "training_excerpt": normalized_text(value)[:300],
                                })
            continue
        else:
            raise ValueError(f"unsupported training-file suffix: {path}")
        scanned_rows += len(rows)
        for row in rows:
            for column in args.training_columns:
                value = row[column]
                for kind, index in (("exact", exact_to_ids), ("compact", compact_to_ids)):
                    normalized = normalized_text(value) if kind == "exact" else compact_text(value)
                    item_ids = index.get(normalized, [])
                    matches[column][kind].update(item_ids)
                    if item_ids and len(examples[column][kind]) < args.max_examples:
                        examples[column][kind].append({
                            "item_id": item_ids[0],
                            "training_excerpt": normalized_text(value)[:300],
                        })

    summary = {
        "audit_type": "direct_text_overlap",
        "evaluation": str(args.evaluation),
        "evaluation_items": evaluation.num_rows,
        "training_files": [str(path) for path in paths],
        "training_rows_scanned": scanned_rows,
        "training_columns": args.training_columns,
        "normalization": {
            "exact": "NFKC, Unicode casefold, collapsed whitespace",
            "compact": "exact normalization plus punctuation/whitespace removal",
        },
        "matches": {
            column: {kind: {"item_count": len(ids), "item_ids": sorted(ids)} for kind, ids in kinds.items()}
            for column, kinds in matches.items()
        },
        "examples": examples,
        "limitation": "A zero result does not establish absence from opaque or web-scale pretraining corpora.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"TRAINING_OVERLAP_AUDIT_OK evaluation_items={evaluation.num_rows} training_rows={scanned_rows}")


if __name__ == "__main__":
    main()
