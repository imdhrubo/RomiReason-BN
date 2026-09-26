"""Fail-closed evaluation job construction and deterministic answer scoring."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pyarrow as pa

_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_MCQ = re.compile(r"^(?:answer\s*[:=-]?\s*)?([A-D])\s*[).,:]?\s*$", re.IGNORECASE)
_FINAL_TAG = re.compile(r"<final>\s*(.*?)\s*</final>\s*$", re.DOTALL | re.IGNORECASE)
_THINK_ANSWER_TAGS = re.compile(
    r"^\s*<think>\s*.*?\s*</think>\s*<answer>\s*(.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_OPTION = re.compile(r"^(?:option|বিকল্প)?\s*([12])$", re.IGNORECASE)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {"protocol_version", "status", "prompt_template", "decoding", "scoring", "analysis"}
    if required - protocol.keys():
        raise ValueError("evaluation protocol is incomplete")
    return protocol


def assert_evaluation_ready(dataset_rows: list[dict[str, Any]], protocol: dict[str, Any], model: dict[str, Any]) -> None:
    """Refuse jobs unless both data and evaluator provenance are frozen."""
    if protocol["status"] != "frozen":
        raise ValueError("evaluation protocol is not frozen")
    if any(row.get("review_status") != "frozen" for row in dataset_rows):
        raise ValueError("dataset is not frozen after expert review")
    for key in ("name", "repository", "revision", "tokenizer_revision", "tokenizer_sha256"):
        if not model.get(key):
            raise ValueError(f"evaluator model missing pinned {key}")


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def write_parquet_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a Parquet artifact, creating its parent directory when needed."""
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(materialized), path)


def load_response_rows(path: Path) -> list[dict[str, Any]]:
    """Load one response JSONL file or a directory of completed response shards.

    Shards are consumed in lexical order and must not repeat a form identifier.
    This permits scoring immutable HPC output directly, without concatenating or
    modifying raw response artifacts.
    """
    paths = [path] if path.is_file() else sorted(path.glob("shard-*.jsonl"))
    if not paths:
        raise ValueError(f"no response JSONL files found at {path}")
    rows: list[dict[str, Any]] = []
    form_ids: set[str] = set()
    for response_path in paths:
        with response_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                form_id = str(row.get("form_id", ""))
                if not form_id:
                    raise ValueError(f"{response_path}:{line_number}: response has no form_id")
                if form_id in form_ids:
                    raise ValueError(f"duplicate response form_id: {form_id}")
                form_ids.add(form_id)
                rows.append(row)
    return rows


def stratified_smoke_rows(
    dataset_rows: list[dict[str, Any]], per_task: int, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic, equal-size item cohort from every task type."""
    if per_task <= 0:
        raise ValueError("per_task must be positive")
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in dataset_rows:
        by_task.setdefault(str(row["task_type"]), []).append(row)
    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for task_type in sorted(by_task):
        candidates = sorted(by_task[task_type], key=lambda row: str(row["item_id"]))
        if len(candidates) < per_task:
            raise ValueError(f"task {task_type!r} has fewer than {per_task} rows")
        selected.extend(rng.sample(candidates, per_task))
    return sorted(selected, key=lambda row: str(row["item_id"]))


def build_evaluation_jobs(dataset_rows: list[dict[str, Any]], protocol: dict[str, Any], model: dict[str, Any]) -> list[dict[str, Any]]:
    """Create answer-key-free model inputs and separate scoring metadata."""
    assert_evaluation_ready(dataset_rows, protocol, model)
    jobs: list[dict[str, Any]] = []
    for item in dataset_rows:
        forms = [("native", 1, item["native_text"]), ("canonical", 1, item["canonical_text"])]
        forms.extend(("llm_generated_variant", int(v["variant_id"]), v["text"]) for v in item["llm_generated_variants"])
        forms.append(("synthetic_variant", 1, item["rule_synthetic_text"]))
        for condition, variant_id, text in forms:
            form_id = f"{item['item_id']}:{condition}:{variant_id}"
            jobs.append({
                "form_id": form_id, "item_id": item["item_id"], "condition": condition,
                "variant_id": variant_id, "task_type": item["task_type"], "expected_answer": item["answer"],
                "model": {key: model[key] for key in ("name", "repository", "revision", "tokenizer_revision", "tokenizer_sha256")},
                "decoding": protocol["decoding"], "scoring": protocol["scoring"],
                "prompt": protocol["prompt_template"].format(item_text=text),
            })
    return jobs


def split_inference_and_scoring_jobs(
    jobs: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep prompts on the inference side and answers only in the scoring key."""
    inference_jobs: list[dict[str, Any]] = []
    scoring_key: list[dict[str, Any]] = []
    for job in jobs:
        if "expected_answer" not in job:
            raise ValueError("evaluation job has no expected_answer")
        inference_jobs.append({key: value for key, value in job.items() if key != "expected_answer"})
        scoring_key.append({key: value for key, value in job.items() if key != "prompt"})
    return inference_jobs, scoring_key


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).translate(_BENGALI_DIGITS).casefold().split())


def _numeric_value(value: str) -> tuple[str, tuple[Fraction, ...]] | None:
    """Canonicalize scalar, fraction, and ratio answer expressions without approximation."""
    text = _normalized(value).replace("ঃ", ":").replace(",", "")
    if not text or re.search(r"[^0-9+./:\-]", text):
        return None
    parts = text.split(":")
    if any(not part or part.count("/") > 1 for part in parts):
        return None
    try:
        values = tuple(Fraction(part) for part in parts)
    except (ValueError, ZeroDivisionError):
        return None
    return ("ratio" if len(values) > 1 else "scalar", values)


def _final_answer(output_text: str) -> str | None:
    """Require exactly one terminal final tag while allowing prior reasoning."""
    if output_text.lower().count("<final>") != 1 or output_text.lower().count("</final>") != 1:
        return None
    match = _FINAL_TAG.search(output_text)
    return match.group(1) if match else None


def _parse_schema_aware_final(output_text: str, expected_answer: str) -> tuple[str, bool]:
    actual = _final_answer(output_text)
    if actual is None:
        return "unscorable", False
    expected = _normalized(expected_answer)
    parsed = _normalized(actual)
    if not parsed:
        return "unscorable", False
    if len(expected) == 1 and expected.upper() in {"A", "B", "C", "D"}:
        return "parsed", parsed.upper() == expected.upper()
    if expected in {"entailment", "contradiction", "neutral"}:
        return "parsed", parsed == expected
    option = _OPTION.fullmatch(expected)
    if option:
        predicted = _OPTION.fullmatch(parsed)
        return "parsed", bool(predicted and predicted.group(1) == option.group(1))
    expected_numeric, parsed_numeric = _numeric_value(expected), _numeric_value(parsed)
    if expected_numeric is not None:
        return "parsed", expected_numeric == parsed_numeric
    return "parsed", parsed == expected


def _parse_schema_aware_think_answer(output_text: str, expected_answer: str) -> tuple[str, bool]:
    """Require the v1.2 think/answer envelope and score its answer field."""
    match = _THINK_ANSWER_TAGS.fullmatch(output_text)
    if match is None:
        return "unscorable", False
    # Reuse the exact answer normalization and schema handling from v1.1.
    return _parse_schema_aware_final(f"<final>{match.group(1)}</final>", expected_answer)


def parse_and_score(
    output_text: str, expected_answer: str, parser: str = "strict_answer_only_v1"
) -> tuple[str, bool]:
    """Score answer-only outputs without explanation extraction or repair."""
    if parser == "final_tag_schema_aware_v1":
        return _parse_schema_aware_final(output_text, expected_answer)
    if parser == "think_answer_schema_aware_v1":
        return _parse_schema_aware_think_answer(output_text, expected_answer)
    if parser != "strict_answer_only_v1":
        raise ValueError(f"unknown scoring parser: {parser}")
    expected = _normalized(expected_answer)
    if len(expected) == 1 and expected.upper() in {"A", "B", "C", "D"}:
        match = _MCQ.fullmatch(output_text)
        if not match:
            return "unscorable", False
        return "parsed", match.group(1).casefold() == expected
    parsed = _normalized(output_text)
    if not parsed:
        return "unscorable", False
    return "parsed", parsed == expected


def score_responses(jobs: Iterable[dict[str, Any]], responses: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join a response artifact keyed by form_id to immutable job metadata."""
    by_id = {str(row["form_id"]): row for row in responses}
    scored: list[dict[str, Any]] = []
    for job in jobs:
        response = by_id.get(job["form_id"])
        if response is None:
            status, correct, text = "missing", False, ""
        else:
            text = str(response.get("output_text", ""))
            parser = job.get("scoring", {}).get("parser", "strict_answer_only_v1")
            status, correct = parse_and_score(text, job["expected_answer"], parser)
        scored.append({**job, "output_text": text, "parse_status": status, "correct": correct,
                       "prompt_tokens": response.get("prompt_tokens") if response else None,
                       "completion_tokens": response.get("completion_tokens") if response else None})
    return scored
