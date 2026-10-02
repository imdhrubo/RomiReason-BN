"""Preregistered item-clustered summaries and bootstrap intervals."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np


def item_level_outcomes(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["item_id"]].append(row)
    outcomes: list[dict[str, Any]] = []
    for item_id, group in sorted(groups.items()):
        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_condition[row["condition"]].append(row)
        required = {"native", "canonical", "llm_generated_variant", "synthetic_variant"}
        if required - set(by_condition) or len(by_condition["llm_generated_variant"]) != 3:
            raise ValueError(f"{item_id}: incomplete paired evaluation rows")
        native = by_condition["native"][0]["correct"]
        canonical = by_condition["canonical"][0]["correct"]
        synthetic = by_condition["synthetic_variant"][0]["correct"]
        llm_rows = by_condition["llm_generated_variant"]
        llm_mean = sum(row["correct"] for row in llm_rows) / len(llm_rows)
        llm_all = all(row["correct"] for row in llm_rows)
        outcomes.append({"item_id": item_id, "task_type": group[0]["task_type"], "native": native,
                         "canonical": canonical, "synthetic": synthetic, "llm_mean": llm_mean, "llm_all": llm_all,
                         "all_forms_same_and_correct": native and canonical and synthetic and llm_all})
    return outcomes


def paired_bootstrap(outcomes: list[dict[str, Any]], left: str, right: str, replicates: int, seed: int) -> dict[str, float]:
    """Item-resampled paired difference: mean(left - right), percentile interval."""
    differences = np.fromiter(
        (float(row[left]) - float(row[right]) for row in outcomes), dtype=np.float64
    )
    point = float(differences.mean())
    # A multinomial draw over the observed paired differences is exactly
    # equivalent to item-resampling with replacement, while avoiding a large
    # replicates-by-items matrix.  This supports both binary endpoints and the
    # primary mean-over-three-LLM-forms endpoint (with thirds as values).
    values, frequencies = np.unique(differences, return_counts=True)
    draws_by_value = np.random.default_rng(seed).multinomial(
        len(differences), frequencies / len(differences), size=replicates
    )
    draws = np.sort(draws_by_value @ values / len(differences))
    return {"estimate": point, "ci_2_5": draws[int(.025 * (replicates - 1))], "ci_97_5": draws[int(.975 * (replicates - 1))]}


def outcome_rates(outcomes: list[dict[str, Any]]) -> dict[str, float]:
    """Item-level accuracy, with mean LLM-form accuracy as the primary endpoint."""
    return {condition: sum(float(row[condition]) for row in outcomes) / len(outcomes)
            for condition in ("native", "canonical", "llm_mean", "synthetic", "llm_all")}


def paired_contrasts(outcomes: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, dict[str, float]]:
    return {f"{left}_minus_native": paired_bootstrap(outcomes, left, "native", replicates, seed)
            for left in ("canonical", "llm_mean", "synthetic", "llm_all")}


def form_level_summary(scored_rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Format validity, accuracy, and token use for every presented form."""
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_condition[row["condition"]].append(row)
    result: dict[str, dict[str, float | int]] = {}
    for condition, rows in sorted(by_condition.items()):
        result[condition] = {
            "forms": len(rows),
            "parsed_forms": sum(row.get("parse_status") == "parsed" for row in rows),
            "parse_rate": sum(row.get("parse_status") == "parsed" for row in rows) / len(rows),
            "correct_forms": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "mean_prompt_tokens": sum(row.get("prompt_tokens", 0) for row in rows) / len(rows),
            "mean_completion_tokens": sum(row.get("completion_tokens", 0) for row in rows) / len(rows),
        }
    return result


def overall_form_summary(scored_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Overall formatting and token-use diagnostics across all presented forms."""
    return {
        "forms": len(scored_rows),
        "parsed_forms": sum(row.get("parse_status") == "parsed" for row in scored_rows),
        "parse_rate": sum(row.get("parse_status") == "parsed" for row in scored_rows) / len(scored_rows),
        "correct_forms": sum(row["correct"] for row in scored_rows),
        "accuracy": sum(row["correct"] for row in scored_rows) / len(scored_rows),
        "mean_prompt_tokens": sum(row.get("prompt_tokens", 0) for row in scored_rows) / len(scored_rows),
        "mean_completion_tokens": sum(row.get("completion_tokens", 0) for row in scored_rows) / len(scored_rows),
    }


def analysis_summary(scored_rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    outcomes = item_level_outcomes(scored_rows)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        by_task[row["task_type"]].append(row)
    task_stratified = {
        task: {"items": len(task_outcomes), "condition_accuracy": outcome_rates(task_outcomes),
               "paired_bootstrap": paired_contrasts(task_outcomes, replicates, seed)}
        for task, task_outcomes in sorted(by_task.items())
    }
    return {"items": len(outcomes),
            "primary_outcome": "mean_accuracy_across_three_llm_generated_forms",
            "condition_accuracy": outcome_rates(outcomes),
            "paired_bootstrap": paired_contrasts(outcomes, replicates, seed),
            "secondary_all_forms_correct_rate": sum(row["all_forms_same_and_correct"] for row in outcomes) / len(outcomes),
            "form_level": form_level_summary(scored_rows),
            "overall_form_level": overall_form_summary(scored_rows),
            "task_stratified": task_stratified,
            "task_item_counts": {task: len(rows) for task, rows in sorted(by_task.items())},
            "mixed_effects_input": "use scored form-level rows with correctness ~ condition * task_type + token covariates + item/model effects"}


def export_cross_run_summary(summary_root: Path, output: Path) -> int:
    """Write one paper-table-ready row per completed run summary."""
    fields = [
        "run_group", "model", "items", "overall_parse_rate", "native_accuracy",
        "canonical_accuracy", "llm_mean_accuracy", "synthetic_accuracy", "llm_all_accuracy",
        "canonical_minus_native", "llm_mean_minus_native", "synthetic_minus_native",
        "llm_all_minus_native", "all_forms_correct_rate",
    ]
    rows: list[dict[str, str | int | float]] = []
    for path in sorted(summary_root.glob("*/summaries/*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rates = summary["condition_accuracy"]
        contrasts = summary["paired_bootstrap"]
        rows.append({
            "run_group": path.parents[1].name,
            "model": path.stem,
            "items": summary["items"],
            "overall_parse_rate": summary["overall_form_level"]["parse_rate"],
            "native_accuracy": rates["native"],
            "canonical_accuracy": rates["canonical"],
            "llm_mean_accuracy": rates["llm_mean"],
            "synthetic_accuracy": rates["synthetic"],
            "llm_all_accuracy": rates["llm_all"],
            "canonical_minus_native": contrasts["canonical_minus_native"]["estimate"],
            "llm_mean_minus_native": contrasts["llm_mean_minus_native"]["estimate"],
            "synthetic_minus_native": contrasts["synthetic_minus_native"]["estimate"],
            "llm_all_minus_native": contrasts["llm_all_minus_native"]["estimate"],
            "all_forms_correct_rate": summary["secondary_all_forms_correct_rate"],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
