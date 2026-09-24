"""Preregistered item-clustered summaries and bootstrap intervals."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


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
        llm_all = all(row["correct"] for row in by_condition["llm_generated_variant"])
        outcomes.append({"item_id": item_id, "task_type": group[0]["task_type"], "native": native,
                         "canonical": canonical, "synthetic": synthetic, "llm_all": llm_all,
                         "all_forms_same_and_correct": native and canonical and synthetic and llm_all})
    return outcomes


def paired_bootstrap(outcomes: list[dict[str, Any]], left: str, right: str, replicates: int, seed: int) -> dict[str, float]:
    """Item-resampled paired difference: mean(left - right), percentile interval."""
    differences = [float(row[left]) - float(row[right]) for row in outcomes]
    point = sum(differences) / len(differences)
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(replicates))
    return {"estimate": point, "ci_2_5": draws[int(.025 * (replicates - 1))], "ci_97_5": draws[int(.975 * (replicates - 1))]}


def analysis_summary(scored_rows: list[dict[str, Any]], replicates: int, seed: int) -> dict[str, Any]:
    outcomes = item_level_outcomes(scored_rows)
    contrasts = {f"{left}_minus_native": paired_bootstrap(outcomes, left, "native", replicates, seed)
                 for left in ("canonical", "synthetic", "llm_all")}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in outcomes:
        by_task[row["task_type"]].append(row)
    return {"items": len(outcomes), "primary_endpoint_rate": sum(row["all_forms_same_and_correct"] for row in outcomes) / len(outcomes),
            "paired_bootstrap": contrasts,
            "task_item_counts": {task: len(rows) for task, rows in sorted(by_task.items())},
            "mixed_effects_input": "use scored form-level rows with correctness ~ condition * task_type + token covariates + item/model effects"}
