"""Evaluator-run planning without downloading weights or performing inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .validation import sha256_file


FORMS_PER_ITEM = 6  # native, canonical, three LLM forms, one rule-synthetic form


def build_model_run_plan(models_path: Path, protocol_path: Path, cohort_items: int) -> dict[str, Any]:
    """Create a fail-closed execution inventory from pinned model metadata."""
    if cohort_items <= 0:
        raise ValueError("cohort_items must be positive")
    models = json.loads(models_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    ready: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for model in models["models"]:
        if not model.get("selected"):
            continue
        required = ("repository", "revision", "tokenizer_revision", "tokenizer_sha256")
        target = ready if all(model.get(field) for field in required) else pending
        target.append({
            "name": model["name"], "repository": model["repository"],
            "revision": model.get("revision"), "tokenizer_revision": model.get("tokenizer_revision"),
            "tokenizer_sha256": model.get("tokenizer_sha256"), "license": model.get("license"),
            "access_status": model.get("access_status"), "forms": cohort_items * FORMS_PER_ITEM,
            "inference_enabled": False,
        })
    return {
        "status": "planned_not_executable_until_dataset_and_protocol_freeze",
        "cohort_items": cohort_items,
        "forms_per_item": FORMS_PER_ITEM,
        "models_ready_for_final_verification": ready,
        "models_pending_access_or_tokenizer_verification": pending,
        "total_planned_forms_when_all_models_ready": cohort_items * FORMS_PER_ITEM * (len(ready) + len(pending)),
        "models_config_sha256": sha256_file(models_path),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_version": protocol["protocol_version"],
        "decoding": protocol["decoding"],
        "required_run_artifacts": [
            "model_and_tokenizer_revisions", "tokenizer_sha256", "prompt_hash", "raw_response",
            "parsed_answer", "parse_status", "correctness", "prompt_tokens", "completion_tokens",
            "runtime_versions", "device_and_dtype", "run_manifest",
        ],
    }
