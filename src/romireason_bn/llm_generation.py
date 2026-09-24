"""Build reproducible, non-submitting Batch API manifests for synthetic variants."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any


VARIANT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "variants": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "variant_id": {"type": "integer", "minimum": 1, "maximum": 3},
                    "text": {"type": "string"},
                },
                "required": ["variant_id", "text"],
            },
        }
    },
    "required": ["variants"],
}

RETRY_VARIANT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "variant_id": {"type": "integer", "minimum": 1, "maximum": 3},
        "text": {"type": "string"},
    },
    "required": ["variant_id", "text"],
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def dynamic_output_token_cap(native_text: str) -> int:
    """Allow room for three complete renderings without charging for unused tokens."""
    return max(400, math.ceil(1.8 * len(native_text) + 60))


def build_batch_requests(
    items: Iterable[dict[str, Any]],
    *,
    prompt: str,
    model: str,
    max_output_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """Build one structured-output Batch request per native item, without I/O."""
    requests: list[dict[str, Any]] = []
    for item in items:
        item_id = item["item_id"]
        source = item["source_provenance"]
        input_text = (
            f"ITEM_ID: {item_id}\n"
            f"TASK_TYPE: {item['task_type']}\n"
            f"BENGALI_ITEM:\n{item['native_text']}"
        )
        output_cap = max_output_tokens or dynamic_output_token_cap(item["native_text"])
        requests.append(
            {
                "custom_id": f"{item_id}:llm-variants-v1",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "instructions": prompt,
                    "input": input_text,
                    "reasoning": {"effort": "none"},
                    "max_output_tokens": output_cap,
                    "store": False,
                    "metadata": {
                        "item_id": item_id,
                        "source_dataset": str(source["source_dataset"]),
                        "generation": "llm_variants_v1",
                    },
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "synthetic_romanized_variants",
                            "strict": True,
                            "schema": VARIANT_SCHEMA,
                        }
                    },
                },
            }
        )
    return requests


def build_retry_requests(
    retry_rows: Iterable[dict[str, Any]],
    items_by_id: dict[str, dict[str, Any]],
    accepted_rows: Iterable[dict[str, Any]],
    *,
    prompt: str,
    model: str,
) -> list[dict[str, Any]]:
    """Build one replacement request per failed variant, retaining prior successes."""
    retained: dict[str, list[str]] = {}
    for row in accepted_rows:
        retained.setdefault(row["item_id"], []).append(row["text"])
    requests: list[dict[str, Any]] = []
    for row in retry_rows:
        item_id, variant_id = row["item_id"], row["variant_id"]
        item = items_by_id[item_id]
        retained_text = "\n---\n".join(retained.get(item_id, [])) or "(none)"
        input_text = (
            f"TARGET_VARIANT_ID: {variant_id}\n"
            f"BENGALI_ITEM:\n{item['native_text']}\n"
            f"RETAINED_VARIANTS (do not copy these spellings exactly):\n{retained_text}"
        )
        requests.append(
            {
                "custom_id": f"{item_id}:retry:variant:{variant_id}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "instructions": prompt,
                    "input": input_text,
                    "reasoning": {"effort": "none"},
                    "max_output_tokens": dynamic_output_token_cap(item["native_text"]),
                    "store": False,
                    "metadata": {"item_id": item_id, "variant_id": str(variant_id), "generation": "llm_variants_retry_v1"},
                    "text": {"format": {"type": "json_schema", "name": "synthetic_romanized_variant_retry", "strict": True, "schema": RETRY_VARIANT_SCHEMA}},
                },
            }
        )
    return requests


def write_batch_requests(path: Path, requests: Iterable[dict[str, Any]]) -> int:
    rows = list(requests)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)
