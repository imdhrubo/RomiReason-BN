"""Reproducible dataset acquisition through Hugging Face Datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_pinned_dataset(
    dataset_id: str,
    revision: str,
    split: str,
    config_name: str | None = None,
) -> Any:
    """Load an explicitly pinned dataset revision.

    The import is deliberately local: schema and validation commands remain
    usable before optional inference/acquisition dependencies are installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Dataset loading requires the 'datasets' package. Install the project dependencies first."
        ) from error
    return load_dataset(dataset_id, name=config_name, split=split, revision=revision)


def write_acquisition_record(
    output_path: Path,
    *,
    dataset_id: str,
    revision: str,
    split: str,
    config_name: str | None,
    row_count: int,
    artifact_paths: list[Path],
) -> None:
    """Write the immutable acquisition facts required for audit and reuse."""
    payload = {
        "dataset_id": dataset_id,
        "revision": revision,
        "split": split,
        "config_name": config_name,
        "row_count": row_count,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(artifact_paths)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
