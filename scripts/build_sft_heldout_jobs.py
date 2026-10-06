#!/usr/bin/env python3
"""Build answer-key-separated held-out evaluation jobs for every SFT run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

# Make the repository package available when this script is run directly from
# an HPC checkout, without requiring a separate editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from romireason_bn.evaluation import (
    build_evaluation_jobs,
    load_parquet_rows,
    load_protocol,
    split_inference_and_scoring_jobs,
)
from romireason_bn.curation import write_jsonl_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split_ids(path: Path, split: str) -> set[str]:
    selected_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            item_id = str(row["item_id"])
            if item_id in selected_ids:
                raise ValueError(f"duplicate {split} item in {path}:{line_number}: {item_id}")
            selected_ids.add(item_id)
    if not selected_ids:
        raise ValueError(f"no {split} items in {path}")
    return selected_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "test"], default="test")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sft-plan", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_ids = read_split_ids(args.split_manifest, args.split)
    dataset_rows = [row for row in load_parquet_rows(args.dataset) if str(row["item_id"]) in selected_ids]
    found_ids = {str(row["item_id"]) for row in dataset_rows}
    if found_ids != selected_ids:
        raise ValueError(f"{args.split} split and dataset disagree: missing={len(selected_ids - found_ids)} extra={len(found_ids - selected_ids)}")
    dataset_rows.sort(key=lambda row: str(row["item_id"]))
    protocol = load_protocol(args.protocol)
    plan = json.loads(args.sft_plan.read_text(encoding="utf-8"))
    model_config = json.loads(args.models.read_text(encoding="utf-8"))
    models_by_repository = {str(row["repository"]): row for row in model_config["models"]}
    runs: list[dict[str, object]] = []
    for planned_model in plan["models"]:
        repository = str(planned_model["repository"])
        model = models_by_repository.get(repository)
        if model is None:
            raise ValueError(f"SFT model is absent from --models: {repository}")
        model_label = repository.rsplit("/", 1)[-1]
        for condition in plan["training_conditions"]:
            condition_name = str(condition["name"])
            for seed in plan["training_seeds"]:
                run_label = f"{model_label}/{condition_name}/seed-{seed}"
                output = args.output_dir / model_label / condition_name / f"seed-{seed}"
                inference_path = output / "inference.jsonl"
                scoring_path = output / "scoring_key.jsonl"
                if inference_path.exists() or scoring_path.exists():
                    raise FileExistsError(f"refusing to overwrite held-out jobs: {output}")
                evaluator = {
                    key: model[key]
                    for key in ("repository", "revision", "tokenizer_revision", "tokenizer_sha256")
                }
                evaluator["name"] = f"{planned_model['name']} SFT {condition_name} seed {seed}"
                jobs = build_evaluation_jobs(dataset_rows, protocol, evaluator)
                inference_jobs, scoring_key = split_inference_and_scoring_jobs(jobs)
                write_jsonl_rows(inference_path, inference_jobs)
                write_jsonl_rows(scoring_path, scoring_key)
                runs.append({
                    "run": run_label,
                    "base_model": repository,
                    "base_revision": model["revision"],
                    "condition": condition_name,
                    "seed": seed,
                    "adapter_path": f"/scratch/$USER/romireason-bn-sft/{run_label}/best_adapter",
                    "inference_jobs": str(inference_path),
                    "inference_jobs_sha256": sha256(inference_path),
                    "scoring_key": str(scoring_path),
                    "scoring_key_sha256": sha256(scoring_path),
                    "forms": len(inference_jobs),
                })
    manifest = {
        "status": f"{args.split}_sft_evaluation_inputs_ready_no_inference_started",
        "dataset": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "split_manifest": str(args.split_manifest),
        "split_manifest_sha256": sha256(args.split_manifest),
        "split": args.split,
        "items": len(dataset_rows),
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "runs": runs,
        "total_forms": sum(int(run["forms"]) for run in runs),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SFT_EVALUATION_JOBS_OK split={args.split} items={len(dataset_rows)} runs={len(runs)} forms={manifest['total_forms']}")


if __name__ == "__main__":
    main()
