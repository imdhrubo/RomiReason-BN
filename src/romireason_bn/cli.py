"""Command-line entry points for the starter repository."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import Counter

from .acquisition import load_pinned_dataset, write_acquisition_record
from .canonicalize import CANONICAL_ENGINE, build_canonical_records
from .llm_generation import build_batch_requests, build_retry_requests, sha256_text, write_batch_requests
from .llm_validation import validate_batch_rows
from .rule_synthetic import build_rule_synthetic_records, load_rule_catalog
from .review_export import assemble_paired_candidate, export_hf_task_release, export_review_candidate, freeze_reviewed_release
from .evaluation import build_evaluation_jobs, load_parquet_rows, load_protocol, load_response_rows, score_responses, split_inference_and_scoring_jobs, stratified_smoke_rows, write_parquet_rows
from .analysis import analysis_summary
from .model_planning import build_model_run_plan
from .curation import (
    build_dedup_input_manifest,
    exact_deduplication,
    exact_deduplication_summary,
    lexical_near_deduplication,
    mcq_order_invariant_deduplication,
    preprocess_dedup_rows,
    read_jsonl_rows,
    strict_near_deduplication,
    validate_source_answers,
    write_jsonl_rows,
)
from .quality import structural_quality_filter
from .release_export import export_release
from .schema import Record
from .seed_freeze import freeze_curated_native, native_dataset_summary
from .semantic_dedup import semantic_deduplicate
from .source_ingest import (
    normalize_bennumeval_rows,
    normalize_bnmmlu_rows,
    read_bluck_directory,
    read_banglamath_csv,
    read_bmwp_workbook,
    normalize_ganit_rows,
)
from .task_mapping import load_task_mapping, map_candidate_tasks
from .validation import (
    audit_sources,
    check_layout,
    load_jsonl,
    sha256_file,
    validate_records,
    contains_bengali,
)


def _print_errors(errors: list[str]) -> int:
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="romireason")
    subparsers = parser.add_subparsers(dest="command", required=True)

    layout = subparsers.add_parser("check-layout")
    layout.add_argument("--root", type=Path, default=Path.cwd())

    data = subparsers.add_parser("validate-data")
    data.add_argument("path", type=Path)

    sources = subparsers.add_parser("audit-sources")
    sources.add_argument("path", type=Path)

    manifest = subparsers.add_parser("hash-manifest")
    manifest.add_argument("path", type=Path)

    bennumeval = subparsers.add_parser("ingest-bennumeval")
    bennumeval.add_argument("--revision", required=True)
    bennumeval.add_argument("--output", type=Path, required=True)
    bennumeval.add_argument("--rejections", type=Path, required=True)
    bennumeval.add_argument("--manifest", type=Path, required=True)

    bnmmlu = subparsers.add_parser("ingest-bnmmlu")
    bnmmlu.add_argument("--revision", required=True)
    bnmmlu.add_argument("--output", type=Path, required=True)
    bnmmlu.add_argument("--rejections", type=Path, required=True)
    bnmmlu.add_argument("--manifest", type=Path, required=True)

    ganit = subparsers.add_parser("ingest-ganit")
    ganit.add_argument("--revision", required=True)
    ganit.add_argument("--output", type=Path, required=True)
    ganit.add_argument("--rejections", type=Path, required=True)
    ganit.add_argument("--manifest", type=Path, required=True)

    bluck = subparsers.add_parser("ingest-bluck")
    bluck.add_argument("--input-root", type=Path, required=True)
    bluck.add_argument("--revision", required=True)
    bluck.add_argument("--output", type=Path, required=True)
    bluck.add_argument("--rejections", type=Path, required=True)
    bluck.add_argument("--manifest", type=Path, required=True)

    banglamath = subparsers.add_parser("ingest-banglamath")
    banglamath.add_argument("--input-path", type=Path, required=True)
    banglamath.add_argument("--revision", required=True)
    banglamath.add_argument("--output", type=Path, required=True)
    banglamath.add_argument("--rejections", type=Path, required=True)
    banglamath.add_argument("--manifest", type=Path, required=True)

    bmwp = subparsers.add_parser("ingest-bmwp")
    bmwp.add_argument("--input-path", type=Path, required=True)
    bmwp.add_argument("--revision", required=True)
    bmwp.add_argument("--output", type=Path, required=True)
    bmwp.add_argument("--rejections", type=Path, required=True)
    bmwp.add_argument("--manifest", type=Path, required=True)

    dedup_manifest = subparsers.add_parser("dedup-build-manifest")
    dedup_manifest.add_argument("--registry", type=Path, required=True)
    dedup_manifest.add_argument("--root", type=Path, default=Path.cwd())
    dedup_manifest.add_argument("--output", type=Path, required=True)

    dedup_preprocess = subparsers.add_parser("dedup-preprocess")
    dedup_preprocess.add_argument("--input", type=Path, required=True)
    dedup_preprocess.add_argument("--output", type=Path, required=True)
    dedup_preprocess.add_argument("--rejections", type=Path, required=True)

    dedup_validate = subparsers.add_parser("dedup-validate-source-answers")
    dedup_validate.add_argument("--input", type=Path, required=True)
    dedup_validate.add_argument("--output", type=Path, required=True)
    dedup_validate.add_argument("--quarantined", type=Path, required=True)

    dedup_exact = subparsers.add_parser("dedup-exact")
    dedup_exact.add_argument("--input", type=Path, required=True)
    dedup_exact.add_argument("--retained", type=Path, required=True)
    dedup_exact.add_argument("--clusters", type=Path, required=True)
    dedup_exact.add_argument("--quarantined", type=Path, required=True)
    dedup_exact.add_argument("--summary", type=Path)

    dedup_strict_near = subparsers.add_parser("dedup-strict-near")
    dedup_strict_near.add_argument("--input", type=Path, required=True)
    dedup_strict_near.add_argument("--retained", type=Path, required=True)
    dedup_strict_near.add_argument("--clusters", type=Path, required=True)

    dedup_lexical_near = subparsers.add_parser("dedup-lexical-near")
    dedup_lexical_near.add_argument("--input", type=Path, required=True)
    dedup_lexical_near.add_argument("--retained", type=Path, required=True)
    dedup_lexical_near.add_argument("--clusters", type=Path, required=True)
    dedup_lexical_near.add_argument("--threshold", type=float, default=0.985)

    dedup_mcq = subparsers.add_parser("dedup-mcq-order-invariant")
    dedup_mcq.add_argument("--input", type=Path, required=True)
    dedup_mcq.add_argument("--retained", type=Path, required=True)
    dedup_mcq.add_argument("--clusters", type=Path, required=True)
    dedup_mcq.add_argument("--quarantined", type=Path, required=True)

    dedup_semantic = subparsers.add_parser("dedup-semantic")
    dedup_semantic.add_argument("--input", type=Path, required=True)
    dedup_semantic.add_argument("--retained", type=Path, required=True)
    dedup_semantic.add_argument("--clusters", type=Path, required=True)
    dedup_semantic.add_argument("--evidence", type=Path, required=True)
    dedup_semantic.add_argument("--summary", type=Path, required=True)
    dedup_semantic.add_argument("--model-id", required=True)
    dedup_semantic.add_argument("--model-revision", required=True)
    dedup_semantic.add_argument("--threshold", type=float, default=0.70)
    dedup_semantic.add_argument("--top-k", type=int, default=20)
    dedup_semantic.add_argument("--require-lexical-evidence", action="store_true")
    dedup_semantic.add_argument("--fuzzy-threshold", type=float, default=0.80)
    dedup_semantic.add_argument("--jaccard-threshold", type=float, default=0.60)

    promote_semantic = subparsers.add_parser("promote-semantic-baseline")
    promote_semantic.add_argument("--input", type=Path, required=True)
    promote_semantic.add_argument("--strict-parent", type=Path, required=True)
    promote_semantic.add_argument("--output", type=Path, required=True)
    promote_semantic.add_argument("--summary", type=Path, required=True)
    promote_semantic.add_argument("--method", default="consensus_semantic_070")

    exclude_sources = subparsers.add_parser("exclude-sources")
    exclude_sources.add_argument("--input", type=Path, required=True)
    exclude_sources.add_argument("--output", type=Path, required=True)
    exclude_sources.add_argument("--summary", type=Path, required=True)
    exclude_sources.add_argument("--sources", nargs="+", required=True)

    release_export = subparsers.add_parser("export-release")
    release_export.add_argument("--input", type=Path, required=True)
    release_export.add_argument("--parquet", type=Path, required=True)
    release_export.add_argument("--workbook", type=Path, required=True)
    release_export.add_argument("--summary", type=Path, required=True)

    canonicalize = subparsers.add_parser("canonicalize")
    canonicalize.add_argument("--input", type=Path, required=True)
    canonicalize.add_argument("--output", type=Path, required=True)
    canonicalize.add_argument("--summary", type=Path, required=True)

    generation_batch = subparsers.add_parser("build-llm-variant-batch")
    generation_batch.add_argument("--input", type=Path, required=True)
    generation_batch.add_argument("--prompt", type=Path, required=True)
    generation_batch.add_argument("--model", required=True)
    generation_batch.add_argument(
        "--max-output-tokens",
        type=int,
        help="Optional fixed cap; omit to use the documented per-item dynamic cap.",
    )
    generation_batch.add_argument("--output", type=Path, required=True)
    generation_batch.add_argument("--manifest", type=Path, required=True)

    validate_generation = subparsers.add_parser("validate-llm-variants")
    validate_generation.add_argument("--items", type=Path, required=True)
    validate_generation.add_argument("--request-batch", type=Path, required=True)
    validate_generation.add_argument("--batch-output", type=Path, required=True)
    validate_generation.add_argument("--accepted", type=Path, required=True)
    validate_generation.add_argument("--rejected", type=Path, required=True)

    retry_batch = subparsers.add_parser("build-llm-retry-batch")
    retry_batch.add_argument("--items", type=Path, required=True)
    retry_batch.add_argument("--rejected", type=Path, required=True)
    retry_batch.add_argument("--accepted", type=Path, required=True)
    retry_batch.add_argument("--prompt", type=Path, required=True)
    retry_batch.add_argument("--model", required=True)
    retry_batch.add_argument("--max-items", type=int, required=True)
    retry_batch.add_argument("--output", type=Path, required=True)
    retry_batch.add_argument("--manifest", type=Path, required=True)

    shard_batch = subparsers.add_parser("shard-llm-variant-batch")
    shard_batch.add_argument("--input", type=Path, required=True)
    shard_batch.add_argument("--shard-size", type=int, default=1000)
    shard_batch.add_argument("--output-dir", type=Path, required=True)
    shard_batch.add_argument("--manifest", type=Path, required=True)

    consolidate_generation = subparsers.add_parser("consolidate-llm-variants")
    consolidate_generation.add_argument("--items", type=Path, required=True)
    consolidate_generation.add_argument("--batches-root", type=Path, required=True)
    consolidate_generation.add_argument("--generation-manifest", type=Path, required=True)
    consolidate_generation.add_argument("--accepted-output", type=Path, required=True)
    consolidate_generation.add_argument("--complete-triplets-output", type=Path, required=True)
    consolidate_generation.add_argument("--summary", type=Path, required=True)

    rule_synthetic = subparsers.add_parser("build-rule-synthetic")
    rule_synthetic.add_argument("--canonical", type=Path, required=True)
    rule_synthetic.add_argument("--llm-complete-triplets", type=Path, required=True)
    rule_synthetic.add_argument("--catalog", type=Path, required=True)
    rule_synthetic.add_argument("--output", type=Path, required=True)
    rule_synthetic.add_argument("--rejected", type=Path, required=True)
    rule_synthetic.add_argument("--manifest", type=Path, required=True)

    review_export = subparsers.add_parser("export-review-candidate")
    review_export.add_argument("--native", type=Path, required=True)
    review_export.add_argument("--canonical", type=Path, required=True)
    review_export.add_argument("--llm-complete-triplets", type=Path, required=True)
    review_export.add_argument("--rule-synthetic", type=Path, required=True)
    review_export.add_argument("--parquet", type=Path, required=True)
    review_export.add_argument("--workbook", type=Path, required=True)
    review_export.add_argument("--manifest", type=Path, required=True)

    freeze_review = subparsers.add_parser("freeze-reviewed-release")
    freeze_review.add_argument("--candidate", type=Path, required=True)
    freeze_review.add_argument("--output", type=Path, required=True)
    freeze_review.add_argument("--manifest", type=Path, required=True)

    hf_release = subparsers.add_parser("export-hf-task-release")
    hf_release.add_argument("--internal", type=Path, required=True)
    hf_release.add_argument("--output", type=Path, required=True)
    hf_release.add_argument("--manifest", type=Path, required=True)

    eval_jobs = subparsers.add_parser("build-evaluation-jobs")
    eval_jobs.add_argument("--dataset", type=Path, required=True)
    eval_jobs.add_argument("--protocol", type=Path, required=True)
    eval_jobs.add_argument("--model", type=Path, required=True)
    eval_jobs.add_argument("--output", type=Path, required=True)
    eval_jobs.add_argument("--scoring-key", type=Path, required=True)

    smoke_cohort = subparsers.add_parser("build-smoke-cohort")
    smoke_cohort.add_argument("--dataset", type=Path, required=True)
    smoke_cohort.add_argument("--per-task", type=int, default=30)
    smoke_cohort.add_argument("--seed", type=int, default=2027)
    smoke_cohort.add_argument("--output", type=Path, required=True)
    smoke_cohort.add_argument("--manifest", type=Path, required=True)

    primary_jobs = subparsers.add_parser("build-primary-evaluation-jobs")
    primary_jobs.add_argument("--dataset", type=Path, required=True)
    primary_jobs.add_argument("--protocol", type=Path, required=True)
    primary_jobs.add_argument("--models", type=Path, required=True)
    primary_jobs.add_argument("--inference-dir", type=Path, required=True)
    primary_jobs.add_argument("--scoring-dir", type=Path, required=True)
    primary_jobs.add_argument("--manifest", type=Path, required=True)

    score_eval = subparsers.add_parser("score-evaluation-responses")
    score_eval.add_argument("--jobs", type=Path, required=True)
    score_eval.add_argument("--responses", type=Path, required=True)
    score_eval.add_argument("--output", type=Path, required=True)

    analyze_eval = subparsers.add_parser("analyze-evaluations")
    analyze_eval.add_argument("--scored", type=Path, required=True)
    analyze_eval.add_argument("--replicates", type=int, default=10000)
    analyze_eval.add_argument("--seed", type=int, default=2027)
    analyze_eval.add_argument("--output", type=Path, required=True)

    model_plan = subparsers.add_parser("plan-evaluator-runs")
    model_plan.add_argument("--models", type=Path, required=True)
    model_plan.add_argument("--protocol", type=Path, required=True)
    model_plan.add_argument("--cohort-items", type=int, required=True)
    model_plan.add_argument("--output", type=Path, required=True)

    map_tasks = subparsers.add_parser("map-tasks")
    map_tasks.add_argument("--input", type=Path, required=True)
    map_tasks.add_argument("--mapping", type=Path, required=True)
    map_tasks.add_argument("--output", type=Path, required=True)
    map_tasks.add_argument("--quarantined", type=Path, required=True)

    quality = subparsers.add_parser("filter-structural-quality")
    quality.add_argument("--input", type=Path, required=True)
    quality.add_argument("--output", type=Path, required=True)
    quality.add_argument("--quarantined", type=Path, required=True)

    freeze_native = subparsers.add_parser("freeze-native")
    freeze_native.add_argument("--input", type=Path, required=True)
    freeze_native.add_argument("--output", type=Path, required=True)
    freeze_native.add_argument("--summary", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check-layout":
        missing = check_layout(args.root)
        if missing:
            return _print_errors([f"missing path: {path}" for path in missing])
        print("LAYOUT_OK")
        return 0
    if args.command == "validate-data":
        records = load_jsonl(args.path)
        errors = validate_records(records)
        if errors:
            return _print_errors(errors)
        print(f"DATA_OK rows={len(records)} items={len({r.item_id for r in records})}")
        return 0
    if args.command == "audit-sources":
        errors = audit_sources(args.path)
        if errors:
            return _print_errors(errors)
        print("SOURCES_OK")
        return 0
    if args.command == "hash-manifest":
        print(sha256_file(args.path))
        return 0
    if args.command == "ingest-bennumeval":
        configs = ("CA", "DS", "CQ", "FiB", "QNLI", "AWP")
        accepted: list[dict[str, str]] = []
        rejected: list[dict[str, str]] = []
        for config in configs:
            rows = list(
                load_pinned_dataset(
                    "ka05ar/BenNumEval", args.revision, "test", config
                )
            )
            valid, invalid = normalize_bennumeval_rows(config, rows)
            accepted.extend(valid)
            rejected.extend(invalid)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in accepted
            ),
            encoding="utf-8",
        )
        args.rejections.parent.mkdir(parents=True, exist_ok=True)
        args.rejections.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rejected
            ),
            encoding="utf-8",
        )
        write_acquisition_record(
            args.manifest,
            dataset_id="ka05ar/BenNumEval",
            revision=args.revision,
            split="test",
            config_name="CA,DS,CQ,FiB,QNLI,AWP",
            row_count=len(accepted),
            artifact_paths=[args.output, args.rejections],
        )
        print(f"INGESTED_OK accepted={len(accepted)} rejected={len(rejected)}")
        return 0
    if args.command == "ingest-bnmmlu":
        rows = list(
            load_pinned_dataset(
                "samanjoy2/BnMMLU", args.revision, "train", "hard"
            )
        )
        accepted, rejected = normalize_bnmmlu_rows(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in accepted
            ),
            encoding="utf-8",
        )
        args.rejections.parent.mkdir(parents=True, exist_ok=True)
        args.rejections.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rejected
            ),
            encoding="utf-8",
        )
        write_acquisition_record(
            args.manifest,
            dataset_id="samanjoy2/BnMMLU",
            revision=args.revision,
            split="train",
            config_name="hard",
            row_count=len(accepted),
            artifact_paths=[args.output, args.rejections],
        )
        print(f"INGESTED_OK accepted={len(accepted)} rejected={len(rejected)}")
        return 0
    if args.command == "ingest-ganit":
        rows = list(
            load_pinned_dataset(
                "dipta007/Ganit", args.revision, "dev", "dev"
            )
        )
        accepted, rejected = normalize_ganit_rows(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in accepted
            ),
            encoding="utf-8",
        )
        args.rejections.parent.mkdir(parents=True, exist_ok=True)
        args.rejections.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rejected
            ),
            encoding="utf-8",
        )
        write_acquisition_record(
            args.manifest,
            dataset_id="dipta007/Ganit",
            revision=args.revision,
            split="dev",
            config_name="dev",
            row_count=len(accepted),
            artifact_paths=[args.output, args.rejections],
        )
        print(f"INGESTED_OK accepted={len(accepted)} rejected={len(rejected)}")
        return 0
    if args.command == "ingest-bluck":
        accepted, rejected = read_bluck_directory(args.input_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in accepted
            ),
            encoding="utf-8",
        )
        args.rejections.parent.mkdir(parents=True, exist_ok=True)
        args.rejections.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rejected
            ),
            encoding="utf-8",
        )
        write_acquisition_record(
            args.manifest,
            dataset_id="minhaj1403/bluck",
            revision=args.revision,
            split="bn_dataset",
            config_name=None,
            row_count=len(accepted),
            artifact_paths=[args.output, args.rejections],
        )
        print(f"INGESTED_OK accepted={len(accepted)} rejected={len(rejected)}")
        return 0
    if args.command == "ingest-banglamath":
        accepted, rejected = read_banglamath_csv(args.input_path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in accepted
            ),
            encoding="utf-8",
        )
        args.rejections.parent.mkdir(parents=True, exist_ok=True)
        args.rejections.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rejected
            ),
            encoding="utf-8",
        )
        write_acquisition_record(
            args.manifest,
            dataset_id="TabiaTanzin/BanglaMATH-A-Bangla-benchmark-dataset-for-testing-LLM-mathematical-reasoning-at-grades-6-7-and-8",
            revision=args.revision,
            split="BanglaMath - Bangla_Math_dataset.csv",
            config_name=None,
            row_count=len(accepted),
            artifact_paths=[args.output, args.rejections],
        )
        print(f"INGESTED_OK accepted={len(accepted)} rejected={len(rejected)}")
        return 0
    if args.command == "ingest-bmwp":
        accepted, rejected = read_bmwp_workbook(args.input_path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in accepted
            ),
            encoding="utf-8",
        )
        args.rejections.parent.mkdir(parents=True, exist_ok=True)
        args.rejections.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rejected
            ),
            encoding="utf-8",
        )
        write_acquisition_record(
            args.manifest,
            dataset_id="SanchitaMondal/BMWP",
            revision=args.revision,
            split="Dataset/BMWP_Dataset.xlsx",
            config_name=None,
            row_count=len(accepted),
            artifact_paths=[args.output, args.rejections],
        )
        print(f"INGESTED_OK accepted={len(accepted)} rejected={len(rejected)}")
        return 0
    if args.command == "dedup-build-manifest":
        rows = build_dedup_input_manifest(args.registry, args.root)
        write_jsonl_rows(args.output, rows)
        print(f"DEDUP_MANIFEST_OK rows={len(rows)}")
        return 0
    if args.command == "dedup-preprocess":
        accepted, rejected = preprocess_dedup_rows(read_jsonl_rows(args.input))
        write_jsonl_rows(args.output, accepted)
        write_jsonl_rows(args.rejections, rejected)
        print(
            f"DEDUP_PREPROCESS_OK accepted={len(accepted)} rejected={len(rejected)}"
        )
        return 0
    if args.command == "dedup-validate-source-answers":
        accepted, quarantined = validate_source_answers(read_jsonl_rows(args.input))
        write_jsonl_rows(args.output, accepted)
        write_jsonl_rows(args.quarantined, quarantined)
        print(
            f"DEDUP_SOURCE_VALIDATION_OK accepted={len(accepted)} "
            f"quarantined={len(quarantined)}"
        )
        return 0
    if args.command == "dedup-exact":
        retained, clusters, quarantined = exact_deduplication(
            read_jsonl_rows(args.input)
        )
        write_jsonl_rows(args.retained, retained)
        write_jsonl_rows(args.clusters, clusters)
        write_jsonl_rows(args.quarantined, quarantined)
        if args.summary:
            summary = exact_deduplication_summary(
                read_jsonl_rows(args.input), retained, clusters, quarantined
            )
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        print(
            "DEDUP_EXACT_OK "
            f"retained={len(retained)} clusters={len(clusters)} "
            f"quarantined={len(quarantined)}"
        )
        return 0
    if args.command == "dedup-strict-near":
        retained, clusters = strict_near_deduplication(read_jsonl_rows(args.input))
        write_jsonl_rows(args.retained, retained)
        write_jsonl_rows(args.clusters, clusters)
        print(
            f"DEDUP_STRICT_NEAR_OK retained={len(retained)} "
            f"clusters={len(clusters)}"
        )
        return 0
    if args.command == "dedup-lexical-near":
        retained, clusters = lexical_near_deduplication(
            read_jsonl_rows(args.input), args.threshold
        )
        write_jsonl_rows(args.retained, retained)
        write_jsonl_rows(args.clusters, clusters)
        print(
            f"DEDUP_LEXICAL_NEAR_OK retained={len(retained)} "
            f"clusters={len(clusters)} threshold={args.threshold}"
        )
        return 0
    if args.command == "dedup-mcq-order-invariant":
        retained, clusters, quarantined = mcq_order_invariant_deduplication(
            read_jsonl_rows(args.input)
        )
        write_jsonl_rows(args.retained, retained)
        write_jsonl_rows(args.clusters, clusters)
        write_jsonl_rows(args.quarantined, quarantined)
        print(
            f"DEDUP_MCQ_OK retained={len(retained)} clusters={len(clusters)} "
            f"quarantined={len(quarantined)}"
        )
        return 0
    if args.command == "dedup-semantic":
        input_rows = read_jsonl_rows(args.input)
        retained, clusters, evidence = semantic_deduplicate(
            input_rows,
            model_id=args.model_id,
            model_revision=args.model_revision,
            semantic_threshold=args.threshold,
            top_k=args.top_k,
            require_lexical_evidence=args.require_lexical_evidence,
            fuzzy_threshold=args.fuzzy_threshold,
            jaccard_threshold=args.jaccard_threshold,
        )
        write_jsonl_rows(args.retained, retained)
        write_jsonl_rows(args.clusters, clusters)
        write_jsonl_rows(args.evidence, evidence)
        summary = {
            "input_rows": len(input_rows),
            "retained_rows": len(retained),
            "collapsed_rows": len(input_rows) - len(retained),
            "clusters": len(clusters),
            "candidate_pairs_merged": len(evidence),
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "semantic_threshold": args.threshold,
            "top_k": args.top_k,
            "require_lexical_evidence": args.require_lexical_evidence,
            "fuzzy_threshold": args.fuzzy_threshold,
            "jaccard_threshold": args.jaccard_threshold,
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"DEDUP_SEMANTIC_OK retained={len(retained)} clusters={len(clusters)} "
            f"collapsed={len(input_rows) - len(retained)} threshold={args.threshold}"
        )
        return 0
    if args.command == "promote-semantic-baseline":
        selected = read_jsonl_rows(args.input)
        strict_parent = read_jsonl_rows(args.strict_parent)
        selected_ids = [row["item_id"] for row in selected]
        parent_ids = {row["item_id"] for row in strict_parent}
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("semantic baseline contains duplicate item IDs")
        if not set(selected_ids) <= parent_ids:
            raise ValueError("semantic baseline contains IDs outside strict parent")
        write_jsonl_rows(args.output, selected)
        summary = native_dataset_summary(selected)
        summary["artifact"] = {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
        }
        summary["selection_provenance"] = {
            "method": args.method,
            "strict_parent_path": str(args.strict_parent),
            "strict_parent_sha256": sha256_file(args.strict_parent),
            "strict_parent_rows": len(strict_parent),
            "removed_rows": len(strict_parent) - len(selected),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"SEMANTIC_BASELINE_PROMOTED_OK rows={len(selected)}")
        return 0
    if args.command == "exclude-sources":
        rows = read_jsonl_rows(args.input)
        excluded = set(args.sources)
        retained = [
            row for row in rows
            if row.get("source_provenance", {}).get("source_dataset") not in excluded
        ]
        unexpected = [
            row["item_id"] for row in rows
            if "source_dataset" not in row.get("source_provenance", {})
        ]
        if unexpected:
            raise ValueError("source exclusion input has rows without source_dataset")
        write_jsonl_rows(args.output, retained)
        summary = {
            "input_rows": len(rows),
            "retained_rows": len(retained),
            "excluded_rows": len(rows) - len(retained),
            "excluded_sources": sorted(excluded),
            "input_sha256": sha256_file(args.input),
            "output_sha256": sha256_file(args.output),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"SOURCES_EXCLUDED_OK retained={len(retained)}")
        return 0
    if args.command == "export-release":
        summary = export_release(read_jsonl_rows(args.input), args.parquet, args.workbook)
        summary["input_path"] = str(args.input)
        summary["input_sha256"] = sha256_file(args.input)
        summary["parquet_path"] = str(args.parquet)
        summary["parquet_sha256"] = sha256_file(args.parquet)
        summary["workbook_path"] = str(args.workbook)
        summary["workbook_sha256"] = sha256_file(args.workbook)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"RELEASE_EXPORT_OK rows={summary['rows']}")
        return 0
    if args.command == "canonicalize":
        records = build_canonical_records(read_jsonl_rows(args.input))
        errors: list[str] = []
        for record in records:
            Record.from_dict(record)
            if contains_bengali(record["text"]):
                errors.append(f"{record['row_id']}: canonical text contains Bengali script")
        if errors:
            return _print_errors(errors)
        write_jsonl_rows(args.output, records)
        summary = {
            "input_rows": len(records),
            "canonical_rows": len(records),
            "canonical_engine": CANONICAL_ENGINE,
            "input_path": str(args.input),
            "input_sha256": sha256_file(args.input),
            "output_path": str(args.output),
            "output_sha256": sha256_file(args.output),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"CANONICALIZE_OK rows={len(records)}")
        return 0
    if args.command == "build-llm-variant-batch":
        prompt = args.prompt.read_text(encoding="utf-8")
        items = read_jsonl_rows(args.input)
        requests = build_batch_requests(
            items,
            prompt=prompt,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        written = write_batch_requests(args.output, requests)
        manifest = {
            "input_path": str(args.input),
            "input_sha256": sha256_file(args.input),
            "prompt_path": str(args.prompt),
            "prompt_sha256": sha256_text(prompt),
            "model": args.model,
            "reasoning_effort": "none",
            "output_token_cap": (
                {"type": "fixed", "value": args.max_output_tokens}
                if args.max_output_tokens
                else {
                    "type": "dynamic",
                    "formula": "max(400, ceil(1.8 * native_text_characters + 60))",
                }
            ),
            "variants_per_item": 3,
            "requests": written,
            "batch_jsonl_path": str(args.output),
            "batch_jsonl_sha256": sha256_file(args.output),
            "submission_status": "not_submitted",
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"LLM_BATCH_MANIFEST_OK requests={written}")
        return 0
    if args.command == "validate-llm-variants":
        request_rows = read_jsonl_rows(args.request_batch)
        request_caps = {
            str(row["custom_id"]): row["body"]["max_output_tokens"]
            for row in request_rows
        }
        requested_ids = {custom_id.split(":", 1)[0] for custom_id in request_caps}
        output_rows = read_jsonl_rows(args.batch_output)
        for row in output_rows:
            row["request_max_output_tokens"] = request_caps.get(str(row.get("custom_id", "")))
        accepted, rejected = validate_batch_rows(
            [row for row in read_jsonl_rows(args.items) if row["item_id"] in requested_ids],
            output_rows,
        )
        write_jsonl_rows(args.accepted, accepted)
        write_jsonl_rows(args.rejected, rejected)
        print(f"LLM_VARIANTS_VALIDATED_OK accepted={len(accepted)} rejected={len(rejected)}")
        return 0
    if args.command == "build-llm-retry-batch":
        rejected_rows = read_jsonl_rows(args.rejected)
        if len(rejected_rows) > args.max_items:
            raise ValueError(
                f"retry budget exceeded: {len(rejected_rows)} rejected variants > {args.max_items}"
            )
        items_by_id = {row["item_id"]: row for row in read_jsonl_rows(args.items)}
        missing = sorted({row["item_id"] for row in rejected_rows} - set(items_by_id))
        if missing:
            raise ValueError(f"rejections contain unknown item IDs: {missing[:3]}")
        prompt = args.prompt.read_text(encoding="utf-8")
        requests = build_retry_requests(rejected_rows, items_by_id, read_jsonl_rows(args.accepted), prompt=prompt, model=args.model)
        written = write_batch_requests(args.output, requests)
        manifest = {
            "retry_of_rejections": str(args.rejected), "retained_variants": str(args.accepted),
            "retry_variants": written, "retry_budget_max_variants": args.max_items,
            "model": args.model,
            "prompt_path": str(args.prompt),
            "prompt_sha256": sha256_text(prompt),
            "submission_status": "not_submitted",
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"LLM_RETRY_BATCH_MANIFEST_OK requests={written} budget={args.max_items}")
        return 0
    if args.command == "shard-llm-variant-batch":
        if args.shard_size <= 0:
            raise ValueError("shard size must be positive")
        rows = read_jsonl_rows(args.input)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shards = []
        for index, start in enumerate(range(0, len(rows), args.shard_size), start=1):
            shard_rows = rows[start : start + args.shard_size]
            shard_path = args.output_dir / f"requests_{index:03d}.jsonl"
            write_jsonl_rows(shard_path, shard_rows)
            shards.append({
                "shard_index": index,
                "requests": len(shard_rows),
                "path": str(shard_path),
                "sha256": sha256_file(shard_path),
                "submission_status": "not_submitted",
            })
        manifest = {
            "input_path": str(args.input),
            "input_sha256": sha256_file(args.input),
            "shard_size": args.shard_size,
            "requests": len(rows),
            "shards": shards,
            "submission_status": "not_submitted",
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"LLM_BATCH_SHARDED_OK shards={len(shards)} requests={len(rows)}")
        return 0
    if args.command == "consolidate-llm-variants":
        items_by_id = {row["item_id"]: row for row in read_jsonl_rows(args.items)}
        accepted_rows = []
        for path in sorted(args.batches_root.glob("batch_*/accepted.jsonl")):
            accepted_rows.extend(read_jsonl_rows(path))
        counts = Counter(row["item_id"] for row in accepted_rows)
        generation = json.loads(args.generation_manifest.read_text(encoding="utf-8"))
        enriched = []
        for row in accepted_rows:
            item = items_by_id[row["item_id"]]
            enriched.append({
                "item_id": row["item_id"],
                "variant_id": row["variant_id"],
                "native_text": item["native_text"],
                "romanized_text": row["text"],
                "answer": item["answer"],
                "task_type": item["task_type"],
                "source_provenance": item["source_provenance"],
                "generation": {
                    "model": generation["model"],
                    "prompt_sha256": generation["prompt_sha256"],
                    "validation_status": "accepted",
                },
            })
        enriched.sort(key=lambda row: (row["item_id"], row["variant_id"]))
        complete = [row for row in enriched if counts[row["item_id"]] == 3]
        write_jsonl_rows(args.accepted_output, enriched)
        write_jsonl_rows(args.complete_triplets_output, complete)
        summary = {
            "accepted_variants": len(enriched),
            "items_with_at_least_one_variant": len(counts),
            "complete_triplet_items": sum(count == 3 for count in counts.values()),
            "variant_count_per_item": dict(sorted(Counter(counts.values()).items())),
            "retry_policy": "skipped_by_project_owner",
            "generation_manifest": str(args.generation_manifest),
            "accepted_output": str(args.accepted_output),
            "accepted_output_sha256": sha256_file(args.accepted_output),
            "complete_triplets_output": str(args.complete_triplets_output),
            "complete_triplets_output_sha256": sha256_file(args.complete_triplets_output),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"LLM_VARIANTS_CONSOLIDATED_OK accepted={len(enriched)} complete_triplets={len(complete)}")
        return 0
    if args.command == "build-rule-synthetic":
        catalog = load_rule_catalog(args.catalog)
        complete_rows = read_jsonl_rows(args.llm_complete_triplets)
        complete_ids = {row["item_id"] for row in complete_rows}
        if len(complete_rows) != 3 * len(complete_ids):
            raise ValueError("LLM complete-triplets input must contain exactly three rows per item")
        records, rejected = build_rule_synthetic_records(
            read_jsonl_rows(args.canonical), complete_ids, catalog
        )
        for record in records:
            Record.from_dict(record)
        if rejected:
            raise ValueError(f"rule-synthetic generation rejected {len(rejected)} items")
        if len(records) != len(complete_ids):
            raise ValueError("rule-synthetic generation did not preserve complete-triplet cohort size")
        write_jsonl_rows(args.output, records)
        write_jsonl_rows(args.rejected, rejected)
        manifest = {
            "catalog_path": str(args.catalog),
            "catalog_sha256": sha256_file(args.catalog),
            "catalog_version": catalog.version,
            "selector_seed": catalog.seed,
            "canonical_input": str(args.canonical),
            "canonical_input_sha256": sha256_file(args.canonical),
            "llm_complete_triplets_input": str(args.llm_complete_triplets),
            "llm_complete_triplets_sha256": sha256_file(args.llm_complete_triplets),
            "cohort_items": len(complete_ids),
            "variants_per_item": 1,
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
            "rejected": str(args.rejected),
            "rejected_count": len(rejected),
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"RULE_SYNTHETIC_OK items={len(records)} rejected={len(rejected)}")
        return 0
    if args.command == "export-review-candidate":
        rows = assemble_paired_candidate(
            read_jsonl_rows(args.native),
            read_jsonl_rows(args.canonical),
            read_jsonl_rows(args.llm_complete_triplets),
            read_jsonl_rows(args.rule_synthetic),
        )
        summary = export_review_candidate(rows, args.parquet, args.workbook)
        manifest = {
            "status": "pending_expert_review",
            "native_input": str(args.native),
            "native_input_sha256": sha256_file(args.native),
            "canonical_input": str(args.canonical),
            "canonical_input_sha256": sha256_file(args.canonical),
            "llm_complete_triplets_input": str(args.llm_complete_triplets),
            "llm_complete_triplets_sha256": sha256_file(args.llm_complete_triplets),
            "rule_synthetic_input": str(args.rule_synthetic),
            "rule_synthetic_sha256": sha256_file(args.rule_synthetic),
            "parquet": str(args.parquet),
            "parquet_sha256": sha256_file(args.parquet),
            "workbook": str(args.workbook),
            "workbook_sha256": sha256_file(args.workbook),
            **summary,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"REVIEW_CANDIDATE_EXPORTED_OK items={summary['items']} review_rows={summary['review_rows']}")
        return 0
    if args.command == "build-evaluation-jobs":
        protocol = load_protocol(args.protocol)
        model = json.loads(args.model.read_text(encoding="utf-8"))
        jobs = build_evaluation_jobs(load_parquet_rows(args.dataset), protocol, model)
        inference_jobs, scoring_key = split_inference_and_scoring_jobs(jobs)
        write_jsonl_rows(args.output, inference_jobs)
        write_jsonl_rows(args.scoring_key, scoring_key)
        print(f"EVALUATION_JOBS_OK forms={len(inference_jobs)}")
        return 0
    if args.command == "build-smoke-cohort":
        selected = stratified_smoke_rows(load_parquet_rows(args.dataset), args.per_task, args.seed)
        write_parquet_rows(args.output, selected)
        task_counts = Counter(str(row["task_type"]) for row in selected)
        manifest = {
            "status": "frozen_smoke_cohort",
            "dataset": str(args.dataset),
            "dataset_sha256": sha256_file(args.dataset),
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
            "seed": args.seed,
            "per_task": args.per_task,
            "items": len(selected),
            "forms_per_model": len(selected) * 6,
            "task_counts": dict(sorted(task_counts.items())),
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"SMOKE_COHORT_OK items={len(selected)} forms_per_model={len(selected) * 6}")
        return 0
    if args.command == "build-primary-evaluation-jobs":
        protocol = load_protocol(args.protocol)
        dataset_rows = load_parquet_rows(args.dataset)
        model_config = json.loads(args.models.read_text(encoding="utf-8"))
        selected_models = [model for model in model_config["models"] if model.get("selected")]
        if not selected_models:
            raise ValueError("no primary models are selected")
        args.inference_dir.mkdir(parents=True, exist_ok=True)
        args.scoring_dir.mkdir(parents=True, exist_ok=True)
        run_models = []
        for model in selected_models:
            jobs = build_evaluation_jobs(dataset_rows, protocol, model)
            inference_jobs, scoring_key = split_inference_and_scoring_jobs(jobs)
            slug = re.sub(r"[^a-z0-9]+", "-", model["name"].lower()).strip("-")
            inference_path = args.inference_dir / f"{slug}.jsonl"
            scoring_path = args.scoring_dir / f"{slug}.jsonl"
            write_jsonl_rows(inference_path, inference_jobs)
            write_jsonl_rows(scoring_path, scoring_key)
            run_models.append({
                "name": model["name"], "repository": model["repository"],
                "forms": len(inference_jobs), "inference_jobs": str(inference_path),
                "inference_jobs_sha256": sha256_file(inference_path),
                "scoring_key": str(scoring_path), "scoring_key_sha256": sha256_file(scoring_path),
                "run_event_log_directory": f"artifacts/evaluation/run_logs/{slug}/",
            })
        manifest = {
            "status": "full_run_inputs_ready_no_inference_started",
            "dataset": str(args.dataset), "dataset_sha256": sha256_file(args.dataset),
            "protocol": str(args.protocol), "protocol_sha256": sha256_file(args.protocol),
            "models": run_models,
            "total_forms": sum(record["forms"] for record in run_models),
            "event_log_schema": "artifacts/evaluation/run_event_log_schema_v1.json",
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"PRIMARY_EVALUATION_JOBS_OK models={len(run_models)} forms={manifest['total_forms']}")
        return 0
    if args.command == "freeze-reviewed-release":
        summary = freeze_reviewed_release(args.candidate, args.output)
        manifest = {
            "status": "frozen_after_expert_review",
            "candidate": str(args.candidate),
            "candidate_sha256": sha256_file(args.candidate),
            "release": str(args.output),
            "release_sha256": sha256_file(args.output),
            **summary,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"REVIEWED_RELEASE_FROZEN_OK items={summary['items']} forms={summary['forms']}")
        return 0
    if args.command == "export-hf-task-release":
        summary = export_hf_task_release(args.internal, args.output)
        manifest = {
            "status": "frozen_hf_task_release",
            "internal_source": str(args.internal),
            "internal_source_sha256": sha256_file(args.internal),
            "hf_release": str(args.output),
            "hf_release_sha256": sha256_file(args.output),
            "columns": ["item_id", "task_type", "answer", "native_text", "canonical_text", "llm_generated_variants", "rule_synthetic_text"],
            **summary,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"HF_TASK_RELEASE_OK items={summary['items']} columns={summary['columns']}")
        return 0
    if args.command == "score-evaluation-responses":
        scored = score_responses(read_jsonl_rows(args.jobs), load_response_rows(args.responses))
        write_jsonl_rows(args.output, scored)
        print(f"EVALUATION_SCORED_OK forms={len(scored)}")
        return 0
    if args.command == "analyze-evaluations":
        if args.replicates <= 0:
            raise ValueError("replicates must be positive")
        summary = analysis_summary(read_jsonl_rows(args.scored), args.replicates, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"EVALUATION_ANALYSIS_OK items={summary['items']}")
        return 0
    if args.command == "plan-evaluator-runs":
        plan = build_model_run_plan(args.models, args.protocol, args.cohort_items)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"EVALUATOR_RUN_PLAN_OK ready={len(plan['models_ready_for_final_verification'])} pending={len(plan['models_pending_access_or_tokenizer_verification'])}")
        return 0
    if args.command == "map-tasks":
        mapped, quarantined = map_candidate_tasks(
            read_jsonl_rows(args.input), load_task_mapping(args.mapping)
        )
        write_jsonl_rows(args.output, mapped)
        write_jsonl_rows(args.quarantined, quarantined)
        print(f"TASK_MAPPING_OK mapped={len(mapped)} quarantined={len(quarantined)}")
        return 0
    if args.command == "filter-structural-quality":
        accepted, quarantined = structural_quality_filter(read_jsonl_rows(args.input))
        write_jsonl_rows(args.output, accepted)
        write_jsonl_rows(args.quarantined, quarantined)
        print(
            f"STRUCTURAL_QUALITY_OK accepted={len(accepted)} "
            f"quarantined={len(quarantined)}"
        )
        return 0
    if args.command == "freeze-native":
        frozen = freeze_curated_native(read_jsonl_rows(args.input))
        write_jsonl_rows(args.output, frozen)
        summary = native_dataset_summary(frozen)
        summary["artifact"] = {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"NATIVE_DATASET_FROZEN_OK rows={len(frozen)}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
