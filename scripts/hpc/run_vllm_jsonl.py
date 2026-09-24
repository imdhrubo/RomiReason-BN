#!/usr/bin/env python3
"""Run one immutable JSONL shard with vLLM and emit raw responses plus events.

This script receives only answer-key-free inference inputs. It never reads a
scoring key. Run it from the transferred project directory on the HPC system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def read_shard(path: Path, shard_index: int, shard_size: int) -> list[dict]:
    start = shard_index * shard_size
    end = start + shard_size
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= end:
                break
            if index >= start:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing shard: {args.output}")
    rows = read_shard(args.jobs, args.shard_index, args.shard_size)
    if not rows:
        raise ValueError("requested shard has no jobs")
    model = rows[0]["model"]
    if any(row["model"] != model for row in rows):
        raise ValueError("a shard must contain exactly one pinned model")
    if any("expected_answer" in row for row in rows):
        raise ValueError("inference inputs must not contain answer keys")

    job_hash = sha256(args.jobs)
    common = {
        "model_name": model["name"], "model_revision": model["revision"],
        "tokenizer_revision": model["tokenizer_revision"], "job_file_sha256": job_hash,
        "shard_id": args.shard_index,
    }
    append_event(args.event_log, {"timestamp_utc": timestamp(), "event": "run_started", **common,
                                  "runtime_versions": "recorded_by_hpc_runner", "device": "cuda",
                                  "dtype_or_quantization": args.dtype, "total_forms": len(rows)})
    append_event(args.event_log, {"timestamp_utc": timestamp(), "event": "shard_started", **common,
                                  "form_id_start": rows[0]["form_id"], "form_id_end": rows[-1]["form_id"],
                                  "requested_forms": len(rows)})
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(
            model["repository"], revision=model["tokenizer_revision"]
        )
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        ) for row in rows]
        decoding = rows[0]["decoding"]
        if any(row["decoding"] != decoding for row in rows):
            raise ValueError("a shard must contain one decoding configuration")
        llm = LLM(
            model=model["repository"], revision=model["revision"],
            tokenizer=model["repository"], tokenizer_revision=model["tokenizer_revision"],
            tensor_parallel_size=args.tensor_parallel_size, dtype=args.dtype,
        )
        sampling = SamplingParams(
            temperature=decoding["temperature"], top_p=decoding["top_p"],
            max_tokens=decoding["max_output_tokens"],
        )
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        response_rows = [
            {
                "form_id": job["form_id"], "output_text": output.outputs[0].text,
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason,
            }
            for job, output in zip(rows, outputs, strict=True)
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in response_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(args.output)
        append_event(args.event_log, {"timestamp_utc": timestamp(), "event": "shard_completed", **common,
                                      "completed_forms": len(response_rows), "raw_response_file": str(args.output),
                                      "raw_response_sha256": sha256(args.output), "elapsed_seconds": None,
                                      "prompt_tokens": sum(row["prompt_tokens"] for row in response_rows),
                                      "completion_tokens": sum(row["completion_tokens"] for row in response_rows),
                                      "truncation_count": sum(row["finish_reason"] == "length" for row in response_rows)})
    except Exception as error:
        append_event(args.event_log, {"timestamp_utc": timestamp(), "event": "shard_failed", **common,
                                      "error_type": type(error).__name__, "error_message": str(error),
                                      "completed_forms": 0})
        raise


if __name__ == "__main__":
    main()
