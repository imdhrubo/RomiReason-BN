#!/usr/bin/env python3
"""Run immutable JSONL shard(s) with Transformers and a PEFT adapter.

This is the compatibility runner for adapter/model pairs that vLLM cannot
serve.  Its input and output contracts intentionally match run_vllm_jsonl.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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


def shard_count(path: Path, shard_size: int) -> int:
    with path.open(encoding="utf-8") as handle:
        total = sum(1 for line in handle if line.strip())
    return (total + shard_size - 1) // shard_size


def render_prompts(tokenizer, rows: list[dict]) -> list[str]:
    prompt_renderer = rows[0].get("prompt_renderer", "chat_template_v1")
    if any(row.get("prompt_renderer", "chat_template_v1") != prompt_renderer for row in rows):
        raise ValueError("a shard must contain one prompt renderer")
    if prompt_renderer == "bangla_instruction_response_v1":
        return [f"### Instruction:\n{row['prompt']}\n\n### Response:\n" for row in rows]
    if prompt_renderer != "chat_template_v1":
        raise ValueError(f"unknown prompt renderer: {prompt_renderer}")
    chat_template_kwargs = rows[0].get("chat_template_kwargs", {})
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object")
    if any(row.get("chat_template_kwargs", {}) != chat_template_kwargs for row in rows):
        raise ValueError("a shard must contain one chat-template configuration")
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--all-shards", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--event-dir", type=Path)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-stride", type=int, default=1)
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--lora-name", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.shard_start < 0 or args.shard_stride < 1 or args.batch_size < 1:
        parser.error("shard start, stride, and batch size must be positive")
    if args.all_shards:
        if args.shard_index is not None or args.output or args.event_log or not args.output_dir or not args.event_dir:
            parser.error("--all-shards requires --output-dir and --event-dir, without single-shard arguments")
        indices = range(args.shard_start, shard_count(args.jobs, args.shard_size), args.shard_stride)
    else:
        if args.shard_index is None or not args.output or not args.event_log:
            parser.error("a single shard requires --shard-index, --output, and --event-log")
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing shard: {args.output}")
        indices = [args.shard_index]

    first_rows = read_shard(args.jobs, 0, args.shard_size)
    if not first_rows:
        raise ValueError("job file has no rows")
    model_spec = first_rows[0]["model"]
    if not str(model_spec["repository"]).startswith("CohereLabs/aya-expanse-"):
        raise ValueError("this compatibility runner is restricted to Aya Expanse models")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.dtype != "bfloat16":
        raise ValueError("only bfloat16 is supported by this runner")
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["repository"], revision=model_spec["tokenizer_revision"]
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_spec["repository"],
        revision=model_spec["revision"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()
    input_device = model.get_input_embeddings().weight.device
    job_hash = sha256(args.jobs)

    for shard_index in indices:
        output = args.output if not args.all_shards else args.output_dir / f"shard-{shard_index:05d}.jsonl"
        event_log = args.event_log if not args.all_shards else args.event_dir / f"shard-{shard_index:05d}.jsonl"
        if output.exists():
            print(f"SKIP completed shard {shard_index}: {output}")
            continue
        rows = read_shard(args.jobs, shard_index, args.shard_size)
        if not rows:
            raise ValueError(f"requested shard {shard_index} has no jobs")
        if any(row["model"] != model_spec for row in rows):
            raise ValueError("a shard must contain exactly one pinned model")
        if any("expected_answer" in row for row in rows):
            raise ValueError("inference inputs must not contain answer keys")
        decoding = rows[0]["decoding"]
        if any(row["decoding"] != decoding for row in rows):
            raise ValueError("a shard must contain one decoding configuration")
        if decoding["temperature"] != 0 or decoding["top_p"] != 1:
            raise ValueError("this deterministic runner requires temperature=0 and top_p=1")
        common = {
            "model_name": model_spec["name"], "model_revision": model_spec["revision"],
            "tokenizer_revision": model_spec["tokenizer_revision"], "job_file_sha256": job_hash,
            "shard_id": shard_index, "lora_name": args.lora_name, "lora_path": str(args.lora_path),
            "lora_adapter_config_sha256": sha256(args.lora_path / "adapter_config.json"),
        }
        append_event(event_log, {"timestamp_utc": timestamp(), "event": "run_started", **common,
                                "runtime_versions": "transformers_peft", "device": "cuda",
                                "dtype_or_quantization": args.dtype, "total_forms": len(rows)})
        append_event(event_log, {"timestamp_utc": timestamp(), "event": "shard_started", **common,
                                "form_id_start": rows[0]["form_id"], "form_id_end": rows[-1]["form_id"],
                                "requested_forms": len(rows)})
        started = time.monotonic()
        try:
            prompts = render_prompts(tokenizer, rows)
            response_rows = []
            for start in range(0, len(rows), args.batch_size):
                batch_rows = rows[start:start + args.batch_size]
                encoded = tokenizer(prompts[start:start + args.batch_size], return_tensors="pt", padding=True)
                encoded = {key: value.to(input_device) for key, value in encoded.items()}
                prompt_width = encoded["input_ids"].shape[1]
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=decoding["max_output_tokens"],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                completions = generated[:, prompt_width:]
                for batch_index, (job, tokens) in enumerate(zip(batch_rows, completions, strict=True)):
                    token_list = tokens.tolist()
                    response_rows.append({
                        "form_id": job["form_id"],
                        "output_text": tokenizer.decode(token_list, skip_special_tokens=True),
                        "prompt_tokens": int(encoded["attention_mask"][batch_index].sum()),
                        "completion_tokens": len(token_list),
                        "finish_reason": "length" if len(token_list) == decoding["max_output_tokens"] else "stop",
                    })
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".partial")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in response_rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            temporary.replace(output)
            append_event(event_log, {"timestamp_utc": timestamp(), "event": "shard_completed", **common,
                                    "completed_forms": len(response_rows), "raw_response_file": str(output),
                                    "raw_response_sha256": sha256(output),
                                    "elapsed_seconds": time.monotonic() - started,
                                    "prompt_tokens": sum(row["prompt_tokens"] for row in response_rows),
                                    "completion_tokens": sum(row["completion_tokens"] for row in response_rows),
                                    "truncation_count": sum(row["finish_reason"] == "length" for row in response_rows)})
        except Exception as error:
            append_event(event_log, {"timestamp_utc": timestamp(), "event": "shard_failed", **common,
                                    "error_type": type(error).__name__, "error_message": str(error),
                                    "completed_forms": 0})
            raise


if __name__ == "__main__":
    main()
