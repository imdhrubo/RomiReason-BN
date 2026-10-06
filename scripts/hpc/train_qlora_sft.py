#!/usr/bin/env python3
"""Single-GPU QLoRA SFT for the frozen RomiReason in-domain conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def load_records(path: Path) -> Dataset:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return Dataset.from_list(rows)


def tokenize_record(record: dict, tokenizer, max_length: int) -> dict:
    messages = record["messages"]
    prompt = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    full = prompt + messages[1]["content"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)
    labels = encoded["input_ids"].copy()
    labels[:min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
    encoded["labels"] = labels
    return encoded


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=quantization,
        torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    raw_train = load_records(args.train)
    raw_development = load_records(args.development)
    train = raw_train.map(
        lambda row: tokenize_record(row, tokenizer, args.max_length),
        remove_columns=raw_train.column_names,
        desc="Tokenizing training records",
    )
    development = raw_development.map(
        lambda row: tokenize_record(row, tokenizer, args.max_length),
        remove_columns=raw_development.column_names,
        desc="Tokenizing development records",
    )
    training_args = TrainingArguments(
        output_dir=str(args.output), overwrite_output_dir=False, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, learning_rate=args.learning_rate,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10,
        eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, bf16=True, tf32=True, gradient_checkpointing=True,
        report_to="none", seed=args.seed, data_seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=development,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
    )
    resume_checkpoint = get_last_checkpoint(str(args.output)) if args.output.exists() else None
    if resume_checkpoint is not None:
        print(f"RESUMING_FROM_CHECKPOINT={resume_checkpoint}", flush=True)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(args.output / "best_adapter"))
    tokenizer.save_pretrained(args.output / "best_adapter")
    metadata = {
        # argparse returns Paths for the filesystem arguments.  Persist their
        # string forms so this completion marker is valid JSON.
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "train_records": len(train),
        "development_records": len(development),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "resumed_from_checkpoint": resume_checkpoint,
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
