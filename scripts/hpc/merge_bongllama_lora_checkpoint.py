#!/usr/bin/env python3
"""Faithfully merge the published BongLLaMA-13B full LoRA checkpoint.

The published state dictionary stores each adapted linear layer as
``base_layer.weight`` plus LoRA A/B matrices, rather than standard Llama
weight names. vLLM cannot load that representation directly. BongLLaMA's
paper specifies rank 64 and alpha 128 for Llama-2 instruction tuning, so the
merged matrix is W + (B @ A) * (128 / 64).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


LORA_ALPHA = 128
LORA_RANK = 64
SMALL_FILES = (
    "config.json", "generation_config.json", "special_tokens_map.json",
    "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
)


def merge_shard(source: Path, destination: Path) -> list[str]:
    import torch
    from safetensors.torch import save_file

    weights = torch.load(source, map_location="cpu", weights_only=True)
    merged: dict[str, object] = {}
    consumed: set[str] = set()
    for name, value in weights.items():
        suffix = ".base_layer.weight"
        if not name.endswith(suffix):
            if ".lora_A." not in name and ".lora_B." not in name:
                merged[name] = value
            continue
        stem = name[: -len(suffix)]
        a_name = f"{stem}.lora_A.default.weight"
        b_name = f"{stem}.lora_B.default.weight"
        if a_name not in weights or b_name not in weights:
            raise KeyError(f"missing LoRA matrices for {stem}")
        a, b = weights[a_name], weights[b_name]
        if a.shape[0] != LORA_RANK or b.shape[1] != LORA_RANK:
            raise ValueError(f"unexpected LoRA rank for {stem}: {tuple(a.shape)}, {tuple(b.shape)}")
        standard_name = f"{stem}.weight"
        merged[standard_name] = (value.float() + (b.float() @ a.float()) * (LORA_ALPHA / LORA_RANK)).to(value.dtype)
        consumed.update((name, a_name, b_name))
    unexpected = {name for name in weights if name not in consumed and (".lora_A." in name or ".lora_B." in name)}
    if unexpected:
        raise ValueError(f"unmerged LoRA tensors: {sorted(unexpected)[:3]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(merged, destination, metadata={"format": "pt"})
    return list(merged)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Downloaded Hugging Face snapshot directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    index = json.loads((source / "pytorch_model.bin.index.json").read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    output.mkdir(parents=True, exist_ok=True)
    output_map: dict[str, str] = {}
    total = len(shard_names)
    for position, shard_name in enumerate(shard_names, start=1):
        destination_name = f"model-{position:05d}-of-{total:05d}.safetensors"
        keys = merge_shard(source / shard_name, output / destination_name)
        output_map.update({key: destination_name for key in keys})
        print(f"merged {position}/{total}: {shard_name}", flush=True)
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum((output / name).stat().st_size for name in set(output_map.values()))},
                    "weight_map": output_map}, indent=2, sort_keys=True) + "\n"
    )
    for filename in SMALL_FILES:
        shutil.copy2(source / filename, output / filename)
    print(f"merged checkpoint ready: {output}", flush=True)


if __name__ == "__main__":
    main()
