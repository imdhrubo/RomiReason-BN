"""Deterministic, auditable sampling for candidate seed pools."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TypeVar


T = TypeVar("T")


def balanced_allocation(total: int, labels: Sequence[str]) -> dict[str, int]:
    """Allocate a total as evenly as possible in the supplied label order."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if not labels:
        raise ValueError("labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    quotient, remainder = divmod(total, len(labels))
    return {label: quotient + (index < remainder) for index, label in enumerate(labels)}


def stratified_sample(
    pools: Mapping[str, Sequence[T]], allocation: Mapping[str, int], seed: int
) -> list[tuple[str, T]]:
    """Sample each named stratum independently using a reproducible seed."""
    if set(pools) != set(allocation):
        raise ValueError("pools and allocation must have the same labels")
    sampler = random.Random(seed)
    selected: list[tuple[str, T]] = []
    for label in allocation:
        count = allocation[label]
        pool = pools[label]
        if count < 0 or count > len(pool):
            raise ValueError(f"invalid sample count for {label}: {count}")
        selected.extend((label, row) for row in sampler.sample(list(pool), count))
    return selected
