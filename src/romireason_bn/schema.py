"""Dataset schema and record-level validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Collection(str, Enum):
    CORE_PAIRED = "core_paired"
    ATTESTED_TRANSFER = "attested_transfer"


class Condition(str, Enum):
    NATIVE = "native"
    CANONICAL = "canonical"
    LLM_GENERATED_VARIANT = "llm_generated_variant"
    SYNTHETIC_VARIANT = "synthetic_variant"
    ATTESTED_NATURAL = "attested_natural"


class Provenance(str, Enum):
    BENCHMARK_SOURCE = "benchmark_source"
    DETERMINISTIC = "deterministic"
    LLM_GENERATED = "llm_generated"
    SYNTHETIC = "synthetic"
    ATTESTED = "attested"


class TaskType(str, Enum):
    BASIC_UNDERSTANDING = "basic_understanding"
    FACTUAL_QA = "factual_qa"
    LOGICAL_REASONING = "logical_reasoning"
    MATH_REASONING = "math_reasoning"


EXPECTED_PROVENANCE = {
    Condition.NATIVE: Provenance.BENCHMARK_SOURCE,
    Condition.CANONICAL: Provenance.DETERMINISTIC,
    Condition.LLM_GENERATED_VARIANT: Provenance.LLM_GENERATED,
    Condition.SYNTHETIC_VARIANT: Provenance.SYNTHETIC,
    Condition.ATTESTED_NATURAL: Provenance.ATTESTED,
}


@dataclass(frozen=True)
class Record:
    row_id: str
    item_id: str
    collection: Collection
    task_type: TaskType
    condition: Condition
    text: str
    answer: str
    source_dataset: str
    source_item_id: str
    provenance: Provenance
    meaning_preserved: bool | None
    answer_preserved: bool | None
    dialectal: bool
    code_mixed: bool
    canonical_engine: str | None
    variant_rule_ids: tuple[str, ...]
    generator_model: str | None
    generator_prompt_hash: str | None
    generator_seed: int | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Record":
        required = {
            "row_id",
            "item_id",
            "collection",
            "task_type",
            "condition",
            "text",
            "answer",
            "source_dataset",
            "source_item_id",
            "provenance",
            "dialectal",
            "code_mixed",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")

        record = cls(
            row_id=str(raw["row_id"]).strip(),
            item_id=str(raw["item_id"]).strip(),
            collection=Collection(raw["collection"]),
            task_type=TaskType(raw["task_type"]),
            condition=Condition(raw["condition"]),
            text=str(raw["text"]).strip(),
            answer=str(raw["answer"]).strip(),
            source_dataset=str(raw["source_dataset"]).strip(),
            source_item_id=str(raw["source_item_id"]).strip(),
            provenance=Provenance(raw["provenance"]),
            meaning_preserved=raw.get("meaning_preserved"),
            answer_preserved=raw.get("answer_preserved"),
            dialectal=raw["dialectal"],
            code_mixed=raw["code_mixed"],
            canonical_engine=_optional_string(raw.get("canonical_engine")),
            variant_rule_ids=tuple(raw.get("variant_rule_ids", [])),
            generator_model=_optional_string(raw.get("generator_model")),
            generator_prompt_hash=_optional_string(raw.get("generator_prompt_hash")),
            generator_seed=raw.get("generator_seed"),
        )
        record.validate()
        return record

    def validate(self) -> None:
        for field_name in (
            "row_id",
            "item_id",
            "text",
            "answer",
            "source_dataset",
            "source_item_id",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be non-empty")

        if type(self.dialectal) is not bool or type(self.code_mixed) is not bool:
            raise ValueError("dialectal and code_mixed must be booleans")
        if self.meaning_preserved is not None and type(self.meaning_preserved) is not bool:
            raise ValueError("meaning_preserved must be boolean or null")
        if self.answer_preserved is not None and type(self.answer_preserved) is not bool:
            raise ValueError("answer_preserved must be boolean or null")
        if self.provenance != EXPECTED_PROVENANCE[self.condition]:
            raise ValueError(
                f"condition {self.condition.value} requires provenance "
                f"{EXPECTED_PROVENANCE[self.condition].value}"
            )

        if self.collection is Collection.CORE_PAIRED:
            if self.condition is Condition.ATTESTED_NATURAL:
                raise ValueError("attested_natural rows cannot enter core_paired")
            if self.condition is not Condition.NATIVE:
                if self.meaning_preserved is not True or self.answer_preserved is not True:
                    raise ValueError("non-native core rows require preservation=true")
        elif self.condition is not Condition.ATTESTED_NATURAL:
            raise ValueError("attested_transfer rows require attested_natural condition")

        if self.condition is Condition.CANONICAL and not self.canonical_engine:
            raise ValueError("canonical rows require canonical_engine")
        if self.condition is Condition.SYNTHETIC_VARIANT and not self.variant_rule_ids:
            raise ValueError("synthetic rows require variant_rule_ids")
        if self.condition is Condition.LLM_GENERATED_VARIANT:
            if not self.generator_model or not self.generator_prompt_hash:
                raise ValueError("LLM-generated rows require generator_model and generator_prompt_hash")
            if type(self.generator_seed) is not int:
                raise ValueError("LLM-generated rows require integer generator_seed")


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None
