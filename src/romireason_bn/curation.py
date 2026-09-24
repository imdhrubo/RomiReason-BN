"""Deterministic first-pass curation utilities for multi-source seed pools."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import ast
import difflib
from collections import defaultdict
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def normalize_for_deduplication(text: str) -> str:
    """Normalize Unicode and harmless layout variation without changing meaning."""
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def item_fingerprint(item: dict[str, Any]) -> str:
    """Hash the task input and gold answer used for exact duplicate clustering."""
    fields = (
        normalize_for_deduplication(str(item["text"])),
        normalize_for_deduplication(str(item["answer"])),
        normalize_for_deduplication(str(item["task_type"])),
    )
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()


def exact_duplicate_clusters(items: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group exact normalized duplicates while retaining each source record."""
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        clusters[item_fingerprint(item)].append(item)
    return dict(clusters)


_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_ADDITIONAL_DIGITS = str.maketrans("૦૧૨૩૪૫૬૭૮૯", "0123456789")


def normalize_answer_for_deduplication(answer: str) -> str:
    """Apply only safe answer-format normalization for comparison."""
    return (
        normalize_for_deduplication(answer)
        .translate(_BENGALI_DIGITS)
        .translate(_ADDITIONAL_DIGITS)
    )


def numeric_signature(text: str) -> list[str]:
    """Extract the ordered numeric tokens without interpreting the problem."""
    normalized = normalize_for_deduplication(text).translate(_BENGALI_DIGITS)
    return re.findall(r"\d+(?:[.,]\d+)?", normalized)


def normalize_strict_near_text(text: str) -> str:
    """Remove punctuation-only variation for a high-precision second pass."""
    normalized = normalize_for_deduplication(text)
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
        and not character.isspace()
    )


def stable_dedup_record_id(
    source_dataset: str, source_revision: str, source_item_id: str, line_number: int
) -> str:
    payload = "\x1f".join(
        (source_dataset, source_revision, source_item_id, str(line_number))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def write_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_dedup_input_manifest(
    registry_path: Path, root: Path
) -> list[dict[str, Any]]:
    """Index every enabled normalized artifact without changing source values."""
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    manifest: list[dict[str, Any]] = []
    for source in registry["sources"]:
        if not source.get("enabled"):
            continue
        artifact_name = source.get("normalized_artifact")
        if not artifact_name:
            raise ValueError(f"{source['name']}: no normalized_artifact configured")
        artifact_path = root / artifact_name
        if not artifact_path.exists():
            raise ValueError(f"{source['name']}: missing artifact {artifact_path}")
        actual_sha256 = sha256_file(artifact_path)
        if actual_sha256 != source["sha256"]:
            raise ValueError(
                f"{source['name']}: artifact checksum differs from source registry"
            )
        for line_number, row in enumerate(read_jsonl_rows(artifact_path), start=1):
            source_item_id = str(row.get("source_item_id", ""))
            if not source_item_id:
                raise ValueError(f"{artifact_path}:{line_number}: missing source_item_id")
            manifest.append(
                {
                    **row,
                    "dedup_record_id": stable_dedup_record_id(
                        source["name"], source["revision"], source_item_id, line_number
                    ),
                    "source_revision": source["revision"],
                    "source_artifact": artifact_name,
                    "source_artifact_sha256": actual_sha256,
                    "source_registry_task_purpose": source["purpose"],
                }
            )
    return manifest


def preprocess_dedup_rows(
    rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Add comparison-only fields and explicitly retain malformed records."""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        try:
            text = str(row["text"])
            answer = str(row["answer"])
            if not text.strip() or not answer.strip():
                raise ValueError("empty text or answer")
            accepted.append(
                {
                    **row,
                    "text_normalized": normalize_for_deduplication(text),
                    "text_strict_near_normalized": normalize_strict_near_text(text),
                    "answer_normalized": normalize_answer_for_deduplication(answer),
                    "numeric_signature": numeric_signature(text),
                    "has_latin_mcq_options": bool(
                        re.search(r"(?:^|\n)A\)\s", text)
                        and re.search(r"(?:^|\n)B\)\s", text)
                    ),
                }
            )
        except (KeyError, ValueError) as error:
            rejected.append(
                {
                    "dedup_record_id": row.get("dedup_record_id"),
                    "source_dataset": row.get("source_dataset"),
                    "source_item_id": row.get("source_item_id"),
                    "reason": str(error),
                }
            )
    return accepted, rejected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_deduplication(
    rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge exact same-answer clusters and quarantine exact answer conflicts."""
    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_text[row["text_normalized"]].append(row)

    retained: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for text, members in sorted(by_text.items()):
        ordered = sorted(members, key=lambda row: row["dedup_record_id"])
        answers = sorted({row["answer_normalized"] for row in ordered})
        if len(ordered) == 1:
            retained.append({**ordered[0], "dedup_action": "unique"})
            continue
        cluster_id = "exact-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        base_cluster = {
            "cluster_id": cluster_id,
            "text_normalized": text,
            "member_record_ids": [row["dedup_record_id"] for row in ordered],
            "member_sources": sorted({row["source_dataset"] for row in ordered}),
            "declared_task_types": sorted({row["task_type"] for row in ordered}),
            "answer_normalized_values": answers,
        }
        if len(answers) == 1:
            canonical = ordered[0]
            clusters.append({**base_cluster, "decision": "merge_exact", "canonical_record_id": canonical["dedup_record_id"]})
            retained.append(
                {
                    **canonical,
                    "dedup_action": "exact_merged",
                    "duplicate_cluster_id": cluster_id,
                    "duplicate_member_count": len(ordered),
                    "duplicate_member_record_ids": base_cluster["member_record_ids"],
                }
            )
        else:
            clusters.append({**base_cluster, "decision": "quarantine_answer_conflict", "canonical_record_id": None})
            quarantined.extend(
                {**row, "dedup_action": "quarantine_answer_conflict", "duplicate_cluster_id": cluster_id}
                for row in ordered
            )
    return retained, clusters, quarantined


def exact_deduplication_summary(
    input_rows: Iterable[dict[str, Any]],
    retained: Iterable[dict[str, Any]],
    clusters: Iterable[dict[str, Any]],
    quarantined: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Create a compact, reproducible accounting of exact-pass outcomes."""
    inputs = list(input_rows)
    retained_rows = list(retained)
    cluster_rows = list(clusters)
    quarantined_rows = list(quarantined)
    decisions: dict[str, int] = defaultdict(int)
    cluster_member_total: dict[str, int] = defaultdict(int)
    for cluster in cluster_rows:
        decision = str(cluster["decision"])
        decisions[decision] += 1
        cluster_member_total[decision] += len(cluster["member_record_ids"])
    source_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"input": 0, "retained": 0, "quarantined": 0}
    )
    for row in inputs:
        source_counts[row["source_dataset"]]["input"] += 1
    for row in retained_rows:
        source_counts[row["source_dataset"]]["retained"] += 1
    for row in quarantined_rows:
        source_counts[row["source_dataset"]]["quarantined"] += 1
    return {
        "input_records": len(inputs),
        "retained_records": len(retained_rows),
        "quarantined_records": len(quarantined_rows),
        "collapsed_duplicate_records": len(inputs)
        - len(retained_rows)
        - len(quarantined_rows),
        "clusters_by_decision": dict(sorted(decisions.items())),
        "cluster_members_by_decision": dict(sorted(cluster_member_total.items())),
        "source_counts": dict(sorted(source_counts.items())),
    }


def strict_near_deduplication(
    rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge only punctuation/layout-equivalent records with matching structure."""
    by_key: dict[tuple[str, str, tuple[str, ...], bool], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (
            row["text_strict_near_normalized"],
            row["answer_normalized"],
            tuple(row["numeric_signature"]),
            bool(row["has_latin_mcq_options"]),
        )
        by_key[key].append(row)

    retained: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    for key, members in sorted(by_key.items()):
        ordered = sorted(members, key=lambda row: row["dedup_record_id"])
        if len(ordered) == 1:
            retained.append({**ordered[0], "strict_near_action": "unique"})
            continue
        cluster_id = "strict-near-" + hashlib.sha256(
            "\x1f".join((key[0], key[1], ",".join(key[2]), str(key[3]))).encode("utf-8")
        ).hexdigest()[:16]
        canonical = ordered[0]
        member_source_record_ids = sorted(
            {
                member_id
                for row in ordered
                for member_id in row.get(
                    "duplicate_member_record_ids", [row["dedup_record_id"]]
                )
            }
        )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "decision": "merge_punctuation_layout_equivalent",
                "canonical_record_id": canonical["dedup_record_id"],
                "member_retained_record_ids": [
                    row["dedup_record_id"] for row in ordered
                ],
                "member_source_record_ids": member_source_record_ids,
                "member_sources": sorted(
                    {row["source_dataset"] for row in ordered}
                ),
                "answer_normalized": key[1],
                "numeric_signature": list(key[2]),
                "has_latin_mcq_options": key[3],
            }
        )
        retained.append(
            {
                **canonical,
                "strict_near_action": "merged",
                "strict_near_cluster_id": cluster_id,
                "strict_near_member_count": len(ordered),
                "strict_near_member_record_ids": [
                    row["dedup_record_id"] for row in ordered
                ],
                "strict_near_member_source_record_ids": member_source_record_ids,
            }
        )
    return retained, clusters


def _lexical_anchors(text: str) -> set[str]:
    """Return spaced character anchors for high-recall candidate proposal."""
    if len(text) < 12:
        return {text}
    width = 9
    positions = {
        0,
        max(0, len(text) // 4 - width // 2),
        max(0, len(text) // 2 - width // 2),
        max(0, (3 * len(text)) // 4 - width // 2),
        len(text) - width,
    }
    return {text[position : position + width] for position in positions}


def lexical_near_deduplication(
    rows: Iterable[dict[str, Any]], threshold: float = 0.985
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge only near-identical strings under answer and structure agreement."""
    ordered = sorted(rows, key=lambda row: row["dedup_record_id"])
    indexed: dict[tuple[str, tuple[str, ...], bool, str], list[int]] = defaultdict(list)
    edges: dict[int, set[int]] = defaultdict(set)
    for index, row in enumerate(ordered):
        base = (
            row["answer_normalized"],
            tuple(row["numeric_signature"]),
            bool(row["has_latin_mcq_options"]),
        )
        comparison = row["text_strict_near_normalized"]
        candidates: set[int] = set()
        for anchor in _lexical_anchors(comparison):
            candidates.update(indexed[(base[0], base[1], base[2], anchor)])
        for candidate_index in sorted(candidates):
            candidate = ordered[candidate_index]
            candidate_comparison = candidate["text_strict_near_normalized"]
            if abs(len(comparison) - len(candidate_comparison)) / max(
                len(comparison), len(candidate_comparison), 1
            ) > 1 - threshold:
                continue
            ratio = difflib.SequenceMatcher(
                None, comparison, candidate_comparison, autojunk=False
            ).ratio()
            if ratio >= threshold and comparison != candidate_comparison:
                edges[index].add(candidate_index)
                edges[candidate_index].add(index)
        for anchor in _lexical_anchors(comparison):
            indexed[(base[0], base[1], base[2], anchor)].append(index)

    components: list[list[int]] = []
    seen: set[int] = set()
    for index in range(len(ordered)):
        if index in seen or not edges[index]:
            continue
        stack, component = [index], []
        seen.add(index)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in edges[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))

    component_by_member = {
        member: component for component in components for member in component
    }
    retained: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []
    processed: set[int] = set()
    for index, row in enumerate(ordered):
        if index in processed:
            continue
        component = component_by_member.get(index)
        if component is None:
            retained.append({**row, "lexical_near_action": "unique"})
            processed.add(index)
            continue
        processed.update(component)
        canonical_index = component[0]
        canonical = ordered[canonical_index]
        canonical_text = canonical["text_strict_near_normalized"]
        if not all(
            difflib.SequenceMatcher(
                None, canonical_text, ordered[member]["text_strict_near_normalized"],
                autojunk=False,
            ).ratio()
            >= threshold
            for member in component
        ):
            retained.extend(
                {**ordered[member], "lexical_near_action": "ambiguous_component_kept"}
                for member in component
            )
            continue
        cluster_id = "lexical-near-" + hashlib.sha256(
            "\x1f".join(ordered[member]["dedup_record_id"] for member in component).encode(
                "utf-8"
            )
        ).hexdigest()[:16]
        member_source_record_ids = sorted(
            {
                member_id
                for member in component
                for member_id in ordered[member].get(
                    "strict_near_member_source_record_ids",
                    ordered[member].get(
                        "duplicate_member_record_ids",
                        [ordered[member]["dedup_record_id"]],
                    ),
                )
            }
        )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "decision": "merge_lexical_near_identical",
                "threshold": threshold,
                "canonical_record_id": canonical["dedup_record_id"],
                "member_retained_record_ids": [
                    ordered[member]["dedup_record_id"] for member in component
                ],
                "member_source_record_ids": member_source_record_ids,
                "member_sources": sorted(
                    {ordered[member]["source_dataset"] for member in component}
                ),
            }
        )
        retained.append(
            {
                **canonical,
                "lexical_near_action": "merged",
                "lexical_near_cluster_id": cluster_id,
                "lexical_near_member_count": len(component),
                "lexical_near_member_record_ids": [
                    ordered[member]["dedup_record_id"] for member in component
                ],
                "lexical_near_member_source_record_ids": member_source_record_ids,
            }
        )
    return retained, clusters


def _parse_latin_mcq(text: str, answer: str) -> tuple[str, list[str], str] | None:
    """Extract a question stem, option texts, and the correct option text."""
    answer_label = answer.strip()
    if answer_label not in {"A", "B", "C", "D"}:
        return None
    matches = list(re.finditer(r"(?:^|\n)([A-D])\)\s+", text))
    if len(matches) != 4 or [match.group(1) for match in matches] != ["A", "B", "C", "D"]:
        return None
    stem = text[: matches[0].start()].strip()
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        options[match.group(1)] = text[match.end() : end].strip()
    if not stem or not all(options.values()):
        return None
    return stem, [options[label] for label in ("A", "B", "C", "D")], options[answer_label]


def mcq_order_invariant_deduplication(
    rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge identical MCQs despite option order; quarantine key-answer conflicts."""
    parsed: dict[tuple[str, tuple[str, ...]], list[tuple[dict[str, Any], str]]] = defaultdict(list)
    unparsed: list[dict[str, Any]] = []
    for row in rows:
        mcq = _parse_latin_mcq(str(row["text"]), str(row["answer"]))
        if mcq is None:
            unparsed.append(row)
            continue
        stem, options, correct_option = mcq
        key = (
            normalize_strict_near_text(stem),
            tuple(sorted(normalize_strict_near_text(option) for option in options)),
        )
        parsed[key].append((row, normalize_strict_near_text(correct_option)))

    retained = list(unparsed)
    clusters: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for key, members in sorted(parsed.items(), key=lambda item: item[0]):
        ordered = sorted(members, key=lambda member: member[0]["dedup_record_id"])
        if len(ordered) == 1:
            retained.append({**ordered[0][0], "mcq_dedup_action": "unique"})
            continue
        correct_options = sorted({correct_option for _, correct_option in ordered})
        cluster_id = "mcq-" + hashlib.sha256(
            "\x1f".join((key[0], *key[1])).encode("utf-8")
        ).hexdigest()[:16]
        base_cluster = {
            "cluster_id": cluster_id,
            "member_record_ids": [row["dedup_record_id"] for row, _ in ordered],
            "member_sources": sorted({row["source_dataset"] for row, _ in ordered}),
            "correct_option_values": correct_options,
        }
        if len(correct_options) != 1:
            clusters.append({**base_cluster, "decision": "quarantine_mcq_answer_conflict"})
            quarantined.extend(
                {
                    **row,
                    "dedup_action": "quarantine_mcq_answer_conflict",
                    "mcq_dedup_cluster_id": cluster_id,
                }
                for row, _ in ordered
            )
            continue
        canonical = ordered[0][0]
        clusters.append(
            {
                **base_cluster,
                "decision": "merge_mcq_order_invariant",
                "canonical_record_id": canonical["dedup_record_id"],
            }
        )
        retained.append(
            {
                **canonical,
                "mcq_dedup_action": "merged",
                "mcq_dedup_cluster_id": cluster_id,
                "mcq_dedup_member_record_ids": base_cluster["member_record_ids"],
            }
        )
    return (
        sorted(retained, key=lambda row: row["dedup_record_id"]),
        clusters,
        quarantined,
    )


def _decimal_expression_value(expression: str) -> Decimal | None:
    """Safely evaluate the simple arithmetic notation used by BMWP provenance."""
    clean = (
        expression.translate(_BENGALI_DIGITS)
        .translate(_ADDITIONAL_DIGITS)
        .replace("×", "*")
        .replace("÷", "/")
        .replace(",", "")
        .replace(" ", "")
    )
    if not clean:
        return None
    try:
        tree = ast.parse(clean, mode="eval")
    except SyntaxError:
        return None

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("unsupported arithmetic expression")

    try:
        return evaluate(tree.body)
    except (ArithmeticError, InvalidOperation, ValueError):
        return None


def _decimal_answer_value(answer: str) -> Decimal | None:
    try:
        return Decimal(normalize_answer_for_deduplication(answer).replace(",", ""))
    except InvalidOperation:
        return None


def validate_source_answers(
    rows: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate BMWP's answer against its supplied equation; preserve other rows."""
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for row in rows:
        if row["source_dataset"] != "BMWP":
            accepted.append({**row, "source_answer_validation": "not_applicable"})
            continue
        equation_value = _decimal_expression_value(str(row.get("source_equation", "")))
        answer_value = _decimal_answer_value(str(row["answer"]))
        if equation_value is None or answer_value is None:
            quarantined.append(
                {
                    **row,
                    "dedup_action": "quarantine_unverifiable_source_answer",
                    "source_answer_validation": "unverifiable",
                }
            )
        elif equation_value == answer_value:
            accepted.append(
                {
                    **row,
                    "source_answer_validation": "equation_match",
                    "source_equation_value": str(equation_value),
                }
            )
        else:
            quarantined.append(
                {
                    **row,
                    "dedup_action": "quarantine_source_answer_mismatch",
                    "source_answer_validation": "equation_mismatch",
                    "source_equation_value": str(equation_value),
                }
            )
    return accepted, quarantined
