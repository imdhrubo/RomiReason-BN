"""Experimental semantic paraphrase deduplication; never mutates strict-v1."""

from __future__ import annotations

import difflib
import hashlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .curation import _parse_latin_mcq, normalize_for_deduplication, numeric_signature


def _semantic_fields(row: dict[str, Any]) -> tuple[str, str, bool, tuple[str, ...]]:
    text = str(row["native_text"])
    parsed = _parse_latin_mcq(text, str(row["answer"]))
    if parsed:
        stem, _, correct_option = parsed
        return (
            normalize_for_deduplication(stem),
            normalize_for_deduplication(correct_option),
            True,
            tuple(numeric_signature(stem)),
        )
    return (
        normalize_for_deduplication(text),
        normalize_for_deduplication(str(row["answer"])),
        False,
        tuple(numeric_signature(text)),
    )


def _jaccard_3gram(first: str, second: str) -> float:
    first_set = {first[index : index + 3] for index in range(max(1, len(first) - 2))}
    second_set = {second[index : index + 3] for index in range(max(1, len(second) - 2))}
    return len(first_set & second_set) / len(first_set | second_set)


def _complete_link_components(
    edges: dict[int, set[int]], ordered_indices: list[int]
) -> list[list[int]]:
    """Partition threshold edges without transitive-chain over-merging.

    A record may join a cluster only when it has a qualifying pairwise edge to
    every existing member.  Thus A~B and B~C cannot silently collapse A and C
    when A~C is not supported.
    """
    unassigned = set(ordered_indices)
    rank = {index: position for position, index in enumerate(ordered_indices)}
    clusters: list[list[int]] = []
    while unassigned:
        seed = min(unassigned, key=rank.__getitem__)
        cluster = [seed]
        unassigned.remove(seed)
        changed = True
        while changed:
            changed = False
            for candidate in sorted(unassigned, key=rank.__getitem__):
                if all(candidate in edges[member] for member in cluster):
                    cluster.append(candidate)
                    unassigned.remove(candidate)
                    changed = True
        clusters.append(cluster)
    return clusters


def semantic_deduplicate(
    rows: Iterable[dict[str, Any]],
    *,
    model_id: str,
    model_revision: str,
    semantic_threshold: float = 0.70,
    top_k: int = 20,
    require_lexical_evidence: bool = False,
    fuzzy_threshold: float = 0.80,
    jaccard_threshold: float = 0.60,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a semantic-dedup branch, clusters, and pair-level evidence.

    The broad mode merges semantically equivalent items at the requested cosine
    threshold.  Consensus mode additionally requires either strong fuzzy or
    character 3-gram overlap.  Both modes retain task, answer, MCQ, and exact
    math-number safeguards.
    """
    from sentence_transformers import SentenceTransformer

    indexed = []
    for row in rows:
        text, answer_key, is_mcq, numbers = _semantic_fields(row)
        indexed.append(
            {**row, "_semantic_text": text, "_answer_key": answer_key,
             "_is_mcq": is_mcq, "_numbers": numbers}
        )
    model = SentenceTransformer(model_id, revision=model_revision)
    embeddings = model.encode(
        [row["_semantic_text"] for row in indexed],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    groups: dict[tuple[str, str, bool], list[int]] = defaultdict(list)
    for index, row in enumerate(indexed):
        groups[(row["task_type"], row["_answer_key"], row["_is_mcq"])].append(index)

    edges: dict[int, set[int]] = defaultdict(set)
    evidence: list[dict[str, Any]] = []
    for group_indices in groups.values():
        if len(group_indices) < 2:
            continue
        matrix = np.asarray(embeddings[group_indices])
        neighbors = NearestNeighbors(
            n_neighbors=min(top_k + 1, len(group_indices)), metric="cosine"
        ).fit(matrix)
        distances, positions = neighbors.kneighbors(matrix)
        for local_index, source_index in enumerate(group_indices):
            for distance, neighbor_position in zip(distances[local_index], positions[local_index]):
                target_index = group_indices[int(neighbor_position)]
                if source_index >= target_index:
                    continue
                source, target = indexed[source_index], indexed[target_index]
                if source["task_type"] == "math_reasoning" and source["_numbers"] != target["_numbers"]:
                    continue
                semantic_score = float(1 - distance)
                if semantic_score < semantic_threshold:
                    continue
                fuzzy_score = difflib.SequenceMatcher(
                    None, source["_semantic_text"], target["_semantic_text"], autojunk=False
                ).ratio()
                jaccard_score = _jaccard_3gram(
                    source["_semantic_text"], target["_semantic_text"]
                )
                lexical_evidence = (
                    fuzzy_score >= fuzzy_threshold
                    or jaccard_score >= jaccard_threshold
                )
                if require_lexical_evidence and not lexical_evidence:
                    continue
                decision = (
                    "merge_consensus_semantic_70"
                    if require_lexical_evidence
                    else "merge_semantic_70"
                )
                edges[source_index].add(target_index)
                edges[target_index].add(source_index)
                evidence.append({
                    "first_item_id": source["item_id"], "second_item_id": target["item_id"],
                    "semantic_cosine": semantic_score, "fuzzy_ratio": fuzzy_score,
                    "char_3gram_jaccard": jaccard_score, "decision": decision,
                })

    retained = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for index, row in enumerate(indexed)
        if not edges.get(index)
    ]
    connected_seen: set[int] = set()
    components: list[list[int]] = []
    active_indices = [index for index, neighbors in edges.items() if neighbors]
    for start in sorted(active_indices, key=lambda index: indexed[index]["item_id"]):
        if start in connected_seen:
            continue
        stack, connected = [start], []
        connected_seen.add(start)
        while stack:
            current = stack.pop()
            connected.append(current)
            for neighbor in edges[current]:
                if neighbor not in connected_seen:
                    connected_seen.add(neighbor)
                    stack.append(neighbor)
        ordered = sorted(connected, key=lambda index: indexed[index]["item_id"])
        components.extend(_complete_link_components(edges, ordered))

    clusters = []
    for component in components:
        component.sort(key=lambda position: indexed[position]["item_id"])
        if len(component) == 1:
            row = indexed[component[0]]
            retained.append({key: value for key, value in row.items() if not key.startswith("_")})
            continue
        canonical = indexed[component[0]]
        mode = "consensus-semantic-70" if require_lexical_evidence else "semantic-70"
        cluster_id = mode + "-" + hashlib.sha256(
            "\x1f".join(indexed[position]["item_id"] for position in component).encode()
        ).hexdigest()[:16]
        clusters.append({
            "cluster_id": cluster_id,
            "decision": (
                "merge_consensus_semantic_70"
                if require_lexical_evidence
                else "merge_semantic_70"
            ),
            "model_id": model_id, "model_revision": model_revision,
            "semantic_threshold": semantic_threshold,
            "require_lexical_evidence": require_lexical_evidence,
            "fuzzy_threshold": fuzzy_threshold,
            "jaccard_threshold": jaccard_threshold,
            "member_item_ids": [indexed[position]["item_id"] for position in component],
            "canonical_item_id": canonical["item_id"],
        })
        prior_provenance = canonical.get("semantic_dedup_provenance")
        current_provenance = {
            "cluster_id": cluster_id,
            "threshold": semantic_threshold,
            "model_id": model_id,
            "model_revision": model_revision,
            "require_lexical_evidence": require_lexical_evidence,
            "fuzzy_threshold": fuzzy_threshold,
            "jaccard_threshold": jaccard_threshold,
            "member_item_ids": [indexed[position]["item_id"] for position in component],
        }
        retained.append({
            **{key: value for key, value in canonical.items() if not key.startswith("_")},
            "semantic_dedup_provenance": {
                **current_provenance,
                **({"prior": prior_provenance} if prior_provenance else {}),
            },
        })
    return retained, clusters, evidence
