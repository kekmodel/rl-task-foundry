"""DB-agnostic rule-based anchor candidate selection.

The sampler is intentionally not a task oracle. It only chooses diverse,
existing rows that are likely to have enough surrounding data for a composer
to inspect. This breaks the stateless-composer tendency to restart every
episode from the first or smallest id.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rl_task_foundry.schema.graph import ForeignKeyEdge, SchemaGraph, TableProfile
from rl_task_foundry.synthesis.affordance_map import (
    _is_measure_column,
    _is_time_column,
    _table_structure,
    _visible_non_key_columns,
)

_SYSTEM_SCHEMAS = frozenset({
    "information_schema",
    "pg_catalog",
    "pg_toast",
})
PREFERRED_ANCHOR_START_SCORE = 0.60
_NEW_STRUCTURE_WEIGHT_MULTIPLIER = 1.35
_REPEAT_TABLE_WEIGHT_MULTIPLIER = 0.35


@dataclass(frozen=True, slots=True)
class AnchorTableCandidate:
    table: TableProfile
    score: float
    structure: str
    preview_columns: tuple[str, ...]
    incoming_edges: tuple[ForeignKeyEdge, ...]
    outgoing_edges: tuple[ForeignKeyEdge, ...]


def build_anchor_table_candidates(graph: SchemaGraph) -> list[AnchorTableCandidate]:
    """Score every table that can supply a concrete anchor row.

    The algorithm stays schema-general: no table names are hard-coded, row-count
    thresholds are avoided, and weak structural signals are combined instead of
    used as hard gates.
    """

    candidates: list[AnchorTableCandidate] = []
    for table in graph.tables:
        if not _is_anchor_eligible(table):
            continue
        incoming = tuple(graph.edges_to(table.table_name, schema_name=table.schema_name))
        outgoing = tuple(graph.edges_from(table.table_name, schema_name=table.schema_name))
        readable = _visible_non_key_columns(table)
        time_columns = [column for column in readable if _is_time_column(column)]
        numeric_columns = [
            column.column_name for column in readable if _is_measure_column(column)
        ]
        structure = _table_structure(
            graph,
            table,
            readable=readable,
        )
        preview_columns = tuple(column.column_name for column in readable[:4])
        score = _table_score(
            table=table,
            readable_count=len(readable),
            incoming_count=len(incoming),
            outgoing_count=len(outgoing),
            numeric_count=len(numeric_columns),
            time_count=len(time_columns),
        )
        candidates.append(
            AnchorTableCandidate(
                table=table,
                score=score,
                structure=structure,
                preview_columns=preview_columns,
                incoming_edges=incoming,
                outgoing_edges=outgoing,
            )
        )
    return candidates


def select_anchor_tables(
    candidates: list[AnchorTableCandidate],
    *,
    limit: int,
    rng: random.Random | None = None,
) -> list[AnchorTableCandidate]:
    """Select anchor tables with score bias and structural diversity.

    This is a single quality-diversity sampler, not top-N pruning. High-score
    tables get more probability, unseen structural classes receive a small
    bonus, and already selected tables receive a repeat penalty. All eligible
    tables keep a path into the sample, but low-quality structural tables are
    not forced into every episode merely to fill a fixed quota.
    """

    if limit <= 0 or not candidates:
        return []
    rng = rng or random.Random()
    selected: list[AnchorTableCandidate] = []
    while len(selected) < limit:
        chosen = _quality_diversity_choice(
            candidates,
            selected=selected,
            rng=rng,
        )
        selected.append(chosen)

    return selected


def score_anchor_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
    """Score one concrete anchor candidate as an RLVR starting point.

    Scores are heuristic diagnostics, not verifier decisions. They quantify
    whether the candidate breaks first-id collapse and exposes enough visible
    row context or relationship surface for a composer to start exploring.
    """

    preview = candidate.get("preview")
    preview_count = _nonempty_preview_count(preview)
    relation_counts = _positive_relation_counts(candidate.get("relationship_summary"))
    positive_relation_count = len(relation_counts)
    relation_total = sum(relation_counts)
    anti_degeneracy = 0.0 if _id_one_like(candidate.get("row_id")) else 1.0
    visible_surface = min(1.0, preview_count / 2.0)
    relation_surface = _relation_surface_score(
        positive_relation_count=positive_relation_count,
        relation_total=relation_total,
    )
    non_dead = 1.0 if preview_count or positive_relation_count else 0.0
    start_score = (
        0.15 * anti_degeneracy
        + 0.35 * visible_surface
        + 0.35 * relation_surface
        + 0.15 * non_dead
    )
    return {
        "rlvr_start_score": round(start_score, 4),
        "anti_degeneracy_score": round(anti_degeneracy, 4),
        "visible_surface_score": round(visible_surface, 4),
        "relation_surface_score": round(relation_surface, 4),
        "preview_field_count": preview_count,
        "positive_relation_count": positive_relation_count,
        "positive_relation_total": relation_total,
        "dead_anchor": non_dead == 0.0,
        "id_one_like": anti_degeneracy == 0.0,
        "preferred": start_score >= PREFERRED_ANCHOR_START_SCORE,
    }


def summarize_anchor_candidate_pool(
    candidates: Sequence[Mapping[str, object]],
    *,
    eligible_table_count: int | None = None,
) -> dict[str, object]:
    """Aggregate candidate metrics for sampler evaluation."""

    scored = [score_anchor_candidate(candidate) for candidate in candidates]
    count = len(scored)
    if count == 0:
        return {
            "candidate_count": 0,
            "rlvr_start_pool_score": 0.0,
        }
    tables = [
        candidate.get("qualified_table") or candidate.get("table")
        for candidate in candidates
    ]
    table_counts = Counter(table for table in tables if isinstance(table, str))
    unique_tables = len(table_counts)
    table_coverage = (
        unique_tables / eligible_table_count
        if eligible_table_count
        else unique_tables / count
    )
    scores = sorted(float(item["rlvr_start_score"]) for item in scored)
    mean_score = sum(scores) / count
    p10_score = scores[max(0, math.floor((count - 1) * 0.1))]
    entropy = _normalized_entropy(table_counts)
    dead_rate = _rate(scored, "dead_anchor")
    id_one_rate = _rate(scored, "id_one_like")
    preview_rate = sum(
        1 for item in scored if int(item["preview_field_count"]) > 0
    ) / count
    relation_rate = sum(
        1 for item in scored if int(item["positive_relation_count"]) > 0
    ) / count
    preferred_rate = _rate(scored, "preferred")
    pool_score = (
        0.25 * mean_score
        + 0.15 * p10_score
        + 0.15 * (1.0 - dead_rate)
        + 0.10 * (1.0 - id_one_rate)
        + 0.20 * min(1.0, table_coverage)
        + 0.15 * entropy
    )
    return {
        "candidate_count": count,
        "unique_table_count": unique_tables,
        "table_coverage": round(table_coverage, 4),
        "table_entropy": round(entropy, 4),
        "mean_rlvr_start_score": round(mean_score, 4),
        "p10_rlvr_start_score": round(p10_score, 4),
        "preferred_rate": round(preferred_rate, 4),
        "preview_rate": round(preview_rate, 4),
        "positive_relation_rate": round(relation_rate, 4),
        "dead_anchor_rate": round(dead_rate, 4),
        "id_one_like_rate": round(id_one_rate, 4),
        "rlvr_start_pool_score": round(pool_score, 4),
    }


def _is_anchor_eligible(table: TableProfile) -> bool:
    if table.schema_name in _SYSTEM_SCHEMAS:
        return False
    if not table.primary_key:
        return False
    if table.row_estimate == 0:
        return False
    return True


def _table_score(
    *,
    table: TableProfile,
    readable_count: int,
    incoming_count: int,
    outgoing_count: int,
    numeric_count: int,
    time_count: int,
) -> float:
    score = 1.0
    score += min(readable_count, 5) * 0.9
    score += min(incoming_count, 6) * 1.15
    score += min(outgoing_count, 4) * 0.35
    score += min(numeric_count, 3) * 0.45
    score += min(time_count, 3) * 0.45

    fk_count = sum(1 for column in table.columns if column.is_foreign_key)
    if readable_count == 0:
        score -= 1.0
    if fk_count >= 2 and readable_count == 0:
        score -= 1.5
    if table.row_estimate is not None and table.row_estimate < 5:
        score -= 0.4

    return max(0.1, score)


def _weighted_choice(
    candidates: list[AnchorTableCandidate],
    *,
    rng: random.Random,
) -> AnchorTableCandidate:
    total = sum(candidate.score for candidate in candidates)
    if total <= 0:
        return rng.choice(candidates)
    threshold = rng.random() * total
    running = 0.0
    for candidate in candidates:
        running += candidate.score
        if running >= threshold:
            return candidate
    return candidates[-1]


def _quality_diversity_choice(
    candidates: list[AnchorTableCandidate],
    *,
    selected: list[AnchorTableCandidate],
    rng: random.Random,
) -> AnchorTableCandidate:
    selected_keys = {
        (candidate.table.schema_name, candidate.table.table_name)
        for candidate in selected
    }
    selected_structures = {candidate.structure for candidate in selected}
    weighted: list[tuple[AnchorTableCandidate, float]] = []
    for candidate in candidates:
        weight = candidate.score
        key = (candidate.table.schema_name, candidate.table.table_name)
        if candidate.structure not in selected_structures:
            weight *= _NEW_STRUCTURE_WEIGHT_MULTIPLIER
        if key in selected_keys:
            weight *= _REPEAT_TABLE_WEIGHT_MULTIPLIER
        weighted.append((candidate, max(0.01, weight)))
    total = sum(weight for _, weight in weighted)
    threshold = rng.random() * total
    running = 0.0
    for candidate, weight in weighted:
        running += weight
        if running >= threshold:
            return candidate
    return weighted[-1][0]


def _nonempty_preview_count(value: object) -> int:
    if not isinstance(value, Mapping):
        return 0
    count = 0
    for item in value.values():
        if item is None:
            continue
        if isinstance(item, str) and not item.strip():
            continue
        count += 1
    return count


def _positive_relation_counts(value: object) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    counts: list[int] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        raw_count = item.get("count")
        try:
            count = int(raw_count) if raw_count is not None else 0
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            counts.append(count)
    return counts


def _relation_surface_score(
    *,
    positive_relation_count: int,
    relation_total: int,
) -> float:
    if positive_relation_count <= 0 or relation_total <= 0:
        return 0.0
    diversity = min(1.0, positive_relation_count / 2.0)
    mass = min(1.0, math.log1p(relation_total) / math.log1p(20))
    return 0.6 * diversity + 0.4 * mass


def _id_one_like(value: object) -> bool:
    if value == 1 or value == "1":
        return True
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return bool(value) and all(item == 1 or item == "1" for item in value)
    return False


def _rate(items: Sequence[Mapping[str, object]], key: str) -> float:
    if not items:
        return 0.0
    return sum(1 for item in items if item.get(key) is True) / len(items)


def _normalized_entropy(counts: Counter[object]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(len(counts))


__all__ = [
    "AnchorTableCandidate",
    "PREFERRED_ANCHOR_START_SCORE",
    "build_anchor_table_candidates",
    "score_anchor_candidate",
    "select_anchor_tables",
    "summarize_anchor_candidate_pool",
]
