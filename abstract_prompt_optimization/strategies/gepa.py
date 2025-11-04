"""Utilities for configuring GEPA with OTT-QA specific judges."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import dspy

from ..config import GEPAConfig
from ..evaluation import compute_evidence_scores, compute_exact_match, parse_prediction


class JudgeFeedback(tuple):
    """Container returned by judges to provide both score and feedback."""

    score: float
    feedback: str

    def __new__(cls, score: float, feedback: str) -> "JudgeFeedback":
        obj = super().__new__(cls, (float(score), str(feedback)))
        obj.score = float(score)
        obj.feedback = str(feedback)
        return obj


def build_ottqa_judge(primary_metric: str):
    """Return a judge callable that scores and critiques model outputs."""

    def judge(example: dspy.Example, prediction: dspy.Prediction, **_: Any) -> JudgeFeedback:
        raw_output = getattr(prediction, "response", None)
        if raw_output is None:
            raw_output = getattr(prediction, "prediction", None)
        if raw_output is None:
            raw_output = str(prediction)

        answer, cells, sentences, json_valid = parse_prediction(raw_output)
        answer = answer or ""

        gold = getattr(example, "response", {}) or {}
        reference_answer = gold.get("answer")
        reference_cells = gold.get("evidence_cells", [])
        reference_sentences = gold.get("evidence_sentences", [])

        references = _as_reference_list(reference_answer)
        em = compute_exact_match(answer, references)
        precision, recall = compute_evidence_scores(
            cells,
            sentences,
            {"table_cells": reference_cells, "passages": reference_sentences},
        )

        score = _select_score(primary_metric, em, precision, recall, json_valid)
        feedback = _compose_feedback(
            score=score,
            em=em,
            json_valid=json_valid,
            answer=answer,
            references=references,
            precision=precision,
            recall=recall,
        )
        return JudgeFeedback(score, feedback)

    return judge


def build_gepa(config: GEPAConfig):
    """Instantiate GEPA with OTT-QA specific configuration."""

    from dspy.teleprompt import GEPA

    options = dict(
        population_size=config.population_size,
        num_generations=config.generations,
        mutation_rate=config.mutation_rate,
        crossover_rate=config.crossover_rate,
        bootstrap_rounds=config.bootstrap_rounds,
        max_bootstrapped_demos=config.max_bootstrapped_demos,
        min_bootstrapped_demos=config.min_bootstrapped_demos,
        judge=build_ottqa_judge(config.score_metric),
        **config.extra_options,
    )
    return GEPA(metric=config.score_metric, **options)


def _select_score(
    primary_metric: str,
    em: bool,
    precision: float | None,
    recall: float | None,
    json_valid: bool,
) -> float:
    metric = primary_metric.lower() if isinstance(primary_metric, str) else "exact_match"
    if metric == "exact_match":
        return 1.0 if em else 0.0
    if metric == "evidence_precision" and precision is not None:
        return float(precision)
    if metric == "evidence_recall" and recall is not None:
        return float(recall)
    if metric == "json_valid":
        return 1.0 if json_valid else 0.0
    # Fallback: prefer EM, otherwise zero.
    return 1.0 if em else 0.0


def _compose_feedback(
    score: float,
    em: bool,
    json_valid: bool,
    answer: str,
    references: Sequence[str],
    precision: float | None,
    recall: float | None,
) -> str:
    issues: list[str] = []
    if not json_valid:
        issues.append("Output was not valid JSON.")
    if not em:
        target = ", ".join(references) if references else "<no reference answer>"
        issues.append(f"Answer mismatch: produced '{answer}' vs. reference '{target}'.")
    if precision is not None and precision < 1.0:
        issues.append(f"Evidence precision {precision:.2f} < 1.00. Cite only supporting cells/sentences.")
    if recall is not None and recall < 1.0:
        issues.append(f"Evidence recall {recall:.2f} < 1.00. Add missing supporting evidence.")

    if not issues:
        return (
            f"Strong response (score={score:.2f}). JSON parsed, answer matched, and evidence coverage"
            " looks solid."
        )
    return " ".join(issues)


def _as_reference_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


__all__ = ["build_gepa", "build_ottqa_judge", "JudgeFeedback"]
