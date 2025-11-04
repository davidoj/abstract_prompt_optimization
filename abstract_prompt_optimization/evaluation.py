"""Metrics and evaluation helpers for OTT-QA experiments."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .data import OTTQASample


_WORD_SPLIT = re.compile(r"\W+")
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


@dataclass
class Prediction:
    """Model output captured during evaluation."""

    raw: str
    answer: Optional[str]
    evidence_cells: List[str]
    evidence_sentences: List[str]
    latency_s: Optional[float]
    usage: Optional[Dict[str, int]]


@dataclass
class ExampleResult:
    """Per-example evaluation details."""

    example: OTTQASample
    prediction: Prediction
    exact_match: bool
    evidence_precision: Optional[float]
    evidence_recall: Optional[float]
    json_valid: bool


@dataclass
class AggregateMetrics:
    """Aggregated experiment metrics."""

    exact_match: float
    evidence_precision: Optional[float]
    evidence_recall: Optional[float]
    json_valid_rate: float
    avg_latency: Optional[float]


@dataclass
class StrategyReport:
    """Final report returned by a strategy run."""

    name: str
    metrics: AggregateMetrics
    per_example: List[ExampleResult]


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    text = text.lower()
    text = _ARTICLE_RE.sub(" ", text)
    tokens = [tok for tok in _WORD_SPLIT.split(text) if tok]
    return " ".join(tokens)


def compute_exact_match(prediction: str, reference_answers: Sequence[str]) -> bool:
    """Return ``True`` if the prediction matches any reference exactly."""

    if not prediction:
        return False

    norm_prediction = normalize_answer(prediction)
    for reference in reference_answers:
        if normalize_answer(reference) == norm_prediction:
            return True
    return False


def _safe_divide(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_evidence_scores(
    predicted_cells: Sequence[str],
    predicted_sentences: Sequence[str],
    reference: Dict[str, Sequence[str]],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute precision and recall of retrieved evidence."""

    reference_cells = set(reference.get("table_cells", []))
    reference_sentences = set(reference.get("passages", []))

    predicted = set(predicted_cells) | set(predicted_sentences)
    reference_total = reference_cells | reference_sentences

    if not predicted:
        return (0.0 if reference_total else None, 0.0 if reference_total else None)

    true_positive = len(predicted & reference_total)
    precision = _safe_divide(true_positive, len(predicted))
    recall = _safe_divide(true_positive, len(reference_total))
    return precision, recall


def parse_prediction(raw_output: str) -> Tuple[Optional[str], List[str], List[str], bool]:
    """Parse JSON predictions from the model output."""

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return None, [], [], False

    answer = parsed.get("answer")
    evidence_cells = parsed.get("evidence_cells", [])
    evidence_sentences = parsed.get("evidence_sentences", [])

    if isinstance(answer, list):
        answer = ", ".join(str(item) for item in answer)
    if not isinstance(answer, str):
        answer = None

    if not isinstance(evidence_cells, list):
        evidence_cells = []
    if not isinstance(evidence_sentences, list):
        evidence_sentences = []

    evidence_cells = [str(item) for item in evidence_cells]
    evidence_sentences = [str(item) for item in evidence_sentences]
    return answer, evidence_cells, evidence_sentences, True


def aggregate_metrics(results: Iterable[ExampleResult]) -> AggregateMetrics:
    """Compute aggregate metrics from per-example results."""

    results = list(results)
    if not results:
        return AggregateMetrics(0.0, None, None, 0.0, None)

    exact_match_scores = [1.0 if r.exact_match else 0.0 for r in results]
    json_scores = [1.0 if r.json_valid else 0.0 for r in results]

    precision_scores = [r.evidence_precision for r in results if r.evidence_precision is not None]
    recall_scores = [r.evidence_recall for r in results if r.evidence_recall is not None]

    latencies = [r.prediction.latency_s for r in results if r.prediction.latency_s is not None]

    exact_match = statistics.mean(exact_match_scores)
    json_valid_rate = statistics.mean(json_scores)

    evidence_precision = statistics.mean(precision_scores) if precision_scores else None
    evidence_recall = statistics.mean(recall_scores) if recall_scores else None
    avg_latency = statistics.mean(latencies) if latencies else None

    return AggregateMetrics(
        exact_match=exact_match,
        evidence_precision=evidence_precision,
        evidence_recall=evidence_recall,
        json_valid_rate=json_valid_rate,
        avg_latency=avg_latency,
    )


__all__ = [
    "Prediction",
    "ExampleResult",
    "AggregateMetrics",
    "StrategyReport",
    "parse_prediction",
    "compute_exact_match",
    "compute_evidence_scores",
    "aggregate_metrics",
]
