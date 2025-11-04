"""Base interfaces for optimization strategies."""

from __future__ import annotations

import abc
import time
from typing import Iterable, List, Sequence

import dspy

from ..data import OTTQASample
from ..evaluation import (
    AggregateMetrics,
    ExampleResult,
    Prediction,
    StrategyReport,
    aggregate_metrics,
    compute_evidence_scores,
    compute_exact_match,
    parse_prediction,
)


class BaseStrategy(abc.ABC):
    """Interface implemented by all optimization strategies."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def compile(self, trainset: Sequence[OTTQASample], valset: Sequence[OTTQASample]) -> dspy.Module:
        """Compile and return a DSPy program ready for inference."""

    def evaluate(
        self,
        program: dspy.Module,
        dataset: Iterable[OTTQASample],
    ) -> StrategyReport:
        """Run the compiled program and collect evaluation metrics."""

        results: List[ExampleResult] = []
        for example in dataset:
            start = time.perf_counter()
            response = program(
                question=example.question,
                table_context=example.table_context,
                text_context=example.text_context,
            )
            latency = time.perf_counter() - start

            raw_output = getattr(response, "response", None)
            if raw_output is None:
                raw_output = getattr(response, "prediction", None)
            if raw_output is None:
                raw_output = str(response)

            answer, cells, sentences, json_valid = parse_prediction(raw_output)
            precision, recall = compute_evidence_scores(cells, sentences, example.evidence)
            em = compute_exact_match(answer or "", example.answer)

            prediction = Prediction(
                raw=raw_output,
                answer=answer,
                evidence_cells=cells,
                evidence_sentences=sentences,
                latency_s=latency,
                usage=getattr(response, "usage", None),
            )
            result = ExampleResult(
                example=example,
                prediction=prediction,
                exact_match=em,
                evidence_precision=precision,
                evidence_recall=recall,
                json_valid=json_valid,
            )
            results.append(result)

        metrics = aggregate_metrics(results)
        return StrategyReport(name=self.name, metrics=metrics, per_example=results)


__all__ = ["BaseStrategy"]
