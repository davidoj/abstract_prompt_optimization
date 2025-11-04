"""Direct prompt optimization via GEPA."""

from __future__ import annotations

from typing import Sequence

import dspy

from ..config import StrategyConfig
from ..data import OTTQASample
from .gepa import build_gepa
from .base import BaseStrategy


class DirectQASignature(dspy.Signature):
    """Answer OTT-QA questions using the provided table and passages."""

    question = dspy.InputField(desc="Natural language question")
    table_context = dspy.InputField(desc="Linearized table context")
    text_context = dspy.InputField(desc="Supporting passages")
    response = dspy.OutputField(desc="JSON with `answer`, `evidence_cells`, `evidence_sentences`")


class DirectQAModule(dspy.Module):
    """DSPy module used for direct prompt optimization."""

    def __init__(self, prompt: str) -> None:
        super().__init__()
        DirectQASignature.__doc__ = prompt
        self.predict = dspy.Predict(DirectQASignature)

    def forward(self, question: str, table_context: str, text_context: str) -> dspy.Prediction:
        return self.predict(
            question=question,
            table_context=table_context,
            text_context=text_context,
        )


class DirectPromptStrategy(BaseStrategy):
    """Optimize the QA prompt directly with GEPA."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(name=config.name)
        self.prompt_template = config.prompt_template
        self.gepa_config = config.gepa
        self.limit_examples = config.limit_examples

    def compile(self, trainset: Sequence[OTTQASample], valset: Sequence[OTTQASample]) -> dspy.Module:
        module = DirectQAModule(self.prompt_template)

        teleprompter = build_gepa(self.gepa_config)
        compiled_module = teleprompter.compile(
            module,
            trainset=_to_dspy_examples(trainset, self.limit_examples),
            valset=_to_dspy_examples(valset, self.limit_examples),
        )
        return compiled_module


def _to_dspy_examples(examples: Sequence[OTTQASample], limit: int | None = None):
    subset = examples[:limit] if limit else examples
    return [
        dspy.Example(
            question=ex.question,
            table_context=ex.table_context,
            text_context=ex.text_context,
            response={
                "answer": ex.answer[0] if ex.answer else "",
                "evidence_cells": ex.evidence.get("table_cells", []),
                "evidence_sentences": ex.evidence.get("passages", []),
            },
        ).with_inputs("question", "table_context", "text_context")
        for ex in subset
    ]


__all__ = ["DirectPromptStrategy"]
