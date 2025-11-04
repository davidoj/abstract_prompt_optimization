"""Abstract prompt optimization with a prompt interpreter."""

from __future__ import annotations

from typing import Sequence

import dspy

from ..config import StrategyConfig
from ..data import OTTQASample
from ..templates import render_strategy_template
from .gepa import build_gepa
from .base import BaseStrategy


class PromptInterpreterSignature(dspy.Signature):
    """Interpret a prompting strategy and produce a QA instruction."""

    dataset_profile = dspy.InputField(desc="Description of OTT-QA characteristics")
    strategy = dspy.InputField(desc="High-level prompting strategy")
    strategy_blueprint = dspy.InputField(
        desc="Original template outlining steps, tactics, examples, and failure modes",
    )
    interpreter_instruction = dspy.OutputField(desc="Instruction text for the QA model")


class AbstractQASignature(dspy.Signature):
    """Answer OTT-QA questions using interpreter guidance and evidence."""

    question = dspy.InputField(desc="Natural language question")
    table_context = dspy.InputField(desc="Linearized table context")
    text_context = dspy.InputField(desc="Supporting passages")
    qa_instruction = dspy.InputField(desc="Instruction synthesized by the interpreter")
    response = dspy.OutputField(desc="JSON with `answer`, `evidence_cells`, `evidence_sentences`")


class StrategyPlannerSignature(dspy.Signature):
    """Produce a high-level prompting strategy for OTT-QA."""

    dataset_profile = dspy.InputField(desc="Characteristics of OTT-QA questions")
    strategy_blueprint = dspy.InputField(
        desc=(
            "Structured template enumerating steps, prompt construction tactics, execution "
            "guidance, examples, and failure modes"
        ),
    )
    strategy = dspy.OutputField(desc="High-level prompting strategy description")


class StrategyPlannerModule(dspy.Module):
    """Module responsible for generating strategy text."""

    def __init__(self, prompt: str) -> None:
        super().__init__()
        StrategyPlannerSignature.__doc__ = (
            "Given the OTT-QA profile and a reusable template, craft a concrete prompting "
            "strategy. Respect the template sections, spell out prompt construction moves "
            "(few-shot placement, repetition, pitfalls) and the execution guidance for the "
            "assistant.\n\n"
            f"Strategy template to follow:\n{prompt}"
        )
        self.predict = dspy.Predict(StrategyPlannerSignature)

    def forward(self, dataset_profile: str, blueprint: str) -> str:
        result = self.predict(dataset_profile=dataset_profile, strategy_blueprint=blueprint)
        text = getattr(result, "strategy", None)
        if isinstance(text, str):
            return text
        return str(result)


class PromptInterpreterModule(dspy.Module):
    """Module that translates strategies into QA instructions."""

    def __init__(self, prompt: str, blueprint: str) -> None:
        super().__init__()
        PromptInterpreterSignature.__doc__ = (
            "Translate the high-level strategy into a concrete QA instruction. Preserve the "
            "prompt construction tactics, execution guidance, and failure modes while "
            "reminding the model about JSON format and evidence citation.\n\n"
            f"Reference strategy template:\n{blueprint}"
        )
        self.predict = dspy.Predict(PromptInterpreterSignature)

    def forward(self, dataset_profile: str, strategy_text: str, strategy_blueprint: str) -> str:
        result = self.predict(
            dataset_profile=dataset_profile,
            strategy=strategy_text,
            strategy_blueprint=strategy_blueprint,
        )
        instruction = getattr(result, "interpreter_instruction", None)
        if isinstance(instruction, str):
            return instruction
        return str(result)


class AbstractQAModule(dspy.Module):
    """Two-stage module that first interprets strategies then answers questions."""

    def __init__(self, interpreter_prompt: str, strategy_prompt: str, qa_prompt: str) -> None:
        super().__init__()
        self.strategy_planner = StrategyPlannerModule(strategy_prompt)
        self.interpreter = PromptInterpreterModule(interpreter_prompt, strategy_prompt)
        AbstractQASignature.__doc__ = qa_prompt
        self.answer = dspy.Predict(AbstractQASignature)
        self.dataset_profile = (
            "You are solving OTT-QA tasks that combine Wikipedia tables and passages. "
            "Return JSON with `answer`, `evidence_cells`, and `evidence_sentences`."
        )
        self.strategy_blueprint = strategy_prompt

    def forward(
        self,
        question: str,
        table_context: str,
        text_context: str,
    ) -> dspy.Prediction:
        strategy_text = self.strategy_planner(
            dataset_profile=self.dataset_profile,
            blueprint=self.strategy_blueprint,
        )
        qa_instruction = self.interpreter(
            dataset_profile=self.dataset_profile,
            strategy_text=strategy_text,
            strategy_blueprint=self.strategy_blueprint,
        )
        return self.answer(
            question=question,
            table_context=table_context,
            text_context=text_context,
            qa_instruction=qa_instruction,
        )


class AbstractPromptStrategy(BaseStrategy):
    """Optimize both the prompt interpreter and QA prompts."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(name=config.name)
        if not config.interpreter_template:
            raise ValueError("AbstractPromptStrategy requires `interpreter_template` in the config.")
        if not config.strategy_template:
            raise ValueError("AbstractPromptStrategy requires `strategy_template` in the config.")
        self.interpreter_template = config.interpreter_template
        raw_strategy_template = config.strategy_template
        if raw_strategy_template is None:
            raise ValueError(
                "AbstractPromptStrategy requires `strategy_template` in the config, either as text or a structured template.",
            )
        self.strategy_template = render_strategy_template(raw_strategy_template)
        self.qa_template = config.prompt_template
        self.gepa_config = config.gepa
        self.limit_examples = config.limit_examples

    def compile(self, trainset: Sequence[OTTQASample], valset: Sequence[OTTQASample]) -> dspy.Module:
        module = AbstractQAModule(
            interpreter_prompt=self.interpreter_template,
            strategy_prompt=self.strategy_template,
            qa_prompt=self.qa_template,
        )
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
            qa_instruction="",
            response={
                "answer": ex.answer[0] if ex.answer else "",
                "evidence_cells": ex.evidence.get("table_cells", []),
                "evidence_sentences": ex.evidence.get("passages", []),
            },
        ).with_inputs("question", "table_context", "text_context", "qa_instruction")
        for ex in subset
    ]


__all__ = ["AbstractPromptStrategy"]
