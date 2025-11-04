"""Experiment runner orchestrating dataset loading and strategy execution."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from .config import ExperimentConfig, StrategyConfig, load_config
from .data import load_split
from .evaluation import StrategyReport
from .llm import configure_llm
from .strategies import AbstractPromptStrategy, BaseStrategy, DirectPromptStrategy


def _instantiate_strategy(config: StrategyConfig) -> BaseStrategy:
    kind = config.kind.lower()
    if kind == "direct":
        return DirectPromptStrategy(config)
    if kind == "abstract":
        return AbstractPromptStrategy(config)
    raise ValueError(f"Unsupported strategy kind: {config.kind}")


def _load_dataset(config: ExperimentConfig):
    max_examples = config.max_examples
    train = load_split(Path(config.dataset.train), max_examples)
    val = load_split(Path(config.dataset.val), max_examples)
    test = load_split(Path(config.dataset.test), max_examples)
    return train, val, test


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_experiment(config_path: Path) -> List[StrategyReport]:
    """Execute all strategies defined in the configuration file."""

    cfg = load_config(config_path)
    configure_llm(cfg.model)

    train, val, test = _load_dataset(cfg)

    reports: List[StrategyReport] = []
    for strategy_cfg in cfg.strategies:
        strategy = _instantiate_strategy(strategy_cfg)
        compiled_program = strategy.compile(train, val)
        report = strategy.evaluate(compiled_program, test)
        reports.append(report)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    _ensure_output_dir(cfg.output_dir)
    output_path = cfg.output_dir / f"reports-{timestamp}.json"
    _serialize_reports(output_path, reports)
    return reports


def _serialize_reports(path: Path, reports: Iterable[StrategyReport]) -> None:
    payload: List[Dict[str, object]] = []
    for report in reports:
        payload.append(
            {
                "strategy": report.name,
                "metrics": {
                    "exact_match": report.metrics.exact_match,
                    "evidence_precision": report.metrics.evidence_precision,
                    "evidence_recall": report.metrics.evidence_recall,
                    "json_valid_rate": report.metrics.json_valid_rate,
                    "avg_latency": report.metrics.avg_latency,
                },
                "predictions": [
                    {
                        "question": example.example.question,
                        "answer": example.prediction.answer,
                        "exact_match": example.exact_match,
                        "evidence_precision": example.evidence_precision,
                        "evidence_recall": example.evidence_recall,
                        "json_valid": example.json_valid,
                        "latency": example.prediction.latency_s,
                        "raw_output": example.prediction.raw,
                        "reference_answers": example.example.answer,
                        "reference_evidence": example.example.evidence,
                        "metadata": example.example.metadata,
                    }
                    for example in report.per_example
                ],
            }
        )

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


__all__ = ["run_experiment"]
