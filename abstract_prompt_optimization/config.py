"""Configuration models for OTT-QA prompt optimization experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .templates import StrategyTemplate


@dataclass
class DatasetPaths:
    """Paths to the OTT-QA dataset splits."""

    train: Path
    val: Path
    test: Path


@dataclass
class ModelConfig:
    """Language model configuration used by DSPy."""

    provider: str = "openrouter"
    model: str = "openrouter/anthropic/claude-3-haiku"
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout: int = 120
    api_key_env: str = "OPENROUTER_API_KEY"
    http_referer: Optional[str] = None
    app_name: Optional[str] = None


@dataclass
class GEPAConfig:
    """Parameters forwarded to the GEPA teleprompter."""

    population_size: int = 6
    generations: int = 8
    bootstrap_rounds: int = 2
    mutation_rate: float = 0.3
    crossover_rate: float = 0.4
    max_bootstrapped_demos: int = 4
    min_bootstrapped_demos: int = 2
    score_metric: str = "exact_match"
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Configuration for a single optimization strategy."""

    name: str
    kind: str
    prompt_template: str
    interpreter_template: Optional[str] = None
    strategy_template: Optional[StrategyTemplate | str] = None
    gepa: GEPAConfig = field(default_factory=GEPAConfig)
    limit_examples: Optional[int] = None


@dataclass
class ExperimentConfig:
    """Top-level configuration for running an experiment."""

    dataset: DatasetPaths
    model: ModelConfig = field(default_factory=ModelConfig)
    strategies: List[StrategyConfig] = field(default_factory=list)
    max_examples: Optional[int] = None
    output_dir: Path = Path("results")


def load_config(path: Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    dataset = DatasetPaths(**raw["dataset"])

    model_kwargs = {
        key: os.path.expandvars(value) if isinstance(value, str) else value
        for key, value in raw.get("model", {}).items()
    }
    model = ModelConfig(**model_kwargs)

    strategies: List[StrategyConfig] = []
    for item in raw.get("strategies", []):
        item_copy = dict(item)
        gepa_config = item_copy.get("gepa", {})
        strategy_template = item_copy.get("strategy_template")
        if isinstance(strategy_template, dict):
            strategy_template = StrategyTemplate.from_mapping(strategy_template)

        strategy = StrategyConfig(
            name=item_copy["name"],
            kind=item_copy["kind"],
            prompt_template=item_copy["prompt_template"],
            interpreter_template=item_copy.get("interpreter_template"),
            strategy_template=strategy_template,
            gepa=GEPAConfig(**gepa_config),
            limit_examples=item_copy.get("limit_examples"),
        )
        strategies.append(strategy)

    cfg = ExperimentConfig(
        dataset=dataset,
        model=model,
        strategies=strategies,
        max_examples=raw.get("max_examples"),
        output_dir=Path(raw.get("output_dir", "results")),
    )
    return cfg


__all__ = [
    "DatasetPaths",
    "ModelConfig",
    "GEPAConfig",
    "StrategyConfig",
    "ExperimentConfig",
    "load_config",
]
