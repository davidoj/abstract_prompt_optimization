"""Utilities for rich prompting strategy templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class StrategyExample:
    """An illustrative example embedded in a prompting strategy."""

    title: str
    description: str
    takeaway: Optional[str] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StrategyExample":
        if "title" not in payload or "description" not in payload:
            raise ValueError("Strategy examples require `title` and `description` fields.")
        return cls(
            title=str(payload["title"]),
            description=str(payload["description"]),
            takeaway=str(payload["takeaway"]) if payload.get("takeaway") else None,
        )


@dataclass
class StrategyTemplate:
    """Rich strategy representation consumed by the prompt interpreter.

    The template lets you separate prompt-construction moves (few-shots, repetition,
    formatting reminders) from execution guidance (analysis order, verification plans),
    while still tracking tactics, failure modes, and illustrative examples.
    """

    overview: Optional[str] = None
    steps: List[str] = field(default_factory=list)
    tactics: List[str] = field(default_factory=list)
    prompt_construction: List[str] = field(default_factory=list)
    execution_guidance: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    reminders: List[str] = field(default_factory=list)
    repetition: Optional[str] = None
    examples: List[StrategyExample] = field(default_factory=list)
    custom_sections: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StrategyTemplate":
        template = cls()
        template.overview = _optional_str(payload.get("overview"))
        template.steps = _string_list(payload.get("steps"))
        template.tactics = _string_list(payload.get("tactics"))
        template.prompt_construction = _string_list(payload.get("prompt_construction"))
        template.execution_guidance = _string_list(payload.get("execution_guidance"))
        template.failure_modes = _string_list(payload.get("failure_modes"))
        template.reminders = _string_list(payload.get("reminders"))
        template.repetition = _optional_str(payload.get("repetition"))
        template.examples = [
            StrategyExample.from_mapping(item)
            for item in _mapping_list(payload.get("examples"))
        ]
        template.custom_sections = {
            str(key): str(value)
            for key, value in payload.get("custom_sections", {}).items()
        }
        return template


def render_strategy_template(template: StrategyTemplate | str) -> str:
    """Render a structured strategy template into a textual description."""

    if isinstance(template, str):
        return template

    parts: List[str] = []
    if template.overview:
        parts.append(f"Overview:\n- {template.overview.strip()}")

    if template.steps:
        parts.append(_render_numbered_section("Step-by-step Plan", template.steps))

    if template.tactics:
        parts.append(_render_bullets("Tactics to Emphasize", template.tactics))

    if template.prompt_construction:
        parts.append(
            _render_bullets(
                "Prompt Construction Strategies",
                template.prompt_construction,
            )
        )

    if template.execution_guidance:
        parts.append(
            _render_bullets(
                "Execution Guidance",
                template.execution_guidance,
            )
        )

    if template.failure_modes:
        parts.append(_render_bullets("Failure Modes to Avoid", template.failure_modes))

    if template.reminders:
        parts.append(_render_bullets("General Reminders", template.reminders))

    if template.repetition:
        parts.append(f"Reinforcement:\n- {template.repetition.strip()}")

    if template.examples:
        example_lines: List[str] = []
        for idx, example in enumerate(template.examples, start=1):
            example_lines.append(f"Example {idx}: {example.title}")
            example_lines.append(f"  Description: {example.description}")
            if example.takeaway:
                example_lines.append(f"  Takeaway: {example.takeaway}")
        parts.append("Illustrative Examples:\n" + "\n".join(example_lines))

    for section, body in template.custom_sections.items():
        parts.append(f"{section.strip()}:\n- {body.strip()}")

    return "\n\n".join(parts)


def _render_numbered_section(title: str, items: List[str]) -> str:
    lines = [f"{idx}. {item}" for idx, item in enumerate(items, start=1)]
    return f"{title}:\n" + "\n".join(lines)


def _render_bullets(title: str, items: List[str]) -> str:
    lines = [f"- {item}" for item in items]
    return f"{title}:\n" + "\n".join(lines)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError("Expected a list of strings for strategy template fields.")


def _mapping_list(value: Any) -> List[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Iterable):
        result: List[Mapping[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                result.append(item)
            else:
                raise TypeError("Strategy examples must be mappings with text fields.")
        return result
    raise TypeError("Strategy examples must be provided as mappings or a list of mappings.")


__all__ = ["StrategyTemplate", "StrategyExample", "render_strategy_template"]
