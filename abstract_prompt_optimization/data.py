"""Utilities for loading OTT-QA style datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class OTTQASample:
    """A single OTT-QA example."""

    question: str
    answer: List[str]
    table_context: str
    text_context: str
    evidence: Dict[str, List[str]]
    metadata: Dict[str, object]


def load_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    """Yield dictionaries from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _stringify_table(table: Dict[str, object]) -> str:
    header = table.get("header", [])
    rows = table.get("rows", [])
    title = table.get("title")

    parts: List[str] = []
    if title:
        parts.append(f"[Table: {title}]")

    if header:
        parts.append(" | ".join(str(h) for h in header))

    for idx, row in enumerate(rows):
        parts.append(f"Row {idx + 1}: " + " | ".join(str(cell) for cell in row))

    return "\n".join(parts)


def _stringify_passages(passages: Iterable[Dict[str, object]]) -> str:
    parts: List[str] = []
    for passage in passages:
        title = passage.get("title", "")
        text = passage.get("text", "")
        if title:
            parts.append(f"[{title}] {text}")
        else:
            parts.append(text)
    return "\n".join(parts)


def load_split(path: Path, limit: Optional[int] = None) -> List[OTTQASample]:
    """Load a dataset split exported from OTT-QA."""

    examples: List[OTTQASample] = []
    for raw in load_jsonl(path):
        answer = raw.get("answer", [])
        if isinstance(answer, str):
            answer = [answer]

        table_context = ""
        text_context = ""
        evidence = {
            "table_cells": raw.get("table_evidence", []),
            "passages": raw.get("passage_evidence", []),
        }

        table = raw.get("table", {}) or {}
        passages = raw.get("passages", []) or []

        if table:
            table_context = _stringify_table(table)
        if passages:
            text_context = _stringify_passages(passages)

        metadata = {
            "id": raw.get("id"),
            "question_id": raw.get("qid"),
            "question_type": raw.get("type"),
            "table_title": table.get("title"),
        }

        examples.append(
            OTTQASample(
                question=raw["question"],
                answer=answer,
                table_context=table_context,
                text_context=text_context,
                evidence=evidence,
                metadata=metadata,
            )
        )

        if limit is not None and len(examples) >= limit:
            break

    return examples


__all__ = [
    "OTTQASample",
    "load_split",
]
