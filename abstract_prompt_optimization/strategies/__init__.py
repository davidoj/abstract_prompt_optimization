"""Optimization strategies for OTT-QA."""

from .base import BaseStrategy
from .direct import DirectPromptStrategy
from .abstract import AbstractPromptStrategy

__all__ = ["BaseStrategy", "DirectPromptStrategy", "AbstractPromptStrategy"]
