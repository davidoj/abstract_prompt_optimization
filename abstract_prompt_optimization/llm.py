"""DSPy language-model utilities."""

from __future__ import annotations

import os
from typing import Optional

import dspy

from .config import ModelConfig


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def configure_llm(model_config: ModelConfig) -> dspy.LM:
    """Configure DSPy to use the requested language model."""

    if model_config.provider.lower() != "openrouter":
        raise ValueError("Only the OpenRouter provider is currently supported.")

    api_key = os.getenv(model_config.api_key_env)
    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Please export {model_config.api_key_env} with your OpenRouter key."
        )

    headers = {
        "HTTP-Referer": model_config.http_referer
        or os.getenv("OPENROUTER_HTTP_REFERER", "https://example.com"),
        "X-Title": model_config.app_name or os.getenv("OPENROUTER_APP_NAME", "abstract-prompt-optimization"),
    }

    lm = dspy.OpenAI(
        model=model_config.model,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        headers=headers,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        timeout=model_config.request_timeout,
    )
    dspy.settings.configure(lm=lm, trace=False)
    return lm


__all__ = ["configure_llm"]
