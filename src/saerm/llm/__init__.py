"""Lightweight interface for invoking Gemini via google-generativeai."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - exercised via unit tests when dependency is available
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled at runtime when package missing
    genai = None

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

_DOTENV_LOADED = False


def _ensure_dotenv_loaded() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED or load_dotenv is None:
        return
    load_dotenv()
    _DOTENV_LOADED = True

Content = Dict[str, Any]


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM client cannot be configured."""


class LLMGenerationError(RuntimeError):
    """Raised when the LLM client cannot complete the generation request."""


class GeminiLLMClient:
    """Minimal wrapper around google-generativeai's GenerativeModel."""

    _DEFAULT_ENV_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def __init__(
        self,
        *,
        model_name: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
    ) -> None:
        if genai is None:
            raise LLMConfigurationError(
                "google-generativeai is not installed. Please add it to your environment."
            )

        self.model_name = model_name
        self.api_key = api_key or self._from_env()

        if not self.api_key:
            raise LLMConfigurationError(
                f"Gemini API key not provided. Set one of {self._DEFAULT_ENV_KEYS} or pass api_key explicitly."
            )

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str, assistant_prompt: Optional[str] = None) -> str:
        """Generate a response for the provided user prompt."""
        contents = self._build_contents(prompt, assistant_prompt)

        try:
            response = self._model.generate_content(contents)
        except Exception as exc:  # pragma: no cover - network failures hard to simulate
            raise LLMGenerationError("Gemini failed to generate a response.") from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMGenerationError("Gemini returned an empty response.")

        return text.strip()

    def _build_contents(
        self, prompt: str, assistant_prompt: Optional[str]
    ) -> List[Content]:
        contents: List[Content] = []

        if assistant_prompt:
            contents.append({"role": "model", "parts": [assistant_prompt]})

        contents.append({"role": "user", "parts": [prompt]})
        return contents

    def _from_env(self) -> Optional[str]:
        _ensure_dotenv_loaded()
        for key in self._DEFAULT_ENV_KEYS:
            value = os.getenv(key)
            if value:
                return value
        return None


_default_client: Optional[GeminiLLMClient] = None


def get_default_client() -> GeminiLLMClient:
    global _default_client
    if _default_client is None:
        _default_client = GeminiLLMClient()
    return _default_client


def generate_llm_response(
    prompt: str,
    assistant_prompt: Optional[str] = None,
    *,
    client: Optional[GeminiLLMClient] = None,
) -> str:
    """Convenience helper that returns the generated text."""
    llm_client = client or get_default_client()
    return llm_client.generate(prompt, assistant_prompt)


__all__ = [
    "GeminiLLMClient",
    "LLMConfigurationError",
    "LLMGenerationError",
    "generate_llm_response",
    "get_default_client",
]
