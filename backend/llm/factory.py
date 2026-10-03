"""Build the configured LLM client. Nothing here runs per frame: the client is only used by
features that answer a person's request (search) or run occasionally."""

from __future__ import annotations

from llm.base import LLMClient, LLMError

PROVIDERS = ("nvidia", "anthropic")
DEFAULT_MODELS = {
    "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",  # the 120B "super" model was retired on 2026-10-03
    "anthropic": "claude-sonnet-5-5",
}


def build_llm(cfg) -> LLMClient:
    """``cfg`` is an ``LLMConfig`` (config/settings.py). Raises LLMError if it can't be used."""
    provider = (cfg.provider or "nvidia").strip().lower()
    model = cfg.model or DEFAULT_MODELS.get(provider, "")
    if provider == "nvidia":
        from llm.openai_compat import NVIDIA_BASE_URL, OpenAICompatibleClient

        if not cfg.nvidia_api_key:
            raise LLMError("LLM_PROVIDER=nvidia needs NVIDIA_API_KEY in .env")
        return OpenAICompatibleClient(model=model, api_key=cfg.nvidia_api_key,
                                      base_url=cfg.base_url or NVIDIA_BASE_URL, provider="nvidia",
                                      timeout_s=cfg.timeout_s)
    if provider == "anthropic":
        from llm.anthropic_client import AnthropicClient

        if not cfg.anthropic_api_key:
            raise LLMError("LLM_PROVIDER=anthropic needs ANTHROPIC_API_KEY in .env")
        return AnthropicClient(model=model, api_key=cfg.anthropic_api_key, effort=cfg.effort or None,
                               timeout_s=cfg.timeout_s)
    raise LLMError(f"unknown LLM_PROVIDER {provider!r}; use one of {', '.join(PROVIDERS)}")
