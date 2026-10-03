"""Any OpenAI-compatible chat endpoint: NVIDIA (the default, Nemotron), Groq, Cerebras, vLLM...

Notes from testing nvidia/nemotron-3-super-120b-a12b on integrate.api.nvidia.com (that model was
retired on 2026-10-03; the default is now nvidia/nemotron-3-ultra-550b-a55b):
- ``tool_choice="required"`` returns HTTP 500, so the model always chooses (``auto``).
- Occasional 5xx blips: the SDK retries them (``max_retries``).
- Rarely, a tool call arrives as text (``<tool_call>{...}</tool_call>``) or with junk after
  the tool name; both are repaired here so the agent sees a normal tool call.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from llm.base import LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

_TEXT_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FUNCTION_TAG = re.compile(r"<function=([A-Za-z0-9_.-]+)>\s*(\{.*?\})?\s*</function>", re.DOTALL)
_NAME_JUNK = re.compile(r"[^A-Za-z0-9_-].*$", re.DOTALL)


def _parse_arguments(raw: str | None) -> tuple[dict[str, Any], str | None]:
    if not raw or not raw.strip():
        return {}, None
    try:
        value = json.loads(raw)
    except ValueError as e:
        return {}, f"arguments are not valid JSON ({e.msg})"
    if not isinstance(value, dict):
        return {}, "arguments must be a JSON object"
    return value, None


def clean_tool_name(name: str, known: set[str]) -> str:
    """'count_events<|channel|>commentary' -> 'count_events' when that is a known tool."""
    if name in known:
        return name
    cleaned = _NAME_JUNK.sub("", name.strip())
    return cleaned if cleaned in known else name


def tool_calls_from_text(text: str, known: set[str]) -> tuple[list[ToolCall], str]:
    """Recover tool calls the model wrote into its text. Returns (calls, text without them)."""
    calls: list[ToolCall] = []
    for match in _TEXT_TOOL_CALL.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except ValueError:
            continue
        name = clean_tool_name(str(obj.get("name", "")), known)
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            args, err = _parse_arguments(args)
        else:
            err = None if isinstance(args, dict) else "arguments must be a JSON object"
            args = args if isinstance(args, dict) else {}
        if name in known:
            calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=args, parse_error=err))
    for match in _FUNCTION_TAG.finditer(text):
        name = clean_tool_name(match.group(1), known)
        args, err = _parse_arguments(match.group(2))
        if name in known:
            calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=args, parse_error=err))
    if not calls:
        return [], text
    stripped = _FUNCTION_TAG.sub("", _TEXT_TOOL_CALL.sub("", text)).strip()
    return calls, stripped


class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str = NVIDIA_BASE_URL, provider: str = "nvidia",
                 timeout_s: float = 90.0, max_retries: int = 4, client: Any = None):
        if client is None:
            import openai

            if not api_key:
                raise LLMError(f"no API key for provider {provider!r}")
            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=max_retries)
        self._client = client
        self.provider = provider
        self.model = model

    # --- conversion ---------------------------------------------------------------------

    def _wire_messages(self, system: str, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m.role == "user":
                wire.append({"role": "user", "content": m.content})
            elif m.role == "tool":
                wire.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            else:
                item: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    item["tool_calls"] = [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in m.tool_calls
                    ]
                wire.append(item)
        return wire

    @staticmethod
    def _wire_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}} for t in tools]

    # --- call ---------------------------------------------------------------------------

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec],
             max_tokens: int = 4096) -> LLMResponse:
        import openai

        kwargs: dict[str, Any] = {"model": self.model, "messages": self._wire_messages(system, messages),
                                  "max_completion_tokens": max_tokens}
        if tools:
            kwargs["tools"] = self._wire_tools(tools)
            kwargs["tool_choice"] = "auto"
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except openai.APIStatusError as e:
            raise LLMError(f"{self.provider} returned HTTP {e.status_code}: {_short(e)}") from e
        except openai.APIError as e:  # connection errors, timeouts
            raise LLMError(f"{self.provider} request failed: {_short(e)}") from e
        if not resp.choices:
            raise LLMError(f"{self.provider} returned no choices")

        choice = resp.choices[0]
        msg = choice.message
        known = {t.name for t in tools}
        calls = []
        for c in msg.tool_calls or []:
            args, err = _parse_arguments(c.function.arguments)
            calls.append(ToolCall(id=c.id or f"call_{uuid.uuid4().hex[:12]}",
                                  name=clean_tool_name(c.function.name or "", known), arguments=args,
                                  parse_error=err))
        text = msg.content or ""
        if not calls and known:
            calls, text = tool_calls_from_text(text, known)

        if calls:
            stop = "tool_calls"
        elif choice.finish_reason == "length":
            stop = "max_tokens"
        elif choice.finish_reason == "content_filter":
            stop = "refusal"
        elif choice.finish_reason in ("stop", None):
            stop = "end"
        else:
            stop = "other"
        usage = Usage(getattr(resp.usage, "prompt_tokens", 0) or 0, getattr(resp.usage, "completion_tokens", 0) or 0)
        return LLMResponse(message=Message(role="assistant", content=text, tool_calls=calls,
                                           raw_provider=self.provider), stop=stop, usage=usage)


def _short(e: Exception, limit: int = 300) -> str:
    text = str(e)
    return text if len(text) <= limit else text[:limit] + "..."
