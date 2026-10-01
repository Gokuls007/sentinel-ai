"""Anthropic Messages API (official ``anthropic`` SDK). Default model: claude-sonnet-5-5.

Assistant turns are replayed with their original content blocks, so thinking blocks that
precede a tool call go back to the API unchanged, as it requires.
"""

from __future__ import annotations

from typing import Any

from llm.base import LLMError, LLMResponse, Message, ToolCall, ToolSpec, Usage

DEFAULT_MODEL = "claude-sonnet-5-5"


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = "", effort: str | None = "medium",
                 timeout_s: float = 90.0, max_retries: int = 4, client: Any = None):
        if client is None:
            import anthropic

            if not api_key:
                raise LLMError("no API key for provider 'anthropic' (set ANTHROPIC_API_KEY)")
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=max_retries)
        self._client = client
        self.model = model
        self.effort = effort

    def _wire_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                # Results of one assistant turn go back together in a single user message.
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list) \
                        and wire[-1]["content"] and wire[-1]["content"][-1].get("type") == "tool_result":
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
            elif m.role == "user":
                wire.append({"role": "user", "content": m.content})
            elif m.raw is not None and m.raw_provider == self.provider:
                wire.append({"role": "assistant", "content": m.raw})
            else:
                blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in m.tool_calls]
                wire.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        return wire

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec],
             max_tokens: int = 4096) -> LLMResponse:
        import anthropic

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self._wire_messages(messages),
            "thinking": {"type": "adaptive"},
        }
        if system:
            kwargs["system"] = system
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                               for t in tools]
        try:
            resp = self._client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            raise LLMError(f"anthropic returned HTTP {e.status_code}: {str(e)[:300]}") from e
        except anthropic.APIError as e:
            raise LLMError(f"anthropic request failed: {str(e)[:300]}") from e

        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, arguments=b.input if isinstance(b.input, dict) else {},
                          parse_error=None if isinstance(b.input, dict) else "arguments must be a JSON object")
                 for b in resp.content if b.type == "tool_use"]
        stop = {"end_turn": "end", "stop_sequence": "end", "tool_use": "tool_calls",
                "max_tokens": "max_tokens", "refusal": "refusal"}.get(resp.stop_reason, "other")
        if calls:
            stop = "tool_calls"
        u = resp.usage
        usage = Usage((u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0)
                      + (getattr(u, "cache_creation_input_tokens", 0) or 0), u.output_tokens or 0)
        return LLMResponse(message=Message(role="assistant", content=text, tool_calls=calls, raw=resp.content,
                                           raw_provider=self.provider), stop=stop, usage=usage)
