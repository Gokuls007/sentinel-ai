"""A scripted LLM client for agent and API tests (no network, no cost)."""

from __future__ import annotations

from llm.base import LLMError, LLMResponse, Message, ToolCall, Usage


def answer(text: str, stop: str = "end") -> LLMResponse:
    return LLMResponse(Message(role="assistant", content=text), stop=stop, usage=Usage(100, 20))


def calls(*specs: tuple[str, dict], text: str = "") -> LLMResponse:
    tcs = [ToolCall(id=f"c{i}", name=name, arguments=args) for i, (name, args) in enumerate(specs)]
    return LLMResponse(Message(role="assistant", content=text, tool_calls=tcs), stop="tool_calls",
                       usage=Usage(100, 20))


class ScriptedLLM:
    provider = "scripted"
    model = "scripted-1"

    def __init__(self, *responses: LLMResponse | Exception):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def chat(self, system, messages, tools, max_tokens=4096):
        self.requests.append({"system": system, "messages": list(messages), "tools": tools})
        if not self.responses:
            raise LLMError("script exhausted")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
