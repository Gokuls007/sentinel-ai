"""Provider-agnostic chat-with-tools interface.

Everything above this layer (the search agent, later the rules compiler) talks to an
``LLMClient`` and never imports a provider SDK. Each provider module converts these
neutral messages to its own wire format and back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Set when the model sent arguments that aren't a JSON object; the agent reports it back.
    parse_error: str | None = None


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # tool only
    # The provider's own assistant message (e.g. Anthropic content blocks with thinking),
    # replayed verbatim to the same provider so nothing is lost between turns.
    raw: Any = None
    raw_provider: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


StopReason = Literal["end", "tool_calls", "max_tokens", "refusal", "other"]


@dataclass
class LLMResponse:
    message: Message  # role "assistant"; append it to the conversation as is
    stop: StopReason
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return self.message.content

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.message.tool_calls


class LLMError(RuntimeError):
    """The provider call failed after the SDK's own retries (auth, quota, outage, bad request)."""


class LLMClient(Protocol):
    provider: str
    model: str

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec],
             max_tokens: int = 4096) -> LLMResponse: ...
