"""Answer plain-English questions about the event log with an LLM and read-only tools.

The agent loop is plain code: ask the model, run the tools it asks for, feed the results
back, stop when it answers or a budget runs out. Every step is yielded as a small dict so
the API can stream progress ("counting falls by zone...") to the dashboard.

Answers must cite the events they rely on as ``[#id]``. Any cited id that no tool returned
in this conversation is removed from the answer and reported, so the dashboard never links
to an event the model made up.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from llm.base import LLMClient, LLMError, Message, Usage
from search.tools import SearchTools, ToolError, tool_specs

CITATION = re.compile(r"\[#(\d+)\]")
BOLD_BARE_ID = re.compile(r"\*\*#(\d+)\*\*")
BARE_ID = re.compile(r"(?<![\[\w#&])#(\d+)\b(?!\])")
MAX_QUESTION_CHARS = 500

SYSTEM_PROMPT = """You answer questions about the event log of Sentinel AI, a safety camera \
system for warehouses and construction sites. Cameras detect events: falls, restricted-zone \
intrusions, overstaying time-limited zones, wrong-direction movement, loitering, risky \
ergonomic postures and near misses. Each event has an id, type, severity, camera, optional \
zone, optional track id (one tracked person on one camera), and start/end times.

How to work:
- Always get facts from the tools. Never guess numbers, times or ids.
- Use count_events for "how many" questions and find_events to list or inspect events.
- Resolve relative dates ("today", "yesterday", "this week", "after 6pm") against the \
current local time below. Weeks start on Monday. "Last week" is the previous Monday-Sunday.
- If the user names a zone loosely ("the dock"), pass it as the zone filter; the tool \
matches names. If it is ambiguous, say which zones exist.
- If nothing matches, say so plainly (for example "No falls were recorded yesterday.").

How to answer:
- Short and direct: the answer first, then at most a few supporting lines.
- When you mention specific events, cite each one as [#id] using ids from tool results.
- Use local times like "Tue 30 Sep, 14:05". Don't mention the tools or JSON.
- Track ids are per camera and are reused over time, so call them "track 12", not "person 12".

Current local time: {now} ({weekday})."""


@dataclass
class SearchResult:
    question: str
    answer: str = ""
    citations: list[int] = field(default_factory=list)
    removed_citations: list[int] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    stop: str = ""  # answered | max_steps | token_budget | llm_error | refusal | max_tokens
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    provider: str = ""
    model: str = ""


def check_citations(answer: str, seen: set[int]) -> tuple[str, list[int], list[int]]:
    """Keep [#id] citations of events the tools returned; strip the rest.

    Models sometimes write a returned event as ``#31`` or ``**#31**`` instead of ``[#31]``.
    Those are normalized to ``[#31]`` when 31 was returned by a tool (so the dashboard
    links them); a bare ``#n`` that no tool returned is left as plain text.
    """
    kept: list[int] = []
    removed: list[int] = []

    def normalize(match: re.Match) -> str:
        event_id = int(match.group(1))
        return f"[#{event_id}]" if event_id in seen else match.group(0)

    answer = BOLD_BARE_ID.sub(normalize, answer or "")
    answer = BARE_ID.sub(normalize, answer)

    def repl(match: re.Match) -> str:
        event_id = int(match.group(1))
        if event_id in seen:
            if event_id not in kept:
                kept.append(event_id)
            return match.group(0)
        if event_id not in removed:
            removed.append(event_id)
        return ""

    cleaned = CITATION.sub(repl, answer)
    cleaned = re.sub(r"[ \t]+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, kept, removed


class SearchAgent:
    def __init__(self, llm: LLMClient, tools: SearchTools, max_steps: int = 6, max_tokens: int = 60_000,
                 max_output_tokens: int = 4096):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_output_tokens = max_output_tokens
        self.specs = tool_specs()

    def ask(self, question: str) -> SearchResult:
        """Run to completion and return the final result."""
        result = SearchResult(question=question)
        for step in self.run(question):
            if step["type"] == "done":
                result = step["result"]
        return result

    def run(self, question: str) -> Iterator[dict[str, Any]]:
        """Yield progress steps; the last one is ``{"type": "done", "result": SearchResult}``."""
        t0 = time.time()
        question = (question or "").strip()[:MAX_QUESTION_CHARS]
        result = SearchResult(question=question, provider=self.llm.provider, model=self.llm.model)
        self.tools.seen_event_ids.clear()
        now = datetime.fromtimestamp(self.tools.now())
        system = SYSTEM_PROMPT.format(now=now.strftime("%Y-%m-%d %H:%M"), weekday=now.strftime("%A"))
        messages: list[Message] = [Message(role="user", content=question)]

        def finish(stop: str, answer: str = "", error: str | None = None) -> dict[str, Any]:
            result.stop, result.error = stop, error
            result.answer, result.citations, result.removed_citations = check_citations(
                answer, self.tools.seen_event_ids)
            result.latency_s = round(time.time() - t0, 3)
            return {"type": "done", "result": result}

        if not question:
            yield finish("llm_error", error="empty question")
            return

        while True:
            last_chance = result.steps >= self.max_steps
            if last_chance:
                messages.append(Message(role="user", content=(
                    "Step limit reached. Answer now from the results you already have, and say "
                    "what you could not check. Do not call any more tools.")))
            try:
                resp = self.llm.chat(system, messages, self.specs, max_tokens=self.max_output_tokens)
            except LLMError as e:
                yield finish("llm_error", error=str(e))
                return
            result.steps += 1
            result.usage.add(resp.usage)
            messages.append(resp.message)

            if not resp.tool_calls:
                if resp.stop == "refusal":
                    yield finish("refusal", resp.text or "The model declined to answer.")
                elif resp.stop == "max_tokens" and not resp.text.strip():
                    yield finish("max_tokens", error="the model ran out of output tokens")
                else:
                    yield finish("answered", resp.text)
                return
            if last_chance:
                yield finish("max_steps", resp.text or "I couldn't finish within the step limit.")
                return

            for call in resp.tool_calls:
                yield {"type": "tool_call", "id": call.id, "name": call.name, "arguments": call.arguments}
                if call.parse_error:
                    output: dict[str, Any] = {"error": call.parse_error}
                else:
                    try:
                        output = self.tools.run(call.name, call.arguments)
                    except (ToolError, TypeError) as e:
                        output = {"error": str(e)}
                result.tool_calls.append({"name": call.name, "arguments": call.arguments,
                                          "error": output.get("error")})
                yield {"type": "tool_result", "id": call.id, "name": call.name, "summary": summarize(output)}
                messages.append(Message(role="tool", tool_call_id=call.id, content=SearchTools.to_text(output)))

            if result.usage.total >= self.max_tokens:
                yield finish("token_budget", error=f"stopped after {result.usage.total} tokens "
                                                   f"(budget {self.max_tokens})")
                return


def summarize(output: dict[str, Any]) -> str:
    """One line for the dashboard's step list."""
    if "error" in output:
        return f"error: {output['error']}"
    if "total_matching" in output:
        return f"{output['returned']} of {output['total_matching']} matching events"
    if "counts" in output:
        return f"{output['total']} events in {len(output['counts'])} groups"
    if "total" in output:
        return f"{output['total']} events"
    if "now" in output:
        return f"{output['weekday']} {output['now'].replace('T', ' ')}"
    if "found" in output:
        return f"event #{output['id']}" if output["found"] else f"event #{output['id']} not found"
    for key in ("zones", "cameras", "types", "severities"):
        if key in output:
            return f"{len(output[key])} {key}"
    return "done"
