"""Provider clients with fake SDK objects: wire format both ways, repairs, errors, factory."""

import json
from types import SimpleNamespace as NS

import pytest

from config.settings import LLMConfig, SentinelConfig
from llm import DEFAULT_MODELS, LLMError, Message, ToolCall, ToolSpec, build_llm
from llm.anthropic_client import AnthropicClient
from llm.openai_compat import OpenAICompatibleClient, clean_tool_name, tool_calls_from_text

TOOLS = [ToolSpec("count_events", "Count.", {"type": "object", "properties": {"group_by": {"type": "string"}}}),
         ToolSpec("current_time", "Now.", {"type": "object", "properties": {}})]
KNOWN = {t.name for t in TOOLS}


class Recorder:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


# --- OpenAI-compatible (NVIDIA) -------------------------------------------------------------

def oa_response(content="", tool_calls=None, finish="stop"):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls), finish_reason=finish)],
              usage=NS(prompt_tokens=50, completion_tokens=7))


def oa_client(response=None, error=None):
    rec = Recorder(response, error)
    return OpenAICompatibleClient("nvidia/nemotron-3-super-120b-a12b", "", client=NS(chat=NS(completions=rec))), rec


def test_openai_wire_format_and_tool_call_parsing():
    call = NS(id="call_1", function=NS(name="count_events", arguments='{"group_by": "zone"}'))
    client, rec = oa_client(oa_response(tool_calls=[call], finish="tool_calls"))
    history = [
        Message(role="user", content="q"),
        Message(role="assistant", tool_calls=[ToolCall("call_0", "current_time", {})]),
        Message(role="tool", tool_call_id="call_0", content='{"now":"x"}'),
    ]
    r = client.chat("sys", history, TOOLS, max_tokens=321)
    kw = rec.kwargs
    assert kw["tool_choice"] == "auto" and kw["max_completion_tokens"] == 321  # "required" 500s on NVIDIA
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["messages"][2]["tool_calls"][0]["function"] == {"name": "current_time", "arguments": "{}"}
    assert kw["messages"][3] == {"role": "tool", "tool_call_id": "call_0", "content": '{"now":"x"}'}
    assert kw["tools"][0]["function"]["name"] == "count_events"
    assert r.stop == "tool_calls" and r.tool_calls[0].arguments == {"group_by": "zone"}
    assert (r.usage.input_tokens, r.usage.output_tokens) == (50, 7)


def test_openai_bad_arguments_and_junk_names_are_repaired_or_flagged():
    calls = [NS(id="a", function=NS(name="count_events<|channel|>commentary", arguments="{not json")),
             NS(id="b", function=NS(name="current_time", arguments="[1]"))]
    client, _ = oa_client(oa_response(tool_calls=calls, finish="tool_calls"))
    r = client.chat("", [Message(role="user", content="q")], TOOLS)
    assert r.tool_calls[0].name == "count_events" and "not valid JSON" in r.tool_calls[0].parse_error
    assert r.tool_calls[1].parse_error == "arguments must be a JSON object"


def test_openai_text_tool_calls_are_recovered():
    text = 'Let me check. <tool_call>{"name": "count_events", "arguments": {"group_by": "type"}}</tool_call>'
    client, _ = oa_client(oa_response(content=text))
    r = client.chat("", [Message(role="user", content="q")], TOOLS)
    assert r.stop == "tool_calls" and r.tool_calls[0].arguments == {"group_by": "type"}
    assert r.text == "Let me check."
    calls, rest = tool_calls_from_text("<function=current_time>{}</function>", KNOWN)
    assert [c.name for c in calls] == ["current_time"] and rest == ""
    assert tool_calls_from_text("<tool_call>{\"name\": \"rm\"}</tool_call>", KNOWN)[0] == []
    assert clean_tool_name("unknown<x>", KNOWN) == "unknown<x>"


@pytest.mark.parametrize(("finish", "stop"), [("stop", "end"), ("length", "max_tokens"),
                                              ("content_filter", "refusal"), ("weird", "other")])
def test_openai_finish_reasons(finish, stop):
    client, _ = oa_client(oa_response(content="hi", finish=finish))
    assert client.chat("", [Message(role="user", content="q")], TOOLS).stop == stop


def test_openai_errors_become_llm_errors():
    import httpx
    import openai

    req = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    err = openai.InternalServerError("boom", response=httpx.Response(500, request=req), body=None)
    client, _ = oa_client(error=err)
    with pytest.raises(LLMError, match="HTTP 500"):
        client.chat("", [Message(role="user", content="q")], TOOLS)
    client, _ = oa_client(error=openai.APIConnectionError(request=req))
    with pytest.raises(LLMError, match="request failed"):
        client.chat("", [Message(role="user", content="q")], TOOLS)


# --- Anthropic ----------------------------------------------------------------------------------

def an_response(blocks, stop_reason="end_turn"):
    return NS(content=blocks, stop_reason=stop_reason,
              usage=NS(input_tokens=40, output_tokens=9, cache_read_input_tokens=10, cache_creation_input_tokens=0))


def an_client(response=None, error=None):
    rec = Recorder(response, error)
    return AnthropicClient(client=NS(messages=rec)), rec


def test_anthropic_request_shape_and_tool_use_parsing():
    blocks = [NS(type="thinking", thinking="..."), NS(type="text", text="Checking."),
              NS(type="tool_use", id="tu_1", name="count_events", input={"group_by": "zone"})]
    client, rec = an_client(an_response(blocks, "tool_use"))
    r = client.chat("sys", [Message(role="user", content="q")], TOOLS, max_tokens=2048)
    kw = rec.kwargs
    assert kw["model"] == "claude-sonnet-5-5" and kw["system"] == "sys" and kw["max_tokens"] == 2048
    assert kw["thinking"] == {"type": "adaptive"} and kw["output_config"] == {"effort": "medium"}
    assert kw["tools"][0] == {"name": "count_events", "description": "Count.",
                              "input_schema": TOOLS[0].parameters}
    assert "temperature" not in kw
    assert r.stop == "tool_calls" and r.text == "Checking." and r.tool_calls[0].id == "tu_1"
    assert r.usage.input_tokens == 50 and r.message.raw is blocks


def test_anthropic_replays_raw_blocks_and_groups_tool_results():
    raw = [NS(type="thinking"), NS(type="tool_use")]
    history = [
        Message(role="user", content="q"),
        Message(role="assistant", tool_calls=[ToolCall("a", "current_time", {}), ToolCall("b", "count_events", {})],
                raw=raw, raw_provider="anthropic"),
        Message(role="tool", tool_call_id="a", content="1"),
        Message(role="tool", tool_call_id="b", content="2"),
        Message(role="assistant", content="from another provider", tool_calls=[ToolCall("c", "current_time", {})],
                raw="ignored", raw_provider="nvidia"),
        Message(role="tool", tool_call_id="c", content="3"),
    ]
    client, rec = an_client(an_response([NS(type="text", text="done")]))
    r = client.chat("", history, TOOLS)
    wire = rec.kwargs["messages"]
    assert wire[1] == {"role": "assistant", "content": raw}
    assert [b["tool_use_id"] for b in wire[2]["content"]] == ["a", "b"]  # one user message
    assert wire[3]["content"][0] == {"type": "text", "text": "from another provider"}
    assert wire[3]["content"][1] == {"type": "tool_use", "id": "c", "name": "current_time", "input": {}}
    assert r.stop == "end" and r.text == "done"


def test_anthropic_errors_and_refusal():
    import anthropic
    import httpx

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.RateLimitError("slow down", response=httpx.Response(429, request=req), body=None)
    client, _ = an_client(error=err)
    with pytest.raises(LLMError, match="HTTP 429"):
        client.chat("", [Message(role="user", content="q")], TOOLS)
    client, _ = an_client(an_response([], "refusal"))
    assert client.chat("", [Message(role="user", content="q")], TOOLS).stop == "refusal"


# --- factory and settings -----------------------------------------------------------------------

def test_factory_defaults_and_missing_keys():
    assert DEFAULT_MODELS == {"nvidia": "nvidia/nemotron-3-super-120b-a12b", "anthropic": "claude-sonnet-5-5"}
    with pytest.raises(LLMError, match="NVIDIA_API_KEY"):
        build_llm(LLMConfig(provider="nvidia"))
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        build_llm(LLMConfig(provider="anthropic"))
    with pytest.raises(LLMError, match="unknown LLM_PROVIDER"):
        build_llm(LLMConfig(provider="hermes", nvidia_api_key="x"))
    nv = build_llm(LLMConfig(provider="nvidia", nvidia_api_key="nvapi-test"))
    assert (nv.provider, nv.model) == ("nvidia", "nvidia/nemotron-3-super-120b-a12b")
    an = build_llm(LLMConfig(provider="anthropic", anthropic_api_key="sk-test", model="claude-opus-5-5"))
    assert (an.provider, an.model) == ("anthropic", "claude-opus-5-5")


def test_settings_read_llm_and_search_env(monkeypatch):
    for k, v in {"LLM_PROVIDER": "Anthropic", "LLM_MODEL": "claude-haiku-4-5", "ANTHROPIC_API_KEY": "sk-x",
                 "SEARCH_MAX_STEPS": "4", "SEARCH_RATE_LIMIT_PER_MIN": "3", "ALLOW_REMOTE_SEARCH": "true"}.items():
        monkeypatch.setenv(k, v)
    cfg = SentinelConfig.from_env(env_file=None)
    assert (cfg.llm.provider, cfg.llm.model, cfg.llm.anthropic_api_key) == ("anthropic", "claude-haiku-4-5", "sk-x")
    assert (cfg.search.max_steps, cfg.search.rate_limit_per_min, cfg.search.allow_remote) == (4, 3, True)
    assert json.dumps(cfg.llm.__dict__)  # plain values only
