"""Wire-format tests for the Anthropic backend. Like test_llm_client.py these build the request and
inspect it; nothing here touches the network."""

import pytest

from smem.config import SystemConfig
from smem.llm import THINKING_MIN_TOKENS, AnthropicLLM
from smem.system import resolve_provider

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def build(model="claude-sonnet-5", **kw):
    """AnthropicLLM without constructing the SDK client (no key needed in tests)."""
    obj = AnthropicLLM.__new__(AnthropicLLM)
    obj.model = model
    obj.constrained_decoding = kw.get("constrained_decoding", True)
    obj.thinking = kw.get("thinking", "off")
    obj.effort = kw.get("effort")
    obj.fallback_model = kw.get("fallback_model")
    return obj


def test_temperature_is_never_sent_and_system_is_omitted_when_empty():
    kw = build().request_kwargs("", "judge this", None, 0.0, 10)
    assert "temperature" not in kw          # removed on Claude 4.6+; a 400 if sent
    assert "system" not in kw               # LLMJudge passes "" and Anthropic wants it absent
    assert kw["messages"] == [{"role": "user", "content": "judge this"}]
    kw = build().request_kwargs("sys", "usr", None, 0.0, 10)
    assert kw["system"] == "sys"


def test_thinking_off_by_default_so_a_ten_token_judge_call_still_returns_text():
    kw = build().request_kwargs("", "usr", None, 0.0, 10)
    assert kw["thinking"] == {"type": "disabled"}
    assert kw["max_tokens"] == 10           # untouched: no thinking tokens to leave room for


def test_adaptive_thinking_raises_the_max_tokens_floor():
    kw = build(thinking="adaptive").request_kwargs("", "usr", None, 0.0, 10)
    assert kw["thinking"] == {"type": "adaptive"}
    assert kw["max_tokens"] == THINKING_MIN_TOKENS
    # a caller that already asked for more keeps it
    assert build(thinking="adaptive").request_kwargs("", "u", None, 0.0, 99999)["max_tokens"] == 99999


def test_haiku_45_gets_neither_thinking_nor_effort():
    kw = build("claude-haiku-4-5", effort="low").request_kwargs("s", "u", None, 0.0, 100)
    assert "thinking" not in kw             # omitting it is how thinking stays off on Haiku
    assert "output_config" not in kw        # effort is a 400 on Haiku 4.5


def test_effort_and_schema_share_output_config():
    kw = build(effort="low").request_kwargs("s", "u", SCHEMA, 0.0, 100)
    assert kw["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert "response_format" not in kw      # that is the OpenAI spelling


def test_unconstrained_arm_asks_for_json_in_the_prompt_only():
    kw = build(constrained_decoding=False).request_kwargs("s", "u", SCHEMA, 0.0, 100)
    assert "output_config" not in kw
    assert kw["system"].endswith("Respond with a single JSON object and nothing else.")
    # with no system prompt the instruction still has to land somewhere
    kw = build(constrained_decoding=False).request_kwargs("", "u", SCHEMA, 0.0, 100)
    assert kw["system"] == "Respond with a single JSON object and nothing else."


def test_fallbacks_are_opt_in():
    assert "fallbacks" not in build().request_kwargs("s", "u", None, 0.0, 100)
    kw = build(fallback_model="claude-opus-4-8").request_kwargs("s", "u", None, 0.0, 100)
    assert kw["fallbacks"] == [{"model": "claude-opus-4-8"}]
    assert kw["betas"] == ["server-side-fallback-2026-06-01"]


def test_thinking_off_above_effort_high_is_rejected_locally():
    # documented 400 on Opus 5; fail in the constructor rather than mid-run
    with pytest.raises(ValueError, match="thinking='off' is rejected"):
        AnthropicLLM("claude-opus-5", thinking="off", effort="max")
    with pytest.raises(ValueError, match="must be 'off' or 'adaptive'"):
        AnthropicLLM("claude-opus-5", thinking="on")


def test_provider_routing():
    # a base_url always means a self-hosted OpenAI-compatible server, even for a claude-* name
    assert resolve_provider("Qwen/Qwen3-8B-AWQ", "http://localhost:8000/v1") == "openai"
    assert resolve_provider("claude-sonnet-5", "http://localhost:8000/v1") == "openai"
    assert resolve_provider("claude-sonnet-5", None) == "anthropic"
    assert resolve_provider("gpt-4.1-mini", None) == "openai"
    assert resolve_provider("gpt-4.1-mini", None, "anthropic") == "anthropic"


def test_claude_config_loads_and_keeps_the_judge_on_openai():
    cfg = SystemConfig.load("configs/claude.yaml")
    assert cfg.models.answer_model == "claude-sonnet-5"
    assert resolve_provider(cfg.models.answer_model, cfg.models.answer_base_url,
                            cfg.models.answer_provider) == "anthropic"
    # the official LongMemEval judge stays where it is, so accuracy stays comparable upstream
    assert resolve_provider(cfg.models.judge_model, None, cfg.models.judge_provider) == "openai"
    assert cfg.anthropic.thinking == "off"
    # extraction is pinned to the local vLLM server regardless
    assert resolve_provider(cfg.models.extract_model, cfg.models.extract_base_url) == "openai"


def test_answer_model_override_changes_the_run_id():
    """The bake-off relies on --ablate models.answer_model=... producing a distinct run."""
    base = SystemConfig.load("configs/claude.yaml")
    other = base.with_overrides({"models.answer_model": "claude-haiku-4-5"})
    assert base.config_hash() != other.config_hash()


# ---- OpenAI-compatible gateways (TokenRouter and friends) ---------------------------------------

def test_gateway_base_url_keeps_claude_models_on_the_openai_wire_format():
    """A gateway speaks OpenAI even for claude-* ids, so base_url has to win over the model name."""
    assert resolve_provider("claude-sonnet-5", "https://api.tokenrouter.com/v1") == "openai"
    assert resolve_provider("gpt-4.1-mini", "https://api.tokenrouter.com/v1") == "openai"


def test_vendor_extra_body_never_reaches_the_answer_or_judge_client():
    """Regression: extract_extra_body used to go to every client. That was invisible while answering
    went to api.openai.com (dropped there), but a gateway has a base_url, so Qwen3's
    chat_template_kwargs would have been forwarded to a router fronting GPT or Claude."""
    from smem.system import build_llm

    cfg = SystemConfig.load("configs/tokenrouter.yaml")
    assert cfg.models.extract_extra_body  # the config really does set one

    answer = build_llm(cfg.models.answer_model, cfg.models.answer_base_url, cfg,
                       provider=cfg.models.answer_provider)
    judge = build_llm(cfg.models.judge_model, cfg.models.judge_base_url, cfg,
                      provider=cfg.models.judge_provider)
    for llm in (answer, judge):
        assert llm.extra_body == {}
        assert "extra_body" not in llm.request_kwargs("s", "u", None, 0.0, 100)

    extract = build_llm(cfg.models.extract_model, cfg.models.extract_base_url, cfg,
                        extra_body=cfg.models.extract_extra_body)
    assert extract.request_kwargs("s", "u", None, 0.0, 100)["extra_body"] == cfg.models.extract_extra_body


def test_substitutions_reports_a_router_serving_a_different_model():
    from smem.llm import OpenAICompatLLM

    llm = OpenAICompatLLM("gpt-4.1-mini", base_url="https://api.tokenrouter.com/v1", api_key="k")
    llm.served_models.update({"gpt-4.1-mini": 48, "gpt-4o-mini": 2})
    assert llm.substitutions() == {"gpt-4o-mini": 2}   # non-empty => not a controlled comparison
    llm.served_models.clear()
    llm.served_models.update({"gpt-4.1-mini": 50})
    assert llm.substitutions() == {}


def test_gateway_503_is_retried_but_a_400_is_not():
    """The gateway goes down for minutes at a time and a run is thousands of calls, so a blip must
    not end it. A 400 (bad model id, bad schema) cannot be fixed by waiting and must fail at once."""
    import httpx
    from openai import APIStatusError

    from smem.llm import OpenAICompatLLM

    def err(status):
        req = httpx.Request("POST", "https://api.tokenrouter.com/v1/chat/completions")
        return APIStatusError("boom", response=httpx.Response(status, request=req), body=None)

    llm = OpenAICompatLLM("m", base_url="https://api.tokenrouter.com/v1", api_key="k",
                          retry_attempts=4, retry_base_delay=0.0, retry_max_delay=0.0)

    calls = {"n": 0}

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise err(503)
        return "served"

    llm.client.chat.completions.create = flaky
    assert llm._create_with_retry({}) == "served"
    assert calls["n"] == 3 and llm.n_retries == 2

    def always_400(**kw):
        calls["n"] += 1
        raise err(400)

    calls["n"], llm.n_retries = 0, 0
    llm.client.chat.completions.create = always_400
    with pytest.raises(APIStatusError):
        llm._create_with_retry({})
    assert calls["n"] == 1 and llm.n_retries == 0   # no waiting on an unfixable error


def test_retry_gives_up_and_reraises_after_the_last_attempt():
    import httpx
    from openai import APIStatusError

    from smem.llm import OpenAICompatLLM

    llm = OpenAICompatLLM("m", base_url="https://x.example/v1", api_key="k",
                          retry_attempts=3, retry_base_delay=0.0, retry_max_delay=0.0)

    def down(**kw):
        req = httpx.Request("POST", "https://x.example/v1/chat/completions")
        raise APIStatusError("down", response=httpx.Response(503, request=req), body=None)

    llm.client.chat.completions.create = down
    with pytest.raises(APIStatusError):
        llm._create_with_retry({})
    assert llm.n_retries == 3
