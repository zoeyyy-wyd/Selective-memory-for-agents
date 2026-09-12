"""LLM access. One OpenAI-compatible client serves both the local vLLM extraction server and the hosted
answering / judge models. Responses are cached on disk by (model, messages, schema) so every stage of
the project is re-runnable without re-paying for calls."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Protocol


class LLM(Protocol):
    model: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str: ...


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, key: str) -> str | None:
        p = self.root / f"{key}.json"
        if p.exists():
            return json.loads(p.read_text())["text"]
        return None

    def put(self, key: str, text: str, meta: dict[str, Any] | None = None) -> None:
        p = self.root / f"{key}.json"
        p.write_text(json.dumps({"text": text, "meta": meta or {}}, ensure_ascii=False))


class OpenAICompatLLM:
    """Works against api.openai.com and against vLLM's OpenAI-compatible server. For vLLM the JSON
    schema is passed as `guided_json` (xgrammar / outlines backend); for OpenAI as a structured output."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        constrained_decoding: bool = True,
        schema_mode: str = "response_format",
        extra_body: dict[str, Any] | None = None,
        max_retries: int = 2,
        timeout: float = 600.0,
        retry_attempts: int = 5,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
    ):
        from openai import OpenAI  # local import keeps the offline path free of network clients

        self.model = model
        self.base_url = base_url
        self.constrained_decoding = constrained_decoding
        # "response_format" is understood by OpenAI, vLLM (routed to its structured-output backend)
        # and llama.cpp; "guided_json" is the older vLLM-only field.
        self.schema_mode = schema_mode
        # Server-specific request fields, e.g. {"chat_template_kwargs": {"enable_thinking": False}}
        # to switch off Qwen3 thinking. Never sent to api.openai.com, which rejects unknown fields.
        self.extra_body = dict(extra_body or {})
        key = api_key or os.environ.get("OPENAI_API_KEY") or ("EMPTY" if base_url else None)
        if key is None:
            raise RuntimeError("OPENAI_API_KEY is not set and no base_url was given")
        # A shared gateway 503s intermittently ('system disk overloaded' on TokenRouter). The SDK
        # retries 5xx with backoff, and an eval run is long enough that giving up after the
        # default 2 attempts loses hours of work to a blip.
        self.client = OpenAI(base_url=base_url, api_key=key, max_retries=max_retries, timeout=timeout)
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        # A routing gateway may serve a request with a model other than the one asked for. Every
        # ablation in this project assumes the answering model is held constant, so substitution is
        # silent corruption of the comparison, not a performance detail: count what actually served.
        self.served_models: Counter[str] = Counter()
        self.retry_attempts = retry_attempts
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.n_retries = 0

    def _is_vllm(self) -> bool:
        """Self-hosted OpenAI-compatible server, i.e. one that takes vLLM's vendor fields. A hosted
        gateway also has a base_url, so this is only ever consulted for extraction, whose server is
        pinned by `extract_base_url`; `extra_body` is now passed in by that caller alone."""
        return bool(self.base_url) and "openai.com" not in self.base_url

    def request_kwargs(self, system: str, user: str, json_schema: dict[str, Any] | None,
                       temperature: float, max_tokens: int) -> dict[str, Any]:
        schema = json_schema if self.constrained_decoding else None
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                  "max_tokens": max_tokens}
        extra_body = dict(self.extra_body) if self._is_vllm() else {}
        if schema is not None:
            if self.schema_mode == "guided_json" and self._is_vllm():
                extra_body["guided_json"] = schema
            else:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": schema, "strict": False},
                }
        elif json_schema is not None:
            # Unconstrained control: we still ask for JSON in the prompt, but do not enforce it.
            kwargs["messages"][0]["content"] += "\nRespond with a single JSON object and nothing else."
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _create_with_retry(self, kwargs: dict[str, Any]):
        """The SDK already retries 5xx, but its backoff tops out inside a minute. A shared gateway can
        be down for far longer -- TokenRouter returned 'system disk overloaded' for several minutes at
        a stretch -- and an eval run is thousands of calls, so one blip would otherwise throw away
        hours. Retries only what can succeed on a second attempt: a 400 (bad model id, bad schema)
        fails immediately, because waiting cannot fix it."""
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        delay = self.retry_base_delay
        for attempt in range(self.retry_attempts):
            try:
                return self.client.chat.completions.create(**kwargs)
            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                last = e
            except APIStatusError as e:
                if e.status_code < 500:
                    raise
                last = e
            self.n_retries += 1
            if attempt == self.retry_attempts - 1:
                raise last
            time.sleep(min(delay, self.retry_max_delay) * (1.0 + 0.25 * random.random()))
            delay *= 2
        raise RuntimeError("unreachable")

    def complete(self, system, user, *, json_schema=None, temperature=0.0, max_tokens=1024) -> str:
        schema = json_schema if self.constrained_decoding else None
        cache_key = DiskCache.key(self.model, system, user, schema, temperature, max_tokens)
        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit

        kwargs = self.request_kwargs(system, user, json_schema, temperature, max_tokens)
        resp = self._create_with_retry(kwargs)
        text = resp.choices[0].message.content or ""
        self.calls += 1
        served = getattr(resp, "model", None) or self.model
        self.served_models[served] += 1
        if resp.usage is not None:
            self.prompt_tokens += resp.usage.prompt_tokens or 0
            self.completion_tokens += resp.usage.completion_tokens or 0
        if self.cache is not None:
            self.cache.put(cache_key, text, {"model": self.model, "served_model": served})
        return text

    @staticmethod
    def _norm_model(name: str) -> str:
        """Gateway ids carry a vendor prefix and come back resolved to a dated snapshot -- asking for
        `openai/gpt-5-mini` is served by `gpt-5-mini-2025-08-07`. Same model, so normalise both away
        before comparing; anything still different is a real reroute."""
        name = name.split("/")[-1].lower()
        name = re.sub(r"[-:]free$", "", name)   # billing tier, not a model: z-ai/glm-5.3-free is served as glm-5.3
        return re.sub(r"[-@]?\d{4}[-_]?\d{2}[-_]?\d{2}$", "", name).rstrip("-@")

    def substitutions(self) -> dict[str, int]:
        """Responses served by a different model than requested. Non-empty means the gateway rerouted
        and the run is not a controlled comparison; report it, do not average over it."""
        want = self._norm_model(self.model)
        return {m: n for m, n in self.served_models.items() if self._norm_model(m) != want}


# Claude models that reject `output_config.effort` and take thinking only as an explicit budget.
# For these, thinking is off simply by omitting the parameter.
_NO_EFFORT_PREFIXES = ("claude-haiku-4-5", "claude-sonnet-4-5", "claude-haiku-3", "claude-3")
# Adaptive thinking draws from max_tokens; leave room or the text block comes back empty.
THINKING_MIN_TOKENS = 4096


class AnthropicLLM:
    """Anthropic Messages API behind the same `LLM` protocol as `OpenAICompatLLM`, so answering and
    judging can be pointed at Claude without touching the read/write pipeline.

    Five things differ from the OpenAI path and are absorbed here rather than leaking outward:

    * `temperature` was removed on Claude 4.6+ and is a 400 if sent. It is accepted and ignored;
      rerun determinism comes from `DiskCache`, as it already did on the OpenAI path.
    * Structured output is `output_config.format`, not `response_format`.
    * Thinking is ON by default on Sonnet 5 / Opus 5 and its tokens are drawn from `max_tokens`, so
      the judge's `max_tokens=10` would be spent before any text is emitted. Thinking is therefore
      off by default here, and `max_tokens` gets a floor whenever it is switched on.
    * An empty system prompt (`LLMJudge` passes "") must be omitted, not sent as "".
    * A policy decline arrives as HTTP 200 with `stop_reason="refusal"`, not as an exception.

    On refusals: the server-side `fallbacks` parameter is deliberately NOT enabled by default. It
    would re-run a declined request on a different model inside the same call, which in an ablation
    would silently put two different answering models in one run and break the controlled
    comparison. Refusals are counted in `n_refusals` and surface as an empty answer (scored wrong)
    so they stay visible. Set `fallback_model` if you would rather trade that visibility for
    coverage; LongMemEval content is benign, so the expected count is zero either way.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        constrained_decoding: bool = True,
        thinking: str = "off",              # "off" | "adaptive"
        effort: str | None = None,          # None | low | medium | high | xhigh | max
        fallback_model: str | None = None,
        max_retries: int = 2,
        timeout: float = 600.0,
    ):
        # Validated before the SDK import so a bad config fails the same way with or without it.
        if thinking not in ("off", "adaptive"):
            raise ValueError(f"thinking must be 'off' or 'adaptive', got {thinking!r}")
        # Documented 400: thinking cannot be disabled above effort 'high' on Opus 5.
        if thinking == "off" and effort in ("xhigh", "max"):
            raise ValueError(f"thinking='off' is rejected at effort={effort!r}; use thinking='adaptive'")

        # Local import keeps the offline path free of network clients.
        from anthropic import Anthropic

        self.model = model
        self.base_url = None                # kept so callers can treat both clients alike
        self.constrained_decoding = constrained_decoding
        self.thinking = thinking
        self.effort = effort
        self.fallback_model = fallback_model
        kwargs: dict[str, Any] = {"max_retries": max_retries, "timeout": timeout}
        # Omitted api_key lets the SDK resolve ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
        # `ant auth login` profile; passing an explicit None would not.
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            kwargs["api_key"] = key
        self.client = Anthropic(**kwargs)
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.n_refusals = 0
        self.n_truncated = 0

    def _supports_effort(self) -> bool:
        return not self.model.startswith(_NO_EFFORT_PREFIXES)

    def request_kwargs(self, system: str, user: str, json_schema: dict[str, Any] | None,
                       temperature: float, max_tokens: int) -> dict[str, Any]:
        """Built separately from the call so the wire format is testable without a network client,
        exactly as `OpenAICompatLLM.request_kwargs` is. `temperature` is ignored (see class docstring)."""
        schema = json_schema if self.constrained_decoding else None
        messages = [{"role": "user", "content": user}]
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        if self._supports_effort():
            if self.thinking == "adaptive":
                kwargs["thinking"] = {"type": "adaptive"}
                kwargs["max_tokens"] = max(max_tokens, THINKING_MIN_TOKENS)
            else:
                kwargs["thinking"] = {"type": "disabled"}
        # Haiku 4.5 and older: omitting `thinking` already means no thinking, and `effort` is a 400.
        output_config: dict[str, Any] = {}
        if self.effort is not None and self._supports_effort():
            output_config["effort"] = self.effort
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        elif json_schema is not None:
            # Unconstrained control arm: ask for JSON in the prompt but do not enforce it, matching
            # the `no_constrained_decoding` ablation on the OpenAI path.
            instruction = "Respond with a single JSON object and nothing else."
            kwargs["system"] = (kwargs["system"] + "\n" + instruction) if system else instruction
        if output_config:
            kwargs["output_config"] = output_config
        if self.fallback_model:
            kwargs["betas"] = ["server-side-fallback-2026-06-01"]
            kwargs["fallbacks"] = [{"model": self.fallback_model}]
        return kwargs

    def complete(self, system, user, *, json_schema=None, temperature=0.0, max_tokens=1024) -> str:
        schema = json_schema if self.constrained_decoding else None
        cache_key = DiskCache.key(self.model, system, user, schema, temperature, max_tokens)
        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit

        kwargs = self.request_kwargs(system, user, json_schema, temperature, max_tokens)
        create = self.client.beta.messages.create if self.fallback_model else self.client.messages.create
        resp = create(**kwargs)
        self.calls += 1
        if resp.usage is not None:
            self.prompt_tokens += resp.usage.input_tokens or 0
            self.completion_tokens += resp.usage.output_tokens or 0
        if resp.stop_reason == "refusal":
            self.n_refusals += 1
            return ""
        if resp.stop_reason == "max_tokens":
            # With thinking on this means the budget went to reasoning; without it, a long answer.
            self.n_truncated += 1
        text = "".join(b.text for b in resp.content if b.type == "text")
        if self.cache is not None:
            self.cache.put(cache_key, text, {"model": self.model, "stop_reason": resp.stop_reason})
        return text


class ScriptedLLM:
    """Test double: answers from a queue, or via a callable(system, user) -> str."""

    def __init__(self, responses: list[str] | None = None, fn=None, model: str = "scripted"):
        self.model = model
        self._responses = list(responses or [])
        self._fn = fn
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, *, json_schema=None, temperature=0.0, max_tokens=1024) -> str:
        self.calls.append((system, user))
        if self._fn is not None:
            return self._fn(system, user)
        if self._responses:
            return self._responses.pop(0)
        return ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Tolerant JSON extraction: strips code fences and trailing prose. Returns None on failure."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
