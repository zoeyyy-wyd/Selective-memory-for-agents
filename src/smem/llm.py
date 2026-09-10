"""LLM access. One OpenAI-compatible client serves both the local vLLM extraction server and the hosted
answering / judge models. Responses are cached on disk by (model, messages, schema) so every stage of
the project is re-runnable without re-paying for calls."""

from __future__ import annotations

import hashlib
import json
import os
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
    ):
        from openai import OpenAI  # local import keeps the offline path free of network clients

        self.model = model
        self.base_url = base_url
        self.constrained_decoding = constrained_decoding
        key = api_key or os.environ.get("OPENAI_API_KEY") or ("EMPTY" if base_url else None)
        if key is None:
            raise RuntimeError("OPENAI_API_KEY is not set and no base_url was given")
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _is_vllm(self) -> bool:
        return bool(self.base_url) and "openai.com" not in self.base_url

    def complete(self, system, user, *, json_schema=None, temperature=0.0, max_tokens=1024) -> str:
        schema = json_schema if self.constrained_decoding else None
        cache_key = DiskCache.key(self.model, system, user, schema, temperature, max_tokens)
        if self.cache is not None:
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                  "max_tokens": max_tokens}
        if schema is not None:
            if self._is_vllm():
                kwargs["extra_body"] = {"guided_json": schema}
            else:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": schema, "strict": False},
                }
        elif json_schema is not None:
            # Unconstrained control: we still ask for JSON in the prompt, but do not enforce it.
            kwargs["messages"][0]["content"] += "\nRespond with a single JSON object and nothing else."

        resp = self.client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        self.calls += 1
        if resp.usage is not None:
            self.prompt_tokens += resp.usage.prompt_tokens or 0
            self.completion_tokens += resp.usage.completion_tokens or 0
        if self.cache is not None:
            self.cache.put(cache_key, text, {"model": self.model})
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
