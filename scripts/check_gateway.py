#!/usr/bin/env python
"""Check an OpenAI-compatible gateway before spending anything on a run.

    python scripts/check_gateway.py                       # uses configs/tokenrouter.yaml
    python scripts/check_gateway.py --config configs/default.yaml

Verifies, in order: the key is loaded, the gateway answers /v1/models, the ids named in the config
exist there, and one real (tiny) call round-trips -- reporting which model actually served it, since
a router is free to substitute and every ablation here assumes the answering model is held constant.
"""

from __future__ import annotations

import argparse
import os
import sys

from smem.config import SystemConfig, load_dotenv
from smem.system import build_llm, resolve_provider


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--live", action="store_true", help="also make one real call (costs a few tokens)")
    args = p.parse_args(argv)

    load_dotenv()
    cfg = SystemConfig.load(args.config)

    for role, env in (("answer", cfg.models.answer_api_key_env), ("judge", cfg.models.judge_api_key_env)):
        key = os.environ.get(env, "")
        if not key:
            print(f"FAIL  {env} is empty ({role} reads it). Paste the key into that line in .env.")
            return 1
        print(f"ok    {role} key {env} loaded ({key[:6]}...{key[-4:]}, {len(key)} chars)")

    for role, model, base_url, provider in [
        ("answer", cfg.models.answer_model, cfg.models.answer_base_url, cfg.models.answer_provider),
        ("judge", cfg.models.judge_model, cfg.models.judge_base_url, cfg.models.judge_provider),
    ]:
        print(f"      {role}: {model} @ {base_url or 'api.openai.com'} "
              f"-> {resolve_provider(model, base_url, provider)}")

    # Check each role's model id against the model list of the endpoint THAT ROLE uses. A direct
    # role (no base_url) is checked against api.openai.com with its own key, never against a gateway.
    from openai import OpenAI

    roles = [("answer", cfg.models.answer_model, cfg.models.answer_base_url, cfg.models.answer_api_key_env),
             ("judge", cfg.models.judge_model, cfg.models.judge_base_url, cfg.models.judge_api_key_env)]
    listed: dict[str, list[str]] = {}
    bad = False
    for role, model, base_url, env in roles:
        where = base_url or "https://api.openai.com/v1"
        if where not in listed:
            try:
                listed[where] = sorted(m.id for m in OpenAI(base_url=base_url, api_key=os.environ[env]).models.list().data)
                print(f"ok    {where} lists {len(listed[where])} models")
            except Exception as e:                                # noqa: BLE001 - report, don't raise
                print(f"FAIL  {where}/models: {type(e).__name__}: {str(e)[:160]}")
                return 1
        ids = listed[where]
        if model in ids:
            print(f"ok    {role} model {model!r} found at {where}")
        else:
            bad = True
            near = [i for i in ids if any(t in i for t in ("gpt", "claude", "sonnet", "haiku", "mini", "glm"))]
            print(f"FAIL  {role} model {model!r} NOT listed at {where}; candidates: " + ", ".join(near[:20] or ids[:20]))
    if bad:
        print(f"      -> put exact ids into {args.config}. Never an 'auto'/'best' alias: a router that "
              "substitutes models breaks the controlled comparison every ablation depends on.")
        return 1

    if args.live:
        llm = build_llm(cfg.models.answer_model, cfg.models.answer_base_url, cfg,
                        provider=cfg.models.answer_provider, api_key_env=cfg.models.answer_api_key_env)
        try:
            out = llm.complete("Answer with one word.", "What is the capital of France?",
                               max_tokens=cfg.models.answer_max_tokens)
        except Exception as e:                                    # noqa: BLE001 - report, don't raise
            print(f"FAIL  live call: {type(e).__name__}: {str(e)[:160]}")
            print("      A 503 here is the gateway, not the config. It has been intermittent; retry, "
                  "and raise models.request_max_retries before a long run.")
            return 1
        if not out.strip():
            print(f"FAIL  live call returned an EMPTY string at max_tokens={cfg.models.answer_max_tokens}. "
                  f"{cfg.models.answer_model} is probably a reasoning model; raise models.answer_max_tokens.")
            return 1
        print(f"ok    live answer call returned {out.strip()[:40]!r}")
        from evals.longmemeval.judge import parse_verdict

        jl = build_llm(cfg.models.judge_model, cfg.models.judge_base_url, cfg,
                       provider=cfg.models.judge_provider, api_key_env=cfg.models.judge_api_key_env)
        jv = jl.complete("", "Question: capital of France?\n\nCorrect Answer: Paris\n\nModel Response: Paris.\n\n"
                         "Is the model response correct? Answer yes or no only.", max_tokens=cfg.models.judge_max_tokens)
        print(f"ok    live judge call: {jv.strip()[:20]!r} -> {parse_verdict(jv)} "
              f"({jl.completion_tokens} output tokens, served by {dict(jl.served_models)})")
        served = getattr(llm, "served_models", None)
        if served:
            print(f"      served by: {dict(served)}")
            if llm.substitutions():
                print(f"WARN  gateway substituted {llm.substitutions()} -- pin the model or runs are not comparable")
    else:
        print("      (add --live to make one real call and see which model actually serves it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
