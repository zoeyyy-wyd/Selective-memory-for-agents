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
    p.add_argument("--config", default="configs/tokenrouter.yaml")
    p.add_argument("--live", action="store_true", help="also make one real call (costs a few tokens)")
    args = p.parse_args(argv)

    load_dotenv()
    cfg = SystemConfig.load(args.config)

    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("FAIL  OPENAI_API_KEY is empty. Paste the gateway key into that line in .env.")
        return 1
    print(f"ok    key loaded ({key[:6]}...{key[-4:]}, {len(key)} chars)")

    for role, model, base_url, provider in [
        ("answer", cfg.models.answer_model, cfg.models.answer_base_url, cfg.models.answer_provider),
        ("judge", cfg.models.judge_model, cfg.models.judge_base_url, cfg.models.judge_provider),
    ]:
        print(f"      {role}: {model} @ {base_url or 'api.openai.com'} "
              f"-> {resolve_provider(model, base_url, provider)}")

    base = cfg.models.answer_base_url
    if base:
        from openai import OpenAI

        try:
            ids = sorted(m.id for m in OpenAI(base_url=base, api_key=key).models.list().data)
        except Exception as e:                                    # noqa: BLE001 - report, don't raise
            print(f"FAIL  {base}/models: {type(e).__name__}: {str(e)[:200]}")
            return 1
        print(f"ok    gateway lists {len(ids)} models")
        for role, want in [("answer", cfg.models.answer_model), ("judge", cfg.models.judge_model)]:
            mark = "ok   " if want in ids else "FAIL "
            print(f"{mark} {role} model {want!r} {'found' if want in ids else 'NOT in the gateway list'}")
        if cfg.models.answer_model not in ids or cfg.models.judge_model not in ids:
            near = [i for i in ids if any(t in i for t in ("gpt", "claude", "sonnet", "haiku", "mini"))]
            print("      candidates: " + ", ".join(near[:25] or ids[:25]))
            print(f"      -> put exact ids into {args.config}. Never an 'auto'/'best' alias: a router that "
                  "substitutes models breaks the controlled comparison every ablation depends on.")
            return 1

    if args.live:
        llm = build_llm(cfg.models.answer_model, cfg.models.answer_base_url, cfg,
                        provider=cfg.models.answer_provider)
        try:
            out = llm.complete("Answer with one word.", "What is the capital of France?", max_tokens=10)
        except Exception as e:                                    # noqa: BLE001 - report, don't raise
            print(f"FAIL  live call: {type(e).__name__}: {str(e)[:160]}")
            print("      A 503 here is the gateway, not the config. It has been intermittent; retry, "
                  "and raise models.request_max_retries before a long run.")
            return 1
        if not out.strip():
            print(f"FAIL  live call returned an EMPTY string at max_tokens=10. {cfg.models.answer_model} "
                  "is most likely a reasoning model that spent the budget before emitting text.")
            return 1
        print(f"ok    live call returned {out.strip()[:40]!r}")
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
