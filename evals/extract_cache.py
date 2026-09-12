"""Prefill the extraction cache in parallel.

  smem-extract --split dev --workers 32            # ~4.6k unique sessions on dev
  smem-extract --split all --workers 32            # ~18.5k on the full set

The evaluation runner extracts serially inside each question, which wastes a GPU server that can
batch dozens of requests. This walks every unique session once (deduplicated by content hash), fills
the on-disk cache, and reports the write error rate. Re-running is free: cached sessions are skipped."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from evals.longmemeval.data import LMEQuestion
from evals.run import build_parser as build_eval_parser
from evals.run import load_questions
from smem.config import SystemConfig, load_dotenv, parse_override
from smem.extract import LLMExtractor
from smem.schemas import Session
from smem.system import build_llm


def unique_sessions(questions: list[LMEQuestion]) -> list[Session]:
    """One representative per content hash. The cache key includes the session id, so the
    representative's id is what later runs must also see; ids are stable per content in LongMemEval."""
    seen: dict[str, Session] = {}
    for q in questions:
        for s in q.sessions:
            seen.setdefault(s.content_hash(), s)
    return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parent = build_eval_parser()
    p = argparse.ArgumentParser(description="Prefill the extraction cache", parents=[parent], conflict_handler="resolve")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args(argv)
    overrides = dict(parse_override(x) for x in args.ablate)
    cfg = SystemConfig.load(args.config, overrides)
    questions = load_questions(args)
    sessions = unique_sessions(questions)
    llm = build_llm(cfg.models.extract_model, cfg.models.extract_base_url, cfg,
                    cfg.extract.constrained_decoding, extra_body=cfg.models.extract_extra_body)
    extractor = LLMExtractor(llm, cfg.extract.cache_dir, cfg.extract.constrained_decoding,
                             cfg.extract.max_episode_tokens, max_turn_tokens=cfg.extract.max_turn_tokens,
                             max_output_tokens=cfg.extract.max_output_tokens)
    print(f"{len(questions)} questions -> {len(sessions)} unique sessions; model {cfg.models.extract_model} "
          f"@ {cfg.models.extract_base_url or 'api.openai.com'}; constrained={cfg.extract.constrained_decoding}")
    t0 = time.time()
    n_ep = n_facts = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extractor.extract, s): s for s in sessions}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="session"):
            r = fut.result()
            n_ep += len(r.episodes)
            n_facts += len(r.facts)
    dt = time.time() - t0
    print(f"done in {dt / 60:.1f} min; {extractor.n_calls} model calls ({len(sessions) - extractor.n_calls} cache hits); "
          f"schema errors {extractor.n_schema_errors} ({extractor.n_schema_errors / max(len(sessions), 1):.2%})")
    print(f"  of which: truncated at max_output_tokens={cfg.extract.max_output_tokens} {extractor.n_truncated}; "
          f"'other' facts with no custom_attribute (kept under a derived key) {extractor.n_missing_custom_attr}")
    print(f"{n_ep} episodes, {n_facts} facts; prompt tokens {llm.prompt_tokens:,}, completion tokens {llm.completion_tokens:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
