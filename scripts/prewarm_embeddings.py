#!/usr/bin/env python
"""Embed every entry a prompt version's cached extractions would produce, on the GPU, in large batches,
so evaluation shards (which embed on the CPU when vLLM owns the card) hit the cache instead of computing.

    python scripts/prewarm_embeddings.py --split dev --prompt-version v3-detail [--batch 256]

Run it once after every re-extraction, before any evaluation. Idempotent: cached texts are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time

from evals.extract_cache import unique_sessions
from evals.longmemeval.data import load_longmemeval
from evals.longmemeval.split import load_split, stratified_split
from smem.config import SystemConfig, load_dotenv
from smem.embed import get_embedder
from smem.extract import HeuristicExtractor, LLMExtractor
from smem.llm import DiskCache


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--prompt-version", default=None, help="defaults to the config's extract.prompt_version")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default=None, help="cuda (default) or cpu")
    args = p.parse_args(argv)
    load_dotenv()
    cfg = SystemConfig.load(args.config)
    version = args.prompt_version or cfg.extract.prompt_version

    qs = load_longmemeval("data/longmemeval_s_cleaned.json")
    try:
        dev, test = load_split(qs, "data/split_dev100_seed0.json")
    except FileNotFoundError:
        dev, test = stratified_split(qs)
    questions = {"dev": dev, "test": test, "all": qs}[args.split]
    sessions = unique_sessions(questions)

    # parse cached extractions only -- never call the extraction model here
    ex = LLMExtractor.__new__(LLMExtractor)
    ex.max_episode_tokens, ex.max_turn_tokens = cfg.extract.max_episode_tokens, cfg.extract.max_turn_tokens
    ex.fallback = HeuristicExtractor(cfg.extract.max_episode_tokens)
    ex.n_schema_errors = ex.n_missing_custom_attr = ex.n_truncated = 0
    cache = DiskCache(cfg.extract.cache_dir)
    texts: set[str] = set()
    missing = 0
    for s in sessions:
        raw = cache.get(DiskCache.key("extract", version, cfg.models.extract_model, True, cfg.extract.max_turn_tokens,
                                      s.session_id, s.content_hash()))
        if raw is None:
            missing += 1
            continue
        res = ex.parse(s, raw)
        texts.update(e.text for e in res.episodes)
        texts.update(f.text for f in res.facts)
    print(f"{len(sessions)} sessions ({missing} without a cached extraction for {version!r}); {len(texts)} distinct entry texts")

    emb = get_embedder(cfg.models.embedder, cfg.models.embed_dim, cfg.models.embed_cache_dir, device=args.device)
    todo = [t for t in texts if not emb.has(t)] if hasattr(emb, "has") else sorted(texts)
    print(f"{len(todo)} not yet cached; embedding in batches of {args.batch} on {args.device or 'default device'}")
    t0 = time.time()
    for i in range(0, len(todo), args.batch):
        emb.encode(todo[i:i + args.batch])
        if (i // args.batch) % 20 == 0:
            print(f"  {min(i + args.batch, len(todo))}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
