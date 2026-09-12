"""Demo: feed a conversation, ask questions, print what was injected and what was evicted.

  smem-demo --conversation demo/sample_conversation.json --store-budget 250 --read-budget 120 \
      --ask "Which city does the user live in now?" --ask "Where did the user live in March?"
  smem-demo --conversation demo/sample_conversation.json --interactive
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from smem.answer import format_entry
from smem.config import SystemConfig, load_dotenv, parse_override
from smem.extract import load_sessions_from_json
from smem.schemas import Fact
from smem.system import build_system


def describe(mem, entry_id: str) -> str:
    e = mem.store.get(entry_id)
    return format_entry(e) if e is not None else f"[{entry_id}] (evicted)"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Selective memory demo")
    p.add_argument("--conversation", required=True, help="JSON list of sessions (see demo/sample_conversation.json)")
    p.add_argument("--config", default="configs/offline.yaml")
    p.add_argument("--backend", choices=["offline", "llm"], default="offline")
    p.add_argument("--store-budget", type=int, default=None)
    p.add_argument("--read-budget", type=int, default=None)
    p.add_argument("--ablate", action="append", default=[])
    p.add_argument("--ask", action="append", default=[], help="question to ask after ingesting everything")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--now", default=None, help="question date, e.g. 2023-06-01")
    args = p.parse_args(argv)

    overrides = dict(parse_override(x) for x in args.ablate)
    if args.store_budget:
        overrides["budget.store_tokens"] = args.store_budget
    if args.read_budget:
        overrides["budget.read_tokens"] = args.read_budget
    cfg = SystemConfig.load(args.config, overrides)
    mem = build_system(cfg, args.backend)
    sessions = load_sessions_from_json(args.conversation)
    print(f"config {cfg.name} (hash {cfg.config_hash()}): write={cfg.write.policy} evict={cfg.evict.policy} "
          f"B={cfg.budget.store_tokens} R={cfg.budget.read_tokens}\n")

    for s in sessions:
        active_before = mem.writer.active_ids()
        pool_before = {i: describe(mem, i) for i in mem.store.ids()}
        rep = mem.ingest_session(s)
        print(f"== session {s.session_id} @ {s.ts:%Y-%m-%d}: {rep.n_episodes} episodes, {rep.n_facts} facts extracted; "
              f"{len(rep.admitted)} admitted; store {rep.store_tokens}/{cfg.budget.store_tokens} tokens, {rep.store_entries} entries")
        for i in rep.admitted:
            print("   +", describe(mem, i))
        reasons = dict(mem.writer.evicted_log)
        for i in sorted(active_before - mem.writer.active_ids()):
            print(f"   - ({reasons.get(i, 'evicted')}) {pool_before.get(i, i)}")
    print()
    stats = mem.stats()
    print("write stats:", stats["write"])
    print("consolidation:", stats["consolidation"])
    print("chains:")
    seen = set()
    for f in mem.store.facts.values():
        if f.id in seen or f.id not in mem.writer.active_ids():
            continue
        chain = mem.store.chain(f.id)
        seen.update(x.id for x in chain)
        if len(chain) > 1:
            print("   " + " -> ".join(f"{x.value} [{x.valid_from:%Y-%m-%d}..{x.valid_to:%Y-%m-%d}]" if x.valid_to else f"{x.value} [{x.valid_from:%Y-%m-%d}..now]" for x in chain))
    print()

    now = datetime.fromisoformat(args.now) if args.now else None

    def ask(question: str) -> None:
        res = mem.ask(question, now=now)
        print(f"Q: {question}\n   temporal: {res.read.constraint.mode}  queries: {res.read.queries}")
        print(f"   injected ({res.read.tokens} tokens, {len(res.read.packed)} entries):")
        for e in sorted(res.read.packed, key=lambda e: -res.read.packed_rel.get(e.id, 0)):
            tag = "fact" if isinstance(e, Fact) else "ep"
            print(f"     rel={res.read.packed_rel.get(e.id, 0):.2f} {tag:4s} {format_entry(e)}")
        print(f"   A: {res.answer}{'  (abstained)' if res.abstained else ''}\n")

    for q in args.ask:
        ask(q)
    if args.interactive:
        print("type a question (empty line to quit)")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            ask(line)
    mem.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
