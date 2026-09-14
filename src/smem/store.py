"""Storage: SQLite for metadata and validity chains (source of truth, written through on every
mutation), an exact inner-product vector index (FAISS when available) and an in-memory BM25 index.
A single history has entries in the low thousands, so everything is also mirrored in dicts."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime

import numpy as np
from rank_bm25 import BM25Okapi

from smem.embed import tokenize
from smem.schemas import RawTurn, Entry, Episode, Fact

try:  # pragma: no cover
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    _HAS_FAISS = False


class VectorIndex:
    """Exact cosine search over normalised vectors. FAISS IndexFlatIP behind an ID map when it is
    installed; a numpy matrix otherwise. Filtered search always goes through numpy because the
    active subset (one sieve) can be much smaller than the pool."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vecs: dict[str, np.ndarray] = {}
        self._int_of: dict[str, int] = {}
        self._str_of: dict[int, str] = {}
        self._next = 0
        self._faiss = faiss.IndexIDMap2(faiss.IndexFlatIP(dim)) if _HAS_FAISS else None

    def add(self, id_: str, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float32)
        if id_ in self._vecs:
            self.remove(id_)
        self._vecs[id_] = vec
        n = self._next
        self._next += 1
        self._int_of[id_] = n
        self._str_of[n] = id_
        if self._faiss is not None:
            self._faiss.add_with_ids(vec[None, :], np.array([n], dtype=np.int64))

    def remove(self, id_: str) -> None:
        if id_ not in self._vecs:
            return
        n = self._int_of.pop(id_)
        self._str_of.pop(n)
        del self._vecs[id_]
        if self._faiss is not None:
            self._faiss.remove_ids(np.array([n], dtype=np.int64))

    def vec(self, id_: str) -> np.ndarray:
        return self._vecs[id_]

    def __len__(self) -> int:
        return len(self._vecs)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._vecs

    def search(self, q: np.ndarray, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        q = np.asarray(q, dtype=np.float32)
        if allowed is not None or self._faiss is None:
            ids = [i for i in self._vecs if allowed is None or i in allowed]
            if not ids:
                return []
            mat = np.stack([self._vecs[i] for i in ids])
            sims = mat @ q
            top = np.argsort(-sims)[:k]
            return [(ids[j], float(sims[j])) for j in top]
        if len(self._vecs) == 0:
            return []
        sims, ints = self._faiss.search(q[None, :], min(k, len(self._vecs)))
        return [(self._str_of[int(n)], float(s)) for s, n in zip(sims[0], ints[0]) if n != -1]

    def max_sim(self, q: np.ndarray, allowed: set[str] | None = None) -> tuple[str | None, float]:
        hits = self.search(q, 1, allowed)
        return hits[0] if hits else (None, 0.0)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY, ts TEXT, session_id TEXT, turn_idx INTEGER, speaker TEXT, text TEXT,
    entities TEXT, tokens INTEGER, consolidated INTEGER, access_count INTEGER, last_access TEXT,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY, entity TEXT, attribute TEXT, value TEXT, valid_from TEXT, valid_to TEXT,
    superseded_by TEXT, sources TEXT, kind TEXT, speaker TEXT, session_id TEXT, tokens INTEGER,
    access_count INTEGER, last_access TEXT, embedding BLOB
);
CREATE INDEX IF NOT EXISTS facts_ea ON facts(entity, attribute);
CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT, turn_idx INTEGER, ts TEXT, speaker TEXT, text TEXT, tokens INTEGER,
    PRIMARY KEY (session_id, turn_idx)
);
"""


def _dt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class MemoryStore:
    def __init__(self, dim: int, path: str = ":memory:"):
        self.dim = dim
        self.db = sqlite3.connect(path)
        self.db.executescript(_SCHEMA)
        self.episodes: dict[str, Episode] = {}
        self.facts: dict[str, Fact] = {}
        self.vectors = VectorIndex(dim)
        self._entity_index: dict[str, set[str]] = defaultdict(set)
        self._pred: dict[str, str] = {}  # fact id -> id of the fact it supersedes
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._dirty = True
        self._insert_order: dict[str, int] = {}
        self._counter = 0
        self.turns: dict[tuple[str, int], RawTurn] = {}
        self._load()

    # ---- persistence ---------------------------------------------------------------------------
    def dump_to(self, path: str) -> None:
        """Copy the live database (including :memory:) to a file."""
        import os
        if os.path.exists(path):
            os.remove(path)
        self.db.commit()   # backup() retries forever (sleeping) while the source has an open transaction
        dest = sqlite3.connect(path)
        with dest:
            self.db.backup(dest)
        dest.close()

    def restore_from(self, path: str) -> None:
        """Replace the contents of this store with a saved database and rebuild every in-memory
        index from it. Reads from a private in-memory copy so later writes never touch the file."""
        src = sqlite3.connect(path)
        self.db.close()
        self.db = sqlite3.connect(":memory:")
        with self.db:
            src.backup(self.db)
        src.close()
        self.episodes.clear(); self.facts.clear(); self.vectors = VectorIndex(self.dim)
        self._entity_index.clear(); self._pred.clear(); self._bm25 = None; self._bm25_ids = []
        self._insert_order.clear(); self._counter = 0; self._dirty = True; self.turns.clear()
        self.db.executescript(_SCHEMA)   # older dumps predate the turns table
        self._load()

    def _load(self) -> None:
        for row in self.db.execute("SELECT * FROM episodes"):
            ep = Episode(
                id=row[0], ts=_dt(row[1]), session_id=row[2], turn_idx=row[3], speaker=row[4], text=row[5],
                entities=json.loads(row[6]), tokens=row[7], consolidated=bool(row[8]), access_count=row[9],
                last_access=_dt(row[10]), embedding=np.frombuffer(row[11], dtype=np.float32).tolist(),
            )
            self._index(ep)
        for row in self.db.execute("SELECT * FROM facts"):
            f = Fact(
                id=row[0], entity=row[1], attribute=row[2], value=row[3], valid_from=_dt(row[4]),
                valid_to=_dt(row[5]), superseded_by=row[6], sources=json.loads(row[7]), kind=row[8],
                speaker=row[9], session_id=row[10], tokens=row[11], access_count=row[12], last_access=_dt(row[13]),
                embedding=np.frombuffer(row[14], dtype=np.float32).tolist(),
            )
            self._index(f)

        for row in self.db.execute("SELECT session_id, turn_idx, ts, speaker, text, tokens FROM turns"):
            t = RawTurn(session_id=row[0], turn_idx=row[1], ts=_dt(row[2]), speaker=row[3], text=row[4], tokens=row[5])
            self.turns[(t.session_id, t.turn_idx)] = t

    # ---- raw turns (payload, not entries) -----------------------------------------------------
    def add_turns(self, turns: Iterable[RawTurn]) -> None:
        rows = []
        for t in turns:
            self.turns[(t.session_id, t.turn_idx)] = t
            rows.append((t.session_id, t.turn_idx, t.ts.isoformat(), t.speaker, t.text, t.tokens))
        self.db.executemany("INSERT OR REPLACE INTO turns VALUES (?, ?, ?, ?, ?, ?)", rows)

    def turn(self, session_id: str, turn_idx: int) -> RawTurn | None:
        return self.turns.get((session_id, turn_idx))

    def _write(self, e: Entry) -> None:
        emb = np.asarray(e.embedding, dtype=np.float32).tobytes()
        if isinstance(e, Episode):
            self.db.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (e.id, e.ts.isoformat(), e.session_id, e.turn_idx, e.speaker, e.text, json.dumps(e.entities),
                 e.tokens, int(e.consolidated), e.access_count,
                 e.last_access.isoformat() if e.last_access else None, emb),
            )
        else:
            self.db.execute(
                "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (e.id, e.entity, e.attribute, e.value, e.valid_from.isoformat(),
                 e.valid_to.isoformat() if e.valid_to else None, e.superseded_by, json.dumps(e.sources), e.kind,
                 e.speaker, e.session_id, e.tokens, e.access_count,
                 e.last_access.isoformat() if e.last_access else None, emb),
            )

    def _index(self, e: Entry) -> None:
        if isinstance(e, Episode):
            self.episodes[e.id] = e
        else:
            self.facts[e.id] = e
            if e.superseded_by:
                self._pred[e.superseded_by] = e.id
        for ent in e.entities:
            self._entity_index[ent.lower()].add(e.id)
        if e.embedding:
            self.vectors.add(e.id, np.asarray(e.embedding, dtype=np.float32))
        if e.id not in self._insert_order:
            self._insert_order[e.id] = self._counter
            self._counter += 1
        self._dirty = True

    # ---- CRUD ----------------------------------------------------------------------------------
    def add(self, e: Entry) -> None:
        self._index(e)
        self._write(e)

    def update(self, e: Entry) -> None:
        self.add(e)

    def remove(self, id_: str) -> None:
        e = self.get(id_)
        if e is None:
            return
        if isinstance(e, Episode):
            del self.episodes[id_]
            self.db.execute("DELETE FROM episodes WHERE id=?", (id_,))
        else:
            del self.facts[id_]
            self.db.execute("DELETE FROM facts WHERE id=?", (id_,))
            if e.superseded_by and self._pred.get(e.superseded_by) == id_:
                del self._pred[e.superseded_by]
        for ent in e.entities:
            self._entity_index[ent.lower()].discard(id_)
        self.vectors.remove(id_)
        self._dirty = True

    def get(self, id_: str) -> Entry | None:
        return self.episodes.get(id_) or self.facts.get(id_)

    def __contains__(self, id_: str) -> bool:
        return id_ in self.episodes or id_ in self.facts

    def __len__(self) -> int:
        return len(self.episodes) + len(self.facts)

    def ids(self) -> list[str]:
        return list(self.episodes) + list(self.facts)

    def entries(self, ids: Iterable[str] | None = None) -> list[Entry]:
        if ids is None:
            return list(self.episodes.values()) + list(self.facts.values())
        out = []
        for i in ids:
            e = self.get(i)
            if e is not None:
                out.append(e)
        return out

    def used_tokens(self, ids: Iterable[str] | None = None) -> int:
        return sum(e.tokens for e in self.entries(ids))

    def insert_order(self, id_: str) -> int:
        return self._insert_order.get(id_, -1)

    def touch(self, id_: str, ts: datetime) -> None:
        e = self.get(id_)
        if e is None:
            return
        e.access_count += 1
        e.last_access = ts
        self._write(e)

    def mark_consolidated(self, ids: Iterable[str]) -> None:
        for i in ids:
            ep = self.episodes.get(i)
            if ep is not None:
                ep.consolidated = True
                self._write(ep)

    # ---- retrieval -----------------------------------------------------------------------------
    def _ensure_bm25(self) -> None:
        if not self._dirty:
            return
        self._bm25_ids = self.ids()
        corpus = [tokenize(self.get(i).text) or ["<empty>"] for i in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._dirty = False

    def search_bm25(self, query: str, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []
        q_tokens = tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        if len(scores) and float(np.max(scores)) <= 0:
            # tiny corpora: every idf is zero, so fall back to plain term overlap
            q_set = set(q_tokens)
            scores = np.array([len(q_set & set(tokenize(self.get(i).text))) for i in self._bm25_ids], dtype=float)
        pairs = [(self._bm25_ids[i], float(s)) for i, s in enumerate(scores)
                 if s > 0 and (allowed is None or self._bm25_ids[i] in allowed)]
        pairs.sort(key=lambda p: -p[1])
        return pairs[:k]

    def search_dense(self, qvec: np.ndarray, k: int, allowed: set[str] | None = None) -> list[tuple[str, float]]:
        return self.vectors.search(qvec, k, allowed)

    def by_entity(self, entity: str) -> set[str]:
        return set(self._entity_index.get(entity.lower(), ()))

    # ---- validity chains -----------------------------------------------------------------------
    def facts_for(self, entity: str, attribute: str) -> list[Fact]:
        out = [f for f in self.facts.values()
               if f.entity.lower() == entity.lower() and f.attribute.lower() == attribute.lower()]
        out.sort(key=lambda f: f.valid_from)
        return out

    def chain(self, fact_id: str) -> list[Fact]:
        """Whole validity chain containing fact_id, oldest first. Follows pointers only, so nodes
        evicted from the store simply do not appear."""
        if fact_id not in self.facts:
            return []
        head = fact_id
        back_seen = {head}
        while head in self._pred and self._pred[head] in self.facts and self._pred[head] not in back_seen:
            head = self._pred[head]
            back_seen.add(head)
        chain = [self.facts[head]]
        fwd_seen = {head}
        cur = self.facts[head]
        while cur.superseded_by and cur.superseded_by in self.facts and cur.superseded_by not in fwd_seen:
            fwd_seen.add(cur.superseded_by)
            cur = self.facts[cur.superseded_by]
            chain.append(cur)
        return chain

    def chain_root(self, fact_id: str) -> str:
        c = self.chain(fact_id)
        return c[0].id if c else fact_id

    def supersede(self, old: Fact, new: Fact, at: datetime) -> None:
        """Knowledge update: never delete, close the old interval and link forward."""
        old.valid_to = at
        old.superseded_by = new.id
        new.valid_from = max(new.valid_from, at)
        self._pred[new.id] = old.id
        self._write(old)
        if new.id in self.facts:
            self._write(new)

    def volatility(self, entity: str, attribute: str) -> float:
        """updates per day over the attribute's observed span; 0 for a single-valued attribute."""
        chain = self.facts_for(entity, attribute)
        if len(chain) < 2:
            return 0.0
        span_days = max((chain[-1].valid_from - chain[0].valid_from).total_seconds() / 86400.0, 1.0)
        return (len(chain) - 1) / span_days

    def close(self) -> None:
        self.db.commit()
        self.db.close()


def key_drift(facts: Iterable[Fact], threshold: float = 0.8) -> dict:
    """Diagnostic, not a fix: near-identical attribute names under one entity, and near-identical
    entity names. Such pairs are where a validity chain silently failed to form. They are reported
    and never auto-merged, because a wrong merge closes a valid fact (hard failure) while a missed
    merge only leaves two independent facts (soft failure)."""
    import difflib

    attrs_by_entity: dict[str, set[str]] = defaultdict(set)
    for f in facts:
        attrs_by_entity[f.entity].add(f.attribute)

    def is_close(a: str, b: str) -> bool:
        # the common drift is a modifier glued on one end: current_location, rachel_colleague, num_x
        if a.startswith(b + "_") or b.startswith(a + "_") or a.endswith("_" + b) or b.endswith("_" + a):
            return True
        return difflib.SequenceMatcher(None, a, b).ratio() >= threshold

    def close_pairs(names: list[str]) -> list[tuple[str, str]]:
        return [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if a != b and is_close(a, b)]

    attr_pairs = [(ent, a, b) for ent, attrs in sorted(attrs_by_entity.items()) for a, b in close_pairs(sorted(attrs))]
    ent_pairs = close_pairs(sorted(attrs_by_entity))
    return {
        "n_entities": len(attrs_by_entity),
        "n_keys": sum(len(v) for v in attrs_by_entity.values()),
        "suspicious_attribute_pairs": attr_pairs,
        "suspicious_entity_pairs": ent_pairs,
    }
