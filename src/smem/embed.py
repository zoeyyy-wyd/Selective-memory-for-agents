"""Embedding backends. All return L2-normalised float32 vectors so cosine similarity is a dot product.

`HashEmbedder` is a deterministic feature-hashing bag of unigrams + bigrams: no model download, good
enough for the offline pipeline and tests. `SentenceTransformerEmbedder` wraps bge-m3 for real runs."""

from __future__ import annotations

import hashlib
import re
from itertools import pairwise
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(["a", "an", "the", "and", "or", "but", "if", "so", "of", "to", "in", "on", "at", "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "am", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "its", "they", "them", "this", "that", "these", "those", "there", "here", "have", "has", "had", "do", "does", "did", "not", "no", "yes", "just", "very", "really", "also", "about", "into", "over", "after", "before", "while", "when", "where", "what", "which", "who", "whom", "how", "why", "can", "could", "would", "should", "will", "shall", "may", "might", "must", "than", "then", "too"])


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _index(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(token.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            toks = tokenize(text)
            content = [t for t in toks if t not in STOPWORDS]
            # content unigrams carry the signal; bigrams (stop words included) keep some word order
            feats = content + content + [f"{a}_{b}" for a, b in pairwise(toks)]
            for f in feats:
                idx, sign = self._index(f)
                out[row, idx] += sign
        return normalize(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        from sentence_transformers import SentenceTransformer  # heavy import kept local

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)


def normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class CachedEmbedder:
    """SQLite-backed cache in front of a model embedder, keyed by (model, text). Sessions are shared
    across LongMemEval questions and configurations, so each text is embedded once per project."""

    def __init__(self, inner: Embedder, name: str, path: str):
        import sqlite3
        from pathlib import Path

        self.inner = inner
        self.name = name
        self.dim = inner.dim
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # several evaluation shards share this file; wait for a writer instead of raising 'database is locked'
        self.db = sqlite3.connect(path, timeout=30)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.name}\x00{text}".encode()).hexdigest()

    def has(self, text: str) -> bool:
        return self.db.execute("SELECT 1 FROM emb WHERE key=?", (self._key(text),)).fetchone() is not None

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        missing: list[int] = []
        for i, t in enumerate(texts):
            row = self.db.execute("SELECT vec FROM emb WHERE key=?", (self._key(t),)).fetchone()
            if row is None:
                missing.append(i)
            else:
                out[i] = np.frombuffer(row[0], dtype=np.float32)
        if missing:
            fresh = self.inner.encode([texts[i] for i in missing])
            for j, i in enumerate(missing):
                out[i] = fresh[j]
                self.db.execute("INSERT OR REPLACE INTO emb VALUES (?, ?)", (self._key(texts[i]), fresh[j].tobytes()))
            self.db.commit()
        return out


_MODEL_CACHE: dict[tuple[str, str | None], SentenceTransformerEmbedder] = {}


def get_embedder(name: str, dim: int = 256, cache_dir: str | None = None, device: str | None = None) -> Embedder:
    if name == "hash":
        return HashEmbedder(dim)
    # build_system runs once per question; reloading a 2 GB encoder from disk each time cost ~15 s
    # a question. The model is stateless, so one instance per (name, device) serves the whole process.
    key = (name, device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = SentenceTransformerEmbedder(name, device=device)
    model = _MODEL_CACHE[key]
    if cache_dir:
        return CachedEmbedder(model, name, f"{cache_dir}/{name.replace('/', '__')}.sqlite")
    return model
