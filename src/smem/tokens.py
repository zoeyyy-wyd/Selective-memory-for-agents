"""Token counting. tiktoken when available; otherwise a length heuristic so the offline path has no
model dependency. Both budgets (B and R) are expressed in these units, so one counter is used everywhere."""

from __future__ import annotations

from functools import lru_cache

try:  # pragma: no cover - exercised implicitly
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    _enc = None


@lru_cache(maxsize=65536)
def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _enc is not None:
        # ShareGPT filler contains literal special-token strings; count them as ordinary text
        return len(_enc.encode(text, disallowed_special=()))
    return max(1, (len(text) + 3) // 4)
