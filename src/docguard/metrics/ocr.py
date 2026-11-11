"""OCR / KIE metrics (proposal §7)."""

from __future__ import annotations

import re
import unicodedata


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds: list[str], refs: list[str]) -> float:
    """Corpus-level character error rate: total edits / total reference characters."""
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    edits = sum(levenshtein(unicodedata.normalize("NFC", p), unicodedata.normalize("NFC", r)) for p, r in zip(preds, refs))
    total = sum(len(r) for r in refs)
    return edits / total if total else 0.0


def normalize_field(value: str | None) -> str:
    if value is None:
        return ""
    v = unicodedata.normalize("NFC", value).upper()
    return re.sub(r"\s+", "", v)


def field_exact_match(preds: list[dict[str, str | None]], refs: list[dict[str, str | None]]) -> dict[str, float]:
    """Per-field and overall exact match after normalization. Fields missing in refs are skipped."""
    hits: dict[str, int] = {}
    counts: dict[str, int] = {}
    for p, r in zip(preds, refs):
        for k, ref in r.items():
            if ref is None:
                continue
            counts[k] = counts.get(k, 0) + 1
            hits[k] = hits.get(k, 0) + (normalize_field(p.get(k)) == normalize_field(ref))
    out = {k: hits[k] / counts[k] for k in counts}
    out["overall"] = sum(hits.values()) / sum(counts.values()) if counts else 0.0
    return out
