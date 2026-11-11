"""Rule-based key information extraction for Vietnamese receipts (baseline KIE).

Input is a list of OCR lines with boxes in original-image coordinates. Rules are
deliberately simple so the KIE step adds ~0 ms and the comparison between model
variants isolates the effect of OCR quality.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class OcrLine:
    text: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float = 1.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class Field:
    value: str | None
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


_DATE_RE = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})")
_AMOUNT_RE = re.compile(r"\d{1,3}(?:[.,\s]\d{3})+|\d{4,}")
_TOTAL_KEYWORDS = (
    "tong cong", "tong tien", "tong thanh toan", "thanh toan", "tong", "total", "cong tien",
)
# Lines that look like totals but are not the amount payable.
_TOTAL_EXCLUDE = ("tien khach dua", "tien thua", "tra lai", "giam gia", "chiet khau", "so luong")


def normalize_amount(text: str) -> str | None:
    m = _AMOUNT_RE.findall(text)
    if not m:
        return None
    digits = re.sub(r"\D", "", m[-1])
    return digits.lstrip("0") or "0"


def normalize_date(text: str) -> str | None:
    m = _DATE_RE.search(text)
    if not m:
        return None
    d, mo, y = (int(g) for g in m.groups())
    if len(m.group(3)) == 2:
        y += 2000
    if not (1 <= d <= 31 and 1 <= mo <= 12 and 1990 <= y <= 2100):
        return None
    return f"{d:02d}/{mo:02d}/{y:04d}"


def _same_row(a: OcrLine, b: OcrLine) -> bool:
    return abs(a.cy - b.cy) <= 0.6 * max(a.height, b.height, 1)


def extract_timestamp(lines: list[OcrLine]) -> Field:
    for ln in lines:
        d = normalize_date(ln.text)
        if d:
            return Field(d, ln.confidence, ln.bbox)
    return Field(None, 0.0)


def extract_total(lines: list[OcrLine]) -> Field:
    candidates: list[Field] = []
    for ln in lines:
        key = strip_accents(ln.text).lower()
        if not any(k in key for k in _TOTAL_KEYWORDS) or any(k in key for k in _TOTAL_EXCLUDE):
            continue
        amount = normalize_amount(ln.text)
        if amount:
            candidates.append(Field(amount, ln.confidence, ln.bbox))
            continue
        # Value printed in a separate box on the same row, to the right of the keyword.
        right = [o for o in lines if o is not ln and _same_row(o, ln) and o.bbox[0] >= ln.bbox[0]]
        for o in sorted(right, key=lambda o: o.bbox[0]):
            amount = normalize_amount(o.text)
            if amount:
                candidates.append(Field(amount, min(ln.confidence, o.confidence), o.bbox))
                break
    if not candidates:
        return Field(None, 0.0)
    # Receipts usually print the final payable total lowest on the page.
    return max(candidates, key=lambda f: f.bbox[1] if f.bbox else -1)


def extract_seller(lines: list[OcrLine], image_height: int) -> Field:
    top = [ln for ln in lines if ln.bbox[3] <= 0.25 * image_height and sum(c.isalpha() for c in ln.text) >= 3]
    if not top:
        return Field(None, 0.0)
    # Store name is typically the tallest text near the top.
    best = max(top, key=lambda ln: (ln.height, -ln.bbox[1]))
    return Field(best.text.strip(), best.confidence, best.bbox)


def extract_fields(lines: list[OcrLine], image_height: int) -> dict[str, Field]:
    lines = sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    return {
        "seller": extract_seller(lines, image_height),
        "timestamp": extract_timestamp(lines),
        "total_amount": extract_total(lines),
    }
