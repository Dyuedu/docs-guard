"""Synthetic document tampering with pixel-accurate masks (proposal §4).

Each operation returns (tampered_rgb, mask) where mask is uint8 {0,255} marking
exactly the pixels that were modified. Targets are boxes whose text contains
digits (amounts, dates), since that is what fraudsters edit.
"""

from __future__ import annotations

import random
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .annotations import TextBox

_DIGIT_RE = re.compile(r"\d")


def digit_boxes(boxes: list[TextBox], min_h: int = 8) -> list[TextBox]:
    return [b for b in boxes if _DIGIT_RE.search(b.text) and b.bbox[3] - b.bbox[1] >= min_h and b.bbox[2] > b.bbox[0]]


def _ink_mask(region: np.ndarray) -> np.ndarray:
    """Dark text pixels inside a light-background region (Otsu), dilated slightly."""
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)


def _sub_box(box: list[int], rng: random.Random, min_frac: float = 0.25) -> list[int]:
    """Random horizontal slice of a text box (e.g. a few digits of an amount)."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    sw = max(4, int(w * rng.uniform(min_frac, 0.6)))
    sx = x1 + rng.randint(0, max(0, w - sw))
    return [sx, y1, min(x2, sx + sw), y2]


def _paste(dst: np.ndarray, patch: np.ndarray, box: list[int]) -> np.ndarray:
    """Paste patch into dst at box; returns mask of changed pixels."""
    x1, y1, x2, y2 = box
    patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    mask = np.zeros(dst.shape[:2], np.uint8)
    region = dst[y1:y2, x1:x2]
    changed = np.any(region != patch, axis=-1)
    dst[y1:y2, x1:x2] = patch
    mask[y1:y2, x1:x2][changed] = 255
    return mask


def copy_move(img: np.ndarray, boxes: list[TextBox], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    cands = digit_boxes(boxes)
    if len(cands) < 2:
        raise ValueError("need at least two digit boxes for copy-move")
    src, dst = rng.sample(cands, 2)
    sb, db = _sub_box(src.bbox, rng), _sub_box(dst.bbox, rng)
    out = img.copy()
    mask = _paste(out, img[sb[1] : sb[3], sb[0] : sb[2]].copy(), db)
    return out, mask


def splice(
    img: np.ndarray, boxes: list[TextBox], donor: np.ndarray, donor_boxes: list[TextBox], rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    cands, dcands = digit_boxes(boxes), digit_boxes(donor_boxes)
    if not cands or not dcands:
        raise ValueError("need digit boxes in both target and donor")
    db = _sub_box(rng.choice(cands).bbox, rng)
    sb = _sub_box(rng.choice(dcands).bbox, rng)
    out = img.copy()
    mask = _paste(out, donor[sb[1] : sb[3], sb[0] : sb[2]].copy(), db)
    return out, mask


def inpaint_and_render(
    img: np.ndarray, boxes: list[TextBox], font_path: str, rng: random.Random
) -> tuple[np.ndarray, np.ndarray, str]:
    """Erase the digits of one box and render a new number in its place. Returns new text too."""
    cands = digit_boxes(boxes)
    if not cands:
        raise ValueError("no digit boxes")
    tb = rng.choice(cands)
    x1, y1, x2, y2 = tb.bbox
    region = img[y1:y2, x1:x2]
    ink = _ink_mask(region)

    out = img.copy()
    full_ink = np.zeros(img.shape[:2], np.uint8)
    full_ink[y1:y2, x1:x2] = ink
    out = cv2.inpaint(out, full_ink, 3, cv2.INPAINT_TELEA)

    new_text = re.sub(r"\d", lambda _: str(rng.randint(0, 9)), tb.text)
    h = y2 - y1
    font = ImageFont.truetype(font_path, max(8, int(h * 0.8)))
    ink_color = tuple(int(v) for v in np.median(region[ink > 0], axis=0)) if (ink > 0).any() else (0, 0, 0)
    pil = Image.fromarray(out)
    ImageDraw.Draw(pil).text((x1, y1 + int(h * 0.05)), new_text, font=font, fill=ink_color)
    out = np.asarray(pil).copy()

    changed = np.any(out != img, axis=-1).astype(np.uint8) * 255
    return out, changed, new_text


def jpeg_recompress(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
