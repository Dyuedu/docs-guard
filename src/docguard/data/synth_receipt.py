"""Synthetic full-page receipt photos with per-line polygons (intent 002, for DBNet).

Design and effect parameters follow SynthDoG (NAVER Corp., MIT,
https://github.com/clovaai/donut/tree/master/synthdog, `config_en.yaml`):
background -> paper -> content -> document effects (perspective) -> photo effects
(shadow, contrast, brightness, blur, noise). Differences: a receipt-specific layout
(header / item table / totals / footer, thermal-printer char grid) replaces SynthDoG's grid
stack, and every text line's quad is exported after the geometric transform.

Implemented with Pillow + OpenCV only (synthtiger does not work with numpy 2).
"""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_CFG = {
    "long_side": [768, 1280],       # canvas long side, px
    "aspect": [1.25, 2.0],          # long / short
    "text_px": [13, 26],            # font pixel size at long side 1024 (scaled with canvas)
    "line_spacing": [1.15, 1.6],
    "paper_margin": [0.03, 0.08],   # fraction of paper width
    "paper_fill": [0.45, 0.92],     # paper height as fraction of canvas height (before rotation)
    "rotate90_prob": 0.3,
    "tilt_deg": [-5, 5],
    "perspective_prob": 0.6,
    "perspective_jitter": 0.05,     # fraction of paper size
    "elastic_prob": 0.15,
    "shadow_prob": 0.7,
    "noise_prob": 0.8,
    "blur_prob": 0.6,
    "motion_blur_prob": 0.3,
    "color_cast_prob": 0.3,
}

_MONO_HINTS = ("mono", "plex", "code", "courier")


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_convex(subject: list, clip: np.ndarray) -> list:
    """Sutherland–Hodgman: clip polygon `subject` by convex CCW polygon `clip` (textbook algorithm)."""
    out = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])  # >= 0: inside
        for j in range(len(inp)):
            p, q = inp[j], inp[(j + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def polygon_iou(a, b) -> float:
    """IoU of two convex quads (float coords).

    Not cv2.intersectConvexConvex: it returns the full area of one quad for nearly parallel,
    touching edges (seen on adjacent tilted text lines), so clip with Sutherland–Hodgman instead.
    """
    pa, pb = np.asarray(a, np.float64), np.asarray(b, np.float64)
    sa, sb = _signed_area(pa), _signed_area(pb)
    if abs(sa) <= 0 or abs(sb) <= 0:
        return 0.0
    if sb < 0:
        pb = pb[::-1]  # clip polygon must be counter-clockwise for the inside test
    inter_poly = _clip_convex([tuple(p) for p in pa], pb)
    inter = abs(_signed_area(np.asarray(inter_poly))) if len(inter_poly) >= 3 else 0.0
    union = abs(sa) + abs(sb) - inter
    return float(inter / union) if union > 0 else 0.0


def _rand(rng: random.Random, lo_hi) -> float:
    return rng.uniform(lo_hi[0], lo_hi[1])


def _np(rng: random.Random) -> np.random.Generator:
    """numpy generator seeded from the Python rng: fast noise fields, still reproducible."""
    return np.random.default_rng(rng.getrandbits(32))


def _field(rng: random.Random, gh: int, gw: int, w: int, h: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Low-frequency random field in [-1, 1], upsampled from a gh×gw grid to h×w."""
    return cv2.resize(_np(rng).uniform(-1, 1, (gh, gw)).astype(np.float32), (w, h), interpolation=interp)


# --- paper + text --------------------------------------------------------------------------------

def _paper_texture(w: int, h: int, rng: random.Random) -> np.ndarray:
    base = rng.randint(225, 255)
    paper = np.full((h, w, 3), base, np.float32)
    paper[..., 2] -= rng.randint(0, 18)        # slight warm/yellow tint (BGR order not used; RGB here)
    paper[..., 1] -= rng.randint(0, 8)
    # low-frequency shading (crumples / uneven lighting)
    paper += _field(rng, 8, 6, w, h, cv2.INTER_CUBIC)[..., None] * rng.uniform(0, 14)
    if rng.random() < 0.3:  # faint horizontal print bands typical of thermal printers
        bands = (np.arange(h) % rng.randint(3, 9) == 0).astype(np.float32) * rng.uniform(0, 5)
        paper -= bands[:, None, None]
    return np.clip(paper, 0, 255).astype(np.uint8)


def _is_header(i: int, text: str, label: str) -> bool:
    return label == "SELLER" or (i <= 5 and not any(c.isdigit() for c in text) and text.upper() == text and len(text) <= 36)


def _render_paper(lines, fonts, rng, cfg, text_px: int):
    """Draw lines on a paper layer; return (paper RGB, alpha, per-line quads in paper coords)."""
    mono = [f for f in fonts if any(h in f.lower() for h in _MONO_HINTS)]
    body_font_path = rng.choice(mono) if mono and rng.random() < 0.85 else rng.choice(fonts)
    body = ImageFont.truetype(body_font_path, text_px)
    head = ImageFont.truetype(rng.choice(fonts), int(text_px * rng.uniform(1.0, 1.5)))

    spacing = _rand(rng, cfg["line_spacing"])
    metrics = []
    for i, (text, label) in enumerate(lines):
        f = head if _is_header(i, text, label) else body
        left, top, right, bottom = f.getbbox(text)
        metrics.append((f, left, top, right - left, bottom - top))
    max_w = max(m[3] for m in metrics)
    margin = int(max_w * _rand(rng, cfg["paper_margin"])) + 4
    paper_w = max_w + 2 * margin
    line_h = int(text_px * spacing)
    pad = max(1, text_px // 8)
    # lay out first (incl. random blank separators), then size the paper to fit every line + its pad
    ys, y = [], margin + int(text_px * 0.3)
    for m in metrics:
        ys.append(y)
        y += max(m[4], line_h) + (line_h - text_px)
        if rng.random() < 0.06:  # blank line / separator
            y += line_h // 2
    paper_h = max(yy + m[4] for yy, m in zip(ys, metrics)) + pad + margin

    paper = Image.fromarray(_paper_texture(paper_w, paper_h, rng))
    draw = ImageDraw.Draw(paper)
    ink = rng.randint(0, 70)
    ink_rgb = (ink, ink + rng.randint(0, 15), ink + rng.randint(0, 25))
    quads = []
    for i, ((text, label), (f, left, top, tw, th), y) in enumerate(zip(lines, metrics, ys)):
        if _is_header(i, text, label):
            x = (paper_w - tw) // 2                              # centred header
        elif rng.random() < 0.08:
            x = paper_w - margin - tw                            # occasionally right-aligned
        else:
            x = margin
        draw.text((x - left, y - top), text, font=f, fill=ink_rgb)
        quads.append([[x - pad, y - pad], [x + tw + pad, y - pad], [x + tw + pad, y + th + pad], [x - pad, y + th + pad]])
    arr = np.asarray(paper).astype(np.float32)
    if rng.random() < 0.5:  # thermal print is often slightly faded / uneven
        fade = (_field(rng, 6, 4, paper_w, paper_h) + 1) * 0.175  # [0, 0.35]
        arr = arr + (255 - arr) * fade[..., None]
    return np.clip(arr, 0, 255).astype(np.uint8), np.full((paper_h, paper_w), 255, np.uint8), np.asarray(quads, np.float32)


# --- background + composition ---------------------------------------------------------------------

def _background(w: int, h: int, rng: random.Random) -> np.ndarray:
    """Mostly neutral surfaces like the MC-OCR photos: tiles, wood, desks, cloth."""
    kind = rng.random()
    gray = rng.randint(60, 225)
    tint = np.array([rng.uniform(-25, 25) for _ in range(3)], np.float32)  # mild hue only
    if rng.random() < 0.35:  # warm wood / beige
        tint = np.array([rng.uniform(10, 45), rng.uniform(0, 20), rng.uniform(-35, -5)], np.float32)
    base = np.clip(gray + tint, 0, 255)
    bg = np.empty((h, w, 3), np.float32)
    bg[:] = base
    if kind < 0.35:  # tiles / grid (floor, table cloth)
        cell = rng.randint(60, 220)
        yy, xx = np.mgrid[0:h, 0:w]
        grout = ((xx % cell) < 3) | ((yy % cell) < 3)
        bg[grout] = base * rng.uniform(0.5, 0.9)
        bg += _field(rng, 10, 10, w, h)[..., None] * rng.uniform(0, 12)
    elif kind < 0.65:  # wood-like streaks (horizontal or vertical)
        streak = _field(rng, 3, 40, w, h) if rng.random() < 0.5 else _field(rng, 40, 3, w, h)
        bg += streak[..., None] * rng.uniform(5, 30)
    bg += _field(rng, 3, 3, w, h, cv2.INTER_CUBIC)[..., None] * rng.uniform(0, 40)   # illumination
    bg += cv2.resize(_np(rng).normal(0, 1, (h // 8 + 1, w // 8 + 1)).astype(np.float32), (w, h))[..., None] * rng.uniform(0, 10)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _homography(paper_w, paper_h, canvas_w, canvas_h, rng, cfg, rot90: bool):
    """Scale + rotate + translate (+ perspective jitter) keeping the paper inside the canvas."""
    angle = (90.0 if rng.random() < 0.5 else -90.0) if rot90 else 0.0
    angle += _rand(rng, cfg["tilt_deg"])
    fill = _rand(rng, cfg["paper_fill"])
    # paper height should fill `fill` of the canvas height (or width when rotated)
    target = (canvas_w if rot90 else canvas_h) * fill
    scale = target / paper_h
    scale = min(scale, 0.95 * (canvas_h if rot90 else canvas_w) / paper_w)  # width must fit too
    corners = np.float32([[0, 0], [paper_w, 0], [paper_w, paper_h], [0, paper_h]])
    R = cv2.getRotationMatrix2D((0, 0), angle, scale)
    pts = cv2.transform(corners[None], R)[0]
    if rng.random() < cfg["perspective_prob"]:
        j = cfg["perspective_jitter"] * max(paper_w, paper_h) * scale
        pts = pts + np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
    # translate so that the quad fits inside the canvas with a random offset
    minx, miny = pts[:, 0].min(), pts[:, 1].min()
    maxx, maxy = pts[:, 0].max(), pts[:, 1].max()
    bw, bh = maxx - minx, maxy - miny
    if bw > canvas_w or bh > canvas_h:  # shrink further if jitter/rotation pushed it out
        s = min(canvas_w / bw, canvas_h / bh) * 0.98
        pts = (pts - [minx, miny]) * s
        minx, miny = 0.0, 0.0
        bw, bh = bw * s, bh * s
    ox = rng.uniform(0, canvas_w - bw) - minx
    oy = rng.uniform(0, canvas_h - bh) - miny
    pts = pts + [ox, oy]
    return cv2.getPerspectiveTransform(corners, pts.astype(np.float32))


def _photo_effects(img: np.ndarray, rng: random.Random, cfg: dict) -> np.ndarray:
    arr = img.astype(np.float32)
    h, w = arr.shape[:2]
    if rng.random() < cfg["color_cast_prob"]:
        arr += np.asarray([rng.uniform(-18, 18) for _ in range(3)], np.float32)
    arr = arr * rng.uniform(0.85, 1.25) + rng.uniform(-30, 10)            # contrast / brightness
    if rng.random() < cfg["noise_prob"]:
        arr += _np(rng).normal(0, 1, (h, w)).astype(np.float32)[..., None] * rng.uniform(1, 7)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if rng.random() < cfg["blur_prob"]:
        arr = cv2.GaussianBlur(arr, (0, 0), rng.uniform(0.3, 1.4))
    if rng.random() < cfg["motion_blur_prob"]:
        k = rng.choice([3, 5])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
        kern = cv2.warpAffine(kern, M, (k, k))
        kern /= max(kern.sum(), 1e-6)
        arr = cv2.filter2D(arr, -1, kern)
    return arr


def render_receipt(lines: list[tuple[str, str]], fonts: list[str], rng: random.Random, cfg: dict | None = None):
    """Render one receipt photo. Returns (RGB uint8 image, boxes) with boxes =
    [{"polygon": [[x,y]*4], "bbox": [x1,y1,x2,y2], "text": str, "label": str}, ...] in line order."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    long_side = int(_rand(rng, cfg["long_side"]))
    short_side = int(long_side / _rand(rng, cfg["aspect"]))
    rot90 = rng.random() < cfg["rotate90_prob"]  # receipt lying sideways -> landscape photo
    canvas_w, canvas_h = (long_side, short_side) if rot90 else (short_side, long_side)
    text_px = max(8, int(round(_rand(rng, cfg["text_px"]) * long_side / 1024)))

    paper, alpha, quads = _render_paper(lines, fonts, rng, cfg, text_px)
    H = _homography(paper.shape[1], paper.shape[0], canvas_w, canvas_h, rng, cfg, rot90)

    bg = _background(canvas_w, canvas_h, rng)
    warped = cv2.warpPerspective(paper, H, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = cv2.warpPerspective(alpha, H, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32) / 255.0
    if rng.random() < cfg["shadow_prob"]:  # soft drop shadow offset from the paper
        dx, dy = rng.randint(-12, 12), rng.randint(2, 16)
        sh = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
        sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(4, 14)) * rng.uniform(0.2, 0.6)
        bg = np.clip(bg.astype(np.float32) * (1 - sh[..., None]), 0, 255).astype(np.uint8)
    a3 = a[..., None]
    comp = (warped.astype(np.float32) * a3 + bg.astype(np.float32) * (1 - a3))
    img = np.clip(comp, 0, 255).astype(np.uint8)

    polys = cv2.perspectiveTransform(quads.reshape(-1, 1, 2), H).reshape(-1, 4, 2)
    if rng.random() < cfg["elastic_prob"]:  # mild elastic warp (SynthDoG alpha<=1): polygons stay approx. valid
        dxs = _field(rng, 6, 5, canvas_w, canvas_h) * 2.0
        dys = _field(rng, 6, 5, canvas_w, canvas_h) * 2.0
        yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
        img = cv2.remap(img, xx + dxs, yy + dys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    img = _photo_effects(img, rng, cfg)

    polys[..., 0] = np.clip(polys[..., 0], 0, canvas_w)
    polys[..., 1] = np.clip(polys[..., 1], 0, canvas_h)
    boxes = []
    for (text, label), p in zip(lines, polys):
        poly = [[round(float(x), 1), round(float(y), 1)] for x, y in p]
        xs, ys = [q[0] for q in poly], [q[1] for q in poly]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(np.ceil(max(xs))), int(np.ceil(max(ys)))
        boxes.append({"polygon": poly, "bbox": [x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)], "text": text, "label": label})
    return img, boxes
