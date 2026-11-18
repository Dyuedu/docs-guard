"""DB label generation: shrink (probability) map and threshold (border) map.

Ported from PaddleOCR (Apache-2.0), which in turn follows WenmuZhou/DBNet.pytorch:
  ppocr/data/imaug/make_shrink_map.py  (MakeShrinkMap)
  ppocr/data/imaug/make_border_map.py  (MakeBorderMap)
Changes: functional interface; shapely replaced by numpy shoelace area / perimeter; polygons are
not modified in place; dynamic shrink-ratio schedule removed. See NOTICE.
"""

from __future__ import annotations

import cv2
import numpy as np
import pyclipper


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _perimeter(poly: np.ndarray) -> float:
    return float(np.linalg.norm(poly - np.roll(poly, -1, axis=0), axis=1).sum())


def _offset(poly: np.ndarray, distance: float) -> list:
    pco = pyclipper.PyclipperOffset()
    pco.AddPath([tuple(p) for p in poly], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    return pco.Execute(distance)


def make_shrink_map(
    polys: list[np.ndarray], ignore: list[bool], h: int, w: int,
    shrink_ratio: float = 0.4, min_text_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (shrink_map, shrink_mask), float32 HxW. Ignored / too small polygons get mask 0."""
    gt = np.zeros((h, w), dtype=np.float32)
    mask = np.ones((h, w), dtype=np.float32)
    for poly, ign in zip(polys, ignore):
        poly = np.asarray(poly, dtype=np.float64).copy()
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        ph = poly[:, 1].max() - poly[:, 1].min()
        pw = poly[:, 0].max() - poly[:, 0].min()
        area = _area(poly)
        if ign or area < 1 or min(ph, pw) < min_text_size:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 0)
            continue
        shrunk = []
        # Paddle: retry with a weaker shrink when the offset splits the polygon.
        for ratio in np.arange(shrink_ratio, 1, shrink_ratio):
            distance = area * (1 - ratio**2) / _perimeter(poly)
            shrunk = _offset(poly, -distance)
            if len(shrunk) == 1:
                break
        if not shrunk:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 0)
            continue
        for s in shrunk:
            cv2.fillPoly(gt, [np.array(s).reshape(-1, 2).astype(np.int32)], 1)
    return gt, mask


def _point_to_segment(xs: np.ndarray, ys: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Distance from each (x, y) to segment p1-p2 (Paddle MakeBorderMap._distance)."""
    sq1 = np.square(xs - p1[0]) + np.square(ys - p1[1])
    sq2 = np.square(xs - p2[0]) + np.square(ys - p2[1])
    sq = np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        cosin = (sq - sq1 - sq2) / (2 * np.sqrt(sq1 * sq2))
        square_sin = np.nan_to_num(1 - np.square(cosin))
        result = np.sqrt(sq1 * sq2 * square_sin / sq)
    obtuse = cosin < 0
    result[obtuse] = np.sqrt(np.fmin(sq1, sq2))[obtuse]
    return np.nan_to_num(result)


def _draw_border(poly: np.ndarray, canvas: np.ndarray, mask: np.ndarray, shrink_ratio: float) -> None:
    area = _area(poly)
    if area <= 0:
        return
    distance = area * (1 - shrink_ratio**2) / _perimeter(poly)
    padded = _offset(poly, distance)
    if not padded:
        return
    padded = np.array(padded[0])
    cv2.fillPoly(mask, [padded.astype(np.int32)], 1.0)

    xmin, xmax = int(padded[:, 0].min()), int(padded[:, 0].max())
    ymin, ymax = int(padded[:, 1].min()), int(padded[:, 1].max())
    width, height = xmax - xmin + 1, ymax - ymin + 1
    local = poly - np.array([xmin, ymin], dtype=poly.dtype)
    xs = np.broadcast_to(np.arange(width, dtype=np.float32).reshape(1, width), (height, width))
    ys = np.broadcast_to(np.arange(height, dtype=np.float32).reshape(height, 1), (height, width))
    dist = np.ones((height, width), dtype=np.float32)
    for i in range(len(local)):
        j = (i + 1) % len(local)
        dist = np.fmin(dist, np.clip(_point_to_segment(xs, ys, local[i], local[j]) / distance, 0, 1))

    h, w = canvas.shape
    x0, x1 = min(max(0, xmin), w - 1), min(max(0, xmax), w - 1)
    y0, y1 = min(max(0, ymin), h - 1), min(max(0, ymax), h - 1)
    canvas[y0 : y1 + 1, x0 : x1 + 1] = np.fmax(
        1 - dist[y0 - ymin : y1 - ymax + height, x0 - xmin : x1 - xmax + width],
        canvas[y0 : y1 + 1, x0 : x1 + 1],
    )


def make_border_map(
    polys: list[np.ndarray], ignore: list[bool], h: int, w: int,
    shrink_ratio: float = 0.4, thresh_min: float = 0.3, thresh_max: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (thresh_map, thresh_mask): thresh_min + (max - min) * (1 - clip(dist / D, 0, 1))."""
    canvas = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    for poly, ign in zip(polys, ignore):
        if ign:
            continue
        poly = np.asarray(poly, dtype=np.float64)
        if np.isnan(poly).any():
            continue
        _draw_border(poly, canvas, mask, shrink_ratio)
    return canvas * (thresh_max - thresh_min) + thresh_min, mask
