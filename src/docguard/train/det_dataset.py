"""DBNet dataset: index.jsonl (synth polygons or MC-OCR bboxes) -> 640 crops + DB label maps.

EAST-style random crop ported from PaddleOCR (Apache-2.0), itself from WenmuZhou/DBNet.pytorch:
  ppocr/data/imaug/random_crop_data.py  (crop_area, EastRandomCropData)
Changes: numpy Generator instead of the global RNG (deterministic per (seed, epoch, index)),
functional interface. Scale / rotate augment and the label wiring follow spec 003. See NOTICE.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from docguard.pipeline import processing as P
from docguard.train.det_labels import make_border_map, make_shrink_map

IGNORE_TEXTS = {"", "###"}


# ---------------------------------------------------------------- EAST random crop (PaddleOCR port)

def _is_poly_outside_rect(poly: np.ndarray, x: float, y: float, w: float, h: float) -> bool:
    if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
        return True
    return poly[:, 1].max() < y or poly[:, 1].min() > y + h


def _split_regions(axis: np.ndarray) -> list[np.ndarray]:
    regions, start = [], 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            regions.append(axis[start:i])
            start = i
    return regions


def _random_select(rng: np.random.Generator, axis: np.ndarray, max_size: int) -> tuple[int, int]:
    xx = rng.choice(axis, size=2)
    return int(np.clip(xx.min(), 0, max_size - 1)), int(np.clip(xx.max(), 0, max_size - 1))


def _region_wise_random_select(rng: np.random.Generator, regions: list[np.ndarray]) -> tuple[int, int]:
    vals = [int(rng.choice(regions[i])) for i in rng.choice(len(regions), 2)]
    return min(vals), max(vals)


def crop_area(rng: np.random.Generator, h: int, w: int, polys: list[np.ndarray],
              min_crop_side_ratio: float = 0.1, max_tries: int = 10) -> tuple[int, int, int, int]:
    """Pick a crop (x, y, w, h) whose borders do not cut through any text polygon."""
    h_array = np.zeros(h, dtype=np.int32)
    w_array = np.zeros(w, dtype=np.int32)
    for points in polys:
        points = np.round(points).astype(np.int32)
        w_array[max(0, points[:, 0].min()) : points[:, 0].max()] = 1
        h_array[max(0, points[:, 1].min()) : points[:, 1].max()] = 1
    h_axis, w_axis = np.where(h_array == 0)[0], np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return 0, 0, w, h
    h_regions, w_regions = _split_regions(h_axis), _split_regions(w_axis)
    for _ in range(max_tries):
        xmin, xmax = (_region_wise_random_select(rng, w_regions) if len(w_regions) > 1
                      else _random_select(rng, w_axis, w))
        ymin, ymax = (_region_wise_random_select(rng, h_regions) if len(h_regions) > 1
                      else _random_select(rng, h_axis, h))
        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            continue
        if any(not _is_poly_outside_rect(p, xmin, ymin, xmax - xmin, ymax - ymin) for p in polys):
            return xmin, ymin, xmax - xmin, ymax - ymin
    return 0, 0, w, h


def east_random_crop(rng: np.random.Generator, image: np.ndarray, polys: list[np.ndarray],
                     ignore: list[bool], size: int, max_tries: int = 10, min_crop_side_ratio: float = 0.1):
    """Crop avoiding text, resize keeping aspect ratio into a size x size zero-padded canvas."""
    care = [p for p, ig in zip(polys, ignore) if not ig]
    cx, cy, cw, ch = crop_area(rng, image.shape[0], image.shape[1], care, min_crop_side_ratio, max_tries)
    scale = min(size / cw, size / ch)
    h, w = int(ch * scale), int(cw * scale)
    canvas = np.zeros((size, size, image.shape[2]), image.dtype)
    canvas[:h, :w] = cv2.resize(image[cy : cy + ch, cx : cx + cw], (w, h))
    out_p, out_i = [], []
    for p, ig in zip(polys, ignore):
        q = (p - (cx, cy)) * scale
        if not _is_poly_outside_rect(q, 0, 0, w, h):
            out_p.append(q)
            out_i.append(ig)
    return canvas, out_p, out_i


# ---------------------------------------------------------------- scale / rotate (spec 003)

def resize_long_side(image: np.ndarray, polys: list[np.ndarray], long_side: float):
    h, w = image.shape[:2]
    r = long_side / max(h, w)
    image = cv2.resize(image, (max(1, round(w * r)), max(1, round(h * r))))
    return image, [p * r for p in polys]


def rotate(image: np.ndarray, polys: list[np.ndarray], deg: float):
    """Rotate about the centre, expanding the canvas so no pixel is lost."""
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(np.ceil(h * sin + w * cos)), int(np.ceil(h * cos + w * sin))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    image = cv2.warpAffine(image, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return image, [np.hstack([p, np.ones((len(p), 1))]) @ m.T for p in polys]


# ---------------------------------------------------------------- dataset

def read_records(jsonl_paths: list[str], image_root: str | None) -> list[dict]:
    """Load records; image paths resolve against image_root, or the jsonl's folder when None."""
    out = []
    for jp in jsonl_paths:
        base = Path(image_root) if image_root else Path(jp).parent
        with open(jp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    rec["image_path"] = str(base / rec["image"])
                    out.append(rec)
    return out


def record_polys(rec: dict) -> tuple[list[np.ndarray], list[bool]]:
    """Polygon when present, else the 4 corners of bbox (MC-OCR); empty / ### text is ignored."""
    polys, ignore = [], []
    for b in rec["boxes"]:
        if b.get("polygon"):
            p = np.asarray(b["polygon"], dtype=np.float64)
        else:
            x1, y1, x2, y2 = b["bbox"]
            p = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
        polys.append(p)
        ignore.append(str(b.get("text", "")).strip() in IGNORE_TEXTS)
    return polys, ignore


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class DetDataset(torch.utils.data.Dataset):
    """train=True: dict of tensors (image 3xSxS + 4 label maps SxS); train=False: raw RGB + GT boxes."""

    def __init__(self, jsonl_paths: list[str], image_root: str | None, train: bool, cfg: dict, seed: int):
        self.records = read_records(list(jsonl_paths), image_root)
        self.train, self.cfg, self.seed = train, cfg, seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        """index may exceed len(): record = index % len, augmentation RNG keyed on (seed, index).

        The training sampler yields epoch * len + i, so each epoch gets fresh crops while staying
        deterministic and independent of worker state (persistent workers keep stale attributes).
        """
        if not self.train:
            rec = self.records[index]  # strict: iteration must stop at len()
            return {"image": load_rgb(rec["image_path"]), "boxes": [b["bbox"] for b in rec["boxes"]],
                    "labels": [b.get("label") for b in rec["boxes"]], "name": rec["image"]}
        rec = self.records[index % len(self.records)]
        rng = np.random.default_rng([self.seed, index])
        return self.make_train_sample(load_rgb(rec["image_path"]), *record_polys(rec), rng)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def make_train_sample(self, image, polys, ignore, rng) -> dict[str, torch.Tensor]:
        c = self.cfg
        lo, hi = c.get("scale_range", [0.7, 1.3])
        image, polys = resize_long_side(image, polys, c.get("input_size", 960) * rng.uniform(lo, hi))
        deg = c.get("rotate_deg", 0)
        if deg:
            image, polys = rotate(image, polys, rng.uniform(-deg, deg))
        size = c.get("crop_size", 640)
        image, polys, ignore = east_random_crop(rng, image, polys, ignore, size)
        r = c.get("shrink_ratio", 0.4)
        gt, gt_mask = make_shrink_map(polys, list(ignore), size, size, r, c.get("min_text_size", 8))
        th, th_mask = make_border_map(polys, list(ignore), size, size, r,
                                      c.get("thresh_min", 0.3), c.get("thresh_max", 0.7))
        return {
            "image": torch.from_numpy(np.ascontiguousarray(P.to_chw(image))),
            "shrink_map": torch.from_numpy(gt), "shrink_mask": torch.from_numpy(gt_mask),
            "thresh_map": torch.from_numpy(th), "thresh_mask": torch.from_numpy(th_mask),
        }


class EpochOffsetSampler(torch.utils.data.Sampler):
    """Infinite shuffled sampler yielding epoch * n + perm_epoch[i] (see DetDataset.__getitem__)."""

    def __init__(self, n: int, seed: int):
        self.n, self.seed = n, seed

    def __iter__(self):
        epoch = 0
        while True:
            g = torch.Generator().manual_seed(self.seed * 100003 + epoch)
            for i in torch.randperm(self.n, generator=g).tolist():
                yield epoch * self.n + i
            epoch += 1
