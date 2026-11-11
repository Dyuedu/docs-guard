"""Canonical train/val/test split by source image, shared by OCR/KIE and tampering.

`data/processed/splits.json` = {"seed": 42, "fractions": [...], "train": [...], "val": [...], "test": [...]}
Every downstream dataset (tamper set, rec crops, det pseudo-labels, calibration) must read
this file so the same receipt never lands in two splits.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SPLITS = ("train", "val", "test")


def make_splits(image_ids: list[str], fractions: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42) -> dict:
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be 3 values summing to 1, got {fractions}")
    ids = sorted(set(image_ids))  # sort first so the result depends only on the id set + seed
    random.Random(seed).shuffle(ids)
    n = len(ids)
    cut1 = int(round(fractions[0] * n))
    cut2 = int(round((fractions[0] + fractions[1]) * n))
    return {"seed": seed, "fractions": list(fractions), "train": ids[:cut1], "val": ids[cut1:cut2], "test": ids[cut2:]}


def save_splits(splits: dict, path: str | Path) -> Path:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=1)
    return Path(path)


def load_splits(path: str | Path) -> dict[str, set[str]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {s: set(d[s]) for s in SPLITS}
    overlap = (out["train"] & out["val"]) | (out["train"] & out["test"]) | (out["val"] & out["test"])
    if overlap:
        raise ValueError(f"splits overlap on {len(overlap)} images, e.g. {sorted(overlap)[:3]}")
    return out


def split_of(splits: dict[str, set[str]], image_id: str) -> str | None:
    return next((s for s in SPLITS if image_id in splits[s]), None)
