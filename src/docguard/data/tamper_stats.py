"""Validation + statistics for the synthetic tampering dataset (intent 001, spec AC-1..AC-4).

Layout produced by scripts/make_tamper_dataset.py:
    {root}/{split}/images/{name}.jpg, {root}/{split}/masks/{name}.png, {root}/{split}/index.jsonl
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TAMPER_OPS = ("copy_move", "splice", "inpaint")

# Thresholds from intent/001-tamper-dataset/spec.md (AC-3, AC-4).
DEFAULT_CFG = {
    "op_min_ratio": 0.20,
    "area_median_min": 0.0001,
    "area_median_max": 0.01,
    "outlier_area_ratio": 0.05,
}


def load_index(split_dir: str | Path) -> list[dict]:
    with open(Path(split_dir) / "index.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def leak_check(index_by_split: dict[str, list[dict]], splits: dict[str, set[str]]) -> list[str]:
    """Return human-readable leak errors; [] means OK (AC-1)."""
    errors: list[str] = []
    seen: dict[str, set[str]] = defaultdict(set)
    for split, rows in index_by_split.items():
        allowed = splits.get(split, set())
        for r in rows:
            seen[r["source"]].add(split)
            if r["source"] not in allowed:
                errors.append(f"{split}/{r['name']}: source {r['source']} is not in canonical split '{split}'")
            donor = r.get("donor")
            if donor is not None and donor not in allowed:
                errors.append(f"{split}/{r['name']}: donor {donor} is not in canonical split '{split}'")
    for source, in_splits in sorted(seen.items()):
        if len(in_splits) > 1:
            errors.append(f"source {source} appears in several splits: {sorted(in_splits)}")
    return errors


def mask_stats(split_dir: str | Path, rows: list[dict]) -> list[dict]:
    """Per-image mask validity and area ratio (AC-2)."""
    split_dir = Path(split_dir)
    out = []
    for r in rows:
        s = {"name": r["name"], "op": r["op"], "h": 0, "w": 0, "area_ratio": 0.0, "valid": False, "error": ""}
        img_p, mask_p = split_dir / "images" / f"{r['name']}.jpg", split_dir / "masks" / f"{r['name']}.png"
        if not img_p.exists() or not mask_p.exists():
            s["error"] = "missing image or mask file"
            out.append(s)
            continue
        with Image.open(img_p) as im:  # header only, no full decode
            w, h = im.size
        mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
        s["h"], s["w"] = h, w
        if mask is None or mask.ndim != 2:
            s["error"] = "mask unreadable or not single-channel"
        elif mask.shape != (h, w):
            s["error"] = f"mask shape {mask.shape} != image {(h, w)}"
        elif not set(np.unique(mask).tolist()) <= {0, 255}:
            s["error"] = "mask values outside {0,255}"
        else:
            s["area_ratio"] = float((mask > 0).mean())
            if r["tampered"] and s["area_ratio"] == 0:
                s["error"] = "tampered image has empty mask"
            elif not r["tampered"] and s["area_ratio"] > 0:
                s["error"] = "authentic image has non-zero mask"
            else:
                s["valid"] = True
        out.append(s)
    return out


def _pct(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values)
    return {"p5": float(np.percentile(a, 5)), "p50": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "n": int(a.size)}


def summarize(
    index_by_split: dict[str, list[dict]],
    stats_by_split: dict[str, list[dict]],
    leak_errors: list[str],
    cfg: dict | None = None,
) -> dict:
    """Counts, op ratios, mask-area percentiles, outliers and pass/fail checks (AC-1..AC-4)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    splits_out, outliers, invalid = {}, [], []
    op_ok, area_ok = True, True

    for split, rows in index_by_split.items():
        stats = stats_by_split.get(split, [])
        fake = [r for r in rows if r["tampered"]]
        op_counts = Counter(r["op"] for r in fake)
        op_ratio = {op: (op_counts[op] / len(fake) if fake else 0.0) for op in TAMPER_OPS}
        fake_stats = [s for s in stats if s["op"] in TAMPER_OPS]
        areas = [s["area_ratio"] for s in fake_stats if s["valid"]]
        area_by_op = {op: _pct([s["area_ratio"] for s in fake_stats if s["valid"] and s["op"] == op]) for op in TAMPER_OPS}

        if fake:
            op_ok &= all(op_ratio[op] >= cfg["op_min_ratio"] for op in TAMPER_OPS)
        if areas:
            med = float(np.median(areas))
            area_ok &= cfg["area_median_min"] <= med <= cfg["area_median_max"]
        outliers += [{"split": split, "name": s["name"], "op": s["op"], "area_ratio": s["area_ratio"]}
                     for s in fake_stats if s["valid"] and s["area_ratio"] > cfg["outlier_area_ratio"]]
        invalid += [{"split": split, "name": s["name"], "error": s["error"]} for s in stats if not s["valid"]]

        splits_out[split] = {
            "n_images": len(rows),
            "n_authentic": len(rows) - len(fake),
            "n_tampered": len(fake),
            "n_sources": len({r["source"] for r in rows}),
            "op_counts": {op: op_counts[op] for op in TAMPER_OPS},
            "op_ratio": op_ratio,
            "area_ratio": {"all": _pct(areas), **area_by_op},
            "fonts": dict(Counter(r["font"] for r in fake if "font" in r)),
        }

    return {
        "config": cfg,
        "splits": splits_out,
        "outliers": sorted(outliers, key=lambda o: -o["area_ratio"]),
        "invalid": invalid,
        "leak_errors": leak_errors,
        "checks": {
            "no_leak": not leak_errors,
            "masks_valid": not invalid,
            "op_balance": op_ok,
            "mask_area": area_ok,
        },
    }


# --- visual review (AC-7) -------------------------------------------------------------------------

_PANEL = 360


def _fit(img: np.ndarray, size: int = _PANEL) -> np.ndarray:
    """Letterbox to size×size on a grey canvas."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    canvas = np.full((size, size, 3), 128, np.uint8)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    canvas[y : y + r.shape[0], x : x + r.shape[1]] = r
    return canvas


def _tile(split_dir: Path, row: dict, crop_pad: int) -> np.ndarray:
    img = cv2.imread(str(split_dir / "images" / f"{row['name']}.jpg"))
    mask = cv2.imread(str(split_dir / "masks" / f"{row['name']}.png"), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    ys, xs = np.nonzero(mask)
    if len(xs):
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    else:
        x1, y1, x2, y2 = w // 3, h // 3, 2 * w // 3, 2 * h // 3
    full = img.copy()
    cv2.rectangle(full, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), max(2, w // 300))
    cx1, cy1 = max(0, x1 - crop_pad), max(0, y1 - crop_pad)
    cx2, cy2 = min(w, x2 + crop_pad), min(h, y2 + crop_pad)
    crop = img[cy1:cy2, cx1:cx2]
    overlay = crop.copy()
    overlay[mask[cy1:cy2, cx1:cx2] > 0] = (0, 0, 255)
    blend = cv2.addWeighted(crop, 0.5, overlay, 0.5, 0)
    tile = np.hstack([_fit(full), _fit(crop), _fit(blend)])
    label = f"{row['name']}  op={row['op']}  q={row.get('jpeg_q')}  {row.get('font') or row.get('donor') or ''}"
    bar = np.full((28, tile.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label[:110], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def preview_grid(split_dir: str | Path, rows: list[dict], out_path: str | Path, crop_pad: int = 48) -> None:
    """One row per sample: [full image + red bbox | zoomed crop | crop with mask overlay]."""
    split_dir, out_path = Path(split_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.vstack([_tile(split_dir, r, crop_pad) for r in rows])
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
