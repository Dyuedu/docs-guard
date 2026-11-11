"""Generate the synthetic tampering dataset from normalized MC-OCR annotations (intent 001).

Splits come from the canonical split file (scripts/make_splits.py), so the same
receipt is in the same split here as in the OCR/KIE data. Every authentic image is
also written (re-compressed with the same JPEG distribution) so the classifier
cannot shortcut on "was re-encoded".

Usage:
    python scripts/make_tamper_dataset.py --ann data/processed/annotations.jsonl \
        --splits data/processed/splits.json --images data/raw/mcocr/train_images \
        --out data/tamper --font data/fonts --tampered-per-image 2 --tampered-per-image-train 4

Validate the output with scripts/tamper_report.py.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from docguard.data.annotations import Sample, read_jsonl
from docguard.data.splits import SPLITS, load_splits
from docguard.data.synth_tamper import copy_move, inpaint_and_render, jpeg_recompress, splice

OPS = ("copy_move", "splice", "inpaint")


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save(out_dir: Path, name: str, img: np.ndarray, mask: np.ndarray) -> None:
    cv2.imwrite(str(out_dir / "images" / f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(str(out_dir / "masks" / f"{name}.png"), mask)


def resolve_fonts(path: str | Path) -> list[str]:
    """A .ttf file, or a directory whose *.ttf files are rotated (sorted for determinism)."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    fonts = sorted(str(f) for f in p.glob("*.ttf")) if p.is_dir() else []
    if not fonts:
        raise FileNotFoundError(f"no .ttf font at {p}")
    return fonts


def generate(
    samples: list[Sample],
    splits: dict[str, set[str]],
    images_dir: str | Path,
    out_root: str | Path,
    fonts: list[str],
    per_image: dict[str, int],
    seed: int = 42,
    log: Callable[..., None] = print,
) -> dict[str, list[dict]]:
    """Write {out_root}/{split}/{images,masks,index.jsonl}; return the index rows per split.

    Deterministic for a fixed (samples order, splits, images, fonts, per_image, seed).
    Inpaint is only drawn when at least one font is given.
    """
    rng = random.Random(seed)
    images_dir, out_root = Path(images_dir), Path(out_root)
    ops = OPS if fonts else ("copy_move", "splice")
    result: dict[str, list[dict]] = {}

    for split in SPLITS:
        items = [s for s in samples if s.image in splits[split]]
        out = out_root / split
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        index: list[dict] = []
        skipped: Counter[str] = Counter()

        for s in items:
            img = load_rgb(images_dir / s.image)
            stem = Path(s.image).stem
            q = rng.randint(70, 95)
            save(out, f"{stem}_auth", jpeg_recompress(img, q), np.zeros(img.shape[:2], np.uint8))
            index.append({"name": f"{stem}_auth", "source": s.image, "op": "authentic", "tampered": 0, "jpeg_q": q})

            for k in range(per_image[split]):
                op = rng.choice(ops)
                extra: dict[str, str] = {}
                try:
                    if op == "copy_move":
                        t, m = copy_move(img, s.boxes, rng)
                    elif op == "splice":
                        donors = [d for d in items if d.image != s.image] or items  # same split only
                        donor = rng.choice(donors)
                        t, m = splice(img, s.boxes, load_rgb(images_dir / donor.image), donor.boxes, rng)
                        extra["donor"] = donor.image
                    else:
                        font = rng.choice(fonts)
                        t, m, _ = inpaint_and_render(img, s.boxes, font, rng)
                        extra["font"] = Path(font).name
                except ValueError:
                    skipped[op] += 1  # receipt lacks suitable digit boxes for this op
                    continue
                if not m.any():
                    skipped[op] += 1
                    continue
                q = rng.randint(70, 95)
                name = f"{stem}_{op}_{k}"
                save(out, name, jpeg_recompress(t, q), m)
                index.append({"name": name, "source": s.image, "op": op, "tampered": 1, "jpeg_q": q, **extra})

        with open(out / "index.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in index)
        log(f"{split}: {len(index)} images from {len(items)} receipts; skipped ops: {dict(skipped)}")
        result[split] = index
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--font", required=True, help="TTF file, or a directory of TTFs to rotate (Vietnamese glyphs)")
    ap.add_argument("--splits", required=True, help="canonical split file from scripts/make_splits.py")
    ap.add_argument("--tampered-per-image", type=int, default=2)
    ap.add_argument("--tampered-per-image-train", type=int, default=None, help="defaults to --tampered-per-image")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = read_jsonl(args.ann)
    canon = load_splits(args.splits)
    known = {s.image for s in samples}
    missing = sum(len(ids - known) for ids in canon.values())
    if missing:
        print(f"warning: {missing} split ids have no annotation and are skipped")

    n = args.tampered_per_image
    per_image = {"train": args.tampered_per_image_train if args.tampered_per_image_train is not None else n, "val": n, "test": n}
    generate(samples, canon, args.images, args.out, resolve_fonts(args.font), per_image, args.seed)


if __name__ == "__main__":
    main()
