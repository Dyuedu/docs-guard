"""Generate the synthetic tampering dataset from normalized MC-OCR annotations.

Splits come from the canonical split file (scripts/make_splits.py), so the same
receipt is in the same split here as in the OCR/KIE data. Every authentic image is
also written (re-compressed with the same JPEG distribution) so the classifier
cannot shortcut on "was re-encoded".

Usage:
    python scripts/make_tamper_dataset.py --ann data/processed/annotations.jsonl \
        --splits data/processed/splits.json --images data/raw/mcocr/train_images \
        --out data/tamper --font data/fonts/Roboto-Regular.ttf
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from docguard.data.annotations import read_jsonl
from docguard.data.splits import load_splits
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--font", required=True, help="TTF font with Vietnamese glyphs")
    ap.add_argument("--splits", required=True, help="canonical split file from scripts/make_splits.py")
    ap.add_argument("--tampered-per-image", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    samples = read_jsonl(args.ann)
    canon = load_splits(args.splits)
    splits = {name: [s for s in samples if s.image in ids] for name, ids in canon.items()}
    missing = sum(len(v) for v in canon.values()) - sum(len(v) for v in splits.values())
    if missing:
        print(f"warning: {missing} split ids have no annotation and are skipped")
    img_dir = Path(args.images)

    for split, items in splits.items():
        out = Path(args.out) / split
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        index = []
        for s in items:
            img = load_rgb(img_dir / s.image)
            stem = Path(s.image).stem
            q = rng.randint(70, 95)
            save(out, f"{stem}_auth", jpeg_recompress(img, q), np.zeros(img.shape[:2], np.uint8))
            index.append({"name": f"{stem}_auth", "source": s.image, "op": "authentic", "tampered": 0, "jpeg_q": q})

            for k in range(args.tampered_per_image):
                op = rng.choice(OPS)
                try:
                    if op == "copy_move":
                        t, m = copy_move(img, s.boxes, rng)
                    elif op == "splice":
                        donor = rng.choice(items)  # donor from the same split only
                        t, m = splice(img, s.boxes, load_rgb(img_dir / donor.image), donor.boxes, rng)
                    else:
                        t, m, _ = inpaint_and_render(img, s.boxes, args.font, rng)
                except ValueError:
                    continue  # receipt lacks suitable digit boxes for this op
                if not m.any():
                    continue
                q = rng.randint(70, 95)
                name = f"{stem}_{op}_{k}"
                save(out, name, jpeg_recompress(t, q), m)
                index.append({"name": name, "source": s.image, "op": op, "tampered": 1, "jpeg_q": q})

        with open(out / "index.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in index)
        print(f"{split}: {len(index)} images from {len(items)} receipts")


if __name__ == "__main__":
    main()
