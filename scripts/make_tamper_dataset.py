"""Generate the synthetic tampering dataset from normalized MC-OCR annotations.

Splits by source image so the same receipt never appears in two splits.
Every authentic image is also written (re-compressed with the same JPEG
distribution) so the classifier cannot shortcut on "was re-encoded".

Usage:
    python scripts/make_tamper_dataset.py --ann data/processed/annotations.jsonl \
        --images data/raw/mcocr/train_images --out data/tamper --font fonts/Roboto-Regular.ttf
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from docguard.data.annotations import read_jsonl
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
    ap.add_argument("--split", default="0.8,0.1,0.1")
    ap.add_argument("--tampered-per-image", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    samples = read_jsonl(args.ann)
    rng.shuffle(samples)
    fr = [float(x) for x in args.split.split(",")]
    n = len(samples)
    cut1, cut2 = int(fr[0] * n), int((fr[0] + fr[1]) * n)
    splits = {"train": samples[:cut1], "val": samples[cut1:cut2], "test": samples[cut2:]}
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
