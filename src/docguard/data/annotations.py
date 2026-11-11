"""Normalized annotation format and MC-OCR 2021 converter.

Normalized format (one JSON object per line, `annotations.jsonl`):
    {"image": "mcocr_public_xxx.jpg", "height": H, "width": W,
     "boxes": [{"bbox": [x1, y1, x2, y2], "text": "...", "label": "TOTAL_COST"}, ...]}
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TextBox:
    bbox: list[int]
    text: str
    label: str = "OTHER"


@dataclass
class Sample:
    image: str
    height: int
    width: int
    boxes: list[TextBox] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {"image": self.image, "height": self.height, "width": self.width,
             "boxes": [{"bbox": b.bbox, "text": b.text, "label": b.label} for b in self.boxes]},
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> "Sample":
        d = json.loads(line)
        return cls(d["image"], d["height"], d["width"], [TextBox(**b) for b in d["boxes"]])


def read_jsonl(path: str | Path) -> list[Sample]:
    with open(path, encoding="utf-8") as f:
        return [Sample.from_json(line) for line in f if line.strip()]


def convert_mcocr_csv(csv_path: str | Path, image_dir: str | Path, out_path: str | Path) -> int:
    """Convert the MC-OCR 2021 KIE csv to normalized jsonl.

    Assumes the public release layout: columns `img_id`, `anno_polygons` (python-literal
    list of dicts with a COCO-style `bbox` = [x, y, w, h]), `anno_texts` and
    `anno_labels` (both joined by '|||'). Verify against your downloaded copy.
    """
    import cv2

    image_dir = Path(image_dir)
    n = 0
    with open(csv_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for row in csv.DictReader(f):
            polys = ast.literal_eval(row["anno_polygons"]) if row.get("anno_polygons") else []
            texts = row.get("anno_texts", "").split("|||")
            labels = row.get("anno_labels", "").split("|||")
            img = cv2.imread(str(image_dir / row["img_id"]))
            if img is None:
                continue
            boxes = []
            for poly, text, label in zip(polys, texts, labels):
                x, y, w, h = (int(round(v)) for v in poly["bbox"])
                boxes.append(TextBox([x, y, x + w, y + h], text.strip(), label.strip()))
            out.write(Sample(row["img_id"], img.shape[0], img.shape[1], boxes).to_json() + "\n")
            n += 1
    return n
