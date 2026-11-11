import csv
import json

import cv2
import numpy as np
import pytest

from docguard.data.annotations import convert_mcocr_csv, read_jsonl
from docguard.data.splits import load_splits, make_splits, save_splits, split_of


def _write_mcocr(tmp_path, rows):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for r in rows:
        cv2.imwrite(str(img_dir / r["img_id"]), np.full((100, 80, 3), 255, np.uint8))
    csv_path = tmp_path / "mcocr.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["img_id", "anno_polygons", "anno_texts", "anno_labels"])
        w.writeheader()
        w.writerows(rows)
    return csv_path, img_dir


def test_convert_mcocr_csv(tmp_path):
    polys = str([{"category_id": 15, "segmentation": [], "bbox": [10, 20, 30, 8]},
                 {"category_id": 18, "segmentation": [], "bbox": [5.4, 60.6, 40, 10]}])
    rows = [
        {"img_id": "a.jpg", "anno_polygons": polys, "anno_texts": "CO.OP MART|||125.000",
         "anno_labels": "SELLER|||TOTAL_COST"},
        {"img_id": "missing.jpg", "anno_polygons": "", "anno_texts": "", "anno_labels": ""},
    ]
    csv_path, img_dir = _write_mcocr(tmp_path, rows[:1])
    with open(csv_path, "a", encoding="utf-8", newline="") as f:  # row whose image is absent -> skipped
        csv.DictWriter(f, fieldnames=list(rows[1])).writerow(rows[1])

    out = tmp_path / "ann.jsonl"
    assert convert_mcocr_csv(csv_path, img_dir, out) == 1
    (s,) = read_jsonl(out)
    assert (s.image, s.height, s.width) == ("a.jpg", 100, 80)
    assert [b.bbox for b in s.boxes] == [[10, 20, 40, 28], [5, 61, 45, 71]]
    assert [b.label for b in s.boxes] == ["SELLER", "TOTAL_COST"]
    assert s.boxes[1].text == "125.000"


def test_make_splits_disjoint_complete_deterministic():
    ids = [f"img_{i}.jpg" for i in range(1000)]
    a = make_splits(ids, (0.8, 0.1, 0.1), seed=7)
    b = make_splits(list(reversed(ids)), (0.8, 0.1, 0.1), seed=7)  # input order must not matter
    assert a == b
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (800, 100, 100)
    assert set(a["train"]) | set(a["val"]) | set(a["test"]) == set(ids)
    assert make_splits(ids, seed=8)["test"] != a["test"]


def test_make_splits_rejects_bad_fractions():
    with pytest.raises(ValueError):
        make_splits(["a"], (0.5, 0.5, 0.5))


def test_save_load_and_overlap_detection(tmp_path):
    p = save_splits(make_splits([f"{i}.jpg" for i in range(10)]), tmp_path / "s.json")
    splits = load_splits(p)
    assert split_of(splits, next(iter(splits["val"]))) == "val"
    assert split_of(splits, "nope.jpg") is None

    d = json.loads(p.read_text())
    d["test"].append(d["train"][0])
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="overlap"):
        load_splits(p)
