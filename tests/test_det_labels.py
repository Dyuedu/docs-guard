"""Spec 003 AC-3 (shrink map), AC-4 (threshold map), AC-9 (dataset determinism)."""

import numpy as np
import pytest

pytest.importorskip("pyclipper")

from docguard.train.det_labels import make_border_map, make_shrink_map

RECT = np.array([[100, 100], [300, 100], [300, 120], [100, 120]], np.float32)  # 200 x 20
D = 200 * 20 * (1 - 0.4**2) / 440  # AC-3: ~7.64 px


def test_shrink_map_rect():
    gt, mask = make_shrink_map([RECT.copy()], [False], 200, 400)
    assert gt.dtype == np.float32 and mask.dtype == np.float32
    lo, hi_x, hi_y = 100 + D, 300 - D, 120 - D  # inner rectangle, +-1 px at the border
    assert gt[int(np.ceil(lo)) + 1 : int(hi_y), int(np.ceil(lo)) + 1 : int(hi_x)].min() == 1.0
    ys, xs = np.nonzero(gt)
    assert xs.min() >= lo - 1 and xs.max() <= hi_x + 1 and ys.min() >= lo - 1 and ys.max() <= hi_y + 1
    assert mask.min() == 1.0


def test_small_polygon_ignored():
    small = np.array([[50, 50], [120, 50], [120, 56], [50, 56]], np.float32)  # short side 6 < 8
    gt, mask = make_shrink_map([small.copy(), RECT.copy()], [False, True], 200, 400)
    assert gt.max() == 0.0
    assert mask[52, 80] == 0.0 and mask[110, 200] == 0.0  # both polygons masked out of the loss
    assert mask[10, 10] == 1.0


def test_threshold_map_values():
    t, m = make_border_map([RECT.copy()], [False], 200, 400)
    assert t.min() >= 0.3 - 1e-6 and t.max() <= 0.7 + 1e-6
    assert t[100, 200] == pytest.approx(0.7, abs=0.02)  # on the top edge
    assert t[120, 150] == pytest.approx(0.7, abs=0.02)  # on the bottom edge
    assert t[110, 200] == pytest.approx(0.3, abs=1e-6)  # 10 px from both long edges >= D
    assert t[50, 200] == pytest.approx(0.3, abs=1e-6)  # far outside
    assert t[104, 200] == pytest.approx(0.3 + 0.4 * (1 - 4 / D), abs=1e-3)  # AC-4 formula
    assert m[94, 200] == 1.0 and m[110, 200] == 1.0  # inside polygon dilated by D
    assert m[90, 200] == 0.0 and m[50, 200] == 0.0


def _write_fixture(root, n=2):
    import json

    import cv2

    (root / "images").mkdir()
    lines = []
    for i in range(n):
        img = np.full((900, 600, 3), 255, np.uint8)
        cv2.putText(img, "HOA DON 123", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(str(root / "images" / f"{i}.jpg"), img)
        boxes = [
            {"polygon": [[55, 175], [260, 175], [260, 210], [55, 210]], "bbox": [55, 175, 260, 210], "text": "HOA DON"},
            {"bbox": [60, 400, 300, 430], "text": "TONG"},
        ]
        lines.append(json.dumps({"image": f"images/{i}.jpg", "height": 900, "width": 600, "boxes": boxes}))
    (root / "index.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return str(root / "index.jsonl")


DS_CFG = {"input_size": 960, "crop_size": 640, "scale_range": [0.7, 1.3], "rotate_deg": 5,
          "shrink_ratio": 0.4, "thresh_min": 0.3, "thresh_max": 0.7, "min_text_size": 8}


def test_dataset_deterministic(tmp_path):
    # AC-9: same seed and index -> identical crop and identical 4 label maps.
    pytest.importorskip("torch")
    from docguard.train.det_dataset import DetDataset

    jp = _write_fixture(tmp_path)
    a, b = DetDataset([jp], None, True, DS_CFG, seed=7), DetDataset([jp], None, True, DS_CFG, seed=7)
    keys = ["image", "shrink_map", "shrink_mask", "thresh_map", "thresh_mask"]
    for idx in range(2):
        sa, sb = a[idx], b[idx]
        assert tuple(sa["image"].shape) == (3, 640, 640)
        for k in keys[1:]:
            assert tuple(sa[k].shape) == (640, 640)
        for k in keys:
            assert np.array_equal(sa[k].numpy(), sb[k].numpy()), k


def test_dataset_eval_mode_returns_raw_image_and_boxes(tmp_path):
    pytest.importorskip("torch")
    from docguard.train.det_dataset import DetDataset

    ds = DetDataset([_write_fixture(tmp_path, 1)], None, False, {}, seed=0)
    assert len(list(ds)) == 1  # iteration terminates (evaluate_det iterates the val set)
    s = ds[0]
    assert s["image"].shape == (900, 600, 3)
    assert s["boxes"] == [[55, 175, 260, 210], [60, 400, 300, 430]]


def test_record_polys_ignores_empty_text_and_uses_bbox_corners():
    # AC-3: empty text is ignored; MC-OCR records without polygon use the 4 bbox corners.
    pytest.importorskip("torch")
    from docguard.train.det_dataset import record_polys

    rec = {"boxes": [{"bbox": [10, 20, 110, 40], "text": " "}, {"bbox": [0, 0, 50, 30], "text": "###"},
                     {"polygon": [[1, 2], [3, 2], [3, 4], [1, 4]], "bbox": [1, 2, 3, 4], "text": "a"}]}
    polys, ignore = record_polys(rec)
    assert ignore == [True, True, False]
    assert polys[0].tolist() == [[10, 20], [110, 20], [110, 40], [10, 40]]
    assert polys[2].tolist() == [[1, 2], [3, 2], [3, 4], [1, 4]]
