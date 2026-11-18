"""Spec 003 AC-7: a fake model emitting the exact GT shrink map round-trips through the serving path."""

import numpy as np
import pytest

pytest.importorskip("pyclipper")

from docguard.config import pipeline_config
from docguard.pipeline import processing as P
from docguard.train.det_labels import make_shrink_map
from docguard.train.eval_det import evaluate_det


def _rect(b):
    x1, y1, x2, y2 = b
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float64)


def test_gt_kernel_roundtrip():
    cfg = pipeline_config()["text_det"]
    s = cfg["input_size"]
    assert s == 960
    # Two fixture images (long side 960 -> ratio 1): a 20x200 line and a 30x600 line (AC-7).
    samples = [
        {"image": np.zeros((960, 720, 3), np.uint8), "boxes": [[100, 100, 300, 120]], "labels": ["SELLER"]},
        {"image": np.zeros((960, 720, 3), np.uint8), "boxes": [[60, 400, 660, 430]], "labels": ["TOTAL_COST"]},
    ]
    maps = []
    for smp in samples:
        _, ratio = P.det_preprocess(smp["image"], s)
        polys = [_rect(np.array(b) * ratio) for b in smp["boxes"]]
        gt, _ = make_shrink_map(polys, [False] * len(polys), s, s, cfg["shrink_ratio"])
        maps.append(gt)
    it = iter(maps)

    def fake_predict(inp):
        assert inp.shape == (1, 3, s, s) and inp.dtype == np.float32
        return next(it)

    m = evaluate_det(fake_predict, samples, {"text_det": cfg})
    assert m["recall"] == 1.0 and m["precision"] == 1.0
    assert m["mean_iou"] >= 0.85
    assert m["recall_by_label"]["SELLER"] == {"recall": 1.0, "n": 1}


def test_label_filter_only_scores_selected_fields():
    # AC-6 (d): GT boxes outside `labels` do not count toward recall.
    cfg = pipeline_config()["text_det"]
    sample = {"image": np.zeros((960, 720, 3), np.uint8),
              "boxes": [[100, 100, 300, 120], [100, 500, 300, 520]], "labels": ["SELLER", "OTHER"]}
    gt, _ = make_shrink_map([_rect(sample["boxes"][0])], [False], 960, 960, cfg["shrink_ratio"])
    m = evaluate_det(lambda inp: gt, [sample], {"text_det": cfg}, labels={"SELLER"})
    assert (m["n_gt"], m["n_match"], m["recall"]) == (1, 1, 1.0)
